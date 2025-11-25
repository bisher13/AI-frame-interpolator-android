package com.videointerpolation.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.util.Log
import ai.onnxruntime.*
import ai.onnxruntime.OnnxJavaType
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer

/**
 * ONNX Runtime-based interpolator. Uses NNAPI EP for acceleration where available.
 * Assumes a model that takes concatenated frames [1,6,H,W] and outputs [1,3,H,W] mid-frame.
 * If your model differs, adjust the input packing accordingly.
 */
class OnnxInterpolator(private val context: Context) {
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputName: String? = null // single-input models
    private var inputNames: List<String> = emptyList() // multi-input models (e.g., RIFE: 2 inputs)
    private var inputH: Int = 256
    private var inputW: Int = 256
    private var nchw: Boolean = true
    private var inputType: OnnxJavaType = OnnxJavaType.FLOAT
    private var outputType: OnnxJavaType = OnnxJavaType.FLOAT
    private var multiInput: Boolean = false
    private var hasTimeInput: Boolean = false
    private var timeInputName: String? = null
    private var requiredChannels: Int = 3
    private var allowDynamicH: Boolean = false
    private var allowDynamicW: Boolean = false

    // NNAPI device selection
    private var nnapiDeviceName: String? = null
    private var useNnapiCpuOnly: Boolean = false

    // Simple reusable buffer caches to avoid repeated DirectByteBuffer allocations per frame.
    // Keys are element counts (floats for float buffers, half-floats for byte buffers measured in shorts).
    private val floatBufCache = mutableMapOf<Int, java.nio.FloatBuffer>()
    private val halfByteBufCache = mutableMapOf<Int, java.nio.ByteBuffer>()

    private fun getFloatBuffer(elements: Int): java.nio.FloatBuffer {
        val existing = floatBufCache[elements]
        if (existing != null) {
            existing.clear()
            return existing
        }
        val buf = ByteBuffer.allocateDirect(4 * elements).order(ByteOrder.nativeOrder()).asFloatBuffer()
        floatBufCache[elements] = buf
        return buf
    }

    private fun getHalfByteBuffer(halfElements: Int): java.nio.ByteBuffer {
        val existing = halfByteBufCache[halfElements]
        if (existing != null) {
            existing.clear()
            // Also clear the asShort view by rewinding
            existing.asShortBuffer().clear()
            return existing
        }
        val buf = ByteBuffer.allocateDirect(2 * halfElements).order(ByteOrder.nativeOrder())
        halfByteBufCache[halfElements] = buf
        return buf
    }

    fun initialize(modelPath: String = "rife_fp16.onnx"): Boolean {
        return try {
            val e = OrtEnvironment.getEnvironment()
            val so = OrtSession.SessionOptions()
            // Best-effort: increase ORT verbosity via reflection to avoid compile-time dependency on newer APIs
            try {
                val soClass = so.javaClass
                val lvlClass = Class.forName("ai.onnxruntime.OrtLoggingLevel")
                val verbose = java.lang.Enum.valueOf(lvlClass as Class<out Enum<*>>, "ORT_LOGGING_LEVEL_VERBOSE")
                try {
                    val m = soClass.getMethod("setLogSeverityLevel", lvlClass)
                    m.invoke(so, verbose)
                } catch (_: Throwable) { /* method not present on this build */ }
                try {
                    val m2 = soClass.getMethod("setLogVerbosityLevel", Int::class.javaPrimitiveType)
                    m2.invoke(so, 1)
                } catch (_: Throwable) { /* method not present on this build */ }
            } catch (_: Throwable) { /* enum/class not present */ }
            
            // Add NNAPI with device targeting if specified
            try {
                val nnapiFlags = mutableMapOf<String, String>()
                
                // Device selection: target specific hardware (DSP, GPU, NPU)
                nnapiDeviceName?.let { device ->
                    if (device.isNotEmpty()) {
                        nnapiFlags["nnapi_accelerator_name"] = device
                        Log.i("OnnxInterpolator", "NNAPI targeting device: $device")
                    }
                }
                
                // CPU-only mode (for debugging or fallback)
                if (useNnapiCpuOnly) {
                    nnapiFlags["use_nchw"] = "1"
                    nnapiFlags["cpu_only"] = "1"
                    Log.i("OnnxInterpolator", "NNAPI CPU-only mode enabled")
                }
                
                // FP16 acceleration hint
                nnapiFlags["allow_fp16"] = "1"
                
                if (nnapiFlags.isNotEmpty()) {
                    // Use reflection to call addNnapi(Map<String,String>) if available
                    try {
                        val method = so.javaClass.getMethod("addNnapi", Map::class.java)
                        method.invoke(so, nnapiFlags)
                        Log.d("OnnxInterpolator", "NNAPI EP enabled with options: $nnapiFlags")
                    } catch (_: NoSuchMethodException) {
                        // Fallback to basic addNnapi() if parameterized version not available
                        so.addNnapi()
                        Log.d("OnnxInterpolator", "NNAPI EP enabled (basic, options not supported)")
                    }
                } else {
                    so.addNnapi()
                    Log.d("OnnxInterpolator", "NNAPI EP enabled")
                }
            } catch (t: Throwable) {
                Log.w("OnnxInterpolator", "NNAPI not available, using CPU: ${t.message}")
            }

            // Best-effort: log available providers without coupling to specific ORT API versions
            try {
                val envCls = Class.forName("ai.onnxruntime.OrtEnvironment")
                val m = envCls.getMethod("getAvailableProviders")
                val providers = m.invoke(null)
                Log.i("OnnxInterpolator", "Available providers: $providers; NNAPI requested")
            } catch (t: Throwable) {
                Log.d("OnnxInterpolator", "Provider list not available on this ORT build: ${t.message}")
            }

            // Load model from assets to a temp file (ORT needs a File path sometimes on Android)
            val tmpFile = extractAssetToCache(modelPath)
            val sess = e.createSession(tmpFile.absolutePath, so)
            env = e
            session = sess

            // Best-effort: log the execution providers actually in use for this session (if API exists)
            try {
                val method = sess.javaClass.getMethod("getExecutionProviders")
                val active = method.invoke(sess)
                Log.i("OnnxInterpolator", "Session execution providers: $active")
            } catch (_: Throwable) { /* ignore if not present */ }

            // Inspect model inputs for shapes/types.
            val inputs = sess.inputInfo
            try {
                for ((name, valInfo) in inputs) {
                    val ti = valInfo.info as TensorInfo
                    Log.i(
                        "OnnxInterpolator",
                        "Input '$name': type=${ti.type}, shape=${ti.shape.contentToString()}"
                    )
                }
            } catch (_: Throwable) { /* best-effort logging */ }
            if (inputs.isEmpty()) return true

            // Identify image inputs (rank 4 tensors) and optional time/scalar input
            val imageInputs = mutableListOf<Pair<String, TensorInfo>>()
            var scalarInput: Pair<String, TensorInfo>? = null
            for ((name, valInfo) in inputs) {
                val ti = valInfo.info as TensorInfo
                if (ti.shape.size == 4) {
                    imageInputs.add(name to ti)
                } else if ((ti.shape.isEmpty() || (ti.shape.size == 1 && (ti.shape[0] == 1L || ti.shape[0] <= 0))) &&
                    (ti.type == OnnxJavaType.FLOAT || ti.type == OnnxJavaType.FLOAT16)) {
                    scalarInput = name to ti
                }
            }

            multiInput = imageInputs.size >= 2
            if (multiInput) {
                // Use the first two as frame inputs
                inputNames = imageInputs.take(2).map { it.first }
                val ti = imageInputs.first().second
                inputType = ti.type
                // Determine layout and default sizes
                if (ti.shape.size == 4) {
                    if (ti.shape[1] == 3L) { // NCHW
                        nchw = true
                        inputH = (if (ti.shape[2] > 0) ti.shape[2] else 256).toInt()
                        inputW = (if (ti.shape[3] > 0) ti.shape[3] else 256).toInt()
                        allowDynamicH = ti.shape[2] <= 0
                        allowDynamicW = ti.shape[3] <= 0
                    } else if (ti.shape[3] == 3L) { // NHWC
                        nchw = false
                        inputH = (if (ti.shape[1] > 0) ti.shape[1] else 256).toInt()
                        inputW = (if (ti.shape[2] > 0) ti.shape[2] else 256).toInt()
                        allowDynamicH = ti.shape[1] <= 0
                        allowDynamicW = ti.shape[2] <= 0
                    }
                }
                requiredChannels = 3
                // Optional time input
                if (scalarInput != null) {
                    hasTimeInput = true
                    timeInputName = scalarInput.first
                }
            } else {
                // Single input networks (concatenated 6 channels)
                val first = inputs.entries.first()
                inputName = first.key
                val info = first.value.info as TensorInfo
                inputType = info.type
                if (info.shape.size == 4) {
                    if (info.shape[1] == 6L || info.shape[1] == 8L) { // NCHW 6 or 8 channels
                        nchw = true
                        inputH = (if (info.shape[2] > 0) info.shape[2] else 256).toInt()
                        inputW = (if (info.shape[3] > 0) info.shape[3] else 256).toInt()
                        allowDynamicH = info.shape[2] <= 0
                        allowDynamicW = info.shape[3] <= 0
                        requiredChannels = info.shape[1].toInt()
                    } else if (info.shape[3] == 6L || info.shape[3] == 8L) { // NHWC 6 or 8 channels
                        nchw = false
                        inputH = (if (info.shape[1] > 0) info.shape[1] else 256).toInt()
                        inputW = (if (info.shape[2] > 0) info.shape[2] else 256).toInt()
                        allowDynamicH = info.shape[1] <= 0
                        allowDynamicW = info.shape[2] <= 0
                        requiredChannels = info.shape[3].toInt()
                    }
                }
                if (requiredChannels != 6 && requiredChannels != 8) {
                    // default to 6 when unclear
                    requiredChannels = 6
                }
            }
            Log.i(
                "OnnxInterpolator",
                "Model inputs=${inputs.size}, multiInput=$multiInput, nchw=$nchw, C=$requiredChannels, H=$inputH, W=$inputW, type=$inputType -> outputType=$outputType"
            )

            // Capture output tensor type as well
            val outFirst = sess.outputInfo.entries.firstOrNull()
            if (outFirst != null) {
                val oinfo = outFirst.value.info as TensorInfo
                outputType = oinfo.type
            }
            true
        } catch (e: Exception) {
            Log.w("OnnxInterpolator", "Failed to init ONNX: ${e.message}")
            false
        }
    }

    fun close() {
        try { session?.close() } catch (_: Exception) {}
        try { env?.close() } catch (_: Exception) {}
        session = null
        env = null
    }

    /**
     * Generate midpoint frame between two frames with ONNX model.
     * If steps>1 are needed, you can blend frame1->mid and mid->frame2 outside.
     */
    fun midpoint(frame1: Bitmap, frame2: Bitmap, overrideW: Int? = null, overrideH: Int? = null): Bitmap? {
        // Guard against recycled inputs
        if (frame1.isRecycled || frame2.isRecycled) {
            Log.w("OnnxInterpolator", "Received recycled bitmap input; aborting midpoint")
            return null
        }
        val sess = session ?: return null
        // Determine inference size (optional override supports alignment optimization)
        val origW = inputW
        val origH = inputH
    val requestedW = overrideW ?: inputW
    val requestedH = overrideH ?: inputH
    // Only honor overrides if the model allows dynamic spatial dims
    val infW = ((if (allowDynamicW) requestedW else inputW)).coerceAtLeast(8)
    val infH = ((if (allowDynamicH) requestedH else inputH)).coerceAtLeast(8)
        val needTempSizeChange = (infW != inputW || infH != inputH)
        if (needTempSizeChange) {
            inputW = infW
            inputH = infH
        }
    // Avoid creating bitmaps larger than a safe threshold to keep memory and time bounded
    val SAFE_MAX_W = 1280
    val SAFE_MAX_H = 768
    val safeW = infW.coerceAtMost(SAFE_MAX_W)
    val safeH = infH.coerceAtMost(SAFE_MAX_H)
    // Prefer zero-padding over resampling when target is larger (e.g., align-to-8 rounding up)
    val s1 = resizeOrPadTo(frame1, safeW, safeH)
    val s2 = resizeOrPadTo(frame2, safeW, safeH)

    val inputs: Map<String, OnnxTensor>
    // Track all created tensors (both map values and any scalar/time tensors) to close exactly once.
    val createdTensors = mutableListOf<OnnxTensor>()
        if (multiInput && inputNames.size >= 2) {
            val t1 = createImageTensor(s1)
            val t2 = createImageTensor(s2)
            createdTensors.add(t1)
            createdTensors.add(t2)
            val map = mutableMapOf(
                inputNames[0] to t1,
                inputNames[1] to t2
            )
            if (hasTimeInput && timeInputName != null) {
                val tVal = 0.5f
                val tTensor = if (inputType == OnnxJavaType.FLOAT16) createScalarHalfTensor(tVal) else createScalarFloatTensor(tVal)
                map[timeInputName!!] = tTensor
                createdTensors.add(tTensor)
            }
            inputs = map
        } else {
            // Single concatenated input
            val fbFloat: FloatBuffer?
            val fbHalfByte: ByteBuffer?
            val tVal = 0.5f
            val shape = if (nchw) longArrayOf(1, requiredChannels.toLong(), inputH.toLong(), inputW.toLong()) else longArrayOf(1, inputH.toLong(), inputW.toLong(), requiredChannels.toLong())
            if (inputType == OnnxJavaType.FLOAT16) {
                fbFloat = null
                fbHalfByte = if (nchw) {
                    if (requiredChannels == 8) packNCHW8HalfByte(s1, s2, tVal) else packNCHW6HalfByte(s1, s2)
                } else {
                    if (requiredChannels == 8) packNHWC8HalfByte(s1, s2, tVal) else packNHWC6HalfByte(s1, s2)
                }
            } else {
                fbHalfByte = null
                fbFloat = if (nchw) {
                    if (requiredChannels == 8) packNCHW8Float(s1, s2, tVal) else packNCHW6Float(s1, s2)
                } else {
                    if (requiredChannels == 8) packNHWC8Float(s1, s2, tVal) else packNHWC6Float(s1, s2)
                }
            }
            val tensor = if (inputType == OnnxJavaType.FLOAT16) {
                OnnxTensor.createTensor(env, fbHalfByte, shape, OnnxJavaType.FLOAT16)
            } else {
                OnnxTensor.createTensor(env, fbFloat, shape)
            }
            inputs = mapOf((inputName ?: return null) to tensor)
            createdTensors.add(tensor)
        }
        s1.recycle(); s2.recycle()

        val result = try {
            sess.run(inputs)
        } catch (e: Exception) {
            Log.w("OnnxInterpolator", "ONNX run failed: ${e.message}")
            // Close all created tensors before returning (avoid double-close)
            for (t in createdTensors) { try { t.close() } catch (_: Exception) {} }
            return null
        }

        val out = result[0]
        var bmp: Bitmap? = null
        try {
            val ov = out as OnnxValue
            if (outputType == OnnxJavaType.FLOAT16) {
                // Read FP16 tensor as raw buffer and convert directly without allocating nested arrays
                val tensor = ov as OnnxTensor
                val bb = tensor.byteBuffer
                bb.order(ByteOrder.nativeOrder())
                val sb = bb.asShortBuffer()
                val total = inputH * inputW * 3
                if (sb.capacity() >= total) {
                    bmp = if (nchw) bitmapFromHalfBufferNCHW(sb) else bitmapFromHalfBufferNHWC(sb)
                } else {
                    Log.w("OnnxInterpolator", "FP16 output buffer smaller than expected: cap=${sb.capacity()} vs needed=$total")
                }
            } else {
                // Try to read FLOAT32 as raw buffer; fallback to array if necessary
                try {
                    val tensor = ov as OnnxTensor
                    val bb = tensor.byteBuffer
                    bb.order(ByteOrder.nativeOrder())
                    val fb = bb.asFloatBuffer()
                    val total = inputH * inputW * 3
                    if (fb.capacity() >= total) {
                        bmp = if (nchw) bitmapFromFloatBufferNCHW(fb) else bitmapFromFloatBufferNHWC(fb)
                    } else {
                        Log.w("OnnxInterpolator", "FP32 output buffer smaller than expected: cap=${fb.capacity()} vs needed=$total")
                    }
                } catch (_: Throwable) {
                    // Fallback: nested array extraction
                    val anyVal = ov.value
                    if (anyVal is Array<*>) {
                        if (anyVal.size == 1 && anyVal[0] is Array<*>) {
                            val firstLevel = anyVal[0] as Array<*>
                            if (firstLevel.size == 3 && firstLevel[0] is Array<*>) {
                                @Suppress("UNCHECKED_CAST")
                                val chw = firstLevel as Array<Array<FloatArray>>
                                bmp = toBitmapFromNCHW(chw)
                            } else if (firstLevel.isNotEmpty() && firstLevel[0] is Array<*>) {
                                @Suppress("UNCHECKED_CAST")
                                val hwc = firstLevel as Array<Array<FloatArray>>
                                bmp = toBitmapFromNHWC(hwc)
                            }
                        }
                    }
                }
            }
        } catch (t: Throwable) {
            Log.w("OnnxInterpolator", "Unable to convert output: ${t.message}")
        } finally {
            // Only close the OrtSession.Result; it will close contained OnnxValues.
            // Avoid closing 'out' separately to prevent double-close warnings.
            try { result.close() } catch (_: Exception) {}
            // Close all tensors we created exactly once
            createdTensors.forEach { t -> try { t.close() } catch (_: Exception) {} }
        }
        // Restore original configured size if we temporarily changed it
        if (needTempSizeChange) {
            inputW = origW
            inputH = origH
        }
        return bmp
    }

    // Direct FP16 decoding (NCHW): reads planes R,G,B then composes pixels.
    private fun bitmapFromHalfBufferNCHW(sb: ShortBuffer): Bitmap {
        val h = inputH
        val w = inputW
        val planeR = FloatArray(w * h)
        val planeG = FloatArray(w * h)
        val planeB = FloatArray(w * h)
        // Read R plane
        for (i in 0 until w * h) planeR[i] = clamp01(halfToFloat(sb.get()))
        // Read G plane
        for (i in 0 until w * h) planeG[i] = clamp01(halfToFloat(sb.get()))
        // Read B plane
        for (i in 0 until w * h) planeB[i] = clamp01(halfToFloat(sb.get()))
        val out = IntArray(w * h)
        var idx = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val r = (planeR[idx] * 255f).toInt()
                val g = (planeG[idx] * 255f).toInt()
                val b = (planeB[idx] * 255f).toInt()
                out[idx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                idx++
            }
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    // Direct FP16 decoding (NHWC): stream pixels directly.
    private fun bitmapFromHalfBufferNHWC(sb: ShortBuffer): Bitmap {
        val h = inputH
        val w = inputW
        val out = IntArray(w * h)
        var idx = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val r = (clamp01(halfToFloat(sb.get())) * 255f).toInt()
                val g = (clamp01(halfToFloat(sb.get())) * 255f).toInt()
                val b = (clamp01(halfToFloat(sb.get())) * 255f).toInt()
                out[idx++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    // Direct FP32 decoding (NCHW)
    private fun bitmapFromFloatBufferNCHW(fb: FloatBuffer): Bitmap {
        val h = inputH
        val w = inputW
        val planeR = FloatArray(w * h)
        val planeG = FloatArray(w * h)
        val planeB = FloatArray(w * h)
        for (i in 0 until w * h) planeR[i] = clamp01(fb.get())
        for (i in 0 until w * h) planeG[i] = clamp01(fb.get())
        for (i in 0 until w * h) planeB[i] = clamp01(fb.get())
        val out = IntArray(w * h)
        var idx = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val r = (planeR[idx] * 255f).toInt()
                val g = (planeG[idx] * 255f).toInt()
                val b = (planeB[idx] * 255f).toInt()
                out[idx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                idx++
            }
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    // Direct FP32 decoding (NHWC)
    private fun bitmapFromFloatBufferNHWC(fb: FloatBuffer): Bitmap {
        val h = inputH
        val w = inputW
        val out = IntArray(w * h)
        var idx = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val r = (clamp01(fb.get()) * 255f).toInt()
                val g = (clamp01(fb.get()) * 255f).toInt()
                val b = (clamp01(fb.get()) * 255f).toInt()
                out[idx++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun createImageTensor(bmp: Bitmap): OnnxTensor {
        val shape = if (nchw) longArrayOf(1, 3, inputH.toLong(), inputW.toLong()) else longArrayOf(1, inputH.toLong(), inputW.toLong(), 3)
        return if (inputType == OnnxJavaType.FLOAT16) {
            val bb = if (nchw) packNCHW3HalfByte(bmp) else packNHWC3HalfByte(bmp)
            OnnxTensor.createTensor(env, bb, shape, OnnxJavaType.FLOAT16)
        } else {
            val fb = if (nchw) packNCHW3Float(bmp) else packNHWC3Float(bmp)
            OnnxTensor.createTensor(env, fb, shape)
        }
    }

    private fun createScalarHalfTensor(value: Float): OnnxTensor {
        val bb = ByteBuffer.allocateDirect(2).order(ByteOrder.nativeOrder())
        bb.asShortBuffer().put(floatToHalf(value))
        bb.rewind()
        val shape = longArrayOf(1)
        return OnnxTensor.createTensor(env, bb, shape, OnnxJavaType.FLOAT16)
    }

    private fun createScalarFloatTensor(value: Float): OnnxTensor {
        val bb = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
        bb.asFloatBuffer().put(value)
        bb.rewind()
        val shape = longArrayOf(1)
        return OnnxTensor.createTensor(env, bb, shape, OnnxJavaType.FLOAT)
    }

    private fun packNCHW6Float(b1: Bitmap, b2: Bitmap): FloatBuffer {
        val size = (1L * 6 * inputH * inputW).toInt()
        val buf = getFloatBuffer(size)
        val p1 = IntArray(inputW * inputH)
        val p2 = IntArray(inputW * inputH)
        b1.getPixels(p1, 0, inputW, 0, 0, inputW, inputH)
        b2.getPixels(p2, 0, inputW, 0, 0, inputW, inputH)
        fun putChannel(pixels: IntArray, channel: Int) {
            for (y in 0 until inputH) {
                val row = y * inputW
                for (x in 0 until inputW) {
                    val v = pixels[row + x]
                    val c = when (channel) {
                        0 -> (v shr 16) and 0xFF
                        1 -> (v shr 8) and 0xFF
                        else -> v and 0xFF
                    }
                    buf.put(c / 255f)
                }
            }
        }
        // b1 RGB then b2 RGB
        putChannel(p1, 0); putChannel(p1, 1); putChannel(p1, 2)
        putChannel(p2, 0); putChannel(p2, 1); putChannel(p2, 2)
        buf.rewind()
        return buf
    }

    private fun packNHWC6Float(b1: Bitmap, b2: Bitmap): FloatBuffer {
        val size = (1L * inputH * inputW * 6).toInt()
        val buf = getFloatBuffer(size)
        val p1 = IntArray(inputW * inputH)
        val p2 = IntArray(inputW * inputH)
        b1.getPixels(p1, 0, inputW, 0, 0, inputW, inputH)
        b2.getPixels(p2, 0, inputW, 0, 0, inputW, inputH)
        for (y in 0 until inputH) {
            val row = y * inputW
            for (x in 0 until inputW) {
                val v1 = p1[row + x]
                val v2 = p2[row + x]
                buf.put(((v1 shr 16) and 0xFF) / 255f)
                buf.put(((v1 shr 8) and 0xFF) / 255f)
                buf.put((v1 and 0xFF) / 255f)
                buf.put(((v2 shr 16) and 0xFF) / 255f)
                buf.put(((v2 shr 8) and 0xFF) / 255f)
                buf.put((v2 and 0xFF) / 255f)
            }
        }
        buf.rewind()
        return buf
    }

    private fun packNCHW8Float(b1: Bitmap, b2: Bitmap, t: Float): FloatBuffer {
        val size = (1L * 8 * inputH * inputW).toInt()
        val buf = getFloatBuffer(size)
        val p1 = IntArray(inputW * inputH)
        val p2 = IntArray(inputW * inputH)
        b1.getPixels(p1, 0, inputW, 0, 0, inputW, inputH)
        b2.getPixels(p2, 0, inputW, 0, 0, inputW, inputH)
        fun putChannel(pixels: IntArray, ch: Int) {
            for (y in 0 until inputH) {
                val row = y * inputW
                for (x in 0 until inputW) {
                    val v = pixels[row + x]
                    val c = when (ch) { 0 -> (v shr 16) and 0xFF; 1 -> (v shr 8) and 0xFF; else -> v and 0xFF }
                    buf.put(c / 255f)
                }
            }
        }
        // frame1 RGB
        putChannel(p1, 0); putChannel(p1, 1); putChannel(p1, 2)
        // frame2 RGB
        putChannel(p2, 0); putChannel(p2, 1); putChannel(p2, 2)
        // t and (1-t) channels
        val tVal = t.coerceIn(0f, 1f)
        val inv = 1f - tVal
        for (y in 0 until inputH) {
            for (x in 0 until inputW) {
                buf.put(tVal)
            }
        }
        for (y in 0 until inputH) {
            for (x in 0 until inputW) {
                buf.put(inv)
            }
        }
        buf.rewind()
        return buf
    }

    private fun packNHWC8Float(b1: Bitmap, b2: Bitmap, t: Float): FloatBuffer {
        val size = (1L * inputH * inputW * 8).toInt()
        val buf = getFloatBuffer(size)
        val p1 = IntArray(inputW * inputH)
        val p2 = IntArray(inputW * inputH)
        b1.getPixels(p1, 0, inputW, 0, 0, inputW, inputH)
        b2.getPixels(p2, 0, inputW, 0, 0, inputW, inputH)
        val tVal = t.coerceIn(0f, 1f)
        val inv = 1f - tVal
        for (y in 0 until inputH) {
            val row = y * inputW
            for (x in 0 until inputW) {
                val v1 = p1[row + x]
                val v2 = p2[row + x]
                buf.put(((v1 shr 16) and 0xFF) / 255f)
                buf.put(((v1 shr 8) and 0xFF) / 255f)
                buf.put((v1 and 0xFF) / 255f)
                buf.put(((v2 shr 16) and 0xFF) / 255f)
                buf.put(((v2 shr 8) and 0xFF) / 255f)
                buf.put((v2 and 0xFF) / 255f)
                buf.put(tVal)
                buf.put(inv)
            }
        }
        buf.rewind()
        return buf
    }

    private fun packNCHW3Float(b: Bitmap): FloatBuffer {
        val size = (1L * 3 * inputH * inputW).toInt()
        val buf = getFloatBuffer(size)
        val p = IntArray(inputW * inputH)
        b.getPixels(p, 0, inputW, 0, 0, inputW, inputH)
        fun putC(ch: Int) {
            for (y in 0 until inputH) {
                val row = y * inputW
                for (x in 0 until inputW) {
                    val v = p[row + x]
                    val c = when (ch) { 0 -> (v shr 16) and 0xFF; 1 -> (v shr 8) and 0xFF; else -> v and 0xFF }
                    buf.put(c / 255f)
                }
            }
        }
        putC(0); putC(1); putC(2)
        buf.rewind()
        return buf
    }

    private fun packNHWC3Float(b: Bitmap): FloatBuffer {
        val size = (1L * inputH * inputW * 3).toInt()
        val buf = getFloatBuffer(size)
        val p = IntArray(inputW * inputH)
        b.getPixels(p, 0, inputW, 0, 0, inputW, inputH)
        for (y in 0 until inputH) {
            val row = y * inputW
            for (x in 0 until inputW) {
                val v = p[row + x]
                buf.put(((v shr 16) and 0xFF) / 255f)
                buf.put(((v shr 8) and 0xFF) / 255f)
                buf.put((v and 0xFF) / 255f)
            }
        }
        buf.rewind()
        return buf
    }

    private fun packNCHW3HalfByte(b: Bitmap): ByteBuffer {
        val size = (1L * 3 * inputH * inputW).toInt()
        val byteBuf = getHalfByteBuffer(size)
        val buf = byteBuf.asShortBuffer()
        val p = IntArray(inputW * inputH)
        b.getPixels(p, 0, inputW, 0, 0, inputW, inputH)
        fun putC(ch: Int) {
            for (y in 0 until inputH) {
                val row = y * inputW
                for (x in 0 until inputW) {
                    val v = p[row + x]
                    val c = when (ch) { 0 -> (v shr 16) and 0xFF; 1 -> (v shr 8) and 0xFF; else -> v and 0xFF }
                    buf.put(floatToHalf(c / 255f))
                }
            }
        }
        putC(0); putC(1); putC(2)
        buf.rewind(); byteBuf.rewind()
        return byteBuf
    }

    private fun packNHWC3HalfByte(b: Bitmap): ByteBuffer {
        val size = (1L * inputH * inputW * 3).toInt()
        val byteBuf = getHalfByteBuffer(size)
        val buf = byteBuf.asShortBuffer()
        val p = IntArray(inputW * inputH)
        b.getPixels(p, 0, inputW, 0, 0, inputW, inputH)
        for (y in 0 until inputH) {
            val row = y * inputW
            for (x in 0 until inputW) {
                val v = p[row + x]
                buf.put(floatToHalf(((v shr 16) and 0xFF) / 255f))
                buf.put(floatToHalf(((v shr 8) and 0xFF) / 255f))
                buf.put(floatToHalf((v and 0xFF) / 255f))
            }
        }
        buf.rewind(); byteBuf.rewind()
        return byteBuf
    }
    private fun packNCHW6HalfByte(b1: Bitmap, b2: Bitmap): ByteBuffer {
        val size = (1L * 6 * inputH * inputW).toInt()
        val byteBuf = getHalfByteBuffer(size)
        val buf = byteBuf.asShortBuffer()
        val p1 = IntArray(inputW * inputH)
        val p2 = IntArray(inputW * inputH)
        b1.getPixels(p1, 0, inputW, 0, 0, inputW, inputH)
        b2.getPixels(p2, 0, inputW, 0, 0, inputW, inputH)
        fun putChannelHalf(pixels: IntArray, channel: Int) {
            for (y in 0 until inputH) {
                val row = y * inputW
                for (x in 0 until inputW) {
                    val v = pixels[row + x]
                    val c = when (channel) {
                        0 -> (v shr 16) and 0xFF
                        1 -> (v shr 8) and 0xFF
                        else -> v and 0xFF
                    }
                    buf.put(floatToHalf(c / 255f))
                }
            }
        }
        putChannelHalf(p1, 0); putChannelHalf(p1, 1); putChannelHalf(p1, 2)
        putChannelHalf(p2, 0); putChannelHalf(p2, 1); putChannelHalf(p2, 2)
        buf.rewind()
        byteBuf.rewind()
        return byteBuf
    }

    private fun packNHWC6HalfByte(b1: Bitmap, b2: Bitmap): ByteBuffer {
        val size = (1L * inputH * inputW * 6).toInt()
        val byteBuf = getHalfByteBuffer(size)
        val buf = byteBuf.asShortBuffer()
        val p1 = IntArray(inputW * inputH)
        val p2 = IntArray(inputW * inputH)
        b1.getPixels(p1, 0, inputW, 0, 0, inputW, inputH)
        b2.getPixels(p2, 0, inputW, 0, 0, inputW, inputH)
        for (y in 0 until inputH) {
            val row = y * inputW
            for (x in 0 until inputW) {
                val v1 = p1[row + x]
                val v2 = p2[row + x]
                buf.put(floatToHalf(((v1 shr 16) and 0xFF) / 255f))
                buf.put(floatToHalf(((v1 shr 8) and 0xFF) / 255f))
                buf.put(floatToHalf((v1 and 0xFF) / 255f))
                buf.put(floatToHalf(((v2 shr 16) and 0xFF) / 255f))
                buf.put(floatToHalf(((v2 shr 8) and 0xFF) / 255f))
                buf.put(floatToHalf((v2 and 0xFF) / 255f))
            }
        }
        buf.rewind()
        byteBuf.rewind()
        return byteBuf
    }

    private fun packNCHW8HalfByte(b1: Bitmap, b2: Bitmap, t: Float): ByteBuffer {
        val size = (1L * 8 * inputH * inputW).toInt()
        val byteBuf = getHalfByteBuffer(size)
        val buf = byteBuf.asShortBuffer()
        val p1 = IntArray(inputW * inputH)
        val p2 = IntArray(inputW * inputH)
        b1.getPixels(p1, 0, inputW, 0, 0, inputW, inputH)
        b2.getPixels(p2, 0, inputW, 0, 0, inputW, inputH)
        fun putChannelHalf(pixels: IntArray, ch: Int) {
            for (y in 0 until inputH) {
                val row = y * inputW
                for (x in 0 until inputW) {
                    val v = pixels[row + x]
                    val c = when (ch) { 0 -> (v shr 16) and 0xFF; 1 -> (v shr 8) and 0xFF; else -> v and 0xFF }
                    buf.put(floatToHalf(c / 255f))
                }
            }
        }
        putChannelHalf(p1, 0); putChannelHalf(p1, 1); putChannelHalf(p1, 2)
        putChannelHalf(p2, 0); putChannelHalf(p2, 1); putChannelHalf(p2, 2)
        val tVal = t.coerceIn(0f, 1f)
        val inv = 1f - tVal
        for (y in 0 until inputH) {
            for (x in 0 until inputW) {
                buf.put(floatToHalf(tVal))
            }
        }
        for (y in 0 until inputH) {
            for (x in 0 until inputW) {
                buf.put(floatToHalf(inv))
            }
        }
        buf.rewind(); byteBuf.rewind()
        return byteBuf
    }

    private fun packNHWC8HalfByte(b1: Bitmap, b2: Bitmap, t: Float): ByteBuffer {
        val size = (1L * inputH * inputW * 8).toInt()
        val byteBuf = getHalfByteBuffer(size)
        val buf = byteBuf.asShortBuffer()
        val p1 = IntArray(inputW * inputH)
        val p2 = IntArray(inputW * inputH)
        b1.getPixels(p1, 0, inputW, 0, 0, inputW, inputH)
        b2.getPixels(p2, 0, inputW, 0, 0, inputW, inputH)
        val tVal = t.coerceIn(0f, 1f)
        val inv = 1f - tVal
        for (y in 0 until inputH) {
            val row = y * inputW
            for (x in 0 until inputW) {
                val v1 = p1[row + x]
                val v2 = p2[row + x]
                buf.put(floatToHalf(((v1 shr 16) and 0xFF) / 255f))
                buf.put(floatToHalf(((v1 shr 8) and 0xFF) / 255f))
                buf.put(floatToHalf((v1 and 0xFF) / 255f))
                buf.put(floatToHalf(((v2 shr 16) and 0xFF) / 255f))
                buf.put(floatToHalf(((v2 shr 8) and 0xFF) / 255f))
                buf.put(floatToHalf((v2 and 0xFF) / 255f))
                buf.put(floatToHalf(tVal))
                buf.put(floatToHalf(inv))
            }
        }
        buf.rewind(); byteBuf.rewind()
        return byteBuf
    }

    private fun toBitmapFromNCHW(chw: Array<Array<FloatArray>>): Bitmap {
        val c = chw.size // 3
        val h = chw[0].size
        val w = chw[0][0].size
        val out = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val r = (clamp01(chw[0][y][x]) * 255f).toInt()
                val g = (clamp01(chw[1][y][x]) * 255f).toInt()
                val b = (clamp01(chw[2][y][x]) * 255f).toInt()
                out[y * w + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.setPixels(out, 0, w, 0, 0, w, h)
        return bmp
    }

    private fun toBitmapFromNHWC(hwc: Array<Array<FloatArray>>): Bitmap {
        val h = hwc.size
        val w = hwc[0].size
        val out = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val r = (clamp01(hwc[y][x][0]) * 255f).toInt()
                val g = (clamp01(hwc[y][x][1]) * 255f).toInt()
                val b = (clamp01(hwc[y][x][2]) * 255f).toInt()
                out[y * w + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.setPixels(out, 0, w, 0, 0, w, h)
        return bmp
    }

    private fun toBitmapFromNCHWHalf(chw: Array<Array<ShortArray>>): Bitmap {
        val h = chw[0].size
        val w = chw[0][0].size
        val out = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val r = (clamp01(halfToFloat(chw[0][y][x])) * 255f).toInt()
                val g = (clamp01(halfToFloat(chw[1][y][x])) * 255f).toInt()
                val b = (clamp01(halfToFloat(chw[2][y][x])) * 255f).toInt()
                out[y * w + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.setPixels(out, 0, w, 0, 0, w, h)
        return bmp
    }

    private fun toBitmapFromNHWCHalf(hwc: Array<Array<ShortArray>>): Bitmap {
        val h = hwc.size
        val w = hwc[0].size
        val out = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val r = (clamp01(halfToFloat(hwc[y][x][0])) * 255f).toInt()
                val g = (clamp01(halfToFloat(hwc[y][x][1])) * 255f).toInt()
                val b = (clamp01(halfToFloat(hwc[y][x][2])) * 255f).toInt()
                out[y * w + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.setPixels(out, 0, w, 0, 0, w, h)
        return bmp
    }

    private fun clamp01(v: Float) = when {
        v < 0f -> 0f
        v > 1f -> 1f
        else -> v
    }

    // IEEE-754 half <-> float converters
    private fun floatToHalf(f: Float): Short {
        val fBits = java.lang.Float.floatToIntBits(f)
        val sign = (fBits ushr 16) and 0x8000
        var valExp = (fBits ushr 23) and 0xFF
        var mant = fBits and 0x7FFFFF
        if (valExp == 255) { // Inf/NaN
            val out = sign or 0x7C00 or (if (mant != 0) 1 else 0)
            return out.toShort()
        }
        valExp -= 127
        if (valExp < -14) {
            // subnormal or zero
            if (valExp < -24) {
                return sign.toShort()
            }
            mant = (mant or 0x800000) ushr (14 - valExp)
            return (sign or (mant + 0x1000 ushr 13)).toShort()
        } else if (valExp > 15) {
            // overflow -> Inf
            return (sign or 0x7C00).toShort()
        }
        valExp += 15
        val out = sign or (valExp shl 10) or (mant + 0x1000 ushr 13)
        return out.toShort()
    }

    private fun halfToFloat(h: Short): Float {
        val bits = h.toInt() and 0xFFFF
        val sign = (bits ushr 15) and 0x00000001
        var exp = (bits ushr 10) and 0x0000001F
        var mant = bits and 0x000003FF
        var fBits: Int
        if (exp == 0) {
            if (mant == 0) {
                fBits = sign shl 31
            } else {
                // subnormal
                exp = 1
                while ((mant and 0x00000400) == 0) {
                    mant = mant shl 1
                    exp -= 1
                }
                mant = mant and 0x000003FF
                exp = exp + (127 - 15)
                mant = mant shl 13
                fBits = (sign shl 31) or (exp shl 23) or mant
            }
        } else if (exp == 31) {
            // Inf/NaN
            fBits = (sign shl 31) or (0xFF shl 23) or (mant shl 13)
        } else {
            // normalized
            exp = exp + (127 - 15)
            mant = mant shl 13
            fBits = (sign shl 31) or (exp shl 23) or mant
        }
        return java.lang.Float.intBitsToFloat(fBits)
    }

    private fun extractAssetToCache(assetName: String): File {
        val f = File(context.cacheDir, assetName)
        if (f.exists()) return f
        context.assets.open(assetName).use { input ->
            f.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        return f
    }

    /**
     * Set the NNAPI device name for hardware targeting.
     * Common values: "qti-dsp", "qti-gpu", "qti-hta", "gpu", "dsp", "npu"
     * Leave empty/null for automatic selection.
     * Changes take effect after next initialize() call.
     */
    fun setNnapiDeviceName(deviceName: String?) {
        nnapiDeviceName = deviceName?.takeIf { it.isNotEmpty() }
        Log.i("OnnxInterpolator", "NNAPI device set to: ${nnapiDeviceName ?: "auto"}")
    }

    /**
     * Enable/disable NNAPI CPU-only mode (for debugging).
     * Changes take effect after next initialize() call.
     */
    fun setNnapiCpuOnly(cpuOnly: Boolean) {
        useNnapiCpuOnly = cpuOnly
        Log.i("OnnxInterpolator", "NNAPI CPU-only: $cpuOnly")
    }

    /**
     * Get available NNAPI devices on this system.
     * Returns list of device names that can be passed to setNnapiDeviceName().
     */
    fun getAvailableNnapiDevices(): List<String> {
        return try {
            // Use reflection to get available NNAPI devices
            val nnapiClass = Class.forName("ai.onnxruntime.providers.NnapiFlags")
            val method = nnapiClass.getMethod("getAvailableDevices")
            @Suppress("UNCHECKED_CAST")
            method.invoke(null) as? List<String> ?: emptyList()
        } catch (t: Throwable) {
            Log.d("OnnxInterpolator", "Cannot enumerate NNAPI devices: ${t.message}")
            // Return common device names as fallback
            listOf("qti-dsp", "qti-gpu", "qti-hta", "gpu", "dsp", "npu", "nnapi-reference")
        }
    }

    /**
     * Get current acceleration status string for display.
     */
    fun getAccelerationInfo(): String {
        return try {
            val sess = session ?: return "Not initialized"
            val method = sess.javaClass.getMethod("getExecutionProviders")
            val providers = method.invoke(sess) as? Set<*>
            when {
                providers?.any { it.toString().contains("NNAPI", ignoreCase = true) } == true -> {
                    val device = nnapiDeviceName ?: "auto"
                    "ONNX NNAPI ($device)"
                }
                providers?.any { it.toString().contains("XNNPACK", ignoreCase = true) } == true -> "ONNX XNNPACK"
                else -> "ONNX CPU"
            }
        } catch (_: Throwable) {
            "ONNX Runtime"
        }
    }


    // Public: get active providers string for UI
    fun getActiveProvidersString(): String {
        return getAccelerationInfo()
    }

    // Helper: prefer padding (fast, quality-neutral) when target is larger; use NN resize when shrinking.
    private fun resizeOrPadTo(src: Bitmap, targetW: Int, targetH: Int): Bitmap {
        if (src.width == targetW && src.height == targetH) {
            return src.copy(src.config, false)
        }
        // If we only need to grow to meet alignment/cap, pad without filtering
        if (targetW >= src.width && targetH >= src.height) {
            val out = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(out)
            canvas.drawBitmap(src, 0f, 0f, null)
            return out
        }
        // Otherwise (downscale), do a fast scale without bilinear filtering to reduce compute
        return Bitmap.createScaledBitmap(src, targetW, targetH, false)
    }
}
