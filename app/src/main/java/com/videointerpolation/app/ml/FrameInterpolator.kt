package com.videointerpolation.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream
import kotlin.math.min

enum class InterpMode { OPTICAL_FLOW, LINEAR, TFLITE, ONNX }

class FrameInterpolator(private val context: Context, private val mode: InterpMode = InterpMode.OPTICAL_FLOW) {
    
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    // Track if a TFLite NNAPI delegate is attached and the requested accelerator name (if any)
    private var tfliteNnapiActive: Boolean = false
    private var tfliteNnapiName: String? = null
    private var onnx: OnnxInterpolator? = null
    private var onnxReady: Boolean = false
    private var gpuOpticalFlow: GpuOpticalFlowInterpolator? = null
    private var gpuOpticalFlowReady: Boolean = false
    private var initialized: Boolean = false
    private var alignToEight: Boolean = true // Ensure alignment to multiples of 8 for better quality
    // Optional NNAPI accelerator name for TFLite (e.g., "qti-dsp", "gpu", "npu")
    private var nnapiAcceleratorName: String? = null
    private var forceUseNnapiTflite: Boolean = false
    
    companion object {
        private const val INPUT_SIZE = 256
        private const val PIXEL_SIZE = 3
        private const val IMAGE_MEAN = 0f
        private const val IMAGE_STD = 255f
    }
    
    /**
     * Initialize the TensorFlow Lite interpreter
     * Note: You'll need to add a TFLite model file to assets folder
     */
    fun initialize(modelPath: String = "rife_fp16.tflite", onnxModelPath: String = "rife_fp16.onnx") {
        try {
            // Try preferred backend first, then fallback to alternates
            when (mode) {
                InterpMode.ONNX -> {
                    // Try ONNX first
                    onnx = OnnxInterpolator(context).also { onnxInterp ->
                        // Configure NNAPI device before initialization
                        nnapiAcceleratorName?.let { onnxInterp.setNnapiDeviceName(it) }
                    }
                    onnxReady = onnx?.initialize(onnxModelPath) == true
                    if (!onnxReady) {
                        Log.w("FrameInterpolator", "ONNX init failed, trying TFLite")
                        initTflite(modelPath)
                    }
                }
                InterpMode.TFLITE -> {
                    initTflite(modelPath)
                    if (interpreter == null) {
                        // If TFLite unavailable, try ONNX as fallback
                        onnx = OnnxInterpolator(context).also { onnxInterp ->
                            nnapiAcceleratorName?.let { onnxInterp.setNnapiDeviceName(it) }
                        }
                        onnxReady = onnx?.initialize(onnxModelPath) == true
                    }
                }
                else -> {
                    // Non-ML modes: optionally prepare ML backends as opportunistic acceleration
                    initTflite(modelPath)
                    if (interpreter == null) {
                        onnx = OnnxInterpolator(context).also { onnxInterp ->
                            nnapiAcceleratorName?.let { onnxInterp.setNnapiDeviceName(it) }
                        }
                        onnxReady = onnx?.initialize(onnxModelPath) == true
                    }
                }
            }
            initialized = true
            
            // Try to initialize GPU optical flow (optional, falls back to CPU if unavailable)
            if (mode == InterpMode.OPTICAL_FLOW || interpreter == null && !onnxReady) {
                gpuOpticalFlow = GpuOpticalFlowInterpolator(context).also { gpuFlow ->
                    nnapiAcceleratorName?.let { gpuFlow.setNnapiDeviceName(it) }
                    gpuOpticalFlowReady = gpuFlow.initialize("optical_flow.onnx")
                }
            }
            
            Log.i("FrameInterpolator", "Initialization complete. ONNX=${onnxReady}, TFLite=${interpreter != null}, GPUOpticalFlow=${gpuOpticalFlowReady}, preferred=$mode, effective=${getEffectiveMode()}")
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun initTflite(modelPath: String) {
        try {
            val options = Interpreter.Options()
            // Prefer NNAPI delegate when requested; otherwise use GPU delegate if available
            val wantNnapi = forceUseNnapiTflite || (nnapiAcceleratorName != null)
            // Reset TFLite delegate tracking before (re)initialization
            tfliteNnapiActive = false
            tfliteNnapiName = null
            if (wantNnapi) {
                try {
                    val nnOptionsCls = Class.forName("org.tensorflow.lite.nnapi.NnApiDelegate\$Options")
                    val nnDelegateCls = Class.forName("org.tensorflow.lite.nnapi.NnApiDelegate")
                    val ctorOpts = nnOptionsCls.getConstructor()
                    val nnOpts = ctorOpts.newInstance()
                    // setAllowFp16(true)
                    try { nnOptionsCls.getMethod("setAllowFp16", Boolean::class.javaPrimitiveType).invoke(nnOpts, true) } catch (_: Throwable) {}
                    // setExecutionPreference(SUSTAINED_SPEED)
                    try {
                        val prefField = nnOptionsCls.getField("EXECUTION_PREFERENCE_SUSTAINED_SPEED")
                        val prefVal = prefField.get(null)
                        nnOptionsCls.getMethod("setExecutionPreference", prefField.type).invoke(nnOpts, prefVal)
                    } catch (_: Throwable) {}
                    // setAcceleratorName(name)
                    val name = nnapiAcceleratorName
                    if (!name.isNullOrBlank()) {
                        try { nnOptionsCls.getMethod("setAcceleratorName", String::class.java).invoke(nnOpts, name) } catch (_: Throwable) {}
                    }
                    // Optional caching (best-effort)
                    try {
                        val cacheDir = context.cacheDir?.absolutePath
                        if (!cacheDir.isNullOrBlank()) {
                            nnOptionsCls.getMethod("setModelCacheDir", String::class.java).invoke(nnOpts, cacheDir)
                            nnOptionsCls.getMethod("setCompilationCacheDir", String::class.java).invoke(nnOpts, cacheDir)
                            nnOptionsCls.getMethod("setModelToken", String::class.java).invoke(nnOpts, "rife_fp16")
                        }
                    } catch (_: Throwable) {}

                    // Create delegate and add to options
                    val ctorDel = nnDelegateCls.getConstructor(nnOptionsCls)
                    val nnDelegate = ctorDel.newInstance(nnOpts)
                    try {
                        options.addDelegate(nnDelegate as org.tensorflow.lite.Delegate)
                        tfliteNnapiActive = true
                        tfliteNnapiName = name
                        android.util.Log.i("FrameInterpolator", "Using TFLite NNAPI delegate${if (!name.isNullOrBlank()) " ($name)" else ""}")
                    } catch (addErr: Throwable) {
                        android.util.Log.w("FrameInterpolator", "Failed to add NNAPI delegate, falling back: ${addErr.message}")
                    }
                } catch (t: Throwable) {
                    android.util.Log.w("FrameInterpolator", "NNAPI delegate not available: ${t.message}")
                    // Fallback: GPU delegate if available
                    val compatList = CompatibilityList()
                    if (compatList.isDelegateSupportedOnThisDevice) {
                        gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                        try { options.addDelegate(gpuDelegate) } catch (e: Throwable) {
                            android.util.Log.w("FrameInterpolator", "Failed to add GPU delegate: ${e.message}")
                        }
                    } else {
                        options.setNumThreads(4)
                    }
                }
            } else {
                val compatList = CompatibilityList()
                if (compatList.isDelegateSupportedOnThisDevice) {
                    gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                    try { options.addDelegate(gpuDelegate) } catch (e: Throwable) {
                        android.util.Log.w("FrameInterpolator", "Failed to add GPU delegate: ${e.message}")
                    }
                } else {
                    options.setNumThreads(4)
                }
            }
            val model = loadModelFile(modelPath)
            interpreter = Interpreter(model, options)
            Log.d("FrameInterpolator", "Loaded TFLite model: $modelPath")
        } catch (e: Exception) {
            Log.w("FrameInterpolator", "TFLite model not available: ${e.message}")
        }
    }
    
    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    /**
     * Interpolate between two frames using a simple averaging algorithm
     * For production, this should use a neural network model
     */
    fun interpolateFrames(frame1: Bitmap, frame2: Bitmap, steps: Int = 1): List<Bitmap> {
        val interpolatedFrames = mutableListOf<Bitmap>()

        // Ensure frames are the same size by scaling frame2 to frame1's size (preserve output resolution)
        val targetW = frame1.width
        val targetH = frame1.height
        val f1 = if (frame1.width == targetW && frame1.height == targetH) frame1
                 else Bitmap.createScaledBitmap(frame1, targetW, targetH, true)
        val f2 = if (frame2.width == targetW && frame2.height == targetH) frame2
                 else Bitmap.createScaledBitmap(frame2, targetW, targetH, true)

        // Choose effective mode depending on what is initialized
        val effectiveMode = getEffectiveMode()

        when (effectiveMode) {
            InterpMode.LINEAR -> {
                for (step in 1..steps) {
                    val alpha = step.toFloat() / (steps + 1)
                    val interpolated = linearInterpolate(f1, f2, alpha, targetW, targetH)
                    interpolatedFrames.add(interpolated)
                }
            }
            InterpMode.ONNX -> {
                val outs = onnxInterpolate(f1, f2, steps)
                // Resize to target if ONNX model works on fixed size
                outs.forEach { out ->
                    val finalBmp = if (out.width != targetW || out.height != targetH) {
                        Bitmap.createScaledBitmap(out, targetW, targetH, true).also { out.recycle() }
                    } else out
                    interpolatedFrames.add(finalBmp)
                }
            }
            InterpMode.TFLITE -> {
                for (step in 1..steps) {
                    val alpha = step.toFloat() / (steps + 1)
                    val out = tfliteInterpolate(f1, f2, alpha)
                    // Resize to target size if model runs on fixed input size
                    val finalBmp = if (out.width != targetW || out.height != targetH) {
                        Bitmap.createScaledBitmap(out, targetW, targetH, true).also { out.recycle() }
                    } else out
                    interpolatedFrames.add(finalBmp)
                }
            }
            InterpMode.OPTICAL_FLOW -> {
                // Try GPU optical flow first, fall back to CPU if unavailable
                if (gpuOpticalFlowReady && gpuOpticalFlow != null) {
                    // GPU-accelerated optical flow via neural network
                    for (step in 1..steps) {
                        val alpha = step.toFloat() / (steps + 1)
                        val mid = gpuOpticalFlow!!.interpolate(f1, f2, alpha)
                        interpolatedFrames.add(mid)
                    }
                } else {
                    // CPU optical flow (original implementation)
                    // Work at a reduced resolution for speed, then upscale if needed
                    val maxFlowWidth = 480
                    val (wFlow, hFlow, s1, s2) = if (targetW > maxFlowWidth) {
                        val scale = maxFlowWidth.toFloat() / targetW
                        val wf = maxFlowWidth
                        val hf = (targetH * scale).toInt().coerceAtLeast(1)
                        val sf1 = Bitmap.createScaledBitmap(f1, wf, hf, true)
                        val sf2 = Bitmap.createScaledBitmap(f2, wf, hf, true)
                        Quad(wf, hf, sf1, sf2)
                    } else Quad(targetW, targetH, f1, f2)

                    val of = OpticalFlowInterpolator(blockSize = 8, searchRadius = 4)
                    // Parallelize per-step interpolation using a fixed thread pool; preserve ordering
                    val cores = Runtime.getRuntime().availableProcessors().coerceAtLeast(2)
                    val poolSize = min(cores, steps.coerceAtLeast(1))
                    val pool = java.util.concurrent.Executors.newFixedThreadPool(poolSize)
                    try {
                        val futures = (1..steps).map { step ->
                            val alpha = step.toFloat() / (steps + 1)
                            java.util.concurrent.Callable {
                                val low = of.interpolate(s1, s2, alpha)
                                if (wFlow != targetW || hFlow != targetH) {
                                    Bitmap.createScaledBitmap(low, targetW, targetH, true).also { low.recycle() }
                                } else low
                            }
                        }.map { pool.submit(it) }
                        futures.forEach { f -> interpolatedFrames.add(f.get()) }
                    } finally {
                        pool.shutdown()
                    }

                    if (s1 !== f1) s1.recycle()
                    if (s2 !== f2) s2.recycle()
                }
            }
        }

        if (f1 !== frame1) f1.recycle()
        if (f2 !== frame2) f2.recycle()

        return interpolatedFrames
    }

    fun isInitialized(): Boolean = initialized
    fun isOnnxReady(): Boolean = onnxReady
    fun isTfliteReady(): Boolean = interpreter != null
    fun getEffectiveMode(): InterpMode = when (mode) {
        InterpMode.ONNX -> if (onnxReady) InterpMode.ONNX else if (interpreter != null) InterpMode.TFLITE else InterpMode.OPTICAL_FLOW
        InterpMode.TFLITE -> if (interpreter != null) InterpMode.TFLITE else if (onnxReady) InterpMode.ONNX else InterpMode.OPTICAL_FLOW
        else -> if (interpreter != null) InterpMode.TFLITE else if (onnxReady) InterpMode.ONNX else mode
    }

    // Human-readable acceleration info for UI
    fun getAccelerationInfo(): String {
        return when (getEffectiveMode()) {
            InterpMode.ONNX -> onnx?.getActiveProvidersString() ?: "ONNX"
            InterpMode.TFLITE -> when {
                tfliteNnapiActive -> "TFLite NNAPI" + (tfliteNnapiName?.let { " ($it)" } ?: "")
                gpuDelegate != null -> "TFLite GPU"
                else -> "TFLite CPU"
            }
            InterpMode.OPTICAL_FLOW -> if (gpuOpticalFlowReady) "GPU OpticalFlow (NNAPI)" else "CPU OpticalFlow"
            InterpMode.LINEAR -> "CPU Linear"
        }
    }

    fun setAlignToEight(enable: Boolean) {
        alignToEight = enable
    }

    // Public: set NNAPI accelerator name for TFLite (e.g., "qti-dsp", "gpu"). Empty to clear.
    fun setNnapiAcceleratorName(name: String?) {
        nnapiAcceleratorName = name?.trim().takeUnless { it.isNullOrEmpty() }
    }

    // Public: prefer TFLite NNAPI path over ONNX when true
    fun setUseTfliteNnapi(use: Boolean) {
        forceUseNnapiTflite = use
    }

    private fun onnxInterpolate(frame1: Bitmap, frame2: Bitmap): Bitmap {
        val o = onnx ?: return linearInterpolate(frame1, frame2, 0.5f, frame1.width, frame1.height)
        // Do NOT recycle input frames here; midpoint internally scales and recycles its own temporaries.
        val out = if (alignToEight) {
            // Rounded up to multiple-of-8, but capped to avoid giant inference sizes that stall UI.
            val alignedW = ((frame1.width + 7) / 8) * 8
            val alignedH = ((frame1.height + 7) / 8) * 8
            // Conservative caps (reduce from previous 2560x1440 to 1280x768) to keep midpoints responsive.
            val MAX_W = 1280
            val MAX_H = 768
            val cappedW = alignedW.coerceAtMost(MAX_W)
            val cappedH = alignedH.coerceAtMost(MAX_H)
            if (cappedW != alignedW || cappedH != alignedH) {
                android.util.Log.i("FrameInterpolator", "Align×8 active: source=${frame1.width}x${frame1.height} aligned=${alignedW}x${alignedH} capped=${cappedW}x${cappedH}")
            } else {
                android.util.Log.i("FrameInterpolator", "Align×8 active: using ${cappedW}x${cappedH}")
            }
            o.midpoint(frame1, frame2, cappedW, cappedH)
        } else {
            o.midpoint(frame1, frame2)
        }
        return out ?: linearInterpolate(frame1, frame2, 0.5f, frame1.width, frame1.height)
    }

    private fun onnxInterpolate(f1: Bitmap, f2: Bitmap, steps: Int): List<Bitmap> {
        if (steps <= 0) return emptyList()
        // Build a dyadic sequence using midpoints until we have enough frames, then subsample to exact count
    // Keep originals untouched; work on immutable copies to avoid recycling source bitmaps used later.
    var seq: MutableList<Bitmap> = mutableListOf(f1.copy(f1.config, false), f2.copy(f2.config, false))
        fun betweenCount() = seq.size - 2
        val generated: MutableList<Bitmap> = mutableListOf()
        var rounds = 0
        while (betweenCount() < steps && rounds < 6) { // limit rounds to avoid explosion
            val newSeq: MutableList<Bitmap> = mutableListOf()
            for (i in 0 until seq.size - 1) {
                val a = seq[i]
                val b = seq[i + 1]
                // Defensive copies so midpoint never sees a bitmap that might be recycled later in this round
                val ac = a.copy(a.config, false)
                val bc = b.copy(b.config, false)
                val mid = onnxInterpolate(ac, bc)
                ac.recycle(); bc.recycle()
                newSeq.add(a)
                newSeq.add(mid)
            }
            newSeq.add(seq.last())
            seq = newSeq
            rounds++
        }

        val m = betweenCount()
        if (m == steps) {
            // Return all between frames
            for (i in 1 until seq.size - 1) generated.add(seq[i])
        } else {
            // Subsample evenly to match requested steps
            val chosen = HashSet<Int>()
            for (k in 1..steps) {
                val posFloat = k * (m + 1f) / (steps + 1f)
                val idx = posFloat.toInt().coerceIn(1, seq.size - 2)
                if (idx !in chosen) {
                    generated.add(seq[idx])
                    chosen.add(idx)
                } else {
                    // find nearest unused
                    var offset = 1
                    var picked = idx
                    while (true) {
                        val left = (idx - offset).coerceAtLeast(1)
                        val right = (idx + offset).coerceAtMost(seq.size - 2)
                        if (left !in chosen) { picked = left; break }
                        if (right !in chosen) { picked = right; break }
                        offset++
                        if (left == 1 && right == seq.size - 2) break
                    }
                    generated.add(seq[picked])
                    chosen.add(picked)
                }
            }
            // recycle unselected intermediates
            for (i in 1 until seq.size - 1) {
                if (i !in chosen) seq[i].recycle()
            }
        }
        // Recycle endpoints copies
        // Recycle only the working copies (not the originals passed in)
        seq.first().recycle()
        seq.last().recycle()
        return generated
    }

    private fun tfliteInterpolate(frame1: Bitmap, frame2: Bitmap, alpha: Float): Bitmap {
        val intrp = interpreter
        if (intrp == null) {
            // Fallback to linear if model isn't available
            return linearInterpolate(frame1, frame2, alpha, frame1.width, frame1.height)
        }

        // Prepare input: concatenate frames and alpha. This is model-specific; we use a generic path.
        val inputW = INPUT_SIZE
        val inputH = INPUT_SIZE
        val s1 = Bitmap.createScaledBitmap(frame1, inputW, inputH, true)
        val s2 = Bitmap.createScaledBitmap(frame2, inputW, inputH, true)

    // Allocate direct buffer (convert Long size to Int explicitly)
    val inputCapacity = (4L * inputW * inputH * (PIXEL_SIZE * 2 + 1)).toInt()
    val inputBuffer = ByteBuffer.allocateDirect(inputCapacity).order(ByteOrder.nativeOrder())

        fun putBitmapNormalized(bmp: Bitmap) {
            val ints = IntArray(inputW * inputH)
            bmp.getPixels(ints, 0, inputW, 0, 0, inputW, inputH)
            var idx = 0
            for (y in 0 until inputH) {
                for (x in 0 until inputW) {
                    val v = ints[idx++]
                    inputBuffer.putFloat(((v shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    inputBuffer.putFloat(((v shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    inputBuffer.putFloat(((v and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                }
            }
        }

        inputBuffer.clear()
        putBitmapNormalized(s1)
        putBitmapNormalized(s2)
        inputBuffer.putFloat(alpha)
        inputBuffer.rewind()

        // Output buffer (HxWx3)
    val outputCapacity = (4L * inputW * inputH * PIXEL_SIZE).toInt()
    val outputBuffer = ByteBuffer.allocateDirect(outputCapacity).order(ByteOrder.nativeOrder())

        try {
            intrp.run(inputBuffer, outputBuffer)
        } catch (e: Exception) {
            Log.w("FrameInterpolator", "TFLite inference failed, falling back: ${e.message}")
            s1.recycle(); s2.recycle()
            return linearInterpolate(frame1, frame2, alpha, frame1.width, frame1.height)
        }

        outputBuffer.rewind()
        val out = Bitmap.createBitmap(inputW, inputH, Bitmap.Config.ARGB_8888)
        val outPixels = IntArray(inputW * inputH)
        var p = 0
        for (y in 0 until inputH) {
            for (x in 0 until inputW) {
                val r = (outputBuffer.getFloat() * IMAGE_STD + IMAGE_MEAN).toInt().coerceIn(0, 255)
                val g = (outputBuffer.getFloat() * IMAGE_STD + IMAGE_MEAN).toInt().coerceIn(0, 255)
                val b = (outputBuffer.getFloat() * IMAGE_STD + IMAGE_MEAN).toInt().coerceIn(0, 255)
                outPixels[p++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        out.setPixels(outPixels, 0, inputW, 0, 0, inputW, inputH)
        s1.recycle(); s2.recycle()
        return out
    }

    private data class Quad(val w: Int, val h: Int, val f1: Bitmap, val f2: Bitmap)
    
    /**
     * Simple linear interpolation between two frames
     * This is a basic implementation - real frame interpolation uses ML models
     */
    private fun linearInterpolate(frame1: Bitmap, frame2: Bitmap, alpha: Float, width: Int, height: Int): Bitmap {
        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        
        val pixels1 = IntArray(width * height)
        val pixels2 = IntArray(width * height)
        
        frame1.getPixels(pixels1, 0, width, 0, 0, width, height)
        frame2.getPixels(pixels2, 0, width, 0, 0, width, height)
        
        val resultPixels = IntArray(width * height)
        
        for (i in pixels1.indices) {
            val pixel1 = pixels1[i]
            val pixel2 = pixels2[i]
            
            val a1 = (pixel1 shr 24) and 0xff
            val r1 = (pixel1 shr 16) and 0xff
            val g1 = (pixel1 shr 8) and 0xff
            val b1 = pixel1 and 0xff
            
            val a2 = (pixel2 shr 24) and 0xff
            val r2 = (pixel2 shr 16) and 0xff
            val g2 = (pixel2 shr 8) and 0xff
            val b2 = pixel2 and 0xff
            
            val a = ((1 - alpha) * a1 + alpha * a2).toInt()
            val r = ((1 - alpha) * r1 + alpha * r2).toInt()
            val g = ((1 - alpha) * g1 + alpha * g2).toInt()
            val b = ((1 - alpha) * b1 + alpha * b2).toInt()
            
            resultPixels[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
        }
        
        result.setPixels(resultPixels, 0, width, 0, 0, width, height)
        return result
    }
    
    /**
     * Convert bitmap to ByteBuffer for TensorFlow Lite input
     */
    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE)
        byteBuffer.order(ByteOrder.nativeOrder())
        
        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        scaledBitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        
        var pixel = 0
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val value = intValues[pixel++]
                byteBuffer.putFloat(((value shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteBuffer.putFloat(((value shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteBuffer.putFloat(((value and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            }
        }
        
        scaledBitmap.recycle()
        return byteBuffer
    }
    
    fun close() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
        tfliteNnapiActive = false
        tfliteNnapiName = null
        onnx?.close()
        onnx = null
        onnxReady = false
        gpuOpticalFlow?.close()
        gpuOpticalFlow = null
        gpuOpticalFlowReady = false
    }
}
