package com.videointerpolation.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.*
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * GPU-accelerated optical flow using a neural network (ONNX Runtime + NNAPI).
 * Uses a lightweight optical flow model (PWC-Net, FastFlowNet, or similar) for fast GPU inference.
 * Falls back to CPU block-matching if model unavailable.
 */
class GpuOpticalFlowInterpolator(private val context: Context) {
    
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var useGpu: Boolean = false
    private var nnapiDeviceName: String? = null
    
    // Fallback CPU optical flow
    private val cpuFlow = OpticalFlowInterpolator(blockSize = 8, searchRadius = 4)
    
    fun initialize(modelPath: String = "optical_flow.onnx"): Boolean {
        return try {
            val e = OrtEnvironment.getEnvironment()
            val so = OrtSession.SessionOptions()
            
            // Enable NNAPI for GPU acceleration
            try {
                val nnapiFlags = mutableMapOf<String, String>()
                nnapiDeviceName?.let { device ->
                    if (device.isNotEmpty()) {
                        nnapiFlags["nnapi_accelerator_name"] = device
                    }
                }
                nnapiFlags["allow_fp16"] = "1"
                
                if (nnapiFlags.isNotEmpty()) {
                    try {
                        val method = so.javaClass.getMethod("addNnapi", Map::class.java)
                        method.invoke(so, nnapiFlags)
                        Log.d(TAG, "NNAPI enabled for optical flow with options: $nnapiFlags")
                    } catch (_: NoSuchMethodException) {
                        so.addNnapi()
                        Log.d(TAG, "NNAPI enabled (basic)")
                    }
                } else {
                    so.addNnapi()
                    Log.d(TAG, "NNAPI enabled")
                }
                useGpu = true
            } catch (t: Throwable) {
                Log.w(TAG, "NNAPI not available for optical flow: ${t.message}")
                useGpu = false
            }
            
            // Load model from assets
            val tmpFile = extractAssetToCache(modelPath)
            if (!tmpFile.exists()) {
                Log.w(TAG, "Optical flow model not found: $modelPath, using CPU fallback")
                return false
            }
            
            val sess = e.createSession(tmpFile.absolutePath, so)
            env = e
            session = sess
            
            Log.i(TAG, "GPU optical flow initialized: ${if (useGpu) "NNAPI" else "CPU"}")
            true
        } catch (e: Exception) {
            Log.w(TAG, "Failed to init GPU optical flow: ${e.message}, using CPU fallback")
            false
        }
    }
    
    /**
     * Compute optical flow from frame1 to frame2 using GPU.
     * Returns intermediate frame at position alpha (0.0 = frame1, 1.0 = frame2).
     */
    fun interpolate(frame1: Bitmap, frame2: Bitmap, alpha: Float): Bitmap {
        val sess = session
        if (sess == null || !useGpu) {
            // Fallback to CPU optical flow
            return cpuFlow.interpolate(frame1, frame2, alpha)
        }
        
        return try {
            // Prepare input: stack frame1 and frame2 as [1, 6, H, W] (NCHW)
            val width = minOf(frame1.width, frame2.width)
            val height = minOf(frame1.height, frame2.height)
            
            // Resize frames if needed (optical flow models typically work on 256x256 or 384x384)
            val targetSize = 256
            val resizedFrame1 = if (width != targetSize || height != targetSize) {
                Bitmap.createScaledBitmap(frame1, targetSize, targetSize, true)
            } else frame1
            
            val resizedFrame2 = if (width != targetSize || height != targetSize) {
                Bitmap.createScaledBitmap(frame2, targetSize, targetSize, true)
            } else frame2
            
            // Convert to float buffer [1, 6, H, W]: frame1 RGB + frame2 RGB
            val inputBuffer = ByteBuffer.allocateDirect(4 * 6 * targetSize * targetSize)
                .order(ByteOrder.nativeOrder())
            val floatBuffer = inputBuffer.asFloatBuffer()
            
            val pixels1 = IntArray(targetSize * targetSize)
            val pixels2 = IntArray(targetSize * targetSize)
            resizedFrame1.getPixels(pixels1, 0, targetSize, 0, 0, targetSize, targetSize)
            resizedFrame2.getPixels(pixels2, 0, targetSize, 0, 0, targetSize, targetSize)
            
            // Pack as NCHW: [R1, G1, B1, R2, G2, B2]
            for (c in 0 until 3) {
                for (y in 0 until targetSize) {
                    for (x in 0 until targetSize) {
                        val idx = y * targetSize + x
                        val pixel1 = pixels1[idx]
                        val value1 = when (c) {
                            0 -> ((pixel1 shr 16) and 0xff) / 255f
                            1 -> ((pixel1 shr 8) and 0xff) / 255f
                            else -> (pixel1 and 0xff) / 255f
                        }
                        floatBuffer.put(value1)
                    }
                }
            }
            
            for (c in 0 until 3) {
                for (y in 0 until targetSize) {
                    for (x in 0 until targetSize) {
                        val idx = y * targetSize + x
                        val pixel2 = pixels2[idx]
                        val value2 = when (c) {
                            0 -> ((pixel2 shr 16) and 0xff) / 255f
                            1 -> ((pixel2 shr 8) and 0xff) / 255f
                            else -> (pixel2 and 0xff) / 255f
                        }
                        floatBuffer.put(value2)
                    }
                }
            }
            
            // Create ONNX tensor
            val shape = longArrayOf(1, 6, targetSize.toLong(), targetSize.toLong())
            val tensor = OnnxTensor.createTensor(env, inputBuffer, shape)
            
            // Run inference
            val inputName = sess.inputNames.first()
            val inputs = mapOf(inputName to tensor)
            val outputs = sess.run(inputs)
            
            // Extract flow [1, 2, H, W]: dx, dy
            val flowTensor = outputs[0].value as OnnxTensor
            val flowData = flowTensor.floatBuffer
            
            // Warp and blend frames using the computed flow
            val result = warpAndBlend(frame1, frame2, flowData, targetSize, targetSize, alpha)
            
            tensor.close()
            outputs.close()
            
            // Resize back to original size if needed
            if (width != targetSize || height != targetSize) {
                val scaled = Bitmap.createScaledBitmap(result, width, height, true)
                result.recycle()
                scaled
            } else {
                result
            }
            
        } catch (e: Exception) {
            Log.w(TAG, "GPU optical flow failed: ${e.message}, using CPU fallback")
            cpuFlow.interpolate(frame1, frame2, alpha)
        }
    }
    
    private fun warpAndBlend(
        frame1: Bitmap,
        frame2: Bitmap,
        flowData: java.nio.FloatBuffer,
        flowW: Int,
        flowH: Int,
        alpha: Float
    ): Bitmap {
        val width = minOf(frame1.width, frame2.width)
        val height = minOf(frame1.height, frame2.height)
        
        val pixels1 = IntArray(width * height)
        val pixels2 = IntArray(width * height)
        frame1.getPixels(pixels1, 0, width, 0, 0, width, height)
        frame2.getPixels(pixels2, 0, width, 0, 0, width, height)
        
        val out = IntArray(width * height)
        
        // Scale flow to match image resolution
        val scaleX = width.toFloat() / flowW
        val scaleY = height.toFloat() / flowH
        
        for (y in 0 until height) {
            for (x in 0 until width) {
                val flowX = (x / scaleX).toInt().coerceIn(0, flowW - 1)
                val flowY = (y / scaleY).toInt().coerceIn(0, flowH - 1)
                val flowIdx = flowY * flowW + flowX
                
                val dx = flowData.get(flowIdx) * scaleX
                val dy = flowData.get(flowW * flowH + flowIdx) * scaleY
                
                // Warp both frames toward intermediate position
                val x1 = x - alpha * dx
                val y1 = y - alpha * dy
                val x2 = x + (1f - alpha) * dx
                val y2 = y + (1f - alpha) * dy
                
                val c1 = bilinearSample(pixels1, width, height, x1, y1)
                val c2 = bilinearSample(pixels2, width, height, x2, y2)
                
                // Blend
                val a = (((c1 ushr 24) * (1f - alpha) + (c2 ushr 24) * alpha)).toInt().coerceIn(0, 255)
                val r = ((((c1 shr 16) and 0xff) * (1f - alpha) + ((c2 shr 16) and 0xff) * alpha)).toInt().coerceIn(0, 255)
                val g = ((((c1 shr 8) and 0xff) * (1f - alpha) + ((c2 shr 8) and 0xff) * alpha)).toInt().coerceIn(0, 255)
                val b = (((c1 and 0xff) * (1f - alpha) + (c2 and 0xff) * alpha)).toInt().coerceIn(0, 255)
                
                out[y * width + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        
        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        result.setPixels(out, 0, width, 0, 0, width, height)
        return result
    }
    
    private fun bilinearSample(pixels: IntArray, w: Int, h: Int, fx: Float, fy: Float): Int {
        val x0 = fx.toInt().coerceIn(0, w - 1)
        val y0 = fy.toInt().coerceIn(0, h - 1)
        val x1 = (x0 + 1).coerceIn(0, w - 1)
        val y1 = (y0 + 1).coerceIn(0, h - 1)
        
        val dx = fx - x0
        val dy = fy - y0
        
        val c00 = pixels[y0 * w + x0]
        val c10 = pixels[y0 * w + x1]
        val c01 = pixels[y1 * w + x0]
        val c11 = pixels[y1 * w + x1]
        
        val a = interpolateChannel(c00 ushr 24, c10 ushr 24, c01 ushr 24, c11 ushr 24, dx, dy)
        val r = interpolateChannel((c00 shr 16) and 0xff, (c10 shr 16) and 0xff, (c01 shr 16) and 0xff, (c11 shr 16) and 0xff, dx, dy)
        val g = interpolateChannel((c00 shr 8) and 0xff, (c10 shr 8) and 0xff, (c01 shr 8) and 0xff, (c11 shr 8) and 0xff, dx, dy)
        val b = interpolateChannel(c00 and 0xff, c10 and 0xff, c01 and 0xff, c11 and 0xff, dx, dy)
        
        return (a shl 24) or (r shl 16) or (g shl 8) or b
    }
    
    private fun interpolateChannel(v00: Int, v10: Int, v01: Int, v11: Int, dx: Float, dy: Float): Int {
        val top = v00 * (1f - dx) + v10 * dx
        val bot = v01 * (1f - dx) + v11 * dx
        return (top * (1f - dy) + bot * dy).toInt().coerceIn(0, 255)
    }
    
    fun setNnapiDeviceName(deviceName: String?) {
        nnapiDeviceName = deviceName?.takeIf { it.isNotEmpty() }
    }
    
    fun close() {
        session?.close()
        session = null
        env = null
    }
    
    private fun extractAssetToCache(assetName: String): File {
        val f = File(context.cacheDir, assetName)
        if (f.exists()) return f
        try {
            context.assets.open(assetName).use { input ->
                f.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        } catch (e: Exception) {
            // Asset doesn't exist
        }
        return f
    }
    
    companion object {
        private const val TAG = "GpuOpticalFlow"
    }
}
