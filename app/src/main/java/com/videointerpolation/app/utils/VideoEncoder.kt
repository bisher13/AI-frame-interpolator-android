package com.videointerpolation.app.utils

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import android.graphics.BitmapFactory

class VideoEncoder {
    
    companion object {
        private const val MIME_TYPE = "video/avc"
        private const val FRAME_RATE = 60 // Target interpolated frame rate
        private const val I_FRAME_INTERVAL = 1
        private const val TIMEOUT_USEC = 10000L
    }
    
    /**
     * Encode frames into a video file
     */
    suspend fun encodeFramesToVideo(
        frames: List<File>,
        outputFile: File,
        width: Int,
        height: Int,
        bitRate: Int = 6000000,
        onProgress: ((current: Int, total: Int) -> Unit)? = null
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            // Enforce even dimensions for YUV420
            val encWidth = if (width % 2 == 0) width else width - 1
            val encHeight = if (height % 2 == 0) height else height - 1

            val format = MediaFormat.createVideoFormat(MIME_TYPE, encWidth, encHeight).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar)
                setInteger(MediaFormat.KEY_BIT_RATE, bitRate)
                setInteger(MediaFormat.KEY_FRAME_RATE, FRAME_RATE)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, I_FRAME_INTERVAL)
            }
            
            val codec = MediaCodec.createEncoderByType(MIME_TYPE)
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            codec.start()
            
            val muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            val trackIndexHolder = intArrayOf(-1)
            var muxerStarted = false
            
            val bufferInfo = MediaCodec.BufferInfo()
            var frameIndex = 0
            val presentationTimeUs = 1000000L / FRAME_RATE
            
            val totalFrames = frames.size
            for (frameFile in frames) {
                val opts = BitmapFactory.Options().apply { inPreferredConfig = android.graphics.Bitmap.Config.RGB_565 }
                val bitmapSrc = BitmapFactory.decodeFile(frameFile.absolutePath, opts)
                if (bitmapSrc != null) {
                    // Prepare bitmap (scale if needed) outside of buffer acquisition so we can recycle properly
                    var workingBitmap = bitmapSrc
                    if (bitmapSrc.width != encWidth || bitmapSrc.height != encHeight) {
                        workingBitmap = Bitmap.createScaledBitmap(bitmapSrc, encWidth, encHeight, true)
                    }

                    val inputBufferIndex = codec.dequeueInputBuffer(TIMEOUT_USEC)
                    if (inputBufferIndex >= 0) {
                        val inputBuffer = codec.getInputBuffer(inputBufferIndex)
                        inputBuffer?.clear()

                        val yuvData = convertBitmapToI420(workingBitmap)
                        inputBuffer?.put(yuvData)

                        codec.queueInputBuffer(
                            inputBufferIndex,
                            0,
                            yuvData.size,
                            frameIndex * presentationTimeUs,
                            0
                        )
                        frameIndex++
                        onProgress?.invoke(frameIndex.coerceAtMost(totalFrames), totalFrames)
                    }
                    if (workingBitmap !== bitmapSrc) workingBitmap.recycle()
                    bitmapSrc.recycle()
                }
                
                // Get encoded data
                drainEncoder(codec, muxer, bufferInfo, false, trackIndexHolder) { started ->
                    muxerStarted = started
                }
            }
            
            // Signal end of stream
            val inputBufferIndex = codec.dequeueInputBuffer(TIMEOUT_USEC)
            if (inputBufferIndex >= 0) {
                codec.queueInputBuffer(inputBufferIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
            }
            
            // Drain remaining data
            drainEncoder(codec, muxer, bufferInfo, true, trackIndexHolder) { }
            
            codec.stop()
            codec.release()
            muxer.stop()
            muxer.release()
            
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
    
    private fun drainEncoder(
        codec: MediaCodec,
        muxer: MediaMuxer,
        bufferInfo: MediaCodec.BufferInfo,
        endOfStream: Boolean,
        trackIndexHolder: IntArray,
        onMuxerStarted: (Boolean) -> Unit
    ) {
        var localMuxerStarted = trackIndexHolder[0] >= 0
        while (true) {
            val outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_USEC)
            
            when {
                outputBufferIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) break
                }
                outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    val newFormat = codec.outputFormat
                    val trackIndex = muxer.addTrack(newFormat)
                    muxer.start()
                    trackIndexHolder[0] = trackIndex
                    localMuxerStarted = true
                    onMuxerStarted(true)
                }
                outputBufferIndex >= 0 -> {
                    val encodedData = codec.getOutputBuffer(outputBufferIndex)
                    
                    if (encodedData != null && bufferInfo.size > 0) {
                        encodedData.position(bufferInfo.offset)
                        encodedData.limit(bufferInfo.offset + bufferInfo.size)
                        
                        try {
                            // Use the actual video track index once the muxer has started
                            if (localMuxerStarted && trackIndexHolder[0] >= 0) {
                                muxer.writeSampleData(trackIndexHolder[0], encodedData, bufferInfo)
                            }
                        } catch (e: Exception) {
                            // Muxer might not be started yet
                        }
                    }
                    
                    codec.releaseOutputBuffer(outputBufferIndex, false)
                    
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        break
                    }
                }
            }
        }
    }
    
    private fun convertBitmapToI420(bitmap: Bitmap): ByteArray {
        val width = bitmap.width
        val height = bitmap.height
        val frameSize = width * height
        val qFrameSize = frameSize / 4
        val yuv = ByteArray(frameSize + 2 * qFrameSize)

        val pixels = IntArray(frameSize)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        var yIndex = 0
        var uIndex = frameSize
        var vIndex = frameSize + qFrameSize

        for (j in 0 until height) {
            for (i in 0 until width) {
                val c = pixels[j * width + i]
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF

                // BT.601 full-range approximate
                val y = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                yuv[yIndex++] = y.coerceIn(0, 255).toByte()
            }
        }

        for (j in 0 until height step 2) {
            for (i in 0 until width step 2) {
                var rSum = 0
                var gSum = 0
                var bSum = 0
                for (dy in 0..1) {
                    for (dx in 0..1) {
                        val c = pixels[(j + dy) * width + (i + dx)]
                        rSum += (c shr 16) and 0xFF
                        gSum += (c shr 8) and 0xFF
                        bSum += c and 0xFF
                    }
                }
                val r = rSum / 4
                val g = gSum / 4
                val b = bSum / 4
                val u = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                val v = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                yuv[uIndex++] = u.coerceIn(0, 255).toByte()
                yuv[vIndex++] = v.coerceIn(0, 255).toByte()
            }
        }

        return yuv
    }
}
