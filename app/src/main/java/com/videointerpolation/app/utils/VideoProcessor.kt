package com.videointerpolation.app.utils

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.Build
import java.io.BufferedOutputStream
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class VideoProcessor(private val context: Context) {

    /**
     * Extract frames from a video file
     */
    suspend fun extractFrames(
        videoUri: Uri,
        outputDir: File,
        onProgress: ((current: Int, total: Int) -> Unit)? = null,
        maxWidth: Int = 720,
        trimStartMs: Long = 0L,
        trimEndMs: Long = 0L
    ): List<File> = withContext(Dispatchers.IO) {
    val frames = mutableListOf<File>()
    val retriever = MediaMetadataRetriever()
        
        try {
            retriever.setDataSource(context, videoUri)
            
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
            val srcW = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toInt() ?: 0
            val srcH = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toInt() ?: 0
            val metaFps = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)?.toFloat()
            val frameRate = when {
                metaFps != null && metaFps > 0 -> metaFps
                else -> 30f
            }
            val frameIntervalUs = (1_000_000f / frameRate).toLong().coerceAtLeast(1L)
            
            // Apply trim if specified
            val startMs = if (trimStartMs > 0 && trimEndMs > trimStartMs) trimStartMs else 0L
            val endMs = if (trimEndMs > trimStartMs && trimEndMs <= durationMs) trimEndMs else durationMs
            val effectiveDurationMs = endMs - startMs
            
            val totalFrames = if (effectiveDurationMs > 0) ((effectiveDurationMs * frameRate) / 1000f).toInt() else 0

            // Compute target scale once; prefer native scaled decode on API 27+
            val (tW, tH) = if (srcW > 0 && srcH > 0 && srcW > maxWidth) {
                val scale = maxWidth.toFloat() / srcW
                val newW = maxWidth
                val newH = (srcH * scale).toInt().coerceAtLeast(1)
                newW to newH
            } else {
                (if (srcW > 0) srcW else maxWidth) to (if (srcH > 0) srcH else maxWidth)
            }
            
            // Start from trim start time
            var frameTime = startMs * 1000L
            val endTimeUs = endMs * 1000L
            var frameCount = 0
            
            while (frameTime < endTimeUs) {
                val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                    // Use native scaled frame extraction for speed (API 27+)
                    retriever.getScaledFrameAtTime(frameTime, MediaMetadataRetriever.OPTION_CLOSEST, tW, tH)
                } else {
                    retriever.getFrameAtTime(frameTime, MediaMetadataRetriever.OPTION_CLOSEST)
                }
                
                if (bitmap != null) {
                    // Fallback downscale if pre-27 or dimension mismatch
                    val scaled = if (bitmap.width != tW || bitmap.height != tH) {
                        Bitmap.createScaledBitmap(bitmap, tW, tH, true)
                    } else bitmap

                    val frameFile = File(outputDir, "frame_${String.format("%05d", frameCount)}.jpg")
                    BufferedOutputStream(FileOutputStream(frameFile), 64 * 1024).use { out ->
                        scaled.compress(Bitmap.CompressFormat.JPEG, 85, out)
                    }
                    frames.add(frameFile)
                    if (scaled !== bitmap) scaled.recycle()
                    bitmap.recycle()
                    frameCount++
                    onProgress?.invoke(frameCount, totalFrames)
                }
                
                frameTime += frameIntervalUs
            }
            
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            retriever.release()
        }
        
        frames
    }
    
    /**
     * Get video metadata
     */
    fun getVideoMetadata(videoUri: Uri): VideoMetadata {
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(context, videoUri)
            
            val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toInt() ?: 0
            val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toInt() ?: 0
            val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0
            val frameRate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)?.toFloat() ?: 30f
            
            VideoMetadata(width, height, duration, frameRate)
        } finally {
            retriever.release()
        }
    }
    
    data class VideoMetadata(
        val width: Int,
        val height: Int,
        val duration: Long,
        val frameRate: Float
    )

    fun preprocessFrame(frame: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        // Downscale the frame to the target resolution
        val scaledFrame = Bitmap.createScaledBitmap(frame, targetWidth, targetHeight, true)
        return scaledFrame
    }

    fun postprocessFrame(frame: Bitmap, originalWidth: Int, originalHeight: Int): Bitmap {
        // Upscale the frame back to the original resolution
        val restoredFrame = Bitmap.createScaledBitmap(frame, originalWidth, originalHeight, true)
        return restoredFrame
    }
}
