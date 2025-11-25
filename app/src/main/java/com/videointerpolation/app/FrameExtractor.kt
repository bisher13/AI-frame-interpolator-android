package com.videointerpolation.app

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream

class FrameExtractor(private val context: Context) {
    
    enum class ExportFormat {
        PNG, JPEG
    }
    
    data class ExtractOptions(
        val outputDir: File,
        val format: ExportFormat = ExportFormat.PNG,
        val quality: Int = 95, // For JPEG
        val intervalMs: Long = 1000, // Extract every 1 second
        val specificTimestamps: List<Long>? = null, // Or extract at specific times
        val prefix: String = "frame"
    )
    
    suspend fun extractFrames(
        videoUri: Uri,
        options: ExtractOptions,
        onProgress: (current: Int, total: Int) -> Unit = { _, _ -> }
    ): Result<List<File>> = withContext(Dispatchers.IO) {
        val retriever = MediaMetadataRetriever()
        val extractedFiles = mutableListOf<File>()
        
        try {
            retriever.setDataSource(context, videoUri)
            
            val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLong() ?: return@withContext Result.failure(Exception("Could not read video duration"))
            
            // Create output directory
            if (!options.outputDir.exists()) {
                options.outputDir.mkdirs()
            }
            
            // Determine timestamps to extract
            val timestamps = if (options.specificTimestamps != null) {
                options.specificTimestamps
            } else {
                generateTimestamps(duration, options.intervalMs)
            }
            
            val total = timestamps.size
            
            // Extract frames
            timestamps.forEachIndexed { index, timestampMs ->
                if (!isActive) return@withContext Result.failure(Exception("Extraction cancelled"))
                
                val frame = retriever.getFrameAtTime(
                    timestampMs * 1000, // Convert to microseconds
                    MediaMetadataRetriever.OPTION_CLOSEST_SYNC
                )
                
                if (frame != null) {
                    val file = saveFrame(frame, index, options)
                    extractedFiles.add(file)
                    frame.recycle()
                }
                
                onProgress(index + 1, total)
            }
            
            Result.success(extractedFiles)
            
        } catch (e: Exception) {
            Result.failure(e)
        } finally {
            retriever.release()
        }
    }
    
    private fun generateTimestamps(durationMs: Long, intervalMs: Long): List<Long> {
        val timestamps = mutableListOf<Long>()
        var currentMs = 0L
        
        while (currentMs <= durationMs) {
            timestamps.add(currentMs)
            currentMs += intervalMs
        }
        
        return timestamps
    }
    
    private fun saveFrame(bitmap: Bitmap, index: Int, options: ExtractOptions): File {
        val extension = when (options.format) {
            ExportFormat.PNG -> "png"
            ExportFormat.JPEG -> "jpg"
        }
        
        val file = File(options.outputDir, "${options.prefix}_${String.format("%05d", index)}.$extension")
        
        FileOutputStream(file).use { out ->
            val format = when (options.format) {
                ExportFormat.PNG -> Bitmap.CompressFormat.PNG
                ExportFormat.JPEG -> Bitmap.CompressFormat.JPEG
            }
            bitmap.compress(format, options.quality, out)
        }
        
        return file
    }
}

// Worker for background frame extraction
class FrameExtractorWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {
    
    override suspend fun doWork(): Result {
        val videoUriString = inputData.getString("videoUri") ?: return Result.failure()
        val videoUri = Uri.parse(videoUriString)
        
        val outputDirPath = inputData.getString("outputDir") ?: return Result.failure()
        val outputDir = File(outputDirPath)
        
        val intervalMs = inputData.getLong("intervalMs", 1000)
        val format = when (inputData.getString("format")) {
            "JPEG" -> FrameExtractor.ExportFormat.JPEG
            else -> FrameExtractor.ExportFormat.PNG
        }
        
        val options = FrameExtractor.ExtractOptions(
            outputDir = outputDir,
            format = format,
            intervalMs = intervalMs
        )
        
        val extractor = FrameExtractor(applicationContext)
        
        val extractResult = extractor.extractFrames(videoUri, options) { current, total ->
            setProgressAsync(
                workDataOf(
                    "progress" to (current * 100 / total),
                    "current" to current,
                    "total" to total
                )
            )
        }
        
        return if (extractResult.isSuccess) {
            val files = extractResult.getOrNull()!!
            Result.success(
                workDataOf(
                    "frameCount" to files.size,
                    "outputDir" to outputDirPath
                )
            )
        } else {
            Result.failure(
                workDataOf("error" to (extractResult.exceptionOrNull()?.message ?: "Unknown error"))
            )
        }
    }
}
