package com.videointerpolation.app.work

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.work.*
import com.videointerpolation.app.data.AppSettings
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Worker for batch processing multiple videos.
 * Processes videos sequentially to avoid memory issues.
 */
class BatchInterpolationWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    private val settings = AppSettings.getInstance(context)
    
    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        try {
            val videoUrisString = inputData.getStringArray(KEY_VIDEO_URIS) ?: return@withContext Result.failure()
            val videoUris = videoUrisString.map { Uri.parse(it) }
            
            setForeground(createForegroundInfo(0, videoUris.size))
            
            val results = mutableListOf<String>()
            val failures = mutableListOf<String>()
            
            videoUris.forEachIndexed { index, uri ->
                if (isStopped) {
                    return@withContext Result.failure()
                }
                
                setForeground(createForegroundInfo(index + 1, videoUris.size))
                
                try {
                    Log.i(TAG, "Processing video ${index + 1}/${videoUris.size}: $uri")
                    
                    // Queue individual video interpolation
                    val videoWorkRequest = OneTimeWorkRequestBuilder<VideoInterpolationWorker>()
                        .setInputData(
                            workDataOf(
                                VideoInterpolationWorker.KEY_INPUT_URI to uri.toString(),
                                VideoInterpolationWorker.KEY_MULTIPLIER to (settings.getInterpolationSteps() + 1),
                                VideoInterpolationWorker.KEY_BITRATE_MBPS to settings.exportBitrateMbps.toInt(),
                                VideoInterpolationWorker.KEY_ALIGN_EIGHT to settings.enableFramePadding,
                                "batch_mode" to true
                            )
                        )
                        .build()
                    
                    // Enqueue and wait for completion
                    val workManager = WorkManager.getInstance(applicationContext)
                    workManager.enqueue(videoWorkRequest)
                    
                    // Monitor work status
                    var completed = false
                    var attempts = 0
                    while (!completed && attempts < 600) { // 10 min timeout
                        val workInfo = workManager.getWorkInfoById(videoWorkRequest.id).await()
                        when (workInfo?.state) {
                            WorkInfo.State.SUCCEEDED -> {
                                completed = true
                                val outputPath = workInfo.outputData.getString("file_path")
                                results.add(outputPath ?: "unknown")
                                Log.i(TAG, "Video ${index + 1} completed: $outputPath")
                            }
                            WorkInfo.State.FAILED, WorkInfo.State.CANCELLED -> {
                                completed = true
                                failures.add(uri.toString())
                                Log.w(TAG, "Video ${index + 1} failed")
                            }
                            else -> {
                                kotlinx.coroutines.delay(1000)
                                attempts++
                            }
                        }
                    }
                    
                    if (!completed) {
                        failures.add(uri.toString())
                        Log.w(TAG, "Video ${index + 1} timed out")
                    }
                    
                } catch (e: Exception) {
                    Log.e(TAG, "Error processing video ${index + 1}: ${e.message}")
                    failures.add(uri.toString())
                }
            }
            
            val outputData = workDataOf(
                KEY_RESULTS to results.toTypedArray(),
                KEY_FAILURES to failures.toTypedArray(),
                KEY_TOTAL to videoUris.size,
                KEY_SUCCESS_COUNT to results.size,
                KEY_FAILURE_COUNT to failures.size
            )
            
            if (failures.isEmpty()) {
                Result.success(outputData)
            } else {
                Result.failure(outputData)
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Batch processing failed: ${e.message}", e)
            Result.failure()
        }
    }

    private fun createForegroundInfo(current: Int, total: Int): ForegroundInfo {
        val title = if (current == 0) {
            "Starting batch processing..."
        } else {
            "Processing video $current of $total"
        }
        
        val notification = androidx.core.app.NotificationCompat.Builder(
            applicationContext,
            "video_interpolation_channel"
        )
            .setContentTitle("Batch Video Interpolation")
            .setContentText(title)
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setProgress(total, current, false)
            .setOngoing(true)
            .build()
        
        return ForegroundInfo(NOTIFICATION_ID, notification)
    }

    companion object {
        private const val TAG = "BatchInterpolation"
        private const val NOTIFICATION_ID = 2001
        
        const val KEY_VIDEO_URIS = "video_uris"
        const val KEY_RESULTS = "results"
        const val KEY_FAILURES = "failures"
        const val KEY_TOTAL = "total"
        const val KEY_SUCCESS_COUNT = "success_count"
        const val KEY_FAILURE_COUNT = "failure_count"
        
        /**
         * Start batch processing for multiple videos.
         */
        fun startBatchProcessing(context: Context, videoUris: List<Uri>): String {
            val inputData = workDataOf(
                KEY_VIDEO_URIS to videoUris.map { it.toString() }.toTypedArray()
            )
            
            val workRequest = OneTimeWorkRequestBuilder<BatchInterpolationWorker>()
                .setInputData(inputData)
                .setExpedited(OutOfQuotaPolicy.RUN_AS_NON_EXPEDITED_WORK_REQUEST)
                .build()
            
            WorkManager.getInstance(context).enqueue(workRequest)
            
            return workRequest.id.toString()
        }
    }
}
