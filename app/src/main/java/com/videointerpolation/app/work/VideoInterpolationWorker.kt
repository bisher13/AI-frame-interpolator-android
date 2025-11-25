package com.videointerpolation.app.work

import android.content.Context
import android.app.PendingIntent
import android.content.Intent
import android.net.Uri
import androidx.work.CoroutineWorker
import androidx.work.ForegroundInfo
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build
import com.videointerpolation.app.R
import com.videointerpolation.app.MainActivity
import com.videointerpolation.app.ml.FrameInterpolator
import com.videointerpolation.app.utils.VideoProcessor
import com.videointerpolation.app.utils.VideoEncoder
import com.videointerpolation.app.utils.PerformanceMonitor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import android.provider.MediaStore
import android.content.ContentValues
import android.os.Environment

/**
 * Worker that performs full video interpolation pipeline in the background.
 * Provides a foreground service notification so work can continue while app is minimized or screen locked.
 */
class VideoInterpolationWorker(
    appContext: Context,
    params: WorkerParameters
) : CoroutineWorker(appContext, params) {

    companion object {
        const val KEY_INPUT_URI = "input_uri"
        const val KEY_MULTIPLIER = "multiplier"
        const val KEY_BITRATE_MBPS = "bitrate_mbps"
        const val KEY_ALIGN_EIGHT = "align_eight"
        private const val CHANNEL_ID = "video_interp_channel"
        private const val NOTIF_ID = 2020
    }

    private val processor = VideoProcessor(appContext)
    private val interpolator = FrameInterpolator(appContext, com.videointerpolation.app.ml.InterpMode.ONNX)
    private val encoder = VideoEncoder()
    private val perfMonitor = PerformanceMonitor(appContext)
    // Human-readable acceleration/provider info (e.g., "ONNX NNAPI", "TFLite GPU", "CPU OpticalFlow")
    private var accelInfo: String = "Detecting…"

    override suspend fun getForegroundInfo(): ForegroundInfo {
        ensureChannel()
        val contentIntent = PendingIntent.getActivity(
            applicationContext,
            0,
            Intent(applicationContext, MainActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            },
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) PendingIntent.FLAG_IMMUTABLE else 0
        )
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle("Interpolating video • ${accelInfo}")
            .setContentText("Preparing...")
            .setContentIntent(contentIntent)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
        // Specify service types that match manifest on newer Android; fallback to no type if not supported
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ForegroundInfo(
                NOTIF_ID,
                notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            )
        } else ForegroundInfo(NOTIF_ID, notification)
    }

    override suspend fun doWork(): Result = withContext(Dispatchers.Default) {
    // Start foreground immediately so notification appears
    try { setForeground(getForegroundInfo()) } catch (_: Throwable) { try { setForegroundAsync(getForegroundInfo()) } catch (_: Throwable) {} }

    val uriStr = inputData.getString(KEY_INPUT_URI) ?: return@withContext Result.failure()
        val multiplier = inputData.getInt(KEY_MULTIPLIER, 2).coerceAtLeast(2)
        val videoUri = Uri.parse(uriStr)

    // Optional dimension alignment for internal inference
    val align = inputData.getBoolean(KEY_ALIGN_EIGHT, false)
    interpolator.setAlignToEight(align)
    interpolator.initialize(modelPath = "rife_fp16.tflite", onnxModelPath = "rife_fp16.onnx")
    // Capture acceleration/provider info for user visibility
    try {
        accelInfo = interpolator.getAccelerationInfo()
        // Refresh the foreground notification title to include acceleration info
        ensureChannel()
        val nm = NotificationManagerCompat.from(applicationContext)
        val contentIntent = PendingIntent.getActivity(
            applicationContext,
            0,
            Intent(applicationContext, MainActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            },
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) PendingIntent.FLAG_IMMUTABLE else 0
        )
        val notif = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle("Interpolating video • ${accelInfo}")
            .setContentText("Preparing…")
            .setContentIntent(contentIntent)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
        nm.notify(NOTIF_ID, notif)
    } catch (_: Throwable) { }

        try {
            // Load trim settings
            val settings = com.videointerpolation.app.data.AppSettings.getInstance(applicationContext)
            val trimStartMs = if (settings.trimEnabled) settings.trimStart else 0L
            val trimEndMs = if (settings.trimEnabled) settings.trimEnd else 0L
            
            val srcMeta = processor.getVideoMetadata(videoUri)
            val trimInfo = if (settings.trimEnabled) " (trimmed ${formatTime(trimStartMs)}-${formatTime(trimEndMs)})" else ""
            updateProgress(0f, "Extracting frames at ${srcMeta.width}x${srcMeta.height}${trimInfo}")
            val framesDir = File(applicationContext.cacheDir, "bg_frames").apply { deleteRecursively(); mkdirs() }
                val frames = processor.extractFrames(
                    videoUri, 
                    framesDir, 
                    { current, total ->
                        val frac = if (total > 0) current.toFloat() / total else 0f
                        // Provide preview of an extracted frame occasionally
                        val preview = if (current % 10 == 0) File(framesDir, String.format("frame_%05d.jpg", current)).takeIf { it.exists() }?.absolutePath else null
                        kotlinx.coroutines.runBlocking { updateProgress(frac * 0.3f, "Extracting frames ${current}/${total}", preview) }
                    }, 
                    maxWidth = srcMeta.width.coerceAtLeast(1),
                    trimStartMs = trimStartMs,
                    trimEndMs = trimEndMs
                )

            if (frames.size < 2) {
                updateProgress(1f, "Not enough frames")
                return@withContext Result.failure()
            }

            updateProgress(0.31f, "Interpolating (${multiplier}x) at source resolution")
            val interpDir = File(applicationContext.cacheDir, "bg_interpolated").apply { deleteRecursively(); mkdirs() }

            val stepsPerPair = (multiplier - 1)
            val allFrames = mutableListOf<File>()
            var produced = 0
            val totalToProduce = frames.size + (frames.size - 1) * stepsPerPair
            
            // Start performance monitoring
            perfMonitor.start(totalToProduce)
            
            for (i in 0 until frames.size - 1) {
                // Copy original
                val dstOrig = File(interpDir, "frame_${String.format("%05d", allFrames.size)}.jpg")
                frames[i].copyTo(dstOrig, overwrite = true)
                allFrames.add(dstOrig); produced++
                val b1 = android.graphics.BitmapFactory.decodeFile(frames[i].absolutePath)
                val b2 = android.graphics.BitmapFactory.decodeFile(frames[i+1].absolutePath)
                val mids = interpolator.interpolateFrames(b1, b2, stepsPerPair)
                mids.forEach { bmp ->
                    val f = File(interpDir, "frame_${String.format("%05d", allFrames.size)}.jpg")
                    java.io.FileOutputStream(f).use { out -> bmp.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out) }
                    allFrames.add(f); produced++
                    bmp.recycle()
                        perfMonitor.updateFrameCount(produced)
                        val stats = perfMonitor.getStats()
                        val previewPath = f.absolutePath
                        val statusMsg = "Interpolating ${produced}/${totalToProduce} • ${String.format("%.1f", stats.fps)} FPS • ETA: ${stats.eta}"
                        updateProgress(0.31f + 0.4f * (produced.toFloat() / totalToProduce), statusMsg, previewPath)
                }
                b1.recycle(); b2.recycle()
            }
            // last frame
            val last = File(interpDir, "frame_${String.format("%05d", allFrames.size)}.jpg")
            frames.last().copyTo(last, overwrite = true)
            allFrames.add(last); produced++
            updateProgress(0.71f, "Encoding video")

            // Determine output dimensions based on first frame
            val firstBmpOpts = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
            android.graphics.BitmapFactory.decodeFile(allFrames.first().absolutePath, firstBmpOpts)
            var encW = firstBmpOpts.outWidth
            var encH = firstBmpOpts.outHeight
            if (encW % 2 != 0) encW -= 1
            if (encH % 2 != 0) encH -= 1

            val outFile = File(applicationContext.getExternalFilesDir(null), "bg_interpolated_${System.currentTimeMillis()}.mp4")
            // Adaptive bitrate scaling based on resolution (target ~7 bits per pixel @ 60fps)
            // Allow caller override via input data (in Mbps); fallback to adaptive calculation if not provided
            val userMbps = inputData.getInt("bitrate_mbps", -1)
            val bitRate = if (userMbps > 0) (userMbps * 1_000_000) else (encW.toLong() * encH.toLong() * 60L * 7L / 100L).toInt().coerceAtLeast(3_000_000)
            val ok = encoder.encodeFramesToVideo(allFrames, outFile, encW, encH, bitRate = bitRate) { cur, tot ->
                val previewFrame = if (cur < allFrames.size) allFrames[cur].absolutePath else null
                kotlinx.coroutines.runBlocking { updateProgress(0.71f + 0.29f * (cur.toFloat() / tot), "Encoding ${cur}/${tot}", previewFrame) }
            }
            if (!ok) {
                updateProgress(1f, "Encoding failed")
                return@withContext Result.failure()
            }
            // Export to MediaStore so it appears in Gallery
            val galleryUri = saveToMediaStore(outFile)
            updateProgress(1f, if (galleryUri != null) "Saved to Gallery" else "Saved (app folder)")
            val output = androidx.work.Data.Builder()
                .putString("gallery_uri", galleryUri?.toString())
                .putString("file_path", outFile.absolutePath)
                .build()
            Result.success(output)
        } catch (e: Exception) {
            updateProgress(1f, "Error: ${e.message}")
            Result.failure()
        } finally {
            interpolator.close()
        }
    }

    private suspend fun updateProgress(frac: Float, msg: String, previewPath: String? = null) {
        val pct = (frac.coerceIn(0f,1f) * 100).toInt()
        ensureChannel()
        val contentIntent = PendingIntent.getActivity(
            applicationContext,
            0,
            Intent(applicationContext, MainActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            },
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) PendingIntent.FLAG_IMMUTABLE else 0
        )
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle("Interpolating video • ${accelInfo}")
            .setContentText(msg)
            .setContentIntent(contentIntent)
            .setProgress(100, pct, false)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
        NotificationManagerCompat.from(applicationContext).notify(NOTIF_ID, notification)
        // Report progress to WorkManager so UI can observe
        val data = if (previewPath != null) workDataOf("progress" to pct, "message" to msg, "preview" to previewPath, "accel" to accelInfo)
                    else workDataOf("progress" to pct, "message" to msg, "accel" to accelInfo)
        setProgress(data)
    }

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            if (nm.getNotificationChannel(CHANNEL_ID) == null) {
                nm.createNotificationChannel(NotificationChannel(CHANNEL_ID, "Video Interpolation", NotificationManager.IMPORTANCE_DEFAULT))
            }
        }
    }

    private fun saveToMediaStore(source: File): Uri? {
        return try {
            val resolver = applicationContext.contentResolver
            val collection = MediaStore.Video.Media.EXTERNAL_CONTENT_URI
            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, source.name)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.DATE_ADDED, System.currentTimeMillis() / 1000)
                put(MediaStore.Video.Media.DATE_TAKEN, System.currentTimeMillis())
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/InterpolatedVideos")
                }
            }
            val uri = resolver.insert(collection, values) ?: return null
            resolver.openOutputStream(uri)?.use { out ->
                source.inputStream().use { it.copyTo(out) }
            } ?: return null
            uri
        } catch (_: Exception) {
            null
        }
    }
    
    private fun formatTime(ms: Long): String {
        val minutes = (ms / 1000) / 60
        val seconds = (ms / 1000) % 60
        return String.format("%d:%02d", minutes, seconds)
    }
}
