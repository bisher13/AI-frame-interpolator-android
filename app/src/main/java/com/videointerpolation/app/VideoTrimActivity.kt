package com.videointerpolation.app

import android.content.Intent
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.videointerpolation.app.data.AppSettings
import com.google.android.material.slider.RangeSlider
import java.util.concurrent.TimeUnit

class VideoTrimActivity : AppCompatActivity() {
    
    private lateinit var settings: AppSettings
    private lateinit var videoUri: Uri
    private var videoDurationMs: Long = 0
    
    private lateinit var rangeSlider: RangeSlider
    private lateinit var tvStartTime: TextView
    private lateinit var tvEndTime: TextView
    private lateinit var tvDuration: TextView
    private lateinit var ivStartPreview: ImageView
    private lateinit var ivEndPreview: ImageView
    private lateinit var pbStartPreview: ProgressBar
    private lateinit var pbEndPreview: ProgressBar
    private lateinit var videoPreview: VideoView
    private lateinit var btnPlayPreview: Button
    
    private var previewUpdateRunnable: Runnable? = null
    private val handler = android.os.Handler(android.os.Looper.getMainLooper())
    
    // Frame cache to avoid reloading
    private val frameCache = mutableMapOf<Long, Bitmap>()
    private val maxCacheSize = 10
    private var isPreviewPlaying = false
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_trim)
        
        settings = AppSettings.getInstance(this)
        videoUri = intent.data ?: run {
            Toast.makeText(this, "No video selected", Toast.LENGTH_SHORT).show()
            finish()
            return
        }
        
        try {
            initializeViews()
            loadVideoMetadata()
            setupTrimControls()
            
            findViewById<Button>(R.id.btnCancelTrim)?.setOnClickListener {
                settings.trimEnabled = false
                Toast.makeText(this, "Trim disabled", Toast.LENGTH_SHORT).show()
                finish()
            }
            
            findViewById<Button>(R.id.btnApplyTrim)?.setOnClickListener {
                applyTrimSettings()
                val values = rangeSlider.values
                val duration = (values[1] - values[0]).toLong()
                Toast.makeText(
                    this, 
                    "✓ Trim applied!\nDuration: ${formatTime(duration)}\nThis will be applied during interpolation.",
                    Toast.LENGTH_LONG
                ).show()
                finish()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Error loading trim UI: ${e.message}", Toast.LENGTH_LONG).show()
            android.util.Log.e("VideoTrimActivity", "Error in onCreate", e)
            finish()
        }
    }
    
    private fun initializeViews() {
        rangeSlider = findViewById(R.id.rangeSliderTrim) ?: run {
            Toast.makeText(this, "Layout error: rangeSliderTrim not found", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        tvStartTime = findViewById(R.id.tvStartTime) ?: run {
            Toast.makeText(this, "Layout error: tvStartTime not found", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        tvEndTime = findViewById(R.id.tvEndTime) ?: run {
            Toast.makeText(this, "Layout error: tvEndTime not found", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        tvDuration = findViewById(R.id.tvTrimDuration) ?: run {
            Toast.makeText(this, "Layout error: tvTrimDuration not found", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        ivStartPreview = findViewById(R.id.ivStartPreview) ?: run {
            Toast.makeText(this, "Layout error: ivStartPreview not found", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        ivEndPreview = findViewById(R.id.ivEndPreview) ?: run {
            Toast.makeText(this, "Layout error: ivEndPreview not found", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        pbStartPreview = findViewById(R.id.pbStartPreview) ?: ProgressBar(this)
        pbEndPreview = findViewById(R.id.pbEndPreview) ?: ProgressBar(this)
        videoPreview = findViewById(R.id.videoPreview) ?: run {
            Toast.makeText(this, "Layout error: videoPreview not found", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        btnPlayPreview = findViewById(R.id.btnPlayPreview) ?: run {
            Toast.makeText(this, "Layout error: btnPlayPreview not found", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        
        // Setup video preview
        setupVideoPreview()
    }
    
    private fun setupVideoPreview() {
        videoPreview.setVideoURI(videoUri)
        
        btnPlayPreview.setOnClickListener {
            if (isPreviewPlaying) {
                stopPreview()
            } else {
                playPreview()
            }
        }
        
        videoPreview.setOnPreparedListener { mp ->
            mp.isLooping = false
            mp.setOnCompletionListener {
                stopPreview()
            }
        }
        
        videoPreview.setOnErrorListener { _, what, extra ->
            Toast.makeText(this, "Error playing preview: $what, $extra", Toast.LENGTH_SHORT).show()
            stopPreview()
            true
        }
    }
    
    private fun playPreview() {
        val values = rangeSlider.values
        val startMs = values[0].toInt()
        val endMs = values[1].toInt()
        
        try {
            videoPreview.seekTo(startMs)
            videoPreview.start()
            isPreviewPlaying = true
            btnPlayPreview.text = "⏸ Stop"
            btnPlayPreview.visibility = android.view.View.GONE
            
            // Stop at trim end point
            handler.postDelayed(object : Runnable {
                override fun run() {
                    if (isPreviewPlaying && videoPreview.isPlaying) {
                        if (videoPreview.currentPosition >= endMs) {
                            stopPreview()
                        } else {
                            handler.postDelayed(this, 100)
                        }
                    }
                }
            }, 100)
        } catch (e: Exception) {
            Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
            stopPreview()
        }
    }
    
    private fun stopPreview() {
        videoPreview.pause()
        isPreviewPlaying = false
        btnPlayPreview.text = "▶ Preview"
        btnPlayPreview.visibility = android.view.View.VISIBLE
    }
    
    private fun loadVideoMetadata() {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(this, videoUri)
            val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            videoDurationMs = duration?.toLong() ?: 0
            
            if (videoDurationMs == 0L) {
                Toast.makeText(this, "Could not read video duration", Toast.LENGTH_SHORT).show()
                finish()
                return
            }
            
            findViewById<TextView>(R.id.tvVideoDuration)?.text = 
                "Video Duration: ${formatTime(videoDurationMs)}"
            
        } catch (e: Exception) {
            Toast.makeText(this, "Error loading video: ${e.message}", Toast.LENGTH_LONG).show()
            android.util.Log.e("VideoTrimActivity", "Error loading metadata", e)
            finish()
        } finally {
            retriever.release()
        }
    }
    
    private fun setupTrimControls() {
        // Configure range slider
        rangeSlider.valueFrom = 0f
        rangeSlider.valueTo = videoDurationMs.toFloat()
        // Set stepSize to 0 for continuous (smooth) selection
        rangeSlider.stepSize = 0f
        
        // Load existing trim settings or set to full video
        val startMs = if (settings.trimEnabled) settings.trimStart else 0L
        val endMs = if (settings.trimEnabled) settings.trimEnd else videoDurationMs
        
        rangeSlider.values = listOf(startMs.toFloat(), endMs.toFloat())
        
        // Update time displays
        updateTimeDisplays(startMs, endMs)
        
        // Load initial frame previews
        updateFramePreviews(startMs, endMs)
        
        // Listen for slider changes
        rangeSlider.addOnChangeListener { _, _, _ ->
            val values = rangeSlider.values
            updateTimeDisplays(values[0].toLong(), values[1].toLong())
            // Debounce preview updates - only update after user stops dragging
            updateFramePreviewsDebounced(values[0].toLong(), values[1].toLong())
        }
        
        // Also update on slider touch release for immediate feedback
        rangeSlider.addOnSliderTouchListener(object : RangeSlider.OnSliderTouchListener {
            override fun onStartTrackingTouch(slider: RangeSlider) {
                // Do nothing
            }
            
            override fun onStopTrackingTouch(slider: RangeSlider) {
                // Force immediate update when user releases
                val values = slider.values
                handler.removeCallbacks(previewUpdateRunnable ?: return)
                updateFramePreviews(values[0].toLong(), values[1].toLong())
            }
        })
        
        // Quick trim buttons
        findViewById<Button>(R.id.btnTrimFirst10s)?.setOnClickListener {
            setTrimRange(0L, minOf(10000L, videoDurationMs))
        }
        
        findViewById<Button>(R.id.btnTrimFirst30s)?.setOnClickListener {
            setTrimRange(0L, minOf(30000L, videoDurationMs))
        }
        
        findViewById<Button>(R.id.btnTrimFirst1m)?.setOnClickListener {
            setTrimRange(0L, minOf(60000L, videoDurationMs))
        }
        
        findViewById<Button>(R.id.btnTrimLast10s)?.setOnClickListener {
            val start = maxOf(0L, videoDurationMs - 10000L)
            setTrimRange(start, videoDurationMs)
        }
        
        findViewById<Button>(R.id.btnTrimLast30s)?.setOnClickListener {
            val start = maxOf(0L, videoDurationMs - 30000L)
            setTrimRange(start, videoDurationMs)
        }
        
        findViewById<Button>(R.id.btnTrimLast1m)?.setOnClickListener {
            val start = maxOf(0L, videoDurationMs - 60000L)
            setTrimRange(start, videoDurationMs)
        }
        
        findViewById<Button>(R.id.btnResetTrim)?.setOnClickListener {
            setTrimRange(0L, videoDurationMs)
        }
        
        // Frame stepping buttons (step by 1 frame at 30fps = ~33ms)
        val frameStep = 33L
        
        findViewById<Button>(R.id.btnStepStartBack)?.setOnClickListener {
            val values = rangeSlider.values
            val newStart = maxOf(0L, values[0].toLong() - frameStep)
            setTrimRange(newStart, values[1].toLong())
        }
        
        findViewById<Button>(R.id.btnStepStartForward)?.setOnClickListener {
            val values = rangeSlider.values
            val newStart = minOf(values[1].toLong() - frameStep, values[0].toLong() + frameStep)
            setTrimRange(newStart, values[1].toLong())
        }
        
        findViewById<Button>(R.id.btnStepEndBack)?.setOnClickListener {
            val values = rangeSlider.values
            val newEnd = maxOf(values[0].toLong() + frameStep, values[1].toLong() - frameStep)
            setTrimRange(values[0].toLong(), newEnd)
        }
        
        findViewById<Button>(R.id.btnStepEndForward)?.setOnClickListener {
            val values = rangeSlider.values
            val newEnd = minOf(videoDurationMs, values[1].toLong() + frameStep)
            setTrimRange(values[0].toLong(), newEnd)
        }
    }
    
    private fun setTrimRange(startMs: Long, endMs: Long) {
        rangeSlider.values = listOf(startMs.toFloat(), endMs.toFloat())
        updateTimeDisplays(startMs, endMs)
        updateFramePreviews(startMs, endMs)
    }
    
    private fun updateTimeDisplays(startMs: Long, endMs: Long) {
        tvStartTime.text = "Start: ${formatTime(startMs)}"
        tvEndTime.text = "End: ${formatTime(endMs)}"
        
        val durationMs = endMs - startMs
        tvDuration.text = "Trim Duration: ${formatTime(durationMs)}"
        
        // Estimate frame count
        val fps = settings.targetFps
        val frameCount = (durationMs / 1000.0 * fps).toInt()
        findViewById<TextView>(R.id.tvEstimatedFrames)?.text = 
            "Estimated frames: ~$frameCount frames at ${fps}fps"
    }
    
    private fun updateFramePreviewsDebounced(startTimeMs: Long, endTimeMs: Long) {
        // Cancel any pending updates
        previewUpdateRunnable?.let { handler.removeCallbacks(it) }
        
        // Create new runnable
        previewUpdateRunnable = Runnable {
            updateFramePreviews(startTimeMs, endTimeMs)
        }
        
        // Schedule update after 250ms delay
        handler.postDelayed(previewUpdateRunnable!!, 250)
    }
    
    private fun updateFramePreviews(startMs: Long, endMs: Long) {
        // Show loading indicators
        runOnUiThread {
            pbStartPreview.visibility = android.view.View.VISIBLE
            pbEndPreview.visibility = android.view.View.VISIBLE
        }
        
        // Update previews in background to avoid blocking UI
        Thread {
            try {
                // Check cache first
                val startFrame = getFrameFromCacheOrLoad(startMs)
                val endFrame = getFrameFromCacheOrLoad(endMs)
                
                // Update UI on main thread
                runOnUiThread {
                    pbStartPreview.visibility = android.view.View.GONE
                    pbEndPreview.visibility = android.view.View.GONE
                    
                    startFrame?.let { 
                        ivStartPreview.setImageBitmap(it)
                        ivStartPreview.visibility = android.view.View.VISIBLE
                    }
                    endFrame?.let { 
                        ivEndPreview.setImageBitmap(it)
                        ivEndPreview.visibility = android.view.View.VISIBLE
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("VideoTrimActivity", "Error loading frame previews", e)
                runOnUiThread {
                    pbStartPreview.visibility = android.view.View.GONE
                    pbEndPreview.visibility = android.view.View.GONE
                }
            }
        }.start()
    }
    
    private fun getFrameFromCacheOrLoad(timeMs: Long): Bitmap? {
        // Round to nearest second for better cache hits
        val cacheKey = (timeMs / 1000) * 1000
        
        // Check cache
        frameCache[cacheKey]?.let { return it }
        
        // Load from video
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(this, videoUri)
            
            // Use OPTION_CLOSEST for faster seeking (no keyframe wait)
            val frame = retriever.getFrameAtTime(
                timeMs * 1000, // Convert to microseconds
                MediaMetadataRetriever.OPTION_CLOSEST
            )
            
            // Scale down for memory efficiency (max 400px width)
            val scaledFrame = frame?.let { scaleBitmap(it, 400) }
            
            // Add to cache
            if (scaledFrame != null) {
                // Limit cache size
                if (frameCache.size >= maxCacheSize) {
                    frameCache.remove(frameCache.keys.first())
                }
                frameCache[cacheKey] = scaledFrame
            }
            
            return scaledFrame
        } finally {
            retriever.release()
        }
    }
    
    private fun scaleBitmap(bitmap: Bitmap, maxWidth: Int): Bitmap {
        if (bitmap.width <= maxWidth) return bitmap
        
        val ratio = maxWidth.toFloat() / bitmap.width
        val newHeight = (bitmap.height * ratio).toInt()
        
        return Bitmap.createScaledBitmap(bitmap, maxWidth, newHeight, true)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Clear cache to free memory
        frameCache.clear()
        handler.removeCallbacks(previewUpdateRunnable ?: return)
    }
    
    private fun applyTrimSettings() {
        val values = rangeSlider.values
        settings.trimEnabled = true
        settings.trimStart = values[0].toLong()
        settings.trimEnd = values[1].toLong()
    }
    
    private fun formatTime(ms: Long): String {
        val hours = TimeUnit.MILLISECONDS.toHours(ms)
        val minutes = TimeUnit.MILLISECONDS.toMinutes(ms) % 60
        val seconds = TimeUnit.MILLISECONDS.toSeconds(ms) % 60
        val millis = ms % 1000 / 10 // Show centiseconds
        
        return if (hours > 0) {
            String.format("%02d:%02d:%02d.%02d", hours, minutes, seconds, millis)
        } else {
            String.format("%02d:%02d.%02d", minutes, seconds, millis)
        }
    }
}
