package com.videointerpolation.app

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import android.content.ContentValues
import android.os.Environment
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.videointerpolation.app.databinding.ActivityMainBinding
import androidx.appcompat.app.AppCompatDelegate
import com.videointerpolation.app.ml.FrameInterpolator
import com.videointerpolation.app.utils.VideoEncoder
import com.videointerpolation.app.utils.VideoProcessor
import com.videointerpolation.app.utils.CacheManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import android.graphics.BitmapFactory
import android.graphics.Bitmap
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    private lateinit var videoProcessor: VideoProcessor
    private lateinit var frameInterpolator: FrameInterpolator
    private lateinit var videoEncoder: VideoEncoder
    private lateinit var cacheManager: CacheManager
    
    private var selectedVideoUri: Uri? = null
    
    companion object {
        private const val REQUEST_VIDEO_PICK = 1001
        private const val REQUEST_PERMISSIONS = 1002
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        )
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Force dark mode for better contrast with black background
        AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        videoProcessor = VideoProcessor(this)
    // Prefer ONNX Runtime when a model is available; falls back to TFLite, then Optical Flow
    frameInterpolator = FrameInterpolator(this, com.videointerpolation.app.ml.InterpMode.ONNX)
        videoEncoder = VideoEncoder()
        cacheManager = CacheManager.getInstance(this)
        
        setupUI()
        checkPermissions()
        
        // Clean old temporary files on startup
        lifecycleScope.launch(Dispatchers.IO) {
            cacheManager.clearOldTempFiles()
        }
        
        // Initialize ML models: try ONNX first (rife_fp16.onnx), then TFLite (rife_fp16.tflite)
        frameInterpolator.initialize(
            modelPath = "rife_fp16.tflite",
            onnxModelPath = "rife_fp16.onnx"
        )

        // Show backend status
        val mode = frameInterpolator.getEffectiveMode()
        val status = when (mode) {
            com.videointerpolation.app.ml.InterpMode.ONNX -> "ONNX model loaded"
            com.videointerpolation.app.ml.InterpMode.TFLITE -> "TFLite model loaded"
            com.videointerpolation.app.ml.InterpMode.OPTICAL_FLOW -> "Optical Flow fallback"
            else -> "Linear blend fallback"
        }
    binding.tvStatus.text = "Ready • $status"
    binding.tvBackend.text = "Backend: ${mode.name} • ${frameInterpolator.getAccelerationInfo()}"
    }
    
    override fun onResume() {
        super.onResume()
        updateTrimStatus()
    }
    
    private fun updateTrimStatus() {
        val settings = com.videointerpolation.app.data.AppSettings.getInstance(this)
        if (settings.trimEnabled) {
            binding.tvTrimStatus.visibility = android.view.View.VISIBLE
            binding.tvTrimStatus.text = "✂ Trim Active: ${formatTime(settings.trimStart)} - ${formatTime(settings.trimEnd)}"
        } else {
            binding.tvTrimStatus.visibility = android.view.View.GONE
        }
    }
    
    private fun formatTime(ms: Long): String {
        val minutes = (ms / 1000) / 60
        val seconds = (ms / 1000) % 60
        return String.format("%d:%02d", minutes, seconds)
    }
    
    private fun setupUI() {
        // Settings button
        binding.btnSettings.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        
        // Trim video button
        binding.btnTrimVideo.setOnClickListener {
            selectedVideoUri?.let { uri ->
                val intent = Intent(this, VideoTrimActivity::class.java)
                intent.data = uri
                startActivity(intent)
            } ?: run {
                Toast.makeText(this, "Please select a video first", Toast.LENGTH_SHORT).show()
            }
        }
        
        // Extract frames button
        binding.btnExtractFrames.setOnClickListener {
            selectedVideoUri?.let { uri ->
                extractFrames(uri)
            } ?: run {
                Toast.makeText(this, "Please select a video first", Toast.LENGTH_SHORT).show()
            }
        }
        
        binding.btnSelectVideo.setOnClickListener {
            selectVideo()
        }
        
        binding.btnProcess.setOnClickListener {
            selectedVideoUri?.let { uri ->
                // Run foreground WorkManager job in background if toggle enabled
                if (binding.switchBackground.isChecked) {
                    enqueueBackgroundInterpolation(uri)
                } else {
                    // Clear previous progress state to avoid appearing frozen
                    binding.progressLinear.visibility = android.view.View.VISIBLE
                    binding.progressLinear.isIndeterminate = false
                    binding.progressLinear.setProgressCompat(0, false)
                    binding.tvStatus.text = "Processing..."
                    processVideo(uri)
                }
            } ?: run {
                Toast.makeText(this, "Please select a video first", Toast.LENGTH_SHORT).show()
            }
        }
        
        // Set interpolation multiplier
        binding.seekBarMultiplier.max = 4
        binding.seekBarMultiplier.progress = 1
        updateMultiplierText(2)

    // Background switch default off
    binding.switchBackground.isChecked = false

        // Bitrate SeekBar: map positions to Mbps (3–40 Mbps for flexibility)
        binding.seekBarBitrate.max = 37 // 3..40
        binding.seekBarBitrate.progress = 5 // default 8 Mbps
        updateBitrateText(mbpsFromProgress(binding.seekBarBitrate.progress))
        binding.seekBarBitrate.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                updateBitrateText(mbpsFromProgress(progress))
            }
            override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {}
        })
        
        binding.seekBarMultiplier.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                val multiplier = (progress + 1) * 2
                updateMultiplierText(multiplier)
            }
            override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {}
        })

        // Model size alignment switch (opt-in performance / stability for some models)
        binding.switchAlign.setOnCheckedChangeListener { _, isChecked ->
            frameInterpolator.setAlignToEight(isChecked)
        }

        // NNAPI accelerator handling
        binding.btnApplyAccelerator.setOnClickListener {
            val name = binding.etAccelerator.text?.toString()?.trim().orEmpty()
            frameInterpolator.setNnapiAcceleratorName(name.ifEmpty { null })
            // Re-initialize TFLite backend with new delegate preference (ONNX kept as fallback if NNAPI fails)
            frameInterpolator.close()
            // Force prefer TFLite path when switchUseNnapi is enabled
            frameInterpolator.setUseTfliteNnapi(binding.switchUseNnapi.isChecked)
            frameInterpolator.initialize(
                modelPath = "rife_fp16.tflite",
                onnxModelPath = "rife_fp16.onnx"
            )
            updateAccelStatus()
        }
        binding.switchUseNnapi.setOnCheckedChangeListener { _, isChecked ->
            frameInterpolator.setUseTfliteNnapi(isChecked)
            // Re-init to apply delegate change
            frameInterpolator.close()
            frameInterpolator.initialize(
                modelPath = "rife_fp16.tflite",
                onnxModelPath = "rife_fp16.onnx"
            )
            updateAccelStatus()
        }
        updateAccelStatus()
    }
    
    private fun updateMultiplierText(multiplier: Int) {
        binding.tvMultiplier.text = "${multiplier}x FPS"
    }

    private fun mbpsFromProgress(p: Int): Int = 3 + p // 3..40 Mbps
    private fun updateBitrateText(mbps: Int) {
        binding.tvBitrate.text = "$mbps Mbps"
    }

    private fun updateAccelStatus() {
        val mode = frameInterpolator.getEffectiveMode()
        val accel = frameInterpolator.getAccelerationInfo()
        binding.tvAccelStatus.text = "Mode: ${mode.name} • $accel"
        binding.tvBackend.text = "Backend: ${mode.name} • $accel"
    }
    
    private fun checkPermissions() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            // Android 13+: request media + notifications
            val toRequest = mutableListOf<String>()
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_VIDEO) != PackageManager.PERMISSION_GRANTED) {
                toRequest.add(Manifest.permission.READ_MEDIA_VIDEO)
            }
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                toRequest.add(Manifest.permission.POST_NOTIFICATIONS)
            }
            if (toRequest.isNotEmpty()) {
                ActivityCompat.requestPermissions(this, toRequest.toTypedArray(), REQUEST_PERMISSIONS)
            }
        } else {
            val permissionsToRequest = REQUIRED_PERMISSIONS.filter {
                ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
            }
            
            if (permissionsToRequest.isNotEmpty()) {
                ActivityCompat.requestPermissions(
                    this,
                    permissionsToRequest.toTypedArray(),
                    REQUEST_PERMISSIONS
                )
            }
        }
    }
    
    private fun selectVideo() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(intent, REQUEST_VIDEO_PICK)
    }
    
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        if (requestCode == REQUEST_VIDEO_PICK && resultCode == Activity.RESULT_OK) {
            data?.data?.let { uri ->
                selectedVideoUri = uri
                binding.tvVideoPath.text = "Video selected: ${uri.lastPathSegment}"
                binding.btnProcess.isEnabled = true
                
                // Display video metadata
                val metadata = videoProcessor.getVideoMetadata(uri)
                binding.tvVideoInfo.text = """
                    Resolution: ${metadata.width}x${metadata.height}
                    Duration: ${metadata.duration / 1000}s
                    Frame Rate: ${metadata.frameRate} fps
                """.trimIndent()
            }
        }
    }
    
    private fun processVideo(videoUri: Uri) {
        binding.btnProcess.isEnabled = false
        binding.tvStatus.text = "Processing..."
        binding.progressLinear.visibility = android.view.View.VISIBLE
        binding.progressLinear.isIndeterminate = false
        binding.progressLinear.setProgressCompat(0, false)

        val multiplier = (binding.seekBarMultiplier.progress + 1) * 2
        
        lifecycleScope.launch {
            try {
                // Step 1: Extract frames at original resolution
                val srcMeta = videoProcessor.getVideoMetadata(videoUri)
                updateStatus("Extracting frames at ${srcMeta.width}x${srcMeta.height}...")
                val framesDir = File(cacheDir, "frames")
                framesDir.deleteRecursively()
                framesDir.mkdirs()
                
                val originalFrames = videoProcessor.extractFrames(videoUri, framesDir, { current, total ->
                    val stageStart = 0
                    val stageWeight = 30
                    val frac = if (total > 0) current.toFloat() / total else 0f
                    setProgress(stageStart + (frac * stageWeight))
                    // Update extraction progress text
                    runOnUiThread {
                        binding.tvExtractProgress.text = "Extraction: ${current}/${total}"
                    }
                }, maxWidth = srcMeta.width.coerceAtLeast(1))
                updateStatus("Extracted ${originalFrames.size} frames")
                
                // Step 2: Interpolate frames
                updateStatus("Interpolating frames (${multiplier}x)...")
                // Show backend used during interpolation
                runOnUiThread {
                    val eff = frameInterpolator.getEffectiveMode().name
                    binding.tvBackend.text = "Backend: ${eff} • ${frameInterpolator.getAccelerationInfo()}"
                }
                val interpolatedDir = File(cacheDir, "interpolated")
                interpolatedDir.deleteRecursively()
                interpolatedDir.mkdirs()
                
                val allFrames = interpolateAllFrames(originalFrames, interpolatedDir, multiplier - 1) { current, total ->
                    val stageStart = 30
                    val stageWeight = 40
                    val frac = if (total > 0) current.toFloat() / total else 0f
                    setProgress(stageStart + (frac * stageWeight))
                }
                updateStatus("Created ${allFrames.size} interpolated frames")
                
                // Step 3: Encode to video
                updateStatus("Encoding video...")
                val firstBounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeFile(allFrames.firstOrNull()?.absolutePath, firstBounds)
                var encW = firstBounds.outWidth.takeIf { it > 0 } ?: videoProcessor.getVideoMetadata(videoUri).width
                var encH = firstBounds.outHeight.takeIf { it > 0 } ?: videoProcessor.getVideoMetadata(videoUri).height
                // Make sure dimensions are even for YUV420 encoder
                if (encW % 2 != 0) encW -= 1
                if (encH % 2 != 0) encH -= 1
                val outputFile = File(getExternalFilesDir(null), "interpolated_${System.currentTimeMillis()}.mp4")
                
                // Recommend bitrate based on resolution and fps
                val recommendedBitrate = (encW.toLong() * encH.toLong() * 60L * 7L / 100L).toInt().coerceAtLeast(3_000_000)

                val userBitrateMbps = mbpsFromProgress(binding.seekBarBitrate.progress)
                val userBitrate = (userBitrateMbps * 1_000_000).coerceAtLeast(1_000_000)
                val success = videoEncoder.encodeFramesToVideo(
                    allFrames,
                    outputFile,
                    encW,
                    encH,
                    bitRate = userBitrate,
                    onProgress = { current, total ->
                        val stageStart = 70
                        val stageWeight = 30
                        val frac = if (total > 0) current.toFloat() / total else 0f
                        setProgress(stageStart + (frac * stageWeight))
                    }
                )
                
                if (success) {
                    // Persist to MediaStore so it appears in Gallery / Photos
                    val galleryUri = saveVideoToGallery(outputFile)
                    if (galleryUri != null) {
                        updateStatus("✓ Video saved: ${outputFile.name}\nGallery URI: $galleryUri")
                        showToast("Video saved to gallery")
                        // Load a small preview thumbnail
                        withContext(Dispatchers.Main) {
                            binding.ivPreview.visibility = android.view.View.VISIBLE
                            try {
                                val thumb = android.provider.MediaStore.Video.Thumbnails.getThumbnail(
                                    contentResolver,
                                    android.content.ContentUris.parseId(galleryUri),
                                    android.provider.MediaStore.Video.Thumbnails.MINI_KIND,
                                    null
                                )
                                if (thumb != null) binding.ivPreview.setImageBitmap(thumb)
                            } catch (_: Exception) {}
                        }
                    } else {
                        updateStatus("✓ Video saved (app folder): ${outputFile.absolutePath}")
                        showToast("Saved (not indexed) — copying failed")
                    }
                } else {
                    updateStatus("✗ Encoding failed")
                    showToast("Failed to encode video")
                }
                
            } catch (e: Exception) {
                updateStatus("✗ Error: ${e.message}")
                showToast("Error: ${e.message}")
                e.printStackTrace()
            } finally {
                withContext(Dispatchers.Main) {
                    binding.progressLinear.visibility = android.view.View.GONE
                    binding.btnProcess.isEnabled = true
                }
            }
        }
    }

    private fun saveVideoToGallery(source: File): Uri? {
        return try {
            val resolver = contentResolver
            val collection = MediaStore.Video.Media.EXTERNAL_CONTENT_URI
            val name = source.name
            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, name)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.DATE_ADDED, System.currentTimeMillis() / 1000)
                put(MediaStore.Video.Media.DATE_TAKEN, System.currentTimeMillis())
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                    put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/InterpolatedVideos")
                }
            }
            val uri = resolver.insert(collection, values) ?: return null
            resolver.openOutputStream(uri)?.use { out ->
                source.inputStream().use { it.copyTo(out) }
            } ?: return null
            uri
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
    
    private suspend fun interpolateAllFrames(
        originalFrames: List<File>,
        outputDir: File,
        stepsPerFrame: Int,
        onProgress: ((current: Int, total: Int) -> Unit)? = null
    ): List<File> = withContext(Dispatchers.Default) {
        val allFrames = mutableListOf<File>()
        var frameIndex = 0
        val totalFramesToProduce = if (originalFrames.size > 1) {
            // originals + interpolations between pairs
            originalFrames.size + (originalFrames.size - 1) * stepsPerFrame
        } else originalFrames.size
        var produced = 0
        
        for (i in 0 until originalFrames.size - 1) {
            // Add original frame
            val originalFile = File(outputDir, "frame_${String.format("%05d", frameIndex++)}.jpg")
            originalFrames[i].copyTo(originalFile, overwrite = true)
            allFrames.add(originalFile)
            produced++
            onProgress?.invoke(produced, totalFramesToProduce)
            
            // Interpolate between this frame and next
            val bitmap1 = BitmapFactory.decodeFile(originalFrames[i].absolutePath)
            val bitmap2 = BitmapFactory.decodeFile(originalFrames[i + 1].absolutePath)
            
            if (bitmap1 != null && bitmap2 != null) {
                val interpolated = frameInterpolator.interpolateFrames(bitmap1, bitmap2, stepsPerFrame)
                
                interpolated.forEach { interpolatedBitmap ->
                    val interpolatedFile = File(outputDir, "frame_${String.format("%05d", frameIndex++)}.jpg")
                    FileOutputStream(interpolatedFile).use { out ->
                        interpolatedBitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out)
                    }
                    allFrames.add(interpolatedFile)
                    // Live preview: show a small thumbnail of the latest interpolated frame
                    try {
                        val thumbW = 160
                        val ratio = interpolatedBitmap.height.toFloat() / interpolatedBitmap.width.coerceAtLeast(1)
                        val thumbH = (thumbW * ratio).toInt().coerceAtLeast(1)
                        val thumb = Bitmap.createScaledBitmap(interpolatedBitmap, thumbW, thumbH, true)
                        withContext(Dispatchers.Main) {
                            binding.ivPreview.visibility = android.view.View.VISIBLE
                            binding.ivPreview.setImageBitmap(thumb)
                            binding.tvPreviewLabel.visibility = android.view.View.VISIBLE
                            binding.tvPreviewLabel.text = "Interpolated (${produced}/${totalFramesToProduce})"
                        }
                    } catch (_: Exception) {}
                    interpolatedBitmap.recycle()
                    produced++
                    onProgress?.invoke(produced, totalFramesToProduce)
                }
                
                bitmap1.recycle()
                bitmap2.recycle()
            }
        }
        
        // Add last frame
        val lastFile = File(outputDir, "frame_${String.format("%05d", frameIndex)}.jpg")
        originalFrames.last().copyTo(lastFile, overwrite = true)
        allFrames.add(lastFile)
        produced++
        onProgress?.invoke(produced, totalFramesToProduce)
        
        allFrames
    }
    
    private suspend fun updateStatus(status: String) {
        withContext(Dispatchers.Main) {
            binding.tvStatus.text = status
        }
    }

    private fun setProgress(percent: Float) {
        val clamped = percent.coerceIn(0f, 100f)
        runOnUiThread {
            binding.progressLinear.setProgressCompat(clamped.toInt(), true)
        }
    }
    
    private suspend fun showToast(message: String) {
        withContext(Dispatchers.Main) {
            Toast.makeText(this@MainActivity, message, Toast.LENGTH_LONG).show()
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        frameInterpolator.close()
    }

    private fun enqueueBackgroundInterpolation(uri: Uri) {
        val multiplier = (binding.seekBarMultiplier.progress + 1) * 2
        val data = androidx.work.Data.Builder()
            .putString(com.videointerpolation.app.work.VideoInterpolationWorker.KEY_INPUT_URI, uri.toString())
            .putInt(com.videointerpolation.app.work.VideoInterpolationWorker.KEY_MULTIPLIER, multiplier)
            .putInt(com.videointerpolation.app.work.VideoInterpolationWorker.KEY_BITRATE_MBPS, mbpsFromProgress(binding.seekBarBitrate.progress))
            .putBoolean(com.videointerpolation.app.work.VideoInterpolationWorker.KEY_ALIGN_EIGHT, binding.switchAlign.isChecked)
            .build()
        val request = androidx.work.OneTimeWorkRequestBuilder<com.videointerpolation.app.work.VideoInterpolationWorker>()
            .setInputData(data)
            .build()
        val wm = androidx.work.WorkManager.getInstance(this)
        wm.enqueue(request)
        wm.getWorkInfoByIdLiveData(request.id).observe(this) { info ->
            if (info != null) {
                // Live background progress reflection
                val pct = info.progress.getInt("progress", -1)
                val msg = info.progress.getString("message")
                val preview = info.progress.getString("preview")
                if (pct >= 0) {
                    setProgress(pct.toFloat())
                    if (!msg.isNullOrBlank()) binding.tvStatus.text = msg
                }
                if (!preview.isNullOrBlank()) {
                    try {
                        val opts = android.graphics.BitmapFactory.Options().apply {
                            inJustDecodeBounds = true
                        }
                        android.graphics.BitmapFactory.decodeFile(preview, opts)
                        val targetW = 160
                        var sample = 1
                        while ((opts.outWidth / sample) > targetW) sample *= 2
                        val opts2 = android.graphics.BitmapFactory.Options().apply { inSampleSize = sample }
                        val bmp = android.graphics.BitmapFactory.decodeFile(preview, opts2)
                        if (bmp != null) {
                            binding.ivPreview.visibility = android.view.View.VISIBLE
                            binding.ivPreview.setImageBitmap(bmp)
                            binding.tvPreviewLabel.visibility = android.view.View.VISIBLE
                            binding.tvPreviewLabel.text = "Preview"
                        }
                    } catch (_: Exception) {}
                }
            }
            if (info != null && info.state.isFinished) {
                val gallery = info.outputData.getString("gallery_uri")
                val filePath = info.outputData.getString("file_path")
                if (gallery != null) {
                    binding.tvStatus.text = "✓ Background saved to gallery: $gallery"
                    binding.ivPreview.visibility = android.view.View.VISIBLE
                    try {
                        val uriBg = Uri.parse(gallery)
                        val thumb = android.provider.MediaStore.Video.Thumbnails.getThumbnail(
                            contentResolver,
                            android.content.ContentUris.parseId(uriBg),
                            android.provider.MediaStore.Video.Thumbnails.MINI_KIND,
                            null
                        )
                        if (thumb != null) binding.ivPreview.setImageBitmap(thumb)
                    } catch (_: Exception) {}
                } else if (filePath != null) {
                    binding.tvStatus.text = "✓ Background saved: $filePath"
                }
            }
        }
        Toast.makeText(this, "Running in background… check notification", Toast.LENGTH_LONG).show()
        binding.tvStatus.text = "Background job queued"
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        if (requestCode == REQUEST_PERMISSIONS) {
            val allGranted = grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }
            if (!allGranted) {
                Toast.makeText(this, "Some permissions denied. Background progress notification or media access may be limited.", Toast.LENGTH_LONG).show()
            }
        }
    }
    
    private fun extractFrames(videoUri: Uri) {
        lifecycleScope.launch {
            try {
                binding.progressLinear.visibility = android.view.View.VISIBLE
                binding.tvStatus.text = "Extracting frames..."
                
                val outputDir = File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "extracted_frames")
                val options = FrameExtractor.ExtractOptions(
                    outputDir = outputDir,
                    format = FrameExtractor.ExportFormat.PNG,
                    intervalMs = 1000 // 1 frame per second
                )
                
                val extractor = FrameExtractor(this@MainActivity)
                val result = extractor.extractFrames(videoUri, options) { current, total ->
                    runOnUiThread {
                        val progress = (current * 100f / total)
                        setProgress(progress)
                        binding.tvStatus.text = "Extracting frames... $current/$total"
                    }
                }
                
                if (result.isSuccess) {
                    val files = result.getOrNull()!!
                    binding.tvStatus.text = "✓ Extracted ${files.size} frames to ${outputDir.absolutePath}"
                    Toast.makeText(this@MainActivity, "Extracted ${files.size} frames", Toast.LENGTH_LONG).show()
                } else {
                    binding.tvStatus.text = "✗ Frame extraction failed: ${result.exceptionOrNull()?.message}"
                }
                
            } catch (e: Exception) {
                binding.tvStatus.text = "✗ Error: ${e.message}"
            } finally {
                binding.progressLinear.visibility = android.view.View.GONE
            }
        }
    }
}
