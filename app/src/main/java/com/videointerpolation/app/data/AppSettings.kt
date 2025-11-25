package com.videointerpolation.app.data

import android.content.Context
import android.content.SharedPreferences

/**
 * Settings manager for video interpolation app.
 * Stores user preferences for quality, speed, export options, etc.
 */
class AppSettings private constructor(context: Context) {
    
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    
    // Quality Presets
    enum class QualityPreset {
        FAST,       // 480p, fewer frames, GPU priority
        BALANCED,   // 720p, standard frame count
        QUALITY     // Original res, more frames, quality priority
    }
    
    // Speed/Slowdown Factor
    enum class SpeedFactor(val multiplier: Int, val displayName: String) {
        X2(2, "2x Slow Motion"),
        X4(4, "4x Slow Motion"),
        X8(8, "8x Slow Motion"),
        X16(16, "16x Slow Motion"),
        CUSTOM(0, "Custom")
    }
    
    // Export Codec
    enum class ExportCodec(val mimeType: String, val displayName: String) {
        H264("video/avc", "H.264 (Widely Compatible)"),
        H265("video/hevc", "H.265/HEVC (Smaller Size)"),
        VP9("video/x-vnd.on2.vp9", "VP9 (Web Optimized)")
    }
    
    // Interpolation Mode
    var interpolationMode: String
        get() = prefs.getString(KEY_INTERP_MODE, "ONNX") ?: "ONNX"
        set(value) = prefs.edit().putString(KEY_INTERP_MODE, value).apply()
    
    // Quality Preset
    var qualityPreset: QualityPreset
        get() = QualityPreset.valueOf(prefs.getString(KEY_QUALITY_PRESET, "BALANCED") ?: "BALANCED")
        set(value) = prefs.edit().putString(KEY_QUALITY_PRESET, value.name).apply()
    
    // Speed Factor
    var speedFactor: SpeedFactor
        get() = SpeedFactor.valueOf(prefs.getString(KEY_SPEED_FACTOR, "X2") ?: "X2")
        set(value) = prefs.edit().putString(KEY_SPEED_FACTOR, value.name).apply()
    
    var customSpeedMultiplier: Int
        get() = prefs.getInt(KEY_CUSTOM_SPEED, 2)
        set(value) = prefs.edit().putInt(KEY_CUSTOM_SPEED, value.coerceIn(2, 60)).apply()
    
    // Target FPS (for output video)
    var targetFps: Int
        get() = prefs.getInt(KEY_TARGET_FPS, 60)
        set(value) = prefs.edit().putInt(KEY_TARGET_FPS, value.coerceIn(30, 240)).apply()
    
    // Export Codec
    var exportCodec: ExportCodec
        get() = ExportCodec.valueOf(prefs.getString(KEY_EXPORT_CODEC, "H264") ?: "H264")
        set(value) = prefs.edit().putString(KEY_EXPORT_CODEC, value.name).apply()
    
    // Export Bitrate (Mbps)
    var exportBitrateMbps: Float
        get() = prefs.getFloat(KEY_EXPORT_BITRATE, 10f)
        set(value) = prefs.edit().putFloat(KEY_EXPORT_BITRATE, value.coerceIn(1f, 50f)).apply()
    
    // Processing Resolution (max dimension)
    var maxProcessingResolution: Int
        get() = prefs.getInt(KEY_MAX_RESOLUTION, when(qualityPreset) {
            QualityPreset.FAST -> 854  // 480p
            QualityPreset.BALANCED -> 1280  // 720p
            QualityPreset.QUALITY -> 0  // Original
        })
        set(value) = prefs.edit().putInt(KEY_MAX_RESOLUTION, value).apply()
    
    // NNAPI Device
    var nnapiDevice: String?
        get() = prefs.getString(KEY_NNAPI_DEVICE, null)
        set(value) = prefs.edit().putString(KEY_NNAPI_DEVICE, value).apply()
    
    var useNnapi: Boolean
        get() = prefs.getBoolean(KEY_USE_NNAPI, true)
        set(value) = prefs.edit().putBoolean(KEY_USE_NNAPI, value).apply()
    
    // Frame Padding (align to 8)
    var enableFramePadding: Boolean
        get() = prefs.getBoolean(KEY_FRAME_PADDING, true)
        set(value) = prefs.edit().putBoolean(KEY_FRAME_PADDING, value).apply()
    
    // Batch Processing
    var batchProcessingEnabled: Boolean
        get() = prefs.getBoolean(KEY_BATCH_ENABLED, false)
        set(value) = prefs.edit().putBoolean(KEY_BATCH_ENABLED, value).apply()
    
    // Video Trimming
    var trimStart: Long
        get() = prefs.getLong(KEY_TRIM_START, 0)
        set(value) = prefs.edit().putLong(KEY_TRIM_START, value).apply()
    
    var trimEnd: Long
        get() = prefs.getLong(KEY_TRIM_END, 0)
        set(value) = prefs.edit().putLong(KEY_TRIM_END, value).apply()
    
    var trimEnabled: Boolean
        get() = prefs.getBoolean(KEY_TRIM_ENABLED, false)
        set(value) = prefs.edit().putBoolean(KEY_TRIM_ENABLED, value).apply()
    
    // Scene Detection
    var sceneDetectionEnabled: Boolean
        get() = prefs.getBoolean(KEY_SCENE_DETECTION, false)
        set(value) = prefs.edit().putBoolean(KEY_SCENE_DETECTION, value).apply()
    
    var sceneThreshold: Float
        get() = prefs.getFloat(KEY_SCENE_THRESHOLD, 0.3f)
        set(value) = prefs.edit().putFloat(KEY_SCENE_THRESHOLD, value.coerceIn(0.1f, 0.9f)).apply()
    
    // Export GIF
    var exportAsGif: Boolean
        get() = prefs.getBoolean(KEY_EXPORT_GIF, false)
        set(value) = prefs.edit().putBoolean(KEY_EXPORT_GIF, value).apply()
    
    var gifMaxDimension: Int
        get() = prefs.getInt(KEY_GIF_MAX_DIM, 480)
        set(value) = prefs.edit().putInt(KEY_GIF_MAX_DIM, value).apply()
    
    // Timelapse Mode
    var timelapseMode: Boolean
        get() = prefs.getBoolean(KEY_TIMELAPSE_MODE, false)
        set(value) = prefs.edit().putBoolean(KEY_TIMELAPSE_MODE, value).apply()
    
    var timelapseSpeedUp: Int
        get() = prefs.getInt(KEY_TIMELAPSE_SPEEDUP, 2)
        set(value) = prefs.edit().putInt(KEY_TIMELAPSE_SPEEDUP, value.coerceIn(2, 60)).apply()
    
    // Convenience accessors with backward compatibility
    var trimStartMs: Long
        get() = trimStart
        set(value) { trimStart = value }
    
    var trimEndMs: Long
        get() = trimEnd
        set(value) { trimEnd = value }
    
    var nnapiDeviceName: String?
        get() = nnapiDevice
        set(value) { nnapiDevice = value }
    
    var maxResolution: Int
        get() = maxProcessingResolution
        set(value) { maxProcessingResolution = value }
    
    // Helper: Get interpolation steps based on quality preset and speed
    fun getInterpolationSteps(): Int {
        val baseSteps = when(qualityPreset) {
            QualityPreset.FAST -> 1  // Insert 1 frame between each pair
            QualityPreset.BALANCED -> 3  // Insert 3 frames
            QualityPreset.QUALITY -> 7  // Insert 7 frames
        }
        
        val multiplier = if (speedFactor == SpeedFactor.CUSTOM) {
            customSpeedMultiplier
        } else {
            speedFactor.multiplier
        }
        
        // Calculate steps to achieve target slowdown
        // If we want 2x slowdown, we need 1 frame between each pair (doubles frame count)
        // If we want 4x slowdown, we need 3 frames between each pair
        return (multiplier - 1).coerceAtLeast(1)
    }
    
    // Reset to defaults
    fun resetToDefaults() {
        prefs.edit().clear().apply()
    }
    
    companion object {
        private const val PREFS_NAME = "video_interpolation_settings"
        
        // Keys
        private const val KEY_INTERP_MODE = "interp_mode"
        private const val KEY_QUALITY_PRESET = "quality_preset"
        private const val KEY_SPEED_FACTOR = "speed_factor"
        private const val KEY_CUSTOM_SPEED = "custom_speed"
        private const val KEY_TARGET_FPS = "target_fps"
        private const val KEY_EXPORT_CODEC = "export_codec"
        private const val KEY_EXPORT_BITRATE = "export_bitrate"
        private const val KEY_MAX_RESOLUTION = "max_resolution"
        private const val KEY_NNAPI_DEVICE = "nnapi_device"
        private const val KEY_USE_NNAPI = "use_nnapi"
        private const val KEY_FRAME_PADDING = "frame_padding"
        private const val KEY_BATCH_ENABLED = "batch_enabled"
        private const val KEY_TRIM_START = "trim_start"
        private const val KEY_TRIM_END = "trim_end"
        private const val KEY_TRIM_ENABLED = "trim_enabled"
        private const val KEY_SCENE_DETECTION = "scene_detection"
        private const val KEY_SCENE_THRESHOLD = "scene_threshold"
        private const val KEY_EXPORT_GIF = "export_gif"
        private const val KEY_GIF_MAX_DIM = "gif_max_dim"
        private const val KEY_TIMELAPSE_MODE = "timelapse_mode"
        private const val KEY_TIMELAPSE_SPEEDUP = "timelapse_speedup"
        
        @Volatile
        private var instance: AppSettings? = null
        
        fun getInstance(context: Context): AppSettings {
            return instance ?: synchronized(this) {
                instance ?: AppSettings(context.applicationContext).also { instance = it }
            }
        }
    }
}
