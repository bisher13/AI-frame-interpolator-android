package com.videointerpolation.app.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.LruCache
import java.io.File
import java.io.FileOutputStream

/**
 * App-wide caching system for thumbnails, processed videos, and temporary data
 */
class CacheManager(private val context: Context) {
    
    companion object {
        private const val CACHE_DIR = "video_cache"
        private const val THUMBNAILS_DIR = "thumbnails"
        private const val TEMP_DIR = "temp"
        private const val MAX_MEMORY_CACHE_SIZE = 10 * 1024 * 1024 // 10MB
        
        @Volatile
        private var instance: CacheManager? = null
        
        fun getInstance(context: Context): CacheManager {
            return instance ?: synchronized(this) {
                instance ?: CacheManager(context.applicationContext).also { instance = it }
            }
        }
    }
    
    // Memory cache for bitmaps (LRU cache)
    private val memoryCache = object : LruCache<String, Bitmap>(MAX_MEMORY_CACHE_SIZE) {
        override fun sizeOf(key: String, bitmap: Bitmap): Int {
            return bitmap.byteCount
        }
    }
    
    // Recent videos list
    private val recentVideos = mutableListOf<VideoInfo>()
    private val maxRecentVideos = 20
    
    data class VideoInfo(
        val uri: Uri,
        val name: String,
        val timestamp: Long,
        val duration: Long,
        val thumbnailPath: String? = null
    )
    
    init {
        // Create cache directories
        getCacheDir().mkdirs()
        getThumbnailsDir().mkdirs()
        getTempDir().mkdirs()
    }
    
    private fun getCacheDir(): File {
        return File(context.cacheDir, CACHE_DIR)
    }
    
    private fun getThumbnailsDir(): File {
        return File(getCacheDir(), THUMBNAILS_DIR)
    }
    
    private fun getTempDir(): File {
        return File(getCacheDir(), TEMP_DIR)
    }
    
    /**
     * Cache a thumbnail in memory and disk
     */
    fun cacheThumbnail(key: String, bitmap: Bitmap): String? {
        try {
            // Add to memory cache
            memoryCache.put(key, bitmap)
            
            // Save to disk
            val file = File(getThumbnailsDir(), "$key.jpg")
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
            }
            return file.absolutePath
        } catch (e: Exception) {
            android.util.Log.e("CacheManager", "Error caching thumbnail", e)
            return null
        }
    }
    
    /**
     * Get cached thumbnail from memory or disk
     */
    fun getThumbnail(key: String): Bitmap? {
        // Check memory cache first
        memoryCache.get(key)?.let { return it }
        
        // Try disk cache
        val file = File(getThumbnailsDir(), "$key.jpg")
        if (file.exists()) {
            try {
                val bitmap = BitmapFactory.decodeFile(file.absolutePath)
                // Add back to memory cache
                bitmap?.let { memoryCache.put(key, it) }
                return bitmap
            } catch (e: Exception) {
                android.util.Log.e("CacheManager", "Error loading thumbnail", e)
            }
        }
        return null
    }
    
    /**
     * Add video to recent list
     */
    fun addRecentVideo(videoInfo: VideoInfo) {
        // Remove if already exists
        recentVideos.removeAll { it.uri == videoInfo.uri }
        
        // Add to front
        recentVideos.add(0, videoInfo)
        
        // Limit size
        while (recentVideos.size > maxRecentVideos) {
            recentVideos.removeAt(recentVideos.size - 1)
        }
    }
    
    /**
     * Get recent videos list
     */
    fun getRecentVideos(): List<VideoInfo> {
        return recentVideos.toList()
    }
    
    /**
     * Create temporary file for processing
     */
    fun createTempFile(prefix: String, suffix: String): File {
        return File.createTempFile(prefix, suffix, getTempDir())
    }
    
    /**
     * Clear old temporary files
     */
    fun clearOldTempFiles(olderThanMs: Long = 24 * 60 * 60 * 1000) {
        val now = System.currentTimeMillis()
        getTempDir().listFiles()?.forEach { file ->
            if (now - file.lastModified() > olderThanMs) {
                file.delete()
            }
        }
    }
    
    /**
     * Clear all caches
     */
    fun clearAllCaches() {
        memoryCache.evictAll()
        getThumbnailsDir().deleteRecursively()
        getTempDir().deleteRecursively()
        getThumbnailsDir().mkdirs()
        getTempDir().mkdirs()
        recentVideos.clear()
    }
    
    /**
     * Get cache size in bytes
     */
    fun getCacheSize(): Long {
        var size = 0L
        getCacheDir().walkTopDown().forEach { file ->
            if (file.isFile) {
                size += file.length()
            }
        }
        return size
    }
    
    /**
     * Format cache size for display
     */
    fun getFormattedCacheSize(): String {
        val size = getCacheSize()
        return when {
            size < 1024 -> "$size B"
            size < 1024 * 1024 -> "${size / 1024} KB"
            else -> "${size / (1024 * 1024)} MB"
        }
    }
}
