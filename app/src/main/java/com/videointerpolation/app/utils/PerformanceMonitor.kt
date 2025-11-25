package com.videointerpolation.app.utils

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.os.SystemClock
import kotlin.math.roundToInt

/**
 * Performance monitoring utility for tracking FPS, memory, and processing speed
 */
class PerformanceMonitor(private val context: Context) {
    
    private var startTime: Long = 0
    private var frameCount: Int = 0
    private var processedFrames: Int = 0
    private var totalFrames: Int = 0
    
    private val activityManager: ActivityManager by lazy {
        context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    }
    
    data class PerformanceStats(
        val fps: Double,
        val eta: String,
        val memoryUsedMb: Long,
        val memoryTotalMb: Long,
        val memoryPercent: Int,
        val elapsedTime: String,
        val progress: Int
    )
    
    /**
     * Start monitoring
     */
    fun start(totalFrames: Int) {
        this.totalFrames = totalFrames
        startTime = SystemClock.elapsedRealtime()
        frameCount = 0
        processedFrames = 0
    }
    
    /**
     * Update frame count
     */
    fun updateFrameCount(processedFrames: Int) {
        this.processedFrames = processedFrames
        frameCount++
    }
    
    /**
     * Get current FPS
     */
    fun getFps(): Double {
        val elapsed = (SystemClock.elapsedRealtime() - startTime) / 1000.0
        return if (elapsed > 0) frameCount / elapsed else 0.0
    }
    
    /**
     * Get ETA (estimated time to completion)
     */
    fun getEta(): String {
        if (processedFrames == 0 || totalFrames == 0) return "Calculating..."
        
        val elapsed = SystemClock.elapsedRealtime() - startTime
        val remainingFrames = totalFrames - processedFrames
        val msPerFrame = elapsed.toDouble() / processedFrames
        val remainingMs = (remainingFrames * msPerFrame).toLong()
        
        return formatTime(remainingMs)
    }
    
    /**
     * Get elapsed time
     */
    fun getElapsedTime(): String {
        val elapsed = SystemClock.elapsedRealtime() - startTime
        return formatTime(elapsed)
    }
    
    /**
     * Get progress percentage
     */
    fun getProgress(): Int {
        return if (totalFrames > 0) {
            ((processedFrames.toDouble() / totalFrames) * 100).roundToInt()
        } else 0
    }
    
    /**
     * Get memory usage
     */
    fun getMemoryUsage(): Pair<Long, Long> {
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        
        val totalMb = memoryInfo.totalMem / (1024 * 1024)
        val availableMb = memoryInfo.availMem / (1024 * 1024)
        val usedMb = totalMb - availableMb
        
        return Pair(usedMb, totalMb)
    }
    
    /**
     * Get native heap memory usage
     */
    fun getHeapMemory(): Pair<Long, Long> {
        val usedMb = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024)
        val maxMb = Runtime.getRuntime().maxMemory() / (1024 * 1024)
        return Pair(usedMb, maxMb)
    }
    
    /**
     * Get all performance stats
     */
    fun getStats(): PerformanceStats {
        val (usedMb, totalMb) = getMemoryUsage()
        val percent = ((usedMb.toDouble() / totalMb) * 100).roundToInt()
        
        return PerformanceStats(
            fps = getFps(),
            eta = getEta(),
            memoryUsedMb = usedMb,
            memoryTotalMb = totalMb,
            memoryPercent = percent,
            elapsedTime = getElapsedTime(),
            progress = getProgress()
        )
    }
    
    /**
     * Format milliseconds to readable time
     */
    private fun formatTime(ms: Long): String {
        val seconds = (ms / 1000) % 60
        val minutes = (ms / (1000 * 60)) % 60
        val hours = ms / (1000 * 60 * 60)
        
        return when {
            hours > 0 -> String.format("%d:%02d:%02d", hours, minutes, seconds)
            minutes > 0 -> String.format("%d:%02d", minutes, seconds)
            else -> "${seconds}s"
        }
    }
    
    /**
     * Check if memory is low
     */
    fun isMemoryLow(): Boolean {
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        return memoryInfo.lowMemory
    }
    
    /**
     * Log performance stats
     */
    fun logStats() {
        val stats = getStats()
        android.util.Log.i("PerformanceMonitor", """
            FPS: ${String.format("%.2f", stats.fps)}
            ETA: ${stats.eta}
            Memory: ${stats.memoryUsedMb}MB / ${stats.memoryTotalMb}MB (${stats.memoryPercent}%)
            Progress: ${stats.progress}%
            Elapsed: ${stats.elapsedTime}
        """.trimIndent())
    }
}
