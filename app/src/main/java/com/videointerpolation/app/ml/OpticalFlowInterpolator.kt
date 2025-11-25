package com.videointerpolation.app.ml

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Lightweight block-matching optical flow and warping for frame interpolation.
 * This is CPU-based and intended for downscaled frames (<= ~480p) for speed.
 */
class OpticalFlowInterpolator(
    private val blockSize: Int = 8,
    private val searchRadius: Int = 4
) {

    /**
     * Generate one intermediate frame using optical flow between two frames.
     * Steps are handled by caller via alpha in (0,1).
     */
    fun interpolate(frame1: Bitmap, frame2: Bitmap, alpha: Float): Bitmap {
        val width = min(frame1.width, frame2.width)
        val height = min(frame1.height, frame2.height)

        // Extract ARGB arrays
        val pixels1 = IntArray(width * height)
        val pixels2 = IntArray(width * height)
        frame1.getPixels(pixels1, 0, width, 0, 0, width, height)
        frame2.getPixels(pixels2, 0, width, 0, 0, width, height)

        // Convert to grayscale for flow estimation
        val gray1 = IntArray(width * height)
        val gray2 = IntArray(width * height)
        for (i in 0 until width * height) {
            val p1 = pixels1[i]
            val r1 = (p1 shr 16) and 0xff
            val g1 = (p1 shr 8) and 0xff
            val b1 = p1 and 0xff
            gray1[i] = (0.299f * r1 + 0.587f * g1 + 0.114f * b1).toInt()

            val p2 = pixels2[i]
            val r2 = (p2 shr 16) and 0xff
            val g2 = (p2 shr 8) and 0xff
            val b2 = p2 and 0xff
            gray2[i] = (0.299f * r2 + 0.587f * g2 + 0.114f * b2).toInt()
        }

        // Estimate flow using block matching (frame1 -> frame2)
        val dx = FloatArray(width * height)
        val dy = FloatArray(width * height)
        computeBlockMatchingFlow(gray1, gray2, width, height, dx, dy)

        // Warp both frames toward the intermediate time and blend
        val out = IntArray(width * height)
        for (y in 0 until height) {
            val row = y * width
            for (x in 0 until width) {
                val idx = row + x
                val fx = dx[idx]
                val fy = dy[idx]

                val x1 = x - alpha * fx
                val y1 = y - alpha * fy
                val x2 = x + (1f - alpha) * fx
                val y2 = y + (1f - alpha) * fy

                val c1 = bilinearSampleColor(pixels1, width, height, x1, y1)
                val c2 = bilinearSampleColor(pixels2, width, height, x2, y2)

                val a = ((c1 ushr 24) * (1f - alpha) + (c2 ushr 24) * alpha).toInt().coerceIn(0, 255)
                val r = ((((c1 shr 16) and 0xff) * (1f - alpha) + ((c2 shr 16) and 0xff) * alpha)).toInt().coerceIn(0, 255)
                val g = ((((c1 shr 8) and 0xff) * (1f - alpha) + ((c2 shr 8) and 0xff) * alpha)).toInt().coerceIn(0, 255)
                val b = (((c1 and 0xff) * (1f - alpha) + (c2 and 0xff) * alpha)).toInt().coerceIn(0, 255)
                out[idx] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        result.setPixels(out, 0, width, 0, 0, width, height)
        return result
    }

    private fun computeBlockMatchingFlow(
        gray1: IntArray,
        gray2: IntArray,
        width: Int,
        height: Int,
        dxOut: FloatArray,
        dyOut: FloatArray
    ) {
        val bs = blockSize
        val sr = searchRadius

        var y = 0
        while (y < height) {
            var x = 0
            while (x < width) {
                val bx = min(bs, width - x)
                val by = min(bs, height - y)

                // Reference block in frame1
                var bestDx = 0
                var bestDy = 0
                var bestCost = Int.MAX_VALUE

                val yStart = max(y - sr, 0)
                val yEnd = min(y + sr, height - by)
                val xStart = max(x - sr, 0)
                val xEnd = min(x + sr, width - bx)

                var yy = yStart
                while (yy <= yEnd) {
                    var xx = xStart
                    while (xx <= xEnd) {
                        // SAD cost between block at (x,y) in gray1 and (xx,yy) in gray2
                        var cost = 0
                        var byy = 0
                        while (byy < by) {
                            val r1 = (y + byy) * width + x
                            val r2 = (yy + byy) * width + xx
                            var bxx = 0
                            while (bxx < bx) {
                                cost += abs(gray1[r1 + bxx] - gray2[r2 + bxx])
                                bxx++
                            }
                            byy++
                        }

                        if (cost < bestCost) {
                            bestCost = cost
                            bestDx = xx - x
                            bestDy = yy - y
                        }
                        xx++
                    }
                    yy++
                }

                // Assign vector to all pixels in the block
                var byy2 = 0
                while (byy2 < by) {
                    val base = (y + byy2) * width + x
                    var bxx2 = 0
                    while (bxx2 < bx) {
                        dxOut[base + bxx2] = bestDx.toFloat()
                        dyOut[base + bxx2] = bestDy.toFloat()
                        bxx2++
                    }
                    byy2++
                }

                x += bs
            }
            y += bs
        }
    }

    private fun bilinearSampleColor(pixels: IntArray, width: Int, height: Int, fx: Float, fy: Float): Int {
        val x = fx.coerceIn(0f, (width - 1).toFloat())
        val y = fy.coerceIn(0f, (height - 1).toFloat())
        val x0 = floor(x).toInt()
        val y0 = floor(y).toInt()
        val x1 = min(x0 + 1, width - 1)
        val y1 = min(y0 + 1, height - 1)
        val dx = x - x0
        val dy = y - y0

        val c00 = pixels[y0 * width + x0]
        val c10 = pixels[y0 * width + x1]
        val c01 = pixels[y1 * width + x0]
        val c11 = pixels[y1 * width + x1]

        fun lerp(a: Int, b: Int, t: Float): Float = a + (b - a) * t

        val a0 = lerp(c00 ushr 24, c10 ushr 24, dx)
        val a1 = lerp(c01 ushr 24, c11 ushr 24, dx)
        val a = lerp(a0.toInt(), a1.toInt(), dy).toInt().coerceIn(0, 255)

        val r0 = lerp((c00 shr 16) and 0xff, (c10 shr 16) and 0xff, dx)
        val r1 = lerp((c01 shr 16) and 0xff, (c11 shr 16) and 0xff, dx)
        val r = lerp(r0.toInt(), r1.toInt(), dy).toInt().coerceIn(0, 255)

        val g0 = lerp((c00 shr 8) and 0xff, (c10 shr 8) and 0xff, dx)
        val g1 = lerp((c01 shr 8) and 0xff, (c11 shr 8) and 0xff, dx)
        val g = lerp(g0.toInt(), g1.toInt(), dy).toInt().coerceIn(0, 255)

        val b0 = lerp(c00 and 0xff, c10 and 0xff, dx)
        val b1 = lerp(c01 and 0xff, c11 and 0xff, dx)
        val b = lerp(b0.toInt(), b1.toInt(), dy).toInt().coerceIn(0, 255)

        return (a shl 24) or (r shl 16) or (g shl 8) or b
    }

    fun interpolateInParallel(frames: List<Pair<Bitmap, Bitmap>>, alpha: Float): List<Bitmap> {
        return frames.parallelStream().map { (frame1, frame2) ->
            interpolate(frame1, frame2, alpha)
        }.toList()
    }
}
