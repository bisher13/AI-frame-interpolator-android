# NNAPI Device Selection for ONNX Runtime

## Summary

Your logs show ONNX Runtime is using NNAPI but falling back to CPU (`nnapi-reference`). I've added **device targeting** to force hardware acceleration (DSP/GPU/NPU).

## Changes Made

### 1. OnnxInterpolator.kt - Added Device Selection
```kotlin
// New fields
private var nnapiDeviceName: String? = null
private var useNnapiCpuOnly: Boolean = false

// New methods
fun setNnapiDeviceName(deviceName: String?)
fun setNnapiCpuOnly(cpuOnly: Boolean)
fun getAvailableNnapiDevices(): List<String>
fun getAccelerationInfo(): String  // Returns "ONNX NNAPI (device)" or "ONNX CPU"
```

**Device targeting via NNAPI flags:**
- `nnapi_accelerator_name`: Target specific hardware (e.g., "qti-dsp", "qti-gpu", "qti-hta")
- `allow_fp16`: Enable FP16 acceleration
- `cpu_only`: Force CPU mode for debugging

### 2. FrameInterpolator.kt - Pass Device to OnnxInterpolator
Now when you call `setNnapiAcceleratorName("qti-dsp")` on FrameInterpolator, it configures **both** TFLite and ONNX Runtime backends.

### 3. Acceleration Info Display
- `FrameInterpolator.getAccelerationInfo()` → `OnnxInterpolator.getAccelerationInfo()`
- Shows "ONNX NNAPI (qti-dsp)" when device is targeted
- Shows "ONNX NNAPI (auto)" when auto-selected
- Shows "ONNX CPU" when falling back to CPU

## Testing Instructions

### Method 1: Using Existing UI (MainActivity)
Your app already has NNAPI accelerator UI:

1. **Build and install:**
   - Open project in Android Studio
   - Build → Make Project (Ctrl+F9)
   - Run → Run 'app' (Shift+F10)

2. **Test device targeting:**
   - Enter `qti-dsp` in the accelerator field
   - Enable the "Use NNAPI" switch
   - Click "Apply"
   - Check the acceleration status text

3. **Verify via logcat:**
   ```powershell
   adb logcat | Select-String -Pattern "OnnxInterpolator|ExecutionPlan"
   ```
   
   Look for:
   - `[INFO] NNAPI targeting device: qti-dsp`
   - `ModelBuilder::findBestDeviceForEachOperation(...) = 1 (qti-dsp)` ← Hardware!
   
   (Not `nnapi-reference` anymore)

### Method 2: Programmatic Test
Add to MainActivity's onCreate:
```kotlin
// Test ONNX device selection
val interpolator = FrameInterpolator(this, InterpMode.ONNX)
interpolator.setNnapiAcceleratorName("qti-dsp")  // or "qti-gpu", "qti-hta"
interpolator.initialize()
Log.i("TEST", "Acceleration: ${interpolator.getAccelerationInfo()}")
```

### Common Device Names by Vendor

**Qualcomm Snapdragon:**
- `qti-dsp` - Hexagon DSP (fastest for AI)
- `qti-gpu` - Adreno GPU
- `qti-hta` - Hexagon Tensor Accelerator (newer chips)

**MediaTek:**
- `mtk-apu` - AI Processing Unit
- `mtk-gpu` - Mali GPU

**Samsung Exynos:**
- `npu` - Neural Processing Unit
- `gpu` - Mali GPU

**Generic:**
- `dsp` - Digital Signal Processor
- `gpu` - GPU acceleration
- Leave empty for auto-selection

## Troubleshooting

### Still seeing `nnapi-reference`?
1. **Check device support:**
   ```kotlin
   val devices = onnxInterpolator.getAvailableNnapiDevices()
   Log.i("TEST", "Available: $devices")
   ```

2. **Try different device names:**
   - `qti-dsp` (most common for Qualcomm)
   - `dsp`
   - `gpu`
   - `npu`

3. **Check model compatibility:**
   Your RIFE model uses `GridSample` operator which may not be supported by all hardware accelerators. If a device doesn't support it, NNAPI will fall back to CPU.

4. **Verify ONNX Runtime version:**
   The device targeting feature requires ONNX Runtime 1.8+. Check `app/build.gradle`:
   ```gradle
   implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'  // or newer
   ```

### Model doesn't work on hardware?
If you get errors or worse quality with hardware acceleration:
- Try `setNnapiCpuOnly(false)` and compare results
- Some operations in your model may not be hardware-optimized
- CPU execution with ONNX Runtime is still very fast

## Why Not TFLite?

The ONNX → TFLite conversion **failed** because:
- Your model uses `GridSample` operator
- onnx-tf (Python converter) doesn't support GridSample
- TFLite doesn't have native GridSample support

**ONNX Runtime is the right choice** for your RIFE model because:
✅ Supports GridSample natively
✅ NNAPI backend works directly with ONNX
✅ Better operator coverage than TFLite
✅ No conversion quality loss

## Logs Analysis

Your current logs show:
```
OnnxInterpolator: NNAPI EP enabled
ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(SUB:0) = 0 (nnapi-reference)
                                                                     ^^^^^^^^^^^^^^^^^^^
                                                                     CPU fallback!
```

After fix, you should see:
```
OnnxInterpolator: NNAPI targeting device: qti-dsp
ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(SUB:0) = 1 (qti-dsp)
                                                                     ^^^^^^^^^^^^^
                                                                     Hardware!
```

## Next Steps

1. **Rebuild the app** in Android Studio
2. **Enter device name** in the UI (try "qti-dsp" first for Qualcomm)
3. **Check logcat** for device assignment
4. **Compare performance** with and without device targeting
5. If still using CPU, **try different device names** from the list above

The code changes are complete and ready to test!
