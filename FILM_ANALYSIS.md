# Google FILM Model Analysis for Android NNAPI

## Model Information

**Name:** FILM (Frame Interpolation for Large Motion)  
**Source:** Google Research  
**Paper:** https://arxiv.org/abs/2202.04901  
**Code:** https://github.com/google-research/frame-interpolation (archived Oct 2024)  
**Pre-trained Models:** https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy

## Architecture Overview

FILM uses **standard TensorFlow operations** unlike RIFE which uses GridSample:

### ✅ Hardware-Friendly Operations (NNAPI Compatible)
- **Conv2D & DepthwiseConv2D** - Multi-scale feature extraction
- **ResizeNearestNeighbor / ResizeBilinear** - For image pyramids
- **Add, Mul, Sub, Concat** - Standard arithmetic
- **ReLU activations** - Well-supported

### ⚠️ Potential Issues
- **Resample** (custom TF op) - May not be in NNAPI delegate
- **High memory usage** - Multi-scale architecture needs ~200MB+ RAM
- **FP32 model** - Needs quantization to FP16/INT8 for mobile

## NNAPI Compatibility: ✅ MUCH BETTER THAN RIFE

**Why FILM is better for hardware acceleration:**

| Feature | RIFE | FILM |
|---------|------|------|
| GridSample | ❌ Yes (unsupported) | ✅ No |
| Standard Conv2D | ✅ Yes | ✅ Yes |
| Dynamic shapes | ❌ Yes | ⚠️ Minimal |
| Custom ops | ❌ GridSample | ⚠️ Resample (TF native) |
| NNAPI Support | ❌ CPU fallback | ✅ Likely GPU/DSP |

## Download & Conversion Steps

### Option 1: TensorFlow Hub (Recommended)

```powershell
# Using your Python 3.9 env
.\.convert-venv\Scripts\python.exe -m pip install tensorflow-hub

# Download FILM
.\.convert-venv\Scripts\python.exe tools/download_film.py
```

The script `tools/download_film.py` will:
1. Download from https://tfhub.dev/google/film/1 (~75MB)
2. Save as `film_models/film_tfhub/`
3. Analyze operators for NNAPI compatibility

### Option 2: Manual Download from Google Drive

1. Download from: https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy
2. Get `film_net/Style/saved_model` (best quality)
3. Extract to `film_models/film_style/`

## Conversion to Mobile Formats

### A. Convert to TFLite (For TFLite backend)

```powershell
.\.convert-venv\Scripts\python.exe -c "
import tensorflow as tf

# Load SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model('film_models/film_tfhub')

# Optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert
tflite_model = converter.convert()

# Save
with open('app/src/main/assets/film_fp16.tflite', 'wb') as f:
    f.write(tflite_model)

print('✅ Converted to film_fp16.tflite')
"
```

### B. Convert to ONNX (For ONNX Runtime backend)

```powershell
.\.convert-venv\Scripts\python.exe -m pip install tf2onnx

.\.convert-venv\Scripts\python.exe -m tf2onnx.convert `
  --saved-model film_models/film_tfhub `
  --output film_fp32.onnx `
  --opset 13

# Optional: Convert FP32 → FP16
.\.convert-venv\Scripts\python.exe -c "
import onnx
from onnxconverter_common import float16

model = onnx.load('film_fp32.onnx')
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, 'app/src/main/assets/film_fp16.onnx')
print('✅ Converted to film_fp16.onnx')
"
```

## Integration into Android App

### Your app already supports both backends!

**Option A: Use TFLite (if conversion succeeds)**
- Model: `app/src/main/assets/film_fp16.tflite`
- Set mode: `InterpMode.TFLITE`
- NNAPI device: `qti-dsp` or `qti-gpu`

**Option B: Use ONNX Runtime (recommended)**
- Model: `app/src/main/assets/film_fp16.onnx`
- Set mode: `InterpMode.ONNX`
- NNAPI device: Auto or `qti-dsp`

### Code Changes Needed

Your existing code should work with minimal changes. Just update model names:

```kotlin
// In MainActivity or wherever you initialize
val interpolator = FrameInterpolator(this, InterpMode.ONNX) // or TFLITE
interpolator.setNnapiAcceleratorName("qti-dsp")
interpolator.initialize(
    modelPath = "film_fp16.tflite",  // or keep rife_fp16.tflite
    onnxModelPath = "film_fp16.onnx"  // or keep rife_fp16.onnx
)
```

## Expected Performance vs RIFE

### Hardware Acceleration
- **FILM**: ✅ GPU/DSP likely to work (standard ops)
- **RIFE**: ❌ CPU fallback (GridSample unsupported)

### Quality
- **FILM**: Better for large motion, more stable
- **RIFE**: Faster but struggles with large motion

### Speed (on hardware)
- **FILM**: ~30-50ms per frame (GPU/DSP)
- **RIFE**: ~80-120ms per frame (CPU fallback)

### Memory
- **FILM**: ~200-300MB RAM (multi-scale)
- **RIFE**: ~80-150MB RAM

## Quick Test Commands

After downloading FILM, test conversion:

```powershell
# Test TFLite conversion
.\.convert-venv\Scripts\python.exe -c "
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('film_models/film_tfhub')
tflite = converter.convert()
print(f'TFLite size: {len(tflite)/1024/1024:.1f}MB')
"

# Test ONNX conversion
.\.convert-venv\Scripts\python.exe -m tf2onnx.convert \
  --saved-model film_models/film_tfhub \
  --output film_test.onnx \
  --opset 13
```

## Troubleshooting

### "Unsupported operations" during TFLite conversion
- FILM's `Resample` op may not convert cleanly
- **Solution**: Use ONNX Runtime instead (better operator support)

### "Out of memory" errors
- FILM needs more RAM than RIFE
- **Solution**: Lower resolution input (resize to 480p or 360p before interpolation)

### Still getting `nnapi-reference` in logs
- Check if TFLite model is FP16 quantized
- Try different NNAPI device names (`qti-gpu` instead of `qti-dsp`)
- FILM should work better than RIFE for hardware acceleration

## Recommendation

**Use ONNX Runtime backend with FILM** because:
1. ✅ Better operator coverage (Resample supported)
2. ✅ Standard Conv2D ops work well on NNAPI
3. ✅ No GridSample issues
4. ✅ Your device targeting code already works
5. ✅ Quality is better than RIFE for large motion

## Next Steps

1. **Download FILM:**
   ```powershell
   .\.convert-venv\Scripts\python.exe tools/download_film.py
   ```

2. **Convert to ONNX:**
   ```powershell
   .\.convert-venv\Scripts\pip install tf2onnx onnxconverter-common
   .\.convert-venv\Scripts\python.exe -m tf2onnx.convert --saved-model film_models/film_tfhub --output app/src/main/assets/film_fp16.onnx --opset 13
   ```

3. **Update Android app** to use `film_fp16.onnx` instead of `rife_fp16.onnx`

4. **Test with device targeting** and check logs for hardware usage

Let me know if you want me to run the download/conversion automatically or if you need help with a specific step!
