# ONNX → TFLite Conversion — Fix for Python 3.13 + TF 2.20

## Problem
Your current environment (Python 3.13.3 + TensorFlow 2.20.0) is **incompatible** with ONNX conversion tools:
- `onnx-tf` requires `tensorflow-addons` (not available for TF 2.20)
- `onnx2tf` requires `ai-edge-litert` (not available for Python 3.13)

## Solution 1: Use Python 3.10 (Recommended)

Create a separate conversion environment:

```powershell
# Install Python 3.10 if not present (from python.org or Microsoft Store)
# Then:

py -3.10 -m venv .convert-venv
.\.convert-venv\Scripts\activate

pip install --upgrade pip
pip install -r tools/requirements-convert.txt

# Convert
python tools/onnx_to_tflite_simple.py rife_fp16.onnx rife_fp16.tflite

# Verify
if (Test-Path rife_fp16.tflite) { "SUCCESS" } else { "FAILED" }

# Deactivate when done
deactivate
```

## Solution 2: Use Docker (Cross-platform, hermetic)

If Python 3.10 isn't available:

```powershell
# Create Dockerfile in tools/
docker build -t onnx-converter -f tools/Dockerfile .
docker run --rm -v ${PWD}:/workspace onnx-converter python /workspace/tools/onnx_to_tflite_simple.py /workspace/rife_fp16.onnx /workspace/rife_fp16.tflite
```

## Solution 3: Use pre-converted model

If you have access to a TFLite version of your RIFE model from another source, copy it to:
```
app/src/main/assets/rife_fp16.tflite
```

## Why this happens

Python 3.13 is very new (April 2025) and many ML converter libraries haven't caught up yet. TensorFlow 2.20 dropped compatibility with tensorflow-addons. The pinned environment (Python 3.10 + TF 2.12) avoids all these issues.

## What's in tools/requirements-convert.txt

```
tensorflow==2.12.0
onnx==1.13.1
onnx2tf==1.20.0
onnx-graphsurgeon>=0.3.27
psutil>=5.9.0
tensorflow-addons==0.21.0
tf-keras
```

This stack is tested and stable on Windows x64.
