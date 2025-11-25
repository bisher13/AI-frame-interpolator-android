# ONNX → TFLite conversion

Two-step flow:
1) Convert ONNX to TensorFlow SavedModel (prefer `onnx2tf`)
2) Convert SavedModel to TFLite using `tf.lite.TFLiteConverter`

## Recommended environment

For reliability, use Python 3.10 with pinned packages:

```powershell
# From project root
python --version
# If not 3.10.x, install Python 3.10 and use it for a dedicated venv

# Create venv (replace py -3.10 if you have multiple Pythons)
py -3.10 -m venv .convert-venv
.\.convert-venv\Scripts\activate

pip install --upgrade pip
pip install -r tools/requirements-convert.txt
```

## Convert

```powershell
# Using the helper script
python tools/onnx_to_tflite.py --onnx rife_fp16.onnx --out rife_fp16.tflite --fp16 --optimize --saved-model-dir build\saved_model
```

If `onnx2tf` is available, the script will use it and then export TFLite. If not, it falls back to `onnx-tf`.

## Troubleshooting

- "cannot import name 'mapping' from 'onnx'":
  - Caused by onnx-tf expecting legacy onnx.mapping. The script shims this, but prefer using the recommended environment.
- Missing modules like `tf_keras`, `onnx_graphsurgeon`, `psutil`, `ai_edge_litert`, `tensorflow_addons`:
  - Install using the requirements file or individually via pip.
- Python 3.12+/3.13 issues:
  - Some converter stacks aren’t ready yet. Prefer Python 3.10 for conversion.

## Manual onnx2tf usage

```powershell
python -m onnx2tf --input_onnx_file_path rife_fp16.onnx --output_folder_path build\onnx2tf_out
python - <<'PY'
import tensorflow as tf
conv = tf.lite.TFLiteConverter.from_saved_model('build/onnx2tf_out/saved_model')
conv.target_spec.supported_types = [tf.float16]
open('rife_fp16.tflite','wb').write(conv.convert())
PY
```
