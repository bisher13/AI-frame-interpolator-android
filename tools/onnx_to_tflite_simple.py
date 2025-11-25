#!/usr/bin/env python3
"""
Minimal ONNX -> TFLite converter using onnx2tf directly.
Bypasses onnx-tf entirely to avoid tensorflow-addons dependency issues.

Requirements:
  pip install onnx tensorflow onnx2tf onnx-graphsurgeon psutil

Usage:
  python tools/onnx_to_tflite_simple.py rife_fp16.onnx rife_fp16.tflite
"""

import sys
import os
import tempfile
import shutil

def main():
    if len(sys.argv) < 3:
        print("Usage: python onnx_to_tflite_simple.py <input.onnx> <output.tflite>")
        return 1
    
    onnx_path = sys.argv[1]
    tflite_path = sys.argv[2]
    
    if not os.path.isfile(onnx_path):
        print(f"[ERR] ONNX file not found: {onnx_path}")
        return 1
    
    # Step 1: Convert ONNX -> SavedModel using onnx2tf
    print("[1/2] Converting ONNX -> SavedModel with onnx2tf...")
    tmpdir = tempfile.mkdtemp(prefix="onnx2tflite_")
    try:
        try:
            from onnx2tf import convert
            convert(
                input_onnx_file_path=onnx_path,
                output_folder_path=tmpdir,
                copy_onnx_input_output_names_to_tflite=False,
            )
        except ImportError as e:
            print(f"[ERR] onnx2tf not installed or missing dependencies: {e}")
            print("Install with: pip install onnx2tf onnx-graphsurgeon psutil")
            return 2
        except Exception as e:
            print(f"[ERR] onnx2tf conversion failed: {e}")
            return 2
        
        # onnx2tf creates saved_model subdir
        saved_model_dir = os.path.join(tmpdir, "saved_model")
        if not os.path.isdir(saved_model_dir):
            print(f"[ERR] SavedModel not found at {saved_model_dir}")
            return 2
        
        # Step 2: Convert SavedModel -> TFLite
        print("[2/2] Converting SavedModel -> TFLite...")
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        os.makedirs(os.path.dirname(os.path.abspath(tflite_path)) or ".", exist_ok=True)
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        
        print(f"[OK] TFLite model written to: {tflite_path}")
        print(f"     Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    sys.exit(main())
