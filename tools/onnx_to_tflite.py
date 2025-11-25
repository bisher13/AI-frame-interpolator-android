#!/usr/bin/env python3
"""
Convert an ONNX model to TensorFlow Lite (.tflite).

Flow:
1) Convert ONNX -> TensorFlow SavedModel (tries onnx2tf if available, else onnx-tf)
2) Convert SavedModel -> TFLite using tf.lite.TFLiteConverter

Usage (PowerShell on Windows):
  python tools/onnx_to_tflite.py --onnx path/to/model.onnx --out path/to/model.tflite --fp16 --optimize

Notes:
- Requires Python 3.9+ and a CPU TensorFlow install.
- Install deps:
    pip install onnx tensorflow onnx-tf
  Optional (often more reliable for TFLite):
    pip install onnx2tf
"""

import argparse
import os
import sys
import tempfile
import shutil
import platform


def _convert_onnx_to_saved_model(onnx_path: str, out_dir: str) -> None:
    """Convert ONNX to TensorFlow SavedModel at out_dir.
    Tries onnx2tf first if installed; falls back to onnx-tf.
    """
    # Try onnx2tf (preferred for TFLite targets)
    try:
        try:
            from onnx2tf import convert as onnx2tf_convert  # type: ignore
        except Exception as e:
            # Broaden handling: any import-time error (missing subdeps like tf_keras, ai_edge_litert) is treated as not available
            raise ModuleNotFoundError(str(e))
        print("[INFO] Using onnx2tf to convert ONNX -> SavedModel ...")
        onnx2tf_convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=out_dir,
            # Favor TFLite compatibility
            copy_onnx_input_output_names_to_tflite=False,
        )
        # onnx2tf generates a SavedModel under out_dir/saved_model
        saved_model_dir = os.path.join(out_dir, "saved_model")
        if os.path.isdir(saved_model_dir):
            # Re-root output to the SavedModel directory for consistency
            # Move saved_model content up one level (out_dir becomes the SavedModel dir)
            tmp = os.path.join(out_dir, "__tmp_saved_model__")
            os.rename(saved_model_dir, tmp)
            # Clean everything else created by onnx2tf
            for name in os.listdir(out_dir):
                p = os.path.join(out_dir, name)
                if os.path.isdir(p) or os.path.isfile(p):
                    if os.path.basename(p) != "__tmp_saved_model__":
                        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            # Move saved model back to out_dir
            for name in os.listdir(tmp):
                os.rename(os.path.join(tmp, name), os.path.join(out_dir, name))
            os.rmdir(tmp)
        print(f"[OK] SavedModel written to: {out_dir}")
        return
    except ModuleNotFoundError:
        print("[INFO] onnx2tf unavailable (or missing optional deps). Falling back to onnx-tf ...")
    except Exception as e:
        print(f"[WARN] onnx2tf failed: {e}. Falling back to onnx-tf ...")

    # Fallback: onnx-tf
    # Try to maintain compatibility with newer onnx packages where onnx.mapping was removed
    import onnx  # type: ignore
    try:
        from onnx import mapping as _onnx_mapping  # type: ignore
    except Exception:
        # Provide a minimal shim for onnx.mapping expected by onnx-tf
        import types, numpy as _np
        from onnx import TensorProto as _TP  # type: ignore
        _TENSOR_TYPE_TO_NP_TYPE = {
            _TP.FLOAT: _np.float32,
            _TP.UINT8: _np.uint8,
            _TP.INT8: _np.int8,
            _TP.UINT16: _np.uint16,
            _TP.INT16: _np.int16,
            _TP.INT32: _np.int32,
            _TP.INT64: _np.int64,
            _TP.BOOL: _np.bool_,
            _TP.FLOAT16: _np.float16,
            _TP.DOUBLE: _np.float64,
            _TP.UINT32: _np.uint32 if hasattr(_np, 'uint32') else _np.uint64,
            _TP.UINT64: _np.uint64,
            _TP.COMPLEX64: _np.complex64 if hasattr(_np, 'complex64') else _np.complex128,
            _TP.COMPLEX128: _np.complex128,
            _TP.STRING: _np.object_,
        }
        _NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in _TENSOR_TYPE_TO_NP_TYPE.items()}
        onnx.mapping = types.SimpleNamespace(  # type: ignore[attr-defined]
            TENSOR_TYPE_TO_NP_TYPE=_TENSOR_TYPE_TO_NP_TYPE,
            NP_TYPE_TO_TENSOR_TYPE=_NP_TYPE_TO_TENSOR_TYPE,
        )
    from onnx_tf.backend import prepare  # type: ignore
    print("[INFO] Using onnx-tf to convert ONNX -> SavedModel ...")
    model = onnx.load(onnx_path)
    tf_rep = prepare(model)
    # onnx-tf exports a SavedModel when given a directory path
    tf_rep.export_graph(out_dir)
    print(f"[OK] SavedModel written to: {out_dir}")


def _convert_saved_model_to_tflite(saved_model_dir: str, tflite_out: str, fp16: bool, optimize: bool, allow_custom_ops: bool) -> None:
    import tensorflow as tf  # type: ignore

    print("[INFO] Converting SavedModel -> TFLite ...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if allow_custom_ops:
        converter.allow_custom_ops = True
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if fp16:
        # Enable float16 weights to reduce size and often speed up on GPUs/NNAPI (with FP16)
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(tflite_out)), exist_ok=True)
    with open(tflite_out, "wb") as f:
        f.write(tflite_model)
    print(f"[OK] TFLite model written to: {tflite_out}")


def main() -> int:
    p = argparse.ArgumentParser(description="Convert ONNX to TensorFlow Lite")
    p.add_argument("--onnx", required=True, help="Path to input .onnx model")
    p.add_argument("--out", required=True, help="Path to output .tflite file")
    p.add_argument("--saved-model-dir", default=None, help="Optional output dir for intermediate SavedModel (kept if provided)")
    p.add_argument("--fp16", action="store_true", help="Enable FP16 weights in TFLite (smaller, faster on some accelerators)")
    p.add_argument("--optimize", action="store_true", help="Enable default TFLite optimizations")
    p.add_argument("--allow-custom-ops", action="store_true", help="Allow custom ops in TFLite converter (use if needed)")
    args = p.parse_args()

    # Preflight: warn if Python version is >=3.12 (converter stacks often break there)
    py_ver = sys.version_info
    if py_ver.major == 3 and py_ver.minor >= 12:
        print(f"[WARN] Detected Python {py_ver.major}.{py_ver.minor}. For best compatibility, use Python 3.10 with requirements in tools/requirements-convert.txt.")
    if platform.system().lower() == 'windows':
        print("[INFO] Running on Windows; long path or antivirus may slow conversion.")

    onnx_path = os.path.abspath(args.onnx)
    tflite_out = os.path.abspath(args.out)
    if not os.path.isfile(onnx_path):
        print(f"[ERR] ONNX file not found: {onnx_path}")
        return 1

    # Prepare SavedModel dir
    if args.saved_model_dir:
        saved_model_dir = os.path.abspath(args.saved_model_dir)
        os.makedirs(saved_model_dir, exist_ok=True)
        keep_saved_model = True
    else:
        tmpdir = tempfile.TemporaryDirectory(prefix="onnx2tflite_")
        saved_model_dir = os.path.join(tmpdir.name, "saved_model")
        keep_saved_model = False

    try:
        os.makedirs(saved_model_dir, exist_ok=True)
        _convert_onnx_to_saved_model(onnx_path, saved_model_dir)
        _convert_saved_model_to_tflite(
            saved_model_dir,
            tflite_out,
            fp16=args.fp16,
            optimize=args.optimize,
            allow_custom_ops=args.allow_custom_ops,
        )
        print("[DONE] Conversion complete.")
    except Exception as e:
        print(f"[ERR] Conversion failed: {e}")
        return 2
    finally:
        if not keep_saved_model:
            # Clean up temp directory
            try:
                tmpdir.cleanup()  # type: ignore
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
