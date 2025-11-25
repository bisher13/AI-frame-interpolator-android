"""Download and inspect Google FILM model from TensorFlow Hub."""
import os
import sys

def download_film_model():
    """Download FILM model from TensorFlow Hub."""
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        print(f"TensorFlow version: {tf.__version__}")
        
        # FILM model on TFHub
        model_url = "https://tfhub.dev/google/film/1"
        print(f"\n[INFO] Downloading FILM from TensorFlow Hub: {model_url}")
        print("[INFO] This may take a few minutes (model is ~75MB)...")
        
        # Download and cache the model
        interpolator = hub.load(model_url)
        print(f"[SUCCESS] Model downloaded and loaded!")
        
        # Get model signatures
        print(f"\n[INFO] Available signatures: {list(interpolator.signatures.keys())}")
        
        # Get default signature (inference function)
        infer = interpolator.signatures['serving_default']
        print(f"\n[INFO] Inputs: {infer.structured_input_signature}")
        print(f"[INFO] Outputs: {infer.structured_outputs}")
        
        # Save as SavedModel for inspection
        output_dir = "film_models/film_tfhub"
        print(f"\n[INFO] Saving to {output_dir}...")
        tf.saved_model.save(interpolator, output_dir)
        print(f"[SUCCESS] Saved to {output_dir}")
        
        return output_dir
        
    except ImportError as e:
        print(f"[ERR] Missing dependency: {e}")
        print("[INFO] Install with: pip install tensorflow tensorflow-hub")
        return None
    except Exception as e:
        print(f"[ERR] Download failed: {e}")
        return None

def inspect_saved_model(model_path):
    """Inspect SavedModel structure and operators."""
    try:
        import tensorflow as tf
        print(f"\n[INFO] Inspecting SavedModel at {model_path}")
        
        # Load and inspect
        model = tf.saved_model.load(model_path)
        concrete_func = model.signatures['serving_default']
        
        # Get graph def
        graph_def = concrete_func.graph.as_graph_def()
        
        # Count operators
        op_types = {}
        for node in graph_def.node:
            op_types[node.op] = op_types.get(node.op, 0) + 1
        
        print(f"\n[INFO] Model has {len(graph_def.node)} nodes")
        print(f"[INFO] Unique operator types: {len(op_types)}")
        print("\n[INFO] Top 20 operators:")
        for op, count in sorted(op_types.items(), key=lambda x: -x[1])[:20]:
            print(f"  {op}: {count}")
        
        # Check for problematic ops
        problematic = []
        if 'ResampleGrad' in op_types or 'Resample' in op_types:
            problematic.append("Resample/ResampleGrad (custom op)")
        # GridSample doesn't exist in TF, but check for custom ops
        custom_ops = [op for op in op_types if op.startswith('Custom') or 'Grid' in op]
        if custom_ops:
            problematic.extend(custom_ops)
        
        if problematic:
            print(f"\n[WARN] Potentially unsupported operators for NNAPI:")
            for op in problematic:
                print(f"  - {op}")
        else:
            print(f"\n[INFO] No obvious GridSample or custom ops detected")
            print("[INFO] Model uses standard TF ops (Conv2D, DepthwiseConv2D, etc.)")
        
        return op_types
        
    except Exception as e:
        print(f"[ERR] Inspection failed: {e}")
        return None

if __name__ == '__main__':
    print("=" * 60)
    print("Google FILM Model Download and Analysis")
    print("=" * 60)
    
    model_path = download_film_model()
    
    if model_path and os.path.exists(model_path):
        ops = inspect_saved_model(model_path)
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Model location: {os.path.abspath(model_path)}")
        print("Model format: TensorFlow SavedModel")
        print("Next steps:")
        print("  1. Convert to TFLite: Use tf.lite.TFLiteConverter")
        print("  2. Convert to ONNX: Use tf2onnx")
        print("  3. Test in Android app")
    else:
        print("\n[ERR] Download failed. Try manual download from:")
        print("https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy")
