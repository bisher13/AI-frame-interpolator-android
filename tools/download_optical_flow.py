"""
Download lightweight GPU-accelerated optical flow models for Android.

Options:
1. RAFT-Lite (mobile-optimized RAFT)
2. PWC-Net mobile variant
3. LiteFlowNet
"""

import os
import sys

def download_raft_lite():
    """Download RAFT-Lite model for mobile inference."""
    try:
        import torch
        import urllib.request
        print("[INFO] Downloading RAFT-Lite checkpoint...")
        
        # RAFT-Lite is a quantized version of RAFT for mobile
        # Original RAFT: https://github.com/princeton-vl/RAFT
        model_url = "https://github.com/princeton-vl/RAFT/raw/master/models/raft-small.pth"
        output_path = "optical_flow_models/raft_small.pth"
        
        os.makedirs("optical_flow_models", exist_ok=True)
        
        if os.path.exists(output_path):
            print(f"[INFO] Model already exists: {output_path}")
            return output_path
        
        print(f"[INFO] Downloading from {model_url}...")
        urllib.request.urlretrieve(model_url, output_path)
        print(f"[SUCCESS] Downloaded to {output_path}")
        
        # TODO: Convert to ONNX/TFLite
        print("\n[NEXT] Convert PyTorch model to ONNX:")
        print("  python tools/convert_raft_to_onnx.py")
        
        return output_path
        
    except Exception as e:
        print(f"[ERR] Failed to download RAFT-Lite: {e}")
        return None

def create_simple_optical_flow_tflite():
    """Create a simple CNN-based optical flow model using TensorFlow."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        print("[INFO] Creating simple CNN optical flow model...")
        
        # Simple FlowNet-style architecture (very lightweight)
        inputs = keras.Input(shape=(None, None, 6), name='images')  # Stacked frame1+frame2
        
        # Encoder
        x = keras.layers.Conv2D(32, 7, strides=2, padding='same', activation='relu')(inputs)
        x = keras.layers.Conv2D(64, 5, strides=2, padding='same', activation='relu')(x)
        x = keras.layers.Conv2D(96, 3, strides=2, padding='same', activation='relu')(x)
        
        # Decoder
        x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
        x = keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
        x = keras.layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(x)
        
        # Flow output (2 channels: dx, dy)
        outputs = keras.layers.Conv2D(2, 3, padding='same', name='flow')(x)
        
        model = keras.Model(inputs, outputs)
        model.summary()
        
        # Convert to TFLite
        print("\n[INFO] Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        output_path = "app/src/main/assets/optical_flow_fp16.tflite"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"[SUCCESS] Saved to {output_path}")
        print(f"[INFO] Model size: {len(tflite_model) / 1024:.1f} KB")
        print("\n[WARN] This is an untrained skeleton model.")
        print("[WARN] For real optical flow, use pre-trained PWC-Net or RAFT-Lite.")
        
        return output_path
        
    except Exception as e:
        print(f"[ERR] Failed to create TFLite model: {e}")
        return None

def recommend_pretrained_models():
    """Recommend pre-trained models available online."""
    print("\n" + "="*60)
    print("PRE-TRAINED GPU OPTICAL FLOW MODELS")
    print("="*60)
    
    recommendations = [
        {
            "name": "PWC-Net (TensorFlow)",
            "url": "https://github.com/philferriere/tfoptflow",
            "size": "~10MB",
            "speed": "Fast (GPU: ~20ms/frame)",
            "quality": "Good",
            "format": "SavedModel → TFLite/ONNX",
            "nnapi": "✅ Standard Conv2D ops"
        },
        {
            "name": "RAFT-Lite (PyTorch)",
            "url": "https://github.com/princeton-vl/RAFT",
            "size": "~5MB (small variant)",
            "speed": "Medium (GPU: ~40ms/frame)",
            "quality": "Excellent",
            "format": "PyTorch → ONNX",
            "nnapi": "⚠️ Some custom ops"
        },
        {
            "name": "LiteFlowNet",
            "url": "https://github.com/twhui/LiteFlowNet",
            "size": "~8MB",
            "speed": "Fast (GPU: ~25ms/frame)",
            "quality": "Very Good",
            "format": "Caffe → ONNX",
            "nnapi": "✅ Standard ops"
        },
        {
            "name": "FastFlowNet (MobileNet backbone)",
            "url": "https://github.com/ltkong218/FastFlowNet",
            "size": "~3MB",
            "speed": "Very Fast (GPU: ~15ms/frame)",
            "quality": "Good",
            "format": "PyTorch → ONNX",
            "nnapi": "✅ Mobile-optimized"
        }
    ]
    
    for i, model in enumerate(recommendations, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   URL: {model['url']}")
        print(f"   Size: {model['size']}")
        print(f"   Speed: {model['speed']}")
        print(f"   Quality: {model['quality']}")
        print(f"   Format: {model['format']}")
        print(f"   NNAPI: {model['nnapi']}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("For Android NNAPI acceleration:")
    print("  1st choice: FastFlowNet (smallest, fastest, mobile-optimized)")
    print("  2nd choice: PWC-Net (good balance of speed/quality)")
    print("  3rd choice: LiteFlowNet (best quality, slightly slower)")
    print("\nAll use standard Conv2D/DepthwiseConv2D → should work on GPU/DSP!")

if __name__ == '__main__':
    print("=" * 60)
    print("GPU Optical Flow Model Downloader")
    print("=" * 60)
    
    recommend_pretrained_models()
    
    print("\n" + "="*60)
    print("QUICK START")
    print("="*60)
    print("\nOption 1: Create untrained skeleton model (for testing integration)")
    print("  → python tools/download_optical_flow.py --create-skeleton")
    
    print("\nOption 2: Manual download and convert")
    print("  1. Get PWC-Net from: https://github.com/philferriere/tfoptflow")
    print("  2. Convert to ONNX: python -m tf2onnx.convert --saved-model ...")
    print("  3. Copy to: app/src/main/assets/optical_flow.onnx")
    
    print("\nOption 3: Use existing RIFE/FILM models")
    print("  → They already do optical flow internally on GPU!")
    print("  → Your NNAPI device targeting already enables GPU for them")
    
    if '--create-skeleton' in sys.argv:
        create_simple_optical_flow_tflite()
