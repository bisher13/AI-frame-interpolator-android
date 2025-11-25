#!/usr/bin/env python3
"""
Simplify ONNX model to remove unsupported operations and optimize for NNAPI.
This attempts to replace GridSample with more hardware-friendly operations.
"""

import onnx
import onnx.helper as helper
from onnx import numpy_helper
import argparse


def simplify_model(input_path, output_path):
    """
    Load ONNX model and attempt simplifications for NNAPI compatibility.
    
    Note: This is a basic template. Full GridSample replacement requires
    understanding your specific model architecture and may need manual work.
    """
    print(f"Loading model: {input_path}")
    model = onnx.load(input_path)
    
    # Try onnx-simplifier first
    try:
        import onnxsim
        print("Running onnx-simplifier...")
        model_simp, check = onnxsim.simplify(
            model,
            skip_fuse_bn=True,  # Keep batch norm separate for better NNAPI support
            skip_optimization=False,
        )
        if check:
            print("✓ Model simplified successfully")
            model = model_simp
        else:
            print("⚠ Simplification may have issues, check output carefully")
    except ImportError:
        print("⚠ onnx-simplifier not installed. Install with: pip install onnx-simplifier")
    except Exception as e:
        print(f"⚠ Simplification failed: {e}")
    
    # Check for problematic operations
    print("\nAnalyzing operations...")
    ops = set()
    for node in model.graph.node:
        ops.add(node.op_type)
    
    problematic_ops = {'GridSample', 'NonMaxSuppression', 'ScatterElements', 
                       'OneHot', 'TopK', 'RoiAlign'}
    found_problems = ops & problematic_ops
    
    if found_problems:
        print(f"⚠ Found NNAPI-incompatible operations: {found_problems}")
        print("\nTo fix GridSample:")
        print("1. Re-export model from PyTorch with torch.nn.functional.grid_sample")
        print("   replaced by torch.nn.functional.affine_grid + grid_sample decomposition")
        print("2. Or use a different interpolation method in the source model")
        print("\nCurrent model will fall back to CPU for these operations.")
    else:
        print("✓ No known problematic operations found")
    
    print(f"\nAll operations in model: {sorted(ops)}")
    
    # Check for dynamic shapes
    print("\nChecking input shapes...")
    for input_tensor in model.graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else -1 
                for dim in input_tensor.type.tensor_type.shape.dim]
        if -1 in shape:
            print(f"⚠ Dynamic shape in '{input_tensor.name}': {shape}")
            print("  Consider fixing input dimensions for better hardware support")
        else:
            print(f"✓ '{input_tensor.name}': {shape}")
    
    # Save the model
    print(f"\nSaving to: {output_path}")
    onnx.save(model, output_path)
    print("✓ Done!")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify ONNX model for NNAPI")
    parser.add_argument("--input", required=True, help="Input ONNX model")
    parser.add_argument("--output", required=True, help="Output ONNX model")
    
    args = parser.parse_args()
    simplify_model(args.input, args.output)
