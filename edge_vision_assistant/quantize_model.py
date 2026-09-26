#!/usr/bin/env python3
"""
quantize_model.py
-----------------
Performs dynamic INT8 quantization on the pre-trained ResNet-18 ONNX model.
Shrinks model size by ~75% and accelerates CPU inference via AVX2 / VNNI integer dot-products.
"""

import sys
from pathlib import Path

def quantize_resnet18():
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    input_model = models_dir / "resnet18-v1-7.onnx"
    output_model = models_dir / "resnet18-v1-7-int8.onnx"

    if not input_model.exists():
        print(f"[ERROR] Input model not found at {input_model}")
        print("Please run 'python3 download_assets.py' first.")
        sys.exit(1)

    print("=" * 65)
    print("⚡ INT8 Quantization Tool for Edge CPU Inference")
    print("=" * 65)
    print(f"Source FP32 Model  : {input_model}")
    print(f"Target INT8 Model  : {output_model}")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("[ERROR] onnxruntime is required for quantization.")
        print("Please install it: pip install onnx onnxruntime")
        sys.exit(1)

    print("\nExecuting Dynamic INT8 Quantization (weights FP32 -> INT8)...")
    quantize_dynamic(
        model_input=str(input_model),
        model_output=str(output_model),
        weight_type=QuantType.QUInt8
    )

    size_fp32_mb = input_model.stat().st_size / (1024.0 * 1024.0)
    size_int8_mb = output_model.stat().st_size / (1024.0 * 1024.0)
    savings_pct = ((size_fp32_mb - size_int8_mb) / size_fp32_mb) * 100.0

    print("\n✅ Quantization Complete!")
    print(f"  • FP32 Model Size : {size_fp32_mb:.2f} MB")
    print(f"  • INT8 Model Size : {size_int8_mb:.2f} MB")
    print(f"  • Memory Savings  : {savings_pct:.1f}% reduction ({size_fp32_mb - size_int8_mb:.2f} MB saved on disk & RAM)")
    print("=" * 65)

if __name__ == "__main__":
    quantize_resnet18()
