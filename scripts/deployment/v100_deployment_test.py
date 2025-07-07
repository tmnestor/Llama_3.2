#!/usr/bin/env python3
"""
V100 16GB Deployment Test for Llama-3.2-Vision.

This script tests different quantization strategies to fit the model
into V100 16GB memory constraints.
"""

import sys
from pathlib import Path

# Add package to path
sys.path.append(str(Path(__file__).parent))

from llama_vision.config import PromptManager, load_config
from llama_vision.model.v100_loader import V100ModelLoader, create_v100_optimized_loader


def test_v100_memory_estimates():
    """Test and display memory estimates for different quantization options."""
    print("\n🔍 V100 16GB Memory Analysis for Llama-3.2-11B-Vision\n")
    print("=" * 70)

    config = load_config()
    v100_loader = V100ModelLoader(config)

    # Get all quantization options
    options = v100_loader.get_quantization_options()

    for quant_type, info in options.items():
        estimate = v100_loader.estimate_memory_usage(quant_type)

        print(f"\n📊 {quant_type.upper()} Quantization")
        print(f"   Description: {info['description']}")
        print(f"   Base model memory: {estimate['base_model_memory_gb']:.1f} GB")
        print("   Overhead breakdown:")
        for overhead_type, overhead_gb in estimate["overhead_breakdown"].items():
            print(f"      - {overhead_type}: {overhead_gb:.1f} GB")
        print(f"   Total estimated: {estimate['total_estimated_gb']:.1f} GB")

        if estimate["fits_v100_16gb"]:
            print(
                f"   ✅ Fits in V100 16GB with {estimate['safety_margin_gb']:.1f} GB margin"
            )
        else:
            print(
                f"   ❌ Too large for V100 16GB (exceeds by {-estimate['safety_margin_gb']:.1f} GB)"
            )

    print("\n" + "=" * 70)
    print("\n💡 Recommendations for V100 16GB deployment:")
    print("   1. Use INT8 quantization for best quality-memory tradeoff")
    print("   2. Use INT4 quantization for maximum memory savings")
    print("   3. Consider mixed precision for critical layers")
    print("   4. Monitor actual GPU memory usage during inference")


def test_v100_loading(quantization_type: str = "int8"):
    """Test loading the model with specified quantization."""
    print(f"\n🚀 Testing V100 deployment with {quantization_type} quantization...\n")

    try:
        # Load configuration
        config = load_config()
        config.use_quantization = True
        config.quantization_type = quantization_type

        # Create V100 loader
        v100_loader = create_v100_optimized_loader(config)

        # Load model
        print("Loading model...")
        model, processor = v100_loader.load_for_v100(quantization_type)

        print("✅ Model loaded successfully!")

        # Test inference
        print("\nTesting inference with sample prompt...")
        prompt_manager = PromptManager()
        prompt = prompt_manager.get_prompt("factual_information_prompt")

        # Simple text generation test
        test_input = processor(
            text=prompt.replace("{image}", "a receipt"), return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(**test_input, max_new_tokens=50, do_sample=False)

        response = processor.decode(output[0], skip_special_tokens=True)
        print(f"Response preview: {response[:100]}...")

        # Log final memory usage
        v100_loader._log_gpu_memory_usage()

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def create_v100_env_file():
    """Create example .env file for V100 deployment."""
    env_content = """# V100 16GB Deployment Configuration
MODEL_PATH=/path/to/Llama-3.2-11B-Vision
DEVICE=cuda
USE_QUANTIZATION=true
QUANTIZATION_TYPE=int8  # Options: int8, int4, mixed_int8_int4

# Generation parameters optimized for V100
MAX_TOKENS=512  # Reduced for memory efficiency
TEMPERATURE=0.3
TOP_P=0.9
DO_SAMPLE=true

# Memory optimization
MAX_NEW_TOKENS=256  # Limit generation length
EXTRACTION_TOKENS=512  # Limit extraction context

# Logging
LOG_LEVEL=INFO

# Tax authority specific
TAX_INVOICE_NER_DEVICE=cuda
TAX_INVOICE_NER_USE_8BIT=true
TAX_INVOICE_NER_QUANTIZATION_TYPE=int8
TAX_INVOICE_NER_MAX_TOKENS=512
TAX_INVOICE_NER_EXTRACTION_MAX_TOKENS=512
"""

    env_file = Path(".env.v100")
    env_file.write_text(env_content)
    print(f"✅ Created {env_file} with V100-optimized settings")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="V100 16GB deployment test")
    parser.add_argument(
        "--test-loading",
        action="store_true",
        help="Test actual model loading (requires GPU)",
    )
    parser.add_argument(
        "--quantization",
        choices=["int8", "int4", "mixed_int8_int4"],
        default="int8",
        help="Quantization type to test",
    )
    parser.add_argument(
        "--create-env", action="store_true", help="Create example .env file for V100"
    )

    args = parser.parse_args()

    # Always show memory estimates
    test_v100_memory_estimates()

    # Create env file if requested
    if args.create_env:
        create_v100_env_file()

    # Test loading if requested
    if args.test_loading:
        # Only import torch if actually loading
        import torch

        if not torch.cuda.is_available():
            print("\n⚠️  No CUDA device available. Skipping loading test.")
            print("   Run this on a V100 machine to test actual loading.")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n🖥️  GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")

            if "V100" not in gpu_name and gpu_memory > 20:
                print("⚠️  Warning: Not running on V100. Results may differ.")

            success = test_v100_loading(args.quantization)
            sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
