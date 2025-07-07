#!/usr/bin/env python3
"""
V100 16GB Quantization Analysis for Llama-3.2-11B-Vision.

Shows memory requirements for different precision levels.
"""


def analyze_quantization_options():
    """Analyze memory requirements for different quantization options."""

    print("\n🔍 Llama-3.2-11B-Vision Quantization Options for V100 16GB\n")
    print("=" * 80)

    # Model parameters
    model_params_billion = 11.0
    v100_memory_gb = 16.0

    # Quantization options with bytes per parameter
    quantization_options = [
        {
            "name": "FP32 (Full Precision)",
            "bytes_per_param": 4,
            "description": "Original precision - highest quality",
            "pros": "Best quality, no loss",
            "cons": "44GB memory - won't fit on V100",
            "recommended": False,
        },
        {
            "name": "FP16 (Half Precision)",
            "bytes_per_param": 2,
            "description": "Standard GPU precision",
            "pros": "Good quality, minimal loss",
            "cons": "22GB memory - won't fit on V100",
            "recommended": False,
        },
        {
            "name": "INT8 (8-bit Quantization)",
            "bytes_per_param": 1,
            "description": "Balanced quality and memory",
            "pros": "Good quality, fits on V100 with headroom",
            "cons": "Slight quality degradation",
            "recommended": True,
        },
        {
            "name": "INT4 (4-bit Quantization)",
            "bytes_per_param": 0.5,
            "description": "Aggressive quantization",
            "pros": "Minimal memory usage, lots of headroom",
            "cons": "Noticeable quality degradation",
            "recommended": False,
        },
    ]

    print(
        f"Model: Llama-3.2-11B-Vision ({model_params_billion:.1f} billion parameters)"
    )
    print(f"Target: V100 GPU with {v100_memory_gb} GB memory")
    print(
        "\nNote: Vision components (vision_tower, multi_modal_projector) kept in FP16"
    )
    print("=" * 80)

    for option in quantization_options:
        # Calculate base model memory
        base_memory = model_params_billion * option["bytes_per_param"]

        # Add overhead estimates
        overhead = {
            "KV Cache": 2.0,  # Key-value cache for generation
            "Activations": 1.5,  # Intermediate activations
            "CUDA Overhead": 1.0,  # CUDA kernels, fragmentation
        }
        total_overhead = sum(overhead.values())
        total_memory = base_memory + total_overhead

        # Check if fits
        fits_v100 = total_memory < v100_memory_gb
        margin = v100_memory_gb - total_memory

        # Print analysis
        print(f"\n{'⭐ ' if option['recommended'] else ''}📊 {option['name']}")
        print(f"   Description: {option['description']}")
        print("   Memory calculation:")
        print(
            f"      - Model weights: {base_memory:.1f} GB ({model_params_billion}B × {option['bytes_per_param']} bytes)"
        )
        print(f"      - KV Cache: {overhead['KV Cache']:.1f} GB")
        print(f"      - Activations: {overhead['Activations']:.1f} GB")
        print(f"      - CUDA Overhead: {overhead['CUDA Overhead']:.1f} GB")
        print(f"      - Total: {total_memory:.1f} GB")

        if fits_v100:
            print(f"   ✅ Fits on V100 with {margin:.1f} GB safety margin")
        else:
            print(f"   ❌ Too large for V100 (exceeds by {-margin:.1f} GB)")

        print(f"   Pros: {option['pros']}")
        print(f"   Cons: {option['cons']}")

    print("\n" + "=" * 80)
    print("\n💡 Recommendations:")
    print("   1. **INT8 (8-bit)** - RECOMMENDED for V100 16GB")
    print("      - Best balance of quality and memory efficiency")
    print("      - ~11.5GB total usage leaves comfortable headroom")
    print("      - Minimal quality loss for OCR/receipt processing")
    print("")
    print("   2. **INT4 (4-bit)** - Only if INT8 doesn't fit")
    print("      - Use when processing very long sequences")
    print("      - Or when running multiple models")
    print("      - Quality may suffer for detailed text extraction")
    print("")
    print("   3. **Implementation in .env file:**")
    print("      USE_QUANTIZATION=true")
    print("      QUANTIZATION_TYPE=int8  # or int4 if needed")
    print("\n" + "=" * 80)


def show_bitsandbytes_config():
    """Show the actual BitsAndBytes configuration for each option."""
    print("\n📝 BitsAndBytes Configuration Examples:\n")

    print("INT8 Configuration (Recommended):")
    print("""```python
BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
    llm_int8_threshold=6.0,
)
```""")

    print("\nINT4 Configuration (Memory-constrained):")
    print("""```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
)
```""")


if __name__ == "__main__":
    analyze_quantization_options()
    show_bitsandbytes_config()
