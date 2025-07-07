#!/usr/bin/env python3
"""Test CPU offloading strategies for V100 16GB."""

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
)

print("🔄 TESTING CPU OFFLOADING FOR V100 16GB")
print("=" * 50)

model_path = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision"
test_image_path = "/home/jovyan/nfs_share/tod/data/examples/test_receipt.png"

processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
test_image = Image.open(test_image_path)

# Strategy 1: Vision on GPU, Language on CPU/GPU hybrid
print("\n1. Testing: Vision on GPU, Language quantized")
print("-" * 45)

try:
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Custom device map: keep vision on GPU, allow language to be quantized
    device_map = {
        # Vision components stay on GPU in FP16
        "vision_model": "cuda:0",
        "multi_modal_projector": "cuda:0",
        # Language model can be quantized and distributed
        "language_model": "auto",
    }

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
        # Skip vision modules from quantization
        llm_int8_skip_modules=["vision_model", "multi_modal_projector"],
    )

    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    print("   ✅ Model loaded successfully")

    # Check memory
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated(0) / 1e9
        print(f"   💾 GPU memory: {memory_used:.1f}GB")

    # Test inference
    print("   Testing inference...")
    inputs = processor(
        text="<|image|>What is in this image?", images=test_image, return_tensors="pt"
    )

    # Move inputs to appropriate devices
    inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )

    response = processor.decode(outputs[0], skip_special_tokens=True)
    print("   ✅ Inference successful!")
    print(f"   Response: {response[:100]}...")

    del model
    torch.cuda.empty_cache()

except Exception as e:
    print(f"   ❌ Failed: {str(e)[:150]}...")
    try:
        pass
    except Exception:
        pass
    torch.cuda.empty_cache()

# Strategy 2: Sequential processing - load vision, process, unload, load language
print("\n\n2. Testing: Sequential processing (vision → language)")
print("-" * 50)

try:
    print("   Loading vision model only...")

    # Load with manual device assignment
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        device_map={
            "vision_model": "cuda:0",
            "multi_modal_projector": "cuda:0",
            "language_model": "cpu",  # Keep language on CPU initially
        },
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    print("   ✅ Model loaded with vision on GPU, language on CPU")

    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated(0) / 1e9
        print(f"   💾 GPU memory (vision only): {memory_used:.1f}GB")

    # Test if we can move language model back to GPU for inference
    print("   Testing inference with CPU offloading...")

    inputs = processor(
        text="<|image|>What is in this image?", images=test_image, return_tensors="pt"
    )

    # Move vision inputs to GPU
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )

    response = processor.decode(outputs[0], skip_special_tokens=True)
    print("   ✅ Sequential processing works!")
    print(f"   Response: {response[:100]}...")

    del model
    torch.cuda.empty_cache()

except Exception as e:
    print(f"   ❌ Failed: {str(e)[:150]}...")
    try:
        pass
    except Exception:
        pass
    torch.cuda.empty_cache()

# Strategy 3: 4-bit quantization with vision skip
print("\n\n3. Testing: 4-bit quantization (vision modules skipped)")
print("-" * 55)

try:
    torch.cuda.empty_cache()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # Don't quantize vision components
        llm_int8_skip_modules=["vision_model", "multi_modal_projector"],
    )

    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    print("   ✅ 4-bit model loaded successfully")

    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated(0) / 1e9
        print(f"   💾 GPU memory: {memory_used:.1f}GB")

    # Test inference
    inputs = processor(
        text="<|image|>What is in this image?", images=test_image, return_tensors="pt"
    )

    inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )

    response = processor.decode(outputs[0], skip_special_tokens=True)
    print("   ✅ 4-bit inference successful!")
    print(f"   Response: {response[:100]}...")

    del model
    torch.cuda.empty_cache()

except Exception as e:
    print(f"   ❌ Failed: {str(e)[:150]}...")
    try:
        pass
    except Exception:
        pass
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("💡 CPU OFFLOADING ANALYSIS:")
print("=" * 60)
print("CPU offloading can help by:")
print("1. Keeping vision components in FP16 (no quantization issues)")
print("2. Quantizing only the language model (safer)")
print("3. Using 4-bit instead of 8-bit (better memory efficiency)")
print("4. Dynamic loading/unloading during inference")
print("\nIf any strategy above works, it can be used for V100 16GB!")
