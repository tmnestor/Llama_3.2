#!/usr/bin/env python3
"""Find working quantization config for V100 16GB deployment."""

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
)

print("🎯 FINDING V100 16GB COMPATIBLE QUANTIZATION")
print("=" * 50)

model_path = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision"
test_image_path = "/home/jovyan/nfs_share/tod/data/examples/test_receipt.png"

# Test different quantization configurations
quantization_configs = [
    {
        "name": "8-bit basic",
        "config": BitsAndBytesConfig(load_in_8bit=True)
    },
    {
        "name": "8-bit with CPU offload",
        "config": BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    },
    {
        "name": "8-bit skip vision modules",
        "config": BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["vision_model", "multi_modal_projector"],
        )
    },
    {
        "name": "4-bit basic",
        "config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    },
    {
        "name": "4-bit with vision skip",
        "config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["vision_model", "multi_modal_projector"],
        )
    }
]

processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
test_image = Image.open(test_image_path)

successful_configs = []

for i, quant_config in enumerate(quantization_configs, 1):
    print(f"\n{i}. Testing: {quant_config['name']}")
    print("-" * 40)
    
    try:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load model with this config
        print("   Loading model...")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quant_config['config'],
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        print("   ✅ Model loaded successfully")
        
        # Check memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1e9
            print(f"   💾 GPU memory: {memory_used:.1f}GB")
            
            if memory_used <= 16:
                print("   ✅ Fits in V100 16GB!")
            else:
                print("   ❌ Exceeds V100 16GB limit")
        
        # Test inference
        print("   Testing inference...")
        inputs = processor(
            text="<|image|>What is in this image?",
            images=test_image,
            return_tensors="pt"
        )
        
        # Move to GPU
        inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print("   ✅ Inference successful!")
        print(f"   Response preview: {response[:50]}...")
        
        # Record successful config
        successful_configs.append({
            'name': quant_config['name'],
            'config': quant_config['config'],
            'memory_gb': memory_used if torch.cuda.is_available() else 0,
            'response': response[:100]
        })
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:100]}...")
        
        # Clean up on error
        try:
            pass
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("\n" + "="*50)
print("📊 RESULTS FOR V100 16GB DEPLOYMENT:")
print("="*50)

if successful_configs:
    print("✅ Working configurations found:")
    for i, config in enumerate(successful_configs, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   Memory: {config['memory_gb']:.1f}GB")
        print(f"   V100 compatible: {'✅ Yes' if config['memory_gb'] <= 16 else '❌ No'}")
        print(f"   Response: {config['response'][:50]}...")
        
    # Recommend best config
    v100_compatible = [c for c in successful_configs if c['memory_gb'] <= 16]
    if v100_compatible:
        best = min(v100_compatible, key=lambda x: x['memory_gb'])
        print("\n🏆 RECOMMENDED FOR V100 16GB:")
        print(f"   Configuration: {best['name']}")
        print(f"   Memory usage: {best['memory_gb']:.1f}GB")
        print(f"   Headroom: {16 - best['memory_gb']:.1f}GB")
else:
    print("❌ No working configurations found")
    print("\nOptions for V100 deployment:")
    print("1. Use a smaller vision model")
    print("2. CPU offload for vision components")
    print("3. Use text-only model with separate OCR")

print("\n💡 Next steps:")
print("1. Use the recommended config in your notebook")
print("2. Test thoroughly on L40S before V100 deployment")
print("3. Monitor memory usage during inference")