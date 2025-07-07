#!/usr/bin/env python3
"""Test different approaches to fix the tensor reshape error."""

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

print("🔧 TESTING TENSOR RESHAPE FIXES")
print("=" * 50)

model_path = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision"
test_image_path = "/home/jovyan/nfs_share/tod/data/examples/test_receipt.png"

# Load model and processor
print("\n1. Loading model with different configurations...")

# Try 1: Load without 8-bit quantization
print("\nTrying without quantization...")
try:
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    print("✅ Model loaded without quantization")
    
    # Test inference
    image = Image.open(test_image_path)
    inputs = processor(
        text="<|image|>What is in this image?",
        images=image,
        return_tensors="pt"
    )
    
    # Move to GPU
    inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response[:100]}...")
    print("✅ Inference successful without quantization!")
    
except Exception as e:
    print(f"❌ Failed: {str(e)}")

# Try 2: Different image processing
print("\n\n2. Testing different image formats...")
try:
    # Test with different image sizes
    test_sizes = [(224, 224), (336, 336), (448, 448)]
    image = Image.open(test_image_path)
    
    for size in test_sizes:
        print(f"\nTesting size {size}...")
        try:
            # Resize image
            resized_image = image.resize(size, Image.Resampling.LANCZOS)
            
            inputs = processor(
                text="<|image|>What is this?",
                images=resized_image,
                return_tensors="pt"
            )
            
            print(f"  Input shapes: {inputs['input_ids'].shape}, pixel_values: {inputs.get('pixel_values', 'N/A')}")
            
            # Try generation with minimal config
            inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    pixel_values=inputs.get('pixel_values'),
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=20,
                )
            
            print(f"  ✅ Size {size} works!")
            
        except Exception as e:
            print(f"  ❌ Size {size} failed: {str(e)[:50]}...")

except Exception as e:
    print(f"❌ Image format testing failed: {e}")

# Try 3: Manual tensor handling
print("\n\n3. Testing manual tensor processing...")
try:
    # Process image and text separately
    image = Image.open(test_image_path)
    
    # Get image features
    image_inputs = processor.image_processor(images=image, return_tensors="pt")
    print(f"Image inputs keys: {list(image_inputs.keys())}")
    
    # Get text tokens
    text = "<|image|>What is in this image?"
    text_inputs = processor.tokenizer(text, return_tensors="pt")
    print(f"Text inputs keys: {list(text_inputs.keys())}")
    
    # Check shapes
    print(f"Pixel values shape: {image_inputs['pixel_values'].shape}")
    print(f"Input IDs shape: {text_inputs['input_ids'].shape}")
    
    # Look for the issue
    if 'aspect_ratio_ids' in image_inputs:
        print(f"Aspect ratio IDs shape: {image_inputs['aspect_ratio_ids'].shape}")
    if 'aspect_ratio_mask' in image_inputs:
        print(f"Aspect ratio mask shape: {image_inputs['aspect_ratio_mask'].shape}")
    
    # Try different generation approaches
    print("\nTrying generation with explicit kwargs...")
    
    # Move to GPU
    pixel_values = image_inputs['pixel_values'].to("cuda:0")
    input_ids = text_inputs['input_ids'].to("cuda:0")
    attention_mask = text_inputs['attention_mask'].to("cuda:0")
    
    # Try with only essential inputs
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=20,
            )
        print("✅ Basic generation works!")
    except Exception as e:
        print(f"❌ Basic generation failed: {str(e)[:100]}...")
        
        # Try with attention mask
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    max_new_tokens=20,
                )
            print("✅ Generation with attention mask works!")
        except Exception as e2:
            print(f"❌ With attention mask failed: {str(e2)[:100]}...")

except Exception as e:
    print(f"❌ Manual processing failed: {e}")

print("\n" + "="*50)
print("RECOMMENDATIONS:")
print("1. The tensor reshape error may be due to 8-bit quantization")
print("2. Try loading the model without quantization if you have enough memory")
print("3. Or use a different quantization config that preserves tensor shapes")
print("4. The issue might be in the vision model's forward pass with quantized weights")