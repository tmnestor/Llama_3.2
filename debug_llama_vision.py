#!/usr/bin/env python3
"""Debug script for Llama-3.2-Vision tensor issues."""

import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
from PIL import Image
import traceback

print("🔍 LLAMA VISION DEBUG SCRIPT")
print("=" * 50)

# Model path
model_path = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision"
test_image = "/home/jovyan/nfs_share/tod/data/examples/test_receipt.png"

print(f"Model path: {model_path}")
print(f"Test image: {test_image}")

# Test 1: Basic processor test
print("\n📋 Test 1: Processor initialization")
try:
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    print("✅ Processor loaded successfully")
    print(f"   Processor type: {type(processor)}")
    print(f"   Image processor: {type(processor.image_processor)}")
    print(f"   Tokenizer type: {type(processor.tokenizer)}")
except Exception as e:
    print(f"❌ Processor loading failed: {e}")
    traceback.print_exc()

# Test 2: Image processing
print("\n📋 Test 2: Image processing")
try:
    image = Image.open(test_image)
    print(f"✅ Image loaded: {image.size}, mode: {image.mode}")
    
    # Test different input formats
    print("\n   Testing input formats:")
    
    # Format 1: Simple text + image
    try:
        inputs1 = processor(
            text="What is in this image?",
            images=image,
            return_tensors="pt"
        )
        print("   ✅ Format 1 (text + image): Success")
        print(f"      Input keys: {list(inputs1.keys())}")
        if 'input_ids' in inputs1:
            print(f"      Input shape: {inputs1['input_ids'].shape}")
    except Exception as e:
        print(f"   ❌ Format 1 failed: {e}")
    
    # Format 2: Image as list
    try:
        inputs2 = processor(
            text="What is in this image?",
            images=[image],
            return_tensors="pt"
        )
        print("   ✅ Format 2 (text + [image]): Success")
    except Exception as e:
        print(f"   ❌ Format 2 failed: {e}")
    
    # Format 3: Using image processor directly
    try:
        image_inputs = processor.image_processor(images=image, return_tensors="pt")
        print("   ✅ Format 3 (image processor only): Success")
        print(f"      Keys: {list(image_inputs.keys())}")
    except Exception as e:
        print(f"   ❌ Format 3 failed: {e}")
        
except Exception as e:
    print(f"❌ Image processing failed: {e}")
    traceback.print_exc()

# Test 3: Check for image token in vocabulary
print("\n📋 Test 3: Tokenizer vocabulary check")
try:
    tokenizer = processor.tokenizer
    
    # Check for image token
    image_token = "<|image|>"
    if image_token in tokenizer.get_vocab():
        image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        print(f"✅ Image token found: '{image_token}' -> ID: {image_token_id}")
    else:
        print(f"❌ Image token '{image_token}' not in vocabulary")
        
    # Check special tokens
    print(f"   Special tokens: {tokenizer.special_tokens_map}")
    print(f"   Additional special tokens: {tokenizer.additional_special_tokens}")
    
    # Try encoding with image token
    try:
        test_text = f"{image_token}\nWhat is this?"
        encoded = tokenizer.encode(test_text, return_tensors="pt")
        print(f"✅ Can encode image token in text")
    except Exception as e:
        print(f"❌ Cannot encode image token: {e}")
        
except Exception as e:
    print(f"❌ Tokenizer check failed: {e}")

# Test 4: Model loading with minimal config
print("\n📋 Test 4: Minimal model loading")
try:
    # Try loading with minimal config
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    )
    print("✅ Model loaded with minimal config")
    print(f"   Model type: {type(model)}")
    print(f"   Has vision model: {hasattr(model, 'vision_model')}")
    
    # Test inference
    if 'inputs1' in locals():
        try:
            print("\n   Testing inference...")
            inputs = inputs1.to("cuda:0")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                )
            print("   ✅ Inference successful!")
        except Exception as e:
            print(f"   ❌ Inference failed: {e}")
            if "image token" in str(e):
                print("   💡 This is the image token count error")
                
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    traceback.print_exc()

# Test 5: Alternative inference approach
print("\n📋 Test 5: Alternative inference approaches")
if 'model' in locals() and 'processor' in locals():
    try:
        image = Image.open(test_image)
        
        # Approach 1: No image token in text
        print("   Approach 1: No image token in prompt")
        try:
            inputs = processor(
                text="Describe this image",  # No image token
                images=image,
                return_tensors="pt"
            ).to("cuda:0")
            
            output_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
            response = processor.decode(output_ids[0], skip_special_tokens=True)
            print(f"   ✅ Success! Response: {response[:50]}...")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            
        # Approach 2: Using the base generation
        print("\n   Approach 2: Base generation with explicit inputs")
        try:
            # Process image
            pixel_values = processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
            
            # Process text without image token
            text_inputs = processor.tokenizer(
                "What is in this receipt?",
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Generate
            outputs = model.generate(
                input_ids=text_inputs["input_ids"].to("cuda:0"),
                pixel_values=pixel_values.to("cuda:0"),
                attention_mask=text_inputs["attention_mask"].to("cuda:0"),
                max_new_tokens=20,
                do_sample=False
            )
            
            response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   ✅ Success! Response: {response[:50]}...")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            
    except Exception as e:
        print(f"❌ Alternative approaches failed: {e}")

print("\n📊 DEBUG SUMMARY")
print("=" * 50)
print("The image token error is likely due to:")
print("1. Mismatch between text tokens and image placeholders")
print("2. Processor expecting different format than model")
print("3. Version incompatibility in token handling")
print("\nRecommended fix: Use approach without image tokens in text")