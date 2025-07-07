#!/usr/bin/env python3
"""Check if Llama-3.2-Vision model has vision capabilities properly loaded."""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

print("🔍 CHECKING LLAMA-3.2-VISION MODEL COMPONENTS")
print("=" * 50)

model_path = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision"

# Load model and processor
print("\n1. Loading model...")
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True,
)

print(f"✅ Model loaded: {type(model)}")

# Check model components
print("\n2. Checking model components:")
print(f"   Has vision_model: {hasattr(model, 'vision_model')}")
print(f"   Has language_model: {hasattr(model, 'language_model')}")
print(f"   Has multi_modal_projector: {hasattr(model, 'multi_modal_projector')}")

if hasattr(model, "vision_model"):
    print(f"   Vision model type: {type(model.vision_model)}")
    print(f"   Vision model device: {next(model.vision_model.parameters()).device}")

# Check processor components
print("\n3. Checking processor components:")
print(f"   Has image_processor: {hasattr(processor, 'image_processor')}")
print(f"   Has tokenizer: {hasattr(processor, 'tokenizer')}")
print(f"   Image processor type: {type(processor.image_processor)}")

# Check config
print("\n4. Checking model config:")
config = model.config
print(f"   Model type: {config.model_type}")
print(f"   Vision config exists: {hasattr(config, 'vision_config')}")
if hasattr(config, "vision_config"):
    print(f"   Vision hidden size: {config.vision_config.hidden_size}")
    print(f"   Vision patch size: {config.vision_config.patch_size}")

# Test vision processing
print("\n5. Testing vision processing:")
try:
    # Create a dummy image
    dummy_image = Image.new("RGB", (224, 224), color="red")

    # Process image
    image_inputs = processor.image_processor(images=dummy_image, return_tensors="pt")
    print("   ✅ Image processing successful")
    print(f"   Image tensor shape: {image_inputs['pixel_values'].shape}")

    # Check if vision model can process it
    if hasattr(model, "vision_model"):
        with torch.no_grad():
            # Get vision features
            vision_outputs = model.vision_model(
                pixel_values=image_inputs["pixel_values"].to(model.device),
                aspect_ratio_ids=image_inputs.get("aspect_ratio_ids", None),
                aspect_ratio_mask=image_inputs.get("aspect_ratio_mask", None),
            )
            print("   ✅ Vision model forward pass successful")
            print(f"   Vision output shape: {vision_outputs[0].shape}")
except Exception as e:
    print(f"   ❌ Vision processing failed: {e}")

# Check for vision-related parameters
print("\n6. Checking vision parameters:")
total_params = sum(p.numel() for p in model.parameters())
if hasattr(model, "vision_model"):
    vision_params = sum(p.numel() for p in model.vision_model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Vision parameters: {vision_params:,}")
    print(f"   Vision params %: {vision_params / total_params * 100:.1f}%")

# Test with actual prompt
print("\n7. Testing with vision-aware prompt:")
try:
    test_prompt = "USER: <image>\nWhat do you see in this image?\nASSISTANT:"

    # Tokenize
    inputs = processor.tokenizer(test_prompt, return_tensors="pt")
    print(f"   Tokenized prompt length: {inputs['input_ids'].shape[1]}")

    # Check for image token
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    print(f"   Image token ID: {image_token_id}")
    print(
        f"   Image token in input: {image_token_id in inputs['input_ids'][0].tolist()}"
    )

except Exception as e:
    print(f"   ❌ Prompt test failed: {e}")

print("\n" + "=" * 50)
print("DIAGNOSIS:")

# Diagnosis
if hasattr(model, "vision_model") and hasattr(model, "multi_modal_projector"):
    print("✅ Model has vision components loaded")
else:
    print("❌ Vision components missing!")

if hasattr(processor, "image_processor"):
    print("✅ Processor can handle images")
else:
    print("❌ Image processor missing!")

# Check model files
print("\n8. Checking model files:")
model_path_obj = Path(model_path)
model_files = list(model_path_obj.iterdir())
print(f"   Files in model directory: {len(model_files)}")
for f in sorted(model_files):
    if any(keyword in f.name.lower() for keyword in ["vision", "image", "multimodal"]):
        print(f"   - {f.name}")

print("\nIf vision components are missing, the model may need to be re-downloaded.")
print("The model should have both vision and language components for multimodal tasks.")
