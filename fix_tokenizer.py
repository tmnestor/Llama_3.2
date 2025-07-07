#!/usr/bin/env python3
"""Fix tokenizer to properly handle image tokens."""


from PIL import Image
from transformers import AutoProcessor

print("🔧 FIXING LLAMA-3.2-VISION TOKENIZER")
print("=" * 50)

model_path = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision"

# Load processor
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
tokenizer = processor.tokenizer

print("\n1. Current tokenizer configuration:")
print(f"   Vocab size: {len(tokenizer)}")
print(f"   Special tokens: {tokenizer.special_tokens_map}")
print(f"   Additional special tokens: {tokenizer.additional_special_tokens}")

# Check various image token formats
image_tokens = ["<image>", "<|image|>", "[IMG]", "<IMG>", "IMAGE", "<|vision_start|>", "<|vision_end|>"]
print("\n2. Checking for image tokens:")
for token in image_tokens:
    if token in tokenizer.get_vocab():
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"   ✅ '{token}' -> ID: {token_id}")
    else:
        print(f"   ❌ '{token}' not found")

# Check the actual special tokens
print("\n3. Looking for special vision tokens:")
vocab = tokenizer.get_vocab()
vision_tokens = {k: v for k, v in vocab.items() if any(word in k.lower() for word in ['image', 'img', 'vision', 'pixel'])}
if vision_tokens:
    print(f"   Found {len(vision_tokens)} vision-related tokens:")
    for token, token_id in list(vision_tokens.items())[:10]:
        print(f"   - '{token}' -> ID: {token_id}")
else:
    print("   No vision-related tokens found")

# Test the processor's default formatting
print("\n4. Testing processor's default image handling:")
test_image = Image.new('RGB', (224, 224), color='red')

# Try different text formats
test_prompts = [
    "What is in this image?",
    "<|image|>What is in this image?",
    "USER: What is in this image?\nASSISTANT:",
    "<|begin_of_text|>What is in this image?",
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n   Test {i}: {prompt[:30]}...")
    try:
        inputs = processor(
            text=prompt,
            images=test_image,
            return_tensors="pt"
        )
        print(f"   ✅ Success - Input IDs shape: {inputs['input_ids'].shape}")
        
        # Decode to see what was tokenized
        decoded = tokenizer.decode(inputs['input_ids'][0][:20])  # First 20 tokens
        print(f"   Decoded start: {decoded[:50]}...")
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:100]}")

# Check if processor adds image tokens automatically
print("\n5. Checking if processor adds image tokens automatically:")
try:
    # Process with just text
    text_only = processor(text="Hello world", return_tensors="pt")
    text_ids = text_only['input_ids'][0].tolist()
    
    # Process with text and image
    text_image = processor(text="Hello world", images=test_image, return_tensors="pt")
    text_image_ids = text_image['input_ids'][0].tolist()
    
    print(f"   Text only length: {len(text_ids)}")
    print(f"   Text+image length: {len(text_image_ids)}")
    
    if len(text_image_ids) > len(text_ids):
        print(f"   ✅ Processor adds {len(text_image_ids) - len(text_ids)} tokens for image")
        # Find the different tokens
        for i, (t1, t2) in enumerate(zip(text_ids, text_image_ids, strict=False)):
            if t1 != t2:
                print(f"   First difference at position {i}: {t1} -> {t2}")
                print(f"   Token: '{tokenizer.decode([t2])}'")
                break
    else:
        print("   ❌ No additional tokens added for image")
        
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*50)
print("RECOMMENDATION:")
print("The processor should handle image tokens automatically.")
print("Don't manually add <image> tokens to your prompts.")
print("Let the processor format the inputs properly.")