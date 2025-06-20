#!/usr/bin/env python3
"""
Minimal test to verify MPS model loading and generation
"""

import torch
from models.extractors.llama_vision_extractor import LlamaVisionExtractor
from PIL import Image


def test_model():
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    if torch.backends.mps.is_available():
        print("✅ MPS acceleration is available")
    else:
        print("❌ MPS acceleration not available")

    print("\n🔄 Loading model...")
    extractor = LlamaVisionExtractor(
        model_path="/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision",
        device="auto",
        max_new_tokens=50  # Very short for testing
    )

    print(f"✅ Model loaded on device: {extractor.device}")
    print(f"   Model device: {extractor.model.device}")

    # Test with a simple prompt first (no image)
    print("\n🔄 Testing text-only generation...")
    try:
        test_prompt = "Hello, how are you?"
        response = extractor._generate_response(test_prompt, None)
        print(f"✅ Text response: {response[:100]}...")
    except Exception as e:
        print(f"❌ Text generation failed: {e}")

    # Test image loading
    print("\n🔄 Testing image loading...")
    try:
        image = Image.open("/Users/tod/Desktop/Llama_3.2/test_receipt.png")
        print(f"✅ Image loaded: {image.size}")

        # Test image encoding only
        if hasattr(extractor.model, 'encode_image'):
            print("🔄 Testing image encoding...")
            image_embeds = extractor.model.encode_image(image)
            print(f"✅ Image encoded: {image_embeds.shape}")

    except Exception as e:
        print(f"❌ Image processing failed: {e}")

if __name__ == "__main__":
    test_model()
