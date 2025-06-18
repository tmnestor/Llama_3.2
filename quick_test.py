#!/usr/bin/env python3
"""
Quick test for receipt extraction with minimal model interaction
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def quick_test():
    print(f"🔄 PyTorch: {torch.__version__}")
    print(f"🔄 MPS available: {torch.backends.mps.is_available()}")
    
    model_path = "/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision"
    
    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("🔄 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=None,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    if torch.backends.mps.is_available():
        model = model.to("mps")
        print("✅ Model moved to MPS")
    else:
        print("✅ Model on CPU")
    
    # Load image
    image = Image.open("/Users/tod/Desktop/Llama_3.2/test_receipt.png")
    print(f"✅ Image loaded: {image.size}")
    
    # Test answer_question if available
    if hasattr(model, 'answer_question'):
        print("🔄 Testing answer_question...")
        try:
            response = model.answer_question(
                image=image,
                question="What store is this?",
                tokenizer=tokenizer,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.1,
            )
            print(f"✅ SUCCESS: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("❌ No answer_question method found")

if __name__ == "__main__":
    quick_test()