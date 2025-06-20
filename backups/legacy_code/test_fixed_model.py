#!/usr/bin/env python3
"""Test model after config fix."""

import os
import traceback

import torch
from PIL import Image


# Set environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    """Test model loading with better error handling."""

    model_path = "/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision"

    try:
        print("1. Loading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"   Tokenizer loaded: {type(tokenizer)}")

        print("2. Loading model...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=None,
            torch_dtype=torch.float32,  # Use float32 to avoid precision issues
            trust_remote_code=True,
            local_files_only=True,
        )
        print(f"   Model loaded: {type(model)}")

        print("3. Loading test image...")
        image = Image.open("test_receipt.png")
        print(f"   Image: {image.size}")

        print("4. Testing answer_question method...")
        if hasattr(model, 'answer_question'):
            # Try with minimal parameters
            print("   Calling answer_question...")
            response = model.answer_question(
                image=image,
                question="What store?",
                tokenizer=tokenizer,
                max_new_tokens=5,
                do_sample=False
            )
            print(f"   SUCCESS! Response: '{response}'")
        else:
            print("   No answer_question method found")

    except Exception as e:
        print(f"ERROR: {e}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
