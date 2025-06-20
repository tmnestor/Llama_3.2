#!/usr/bin/env python3
"""Test different model interfaces to find working one."""

import os

import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


# Set environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    """Test model interfaces."""

    model_path = "/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=None,
        torch_dtype=torch.float32,  # Use float32 for stability
        trust_remote_code=True,
        local_files_only=True,
    )

    # Don't move to MPS yet - test on CPU first
    print(f"Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")

    # Load a simple image
    image = Image.open("test_receipt.png")
    print(f"Image loaded: {image.size}")

    # Test different interfaces step by step
    print("\n=== Testing Interfaces ===")

    # 1. Check if answer_question exists and what it expects
    if hasattr(model, 'answer_question'):
        print("1. Found answer_question method")
        import inspect
        sig = inspect.signature(model.answer_question)
        print(f"   Signature: {sig}")

        # Try calling with minimal parameters - no temperature
        try:
            print("   Attempting simple call...")
            response = model.answer_question(
                image=image,
                question="What store is this?",
                tokenizer=tokenizer,
                max_new_tokens=10,
                do_sample=False
            )
            print(f"   Success! Response: {response}")
            return
        except Exception as e:
            print(f"   Failed: {e}")

    # 2. Check if there's a generate method that takes images
    if hasattr(model, 'generate') and hasattr(model, 'vision_model'):
        print("2. Found vision model, trying generate...")
        try:
            # Process image to model format
            inputs = tokenizer("What store is this?", return_tensors="pt")

            # Try calling generate (this might not work but let's see)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   Success! Response: {response}")
            return

        except Exception as e:
            print(f"   Failed: {e}")

    # 3. Check what methods are available
    print("3. Available methods:")
    methods = [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))]
    for method in sorted(methods)[:20]:  # Show first 20
        print(f"   - {method}")

    print("\nModel inspection complete")

if __name__ == "__main__":
    main()
