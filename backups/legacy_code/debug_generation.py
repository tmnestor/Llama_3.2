#!/usr/bin/env python3
"""Debug generation timeout issue."""

import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


# Set environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def test_model_loading():
    """Test model loading step by step."""
    print("=== Model Loading Debug ===")

    model_path = "/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision"

    print(f"1. Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"   Tokenizer loaded: {type(tokenizer)}")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Pad token: {tokenizer.pad_token}")

    print("2. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=None,
        torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    )
    print(f"   Model loaded: {type(model)}")
    print(f"   Model device: {next(model.parameters()).device}")

    # Check model methods
    print("3. Checking model interface:")
    print(f"   Has answer_question: {hasattr(model, 'answer_question')}")
    print(f"   Has encode_image: {hasattr(model, 'encode_image')}")
    print(f"   Has chat: {hasattr(model, 'chat')}")

    # Move to MPS if available
    if torch.backends.mps.is_available():
        print("4. Moving model to MPS...")
        model = model.to("mps")
        print(f"   Model device after move: {next(model.parameters()).device}")

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   Set pad token to: {tokenizer.pad_token}")

    return model, tokenizer

def test_image_loading():
    """Test image loading."""
    print("\n=== Image Loading Debug ===")

    image_path = "test_receipt.png"
    print(f"1. Loading image from {image_path}")

    if not Path(image_path).exists():
        print("   ERROR: Image file not found!")
        return None

    image = Image.open(image_path)
    print(f"   Image loaded: {image.size} {image.mode}")
    return image

def test_generation_minimal(model, tokenizer, image):
    """Test minimal generation with timeout."""
    print("\n=== Generation Debug ===")

    # Simple prompt
    prompt = "What store is this?"
    print(f"1. Testing with prompt: '{prompt}'")

    # Test with timeout
    try:
        print("2. Calling answer_question with timeout...")

        # Set a timeout using signal (Unix only)
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Generation timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout

        start_time = time.time()

        response = model.answer_question(
            image=image,
            question=prompt,
            tokenizer=tokenizer,
            max_new_tokens=20,  # Very short
            do_sample=False,
            temperature=0.1,
        )

        signal.alarm(0)  # Cancel timeout

        elapsed = time.time() - start_time
        print(f"3. Generation completed in {elapsed:.2f}s")
        print(f"   Response: '{response}'")

        return response

    except TimeoutError:
        print("   ERROR: Generation timed out after 30s")
        return None
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        print(f"   ERROR: Generation failed: {e}")
        return None

def main():
    """Main debug function."""
    print("Starting Llama-Vision generation debug")

    # Test model loading
    try:
        model, tokenizer = test_model_loading()
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    # Test image loading
    image = test_image_loading()
    if image is None:
        return

    # Test generation
    test_generation_minimal(model, tokenizer, image)

    print("\nDebug complete")

if __name__ == "__main__":
    main()
