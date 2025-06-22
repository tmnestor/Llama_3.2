#!/usr/bin/env python3
"""
Debug script to test text-only generation without vision.
This isolates whether the issue is with vision processing or text generation.
"""

import logging
import time

import torch
from transformers import AutoTokenizer
from transformers import MllamaForConditionalGeneration


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_text_only_generation():
    """Test text-only generation to isolate the issue."""

    logger.info("🔬 Testing text-only generation...")

    try:
        # Load model and tokenizer
        logger.info("📥 Loading model...")
        model_path = "/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision"

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            device_map=None,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        logger.info("✅ Model loaded successfully")

        # Test simple text generation
        logger.info("🔄 Testing simple text generation...")
        start_time = time.time()

        prompt = "Hello world"
        inputs = tokenizer(prompt, return_tensors="pt")

        logger.info("🔄 Starting generation...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generation_time = time.time() - start_time
        logger.info(f"⏱️  Generation completed in {generation_time:.2f}s")

        # Decode response
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        logger.info(f"📄 Response: '{response}'")

        if response.strip():
            logger.info("✅ Text generation is working!")
        else:
            logger.info("⚠️  Empty response from text generation")

        return True

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 TESTING TEXT-ONLY GENERATION")
    print("=" * 60)
    print()

    success = test_text_only_generation()

    print()
    print("=" * 60)
    if success:
        print("✅ TEST COMPLETED")
    else:
        print("❌ TEST FAILED")
    print("=" * 60)
