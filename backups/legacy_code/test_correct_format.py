#!/usr/bin/env python3
"""
Test script using the EXACT format from Meta's documentation.
"""

import logging

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import MllamaForConditionalGeneration


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_exact_meta_format():
    """Test using the exact format from Meta documentation."""

    logger.info("🚀 Testing EXACT Meta documentation format...")

    try:
        # Load exactly as in documentation
        model_path = "/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision"

        logger.info("📥 Loading model and processor...")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            device_map=None,  # Load on CPU first
            torch_dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

        # Move to MPS
        if torch.backends.mps.is_available():
            model = model.to("mps")
            logger.info("✅ Model moved to MPS")

        # Load image
        image = Image.open("test_receipt.png")

        # EXACT format from documentation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What store name is on this receipt?"}
                ]
            }
        ]

        # Apply chat template
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        logger.info(f"📝 Chat template result: {input_text}")

        # Process inputs - EXACT as documentation
        inputs = processor(image, input_text, return_tensors="pt").to(model.device)

        logger.info(f"📊 Input keys: {list(inputs.keys())}")
        logger.info(f"📊 Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")

        # Generate - EXACT as documentation
        logger.info("🔄 Starting generation...")
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)

        # Decode
        result = processor.decode(output[0], skip_special_tokens=True)

        logger.info(f"📄 Full output: {result}")

        # Extract just the new tokens
        input_length = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_length:]
        response = processor.decode(new_tokens, skip_special_tokens=True)

        logger.info(f"📄 Response only: '{response}'")

        return response

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_simple_prompt():
    """Test with an even simpler prompt format."""

    logger.info("🔧 Testing simple prompt format...")

    try:
        model_path = "/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision"

        logger.info("📥 Loading model and processor...")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            device_map=None,  # Load on CPU first
            torch_dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

        # Move to MPS
        if torch.backends.mps.is_available():
            model = model.to("mps")
            logger.info("✅ Model moved to MPS")

        # Load image
        image = Image.open("test_receipt.png")

        # Simple prompt format
        prompt = "<|image|>What store is this?"

        logger.info(f"📝 Simple prompt: {prompt}")

        # Process
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

        logger.info(f"📊 Input keys: {list(inputs.keys())}")

        # Generate
        logger.info("🔄 Starting generation...")
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        # Decode full output
        result = processor.decode(output[0], skip_special_tokens=True)
        logger.info(f"📄 Full result: {result}")

        return result

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🧪 TESTING CORRECT LLAMA-3.2-VISION FORMAT")
    logger.info("=" * 60)

    # Test 1: Exact Meta format
    logger.info("\n📋 TEST 1: Exact Meta Documentation Format")
    result1 = test_exact_meta_format()

    # Test 2: Simple format
    logger.info("\n📋 TEST 2: Simple Prompt Format")
    result2 = test_simple_prompt()

    logger.info("\n" + "=" * 60)
    logger.info("🏁 RESULTS:")
    logger.info(f"Meta format: {result1}")
    logger.info(f"Simple format: {result2}")
    logger.info("=" * 60)
