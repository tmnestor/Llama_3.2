#!/usr/bin/env python3
"""
Test CPU-only to isolate the format issue from MPS memory problems.
"""

import logging

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import MllamaForConditionalGeneration


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_cpu_format():
    """Test using CPU to isolate format issues from MPS memory."""

    logger.info("🖥️ Testing on CPU to isolate format issue...")

    try:
        model_path = "/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision"

        logger.info("📥 Loading model and processor on CPU...")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            device_map=None,  # CPU only
            torch_dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

        # Stay on CPU
        logger.info("🖥️ Keeping model on CPU")

        # Load image
        image = Image.open("test_receipt.png")
        logger.info(f"📸 Image loaded: {image.size}")

        # Test EXACT documentation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What do you see?"}
                ]
            }
        ]

        # Apply chat template
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        logger.info(f"📝 Chat template length: {len(input_text)} chars")
        logger.info(f"📝 First 200 chars: {input_text[:200]}...")

        # Process inputs with CORRECTED format
        logger.info("🔄 Processing inputs with images= and text= format...")
        inputs = processor(images=image, text=input_text, return_tensors="pt")

        logger.info(f"📊 Input keys: {list(inputs.keys())}")
        logger.info(f"📊 Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")

        # Generate with minimal tokens to save time
        logger.info("🔄 Starting generation (CPU - will be slow)...")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,  # Very small for CPU test
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )

        # Decode just the new tokens
        input_length = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_length:]
        response = processor.decode(new_tokens, skip_special_tokens=True)

        logger.info(f"📄 Generated response: '{response}'")

        # Check for exclamation marks specifically
        if "!" in response and len(response.strip("! \n")) == 0:
            logger.error("❌ Still generating only exclamation marks!")
            return False
        else:
            logger.info("✅ Generated meaningful text!")
            return True

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🖥️ CPU-ONLY FORMAT TEST")
    logger.info("=" * 60)

    success = test_cpu_format()

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✅ CPU test successful - issue is likely MPS-specific")
    else:
        logger.info("❌ CPU test failed - issue is fundamental")
    logger.info("=" * 60)
