#!/usr/bin/env python3
"""
Debug script to examine the actual token IDs being generated.
This will help us understand WHY we're getting exclamation marks.
"""

import logging

import torch
from models.extractors.llama_vision_extractor import LlamaVisionExtractor


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def debug_token_generation():
    """Debug the actual token IDs being generated."""

    logger.info("🔍 DEBUGGING TOKEN IDS - Finding the root cause")

    try:
        # Load model
        extractor = LlamaVisionExtractor(
            model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            use_8bit=False,
            max_new_tokens=10,  # Small number for detailed analysis
            device="mps",
        )

        # Check what token ID corresponds to "!"
        exclamation_token = extractor.tokenizer.encode("!", add_special_tokens=False)
        logger.info(f"🔍 Exclamation mark '!' token ID: {exclamation_token}")

        # Check other special tokens
        logger.info(f"🔍 EOS token: '{extractor.tokenizer.eos_token}' = ID {extractor.tokenizer.eos_token_id}")
        logger.info(f"🔍 PAD token: '{extractor.tokenizer.pad_token}' = ID {extractor.tokenizer.pad_token_id}")
        logger.info(f"🔍 BOS token: '{extractor.tokenizer.bos_token}' = ID {extractor.tokenizer.bos_token_id}")

        # Manually create the generation process to examine each step
        image = extractor._preprocess_image("test_receipt.png")
        prompt = "What is the store name on this receipt?"

        # Step 1: Create messages
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]

        # Step 2: Apply chat template
        input_text = extractor.processor.apply_chat_template(messages, add_generation_prompt=True)
        logger.info(f"📝 Chat template: {repr(input_text)}")

        # Step 3: Process inputs
        inputs = extractor.processor(images=image, text=input_text, return_tensors="pt").to(extractor.device)

        logger.info(f"📊 Input token IDs: {inputs['input_ids']}")
        logger.info(f"📊 Input tokens decoded: {extractor.tokenizer.decode(inputs['input_ids'][0])}")

        # Step 4: Generate with detailed output
        logger.info("🔄 Starting generation with detailed logging...")

        with torch.no_grad():
            outputs = extractor.model.generate(
                **inputs,
                max_new_tokens=5,  # Very small for detailed analysis
                do_sample=False,
                pad_token_id=extractor.tokenizer.eos_token_id,
                eos_token_id=extractor.tokenizer.eos_token_id,
                return_dict_in_generate=True,  # Get detailed output
                output_scores=True,  # Get generation scores
            )

        # Analyze the generated tokens
        generated_ids = outputs.sequences[0]
        input_length = inputs["input_ids"].shape[1]
        new_token_ids = generated_ids[input_length:]

        logger.info(f"📊 Generated token IDs: {new_token_ids.tolist()}")

        # Decode each token individually
        for i, token_id in enumerate(new_token_ids):
            token_text = extractor.tokenizer.decode([token_id], skip_special_tokens=False)
            logger.info(f"📊 Token {i}: ID={token_id.item()} -> '{token_text}'")

        # Check if all tokens are the same
        unique_tokens = set(new_token_ids.tolist())
        if len(unique_tokens) == 1:
            repeated_token = list(unique_tokens)[0]
            logger.error(f"❌ PROBLEM FOUND: Model is stuck generating token ID {repeated_token} repeatedly!")

            # Check what this token should be
            token_text = extractor.tokenizer.decode([repeated_token], skip_special_tokens=False)
            logger.error(f"❌ Token ID {repeated_token} decodes to: '{token_text}'")

            # Check if this is a special token
            special_tokens = {
                "eos": extractor.tokenizer.eos_token_id,
                "pad": extractor.tokenizer.pad_token_id,
                "bos": extractor.tokenizer.bos_token_id,
            }

            for name, token_id in special_tokens.items():
                if token_id == repeated_token:
                    logger.error(f"❌ This is the {name.upper()} token! Generation is stuck on {name} token.")
                    return False

            logger.error("❌ This is not a standard special token - investigating further...")

            # Check the generation scores for this token
            if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
                first_score = outputs.scores[0][0]  # First generated token scores
                top_tokens = torch.topk(first_score, 5)
                logger.info("🔍 Top 5 token predictions for first generated token:")
                for score, token_id in zip(top_tokens.values, top_tokens.indices, strict=False):
                    token_text = extractor.tokenizer.decode([token_id], skip_special_tokens=False)
                    logger.info(f"  Token {token_id}: '{token_text}' (score: {score:.4f})")

        return False

    except Exception as e:
        logger.error(f"❌ Error during token debugging: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🔍 TOKEN ID DEBUGGING - FINDING THE EXCLAMATION MARK BUG")
    logger.info("=" * 80)

    debug_token_generation()

    logger.info("=" * 80)
