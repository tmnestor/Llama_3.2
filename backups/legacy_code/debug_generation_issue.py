#!/usr/bin/env python3
"""
Debug script to isolate the exclamation marks generation issue.
Tests various aspects of the model to find the root cause.
"""

import logging

import torch
from models.extractors.llama_vision_extractor import LlamaVisionExtractor


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_tokenizer_only():
    """Test tokenizer configuration and basic functionality."""
    logger.info("🔧 TESTING TOKENIZER ONLY...")

    try:
        extractor = LlamaVisionExtractor(
            model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            use_8bit=False,
            max_new_tokens=20,
            device="cpu",  # Keep on CPU for faster testing
        )

        # Test tokenizer directly
        test_text = "What is the store name on this receipt?"
        tokens = extractor.tokenizer.encode(test_text)
        decoded = extractor.tokenizer.decode(tokens)

        logger.info(f"Original text: {test_text}")
        logger.info(f"Tokenized: {tokens}")
        logger.info(f"Decoded: {decoded}")

        # Check special tokens
        logger.info(f"EOS token: {extractor.tokenizer.eos_token} (ID: {extractor.tokenizer.eos_token_id})")
        logger.info(f"PAD token: {extractor.tokenizer.pad_token} (ID: {extractor.tokenizer.pad_token_id})")
        logger.info(f"BOS token: {extractor.tokenizer.bos_token} (ID: {extractor.tokenizer.bos_token_id})")

    except Exception as e:
        logger.error(f"Tokenizer test failed: {e}")
        return False

    return True

def test_text_only_generation():
    """Test text-only generation without images."""
    logger.info("📝 TESTING TEXT-ONLY GENERATION...")

    try:
        extractor = LlamaVisionExtractor(
            model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            use_8bit=False,
            max_new_tokens=20,
            device="cpu",  # Keep on CPU for faster testing
        )

        # Try direct text generation
        prompt = "The capital of France is"
        tokens = extractor.tokenizer.encode(prompt, return_tensors="pt")

        logger.info(f"Input prompt: {prompt}")
        logger.info(f"Input tokens shape: {tokens.shape}")

        with torch.no_grad():
            outputs = extractor.model.generate(
                tokens,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
                pad_token_id=extractor.tokenizer.eos_token_id,
                eos_token_id=extractor.tokenizer.eos_token_id,
            )

        # Decode response
        response = extractor.tokenizer.decode(outputs[0][tokens.shape[-1]:], skip_special_tokens=True)
        logger.info(f"Generated response: '{response}'")

        return "!" not in response or len(response.strip("!")) > 0

    except Exception as e:
        logger.error(f"Text-only generation failed: {e}")
        return False

def test_generation_parameters():
    """Test different generation parameters."""
    logger.info("⚙️ TESTING GENERATION PARAMETERS...")

    try:
        extractor = LlamaVisionExtractor(
            model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            use_8bit=False,
            max_new_tokens=20,
            device="cpu",
        )

        prompt = "Hello, my name is"
        tokens = extractor.tokenizer.encode(prompt, return_tensors="pt")

        # Test different parameter combinations
        configs = [
            {"do_sample": False, "temperature": 1.0, "name": "Deterministic"},
            {"do_sample": True, "temperature": 0.7, "name": "Low Temperature"},
            {"do_sample": True, "temperature": 1.0, "name": "Normal Temperature"},
            {"do_sample": False, "num_beams": 3, "name": "Beam Search"},
        ]

        for config in configs:
            config_name = config.pop("name")
            logger.info(f"Testing {config_name}: {config}")

            with torch.no_grad():
                outputs = extractor.model.generate(
                    tokens,
                    max_new_tokens=5,
                    pad_token_id=extractor.tokenizer.eos_token_id,
                    eos_token_id=extractor.tokenizer.eos_token_id,
                    **config
                )

            response = extractor.tokenizer.decode(outputs[0][tokens.shape[-1]:], skip_special_tokens=True)
            logger.info(f"{config_name} result: '{response}'")

    except Exception as e:
        logger.error(f"Parameter testing failed: {e}")
        return False

    return True

def test_chat_template():
    """Test the chat template formatting."""
    logger.info("💬 TESTING CHAT TEMPLATE...")

    try:
        extractor = LlamaVisionExtractor(
            model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            use_8bit=False,
            max_new_tokens=20,
            device="cpu",
        )

        # Test chat template
        messages = [{"role": "user", "content": "What is 2+2?"}]

        # Apply chat template
        chat_text = extractor.processor.apply_chat_template(messages, add_generation_prompt=True)
        logger.info(f"Chat template result: '{chat_text}'")

        # Test tokenization of chat template
        tokens = extractor.tokenizer.encode(chat_text, return_tensors="pt")
        decoded = extractor.tokenizer.decode(tokens[0])
        logger.info(f"Chat template tokens: {tokens.shape}")
        logger.info(f"Decoded: '{decoded}'")

    except Exception as e:
        logger.error(f"Chat template test failed: {e}")
        return False

    return True

def main():
    """Run all debug tests."""
    logger.info("🚀 STARTING DEBUG TESTS FOR EXCLAMATION MARKS ISSUE")
    logger.info("=" * 60)

    results = {}

    # Run tests
    results["tokenizer"] = test_tokenizer_only()
    results["text_generation"] = test_text_only_generation()
    results["parameters"] = test_generation_parameters()
    results["chat_template"] = test_chat_template()

    # Summary
    logger.info("=" * 60)
    logger.info("🏁 DEBUG TEST RESULTS:")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    if all(results.values()):
        logger.info("🎉 All basic tests passed - issue may be vision-specific")
    else:
        logger.info("🔧 Found issues in basic functionality - needs investigation")

if __name__ == "__main__":
    main()
