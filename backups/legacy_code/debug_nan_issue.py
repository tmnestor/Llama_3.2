#!/usr/bin/env python3
"""
Debug script to check for NaN values in model outputs and test MPS vs CPU.
"""

import logging

import torch
from models.extractors.llama_vision_extractor import LlamaVisionExtractor


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def check_for_nans_in_model():
    """Check if the model produces NaN values in its forward pass."""

    logger.info("🔍 CHECKING FOR NaN VALUES IN MODEL COMPUTATION")

    try:
        # Test 1: Load model on CPU first
        logger.info("📱 Test 1: Loading model on CPU...")
        extractor_cpu = LlamaVisionExtractor(
            model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            use_8bit=False,
            max_new_tokens=3,
            device="cpu",  # Force CPU
        )

        # Test basic text generation on CPU
        logger.info("🔤 Testing text-only generation on CPU...")
        test_input = "The capital of France is"
        tokens = extractor_cpu.tokenizer.encode(test_input, return_tensors="pt")

        with torch.no_grad():
            cpu_output = extractor_cpu.model.generate(
                tokens,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=extractor_cpu.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Check CPU scores for NaN
        cpu_scores = cpu_output.scores[0][0]  # First token scores
        cpu_has_nan = torch.isnan(cpu_scores).any()
        logger.info(f"📊 CPU text generation - NaN in scores: {cpu_has_nan}")
        if not cpu_has_nan:
            top_cpu_token = torch.argmax(cpu_scores).item()
            cpu_text = extractor_cpu.tokenizer.decode([top_cpu_token])
            logger.info(f"📊 CPU top token: {top_cpu_token} -> '{cpu_text}'")

        # Test 2: Try vision on CPU (if it doesn't crash)
        logger.info("👁️ Testing vision processing on CPU...")
        image = extractor_cpu._preprocess_image("test_receipt.png")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What store?"}]}]
        input_text = extractor_cpu.processor.apply_chat_template(messages, add_generation_prompt=True)

        vision_inputs = extractor_cpu.processor(images=image, text=input_text, return_tensors="pt")

        logger.info("🔄 Running forward pass to check for NaN...")
        with torch.no_grad():
            # Just do a forward pass to check for NaN
            cpu_vision_output = extractor_cpu.model.generate(
                **vision_inputs,
                max_new_tokens=2,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Check vision scores for NaN on CPU
        cpu_vision_scores = cpu_vision_output.scores[0][0]
        cpu_vision_has_nan = torch.isnan(cpu_vision_scores).any()
        logger.info(f"📊 CPU vision generation - NaN in scores: {cpu_vision_has_nan}")

        if cpu_vision_has_nan:
            logger.error("❌ CPU vision processing also produces NaN - MODEL ISSUE!")
            return False
        else:
            logger.info("✅ CPU vision processing works without NaN")

        # Test 3: Compare with MPS
        logger.info("📱 Test 3: Loading model on MPS...")
        extractor_mps = LlamaVisionExtractor(
            model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            use_8bit=False,
            max_new_tokens=3,
            device="mps",
        )

        # Test vision on MPS
        logger.info("👁️ Testing vision processing on MPS...")
        vision_inputs_mps = extractor_mps.processor(images=image, text=input_text, return_tensors="pt").to("mps")

        with torch.no_grad():
            mps_vision_output = extractor_mps.model.generate(
                **vision_inputs_mps,
                max_new_tokens=2,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Check MPS scores for NaN
        mps_vision_scores = mps_vision_output.scores[0][0]
        mps_vision_has_nan = torch.isnan(mps_vision_scores).any()
        logger.info(f"📊 MPS vision generation - NaN in scores: {mps_vision_has_nan}")

        if mps_vision_has_nan and not cpu_vision_has_nan:
            logger.error("❌ MPS causes NaN but CPU doesn't - MPS NUMERICAL ISSUE!")
            return "mps_issue"
        elif mps_vision_has_nan and cpu_vision_has_nan:
            logger.error("❌ Both CPU and MPS produce NaN - MODEL CORRUPTION!")
            return "model_issue"
        else:
            logger.info("✅ No NaN detected on either device")
            return "unknown"

    except Exception as e:
        logger.error(f"❌ Error during NaN testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_dtype_fix():
    """Test if changing dtype fixes the NaN issue."""

    logger.info("🔧 TESTING DTYPE FIX FOR NaN ISSUE")

    try:
        # Try with float32 instead of float16
        logger.info("🔄 Testing with float32 dtype...")

        import torch
        from transformers import AutoProcessor
        from transformers import MllamaForConditionalGeneration

        model = MllamaForConditionalGeneration.from_pretrained(
            "/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            device_map=None,
            torch_dtype=torch.float32,  # Use float32 instead of float16
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

        processor = AutoProcessor.from_pretrained(
            "/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            local_files_only=True
        )

        # Move to MPS
        if torch.backends.mps.is_available():
            model = model.to("mps")
            logger.info("✅ Float32 model moved to MPS")

        # Test generation
        image = extractor._preprocess_image("test_receipt.png")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Store name?"}]}]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=input_text, return_tensors="pt").to("mps")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Check for NaN in float32
        scores = output.scores[0][0]
        has_nan = torch.isnan(scores).any()
        logger.info(f"📊 Float32 MPS - NaN in scores: {has_nan}")

        if not has_nan:
            # Decode the response
            input_length = inputs["input_ids"].shape[1]
            new_tokens = output.sequences[0][input_length:]
            response = processor.decode(new_tokens, skip_special_tokens=True)
            logger.info(f"📄 Float32 response: '{response}'")

            if response and "!" not in response:
                logger.info("🎉 FLOAT32 FIX WORKS!")
                return True

        return False

    except Exception as e:
        logger.error(f"❌ Error testing float32: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🔍 NaN DEBUGGING - FINDING THE NUMERICAL ISSUE")
    logger.info("=" * 80)

    # Test 1: Check for NaN in different configurations
    nan_result = check_for_nans_in_model()

    # Test 2: Try dtype fix if MPS has issues
    if nan_result == "mps_issue":
        logger.info("\n" + "=" * 80)
        logger.info("🔧 TRYING DTYPE FIX...")
        logger.info("=" * 80)
        dtype_fix = test_dtype_fix()

        if dtype_fix:
            logger.info("🎉 SOLUTION FOUND: Use float32 instead of float16 on MPS!")
        else:
            logger.info("❌ Dtype fix didn't work")

    logger.info("=" * 80)
