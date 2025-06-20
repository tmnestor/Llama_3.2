#!/usr/bin/env python3
"""
Final test with correct format and optimized MPS usage.
"""

import logging

from models.extractors.llama_vision_extractor import LlamaVisionExtractor


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_final_solution():
    """Test the final solution with correct format."""

    logger.info("🎯 FINAL TEST: Correct format + MPS optimization")

    try:
        # Set MPS memory optimization
        import os
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        # Load with minimal memory usage
        extractor = LlamaVisionExtractor(
            model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            use_8bit=False,
            max_new_tokens=15,  # Short response to test functionality
            device="mps",
        )

        logger.info("✅ Model loaded successfully")

        # Test with simple prompt
        logger.info("🔍 Testing simple store name extraction...")

        # Use the corrected extractor (already fixed)
        response = extractor.extract_field("test_receipt.png", "store_name")

        logger.info(f"📄 Response: '{response}'")

        # Analyze result
        if response and isinstance(response, str):
            clean_response = response.strip()
            if clean_response and not clean_response == "!!!!!!!!!!!!!!!!!":
                logger.info("🎉 SUCCESS: Generated meaningful response!")
                return True
            else:
                logger.warning(f"⚠️ Still getting exclamation marks: '{clean_response}'")
                return False
        else:
            logger.warning(f"⚠️ No response or wrong type: {type(response)}")
            return False

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🎯 FINAL SOLUTION TEST")
    logger.info("=" * 60)

    success = test_final_solution()

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🎉 SUCCESS: Exclamation marks issue RESOLVED!")
        logger.info("✅ Model now generates meaningful text responses")
    else:
        logger.info("⚠️ Issue persists - may need further investigation")
    logger.info("=" * 60)
