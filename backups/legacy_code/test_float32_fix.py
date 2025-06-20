#!/usr/bin/env python3
"""
Quick test of the float32 fix with minimal generation.
"""

import logging

from models.extractors.llama_vision_extractor import LlamaVisionExtractor


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_float32_fix():
    """Test the float32 fix with minimal generation."""

    logger.info("🎯 TESTING FLOAT32 FIX")

    try:
        # Load with float32 on MPS
        extractor = LlamaVisionExtractor(
            model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            use_8bit=False,
            max_new_tokens=5,  # Very minimal for quick test
            device="mps",
        )

        logger.info("✅ Model loaded with float32")

        # Test simple extraction
        logger.info("🔍 Testing store name extraction...")
        response = extractor.extract_field("test_receipt.png", "store_name")

        logger.info(f"📄 Response: '{response}'")

        # Check if we got meaningful output
        if response and isinstance(response, str):
            clean_response = response.strip()
            if clean_response and not all(c == '!' for c in clean_response):
                logger.info("🎉 SUCCESS: Got meaningful text instead of exclamation marks!")
                logger.info(f"🎯 Actual response: '{clean_response}'")
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
    logger.info("🎯 FLOAT32 FIX TEST")
    logger.info("=" * 60)

    success = test_float32_fix()

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🎉 EXCLAMATION MARKS BUG FIXED!")
        logger.info("✅ Float32 on MPS solves the numerical instability")
    else:
        logger.info("❌ Issue still persists")
    logger.info("=" * 60)
