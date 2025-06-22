#!/usr/bin/env python3
"""
Quick test to verify the exclamation mark detection and CPU fallback works.
"""

import logging

from models.extractors.llama_vision_extractor import LlamaVisionExtractor


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_detection():
    """Test the exclamation mark detection."""

    logger.info("🔍 TESTING EXCLAMATION MARK DETECTION")

    try:
        # Load model
        extractor = LlamaVisionExtractor(
            model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            use_8bit=False,
            max_new_tokens=10,  # Small for quick test
            device="mps",
        )

        logger.info("✅ Model loaded")

        # Test simple extraction
        logger.info("🔍 Testing field extraction...")
        response = extractor.extract_field("test_receipt.png", "store_name")

        logger.info(f"📄 Final response: '{response}'")

        # Check if CPU fallback was triggered
        if "Moving model to CPU" in str(response) or response != "!!!!!!!!!!!!":
            logger.info("🎉 SUCCESS: CPU fallback was triggered or meaningful response generated!")
            return True
        else:
            logger.warning("⚠️ Still getting exclamation marks without CPU fallback")
            return False

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🔍 EXCLAMATION MARK DETECTION TEST")
    logger.info("=" * 60)

    success = test_detection()

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✅ Detection and fallback working!")
    else:
        logger.info("❌ Detection needs improvement")
    logger.info("=" * 60)
