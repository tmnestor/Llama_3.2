#!/usr/bin/env python3
"""
Quick test of the NER system with limited entities.
"""

import logging
import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.extractors.work_expense_ner_extractor import WorkExpenseNERExtractor


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def quick_ner_test():
    """Quick test with just a few entities."""

    logger.info("🚀 QUICK NER TEST")
    logger.info("=" * 40)

    try:
        # Initialize NER extractor
        logger.info("📥 Loading NER extractor...")
        ner_extractor = WorkExpenseNERExtractor(
            config_path="config/extractor/work_expense_ner_config.yaml"
        )

        logger.info(f"✅ Loaded with {len(ner_extractor.get_available_entities())} entity types")

        # Test with just 3 business entities
        if not Path("test_receipt.png").exists():
            logger.warning("⚠️  test_receipt.png not found - create a test image")
            return

        logger.info("🔍 Testing with 3 business entities...")
        business_entities = ["BUSINESS_NAME", "TOTAL_AMOUNT", "INVOICE_DATE"]

        result = ner_extractor.extract_entities(
            "test_receipt.png",
            entity_types=business_entities
        )

        logger.info("✅ Extraction completed!")
        logger.info(f"📊 Extracted {len(result['entities'])} entities:")

        for entity in result['entities']:
            logger.info(f"  • {entity['label']}: '{entity['text']}' (confidence: {entity['confidence']:.2f})")

        return True

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = quick_ner_test()
    if success:
        logger.info("🎉 Quick test passed!")
    else:
        logger.info("❌ Quick test failed!")
