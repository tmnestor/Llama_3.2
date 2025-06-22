#!/usr/bin/env python3
"""
Basic tax invoice NER extraction example.

Demonstrates simple entity extraction from a tax invoice image.
"""

import logging
from pathlib import Path

from tax_invoice_ner import WorkExpenseNERExtractor


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run basic extraction example."""

    logger.info("🚀 Basic Tax Invoice NER Extraction Example")

    # Initialize extractor
    extractor = WorkExpenseNERExtractor()

    # Example image path
    image_path = "test_receipt.png"

    if not Path(image_path).exists():
        logger.warning(f"Test image not found: {image_path}")
        logger.info("Please provide a tax invoice image for testing")
        return

    # Extract all entities
    logger.info(f"Extracting entities from {image_path}...")
    result = extractor.extract_entities(image_path)

    # Display results
    logger.info(f"Extracted {len(result['entities'])} entities:")

    for entity in result["entities"]:
        confidence_emoji = (
            "🟢" if entity["confidence"] > 0.8 else "🟡" if entity["confidence"] > 0.6 else "🔴"
        )
        logger.info(
            f"  {confidence_emoji} {entity['label']}: '{entity['text']}' (confidence: {entity['confidence']:.2f})"
        )

    logger.info("✅ Basic extraction completed!")


if __name__ == "__main__":
    main()
