#!/usr/bin/env python3
"""
Targeted entity extraction example.

Demonstrates extracting specific entity types from tax invoices.
"""

import logging
from pathlib import Path

from tax_invoice_ner import WorkExpenseNERExtractor


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run targeted extraction example."""

    logger.info("🎯 Targeted Tax Invoice Entity Extraction Example")

    # Initialize extractor
    extractor = WorkExpenseNERExtractor()

    # Example image path
    image_path = "test_receipt.png"

    if not Path(image_path).exists():
        logger.warning(f"Test image not found: {image_path}")
        logger.info("Please provide a tax invoice image for testing")
        return

    # Define entity groups for targeted extraction
    entity_groups = {
        "Business Information": ["BUSINESS_NAME", "VENDOR_NAME", "ABN", "GST_NUMBER"],
        "Financial Details": ["TOTAL_AMOUNT", "SUBTOTAL", "TAX_AMOUNT", "TAX_RATE"],
        "Date Information": ["INVOICE_DATE", "DUE_DATE"],
        "Contact Details": ["CONTACT_PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"],
    }

    # Extract each group
    for group_name, entity_types in entity_groups.items():
        logger.info(f"\n📋 Extracting {group_name}:")

        result = extractor.extract_entities(image_path, entity_types=entity_types)

        if result["entities"]:
            for entity in result["entities"]:
                confidence_color = (
                    "🟢" if entity["confidence"] > 0.8 else "🟡" if entity["confidence"] > 0.6 else "🔴"
                )
                logger.info(f"  {confidence_color} {entity['label']}: '{entity['text']}'")
        else:
            logger.info("  ⚪ No entities found in this category")

    logger.info("\n✅ Targeted extraction completed!")


if __name__ == "__main__":
    main()
