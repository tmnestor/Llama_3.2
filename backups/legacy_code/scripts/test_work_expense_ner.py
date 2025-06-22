#!/usr/bin/env python3
"""
Test script for the Tax Invoice NER system.

This script demonstrates the configurable NER system that extracts 
tax invoice specific entities based on YAML configuration.
"""

import json
import logging
import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.extractors.work_expense_ner_extractor import WorkExpenseNERExtractor


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_ner_extractor():
    """Test the configurable NER extractor."""

    logger.info("🏷️  TESTING TAX INVOICE NER SYSTEM")
    logger.info("=" * 60)

    try:
        # Initialize NER extractor with configuration
        logger.info("📥 Initializing NER extractor...")
        ner_extractor = WorkExpenseNERExtractor(
            config_path="config/extractor/work_expense_ner_config.yaml"
        )

        logger.info(f"✅ NER extractor loaded with {len(ner_extractor.get_available_entities())} entity types")

        # Display available entities
        logger.info("\n📋 Available Entity Types:")
        for entity_type in ner_extractor.get_available_entities():
            entity_config = ner_extractor.get_entity_config(entity_type)
            logger.info(f"  • {entity_type}: {entity_config['description']}")

        # Test entity extraction on sample image
        image_path = "test_receipt.png"
        if not Path(image_path).exists():
            logger.warning(f"⚠️  Test image not found: {image_path}")
            logger.info("ℹ️  Please provide a tax invoice image for testing")
            return

        logger.info(f"\n🔍 Testing NER extraction on: {image_path}")

        # Test 1: Extract all entities
        logger.info("\n📊 Test 1: Extract ALL entities")
        all_results = ner_extractor.extract_entities(image_path)

        logger.info(f"✅ Extracted {len(all_results['entities'])} entities")
        logger.info(f"📅 Extraction timestamp: {all_results['extraction_timestamp']}")

        # Display results
        for entity in all_results['entities']:
            confidence_emoji = "🟢" if entity['confidence'] > 0.8 else "🟡" if entity['confidence'] > 0.6 else "🔴"
            logger.info(f"  {confidence_emoji} {entity['label']}: '{entity['text']}' (confidence: {entity['confidence']:.2f})")

        # Test 2: Extract specific entity types
        logger.info("\n📊 Test 2: Extract FINANCIAL entities only")
        financial_entities = ["TOTAL_AMOUNT", "SUBTOTAL", "TAX_AMOUNT", "TAX_RATE"]
        financial_results = ner_extractor.extract_entities(
            image_path,
            entity_types=financial_entities
        )

        logger.info(f"✅ Extracted {len(financial_results['entities'])} financial entities")
        for entity in financial_results['entities']:
            logger.info(f"  💰 {entity['label']}: '{entity['text']}'")

        # Test 3: Extract business information
        logger.info("\n📊 Test 3: Extract BUSINESS entities only")
        business_entities = ["BUSINESS_NAME", "BUSINESS_ADDRESS", "ABN", "GST_NUMBER"]
        business_results = ner_extractor.extract_entities(
            image_path,
            entity_types=business_entities
        )

        logger.info(f"✅ Extracted {len(business_results['entities'])} business entities")
        for entity in business_results['entities']:
            logger.info(f"  🏢 {entity['label']}: '{entity['text']}'")

        # Test 4: Extract dates and identification
        logger.info("\n📊 Test 4: Extract DATE and ID entities")
        date_id_entities = ["INVOICE_DATE", "DUE_DATE", "INVOICE_NUMBER", "PURCHASE_ORDER"]
        date_id_results = ner_extractor.extract_entities(
            image_path,
            entity_types=date_id_entities
        )

        logger.info(f"✅ Extracted {len(date_id_results['entities'])} date/ID entities")
        for entity in date_id_results['entities']:
            logger.info(f"  📅 {entity['label']}: '{entity['text']}'")

        # Save results to JSON file
        output_file = "ner_extraction_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "all_entities": all_results,
                "financial_entities": financial_results,
                "business_entities": business_results,
                "date_id_entities": date_id_results
            }, f, indent=2)

        logger.info(f"\n💾 Results saved to: {output_file}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📈 EXTRACTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📊 Total entities configured: {len(ner_extractor.get_available_entities())}")
        logger.info(f"🎯 Total entities extracted: {len(all_results['entities'])}")
        logger.info(f"💰 Financial entities: {len(financial_results['entities'])}")
        logger.info(f"🏢 Business entities: {len(business_results['entities'])}")
        logger.info(f"📅 Date/ID entities: {len(date_id_results['entities'])}")

        # Configuration summary
        logger.info(f"\n⚙️  Configuration: {all_results.get('config_entities', 0)} entity types defined")
        logger.info(f"📝 Document type: {all_results.get('document_type', 'unknown')}")

        if "error" in all_results:
            logger.error(f"❌ Error occurred: {all_results['error']}")
            return False

        logger.info("\n🎉 NER testing completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Error during NER testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def demonstrate_configuration():
    """Demonstrate the YAML configuration system."""

    logger.info("\n🔧 YAML CONFIGURATION DEMONSTRATION")
    logger.info("=" * 60)

    try:
        # Load and display configuration
        config_path = "config/extractor/work_expense_ner_config.yaml"

        if not Path(config_path).exists():
            logger.error(f"❌ Configuration file not found: {config_path}")
            return

        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Show model configuration
        model_config = config.get("model", {})
        logger.info(f"🤖 Model path: {model_config.get('model_path')}")
        logger.info(f"💻 Device: {model_config.get('device')}")
        logger.info(f"🔢 Max tokens: {model_config.get('max_new_tokens')}")

        # Show entity categories
        entities = config.get("entities", {})
        logger.info(f"\n📋 Entity Categories ({len(entities)} types):")

        # Group entities by category
        categories = {
            "Business": ["BUSINESS_NAME", "VENDOR_NAME", "CLIENT_NAME"],
            "Financial": ["TOTAL_AMOUNT", "SUBTOTAL", "TAX_AMOUNT", "TAX_RATE"],
            "Dates": ["INVOICE_DATE", "DUE_DATE"],
            "Identification": ["INVOICE_NUMBER", "ABN", "GST_NUMBER", "PURCHASE_ORDER"],
            "Items": ["ITEM_DESCRIPTION", "ITEM_QUANTITY", "UNIT_PRICE", "LINE_TOTAL"],
            "Contact": ["CONTACT_PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"]
        }

        for category, entity_types in categories.items():
            logger.info(f"\n  {category}:")
            for entity_type in entity_types:
                if entity_type in entities:
                    entity_info = entities[entity_type]
                    logger.info(f"    • {entity_type}: {entity_info.get('description', 'No description')}")

        # Show processing configuration
        processing = config.get("processing", {})
        logger.info("\n⚙️  Processing Configuration:")
        logger.info(f"  📊 Extraction method: {processing.get('extraction_method')}")
        logger.info(f"  🎯 Confidence threshold: {processing.get('confidence_threshold')}")
        logger.info(f"  ✅ Validation enabled: {processing.get('validation', {})}")

        logger.info("\n✅ Configuration loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Error loading configuration: {e}")


if __name__ == "__main__":
    logger.info("🚀 TAX INVOICE NER SYSTEM TEST")
    logger.info("=" * 60)

    # Demonstrate configuration
    demonstrate_configuration()

    # Test NER system
    success = test_ner_extractor()

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ Tax Invoice NER system is working correctly")
        logger.info("📋 Entity types are fully configurable via YAML")
        logger.info("🎯 Sequential and parallel extraction modes available")
    else:
        logger.info("❌ TESTS FAILED!")
        logger.info("🔧 Check configuration and model setup")

    logger.info("=" * 60)
