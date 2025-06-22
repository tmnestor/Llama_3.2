#!/usr/bin/env python3
"""
Configuration management demonstration.

Shows how to work with YAML configuration files and entity definitions.
"""

import logging

from tax_invoice_ner import ConfigManager


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate configuration management."""

    logger.info("⚙️  Tax Invoice NER Configuration Demo")

    try:
        # Load configuration
        config_manager = ConfigManager("config/extractor/work_expense_ner_config.yaml")

        # Display model configuration
        model_config = config_manager.get_model_config()
        logger.info("\n🤖 Model Configuration:")
        logger.info(f"  • Path: {model_config['model_path']}")
        logger.info(f"  • Device: {model_config['device']}")
        logger.info(f"  • Max tokens: {model_config['max_new_tokens']}")

        # Display available entities
        entity_types = config_manager.get_entity_types()
        logger.info(f"\n📋 Available Entity Types ({len(entity_types)} total):")

        # Group by category for display
        categories = {
            "Business": ["BUSINESS_NAME", "VENDOR_NAME", "CLIENT_NAME"],
            "Financial": ["TOTAL_AMOUNT", "SUBTOTAL", "TAX_AMOUNT", "TAX_RATE"],
            "Dates": ["INVOICE_DATE", "DUE_DATE"],
            "Identification": ["INVOICE_NUMBER", "ABN", "GST_NUMBER"],
        }

        for category, entities in categories.items():
            logger.info(f"\n  {category}:")
            for entity_type in entities:
                if entity_type in entity_types:
                    config = config_manager.get_entity_config(entity_type)
                    logger.info(f"    • {entity_type}: {config['description']}")

        # Display processing configuration
        processing_config = config_manager.get_processing_config()
        logger.info("\n⚙️  Processing Configuration:")
        logger.info(f"  • Extraction method: {processing_config['extraction_method']}")
        logger.info(f"  • Confidence threshold: {config_manager.get_confidence_threshold()}")

        # Display validation settings
        validation_config = config_manager.get_validation_config()
        logger.info(f"  • Validation settings: {list(validation_config.keys())}")

        # Demonstrate configuration updates
        logger.info("\n🔧 Configuration Updates:")
        original_device = model_config["device"]
        config_manager.update_device("cuda")
        logger.info(f"  • Updated device: {original_device} → cuda")

        # Reset
        config_manager.update_device(original_device)
        logger.info(f"  • Reset device: cuda → {original_device}")

        logger.info("\n✅ Configuration demo completed!")

    except Exception as e:
        logger.error(f"❌ Configuration demo failed: {e}")


if __name__ == "__main__":
    main()
