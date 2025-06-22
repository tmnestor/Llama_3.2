"""
Pytest configuration and fixtures for tax invoice NER tests.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import yaml
from PIL import Image

from tax_invoice_ner.config.config_manager import ConfigManager


@pytest.fixture
def test_config() -> dict:
    """Create test configuration dictionary."""
    return {
        "model": {
            "model_path": "/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            "device": "cpu",
            "use_8bit": False,
            "max_new_tokens": 256,
        },
        "entities": {
            "BUSINESS_NAME": {
                "description": "Name of the business/company issuing the invoice",
                "examples": ["ABC Corp", "Smith & Associates"],
                "patterns": ["company", "business", "corporation"],
            },
            "TOTAL_AMOUNT": {
                "description": "Total invoice amount including tax",
                "examples": ["$1,234.56", "€500.00"],
                "patterns": ["total", "amount due"],
                "format": "currency",
            },
        },
        "processing": {
            "extraction_method": "sequential",
            "confidence_threshold": 0.7,
            "validation": {
                "currency_validation": True,
                "date_validation": True,
            },
        },
        "prompts": {
            "general_ner": "Extract entities from this tax invoice.",
            "specific_entity": "Find {entity_type} in this invoice.",
        },
    }


@pytest.fixture
def test_config_file(test_config: dict) -> Generator[str, None, None]:
    """Create temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    Path(config_path).unlink(missing_ok=True)


@pytest.fixture
def config_manager(test_config_file: str) -> ConfigManager:
    """Create ConfigManager instance with test configuration."""
    return ConfigManager(test_config_file)


@pytest.fixture
def test_image() -> Generator[str, None, None]:
    """Create test image file."""
    # Create a simple test image
    image = Image.new("RGB", (100, 100), color="white")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f.name)
        image_path = f.name

    yield image_path

    # Cleanup
    Path(image_path).unlink(missing_ok=True)


@pytest.fixture
def mock_extraction_result() -> dict:
    """Mock extraction result."""
    return {
        "entities": [
            {
                "text": "ABC Corporation",
                "label": "BUSINESS_NAME",
                "confidence": 0.95,
                "start_pos": 0,
                "end_pos": 14,
            },
            {
                "text": "$1,234.56",
                "label": "TOTAL_AMOUNT",
                "confidence": 0.88,
                "start_pos": 20,
                "end_pos": 29,
            },
        ],
        "document_type": "tax_invoice",
        "extraction_timestamp": "2024-06-20T16:52:00",
        "entity_types_requested": ["BUSINESS_NAME", "TOTAL_AMOUNT"],
        "config_entities": 2,
    }
