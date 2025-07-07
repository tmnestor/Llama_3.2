"""
Tests for configuration manager module.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from tax_invoice_ner.config.config_manager import ConfigManager


class TestConfigManager:
    """Test suite for ConfigManager class."""

    def test_load_valid_config(self, test_config_file: str):
        """Test loading valid configuration file."""
        config_manager = ConfigManager(test_config_file)

        assert config_manager.config is not None
        assert "model" in config_manager.config
        assert "entities" in config_manager.config
        assert "processing" in config_manager.config

    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(ValueError, match="Configuration file not found"):
            ConfigManager("nonexistent_config.yaml")

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: {\n")
            invalid_config_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid YAML configuration"):
                ConfigManager(invalid_config_path)
        finally:
            Path(invalid_config_path).unlink(missing_ok=True)

    def test_validate_config_missing_sections(self):
        """Test configuration validation with missing sections."""
        incomplete_config = {"model": {"model_path": "test"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(incomplete_config, f)
            config_path = f.name

        try:
            with pytest.raises(
                ValueError, match="Missing required configuration section"
            ):
                ConfigManager(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_get_model_config(self, config_manager: ConfigManager):
        """Test getting model configuration."""
        model_config = config_manager.get_model_config()

        assert "model_path" in model_config
        assert "device" in model_config
        assert "max_new_tokens" in model_config

    def test_get_entities(self, config_manager: ConfigManager):
        """Test getting entity definitions."""
        entities = config_manager.get_entities()

        assert "BUSINESS_NAME" in entities
        assert "TOTAL_AMOUNT" in entities
        assert entities["BUSINESS_NAME"]["description"] is not None

    def test_get_entity_types(self, config_manager: ConfigManager):
        """Test getting entity type list."""
        entity_types = config_manager.get_entity_types()

        assert isinstance(entity_types, list)
        assert "BUSINESS_NAME" in entity_types
        assert "TOTAL_AMOUNT" in entity_types

    def test_get_entity_config(self, config_manager: ConfigManager):
        """Test getting specific entity configuration."""
        entity_config = config_manager.get_entity_config("BUSINESS_NAME")

        assert "description" in entity_config
        assert "examples" in entity_config
        assert (
            entity_config["description"]
            == "Name of the business/company issuing the invoice"
        )

    def test_get_entity_config_invalid(self, config_manager: ConfigManager):
        """Test getting configuration for non-existent entity."""
        with pytest.raises(ValueError, match="Unknown entity type"):
            config_manager.get_entity_config("NONEXISTENT_ENTITY")

    def test_get_processing_config(self, config_manager: ConfigManager):
        """Test getting processing configuration."""
        processing_config = config_manager.get_processing_config()

        assert "extraction_method" in processing_config
        assert "confidence_threshold" in processing_config
        assert processing_config["extraction_method"] == "sequential"

    def test_get_confidence_threshold(self, config_manager: ConfigManager):
        """Test getting confidence threshold."""
        threshold = config_manager.get_confidence_threshold()

        assert isinstance(threshold, float)
        assert threshold == 0.7

    def test_get_validation_config(self, config_manager: ConfigManager):
        """Test getting validation configuration."""
        validation_config = config_manager.get_validation_config()

        assert "currency_validation" in validation_config
        assert "date_validation" in validation_config
        assert validation_config["currency_validation"] is True

    def test_update_model_path(self, config_manager: ConfigManager):
        """Test updating model path."""
        new_path = "/new/model/path"
        config_manager.update_model_path(new_path)

        assert config_manager.get_model_config()["model_path"] == new_path

    def test_update_device(self, config_manager: ConfigManager):
        """Test updating device."""
        new_device = "cuda"
        config_manager.update_device(new_device)

        assert config_manager.get_model_config()["device"] == new_device
