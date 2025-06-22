"""
Configuration manager for tax invoice NER system.

Handles loading, validation, and management of YAML configuration files
for entity definitions and processing settings.
"""

import logging
from pathlib import Path
from typing import Any

import yaml


class ConfigManager:
    """Manages configuration for tax invoice NER system."""

    def __init__(self, config_path: str | None = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "config/extractor/work_expense_ner_config.yaml"
        self.config = self.load_config(self.config_path)

    def load_config(self, config_path: str) -> dict[str, Any]:
        """Load YAML configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If configuration file not found or invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise ValueError(f"Configuration file not found: {config_path}")

        try:
            with config_file.open("r") as f:
                config = yaml.safe_load(f)

            self.logger.info(f"Loaded configuration from {config_path}")
            self._validate_config(config)
            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")

    def _validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration structure.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = ["model", "entities", "processing"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate model configuration
        model_config = config["model"]
        required_model_keys = ["model_path", "device", "max_new_tokens"]
        for key in required_model_keys:
            if key not in model_config:
                raise ValueError(f"Missing required model config key: {key}")

        # Validate entities
        entities = config["entities"]
        if not entities:
            raise ValueError("No entities defined in configuration")

        for entity_name, entity_config in entities.items():
            if "description" not in entity_config:
                raise ValueError(f"Entity {entity_name} missing description")

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration.

        Returns:
            Model configuration dictionary
        """
        return self.config["model"]

    def get_entities(self) -> dict[str, Any]:
        """Get entity definitions.

        Returns:
            Entity definitions dictionary
        """
        return self.config["entities"]

    def get_entity_types(self) -> list[str]:
        """Get list of available entity types.

        Returns:
            List of entity type names
        """
        return list(self.config["entities"].keys())

    def get_entity_config(self, entity_type: str) -> dict[str, Any]:
        """Get configuration for specific entity type.

        Args:
            entity_type: Entity type name

        Returns:
            Entity configuration dictionary

        Raises:
            ValueError: If entity type not found
        """
        if entity_type not in self.config["entities"]:
            raise ValueError(f"Unknown entity type: {entity_type}")

        return self.config["entities"][entity_type]

    def get_processing_config(self) -> dict[str, Any]:
        """Get processing configuration.

        Returns:
            Processing configuration dictionary
        """
        return self.config["processing"]

    def get_prompts_config(self) -> dict[str, Any]:
        """Get prompts configuration.

        Returns:
            Prompts configuration dictionary
        """
        return self.config.get("prompts", {})

    def update_model_path(self, model_path: str) -> None:
        """Update model path in configuration.

        Args:
            model_path: New model path
        """
        self.config["model"]["model_path"] = model_path
        self.logger.info(f"Updated model path to: {model_path}")

    def update_device(self, device: str) -> None:
        """Update device in configuration.

        Args:
            device: Device string (cpu, cuda, mps)
        """
        self.config["model"]["device"] = device
        self.logger.info(f"Updated device to: {device}")

    def get_confidence_threshold(self) -> float:
        """Get confidence threshold for entity extraction.

        Returns:
            Confidence threshold value
        """
        return self.config["processing"].get("confidence_threshold", 0.7)

    def get_validation_config(self) -> dict[str, Any]:
        """Get validation configuration.

        Returns:
            Validation configuration dictionary
        """
        return self.config["processing"].get("validation", {})
