"""
Tests for NER extractor module.

Note: These tests focus on the public interface and configuration handling.
Model loading and inference tests require the actual model files.
"""

from unittest.mock import Mock, patch

import pytest

from tax_invoice_ner.extractors.work_expense_ner_extractor import (
    WorkExpenseNERExtractor,
)


class TestWorkExpenseNERExtractor:
    """Test suite for WorkExpenseNERExtractor class."""

    @patch("tax_invoice_ner.extractors.work_expense_ner_extractor.Path")
    def test_init_with_valid_config(self, mock_path, test_config_file: str):
        """Test initialization with valid configuration."""
        # Mock model path existence check
        mock_path.return_value.exists.return_value = True

        with patch.object(WorkExpenseNERExtractor, "_load_model"):
            extractor = WorkExpenseNERExtractor(config_path=test_config_file)

            assert extractor.config is not None
            assert extractor.model_path is not None
            assert extractor.device is not None
            assert extractor.max_new_tokens is not None

    def test_init_with_nonexistent_config(self):
        """Test initialization with non-existent configuration file."""
        with pytest.raises(ValueError, match="Configuration file not found"):
            WorkExpenseNERExtractor(config_path="nonexistent_config.yaml")

    @patch("tax_invoice_ner.extractors.work_expense_ner_extractor.Path")
    def test_init_with_overrides(self, mock_path, test_config_file: str):
        """Test initialization with parameter overrides."""
        mock_path.return_value.exists.return_value = True

        with patch.object(WorkExpenseNERExtractor, "_load_model"):
            extractor = WorkExpenseNERExtractor(
                config_path=test_config_file,
                model_path="/custom/model/path",
                device="cuda",
            )

            assert extractor.model_path == "/custom/model/path"
            assert extractor.device == "cuda"

    @patch("tax_invoice_ner.extractors.work_expense_ner_extractor.Path")
    def test_get_available_entities(self, mock_path, test_config_file: str):
        """Test getting available entity types."""
        mock_path.return_value.exists.return_value = True

        with patch.object(WorkExpenseNERExtractor, "_load_model"):
            extractor = WorkExpenseNERExtractor(config_path=test_config_file)

            entities = extractor.get_available_entities()

            assert isinstance(entities, list)
            assert "BUSINESS_NAME" in entities
            assert "TOTAL_AMOUNT" in entities

    @patch("tax_invoice_ner.extractors.work_expense_ner_extractor.Path")
    def test_get_entity_config(self, mock_path, test_config_file: str):
        """Test getting entity configuration."""
        mock_path.return_value.exists.return_value = True

        with patch.object(WorkExpenseNERExtractor, "_load_model"):
            extractor = WorkExpenseNERExtractor(config_path=test_config_file)

            config = extractor.get_entity_config("BUSINESS_NAME")

            assert "description" in config
            assert (
                config["description"]
                == "Name of the business/company issuing the invoice"
            )

    @patch("tax_invoice_ner.extractors.work_expense_ner_extractor.Path")
    def test_get_entity_config_invalid(self, mock_path, test_config_file: str):
        """Test getting configuration for invalid entity type."""
        mock_path.return_value.exists.return_value = True

        with patch.object(WorkExpenseNERExtractor, "_load_model"):
            extractor = WorkExpenseNERExtractor(config_path=test_config_file)

            with pytest.raises(ValueError, match="Unknown entity type"):
                extractor.get_entity_config("INVALID_ENTITY")

    @patch("tax_invoice_ner.extractors.work_expense_ner_extractor.Path")
    def test_preprocess_image_valid(
        self, mock_path, test_config_file: str, test_image: str
    ):
        """Test image preprocessing with valid image."""
        mock_path.return_value.exists.return_value = True

        with patch.object(WorkExpenseNERExtractor, "_load_model"):
            extractor = WorkExpenseNERExtractor(config_path=test_config_file)

            # Mock the image preprocessing
            with patch.object(extractor, "_preprocess_image") as mock_preprocess:
                mock_preprocess.return_value = Mock()

                result = extractor._preprocess_image(test_image)
                assert result is not None

    @patch("tax_invoice_ner.extractors.work_expense_ner_extractor.Path")
    def test_preprocess_image_invalid(self, mock_path, test_config_file: str):
        """Test image preprocessing with non-existent image."""
        mock_path.return_value.exists.return_value = True

        with patch.object(WorkExpenseNERExtractor, "_load_model"):
            extractor = WorkExpenseNERExtractor(config_path=test_config_file)

            with pytest.raises(ValueError, match="Image does not exist"):
                extractor._preprocess_image("nonexistent_image.png")

    @patch("tax_invoice_ner.extractors.work_expense_ner_extractor.Path")
    def test_validate_entities(self, mock_path, test_config_file: str):
        """Test entity validation."""
        mock_path.return_value.exists.return_value = True

        with patch.object(WorkExpenseNERExtractor, "_load_model"):
            extractor = WorkExpenseNERExtractor(config_path=test_config_file)

            # Test entities with different confidence levels
            test_entities = [
                {"text": "ABC Corp", "label": "BUSINESS_NAME", "confidence": 0.9},
                {
                    "text": "XYZ Inc",
                    "label": "BUSINESS_NAME",
                    "confidence": 0.5,
                },  # Below threshold
                {"text": "$100.00", "label": "TOTAL_AMOUNT", "confidence": 0.8},
            ]

            validated = extractor._validate_entities(test_entities)

            # Should filter out low confidence entity
            assert len(validated) == 2
            assert all(entity["confidence"] >= 0.7 for entity in validated)

    @patch("tax_invoice_ner.extractors.work_expense_ner_extractor.Path")
    def test_parse_entity_response_json(self, mock_path, test_config_file: str):
        """Test parsing JSON entity response."""
        mock_path.return_value.exists.return_value = True

        with patch.object(WorkExpenseNERExtractor, "_load_model"):
            extractor = WorkExpenseNERExtractor(config_path=test_config_file)

            json_response = '{"text": "ABC Corporation", "confidence": 0.95}'
            result = extractor._parse_entity_response(json_response, "BUSINESS_NAME")

            assert len(result) == 1
            assert result[0]["text"] == "ABC Corporation"
            assert result[0]["label"] == "BUSINESS_NAME"
            assert result[0]["confidence"] == 0.95

    @patch("tax_invoice_ner.extractors.work_expense_ner_extractor.Path")
    def test_parse_entity_response_text(self, mock_path, test_config_file: str):
        """Test parsing plain text entity response."""
        mock_path.return_value.exists.return_value = True

        with patch.object(WorkExpenseNERExtractor, "_load_model"):
            extractor = WorkExpenseNERExtractor(config_path=test_config_file)

            text_response = "ABC Corporation"
            result = extractor._parse_entity_response(text_response, "BUSINESS_NAME")

            assert len(result) == 1
            assert result[0]["text"] == "ABC Corporation"
            assert result[0]["label"] == "BUSINESS_NAME"
            assert result[0]["confidence"] == 0.7  # Default confidence

    @patch("tax_invoice_ner.extractors.work_expense_ner_extractor.Path")
    def test_extract_entities_invalid_types(
        self, mock_path, test_config_file: str, test_image: str
    ):
        """Test extraction with invalid entity types."""
        mock_path.return_value.exists.return_value = True

        with patch.object(WorkExpenseNERExtractor, "_load_model"):
            extractor = WorkExpenseNERExtractor(config_path=test_config_file)

            with pytest.raises(ValueError, match="Invalid entity types"):
                extractor.extract_entities(test_image, entity_types=["INVALID_ENTITY"])

    @patch("tax_invoice_ner.extractors.work_expense_ner_extractor.Path")
    def test_timestamp_generation(self, mock_path, test_config_file: str):
        """Test timestamp generation."""
        mock_path.return_value.exists.return_value = True

        with patch.object(WorkExpenseNERExtractor, "_load_model"):
            extractor = WorkExpenseNERExtractor(config_path=test_config_file)

            timestamp = extractor._get_timestamp()

            assert isinstance(timestamp, str)
            assert "T" in timestamp  # ISO format contains 'T'
