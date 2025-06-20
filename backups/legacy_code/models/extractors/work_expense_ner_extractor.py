#!/usr/bin/env python3
"""
Tax Invoice Named Entity Recognition (NER) Extractor using Llama-3.2-Vision.

This module implements configurable NER for tax invoices with entity types
defined in YAML configuration files.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers import MllamaForConditionalGeneration


class WorkExpenseNERExtractor:
    """Tax Invoice NER extractor with configurable entity types."""

    def __init__(
        self,
        config_path: str = "config/extractor/work_expense_ner_config.yaml",
        model_path: str | None = None,
        device: str | None = None,
    ):
        """Initialize the NER extractor.

        Args:
            config_path: Path to YAML configuration file
            model_path: Override model path from config
            device: Override device from config
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = self._load_config(config_path)

        # Override config with parameters if provided
        if model_path:
            self.config["model"]["model_path"] = model_path
        if device:
            self.config["model"]["device"] = device

        self.model_path = self.config["model"]["model_path"]
        self.device = self.config["model"]["device"]
        self.max_new_tokens = self.config["model"]["max_new_tokens"]

        # Load entity definitions
        self.entities = self.config["entities"]
        self.entity_types = list(self.entities.keys())

        # Initialize model components
        self._load_model()

        self.logger.info(f"NER extractor initialized with {len(self.entity_types)} entity types")

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load YAML configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise ValueError(f"Configuration file not found: {config_path}")

        with config_file.open("r") as f:
            config = yaml.safe_load(f)

        self.logger.info(f"Loaded configuration from {config_path}")
        return config

    def _load_model(self):
        """Load the Llama-Vision model and components."""
        # Check if model path exists
        if not Path(self.model_path).exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")

        self.logger.info(f"Loading Llama-Vision model from {self.model_path}")

        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        # Load model with appropriate settings
        model_dtype = torch.float16
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map=None,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

        # Device handling
        if self.device == "cpu":
            self.device = "cpu"
            self.logger.info("Using CPU for stable NER processing")
        elif self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.device = "cuda"
            self.logger.info("Using CUDA for NER processing")
        else:
            self.device = "cpu"
            self.logger.info("Defaulting to CPU for stable NER processing")

        # Set model to evaluation mode
        self.model.eval()

        # Set pad token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.info("Llama-Vision NER model loaded successfully")

    def extract_entities(
        self,
        image_path: str | Path,
        entity_types: list[str] | None = None
    ) -> dict[str, Any]:
        """Extract specified entities from tax invoice image.

        Args:
            image_path: Path to tax invoice image
            entity_types: List of entity types to extract (default: all)

        Returns:
            Dictionary with extracted entities
        """
        # Default to all entity types if none specified
        if entity_types is None:
            entity_types = self.entity_types

        # Validate entity types
        invalid_types = [et for et in entity_types if et not in self.entity_types]
        if invalid_types:
            raise ValueError(f"Invalid entity types: {invalid_types}")

        # Load and preprocess image
        image = self._preprocess_image(image_path)

        # Extract entities
        result = {
            "entities": [],
            "document_type": "tax_invoice",
            "extraction_timestamp": self._get_timestamp(),
            "entity_types_requested": entity_types,
            "config_entities": len(self.entity_types)
        }

        try:
            if self.config["processing"]["extraction_method"] == "sequential":
                result["entities"] = self._extract_entities_sequential(image, entity_types)
            else:
                result["entities"] = self._extract_entities_parallel(image, entity_types)

            # Apply post-processing validation
            result["entities"] = self._validate_entities(result["entities"])

            self.logger.info(f"Extracted {len(result['entities'])} entities from {image_path}")

        except Exception as e:
            self.logger.error(f"Failed to extract entities from {image_path}: {e}")
            result["error"] = str(e)

        return result

    def _extract_entities_sequential(
        self,
        image: Image.Image,
        entity_types: list[str]
    ) -> list[dict[str, Any]]:
        """Extract entities sequentially (one at a time).

        Args:
            image: PIL Image object
            entity_types: List of entity types to extract

        Returns:
            List of extracted entities
        """
        entities = []

        for entity_type in entity_types:
            self.logger.info(f"Extracting {entity_type}...")

            try:
                # Generate entity-specific prompt
                prompt = self._generate_entity_prompt(entity_type)

                # Generate response
                response = self._generate_response(prompt, image)

                # Parse entities from response
                extracted_entities = self._parse_entity_response(response, entity_type)
                entities.extend(extracted_entities)

            except Exception as e:
                self.logger.error(f"Failed to extract {entity_type}: {e}")

        return entities

    def _extract_entities_parallel(
        self,
        image: Image.Image,
        entity_types: list[str]
    ) -> list[dict[str, Any]]:
        """Extract all entities in single prompt.

        Args:
            image: PIL Image object
            entity_types: List of entity types to extract

        Returns:
            List of extracted entities
        """
        # Generate general NER prompt
        prompt = self._generate_general_ner_prompt(entity_types)

        # Generate response
        response = self._generate_response(prompt, image)

        # Parse all entities from response
        entities = self._parse_general_ner_response(response)

        return entities

    def _generate_entity_prompt(self, entity_type: str) -> str:
        """Generate prompt for specific entity type.

        Args:
            entity_type: Type of entity to extract

        Returns:
            Formatted prompt string
        """
        entity_config = self.entities[entity_type]

        prompt_template = self.config["prompts"]["specific_entity"]

        return prompt_template.format(
            entity_type=entity_type,
            entity_description=entity_config["description"],
            entity_examples=", ".join(entity_config["examples"])
        )

    def _generate_general_ner_prompt(self, entity_types: list[str]) -> str:
        """Generate prompt for extracting multiple entity types.

        Args:
            entity_types: List of entity types to extract

        Returns:
            Formatted prompt string
        """
        prompt_template = self.config["prompts"]["general_ner"]

        entity_descriptions = []
        for entity_type in entity_types:
            entity_config = self.entities[entity_type]
            entity_descriptions.append(
                f"{entity_type}: {entity_config['description']}"
            )

        return prompt_template.format(
            entity_types=", ".join(entity_descriptions)
        )

    def _generate_response(self, prompt: str, image: Image.Image) -> str:
        """Generate response from Llama-Vision model.

        Args:
            prompt: Text prompt
            image: PIL Image

        Returns:
            Generated response text
        """
        import signal
        import time

        def timeout_handler(signum, frame):
            raise TimeoutError("Generation timed out after 12 hours")

        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(43200)  # 12 hours

        generation_start = time.time()

        try:
            # Prepare messages
            self.logger.debug("🔄 Preparing messages for NER extraction...")
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]

            # Apply chat template
            self.logger.debug("🔄 Applying chat template...")
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            # Process inputs
            self.logger.debug("🔄 Processing inputs...")
            inputs = self.processor(
                images=image,
                text=input_text,
                return_tensors="pt",
            ).to(self.device)

            # Generate response with simple progress logging
            self.logger.info(f"🔄 Starting NER generation ({self.max_new_tokens} tokens)...")
            generation_step_start = time.time()

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            generation_time = time.time() - generation_step_start
            self.logger.info(f"✅ NER generation completed in {generation_time:.1f}s")

            # Decode response
            self.logger.debug("🔄 Decoding response...")
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )

        except TimeoutError:
            self.logger.error("Generation timed out")
            return ""
        except Exception as e:
            self.logger.error(f"Error in generation: {e}")
            return ""
        finally:
            signal.alarm(0)

        return response

    def _parse_entity_response(
        self,
        response: str,
        entity_type: str
    ) -> list[dict[str, Any]]:
        """Parse entity-specific response.

        Args:
            response: Raw response from model
            entity_type: Type of entity being parsed

        Returns:
            List of extracted entities
        """
        entities = []

        try:
            # Try to parse as JSON first
            if response.strip().startswith("{") or response.strip().startswith("["):
                json_data = json.loads(response)

                if isinstance(json_data, list):
                    # List of entities
                    for item in json_data:
                        if isinstance(item, dict) and "text" in item:
                            entities.append({
                                "text": item["text"],
                                "label": entity_type,
                                "confidence": item.get("confidence", 0.8),
                                "start_pos": item.get("start_pos"),
                                "end_pos": item.get("end_pos")
                            })
                elif isinstance(json_data, dict):
                    # Single entity or entities dict
                    if "entities" in json_data:
                        for entity in json_data["entities"]:
                            if entity.get("label") == entity_type:
                                entities.append(entity)
                    elif "text" in json_data:
                        entities.append({
                            "text": json_data["text"],
                            "label": entity_type,
                            "confidence": json_data.get("confidence", 0.8),
                            "start_pos": json_data.get("start_pos"),
                            "end_pos": json_data.get("end_pos")
                        })
            else:
                # Parse as plain text
                text = response.strip()
                if text and text != "null" and text != "None":
                    entities.append({
                        "text": text,
                        "label": entity_type,
                        "confidence": 0.7,  # Default confidence for text extraction
                        "start_pos": None,
                        "end_pos": None
                    })

        except json.JSONDecodeError:
            # Fallback to text extraction
            text = response.strip()
            if text and text != "null" and text != "None":
                entities.append({
                    "text": text,
                    "label": entity_type,
                    "confidence": 0.6,  # Lower confidence for non-JSON
                    "start_pos": None,
                    "end_pos": None
                })

        return entities

    def _parse_general_ner_response(self, response: str) -> list[dict[str, Any]]:
        """Parse general NER response with multiple entities.

        Args:
            response: Raw response from model

        Returns:
            List of extracted entities
        """
        entities = []

        try:
            # Find JSON in response
            json_match = re.search(r"(\{.*\})", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                json_data = json.loads(json_str)

                if "entities" in json_data:
                    entities = json_data["entities"]

        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON response: {response[:100]}...")

        return entities

    def _validate_entities(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate and filter entities based on configuration.

        Args:
            entities: List of extracted entities

        Returns:
            List of validated entities
        """
        validated_entities = []
        validation_config = self.config["processing"]["validation"]
        confidence_threshold = self.config["processing"]["confidence_threshold"]

        for entity in entities:
            # Check confidence threshold
            confidence = entity.get("confidence", 0.0)
            if confidence < confidence_threshold:
                continue

            # Apply format-specific validation
            entity_type = entity["label"]
            if entity_type in self.entities:
                entity_config = self.entities[entity_type]
                entity_format = entity_config.get("format")

                if entity_format and not self._validate_format(entity["text"], entity_format, validation_config):
                    continue

            validated_entities.append(entity)

        return validated_entities

    def _validate_format(
        self,
        text: str,
        format_type: str,
        validation_config: dict[str, Any]
    ) -> bool:
        """Validate entity text against format requirements.

        Args:
            text: Entity text to validate
            format_type: Format type (currency, date, email, etc.)
            validation_config: Validation configuration

        Returns:
            True if valid, False otherwise
        """
        if format_type == "currency" and validation_config.get("currency_validation"):
            return bool(re.match(r"[\$€£¥]\s*[\d,]+\.?\d*", text))
        elif format_type == "date" and validation_config.get("date_validation"):
            return bool(re.match(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{1,2}-\d{1,2}", text))
        elif format_type == "email" and validation_config.get("email_validation"):
            return bool(re.match(r"[^@]+@[^@]+\.[^@]+", text))
        elif format_type == "phone" and validation_config.get("phone_validation"):
            return bool(re.match(r"[\+]?[\d\s\-\(\)]{8,}", text))
        elif format_type == "abn" and validation_config.get("abn_validation"):
            # Australian Business Number validation
            return bool(re.match(r"\d{2}\s?\d{3}\s?\d{3}\s?\d{3}", text))

        return True  # No validation required

    def _preprocess_image(self, image_path: str | Path) -> Image.Image:
        """Preprocess image for NER extraction.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed PIL Image
        """
        if isinstance(image_path, str):
            image_path = Path(image_path)

        if not image_path.exists():
            raise ValueError(f"Image does not exist: {image_path}")

        image = Image.open(image_path).convert("RGB")
        return image

    def _get_timestamp(self) -> str:
        """Get current timestamp for extraction.

        Returns:
            ISO format timestamp string
        """
        from datetime import datetime
        return datetime.now().isoformat()

    def get_available_entities(self) -> list[str]:
        """Get list of available entity types.

        Returns:
            List of entity type names
        """
        return self.entity_types

    def get_entity_config(self, entity_type: str) -> dict[str, Any]:
        """Get configuration for specific entity type.

        Args:
            entity_type: Entity type name

        Returns:
            Entity configuration dictionary
        """
        if entity_type not in self.entities:
            raise ValueError(f"Unknown entity type: {entity_type}")

        return self.entities[entity_type]
