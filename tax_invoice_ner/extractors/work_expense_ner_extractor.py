"""
Tax Invoice Named Entity Recognition (NER) Extractor using Llama-3.2-Vision.

This module implements configurable NER for tax invoices with entity types
defined in YAML configuration files.
"""

import json
import logging
import re
import signal
import time
from datetime import datetime
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

        # Compile frequently used regex patterns for performance
        self._compile_regex_patterns()

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
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # Load model with GPU optimization settings for KFP Discovery
        model_dtype = torch.float16

        # Configure attention implementation for GPU optimization
        attn_impl = "eager"  # Default for compatibility
        if self.config["model"].get("enable_flash_attention", False):
            attn_impl = "flash_attention_2"
        elif self.config["model"].get("enable_memory_efficient_attention", True):
            attn_impl = "sdpa"  # Scaled Dot Product Attention (PyTorch 2.0+)

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map=None,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )

        # Enhanced device handling for KFP Discovery optimization
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                self.model = self.model.to("cuda")
                # GPU memory optimization for KFP Discovery
                if hasattr(self.config["model"], "gpu_memory_fraction"):
                    torch.cuda.set_per_process_memory_fraction(
                        self.config["model"].get("gpu_memory_fraction", 0.8)
                    )
                self.logger.info("Using CUDA for optimized NER processing on KFP Discovery")
                self.logger.info(f"GPU: {torch.cuda.get_device_name()} with {torch.cuda.get_device_properties(0).total_memory // 1024**2}MB")
            else:
                self.device = "cpu"
                self.logger.info("CUDA not available, using CPU for NER processing")
        elif self.device == "cpu":
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

    def _compile_regex_patterns(self):
        """Compile frequently used regex patterns for better performance."""

        # Business name patterns (used in loops) - CORRECTED
        self.business_patterns_compiled = [
            re.compile(r"from\s+([A-Z](?:[A-Z\s&.,-]){2,30}?)\s+and\s+the\b", re.IGNORECASE),
            re.compile(r"from\s+([A-Z](?:[A-Z\s&.,-]){2,30}?)\s+and\b", re.IGNORECASE),
            re.compile(
                r"invoice\s+(?:is\s+)?from\s+([A-Z](?:[A-Z\s&.,-]){2,30}?)(?:\s+and|\s+[a-z])",
                re.IGNORECASE,
            ),
            re.compile(r'"([A-Z](?:[A-Z\s&.,-]){2,25})"', re.IGNORECASE),
            re.compile(r"business\s+name:?\s*([A-Z](?:[A-Z\s&.,-]){2,30})", re.IGNORECASE),
        ]

        # Currency amount patterns (used frequently) - ENHANCED
        self.amount_patterns_compiled = [
            re.compile(r"(?:AUD|\(AUD\))\s*(-?\$\s*[\d,]+\.?\d*)", re.IGNORECASE),
            re.compile(r"(-?\$\s*[\d,]+\.?\d*)\s*(?:AUD|\(AUD\))", re.IGNORECASE),
            re.compile(
                r"total\s+(?:amount\s+)?(?:is\s+)?(?:AUD|\(AUD\))?\s*(-?[\$€£¥]?[\d,]+\.?\d*)",
                re.IGNORECASE,
            ),
            re.compile(r"(-?[\$€£¥][\d,]+\.?\d*)", re.IGNORECASE),
            re.compile(r"amount:?\s*(?:AUD|\(AUD\))?\s*(-?[\$€£¥]?[\d,]+\.?\d*)", re.IGNORECASE),
            re.compile(r"(-?\d+\.\d{2})", re.IGNORECASE),
            # Bank statement specific patterns for negative amounts
            re.compile(r"withdrawal:?\s*(-?\$\s*[\d,]+\.?\d*)", re.IGNORECASE),
            re.compile(r"debit:?\s*(-?\$\s*[\d,]+\.?\d*)", re.IGNORECASE),
            re.compile(r"balance:?\s*(-?\$\s*[\d,]+\.?\d*)", re.IGNORECASE),
            # Australian bank statement patterns with CR notation
            re.compile(r"(\$?\s*[\d,]+\.\d{2})\s*\(?CR\)?", re.IGNORECASE),  # Amount followed by CR
            re.compile(r"balance:?\s*(\$?\s*[\d,]+\.\d{2})\s*\(?CR\)?", re.IGNORECASE),  # Balance with CR
        ]

        # AUD notation detection (used frequently)
        self.aud_notation_pattern = re.compile(r"(?:AUD|\(AUD\))", re.IGNORECASE)

        # Date patterns (used in multiple contexts) - OPTIMIZED
        self.date_patterns_compiled = [
            (re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})"), "DD/MM/YYYY"),
            (re.compile(r"(\d{1,2})-(\d{1,2})-(\d{4})"), "DD-MM-YYYY"),
            (re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})"), "DD.MM.YYYY"),
            (re.compile(r"(\d{1,2})/(\d{1,2})/(\d{2})"), "DD/MM/YY"),
            (re.compile(r"(\d{1,2})-(\d{1,2})-(\d{2})"), "DD-MM-YY"),
            (re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})"), "YYYY-MM-DD"),
            (
                re.compile(
                    r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})",
                    re.IGNORECASE,
                ),
                "DD Month YYYY",
            ),
            (
                re.compile(
                    r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
                    re.IGNORECASE,
                ),
                "DD Month YYYY",
            ),
            (
                re.compile(
                    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})",
                    re.IGNORECASE,
                ),
                "Month DD YYYY",
            ),
            (
                re.compile(
                    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
                    re.IGNORECASE,
                ),
                "Month DD YYYY",
            ),
        ]

        # Date context patterns (compiled for speed)
        self.invoice_date_contexts_compiled = [
            re.compile(r"invoice\s+date:?\s*([^\n\r,]+)", re.IGNORECASE),
            re.compile(r"date:?\s*([^\n\r,]+)", re.IGNORECASE),
            re.compile(r"issued:?\s*([^\n\r,]+)", re.IGNORECASE),
            re.compile(r"tax\s+invoice:?\s*([^\n\r,]+)", re.IGNORECASE),
        ]

        self.due_date_contexts_compiled = [
            re.compile(r"due\s+date:?\s*([^\n\r,]+)", re.IGNORECASE),
            re.compile(r"payment\s+due:?\s*([^\n\r,]+)", re.IGNORECASE),
            re.compile(r"pay\s+by:?\s*([^\n\r,]+)", re.IGNORECASE),
            re.compile(r"due:?\s*([^\n\r,]+)", re.IGNORECASE),
        ]

        # ABN patterns (used in loops) - ENHANCED
        self.abn_patterns_compiled = [
            re.compile(r"(?:ABN:?\s*)?(\d{2}\s\d{3}\s\d{3}\s\d{3})", re.IGNORECASE),
            re.compile(r"(?:ABN:?\s*)?(\d{11})", re.IGNORECASE),
            re.compile(r"(?:ABN:?\s*)?(\d{2}[\s\-\.]\d{3}[\s\-\.]\d{3}[\s\-\.]\d{3})", re.IGNORECASE),
            re.compile(
                r"australian\s+business\s+number:?\s*(\d{2}[\s\-\.]?\d{3}[\s\-\.]?\d{3}[\s\-\.]?\d{3})",
                re.IGNORECASE,
            ),
            re.compile(
                r"business\s+number:?\s*(\d{2}[\s\-\.]?\d{3}[\s\-\.]?\d{3}[\s\-\.]?\d{3})", re.IGNORECASE
            ),
        ]

        # ABN cleaning pattern
        self.abn_clean_pattern = re.compile(r"[\s\-\.]")

        # URL/Website patterns (compiled for performance)
        self.url_patterns_compiled = [
            re.compile(
                r"(?:website|web|url):?\s*((?:https?://)?(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)",
                re.IGNORECASE,
            ),
            re.compile(r"(https?://[a-zA-Z0-9][a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)", re.IGNORECASE),
            re.compile(r"(www\.[a-zA-Z0-9][a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)", re.IGNORECASE),
            re.compile(
                r"([a-zA-Z0-9][a-zA-Z0-9.-]+\.(?:com|org|net|edu|gov|com\.au|org\.au|net\.au)(?:/[^\s]*)?)",
                re.IGNORECASE,
            ),
        ]

        # Banking patterns (compiled for performance)
        self.bsb_patterns_compiled = [
            re.compile(r"(?:bsb|bank state branch):?\s*(\d{3}[\s\-]?\d{3})", re.IGNORECASE),
            re.compile(r"(\d{3}[\s\-]\d{3})(?:\s|$)", re.IGNORECASE),  # Standalone BSB format
        ]

        self.account_patterns_compiled = [
            re.compile(r"(?:account|acc|account no|account number):?\s*([\d\s\-]{6,})", re.IGNORECASE),
            re.compile(r"(\d{6,12})(?:\s|$)", re.IGNORECASE),  # Standalone account numbers
        ]

        self.bank_name_patterns_compiled = [
            re.compile(r"(commonwealth bank|cba|commbank)", re.IGNORECASE),
            re.compile(r"(westpac|wbc)", re.IGNORECASE),
            re.compile(r"(national australia bank|nab)", re.IGNORECASE),
            re.compile(r"(australia and new zealand banking|anz)", re.IGNORECASE),
            re.compile(r"(bendigo bank|bendigo)", re.IGNORECASE),
            re.compile(r"(suncorp|suncorp bank)", re.IGNORECASE),
            re.compile(r"(macquarie bank|macquarie)", re.IGNORECASE),
            re.compile(r"(ing bank|ing)", re.IGNORECASE),
            re.compile(r"([a-z\s]+(?:bank|credit union|building society))", re.IGNORECASE),
        ]

        # Date splitting pattern
        self.date_split_pattern = re.compile(r"[/\-\.]")

        # DD Month YYYY and Month DD YYYY matching patterns
        self.dd_month_yyyy_pattern = re.compile(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})")
        self.month_dd_yyyy_pattern = re.compile(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})")

        # JSON extraction pattern (used frequently)
        self.json_pattern = re.compile(r"(\{.*\})", re.DOTALL)

        # Validation patterns (compiled for performance) - CORRECTED
        self.currency_validation_pattern = re.compile(
            r"^(?:(?:AUD|\(AUD\))\s*)?-?\$\s*[\d,]+\.\d{2}(?:\s*(?:AUD|\(AUD\)|\(?CR\)?))?$"
        )
        self.date_validation_pattern = re.compile(
            r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$|^\d{4}-\d{1,2}-\d{1,2}$"
        )
        self.email_validation_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        self.phone_validation_pattern = re.compile(
            r"^(?:\+61\s?)?(?:\(0\d\)|0\d)\s?\d{4}\s?\d{4}$|^(?:\+61\s?)?4\d{2}\s?\d{3}\s?\d{3}$"
        )  # Australian format
        self.abn_validation_pattern = re.compile(r"^\d{2}\s?\d{3}\s?\d{3}\s?\d{3}$")
        self.url_validation_pattern = re.compile(
            r"^(?:https?://)?(?:www\.)?[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*(?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?$"
        )
        self.bsb_validation_pattern = re.compile(
            r"^\d{3}[\s\-]?\d{3}$"
        )  # BSB format: 123-456 or 123456 or 123 456
        self.numeric_validation_pattern = re.compile(r"^\d+[\s\-]*\d*$")  # Numeric with optional separators

        self.logger.info("Compiled regex patterns for optimized performance")

    def extract_entities(
        self, image_path: str | Path, entity_types: list[str] | None = None
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
            "config_entities": len(self.entity_types),
        }

        try:
            if self.config["processing"]["extraction_method"] == "sequential":
                result["entities"] = self._extract_entities_sequential(image, entity_types)
            else:
                result["entities"] = self._extract_entities_parallel(image, entity_types)

            # Calculate positions for extracted entities
            result["entities"] = self._calculate_entity_positions(result["entities"])

            # Apply post-processing validation
            result["entities"] = self._validate_entities(result["entities"])

            self.logger.info(f"Extracted {len(result['entities'])} entities from {image_path}")

        except Exception as e:
            self.logger.error(f"Failed to extract entities from {image_path}: {e}")
            result["error"] = str(e)

        return result

    def _extract_entities_sequential(
        self, image: Image.Image, entity_types: list[str]
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
        self, image: Image.Image, entity_types: list[str]
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

        # Store raw response for position calculation
        self._last_raw_response = response

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
            entity_examples=", ".join(entity_config["examples"]),
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
            entity_descriptions.append(f"{entity_type}: {entity_config['description']}")

        return prompt_template.format(entity_types=", ".join(entity_descriptions))

    def _generate_response(self, prompt: str, image: Image.Image) -> str:
        """Generate response from Llama-Vision model.

        Args:
            prompt: Text prompt
            image: PIL Image

        Returns:
            Generated response text
        """

        def timeout_handler(signum, frame):
            raise TimeoutError("Generation timed out after 12 hours")

        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(43200)  # 12 hours

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
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

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
                # GPU optimization: Enable memory efficient inference
                if self.device == "cuda":
                    torch.cuda.empty_cache()  # Clear GPU cache before generation

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,  # Greedy decoding for speed
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for faster generation
                    # GPU optimization settings
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict_in_generate=False,
                )

            generation_time = time.time() - generation_step_start
            self.logger.info(f"✅ NER generation completed in {generation_time:.1f}s")

            # Decode response
            self.logger.debug("🔄 Decoding response...")
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )

            # Debug: Log the raw response
            self.logger.info(f"🔍 Raw model response: {response}")  # Full response for debugging

        except TimeoutError:
            self.logger.error("Generation timed out")
            return ""
        except Exception as e:
            self.logger.error(f"Error in generation: {e}")
            return ""
        finally:
            signal.alarm(0)

        return response

    def _parse_entity_response(self, response: str, entity_type: str) -> list[dict[str, Any]]:
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
                            entities.append(
                                {
                                    "text": item["text"],
                                    "label": entity_type,
                                    "confidence": item.get("confidence", 0.8),
                                    "start_pos": item.get("start_pos"),
                                    "end_pos": item.get("end_pos"),
                                }
                            )
                elif isinstance(json_data, dict):
                    # Single entity or entities dict
                    if "entities" in json_data:
                        for entity in json_data["entities"]:
                            if entity.get("label") == entity_type:
                                entities.append(entity)
                    elif "text" in json_data:
                        entities.append(
                            {
                                "text": json_data["text"],
                                "label": entity_type,
                                "confidence": json_data.get("confidence", 0.8),
                                "start_pos": json_data.get("start_pos"),
                                "end_pos": json_data.get("end_pos"),
                            }
                        )
            else:
                # Parse as plain text
                text = response.strip()
                if text and text != "null" and text != "None":
                    entities.append(
                        {
                            "text": text,
                            "label": entity_type,
                            "confidence": 0.7,  # Default confidence for text extraction
                            "start_pos": None,
                            "end_pos": None,
                        }
                    )

        except json.JSONDecodeError:
            # Fallback to text extraction
            text = response.strip()
            if text and text != "null" and text != "None":
                entities.append(
                    {
                        "text": text,
                        "label": entity_type,
                        "confidence": 0.6,  # Lower confidence for non-JSON
                        "start_pos": None,
                        "end_pos": None,
                    }
                )

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
            # Find JSON in response using compiled pattern
            json_match = self.json_pattern.search(response)
            if json_match:
                json_str = json_match.group(1)
                json_data = json.loads(json_str)

                if "entities" in json_data:
                    entities = json_data["entities"]
            else:
                # Fallback: Parse plain text response for known patterns
                self.logger.info("No JSON found, attempting text parsing...")
                entities = self._parse_text_response(response)

        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON response, trying text parsing...")
            entities = self._parse_text_response(response)

        return entities

    def _parse_text_response(self, response: str) -> list[dict[str, Any]]:
        """Parse plain text response for entity extraction.

        Args:
            response: Plain text response from model

        Returns:
            List of extracted entities
        """
        entities = []

        # Business name patterns using compiled patterns (optimized)
        for i, pattern in enumerate(self.business_patterns_compiled):
            match = pattern.search(response)
            if match:
                business_name = match.group(1).strip()
                # Debug logging
                self.logger.debug(f"🔍 Business pattern {i + 1} matched: '{business_name}'")
                if len(business_name) > 2:  # Filter out very short matches
                    entities.append(
                        {
                            "text": business_name,
                            "label": "BUSINESS_NAME",
                            "confidence": 0.85,
                            "start_pos": None,
                            "end_pos": None,
                        }
                    )
                break

        amounts_found = set()  # Track found amounts to avoid duplicates

        # Check if AUD notation is present using compiled pattern (optimized)
        has_aud_notation = bool(self.aud_notation_pattern.search(response))

        # Currency amount patterns using compiled patterns (optimized)
        for pattern in self.amount_patterns_compiled:
            matches = pattern.findall(response)
            for amount in matches:
                # Clean and format amount, preserving negative sign and detecting CR notation
                clean_amount = amount.replace(",", "").replace("$", "").strip()

                # Check for Australian bank CR (credit) notation in the surrounding context
                is_credit = False
                pattern_match = pattern.search(response)
                if pattern_match:
                    # Look for CR notation near the amount
                    match_context = response[max(0, pattern_match.start() - 20) : pattern_match.end() + 20]
                    is_credit = bool(re.search(r"\(?CR\)?", match_context, re.IGNORECASE))

                # Determine if this is a negative amount
                is_negative = clean_amount.startswith("-")
                if is_negative:
                    clean_amount = clean_amount[1:]  # Remove the minus sign temporarily

                if "." in clean_amount or clean_amount.isdigit():
                    try:
                        float_val = float(clean_amount)
                        if float_val >= 0:  # Valid amount (allow zero)
                            # Apply negative sign if present (unless it's a CR which indicates positive balance)
                            if is_negative and not is_credit:
                                float_val = -float_val
                            # CR notation in Australian bank statements typically indicates a positive balance/credit

                            # Format with AUD notation if detected in document
                            if has_aud_notation:
                                if is_negative and not is_credit:
                                    formatted_amount = f"AUD -${abs(float_val):.2f}"
                                else:
                                    formatted_amount = f"AUD ${abs(float_val):.2f}"
                                    if is_credit:
                                        formatted_amount += " CR"
                            else:
                                if is_negative and not is_credit:
                                    formatted_amount = f"-${abs(float_val):.2f}"
                                else:
                                    formatted_amount = f"${abs(float_val):.2f}"
                                    if is_credit:
                                        formatted_amount += " CR"

                            if formatted_amount not in amounts_found:  # Avoid duplicates
                                amounts_found.add(formatted_amount)

                                # Determine entity type based on context, sign, and CR notation
                                entity_label = "TOTAL_AMOUNT"
                                if is_negative and not is_credit:
                                    # Check context for withdrawal/debit patterns
                                    if any(
                                        keyword in response.lower()
                                        for keyword in ["withdrawal", "debit", "wd", "withdraw"]
                                    ):
                                        entity_label = "WITHDRAWAL_AMOUNT"
                                    elif "balance" in response.lower():
                                        entity_label = "ACCOUNT_BALANCE"
                                elif is_credit or "balance" in response.lower():
                                    # CR amounts or balance context typically indicate account balance
                                    entity_label = "ACCOUNT_BALANCE"
                                elif any(
                                    keyword in response.lower()
                                    for keyword in ["deposit", "credit", "payment in"]
                                ):
                                    entity_label = "DEPOSIT_AMOUNT"

                                entities.append(
                                    {
                                        "text": formatted_amount,
                                        "label": entity_label,
                                        "confidence": 0.85,
                                        "start_pos": None,
                                        "end_pos": None,
                                    }
                                )
                    except ValueError:
                        continue

        # Date parsing patterns (extensive Australian formats)
        self._extract_dates(response, entities)

        # Australian Business Number (ABN) parsing
        self._extract_abn(response, entities)

        # Website/URL parsing
        self._extract_urls(response, entities)

        # Banking information parsing
        self._extract_banking_info(response, entities)

        # Remove duplicates
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity["text"], entity["label"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def _extract_dates(self, response: str, entities: list[dict[str, Any]]) -> None:
        """Extract dates with extensive Australian format support.

        Args:
            response: Text response to parse
            entities: List to append found date entities to
        """

        # Use compiled date patterns for better performance
        date_patterns = self.date_patterns_compiled

        # Use compiled context patterns for better performance
        invoice_date_contexts = self.invoice_date_contexts_compiled
        due_date_contexts = self.due_date_contexts_compiled

        def parse_date_string(date_str: str, format_hint: str) -> str | None:
            """Parse date string and return standardized format."""
            date_str = date_str.strip()

            # Month name mappings
            month_map = {
                "jan": "01",
                "january": "01",
                "feb": "02",
                "february": "02",
                "mar": "03",
                "march": "03",
                "apr": "04",
                "april": "04",
                "may": "05",
                "jun": "06",
                "june": "06",
                "jul": "07",
                "july": "07",
                "aug": "08",
                "august": "08",
                "sep": "09",
                "september": "09",
                "oct": "10",
                "october": "10",
                "nov": "11",
                "november": "11",
                "dec": "12",
                "december": "12",
            }

            try:
                if format_hint == "DD/MM/YYYY":
                    parts = self.date_split_pattern.split(date_str)
                    if len(parts) == 3:
                        day, month, year = parts
                        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

                elif format_hint == "DD/MM/YY":
                    parts = self.date_split_pattern.split(date_str)
                    if len(parts) == 3:
                        day, month, year = parts
                        year_int = int(year)
                        if year_int < 50:
                            year_int += 2000
                        else:
                            year_int += 1900
                        return f"{year_int:04d}-{int(month):02d}-{int(day):02d}"

                elif format_hint == "YYYY-MM-DD":
                    parts = date_str.split("-")
                    if len(parts) == 3:
                        year, month, day = parts
                        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

                elif format_hint == "DD Month YYYY":
                    match = self.dd_month_yyyy_pattern.match(date_str)
                    if match:
                        day, month_name, year = match.groups()
                        month_num = month_map.get(month_name.lower())
                        if month_num:
                            return f"{int(year):04d}-{month_num}-{int(day):02d}"

                elif format_hint == "Month DD YYYY":
                    match = self.month_dd_yyyy_pattern.match(date_str)
                    if match:
                        month_name, day, year = match.groups()
                        month_num = month_map.get(month_name.lower())
                        if month_num:
                            return f"{int(year):04d}-{month_num}-{int(day):02d}"

            except (ValueError, IndexError):
                pass

            return None

        # Extract invoice dates using compiled patterns
        for context_pattern in invoice_date_contexts:
            match = context_pattern.search(response)
            if match:
                date_context = match.group(1)
                for pattern, format_hint in date_patterns:
                    date_match = pattern.search(date_context)
                    if date_match:
                        parsed_date = parse_date_string(date_match.group(0), format_hint)
                        if parsed_date:
                            entities.append(
                                {
                                    "text": parsed_date,
                                    "label": "INVOICE_DATE",
                                    "confidence": 0.9,
                                    "start_pos": None,
                                    "end_pos": None,
                                }
                            )
                        break
                break

        # Extract due dates using compiled patterns
        for context_pattern in due_date_contexts:
            match = context_pattern.search(response)
            if match:
                date_context = match.group(1)
                for pattern, format_hint in date_patterns:
                    date_match = pattern.search(date_context)
                    if date_match:
                        parsed_date = parse_date_string(date_match.group(0), format_hint)
                        if parsed_date:
                            entities.append(
                                {
                                    "text": parsed_date,
                                    "label": "DUE_DATE",
                                    "confidence": 0.9,
                                    "start_pos": None,
                                    "end_pos": None,
                                }
                            )
                        break
                break

    def _extract_abn(self, response: str, entities: list[dict[str, Any]]) -> None:
        """Extract Australian Business Numbers with validation.

        Args:
            response: Text response to parse
            entities: List to append found ABN entities to
        """

        def validate_abn(abn_digits: str) -> bool:
            """Validate ABN using Australian Government algorithm."""
            if len(abn_digits) != 11:
                return False

            # ABN validation weights
            weights = [10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

            try:
                # Convert first digit by subtracting 1
                digits = [int(abn_digits[0]) - 1] + [int(d) for d in abn_digits[1:]]

                # Calculate weighted sum
                total = sum(digit * weight for digit, weight in zip(digits, weights, strict=False))

                # Check if divisible by 89
                return total % 89 == 0
            except (ValueError, IndexError):
                return False

        def format_abn(abn_digits: str) -> str:
            """Format ABN as XX XXX XXX XXX."""
            return f"{abn_digits[:2]} {abn_digits[2:5]} {abn_digits[5:8]} {abn_digits[8:11]}"

        # Use compiled ABN patterns for better performance
        for pattern in self.abn_patterns_compiled:
            matches = pattern.findall(response)
            for match in matches:
                # Clean ABN using compiled pattern (optimized)
                clean_abn = self.abn_clean_pattern.sub("", match)

                if len(clean_abn) == 11 and clean_abn.isdigit():
                    if validate_abn(clean_abn):
                        formatted_abn = format_abn(clean_abn)
                        entities.append(
                            {
                                "text": formatted_abn,
                                "label": "ABN",
                                "confidence": 0.95,
                                "start_pos": None,
                                "end_pos": None,
                            }
                        )
                    else:
                        # Still include invalid ABNs but with lower confidence
                        formatted_abn = format_abn(clean_abn)
                        entities.append(
                            {
                                "text": f"{formatted_abn} (unvalidated)",
                                "label": "ABN",
                                "confidence": 0.6,
                                "start_pos": None,
                                "end_pos": None,
                            }
                        )

    def _extract_urls(self, response: str, entities: list[dict[str, Any]]) -> None:
        """Extract website URLs and web addresses.

        Args:
            response: Text response to parse
            entities: List to append found URL entities to
        """

        def normalize_url(url: str) -> str:
            """Normalize URL format for consistency."""
            url = url.strip()

            # Remove trailing punctuation that might be picked up
            url = url.rstrip(".,;!?)")

            # Add protocol if missing for full URLs
            if url.startswith("www.") and not url.startswith(("http://", "https://")):
                url = "https://" + url
            elif not url.startswith(("http://", "https://", "www.")) and "." in url:
                # For domain-only URLs like "company.com.au"
                if not url.startswith("www."):
                    url = "www." + url
                url = "https://" + url

            return url

        urls_found = set()  # Track found URLs to avoid duplicates

        # Use compiled URL patterns for better performance
        for pattern in self.url_patterns_compiled:
            matches = pattern.findall(response)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle multiple groups in regex
                    url = next((m for m in match if m), "")
                else:
                    url = match

                if url and len(url) > 4:  # Minimum reasonable URL length
                    normalized_url = normalize_url(url)

                    if normalized_url not in urls_found:
                        urls_found.add(normalized_url)
                        entities.append(
                            {
                                "text": normalized_url,
                                "label": "WEBSITE",
                                "confidence": 0.8,
                                "start_pos": None,
                                "end_pos": None,
                            }
                        )

    def _extract_banking_info(self, response: str, entities: list[dict[str, Any]]) -> None:
        """Extract banking information from bank statements.

        Args:
            response: Text response to parse
            entities: List to append found banking entities to
        """

        # Extract BSB codes
        bsb_found = set()
        for pattern in self.bsb_patterns_compiled:
            matches = pattern.findall(response)
            for match in matches:
                clean_bsb = match.replace(" ", "-").replace("--", "-")
                if clean_bsb not in bsb_found and len(clean_bsb) >= 6:
                    bsb_found.add(clean_bsb)
                    entities.append(
                        {
                            "text": clean_bsb,
                            "label": "BSB",
                            "confidence": 0.9,
                            "start_pos": None,
                            "end_pos": None,
                        }
                    )

        # Extract account numbers
        account_found = set()
        for pattern in self.account_patterns_compiled:
            matches = pattern.findall(response)
            for match in matches:
                clean_account = match.strip().replace(" ", "").replace("-", "")
                # Only consider likely account numbers (6-12 digits)
                if (
                    clean_account not in account_found
                    and clean_account.isdigit()
                    and 6 <= len(clean_account) <= 12
                ):
                    account_found.add(clean_account)
                    entities.append(
                        {
                            "text": clean_account,
                            "label": "ACCOUNT_NUMBER",
                            "confidence": 0.85,
                            "start_pos": None,
                            "end_pos": None,
                        }
                    )

        # Extract bank names
        bank_names_found = set()
        for pattern in self.bank_name_patterns_compiled:
            matches = pattern.findall(response)
            for match in matches:
                bank_name = match.strip()
                if bank_name and len(bank_name) > 2:
                    # Normalize common bank names
                    bank_name_normalized = self._normalize_bank_name(bank_name)
                    if bank_name_normalized not in bank_names_found:
                        bank_names_found.add(bank_name_normalized)
                        entities.append(
                            {
                                "text": bank_name_normalized,
                                "label": "BANK_NAME",
                                "confidence": 0.9,
                                "start_pos": None,
                                "end_pos": None,
                            }
                        )

    def _normalize_bank_name(self, bank_name: str) -> str:
        """Normalize bank name to standard format."""
        bank_name = bank_name.strip().title()

        # Standard bank name mappings
        name_mappings = {
            "Cba": "Commonwealth Bank",
            "Commbank": "Commonwealth Bank",
            "Commonwealth Bank Of Australia": "Commonwealth Bank",
            "Wbc": "Westpac",
            "Westpac Banking Corporation": "Westpac",
            "National Australia Bank": "NAB",
            "Australia And New Zealand Banking": "ANZ",
            "Anz Bank": "ANZ",
            "Bendigo And Adelaide Bank": "Bendigo Bank",
            "Ing Bank": "ING Bank",
            "Macquarie Bank Limited": "Macquarie Bank",
        }

        return name_mappings.get(bank_name, bank_name)

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
                self.logger.debug(
                    f"Filtering out entity '{entity['text']}' due to low confidence: {confidence:.2f}"
                )
                continue

            # Check if entity type is valid
            entity_type = entity["label"]
            if entity_type not in self.entities:
                self.logger.warning(
                    f"Unknown entity type '{entity_type}' for text '{entity['text']}' - filtering out"
                )
                continue

            # Apply content validation (length, non-empty, etc.)
            if not self._validate_entity_content(entity):
                continue

            # Apply format-specific validation
            entity_config = self.entities[entity_type]
            entity_format = entity_config.get("format")

            if entity_format and not self._validate_format(
                entity["text"], entity_format, validation_config
            ):
                self.logger.debug(
                    f"Filtering out entity '{entity['text']}' due to invalid format: {entity_format}"
                )
                continue

            # Apply business rule validation
            if not self._validate_business_rules(entity, entity_config):
                continue

            validated_entities.append(entity)

        # Remove duplicate entities
        validated_entities = self._remove_duplicate_entities(validated_entities)

        self.logger.info(f"Validation complete: {len(entities)} -> {len(validated_entities)} entities")
        return validated_entities

    def _validate_entity_content(self, entity: dict[str, Any]) -> bool:
        """Validate entity content for basic quality checks.

        Args:
            entity: Entity dictionary to validate

        Returns:
            True if entity content is valid
        """
        text = entity.get("text", "").strip()

        # Check for empty or null content
        if not text or text.lower() in {"null", "none", "n/a", "not found", "unknown"}:
            self.logger.debug(f"Filtering out empty/null entity: '{text}'")
            return False

        # Check minimum length (avoid single character extractions unless they make sense)
        entity_type = entity.get("label", "")
        if entity_type not in {"TAX_RATE"} and len(text) < 2:
            self.logger.debug(f"Filtering out too-short entity: '{text}' (type: {entity_type})")
            return False

        # Check maximum reasonable length
        if len(text) > 500:
            self.logger.debug(f"Filtering out overly long entity: '{text[:50]}...'")
            return False

        # Check for common extraction errors (repeated characters, gibberish)
        if len(set(text)) == 1 and len(text) > 3:  # "aaaa" type strings
            self.logger.debug(f"Filtering out repeated character string: '{text}'")
            return False

        return True

    def _validate_business_rules(self, entity: dict[str, Any], entity_config: dict[str, Any]) -> bool:
        """Apply business-specific validation rules.

        Args:
            entity: Entity to validate
            entity_config: Configuration for the entity type

        Returns:
            True if entity passes business rules
        """
        text = entity.get("text", "")
        entity_type = entity.get("label", "")

        # Currency amount validation
        if entity_type in {
            "TOTAL_AMOUNT",
            "SUBTOTAL",
            "TAX_AMOUNT",
            "UNIT_PRICE",
            "LINE_TOTAL",
            "WITHDRAWAL_AMOUNT",
            "DEPOSIT_AMOUNT",
            "ACCOUNT_BALANCE",
        }:
            # Check for reasonable amount ranges
            try:
                # Extract numeric value from currency text, preserving negative sign
                # Remove currency symbols but keep minus sign
                clean_text = re.sub(r"[^\d.\-]", "", text.replace(",", ""))
                numeric_value = float(clean_text)

                # Allow negative amounts for bank statement entities (withdrawals, balances)
                if entity_type in {"WITHDRAWAL_AMOUNT", "ACCOUNT_BALANCE"}:
                    # For bank statements, negative amounts are valid (withdrawals, overdrafts)
                    if abs(numeric_value) > 1000000:  # Check absolute value for reasonableness
                        self.logger.debug(f"Filtering out unreasonably large amount: {text}")
                        return False
                else:
                    # For invoices/receipts, negative amounts are unusual
                    if numeric_value < 0:
                        self.logger.debug(f"Filtering out negative amount for {entity_type}: {text}")
                        return False
                    if numeric_value > 1000000:  # Over $1M seems unreasonable for most expenses
                        self.logger.debug(f"Filtering out unreasonably large amount: {text}")
                        return False
            except ValueError:
                pass  # Let format validation handle invalid currency formats

        # Date validation
        if entity_type in {"INVOICE_DATE", "DUE_DATE", "TRANSACTION_DATE"}:
            # Check if date is reasonable (not too far in past/future)
            try:
                # Simple date extraction - this could be enhanced
                if len(text) >= 8:  # Minimum date length
                    # Note: Future enhancement could add date range validation here
                    # to check if dates are within reasonable business ranges
                    pass
            except Exception:
                pass  # Let format validation handle date parsing

        # Business name validation
        if entity_type in {"BUSINESS_NAME", "VENDOR_NAME", "CLIENT_NAME", "BANK_NAME"}:
            # Check for reasonable business name characteristics
            if len(text) < 3:
                self.logger.debug(f"Filtering out too-short business name: '{text}'")
                return False
            if text.isdigit():
                self.logger.debug(f"Filtering out numeric-only business name: '{text}'")
                return False

        # Phone number basic checks
        if entity_type == "PHONE_NUMBER":
            # Remove all non-digits
            digits_only = re.sub(r"\D", "", text)
            if len(digits_only) < 8 or len(digits_only) > 15:  # Reasonable phone number length
                self.logger.debug(f"Filtering out invalid phone number length: '{text}'")
                return False

        # Email validation
        if entity_type == "EMAIL_ADDRESS":
            if "@" not in text or text.count("@") != 1:
                self.logger.debug(f"Filtering out invalid email format: '{text}'")
                return False

        return True

    def _remove_duplicate_entities(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate entities, keeping the one with highest confidence.

        Args:
            entities: List of entities to deduplicate

        Returns:
            List of deduplicated entities
        """
        if not entities:
            return entities

        # Group entities by (label, text) pairs
        entity_groups = {}
        for entity in entities:
            key = (entity.get("label"), entity.get("text", "").strip().lower())
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)

        # Keep the entity with highest confidence from each group
        deduplicated = []
        for group in entity_groups.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Sort by confidence and take the highest
                best_entity = max(group, key=lambda e: e.get("confidence", 0.0))
                deduplicated.append(best_entity)
                self.logger.debug(
                    f"Removed {len(group) - 1} duplicate(s) for entity: '{best_entity['text']}'"
                )

        return deduplicated

    def _validate_format(self, text: str, format_type: str, validation_config: dict[str, Any]) -> bool:
        """Validate entity text against format requirements.

        Args:
            text: Entity text to validate
            format_type: Format type (currency, date, email, etc.)
            validation_config: Validation configuration

        Returns:
            True if valid, False otherwise
        """
        if format_type == "currency" and validation_config.get("currency_validation"):
            # Use compiled currency validation pattern (optimized)
            return bool(self.currency_validation_pattern.match(text))
        elif format_type == "date" and validation_config.get("date_validation"):
            # Use compiled date validation pattern (optimized)
            return bool(self.date_validation_pattern.match(text))
        elif format_type == "email" and validation_config.get("email_validation"):
            # Use compiled email validation pattern (improved & optimized)
            return bool(self.email_validation_pattern.match(text))
        elif format_type == "phone" and validation_config.get("phone_validation"):
            # Use compiled Australian phone validation pattern (improved & optimized)
            return bool(self.phone_validation_pattern.match(text))
        elif format_type == "abn" and validation_config.get("abn_validation"):
            # Use compiled ABN validation pattern (optimized)
            return bool(self.abn_validation_pattern.match(text))
        elif format_type == "url" and validation_config.get("url_validation"):
            # Use compiled URL validation pattern (optimized)
            return bool(self.url_validation_pattern.match(text))
        elif format_type == "bsb" and validation_config.get("bsb_validation"):
            # Use compiled BSB validation pattern (optimized)
            return bool(self.bsb_validation_pattern.match(text))
        elif format_type == "numeric" and validation_config.get("numeric_validation"):
            # Use compiled numeric validation pattern (optimized)
            return bool(self.numeric_validation_pattern.match(text))

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

    def _calculate_entity_positions(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Calculate start_pos and end_pos for extracted entities based on model response.

        Args:
            entities: List of extracted entities

        Returns:
            List of entities with calculated positions
        """
        if not hasattr(self, '_last_raw_response') or not self._last_raw_response:
            self.logger.warning("No raw response available for position calculation")
            return entities

        response_text = self._last_raw_response.lower()
        updated_entities = []

        for entity in entities:
            entity_text = entity.get("text", "").strip()
            if not entity_text:
                updated_entities.append(entity)
                continue

            # Try exact match first
            start_pos = response_text.find(entity_text.lower())

            if start_pos == -1:
                # Try fuzzy matching for common variations
                start_pos = self._fuzzy_find_position(entity_text, response_text)

            if start_pos != -1:
                end_pos = start_pos + len(entity_text)
                entity["start_pos"] = start_pos
                entity["end_pos"] = end_pos

                # Add source snippet for validation (10 chars before/after)
                snippet_start = max(0, start_pos - 10)
                snippet_end = min(len(self._last_raw_response), end_pos + 10)
                entity["source_snippet"] = self._last_raw_response[snippet_start:snippet_end]

                self.logger.debug(f"Found position for '{entity_text}': {start_pos}-{end_pos}")
            else:
                self.logger.debug(f"Could not find position for entity: '{entity_text}'")
                entity["start_pos"] = None
                entity["end_pos"] = None
                entity["source_snippet"] = None

            updated_entities.append(entity)

        return updated_entities

    def _fuzzy_find_position(self, entity_text: str, response_text: str) -> int:
        """Find entity position using fuzzy matching for common variations.

        Args:
            entity_text: Text to find
            response_text: Text to search in (already lowercase)

        Returns:
            Start position or -1 if not found
        """
        entity_lower = entity_text.lower()

        # Common variations to try
        variations = [
            entity_lower,
            entity_lower.replace("$", ""),  # Remove currency symbols
            entity_lower.replace(",", ""),  # Remove commas
            entity_lower.replace(" ", ""),  # Remove spaces
            entity_lower.replace("-", ""),  # Remove hyphens
            entity_lower.replace(".", ""),  # Remove periods
        ]

        # Also try partial matches for business names
        if len(entity_lower) > 5:
            variations.append(entity_lower[:5])  # First 5 characters
            variations.append(entity_lower[-5:])  # Last 5 characters

        for variation in variations:
            if variation and len(variation) > 1:
                pos = response_text.find(variation)
                if pos != -1:
                    return pos

        return -1
