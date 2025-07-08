"""Inference engine for Llama-3.2-Vision package."""

import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ..config import LlamaConfig
from ..utils import setup_logging

try:
    import requests
except ImportError:
    requests = None


class LlamaInferenceEngine:
    """Handle inference with CUDA fixes and optimization."""

    def __init__(self, model: Any, processor: Any, config: LlamaConfig):
        """Initialize inference engine.

        Args:
            model: Loaded Llama model
            processor: Loaded processor
            config: Configuration object
        """
        self.model = model
        self.processor = processor
        self.config = config
        self.logger = setup_logging(config.log_level)

        # Store device information
        self.device = self._detect_model_device()
        self.logger.info(f"Inference engine initialized on device: {self.device}")

    def _detect_model_device(self) -> str:
        """Detect which device the model is loaded on."""
        if hasattr(self.model, "device"):
            return str(self.model.device)
        elif hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            # Multi-device model - use first device
            devices = list(self.model.hf_device_map.values())
            return str(devices[0]) if devices else "cpu"
        else:
            return "cpu"

    def _preprocess_image(self, image_path: str) -> Image.Image:
        """Preprocess image for Llama-3.2-Vision compatibility.

        Args:
            image_path: Path to image file or HTTP URL

        Returns:
            Preprocessed PIL Image
        """
        # Load image
        if image_path.startswith("http"):
            if requests is None:
                raise ImportError(
                    "requests library not available for HTTP image loading"
                )
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if too large (Llama has size limits)
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            self.logger.info(f"Image resized to {image.size} (max: {max_size})")

        return image

    def _prepare_inputs(self, image: Image.Image, prompt: str) -> dict[str, Any]:
        """Prepare inputs for model inference.

        Args:
            image: Preprocessed PIL Image
            prompt: Text prompt (should include <|image|> token)

        Returns:
            Processed inputs dictionary
        """
        # Ensure prompt includes image token
        if not prompt.startswith("<|image|>"):
            prompt_with_image = f"<|image|>{prompt}"
        else:
            prompt_with_image = prompt

        # Process inputs
        inputs = self.processor(
            text=prompt_with_image, images=image, return_tensors="pt"
        )

        self.logger.debug(
            f"Input shapes - IDs: {inputs['input_ids'].shape}, Pixels: {inputs['pixel_values'].shape}"
        )

        # Move to correct device
        if self.device != "cpu":
            device_target = (
                self.device.split(":")[0] if ":" in self.device else self.device
            )
            inputs = {
                k: v.to(device_target) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

        return inputs

    def _clean_response(self, response: str) -> str:
        """Clean response from repetitive text and artifacts.

        Args:
            response: Raw model response

        Returns:
            Cleaned response text
        """
        import re

        # Remove excessive repetition of ANY word repeated 3+ times consecutively
        # This catches "au au au", "the the the", "hello hello hello", etc.
        response = re.sub(r"\b(\w+)(\s+\1){2,}", r"\1", response, flags=re.IGNORECASE)

        # Remove excessive repetition of longer phrases (up to 3 words) repeated 3+ times
        # This catches "Thank you Thank you Thank you" or "Visit costco Visit costco" etc.
        response = re.sub(
            r"\b((?:\w+\s+){1,3})(?:\1){2,}", r"\1", response, flags=re.IGNORECASE
        )

        # Remove excessive repetition of any short token/phrase (1-5 chars) repeated 5+ times
        response = re.sub(
            r"\b(\w{1,5})\s+(?:\1\s+){4,}", "", response, flags=re.IGNORECASE
        )

        # Stop at common receipt endings
        stop_patterns = [
            r"Thank you.*$",
            r"Visit.*costco\.au.*$",
            r"Member #\d+.*$",
            r"\d{2}/\d{2}/\d{4}.*Thank.*$",
        ]

        for pattern in stop_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                response = response[: match.start()].strip()
                break

        # Clean up excessive whitespace
        response = re.sub(r"\s+", " ", response)

        # Limit response length to reasonable size for receipts
        if len(response) > 1000:
            response = response[:1000] + "..."

        return response.strip()

    def predict(self, image_path: str, prompt: str) -> str:
        """Generate prediction with CUDA-safe parameters.

        Args:
            image_path: Path to image file or HTTP URL
            prompt: Text prompt for extraction

        Returns:
            Generated response text
        """
        try:
            start_time = time.time()

            # Preprocess image
            image = self._preprocess_image(image_path)

            # Prepare inputs
            inputs = self._prepare_inputs(image, prompt)

            # Generate with CUDA-safe parameters (NO repetition_penalty)
            # Use optimized settings for receipt extraction
            generation_kwargs = {
                **inputs,
                "max_new_tokens": self.config.max_tokens,
                "do_sample": self.config.do_sample,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "use_cache": True,
            }

            # Only add sampling parameters if sampling is enabled
            if self.config.do_sample:
                generation_kwargs.update(
                    {
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "top_k": self.config.top_k,
                    }
                )

            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)

            # Decode response - extract only the new tokens
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )

            # Clean response from repetitive text and common artifacts
            response = self._clean_response(response)

            inference_time = time.time() - start_time
            self.logger.info(f"Inference completed in {inference_time:.2f}s")

            return response.strip()

        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return f"Error: {str(e)}"

    def classify_document(self, image_path: str) -> dict[str, Any]:
        """Classify document type using the model.

        Args:
            image_path: Path to image file

        Returns:
            Classification result dictionary
        """
        # Use the improved classification prompt from prompts.yaml
        try:
            from ..config import PromptManager

            prompt_manager = PromptManager()
            classification_prompt = prompt_manager.get_prompt(
                "document_classification_prompt"
            )
        except Exception:
            # Fallback to embedded prompt if PromptManager fails
            classification_prompt = """<|image|>Analyze document structure and format. Classify based on layout patterns:

- fuel_receipt: Contains fuel quantities (L, litres), price per unit
- tax_invoice: Formal invoice layout, tax calculations
- receipt: Product lists, subtotals, retail format
- bank_statement: Account numbers, transaction records
- unknown: Cannot determine format

Output document type only."""

        try:
            response = self.predict(image_path, classification_prompt)
            response_lower = response.lower()

            # Parse classification response with improved fuel detection
            # First check OCR content for fuel indicators (override classification if needed)
            response_text = response.lower()

            # Look for fuel indicators in the actual OCR text
            fuel_indicators = [
                "13ulp",
                "ulp",
                "unleaded",
                "diesel",
                "litre",
                " l ",
                ".l ",
                "price/l",
                "per litre",
                "fuel",
            ]
            has_fuel_content = any(
                indicator in response_text for indicator in fuel_indicators
            )

            # Look for quantity patterns that indicate fuel
            import re

            fuel_quantity_pattern = r"\d+\.\d{2,3}\s*l\b|\d+\s*litre"
            has_fuel_quantity = bool(re.search(fuel_quantity_pattern, response_text))

            # Look for bank statement indicators in the actual OCR text
            bank_indicators = [
                "account",
                "balance",
                "transaction",
                "deposit",
                "withdrawal",
                "bsb",
                "opening balance",
                "closing balance",
                "statement period",
                "account number",
                "sort code",
                "debit",
                "credit",
                "available balance",
                "current balance",
            ]
            has_bank_content = any(
                indicator in response_text for indicator in bank_indicators
            )

            # Look for account number patterns (Australian BSB + Account format)
            bank_account_pattern = (
                r"\d{3}-\d{3}\s+\d{4,10}|\bBSB\b|\baccount\s+number\b"
            )
            has_bank_account = bool(
                re.search(bank_account_pattern, response_text, re.IGNORECASE)
            )

            if "fuel_receipt" in response_lower or "fuel receipt" in response_lower:
                doc_type = "fuel_receipt"
                confidence = 0.90
            elif has_fuel_content or has_fuel_quantity:
                # Override other classifications if we see clear fuel indicators
                doc_type = "fuel_receipt"
                confidence = 0.95
                self.logger.info(
                    "Overriding classification to fuel_receipt based on content indicators"
                )
            elif "fuel" in response_lower or "petrol" in response_lower:
                doc_type = "fuel_receipt"
                confidence = 0.85
            elif "tax_invoice" in response_lower or "tax invoice" in response_lower:
                doc_type = "tax_invoice"
                confidence = 0.85
            elif "tax" in response_lower and "invoice" in response_lower:
                doc_type = "tax_invoice"
                confidence = 0.80
            elif (
                "bank_statement" in response_lower or "bank statement" in response_lower
            ):
                doc_type = "bank_statement"
                confidence = 0.90
            elif has_bank_content or has_bank_account:
                # Override other classifications if we see clear bank indicators
                doc_type = "bank_statement"
                confidence = 0.95
                self.logger.info(
                    "Overriding classification to bank_statement based on content indicators"
                )
            elif "bank" in response_lower:
                doc_type = "bank_statement"
                confidence = 0.75
            elif "receipt" in response_lower:
                doc_type = "receipt"
                confidence = 0.75
            elif "invoice" in response_lower:
                doc_type = "tax_invoice"  # Default invoices to tax_invoice
                confidence = 0.70
            else:
                doc_type = "unknown"
                confidence = 0.50

            return {
                "document_type": doc_type,
                "confidence": confidence,
                "classification_response": response,
                "is_business_document": doc_type
                in [
                    "receipt",
                    "tax_invoice",
                    "fuel_receipt",
                    "bank_statement",
                    "invoice",
                ]
                and confidence > 0.7,
            }

        except Exception as e:
            self.logger.error(f"Document classification failed: {e}")
            return {
                "document_type": "unknown",
                "confidence": 0.0,
                "classification_response": f"Error: {str(e)}",
                "is_business_document": False,
            }

    def batch_predict(self, image_paths: list[str], prompts: list[str]) -> list[str]:
        """Process multiple images in batch.

        Args:
            image_paths: List of image paths
            prompts: List of prompts (one per image)

        Returns:
            List of prediction responses
        """
        if len(image_paths) != len(prompts):
            raise ValueError("Number of images and prompts must match")

        results = []
        total_images = len(image_paths)

        self.logger.info(f"Starting batch processing of {total_images} images")

        for i, (image_path, prompt) in enumerate(
            zip(image_paths, prompts, strict=False), 1
        ):
            self.logger.info(
                f"Processing image {i}/{total_images}: {Path(image_path).name}"
            )

            try:
                result = self.predict(image_path, prompt)
                results.append(result)

            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                results.append(f"Error: {str(e)}")

            # Memory cleanup between images if enabled
            if self.config.memory_cleanup_enabled and i < total_images:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if self.config.memory_cleanup_delay > 0:
                    time.sleep(self.config.memory_cleanup_delay)

        self.logger.info(f"Batch processing completed: {len(results)} results")
        return results

    def classify_document_modern(self, image_path: str) -> dict[str, Any]:
        """Classify document using modern registry architecture.

        Args:
            image_path: Path to image file

        Returns:
            Classification result dictionary
        """
        try:
            from ..extraction.extraction_engine import DocumentExtractionEngine

            # Get OCR text first using the model
            classification_prompt = """<|image|>Read all visible text from this document for classification purposes."""
            ocr_response = self.predict(image_path, classification_prompt)

            # Use modern classification
            engine = DocumentExtractionEngine()
            classification_result = engine.classify_document(ocr_response)

            # Convert to legacy format for compatibility
            return {
                "document_type": classification_result.document_type,
                "confidence": classification_result.confidence,
                "classification_response": classification_result.classification_response,
                "is_business_document": classification_result.is_business_document,
                "indicators_found": classification_result.indicators_found,
            }

        except Exception as e:
            self.logger.error(f"Modern document classification failed: {e}")
            # Fallback to legacy classification
            return self.classify_document(image_path)
