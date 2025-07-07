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
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    # repetition_penalty removed to avoid CUDA ScatterGatherKernel error
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode response - extract only the new tokens
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )

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
        classification_prompt = """<|image|>Analyze this document and classify it as one of:
- receipt: Store/business receipt for purchases
- tax_invoice: Official tax invoice with ABN details
- fuel_receipt: Petrol/fuel station receipt
- bank_statement: Bank account statement or transaction history
- unknown: Cannot determine or not a business document

Respond with just the classification type."""

        try:
            response = self.predict(image_path, classification_prompt)
            response_lower = response.lower()

            # Parse classification response
            if "receipt" in response_lower and "fuel" not in response_lower:
                doc_type = "receipt"
                confidence = 0.85
            elif "fuel" in response_lower and "receipt" in response_lower:
                doc_type = "fuel_receipt"
                confidence = 0.80
            elif "tax" in response_lower and "invoice" in response_lower:
                doc_type = "tax_invoice"
                confidence = 0.80
            elif "invoice" in response_lower:
                doc_type = "invoice"
                confidence = 0.75
            elif "bank" in response_lower:
                doc_type = "bank_statement"
                confidence = 0.75
            else:
                doc_type = "unknown"
                confidence = 0.50

            return {
                "document_type": doc_type,
                "confidence": confidence,
                "classification_response": response,
                "is_business_document": doc_type
                in ["receipt", "tax_invoice", "fuel_receipt", "invoice"]
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
