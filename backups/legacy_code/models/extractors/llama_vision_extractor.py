"""
Zero-shot receipt information extractor using Llama-3.2-1B-Vision.

This module implements ab initio (from first principles) receipt information extraction
without requiring any receipt-specific training, using only the pre-trained capabilities
of Llama-Vision with specialized prompt engineering.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers import MllamaForConditionalGeneration


class LlamaVisionExtractor:
    """Zero-shot receipt information extractor using Llama-Vision."""

    def __init__(
        self,
        model_path: str = "/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision",
        device: str = "cpu",  # Default to CPU for stability
        use_8bit: bool = False,
        max_new_tokens: int = 256,
    ):
        """Initialize the Llama-Vision extractor.

        Args:
            model_path: Path to Llama-Vision model
            device: Device to run inference on
            use_8bit: Whether to use 8-bit quantization
            max_new_tokens: Maximum number of tokens to generate
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens

        # Check if model path exists
        if not Path(model_path).exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        # Load model with appropriate settings
        self.logger.info(f"Loading Llama-Vision model from {model_path}")

        # Configure quantization if needed
        quantization_config = None
        if use_8bit:
            # Check if we're on Mac M1 where bitsandbytes doesn't work properly
            import platform

            if platform.machine() == "arm64" and platform.system() == "Darwin":
                self.logger.warning(
                    "8-bit quantization not supported on Mac M1, using float16 instead"
                )
                quantization_config = None
            else:
                try:
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_skip_modules=["vision_model", "vision_tower"],
                    )
                    self.logger.info("Using 8-bit quantization for memory efficiency")
                except Exception as e:
                    self.logger.warning(f"8-bit quantization not available: {e}")
                    quantization_config = None

        # Load processor, tokenizer and model for official Llama-Vision
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        # Use appropriate dtype for device
        model_dtype = torch.float16  # Use float16 for better performance
        self.logger.info(f"Using dtype: {model_dtype} for device: {device}")

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            device_map=None,  # Load on CPU first
            torch_dtype=model_dtype,  # Use float32 on MPS, float16 elsewhere
            quantization_config=quantization_config,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,  # Load model with minimal CPU memory
            attn_implementation="eager",  # Use eager attention for memory efficiency
        )

        # Simple device handling - prioritize CPU for vision stability
        if device == "cpu":
            self.device = "cpu"
            self.logger.info("🖥️ Using CPU for stable vision processing")
        elif device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.device = "cuda"
            self.logger.info("🚀 Using CUDA for vision processing")
        else:
            self.device = "cpu"
            self.logger.info("🖥️ Defaulting to CPU for stable vision processing")

        self.logger.info(f"🎯 Final device: {self.device}")

        # Set model to evaluation mode
        self.model.eval()

        # Apply PyTorch dynamic quantization if requested and on CPU
        if use_8bit and self.device == "cpu":
            self.logger.info("Setting up PyTorch quantization backend for Mac M1")
            # Set quantization backend for ARM/Mac M1
            torch.backends.quantized.engine = "qnnpack"

            self.logger.info("Applying PyTorch dynamic quantization for CPU inference")
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},  # Quantize Linear layers
                    dtype=torch.qint8,
                )
                self.logger.info("Dynamic quantization applied successfully")
            except Exception as e:
                self.logger.warning(f"Dynamic quantization failed: {e}")
                self.logger.info("Continuing with float16 precision")

        # Set pad token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.info("Llama-Vision model loaded successfully")

    def _preprocess_image(self, image_path: str | Path) -> Image.Image:
        """Preprocess image for Llama-Vision.

        Args:
            image_path: Path to receipt image

        Returns:
            Preprocessed PIL Image
        """
        # Load image
        if isinstance(image_path, str):
            image_path = Path(image_path)

        if not image_path.exists():
            raise ValueError(f"Image does not exist: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # Llama-Vision uses PIL Image directly in most implementations
        # Additional preprocessing may be handled by the processor
        return image

    def _get_extraction_prompt(self, field: str | None = None) -> str:
        """Get appropriate extraction prompt.

        Args:
            field: Specific field to extract, or None for all fields

        Returns:
            Formatted extraction prompt
        """
        # Optimized prompts for the 1B model - simpler and more direct

        # Field-specific prompts (optimized for small model)
        field_prompts = {
            "store_name": "What is the store name on this receipt?",
            "date": "What is the date on this receipt? Use format YYYY-MM-DD.",
            "time": "What is the time on this receipt? Use format HH:MM.",
            "total": "What is the total amount on this receipt?",
            "total_amount": "What is the total amount on this receipt?",
            "payment_method": "What payment method was used on this receipt?",
            "receipt_id": "What is the receipt number or ID?",
            "items": "List all items and prices from this receipt.",
        }

        # Full extraction prompt (simplified for 1B model)
        full_extraction_prompt = """Look at this receipt image and extract the following information in JSON format:

{
  "store_name": "",
  "date": "YYYY-MM-DD", 
  "time": "HH:MM",
  "total_amount": "",
  "payment_method": "",
  "receipt_id": "",
  "items": [{"item_name": "", "price": ""}],
  "tax_info": ""
}

Fill in the actual values from the receipt. If a field is not visible, use null."""

        if field is not None and field in field_prompts:
            return field_prompts[field]
        else:
            return full_extraction_prompt

    def extract_field(self, image_path: str | Path, field: str) -> Any:
        """Extract a specific field from receipt.

        Args:
            image_path: Path to receipt image
            field: Field to extract

        Returns:
            Extracted field value
        """
        # Process image
        image = self._preprocess_image(image_path)

        # Create prompt
        prompt = self._get_extraction_prompt(field)

        # Generate response
        try:
            # For Llama-Vision, we need to handle both image and text inputs
            response = self._generate_response(prompt, image)

            # Parse and return the field value
            return self._parse_field(response, field)

        except Exception as e:
            self.logger.error(f"Failed to extract field {field} from {image_path}: {e}")
            return None

    def extract_all_fields(self, image_path: str | Path) -> dict[str, Any]:
        """Extract all receipt information fields.

        Args:
            image_path: Path to receipt image

        Returns:
            Dictionary with all extracted fields
        """
        # Process image
        image = self._preprocess_image(image_path)

        # Create prompt for full extraction
        prompt = self._get_extraction_prompt()

        try:
            # Generate response
            response = self._generate_response(prompt, image)

            # Parse the JSON response
            return self._parse_json_response(response)

        except Exception as e:
            self.logger.error(f"Failed to extract fields from {image_path}: {e}")
            return self._get_empty_result()

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

        # Set 12 hour timeout for generation (11B vision model on CPU can be very slow)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(43200)

        # Progress tracking
        generation_start_time = time.time()

        try:
            # Use the official MllamaForConditionalGeneration model with processor
            self.logger.info("🔄 Step 1: Preparing messages...")
            step_start = time.time()

            # Use the EXACT pattern from working examples
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]
            self.logger.info(f"⏱️  Step 1 completed in {time.time() - step_start:.2f}s")

            # Apply chat template EXACTLY like working examples
            self.logger.info("🔄 Step 2: Applying chat template...")
            step_start = time.time()
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            self.logger.info(f"⏱️  Step 2 completed in {time.time() - step_start:.2f}s")

            # Process EXACTLY like working examples - image first, then text
            self.logger.info("🔄 Step 3: Processing inputs (image + text)...")
            step_start = time.time()
            inputs = self.processor(
                images=image,
                text=input_text,
                return_tensors="pt",
            ).to(self.device)
            self.logger.info(f"⏱️  Step 3 completed in {time.time() - step_start:.2f}s")

            # Debug: print input keys
            self.logger.info(f"📊 Available input keys: {list(inputs.keys())}")
            self.logger.info(f"📊 Input shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in inputs.items()]}")

            # Try the direct approach as mentioned in Meta blog - "drop-in replacement"
            # The model should handle cross-attention internally
            self.logger.info("🔄 Step 4: Starting model generation...")
            self.logger.info("⏰ Progress updates will be shown every 5 minutes...")
            step_start = time.time()

            # Progress monitoring with proper thread management
            import threading

            # Thread-safe progress monitoring
            progress_stop_event = threading.Event()
            progress_thread = None

            def log_progress():
                elapsed = time.time() - step_start
                elapsed_total = time.time() - generation_start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)

                total_hours = int(elapsed_total // 3600)
                total_minutes = int((elapsed_total % 3600) // 60)

                progress_pct = (elapsed_total / 43200) * 100  # Out of 12 hours

                self.logger.info(f"⏰ Generation progress: {hours:02d}:{minutes:02d}:{seconds:02d} "
                               f"(Total: {total_hours:02d}:{total_minutes:02d}, {progress_pct:.1f}% of 12h timeout)")

            def progress_monitor():
                while not progress_stop_event.is_set():
                    if progress_stop_event.wait(300):  # Wait 5 minutes or until stopped
                        break
                    try:
                        if not progress_stop_event.is_set():
                            log_progress()
                    except Exception:
                        break  # Exit if error occurs

            # Only start progress monitoring for long operations
            if self.max_new_tokens > 50:  # Only for significant generations
                progress_thread = threading.Thread(target=progress_monitor, daemon=True)
                progress_thread.start()

            with torch.no_grad():
                # Standard generation on current device
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            generation_time = time.time() - step_start

            # Final progress log
            hours = int(generation_time // 3600)
            minutes = int((generation_time % 3600) // 60)
            seconds = int(generation_time % 60)
            self.logger.info(f"⏱️  Step 4 (GENERATION) completed in {generation_time:.2f}s ({hours:02d}:{minutes:02d}:{seconds:02d})")

            # Stop progress monitoring
            if progress_thread is not None:
                progress_stop_event.set()  # Signal thread to stop
                progress_thread.join(timeout=1.0)  # Wait up to 1 second for cleanup

            # Extract only the new tokens (response)
            self.logger.info("🔄 Step 5: Decoding response...")
            step_start = time.time()
            self.logger.info(f"📊 Input IDs shape: {inputs['input_ids'].shape}")
            self.logger.info(f"📊 Output shape: {outputs[0].shape}")

            # Decode only the new generated tokens
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )
            self.logger.info(f"⏱️  Step 5 completed in {time.time() - step_start:.2f}s")
            self.logger.info(f"📄 Generated response: '{response}'")

        except TimeoutError:
            self.logger.error("Generation timed out after 60 minutes")
            return ""
        except Exception as e:
            self.logger.error(f"Error in _generate_response: {e}")
            self.logger.error(f"Exception type: {type(e)}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return ""
        finally:
            # Cancel the timeout
            signal.alarm(0)

            # Ensure progress monitoring thread is stopped
            try:
                if 'progress_thread' in locals() and progress_thread is not None:
                    if 'progress_stop_event' in locals():
                        progress_stop_event.set()
                    progress_thread.join(timeout=0.5)
            except Exception:
                pass  # Ignore cleanup errors

        return response

    def _parse_field(self, response: str, field: str) -> Any:
        """Parse field-specific response.

        Args:
            response: Raw response from model
            field: Field type

        Returns:
            Parsed and validated field value
        """
        if field == "date":
            # Extract date in YYYY-MM-DD format
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", response)
            if date_match:
                return date_match.group(0)

            # Try other date formats and convert
            patterns = [
                # MM/DD/YYYY
                (
                    r"(\d{1,2})/(\d{1,2})/(\d{4})",
                    lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}",
                ),
                # DD/MM/YYYY
                (
                    r"(\d{1,2})/(\d{1,2})/(\d{4})",
                    lambda m: f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}",
                ),
                # MM-DD-YYYY
                (
                    r"(\d{1,2})-(\d{1,2})-(\d{4})",
                    lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}",
                ),
            ]

            for pattern, formatter in patterns:
                match = re.search(pattern, response)
                if match:
                    return formatter(match)

            return None

        elif field == "total":
            # Extract amount with currency
            amount_match = re.search(r"(\$|€|£|\d)[\d\s,.]+", response)
            if amount_match:
                return amount_match.group(0).strip()
            return None

        elif field == "items":
            # Try to extract JSON array from response
            try:
                # Find JSON-like structure in response
                json_match = re.search(r"\[.*\]", response, re.DOTALL)
                if json_match:
                    items_json = json_match.group(0)
                    return json.loads(items_json)
                return []
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse items JSON: {response}")
                return []

        elif field == "store_name":
            # Return the first line or full response if short
            if "\n" in response:
                return response.split("\n")[0].strip()
            return response.strip()

        # Default case: return the response as is
        return response.strip() if response else None

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse and validate JSON response.

        Args:
            response: Raw JSON response from model

        Returns:
            Validated and normalized extraction results
        """
        # Standard structure for output
        result = self._get_empty_result()
        result["raw_extraction"] = response  # Store raw response for debugging

        # Sanitize response to extract valid JSON
        # Sometimes model outputs extra text before/after JSON
        json_match = re.search(r"(\{.*\})", response, re.DOTALL)
        if not json_match:
            self.logger.warning(f"No valid JSON found in response: {response[:100]}...")
            return result

        json_str = json_match.group(1)

        # Parse JSON with error handling
        try:
            extracted = json.loads(json_str)

            # Map extracted fields to standard structure
            field_mappings = {
                "store_name": [
                    "store_name",
                    "store",
                    "business_name",
                    "business",
                    "merchant",
                    "merchant_name",
                ],
                "date": ["date", "date_of_purchase", "purchase_date"],
                "time": ["time", "time_of_purchase", "purchase_time"],
                "total_amount": ["total_amount", "total", "amount", "total_price"],
                "payment_method": [
                    "payment_method",
                    "payment",
                    "payment_type",
                    "method",
                ],
                "receipt_id": [
                    "receipt_id",
                    "receipt_number",
                    "id",
                    "transaction_id",
                    "receipt_no",
                ],
                "items": ["items", "line_items", "products", "purchases"],
                "tax_info": ["tax_info", "tax", "tax_amount", "tax_rate", "gst", "vat"],
                "discounts": ["discounts", "promotions", "discount_amount", "savings"],
            }

            # Fill in the result structure from extracted data
            for result_key, possible_keys in field_mappings.items():
                for key in possible_keys:
                    if key in extracted and extracted[key] is not None:
                        result[result_key] = extracted[key]
                        break

            # Apply validation to specific fields
            if result["date"]:
                # Normalize date format
                date_match = re.search(r"\d{4}-\d{1,2}-\d{1,2}", str(result["date"]))
                if date_match:
                    # Ensure proper zero-padding
                    date_parts = str(result["date"]).split("-")
                    if len(date_parts) == 3:
                        y, m, d = date_parts
                        result["date"] = f"{y}-{m.zfill(2)}-{d.zfill(2)}"

            # Normalize total amount format
            if result["total_amount"]:
                # Strip any extra text around the amount
                amount_match = re.search(
                    r"(\$|€|£|\d)[\d\s,.]+", str(result["total_amount"])
                )
                if amount_match:
                    result["total_amount"] = amount_match.group(0).strip()

            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Invalid JSON: {json_str[:100]}...")
            return result

    def _get_empty_result(self) -> dict[str, Any]:
        """Get empty result structure.

        Returns:
            Empty result dictionary with all fields set to None/empty
        """
        return {
            "store_name": None,
            "date": None,
            "time": None,
            "total_amount": None,
            "payment_method": None,
            "receipt_id": None,
            "items": [],
            "tax_info": None,
            "discounts": None,
        }
