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
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaVisionExtractor:
    """Zero-shot receipt information extractor using Llama-Vision."""
    
    def __init__(
        self,
        model_path: str = "/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision",
        device: str = "auto",
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
        
        # Configure quantization if needed (optional for 1B model)
        quantization_config = None
        if use_8bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=["vision_model", "vision_tower"]
                )
                self.logger.info("Using 8-bit quantization for memory efficiency")
            except ImportError:
                self.logger.warning("BitsAndBytesConfig not available, using default precision")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=None,  # Disable device mapping for custom models
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
            quantization_config=quantization_config,
            trust_remote_code=True,
            local_files_only=True,
        )
        
        # Move model to appropriate device after loading
        if device == "mps" and torch.backends.mps.is_available():
            self.model = self.model.to("mps")
            self.device = "mps"
        elif device == "auto":
            if torch.backends.mps.is_available():
                self.model = self.model.to("mps") 
                self.device = "mps"
            elif torch.cuda.is_available():
                self.model = self.model.to("cuda")
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Set pad token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.logger.info("Llama-Vision model loaded successfully")
    
    def _preprocess_image(self, image_path: Union[str, Path]) -> Image.Image:
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
    
    def _get_extraction_prompt(self, field: Optional[str] = None) -> str:
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
    
    def extract_field(self, image_path: Union[str, Path], field: str) -> Any:
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
    
    def extract_all_fields(self, image_path: Union[str, Path]) -> Dict[str, Any]:
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
        try:
            # Check if this is the custom Llamavision model with answer_question method
            if hasattr(self.model, 'answer_question'):
                # Use the built-in answer_question method (recommended approach)
                self.logger.debug("Using Llamavision answer_question interface")
                
                response = self.model.answer_question(
                    image=image,
                    question=prompt,
                    tokenizer=self.tokenizer,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                )
                
            elif hasattr(self.model, 'encode_image') and hasattr(self.model, 'generate'):
                # Custom Llamavision model (kadirnar/Llama-3.2-1B-Vision) - fallback method
                self.logger.debug("Using custom Llamavision interface")
                
                # Encode the image
                image_embeds = self.model.encode_image(image)
                
                # Generate response using the custom generate method
                output_ids = self.model.generate(
                    image_embeds=image_embeds,
                    prompt=prompt,
                    tokenizer=self.tokenizer,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                )
                
                # Decode the response
                response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Remove the prompt from the response if it's included
                if prompt in response:
                    response = response.replace(prompt, "").strip()
                    
            elif hasattr(self.model, 'chat'):
                # Standard chat interface (official models)
                self.logger.debug("Using standard chat interface")
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    messages=[{
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    }],
                    temperature=0.1,
                    max_new_tokens=self.max_new_tokens
                )
            else:
                # Fallback - text-only (shouldn't be used for vision models)
                self.logger.warning("No vision interface found, falling back to text-only generation")
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if prompt in response:
                    response = response.replace(prompt, "").strip()
        
        except Exception as e:
            self.logger.error(f"Error in _generate_response: {e}")
            return ""
        
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
            date_match = re.search(r'\d{4}-\d{2}-\d{2}', response)
            if date_match:
                return date_match.group(0)
            
            # Try other date formats and convert
            patterns = [
                # MM/DD/YYYY
                (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
                # DD/MM/YYYY
                (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"),
                # MM-DD-YYYY
                (r'(\d{1,2})-(\d{1,2})-(\d{4})', lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
            ]
            
            for pattern, formatter in patterns:
                match = re.search(pattern, response)
                if match:
                    return formatter(match)
            
            return None
            
        elif field == "total":
            # Extract amount with currency
            amount_match = re.search(r'(\$|€|£|\d)[\d\s,.]+', response)
            if amount_match:
                return amount_match.group(0).strip()
            return None
            
        elif field == "items":
            # Try to extract JSON array from response
            try:
                # Find JSON-like structure in response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    items_json = json_match.group(0)
                    return json.loads(items_json)
                return []
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse items JSON: {response}")
                return []
                
        elif field == "store_name":
            # Return the first line or full response if short
            if '\n' in response:
                return response.split('\n')[0].strip()
            return response.strip()
            
        # Default case: return the response as is
        return response.strip() if response else None
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
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
        json_match = re.search(r'(\{.*\})', response, re.DOTALL)
        if not json_match:
            self.logger.warning(f"No valid JSON found in response: {response[:100]}...")
            return result
            
        json_str = json_match.group(1)
        
        # Parse JSON with error handling
        try:
            extracted = json.loads(json_str)
            
            # Map extracted fields to standard structure
            field_mappings = {
                "store_name": ["store_name", "store", "business_name", "business", "merchant", "merchant_name"],
                "date": ["date", "date_of_purchase", "purchase_date"],
                "time": ["time", "time_of_purchase", "purchase_time"],
                "total_amount": ["total_amount", "total", "amount", "total_price"],
                "payment_method": ["payment_method", "payment", "payment_type", "method"],
                "receipt_id": ["receipt_id", "receipt_number", "id", "transaction_id", "receipt_no"],
                "items": ["items", "line_items", "products", "purchases"],
                "tax_info": ["tax_info", "tax", "tax_amount", "tax_rate", "gst", "vat"],
                "discounts": ["discounts", "promotions", "discount_amount", "savings"]
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
                date_match = re.search(r'\d{4}-\d{1,2}-\d{1,2}', str(result["date"]))
                if date_match:
                    # Ensure proper zero-padding
                    date_parts = str(result["date"]).split('-')
                    if len(date_parts) == 3:
                        y, m, d = date_parts
                        result["date"] = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
            
            # Normalize total amount format
            if result["total_amount"]:
                # Strip any extra text around the amount
                amount_match = re.search(r'(\$|€|£|\d)[\d\s,.]+', str(result["total_amount"]))
                if amount_match:
                    result["total_amount"] = amount_match.group(0).strip()
                    
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Invalid JSON: {json_str[:100]}...")
            return result
    
    def _get_empty_result(self) -> Dict[str, Any]:
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