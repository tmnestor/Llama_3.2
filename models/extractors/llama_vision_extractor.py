"""
Zero-shot receipt information extractor using Llama-3.2-11B-Vision.

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
        model_path: str = "/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
        device: str = "cuda",
        use_8bit: bool = True,
        max_new_tokens: int = 1024,
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
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quantization_config,
            trust_remote_code=True,
            local_files_only=True,
        )
        
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
        base_prompt = """<image>

You are a receipt information extraction assistant. Analyze the receipt in the image and extract the following information accurately.

"""
        
        # Full extraction prompt (all fields)
        full_extraction_prompt = base_prompt + """Extract ALL of the following information from the receipt:
1. Store/Business Name
2. Date of Purchase (YYYY-MM-DD format)
3. Time of Purchase (HH:MM format)
4. Total Amount (include currency)
5. Payment Method
6. Receipt Number/ID
7. Individual Items (with prices)
8. Tax Information (amount and/or percentage)
9. Discounts/Promotions (if any)

Provide your answer as a structured JSON object with these fields. For any field that cannot be found in the receipt, use null.
Format your response as a valid JSON object only, with no additional commentary or explanation. Ensure that numeric values are formatted accordingly.

{
  "store_name": "...",
  "date": "YYYY-MM-DD",
  "time": "HH:MM",
  "total_amount": "...",
  "payment_method": "...",
  "receipt_id": "...",
  "items": [{"item_name": "...", "quantity": 1, "price": "..."}],
  "tax_info": "...",
  "discounts": "..."
}"""
        
        # Field-specific prompts if needed
        field_prompts = {
            "store_name": base_prompt + "What is the store or business name on this receipt? Extract just the name.",
            "date": base_prompt + "What is the date of purchase on this receipt? Format as YYYY-MM-DD.",
            "total": base_prompt + "What is the total amount on this receipt? Include the currency symbol if visible.",
            "items": base_prompt + """Extract all individual items with their prices from this receipt.
Format your answer as a valid JSON array of objects with 'item_name', 'quantity', and 'price' fields.""",
        }
        
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
        # Handle the multimodal input for Llama-Vision
        # Note: The exact implementation may vary based on the specific Llama-Vision variant
        # This is a general approach for vision-language models
        
        with torch.no_grad():
            # For models that support direct image-text input
            if hasattr(self.model, 'chat'):
                # Use chat interface if available
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
                # Fallback to standard generation with processor
                # Note: This part may need adjustment based on the specific model implementation
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate response
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False
                )
                
                # Decode generated text
                response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                # Remove the prompt from the response
                if prompt in response:
                    response = response.replace(prompt, "").strip()
        
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