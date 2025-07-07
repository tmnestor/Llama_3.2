"""JSON format extraction for Llama-3.2-Vision package (legacy support)."""

import json
import re
from typing import Any, Dict, Optional

from ..utils import setup_logging


class JSONExtractor:
    """Extract data from JSON format responses (legacy support)."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize JSON extractor.

        Args:
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)

    def extract(self, response: str) -> Dict[str, Any]:
        """Extract JSON data from response with error handling.

        Args:
            response: Model response text

        Returns:
            Extracted JSON data or empty dict if parsing fails
        """
        self.logger.debug(f"Extracting JSON from response: {response[:100]}...")

        # Try direct JSON parsing first
        try:
            data = json.loads(response)
            self.logger.info("Successfully parsed direct JSON response")
            return data
        except json.JSONDecodeError:
            pass

        # Try to find JSON within the response
        json_match = self._find_json_in_text(response)
        if json_match:
            try:
                data = json.loads(json_match)
                self.logger.info("Successfully extracted JSON from response text")
                return data
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON parsing failed: {e}")

        # Fallback to KEY-VALUE parsing if JSON fails
        self.logger.info("JSON parsing failed, falling back to KEY-VALUE extraction")
        return self._fallback_key_value_parsing(response)

    def _find_json_in_text(self, text: str) -> Optional[str]:
        """Find JSON object within text.

        Args:
            text: Text to search for JSON

        Returns:
            JSON string if found, None otherwise
        """
        # Look for JSON object patterns
        json_patterns = [
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Nested objects
            r"\{[^{}]+\}",  # Simple objects
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                # Clean up the match
                cleaned = match.strip()
                if cleaned.startswith("{") and cleaned.endswith("}"):
                    # Check if it looks like valid JSON
                    if '"' in cleaned and ":" in cleaned:
                        return cleaned

        return None

    def _fallback_key_value_parsing(self, response: str) -> Dict[str, Any]:
        """Fallback to KEY-VALUE parsing when JSON fails.

        Args:
            response: Response text

        Returns:
            Parsed key-value data
        """
        # Import here to avoid circular imports
        from .key_value_extraction import KeyValueExtractor

        kv_extractor = KeyValueExtractor(self.logger.level)
        return kv_extractor.extract(response)

    def clean_json_response(self, response: str) -> str:
        """Clean response text to improve JSON parsing.

        Args:
            response: Raw response text

        Returns:
            Cleaned response text
        """
        # Remove common prefixes/suffixes that break JSON
        cleaned = response.strip()

        # Remove markdown code blocks
        cleaned = re.sub(r"```json\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        # Remove common prefixes
        prefixes_to_remove = [
            "Here's the extracted information:",
            "The extracted data is:",
            "Based on the receipt:",
            "Here is the JSON:",
        ]

        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()

        return cleaned

    def validate_json_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize JSON structure.

        Args:
            data: Parsed JSON data

        Returns:
            Validation results
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "normalized_data": data.copy(),
        }

        # Check for required fields
        required_fields = ["supplier_name", "invoice_date", "total_amount"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            validation["errors"].append(f"Missing required fields: {missing_fields}")
            validation["is_valid"] = False

        # Normalize field names
        field_mappings = {
            "store_name": "supplier_name",
            "business_name": "supplier_name",
            "date": "invoice_date",
            "transaction_date": "invoice_date",
            "total": "total_amount",
            "amount": "total_amount",
            "tax": "gst_amount",
            "gst": "gst_amount",
        }

        for old_field, new_field in field_mappings.items():
            if old_field in data and new_field not in data:
                validation["normalized_data"][new_field] = data[old_field]
                validation["warnings"].append(f"Mapped {old_field} to {new_field}")

        return validation
