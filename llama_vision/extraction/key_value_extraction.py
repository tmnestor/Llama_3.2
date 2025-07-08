"""KEY-VALUE format extraction for Llama-3.2-Vision package."""

import re
from typing import Any, Dict

from ..utils import setup_logging


class KeyValueExtractor:
    """Extract data from KEY-VALUE format responses following InternVL patterns."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize KEY-VALUE extractor.

        Args:
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)

    def extract(self, response: str) -> Dict[str, Any]:
        """Parse KEY-VALUE format following InternVL patterns.

        Args:
            response: Model response text

        Returns:
            Dictionary of extracted key-value pairs
        """
        self.logger.debug(f"Extracting from response: {response[:100]}...")

        extracted = {}

        # Standard KEY-VALUE patterns (enhanced for Llama responses)
        kv_patterns = [
            (r"DATE:\s*([^\n\r]+)", "DATE"),
            (r"STORE:\s*([^\n\r]+)", "STORE"),
            (r"ABN:\s*([^\n\r]+)", "ABN"),
            (r"PAYER:\s*([^\n\r]+)", "PAYER"),
            (r"TAX:\s*([^\n\r]+)", "TAX"),
            (r"GST:\s*([^\n\r]+)", "GST"),
            (r"TOTAL:\s*([^\n\r]+)", "TOTAL"),
            (r"SUBTOTAL:\s*([^\n\r]+)", "SUBTOTAL"),
            (r"RECEIPT:\s*([^\n\r]+)", "RECEIPT"),
            (r"INVOICE_NUMBER:\s*([^\n\r]+)", "INVOICE_NUMBER"),
            (r"PAYMENT_METHOD:\s*([^\n\r]+)", "PAYMENT_METHOD"),
            # Enhanced patterns for raw text extraction
            (r"(?:TOTAL|Total)[\s:]*\$?(\d+\.?\d*)", "EXTRACTED_TOTAL"),
            (r"(?:TAX|GST|tax)[\s:]*\$?(\d+\.?\d*)", "EXTRACTED_TAX"),
            (r"(\d{2}/\d{2}/\d{4})", "EXTRACTED_DATE"),
            (
                r"(TARGET|WOOLWORTHS|COLES|BUNNINGS|BUNNINGS WAREHOUSE)",
                "EXTRACTED_STORE",
            ),
            (r"(?:ABN|abn)[\s:]*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})", "EXTRACTED_ABN"),
        ]

        # Extract basic key-value pairs
        for pattern, key in kv_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value and value not in ["", "N/A", "Not visible", "Not available"]:
                    extracted[key] = value

        # Extract list-based fields (products, quantities, prices)
        list_patterns = [
            (r"PRODUCTS:\s*([^\n\r]+)", "PRODUCTS"),
            (r"ITEMS:\s*([^\n\r]+)", "ITEMS"),
            (r"QUANTITIES:\s*([^\n\r]+)", "QUANTITIES"),
            (r"PRICES:\s*([^\n\r]+)", "PRICES"),
        ]

        for pattern, key in list_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value and "|" in value:
                    # Split by pipe separator
                    items = [item.strip() for item in value.split("|") if item.strip()]
                    if items:
                        extracted[key] = items
                elif value:
                    # Single item
                    extracted[key] = [value]

        # Validate and clean extracted data
        cleaned = self._clean_extracted_data(extracted)

        self.logger.info(f"Extracted {len(cleaned)} fields from KEY-VALUE response")
        if cleaned:
            self.logger.debug(f"Extracted fields: {list(cleaned.keys())}")
        else:
            self.logger.warning("No KEY-VALUE fields extracted from response")
            self.logger.debug(f"Response snippet: {response[:200]}...")

        return cleaned

    def _clean_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate extracted data.

        Args:
            data: Raw extracted data

        Returns:
            Cleaned data dictionary
        """
        cleaned = {}

        for key, value in data.items():
            if isinstance(value, str):
                # Clean string values
                cleaned_value = value.strip()

                # Remove common prefixes/suffixes
                prefixes_to_remove = ["$", "AUD", "AU$"]
                for prefix in prefixes_to_remove:
                    if cleaned_value.startswith(prefix) and key in [
                        "TOTAL",
                        "TAX",
                        "GST",
                        "SUBTOTAL",
                    ]:
                        # Keep currency symbol for amounts
                        break

                if cleaned_value:
                    cleaned[key] = cleaned_value

            elif isinstance(value, list):
                # Clean list values
                cleaned_list = [item.strip() for item in value if item and item.strip()]
                if cleaned_list:
                    cleaned[key] = cleaned_list

        # Add derived fields for compatibility
        cleaned = self._add_derived_fields(cleaned)

        return cleaned

    def _add_derived_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add derived fields for compatibility with different systems.

        Args:
            data: Cleaned extracted data

        Returns:
            Data with derived fields added
        """
        # Create aliases for common fields
        field_aliases = {
            "STORE": ["supplier_name", "business_name", "vendor_name"],
            "DATE": ["invoice_date", "transaction_date"],
            "TOTAL": ["total_amount"],
            "TAX": ["gst_amount", "tax_amount"],
            "GST": ["gst_amount", "tax_amount"],
            "ABN": ["supplier_abn", "business_abn"],
            "PAYER": ["payer_name", "customer_name"],
            "PAYMENT_METHOD": ["payment_method"],
            "PRODUCTS": ["items"],
            "ITEMS": ["products"],
        }

        for primary_key, aliases in field_aliases.items():
            if primary_key in data:
                for alias in aliases:
                    if alias not in data:
                        data[alias] = data[primary_key]

        # Normalize amounts (remove currency symbols for numeric fields)
        amount_fields = ["total_amount", "gst_amount", "tax_amount"]
        for field in amount_fields:
            if field in data and isinstance(data[field], str):
                # Extract numeric value
                numeric_match = re.search(
                    r"[\d,]+\.?\d*", data[field].replace("$", "").replace(",", "")
                )
                if numeric_match:
                    data[f"{field}_numeric"] = numeric_match.group(0)

        return data

    def extract_with_validation(self, response: str) -> Dict[str, Any]:
        """Extract and validate KEY-VALUE data with quality scoring.

        Args:
            response: Model response text

        Returns:
            Dictionary with extracted data and quality metrics
        """
        extracted_data = self.extract(response)

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(extracted_data)

        return {
            "extracted_data": extracted_data,
            "quality_metrics": quality_metrics,
            "extraction_method": "KEY_VALUE",
            "field_count": len(extracted_data),
        }

    def _calculate_quality_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for extracted data.

        Args:
            data: Extracted data dictionary

        Returns:
            Quality metrics dictionary
        """
        # Check for essential business fields
        essential_fields = ["STORE", "DATE", "TOTAL"]
        has_essential = sum(1 for field in essential_fields if field in data)

        # Check for Australian compliance fields
        compliance_fields = ["ABN", "GST", "TAX"]
        has_compliance = sum(1 for field in compliance_fields if field in data)

        # Check for product details
        product_fields = ["PRODUCTS", "ITEMS", "QUANTITIES", "PRICES"]
        has_products = any(field in data for field in product_fields)

        # Calculate overall quality score
        quality_score = (
            (has_essential / len(essential_fields)) * 0.5  # 50% for essential fields
            + (has_compliance / len(compliance_fields)) * 0.3  # 30% for compliance
            + (1.0 if has_products else 0.0) * 0.2  # 20% for product details
        )

        return {
            "quality_score": quality_score,
            "has_essential_fields": has_essential == len(essential_fields),
            "has_compliance_fields": has_compliance > 0,
            "has_product_details": has_products,
            "field_completeness": len(data) / 10,  # Assume 10 fields is complete
            "extraction_confidence": min(
                quality_score * 1.2, 1.0
            ),  # Boost confidence slightly
        }
