"""Fuel receipt document type handler."""

import re
from typing import Any, Dict, List, Pattern

from .document_handlers import DocumentPattern, DocumentTypeHandler, ExtractionResult


class FuelReceiptHandler(DocumentTypeHandler):
    """Handler for fuel receipt documents."""

    @property
    def document_type(self) -> str:
        return "fuel_receipt"

    @property
    def display_name(self) -> str:
        return "Fuel Receipt"

    def get_classification_indicators(self) -> List[str]:
        """Get text indicators for fuel receipts."""
        return [
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
            "petrol",
            "costco",
            "shell",
            "bp",
            "coles express",
            "7-eleven",
        ]

    def get_classification_patterns(self) -> List[Pattern]:
        """Get regex patterns for fuel receipt classification."""
        return [
            re.compile(r"\d+\.\d{2,3}\s*l\b", re.IGNORECASE),  # Fuel quantity pattern
            re.compile(r"\d{3}/l", re.IGNORECASE),  # Price per litre in cents
            re.compile(r"\$\d+\.\d{2}/l", re.IGNORECASE),  # Price per litre in dollars
            re.compile(r"(13ulp|u91|u95|u98|e10)", re.IGNORECASE),  # Fuel type codes
        ]

    def get_prompt_name(self) -> str:
        """Get prompt name for fuel receipts."""
        return "fuel_receipt_extraction_prompt"

    def get_field_patterns(self) -> List[DocumentPattern]:
        """Get field extraction patterns for fuel receipts."""
        return [
            DocumentPattern(
                pattern=r"DATE:\s*([^\n\r]+)",
                field_name="DATE",
                field_type="date",
                required=True,
            ),
            DocumentPattern(
                pattern=r"STORE:\s*([^\n\r]+)",
                field_name="STORE",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"ABN:\s*([^\n\r]+)",
                field_name="ABN",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"PAYER:\s*([^\n\r]+)",
                field_name="PAYER",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"TAX:\s*([^\n\r]+)",
                field_name="TAX",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"TOTAL:\s*([^\n\r]+)",
                field_name="TOTAL",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"PRODUCTS:\s*([^\n\r]+)",
                field_name="PRODUCTS",
                field_type="list",
                required=True,
            ),
            DocumentPattern(
                pattern=r"QUANTITIES:\s*([^\n\r]+)",
                field_name="QUANTITIES",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"PRICES:\s*([^\n\r]+)",
                field_name="PRICES",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"PAYMENT_METHOD:\s*([^\n\r]+)",
                field_name="PAYMENT_METHOD",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"RECEIPT:\s*([^\n\r]+)",
                field_name="RECEIPT",
                field_type="string",
                required=False,
            ),
        ]

    def get_field_mappings(self) -> Dict[str, List[str]]:
        """Get field mappings for standardization."""
        return {
            # Standard compliance fields
            "supplier_name": ["STORE"],
            "total_amount": ["TOTAL"],
            "gst_amount": ["TAX"],
            "invoice_date": ["DATE"],
            "supplier_abn": ["ABN"],
            "items": ["PRODUCTS"],
            "quantities": ["QUANTITIES"],
            "prices": ["PRICES"],
            "payment_method": ["PAYMENT_METHOD"],
            "receipt_number": ["RECEIPT"],
            "invoice_number": ["RECEIPT"],
            # Fuel-specific fields
            "fuel_type": ["PRODUCTS"],
            "fuel_quantity": ["QUANTITIES"],
            "price_per_litre": ["PRICES"],
            "fuel_station": ["STORE"],
            "member_number": ["PAYER"],
        }

    def extract_fields(self, response: str) -> ExtractionResult:
        """Extract fields with fallback pattern matching like TaxAuthorityParser.

        Args:
            response: Model response text

        Returns:
            Extraction result with fields and metadata
        """
        # First try the standard KEY-VALUE approach
        result = super().extract_fields(response)

        # If we found less than 6 meaningful fields, use fallback pattern matching
        # We expect 8+ fields for fuel receipts, so <6 indicates KEY-VALUE parsing failed
        if result.field_count < 6:
            self.logger.debug(
                f"KEY-VALUE parsing found only {result.field_count} fields, trying fallback pattern matching..."
            )
            fallback_fields = self._extract_from_raw_text(response)

            # Merge fallback fields with any successful KEY-VALUE fields
            combined_fields = result.fields.copy()
            combined_fields.update(fallback_fields)

            # Apply field mappings
            mappings = self.get_field_mappings()
            normalized = self._apply_field_mappings(combined_fields, mappings)

            # Recalculate compliance score
            required_patterns = [p for p in self.get_field_patterns() if p.required]
            required_found = sum(
                1 for p in required_patterns if p.field_name in normalized
            )
            compliance_score = (
                required_found / len(required_patterns) if required_patterns else 1.0
            )

            return ExtractionResult(
                fields=normalized,
                extraction_method=f"{self.document_type}_handler_with_fallback",
                compliance_score=compliance_score,
                field_count=len(normalized),
            )
        else:
            self.logger.debug(
                f"KEY-VALUE parsing successful with {result.field_count} fields, skipping fallback"
            )
            return result

    def _extract_from_raw_text(self, response: str) -> Dict[str, Any]:
        """Extract fields from raw OCR text using AWK-style processing.

        Much more maintainable than complex regex patterns.

        Args:
            response: Raw model response text

        Returns:
            Extracted fields dictionary
        """
        # Use AWK-style extractor for cleaner, more maintainable extraction
        from .awk_extractor import FuelReceiptAwkExtractor

        awk_extractor = FuelReceiptAwkExtractor(self.log_level)
        extracted = awk_extractor.extract_fuel_fields(response)

        self.logger.debug(
            f"AWK fallback extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted
