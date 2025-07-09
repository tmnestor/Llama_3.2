"""Meal receipt document type handler."""

import re
from typing import Any, Dict, List, Pattern

from .document_handlers import DocumentPattern, DocumentTypeHandler, ExtractionResult


class MealReceiptHandler(DocumentTypeHandler):
    """Handler for meal receipt documents."""

    @property
    def document_type(self) -> str:
        return "meal_receipt"

    @property
    def display_name(self) -> str:
        return "Meal Receipt"

    def get_classification_indicators(self) -> List[str]:
        """Get text indicators for meal receipts."""
        return [
            "restaurant",
            "cafe",
            "catering",
            "lunch",
            "dinner",
            "meal",
            "food",
            "kitchen",
            "bistro",
            "diner",
            "eatery",
            "grill",
            "pub",
            "bar",
            "hotel",
            "resort",
            "buffet",
            "takeaway",
            "delivery",
            "uber eats",
            "menulog",
            "doordash",
            "coffee",
            "breakfast",
            "brunch",
            "dining",
            "served",
            "table",
            "order",
            "menu",
            "waiter",
            "service charge",
            "tip",
        ]

    def get_classification_patterns(self) -> List[Pattern]:
        """Get regex patterns for meal receipt classification."""
        return [
            re.compile(r"restaurant|cafe|catering", re.IGNORECASE),
            re.compile(r"lunch|dinner|meal|food", re.IGNORECASE),
            re.compile(r"table\s+\d+", re.IGNORECASE),  # Table number
            re.compile(r"order\s+\d+", re.IGNORECASE),  # Order number
            re.compile(r"waiter|server|service", re.IGNORECASE),
            re.compile(r"kitchen|chef|cook", re.IGNORECASE),
            re.compile(r"menu|dish|course", re.IGNORECASE),
            re.compile(r"uber\s+eats|menulog|doordash", re.IGNORECASE),
            re.compile(r"coffee|beverage|drink", re.IGNORECASE),
            re.compile(r"breakfast|brunch|dining", re.IGNORECASE),
        ]

    def get_prompt_name(self) -> str:
        """Get prompt name for meal receipts."""
        return "meal_receipt_extraction_prompt"

    def get_field_patterns(self) -> List[DocumentPattern]:
        """Get field extraction patterns for meal receipts."""
        return [
            DocumentPattern(
                pattern=r"DATE:\s*([^\n\r]+)",
                field_name="DATE",
                field_type="date",
                required=True,
            ),
            DocumentPattern(
                pattern=r"RESTAURANT:\s*([^\n\r]+)",
                field_name="RESTAURANT",
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
                pattern=r"ADDRESS:\s*([^\n\r]+)",
                field_name="ADDRESS",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"TOTAL:\s*([^\n\r]+)",
                field_name="TOTAL",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"GST:\s*([^\n\r]+)",
                field_name="GST",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"ITEMS:\s*([^\n\r]+)",
                field_name="ITEMS",
                field_type="list",
                required=True,
            ),
            DocumentPattern(
                pattern=r"QUANTITIES:\s*([^\n\r]+)",
                field_name="QUANTITIES",
                field_type="string",
                required=False,
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
                pattern=r"TABLE:\s*([^\n\r]+)",
                field_name="TABLE",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"ORDER:\s*([^\n\r]+)",
                field_name="ORDER",
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
            "supplier_name": ["RESTAURANT"],
            "total_amount": ["TOTAL"],
            "gst_amount": ["GST"],
            "invoice_date": ["DATE"],
            "supplier_abn": ["ABN"],
            "items": ["ITEMS"],
            "quantities": ["QUANTITIES"],
            "prices": ["PRICES"],
            "payment_method": ["PAYMENT_METHOD"],
            "receipt_number": ["RECEIPT"],
            "invoice_number": ["RECEIPT", "ORDER"],
            # Meal-specific fields
            "restaurant_name": ["RESTAURANT"],
            "meal_items": ["ITEMS"],
            "meal_date": ["DATE"],
            "table_number": ["TABLE"],
            "order_number": ["ORDER"],
            "venue_address": ["ADDRESS"],
            "dining_type": ["DINING_TYPE"],
        }

    def extract_fields(self, response: str) -> ExtractionResult:
        """Extract fields with fallback pattern matching.

        Args:
            response: Model response text

        Returns:
            Extraction result with fields and metadata
        """
        # First try the standard KEY-VALUE approach
        result = super().extract_fields(response)

        # If we found less than 5 meaningful fields, use fallback pattern matching
        # We expect 6+ fields for meal receipts, so <5 indicates KEY-VALUE parsing failed
        if result.field_count < 5:
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

        Args:
            response: Raw model response text

        Returns:
            Extracted fields dictionary
        """
        # Use AWK-style extractor for cleaner, more maintainable extraction
        from .awk_extractor import MealReceiptAwkExtractor

        awk_extractor = MealReceiptAwkExtractor(self.log_level)
        extracted = awk_extractor.extract_meal_fields(response)

        self.logger.debug(
            f"AWK fallback extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted
