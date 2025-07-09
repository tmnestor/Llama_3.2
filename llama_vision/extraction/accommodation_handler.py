"""Accommodation document type handler."""

import re
from typing import Any, Dict, List, Pattern

from .document_handlers import DocumentPattern, DocumentTypeHandler, ExtractionResult


class AccommodationHandler(DocumentTypeHandler):
    """Handler for accommodation documents."""

    @property
    def document_type(self) -> str:
        return "accommodation"

    @property
    def display_name(self) -> str:
        return "Accommodation"

    def get_classification_indicators(self) -> List[str]:
        """Get text indicators for accommodation documents."""
        return [
            "hotel",
            "motel",
            "accommodation",
            "booking",
            "airbnb",
            "lodging",
            "resort",
            "inn",
            "hostel",
            "b&b",
            "bed and breakfast",
            "guest house",
            "apartment",
            "suite",
            "room",
            "check-in",
            "check-out",
            "checkout",
            "checkin",
            "stay",
            "night",
            "nights",
            "reservation",
            "booking.com",
            "expedia",
            "hotels.com",
            "trivago",
            "wotif",
            "agoda",
            "hilton",
            "marriott",
            "hyatt",
            "ibis",
            "novotel",
            "pullman",
            "crown",
            "intercontinental",
            "holiday inn",
            "best western",
            "accor",
        ]

    def get_classification_patterns(self) -> List[Pattern]:
        """Get regex patterns for accommodation classification."""
        return [
            re.compile(r"hotel|motel|accommodation", re.IGNORECASE),
            re.compile(r"booking|reservation", re.IGNORECASE),
            re.compile(r"check.?in|check.?out", re.IGNORECASE),
            re.compile(r"room\s+\d+", re.IGNORECASE),  # Room number
            re.compile(r"\d+\s+nights?", re.IGNORECASE),  # Number of nights
            re.compile(r"guest\s+name", re.IGNORECASE),
            re.compile(r"arrival|departure", re.IGNORECASE),
            re.compile(r"booking\.com|expedia|hotels\.com", re.IGNORECASE),
            re.compile(r"hilton|marriott|hyatt|ibis|novotel", re.IGNORECASE),
            re.compile(r"suite|apartment|lodge", re.IGNORECASE),
        ]

    def get_prompt_name(self) -> str:
        """Get prompt name for accommodation documents."""
        return "accommodation_extraction_prompt"

    def get_field_patterns(self) -> List[DocumentPattern]:
        """Get field extraction patterns for accommodation documents."""
        return [
            DocumentPattern(
                pattern=r"DATE:\s*([^\n\r]+)",
                field_name="DATE",
                field_type="date",
                required=True,
            ),
            DocumentPattern(
                pattern=r"HOTEL:\s*([^\n\r]+)",
                field_name="HOTEL",
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
                pattern=r"GUEST:\s*([^\n\r]+)",
                field_name="GUEST",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"ROOM:\s*([^\n\r]+)",
                field_name="ROOM",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"NIGHTS:\s*([^\n\r]+)",
                field_name="NIGHTS",
                field_type="number",
                required=False,
            ),
            DocumentPattern(
                pattern=r"CHECK_IN:\s*([^\n\r]+)",
                field_name="CHECK_IN",
                field_type="date",
                required=False,
            ),
            DocumentPattern(
                pattern=r"CHECK_OUT:\s*([^\n\r]+)",
                field_name="CHECK_OUT",
                field_type="date",
                required=False,
            ),
            DocumentPattern(
                pattern=r"PAYMENT_METHOD:\s*([^\n\r]+)",
                field_name="PAYMENT_METHOD",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"BOOKING:\s*([^\n\r]+)",
                field_name="BOOKING",
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
            "supplier_name": ["HOTEL"],
            "total_amount": ["TOTAL"],
            "gst_amount": ["GST"],
            "invoice_date": ["DATE"],
            "supplier_abn": ["ABN"],
            "payment_method": ["PAYMENT_METHOD"],
            "receipt_number": ["RECEIPT", "BOOKING"],
            "invoice_number": ["RECEIPT", "BOOKING"],
            # Accommodation-specific fields
            "hotel_name": ["HOTEL"],
            "guest_name": ["GUEST"],
            "room_number": ["ROOM"],
            "number_of_nights": ["NIGHTS"],
            "check_in_date": ["CHECK_IN"],
            "check_out_date": ["CHECK_OUT"],
            "booking_reference": ["BOOKING"],
            "accommodation_address": ["ADDRESS"],
            "accommodation_date": ["DATE"],
            "stay_duration": ["NIGHTS"],
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

        # If we found less than 4 meaningful fields, use fallback pattern matching
        # We expect 5+ fields for accommodation, so <4 indicates KEY-VALUE parsing failed
        if result.field_count < 4:
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
        from .awk_extractor import AccommodationAwkExtractor

        awk_extractor = AccommodationAwkExtractor(self.log_level)
        extracted = awk_extractor.extract_accommodation_fields(response)

        self.logger.debug(
            f"AWK fallback extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted
