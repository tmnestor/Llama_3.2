"""Travel document type handler."""

import re
from typing import Any, Dict, List, Pattern

from .document_handlers import DocumentPattern, DocumentTypeHandler, ExtractionResult


class TravelDocumentHandler(DocumentTypeHandler):
    """Handler for travel documents."""

    @property
    def document_type(self) -> str:
        return "travel_document"

    @property
    def display_name(self) -> str:
        return "Travel Document"

    def get_classification_indicators(self) -> List[str]:
        """Get text indicators for travel documents."""
        return [
            "qantas",
            "jetstar",
            "virgin",
            "flight",
            "airline",
            "boarding pass",
            "boarding",
            "gate",
            "seat",
            "departure",
            "arrival",
            "terminal",
            "aircraft",
            "plane",
            "airport",
            "sydney",
            "melbourne",
            "brisbane",
            "perth",
            "adelaide",
            "darwin",
            "cairns",
            "gold coast",
            "canberra",
            "hobart",
            "domestic",
            "international",
            "train",
            "railway",
            "bus",
            "coach",
            "greyhound",
            "transit",
            "transport",
            "ticket",
            "travel",
            "journey",
            "route",
            "schedule",
            "platform",
            "carriage",
            "passenger",
            "booking reference",
            "confirmation",
            "itinerary",
            "e-ticket",
            "boarding time",
            "check-in",
        ]

    def get_classification_patterns(self) -> List[Pattern]:
        """Get regex patterns for travel document classification."""
        return [
            re.compile(r"qantas|jetstar|virgin|tigerair", re.IGNORECASE),
            re.compile(r"flight|airline|boarding", re.IGNORECASE),
            re.compile(r"gate\s+[a-z]?\d+", re.IGNORECASE),  # Gate number
            re.compile(r"seat\s+\d+[a-z]", re.IGNORECASE),  # Seat number
            re.compile(r"departure|arrival", re.IGNORECASE),
            re.compile(r"sydney|melbourne|brisbane|perth", re.IGNORECASE),
            re.compile(r"terminal\s+\d+", re.IGNORECASE),
            re.compile(r"boarding\s+pass", re.IGNORECASE),
            re.compile(r"train|railway|bus|coach", re.IGNORECASE),
            re.compile(r"platform\s+\d+", re.IGNORECASE),
            re.compile(r"booking\s+reference", re.IGNORECASE),
            re.compile(r"confirmation\s+code", re.IGNORECASE),
        ]

    def get_prompt_name(self) -> str:
        """Get prompt name for travel documents."""
        return "travel_document_extraction_prompt"

    def get_field_patterns(self) -> List[DocumentPattern]:
        """Get field extraction patterns for travel documents."""
        return [
            DocumentPattern(
                pattern=r"DATE:\s*([^\n\r]+)",
                field_name="DATE",
                field_type="date",
                required=True,
            ),
            DocumentPattern(
                pattern=r"CARRIER:\s*([^\n\r]+)",
                field_name="CARRIER",
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
                pattern=r"PASSENGER:\s*([^\n\r]+)",
                field_name="PASSENGER",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"ORIGIN:\s*([^\n\r]+)",
                field_name="ORIGIN",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"DESTINATION:\s*([^\n\r]+)",
                field_name="DESTINATION",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"DEPARTURE:\s*([^\n\r]+)",
                field_name="DEPARTURE",
                field_type="datetime",
                required=False,
            ),
            DocumentPattern(
                pattern=r"ARRIVAL:\s*([^\n\r]+)",
                field_name="ARRIVAL",
                field_type="datetime",
                required=False,
            ),
            DocumentPattern(
                pattern=r"FLIGHT:\s*([^\n\r]+)",
                field_name="FLIGHT",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"SEAT:\s*([^\n\r]+)",
                field_name="SEAT",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"GATE:\s*([^\n\r]+)",
                field_name="GATE",
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
                pattern=r"TICKET:\s*([^\n\r]+)",
                field_name="TICKET",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"PAYMENT_METHOD:\s*([^\n\r]+)",
                field_name="PAYMENT_METHOD",
                field_type="string",
                required=False,
            ),
        ]

    def get_field_mappings(self) -> Dict[str, List[str]]:
        """Get field mappings for standardization."""
        return {
            # Standard compliance fields
            "supplier_name": ["CARRIER"],
            "total_amount": ["TOTAL"],
            "gst_amount": ["GST"],
            "invoice_date": ["DATE"],
            "supplier_abn": ["ABN"],
            "payment_method": ["PAYMENT_METHOD"],
            "receipt_number": ["TICKET", "BOOKING"],
            "invoice_number": ["TICKET", "BOOKING"],
            # Travel-specific fields
            "airline_name": ["CARRIER"],
            "passenger_name": ["PASSENGER"],
            "departure_city": ["ORIGIN"],
            "arrival_city": ["DESTINATION"],
            "departure_time": ["DEPARTURE"],
            "arrival_time": ["ARRIVAL"],
            "flight_number": ["FLIGHT"],
            "seat_number": ["SEAT"],
            "gate_number": ["GATE"],
            "booking_reference": ["BOOKING"],
            "ticket_number": ["TICKET"],
            "travel_date": ["DATE"],
            "travel_route": ["ORIGIN", "DESTINATION"],
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
        # We expect 6+ fields for travel documents, so <5 indicates KEY-VALUE parsing failed
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
        from .awk_extractor import TravelDocumentAwkExtractor

        awk_extractor = TravelDocumentAwkExtractor(self.log_level)
        extracted = awk_extractor.extract_travel_fields(response)

        self.logger.debug(
            f"AWK fallback extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted