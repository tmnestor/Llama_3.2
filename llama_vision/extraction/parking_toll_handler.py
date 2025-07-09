"""Parking toll document type handler."""

import re
from typing import Any, Dict, List, Pattern

from .document_handlers import DocumentPattern, DocumentTypeHandler, ExtractionResult


class ParkingTollHandler(DocumentTypeHandler):
    """Handler for parking and toll documents."""

    @property
    def document_type(self) -> str:
        return "parking_toll"

    @property
    def display_name(self) -> str:
        return "Parking Toll"

    def get_classification_indicators(self) -> List[str]:
        """Get text indicators for parking and toll documents."""
        return [
            "parking",
            "car park",
            "wilson parking",
            "secure parking",
            "care park",
            "meter",
            "parking meter",
            "toll",
            "citylink",
            "eastlink",
            "westlink",
            "m1",
            "m2",
            "m3",
            "m4",
            "m5",
            "m7",
            "m8",
            "motorway",
            "freeway",
            "tunnel",
            "bridge",
            "harbour bridge",
            "sydney harbour tunnel",
            "lane cove tunnel",
            "cross city tunnel",
            "electronic toll",
            "e-toll",
            "etag",
            "tag",
            "transponder",
            "vehicle",
            "registration",
            "licence plate",
            "entry time",
            "exit time",
            "duration",
            "hourly rate",
            "daily rate",
            "zone",
            "level",
            "space",
            "bay",
            "ticket",
            "receipt",
            "validation",
            "permit",
        ]

    def get_classification_patterns(self) -> List[Pattern]:
        """Get regex patterns for parking toll classification."""
        return [
            re.compile(r"parking|car\s+park", re.IGNORECASE),
            re.compile(r"wilson|secure\s+parking|care\s+park", re.IGNORECASE),
            re.compile(r"toll|citylink|eastlink", re.IGNORECASE),
            re.compile(r"m[1-8]|motorway|freeway", re.IGNORECASE),
            re.compile(r"entry\s+time|exit\s+time", re.IGNORECASE),
            re.compile(r"duration|hourly\s+rate", re.IGNORECASE),
            re.compile(r"vehicle|registration", re.IGNORECASE),
            re.compile(r"level\s+\d+|zone\s+\d+", re.IGNORECASE),
            re.compile(r"etag|e-toll|transponder", re.IGNORECASE),
            re.compile(r"harbour\s+bridge|tunnel", re.IGNORECASE),
            re.compile(r"ticket|validation|permit", re.IGNORECASE),
        ]

    def get_prompt_name(self) -> str:
        """Get prompt name for parking toll documents."""
        return "parking_toll_extraction_prompt"

    def get_field_patterns(self) -> List[DocumentPattern]:
        """Get field extraction patterns for parking toll documents."""
        return [
            DocumentPattern(
                pattern=r"DATE:\s*([^\n\r]+)",
                field_name="DATE",
                field_type="date",
                required=True,
            ),
            DocumentPattern(
                pattern=r"OPERATOR:\s*([^\n\r]+)",
                field_name="OPERATOR",
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
                pattern=r"LOCATION:\s*([^\n\r]+)",
                field_name="LOCATION",
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
                pattern=r"VEHICLE:\s*([^\n\r]+)",
                field_name="VEHICLE",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"ENTRY:\s*([^\n\r]+)",
                field_name="ENTRY",
                field_type="datetime",
                required=False,
            ),
            DocumentPattern(
                pattern=r"EXIT:\s*([^\n\r]+)",
                field_name="EXIT",
                field_type="datetime",
                required=False,
            ),
            DocumentPattern(
                pattern=r"DURATION:\s*([^\n\r]+)",
                field_name="DURATION",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"ZONE:\s*([^\n\r]+)",
                field_name="ZONE",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"LEVEL:\s*([^\n\r]+)",
                field_name="LEVEL",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"SPACE:\s*([^\n\r]+)",
                field_name="SPACE",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"RATE:\s*([^\n\r]+)",
                field_name="RATE",
                field_type="number",
                required=False,
            ),
            DocumentPattern(
                pattern=r"PAYMENT_METHOD:\s*([^\n\r]+)",
                field_name="PAYMENT_METHOD",
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
            "supplier_name": ["OPERATOR"],
            "total_amount": ["TOTAL"],
            "gst_amount": ["GST"],
            "invoice_date": ["DATE"],
            "supplier_abn": ["ABN"],
            "payment_method": ["PAYMENT_METHOD"],
            "receipt_number": ["RECEIPT", "TICKET"],
            "invoice_number": ["RECEIPT", "TICKET"],
            # Parking/toll-specific fields
            "parking_operator": ["OPERATOR"],
            "parking_location": ["LOCATION"],
            "vehicle_registration": ["VEHICLE"],
            "entry_time": ["ENTRY"],
            "exit_time": ["EXIT"],
            "parking_duration": ["DURATION"],
            "parking_zone": ["ZONE"],
            "parking_level": ["LEVEL"],
            "parking_space": ["SPACE"],
            "hourly_rate": ["RATE"],
            "ticket_number": ["TICKET"],
            "toll_operator": ["OPERATOR"],
            "toll_location": ["LOCATION"],
            "toll_amount": ["TOTAL"],
            "toll_date": ["DATE"],
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
        # We expect 5+ fields for parking/toll, so <4 indicates KEY-VALUE parsing failed
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
        from .awk_extractor import ParkingTollAwkExtractor

        awk_extractor = ParkingTollAwkExtractor(self.log_level)
        extracted = awk_extractor.extract_parking_toll_fields(response)

        self.logger.debug(
            f"AWK fallback extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted
