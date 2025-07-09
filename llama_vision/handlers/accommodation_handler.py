"""
Accommodation Handler for Australian Tax Document Processing

This handler specializes in processing accommodation receipts for business travel claims
with ATO compliance validation.
"""

import re
from typing import Any, Dict, List

from ..extraction.australian_tax_classifier import DocumentType
from ..utils import setup_logging
from .base_ato_handler import BaseATOHandler

logger = setup_logging()


class AccommodationHandler(BaseATOHandler):
    """Handler for Australian accommodation processing with ATO compliance."""

    def __init__(self):
        super().__init__(DocumentType.ACCOMMODATION)
        logger.info("AccommodationHandler initialized for business travel claims")

    def _extract_fields_primary(self, document_text: str) -> Dict[str, Any]:
        """Extract fields using primary extraction method for accommodation."""
        extracted_fields = {}

        # Extract hotel name
        hotel_patterns = [
            r"(HILTON|MARRIOTT|HYATT|IBIS|MERCURE|NOVOTEL|CROWNE PLAZA|HOLIDAY INN)",
            r"([A-Z][A-Za-z\s&]+(?:HOTEL|MOTEL|RESORT|INN))",
            r"([A-Z\s]+(?:HOTEL|MOTEL|RESORT))",
        ]

        for pattern in hotel_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["hotel_name"] = match.group(1).strip()
                break

        # Extract dates
        checkin_patterns = [
            r"(?:check.?in|arrival)[\s:]*(\d{1,2}/\d{1,2}/\d{4})",
            r"(?:from|start)[\s:]*(\d{1,2}/\d{1,2}/\d{4})",
        ]

        for pattern in checkin_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["date_checkin"] = match.group(1)
                break

        checkout_patterns = [
            r"(?:check.?out|departure)[\s:]*(\d{1,2}/\d{1,2}/\d{4})",
            r"(?:to|end)[\s:]*(\d{1,2}/\d{1,2}/\d{4})",
        ]

        for pattern in checkout_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["date_checkout"] = match.group(1)
                break

        # Extract number of nights
        nights_patterns = [r"(\d+)\s*nights?", r"nights?[\s:]*(\d+)"]

        for pattern in nights_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["nights"] = match.group(1)
                break

        # Extract room type
        room_patterns = [
            r"(standard|deluxe|suite|king|queen|twin|single|double)\s*(?:room|bed)",
            r"room[\s:]*([A-Za-z\s]+)",
        ]

        for pattern in room_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["room_type"] = match.group(1).strip().title()
                break

        # Extract room rate
        rate_patterns = [
            r"(?:rate|room rate)[\s:]*\$?(\d+\.\d{2})",
            r"per night[\s:]*\$?(\d+\.\d{2})",
        ]

        for pattern in rate_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["room_rate"] = match.group(1)
                break

        # Extract guest name
        guest_patterns = [
            r"(?:guest|customer|name)[\s:]*([A-Za-z\s]+)",
            r"(?:mr|ms|mrs|dr)[\s]*([A-Za-z\s]+)",
        ]

        for pattern in guest_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["guest_name"] = match.group(1).strip()
                break

        # Extract amounts
        subtotal_patterns = [
            r"subtotal[\s:]*\$?(\d+\.\d{2})",
            r"room[\s]*total[\s:]*\$?(\d+\.\d{2})",
        ]

        for pattern in subtotal_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["subtotal"] = match.group(1)
                break

        gst_patterns = [r"gst[\s:]*\$?(\d+\.\d{2})", r"tax[\s:]*\$?(\d+\.\d{2})"]

        for pattern in gst_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["gst_amount"] = match.group(1)
                break

        total_patterns = [r"total[\s:]*\$?(\d+\.\d{2})", r"amount[\s:]*\$?(\d+\.\d{2})"]

        for pattern in total_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["total_amount"] = match.group(1)
                break

        logger.debug(
            f"Primary extraction yielded {len(extracted_fields)} fields for accommodation"
        )
        return extracted_fields

    def _get_required_fields(self) -> List[str]:
        """Get required fields for accommodation processing."""
        return ["date_checkin", "hotel_name", "total_amount", "nights"]

    def _get_optional_fields(self) -> List[str]:
        """Get optional fields for accommodation processing."""
        return [
            "date_checkout",
            "room_type",
            "room_rate",
            "guest_name",
            "subtotal",
            "gst_amount",
            "hotel_abn",
            "address",
            "payment_method",
        ]

    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for accommodation fields."""
        return {
            "date_checkin": self._validate_australian_date,
            "date_checkout": self._validate_australian_date,
            "nights": self._validate_nights_count,
            "total_amount": self._validate_currency_amount,
            "gst_amount": self._validate_currency_amount,
            "room_rate": self._validate_currency_amount,
        }

    def _get_ato_thresholds(self) -> Dict[str, Any]:
        """Get ATO-specific thresholds for accommodation."""
        return {
            "receipt_required_threshold": 82.50,
            "business_purpose_required": True,
            "reasonable_rate_threshold": 400.00,  # Per night
            "gst_rate": 0.10,
        }

    def _get_awk_rules(self) -> List[Dict[str, Any]]:
        """Get AWK extraction rules for accommodation."""
        return [
            {
                "field": "hotel_name",
                "pattern": r"([A-Z][A-Za-z\s&]+(?:HOTEL|MOTEL|RESORT|INN))",
                "line_filter": lambda line: any(
                    word in line.upper() for word in ["HOTEL", "MOTEL", "RESORT", "INN"]
                ),
                "transform": lambda x: x.strip(),
            },
            {
                "field": "date_checkin",
                "pattern": r"\d{1,2}/\d{1,2}/\d{4}",
                "line_filter": lambda line: any(
                    word in line.lower() for word in ["check in", "arrival", "from"]
                ),
                "transform": lambda x: x.strip(),
            },
            {
                "field": "total_amount",
                "pattern": r"\$?(\d+\.\d{2})",
                "line_filter": lambda line: any(
                    word in line.lower() for word in ["total", "amount"]
                ),
                "transform": lambda x: x.replace("$", "").strip(),
            },
        ]

    def _validate_australian_date(self, date_str: str) -> str:
        """Validate Australian date format."""
        if not date_str:
            return date_str

        date_patterns = [
            (r"(\d{1,2})/(\d{1,2})/(\d{4})", r"\1/\2/\3"),
            (r"(\d{1,2})-(\d{1,2})-(\d{4})", r"\1/\2/\3"),
        ]

        for pattern, replacement in date_patterns:
            match = re.match(pattern, date_str.strip())
            if match:
                return re.sub(pattern, replacement, date_str.strip())

        return date_str

    def _validate_nights_count(self, nights_str: str) -> str:
        """Validate nights count."""
        if not nights_str:
            return nights_str

        try:
            nights = int(nights_str)
            if 1 <= nights <= 30:  # Reasonable range
                return str(nights)
            return nights_str
        except ValueError:
            return nights_str

    def _validate_currency_amount(self, amount_str: str) -> str:
        """Validate currency amount format."""
        if not amount_str:
            return amount_str

        clean_amount = re.sub(r"[^\d.]", "", amount_str)

        try:
            amount = float(clean_amount)
            if 0.01 <= amount <= 5000.0:  # Reasonable accommodation range
                return f"{amount:.2f}"
            return clean_amount
        except ValueError:
            return amount_str
