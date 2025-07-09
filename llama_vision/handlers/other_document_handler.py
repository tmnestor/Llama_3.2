"""
Other Document Handler for Australian Tax Document Processing

This handler processes miscellaneous documents that don't fit specific categories
with ATO compliance validation.
"""

import re
from typing import Any, Dict, List

from ..extraction.australian_tax_classifier import DocumentType
from ..utils import setup_logging
from .base_ato_handler import BaseATOHandler

logger = setup_logging()


class OtherDocumentHandler(BaseATOHandler):
    """Handler for other/miscellaneous Australian documents with ATO compliance."""

    def __init__(self):
        super().__init__(DocumentType.OTHER)
        logger.info("OtherDocumentHandler initialized for miscellaneous documents")

    def _extract_fields_primary(self, document_text: str) -> Dict[str, Any]:
        """Extract fields using primary extraction method for other documents."""
        extracted_fields = {}

        # Generic field extraction patterns
        date_patterns = [r"(\d{1,2}/\d{1,2}/\d{4})", r"(\d{1,2}-\d{1,2}-\d{4})"]

        for pattern in date_patterns:
            match = re.search(pattern, document_text)
            if match:
                extracted_fields["date"] = match.group(1)
                break

        # Extract business/supplier name
        business_patterns = [
            r"([A-Z][A-Za-z\s&]+(?:PTY\s+LTD|LIMITED|COMPANY))",
            r"([A-Z\s]+(?:SERVICES|SOLUTIONS|CONSULTING))",
        ]

        for pattern in business_patterns:
            match = re.search(pattern, document_text)
            if match:
                extracted_fields["supplier_name"] = match.group(1).strip()
                break

        # Extract total amount
        total_patterns = [r"total[\s:]*\$?(\d+\.\d{2})", r"amount[\s:]*\$?(\d+\.\d{2})"]

        for pattern in total_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["total_amount"] = match.group(1)
                break

        # Extract ABN if present
        abn_patterns = [
            r"abn[\s:]*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
            r"(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
        ]

        for pattern in abn_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["abn"] = match.group(1).strip()
                break

        # Extract GST if present
        gst_patterns = [r"gst[\s:]*\$?(\d+\.\d{2})", r"tax[\s:]*\$?(\d+\.\d{2})"]

        for pattern in gst_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["gst_amount"] = match.group(1)
                break

        logger.debug(
            f"Primary extraction yielded {len(extracted_fields)} fields for other document"
        )
        return extracted_fields

    def _get_required_fields(self) -> List[str]:
        """Get required fields for other document processing."""
        return ["date", "supplier_name", "total_amount"]

    def _get_optional_fields(self) -> List[str]:
        """Get optional fields for other document processing."""
        return ["abn", "gst_amount", "description", "reference_number"]

    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for other document fields."""
        return {
            "date": self._validate_australian_date,
            "abn": self._validate_abn_format,
            "total_amount": self._validate_currency_amount,
            "gst_amount": self._validate_currency_amount,
        }

    def _get_ato_thresholds(self) -> Dict[str, Any]:
        """Get ATO-specific thresholds for other documents."""
        return {
            "receipt_required_threshold": 82.50,
            "abn_required_threshold": 82.50,
            "gst_rate": 0.10,
        }

    def _get_awk_rules(self) -> List[Dict[str, Any]]:
        """Get AWK extraction rules for other documents."""
        return [
            {
                "field": "date",
                "pattern": r"\d{1,2}/\d{1,2}/\d{4}",
                "line_filter": lambda line: any(
                    word in line.lower() for word in ["date", "dated"]
                ),
                "transform": lambda x: x.strip(),
            },
            {
                "field": "supplier_name",
                "pattern": r"([A-Z][A-Za-z\s&]+(?:PTY\s+LTD|LIMITED|COMPANY))",
                "line_filter": lambda line: any(
                    word in line.upper() for word in ["PTY LTD", "LIMITED", "COMPANY"]
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

    def _validate_abn_format(self, abn_str: str) -> str:
        """Validate ABN format."""
        if not abn_str:
            return abn_str

        clean_abn = re.sub(r"[^\d]", "", abn_str)

        if len(clean_abn) == 11:
            return (
                f"{clean_abn[:2]} {clean_abn[2:5]} {clean_abn[5:8]} {clean_abn[8:11]}"
            )

        return abn_str

    def _validate_currency_amount(self, amount_str: str) -> str:
        """Validate currency amount format."""
        if not amount_str:
            return amount_str

        clean_amount = re.sub(r"[^\d.]", "", amount_str)

        try:
            amount = float(clean_amount)
            if 0.01 <= amount <= 100000.0:
                return f"{amount:.2f}"
            return clean_amount
        except ValueError:
            return amount_str
