"""
Tax Invoice Handler for Australian Tax Document Processing

This handler specializes in processing tax invoices for business expense claims
with ATO compliance validation.
"""

import re
from typing import Any, Dict, List

from ..extraction.australian_tax_classifier import DocumentType
from ..utils import setup_logging
from .base_ato_handler import BaseATOHandler

logger = setup_logging()


class TaxInvoiceHandler(BaseATOHandler):
    """Handler for Australian tax invoice processing with ATO compliance."""

    def __init__(self):
        super().__init__(DocumentType.TAX_INVOICE)
        logger.info("TaxInvoiceHandler initialized for business expense claims")

    def _extract_fields_primary(self, document_text: str) -> Dict[str, Any]:
        """Extract fields using primary extraction method for tax invoices."""
        extracted_fields = {}

        # Extract document type
        document_type_patterns = [
            r"(tax invoice)",
            r"(invoice)",
            r"(gst invoice)",
            r"(tax inv)",
        ]

        for pattern in document_type_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["document_type"] = match.group(1).upper()
                break

        # Extract supplier information
        supplier_patterns = [
            r"(?:supplier|from|bill from|invoice from)[\s:]*([A-Z][A-Za-z\s&]+(?:PTY\s+LTD|LIMITED|COMPANY|CORPORATION|CORP|INC))",
            r"([A-Z][A-Za-z\s&]+(?:PTY\s+LTD|LIMITED|COMPANY|CORPORATION|CORP|INC))",
        ]

        for pattern in supplier_patterns:
            match = re.search(pattern, document_text, re.MULTILINE)
            if match:
                extracted_fields["supplier_name"] = match.group(1).strip()
                break

        # Extract supplier ABN
        abn_patterns = [
            r"abn[\s:]*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
            r"australian business number[\s:]*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
            r"(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
        ]

        for pattern in abn_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["supplier_abn"] = match.group(1).strip()
                break

        # Extract supplier address
        address_patterns = [
            r"(\d+\s+[A-Za-z\s]+(?:street|st|road|rd|avenue|ave|drive|dr|lane|ln)[A-Za-z\s,]*(?:NSW|VIC|QLD|WA|SA|TAS|NT|ACT)\s+\d{4})",
            r"([A-Za-z\s]+(?:NSW|VIC|QLD|WA|SA|TAS|NT|ACT)\s+\d{4})",
        ]

        for pattern in address_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["supplier_address"] = match.group(1).strip()
                break

        # Extract customer information
        customer_patterns = [
            r"(?:customer|to|bill to|invoice to)[\s:]*([A-Za-z\s&]+(?:PTY\s+LTD|LIMITED|COMPANY|CORPORATION|CORP|INC)?)",
            r"(?:client|customer)[\s:]*([A-Za-z\s&]+)",
        ]

        for pattern in customer_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["customer_name"] = match.group(1).strip()
                break

        # Extract customer ABN
        customer_abn_patterns = [
            r"(?:customer|client)\s+abn[\s:]*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
            r"(?:to|bill to).*?abn[\s:]*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
        ]

        for pattern in customer_abn_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE | re.DOTALL)
            if match:
                extracted_fields["customer_abn"] = match.group(1).strip()
                break

        # Extract invoice number
        invoice_patterns = [
            r"invoice\s+(?:no|number|#)[\s:]*([A-Za-z0-9\-]+)",
            r"inv[\s#:]*([A-Za-z0-9\-]+)",
            r"(?:invoice|inv)\s+([A-Za-z0-9\-]+)",
        ]

        for pattern in invoice_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["invoice_number"] = match.group(1).strip()
                break

        # Extract date
        date_patterns = [
            r"(?:date|dated|invoice date)[\s:]*(\d{1,2}/\d{1,2}/\d{4})",
            r"(\d{1,2}/\d{1,2}/\d{4})",
            r"(\d{1,2}-\d{1,2}-\d{4})",
            r"(\d{4}-\d{1,2}-\d{1,2})",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["date"] = match.group(1)
                break

        # Extract due date
        due_date_patterns = [
            r"(?:due date|payment due)[\s:]*(\d{1,2}/\d{1,2}/\d{4})",
            r"due[\s:]*(\d{1,2}/\d{1,2}/\d{4})",
        ]

        for pattern in due_date_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["due_date"] = match.group(1)
                break

        # Extract description
        description_patterns = [
            r"(?:description|services|goods|for)[\s:]*([A-Za-z\s]+)",
            r"(?:professional|consulting|legal|accounting)[\s]*([A-Za-z\s]+)",
        ]

        for pattern in description_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["description"] = match.group(1).strip()
                break

        # Extract subtotal
        subtotal_patterns = [
            r"subtotal[\s:]*\$?(\d+\.\d{2})",
            r"(?:sub total|net amount)[\s:]*\$?(\d+\.\d{2})",
            r"amount[\s:]*\$?(\d+\.\d{2})",
        ]

        for pattern in subtotal_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["subtotal"] = match.group(1)
                break

        # Extract GST amount
        gst_patterns = [
            r"gst[\s:]*\$?(\d+\.\d{2})",
            r"(?:goods\s+services\s+tax|g\.s\.t\.)[\s:]*\$?(\d+\.\d{2})",
            r"tax[\s:]*\$?(\d+\.\d{2})",
        ]

        for pattern in gst_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["gst_amount"] = match.group(1)
                break

        # Extract total amount
        total_patterns = [
            r"total[\s:]*\$?(\d+\.\d{2})",
            r"(?:total amount|amount due)[\s:]*\$?(\d+\.\d{2})",
            r"pay[\s:]*\$?(\d+\.\d{2})",
        ]

        for pattern in total_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["total_amount"] = match.group(1)
                break

        # Extract payment terms
        payment_terms_patterns = [
            r"(?:payment terms|terms)[\s:]*([A-Za-z\s\d]+)",
            r"(?:net|due)\s+(\d+\s+days?)",
            r"(net\s+\d+)",
        ]

        for pattern in payment_terms_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["payment_terms"] = match.group(1).strip()
                break

        logger.debug(
            f"Primary extraction yielded {len(extracted_fields)} fields for tax invoice"
        )
        return extracted_fields

    def _get_required_fields(self) -> List[str]:
        """Get required fields for tax invoice processing."""
        return ["date", "supplier_name", "supplier_abn", "gst_amount", "total_amount"]

    def _get_optional_fields(self) -> List[str]:
        """Get optional fields for tax invoice processing."""
        return [
            "document_type",
            "supplier_address",
            "customer_name",
            "customer_abn",
            "invoice_number",
            "due_date",
            "description",
            "subtotal",
            "payment_terms",
        ]

    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for tax invoice fields."""
        return {
            "date": self._validate_australian_date,
            "due_date": self._validate_australian_date,
            "supplier_abn": self._validate_abn_format,
            "customer_abn": self._validate_abn_format,
            "total_amount": self._validate_currency_amount,
            "gst_amount": self._validate_currency_amount,
            "subtotal": self._validate_currency_amount,
            "invoice_number": self._validate_invoice_number,
            "supplier_name": self._validate_business_name,
            "customer_name": self._validate_business_name,
        }

    def _get_ato_thresholds(self) -> Dict[str, Any]:
        """Get ATO-specific thresholds for tax invoices."""
        return {
            "receipt_required_threshold": 82.50,
            "abn_required_threshold": 0.0,  # Always required for tax invoices
            "gst_rate": 0.10,
            "gst_tolerance": 0.02,
            "tax_invoice_minimum_amount": 82.50,
        }

    def _get_awk_rules(self) -> List[Dict[str, Any]]:
        """Get AWK extraction rules for tax invoices."""
        return [
            {
                "field": "document_type",
                "pattern": r"(TAX INVOICE|INVOICE|GST INVOICE)",
                "line_filter": lambda line: any(
                    word in line.upper()
                    for word in ["TAX INVOICE", "INVOICE", "GST INVOICE"]
                ),
                "transform": lambda x: x.strip().upper(),
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
                "field": "supplier_abn",
                "pattern": r"(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
                "line_filter": lambda line: "abn" in line.lower()
                or "australian business number" in line.lower(),
                "transform": lambda x: x.strip(),
            },
            {
                "field": "date",
                "pattern": r"\d{1,2}/\d{1,2}/\d{4}",
                "line_filter": lambda line: any(
                    word in line.lower() for word in ["date", "dated", "invoice date"]
                ),
                "transform": lambda x: x.strip(),
            },
            {
                "field": "invoice_number",
                "pattern": r"([A-Za-z0-9\-]+)",
                "line_filter": lambda line: any(
                    word in line.lower() for word in ["invoice", "inv", "number", "no"]
                ),
                "transform": lambda x: x.strip(),
            },
            {
                "field": "subtotal",
                "pattern": r"\$?(\d+\.\d{2})",
                "line_filter": lambda line: any(
                    word in line.lower()
                    for word in ["subtotal", "sub total", "net amount"]
                ),
                "transform": lambda x: x.replace("$", "").strip(),
            },
            {
                "field": "gst_amount",
                "pattern": r"\$?(\d+\.\d{2})",
                "line_filter": lambda line: any(
                    word in line.lower()
                    for word in ["gst", "tax", "goods services tax"]
                ),
                "transform": lambda x: x.replace("$", "").strip(),
            },
            {
                "field": "total_amount",
                "pattern": r"\$?(\d+\.\d{2})",
                "line_filter": lambda line: any(
                    word in line.lower() for word in ["total", "amount due", "pay"]
                ),
                "transform": lambda x: x.replace("$", "").strip(),
            },
            {
                "field": "description",
                "pattern": r"([A-Za-z\s]+)",
                "line_filter": lambda line: any(
                    word in line.lower()
                    for word in [
                        "description",
                        "services",
                        "professional",
                        "consulting",
                    ]
                ),
                "transform": lambda x: x.strip(),
            },
        ]

    def _validate_australian_date(self, date_str: str) -> str:
        """Validate Australian date format."""
        if not date_str:
            return date_str

        # Convert to DD/MM/YYYY format
        date_patterns = [
            (r"(\d{1,2})/(\d{1,2})/(\d{4})", r"\1/\2/\3"),
            (r"(\d{1,2})-(\d{1,2})-(\d{4})", r"\1/\2/\3"),
            (r"(\d{4})-(\d{1,2})-(\d{1,2})", r"\2/\3/\1"),
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

        # Remove non-digits and format as XX XXX XXX XXX
        clean_abn = re.sub(r"[^\d]", "", abn_str)

        if len(clean_abn) == 11:
            return (
                f"{clean_abn[:2]} {clean_abn[2:5]} {clean_abn[5:8]} {clean_abn[8:11]}"
            )
        else:
            logger.warning(f"Invalid ABN format: {abn_str}")
            return abn_str

    def _validate_currency_amount(self, amount_str: str) -> str:
        """Validate currency amount format."""
        if not amount_str:
            return amount_str

        # Remove currency symbols and validate
        clean_amount = re.sub(r"[^\d.]", "", amount_str)

        try:
            amount = float(clean_amount)
            if 0.01 <= amount <= 100000.0:  # Reasonable invoice range
                return f"{amount:.2f}"
            else:
                logger.warning(f"Invoice amount {amount} outside reasonable range")
                return clean_amount
        except ValueError:
            logger.warning(f"Invalid currency amount format: {amount_str}")
            return amount_str

    def _validate_invoice_number(self, invoice_str: str) -> str:
        """Validate invoice number format."""
        if not invoice_str:
            return invoice_str

        # Clean and validate invoice number
        clean_invoice = re.sub(r"[^\w\-]", "", invoice_str)

        if len(clean_invoice) >= 3:
            return clean_invoice.upper()
        else:
            logger.warning(f"Invalid invoice number format: {invoice_str}")
            return invoice_str

    def _validate_business_name(self, name_str: str) -> str:
        """Validate business name format."""
        if not name_str:
            return name_str

        # Clean and format business name
        clean_name = name_str.strip()

        # Convert to title case for consistency
        business_suffixes = [
            "PTY LTD",
            "LIMITED",
            "COMPANY",
            "CORPORATION",
            "CORP",
            "INC",
        ]

        for suffix in business_suffixes:
            if suffix in clean_name.upper():
                # Keep suffix in uppercase
                clean_name = re.sub(
                    f"\\b{suffix}\\b", suffix, clean_name, flags=re.IGNORECASE
                )
                break

        return clean_name
