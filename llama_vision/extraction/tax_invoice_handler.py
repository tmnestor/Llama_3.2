"""Tax invoice document type handler."""

import re
from typing import Dict, List, Pattern

from .document_handlers import DocumentPattern, DocumentTypeHandler


class TaxInvoiceHandler(DocumentTypeHandler):
    """Handler for tax invoice documents."""

    @property
    def document_type(self) -> str:
        return "tax_invoice"

    @property
    def display_name(self) -> str:
        return "Tax Invoice"

    def get_classification_indicators(self) -> List[str]:
        """Get text indicators for tax invoices."""
        return [
            "tax invoice",
            "invoice",
            "tax receipt",
            "gst",
            "abn",
            "invoice number",
            "invoice date",
            "tax",
            "line items",
            "supplier",
            "vendor",
            "business",
            "formal",
        ]

    def get_classification_patterns(self) -> List[Pattern]:
        """Get regex patterns for tax invoice classification."""
        return [
            re.compile(r"tax\s+invoice", re.IGNORECASE),  # Tax invoice header
            re.compile(r"invoice\s+number", re.IGNORECASE),  # Invoice number
            re.compile(
                r"abn:?\s*\d{2}\s\d{3}\s\d{3}\s\d{3}", re.IGNORECASE
            ),  # ABN format
            re.compile(r"gst:?\s*\$?\d+\.\d{2}", re.IGNORECASE),  # GST amount
            re.compile(r"line\s+items?", re.IGNORECASE),  # Line items
        ]

    def get_prompt_name(self) -> str:
        """Get prompt name for tax invoices."""
        return "tax_invoice_extraction_prompt"

    def get_field_patterns(self) -> List[DocumentPattern]:
        """Get field extraction patterns for tax invoices."""
        return [
            DocumentPattern(
                pattern=r"DATE:\s*([^\n\r]+)",
                field_name="DATE",
                field_type="date",
                required=True,
            ),
            DocumentPattern(
                pattern=r"VENDOR:\s*([^\n\r]+)",
                field_name="VENDOR",
                field_type="string",
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
                required=True,
            ),
            DocumentPattern(
                pattern=r"GST:\s*([^\n\r]+)",
                field_name="GST",
                field_type="number",
                required=True,
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
                pattern=r"INVOICE_NUMBER:\s*([^\n\r]+)",
                field_name="INVOICE_NUMBER",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"ITEMS:\s*([^\n\r]+)",
                field_name="ITEMS",
                field_type="list",
                required=False,
            ),
            DocumentPattern(
                pattern=r"QUANTITIES:\s*([^\n\r]+)",
                field_name="QUANTITIES",
                field_type="list",
                required=False,
            ),
            DocumentPattern(
                pattern=r"PRICES:\s*([^\n\r]+)",
                field_name="PRICES",
                field_type="list",
                required=False,
            ),
        ]

    def get_field_mappings(self) -> Dict[str, List[str]]:
        """Get field mappings for standardization."""
        return {
            # Standard compliance fields
            "supplier_name": ["VENDOR", "STORE"],
            "total_amount": ["TOTAL"],
            "gst_amount": ["GST", "TAX"],
            "invoice_date": ["DATE"],
            "supplier_abn": ["ABN"],
            "items": ["ITEMS"],
            "quantities": ["QUANTITIES"],
            "prices": ["PRICES"],
            "invoice_number": ["INVOICE_NUMBER"],
            # Tax invoice specific fields
            "vendor_name": ["VENDOR"],
            "business_name": ["STORE", "VENDOR"],
            "transaction_date": ["DATE"],
            "tax_amount": ["GST", "TAX"],
        }
