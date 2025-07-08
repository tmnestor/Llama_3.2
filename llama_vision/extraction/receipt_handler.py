"""General receipt document type handler."""

import re
from typing import Dict, List, Pattern

from .document_handlers import DocumentPattern, DocumentTypeHandler


class ReceiptHandler(DocumentTypeHandler):
    """Handler for general retail receipt documents."""
    
    @property
    def document_type(self) -> str:
        return "receipt"
    
    @property
    def display_name(self) -> str:
        return "Retail Receipt"
    
    def get_classification_indicators(self) -> List[str]:
        """Get text indicators for general receipts."""
        return [
            "receipt",
            "woolworths",
            "coles",
            "aldi",
            "iga",
            "total",
            "subtotal",
            "tax",
            "gst",
            "items",
            "quantity",
            "price",
            "payment",
            "eftpos",
            "cash",
            "credit",
            "debit"
        ]
    
    def get_classification_patterns(self) -> List[Pattern]:
        """Get regex patterns for receipt classification."""
        return [
            re.compile(r"total:?\s*\$?\d+\.\d{2}", re.IGNORECASE),    # Total amount
            re.compile(r"gst:?\s*\$?\d+\.\d{2}", re.IGNORECASE),     # GST amount
            re.compile(r"subtotal:?\s*\$?\d+\.\d{2}", re.IGNORECASE), # Subtotal
            re.compile(r"qty\s*\d+", re.IGNORECASE),                 # Quantity
        ]
    
    def get_prompt_name(self) -> str:
        """Get prompt name for general receipts."""
        return "business_receipt_extraction_prompt"
    
    def get_field_patterns(self) -> List[DocumentPattern]:
        """Get field extraction patterns for receipts."""
        return [
            DocumentPattern(
                pattern=r"DATE:\s*([^\n\r]+)",
                field_name="DATE",
                field_type="date",
                required=True
            ),
            DocumentPattern(
                pattern=r"STORE:\s*([^\n\r]+)",
                field_name="STORE",
                field_type="string", 
                required=True
            ),
            DocumentPattern(
                pattern=r"ABN:\s*([^\n\r]+)",
                field_name="ABN",
                field_type="string",
                required=False
            ),
            DocumentPattern(
                pattern=r"PAYER:\s*([^\n\r]+)",
                field_name="PAYER",
                field_type="string",
                required=False
            ),
            DocumentPattern(
                pattern=r"TAX:\s*([^\n\r]+)",
                field_name="TAX",
                field_type="number",
                required=True
            ),
            DocumentPattern(
                pattern=r"GST:\s*([^\n\r]+)",
                field_name="GST", 
                field_type="number",
                required=True
            ),
            DocumentPattern(
                pattern=r"TOTAL:\s*([^\n\r]+)",
                field_name="TOTAL",
                field_type="number",
                required=True
            ),
            DocumentPattern(
                pattern=r"SUBTOTAL:\s*([^\n\r]+)",
                field_name="SUBTOTAL",
                field_type="number",
                required=False
            ),
            DocumentPattern(
                pattern=r"PRODUCTS:\s*([^\n\r]+)",
                field_name="PRODUCTS",
                field_type="list",
                required=True
            ),
            DocumentPattern(
                pattern=r"QUANTITIES:\s*([^\n\r]+)",
                field_name="QUANTITIES",
                field_type="list",
                required=False
            ),
            DocumentPattern(
                pattern=r"PRICES:\s*([^\n\r]+)",
                field_name="PRICES",
                field_type="list",
                required=False
            ),
            DocumentPattern(
                pattern=r"PAYMENT_METHOD:\s*([^\n\r]+)",
                field_name="PAYMENT_METHOD",
                field_type="string",
                required=False
            ),
            DocumentPattern(
                pattern=r"RECEIPT:\s*([^\n\r]+)",
                field_name="RECEIPT",
                field_type="string",
                required=False
            ),
        ]
    
    def get_field_mappings(self) -> Dict[str, List[str]]:
        """Get field mappings for standardization."""
        return {
            # Standard compliance fields
            "supplier_name": ["STORE"],
            "total_amount": ["TOTAL"],
            "gst_amount": ["GST", "TAX"],
            "invoice_date": ["DATE"],
            "supplier_abn": ["ABN"],
            "items": ["PRODUCTS"],
            "quantities": ["QUANTITIES"], 
            "prices": ["PRICES"],
            "payment_method": ["PAYMENT_METHOD"],
            "receipt_number": ["RECEIPT"],
            "invoice_number": ["RECEIPT"],
            
            # Receipt-specific fields
            "business_name": ["STORE"],
            "transaction_date": ["DATE"],
            "subtotal_amount": ["SUBTOTAL"],
            "customer_name": ["PAYER"],
        }