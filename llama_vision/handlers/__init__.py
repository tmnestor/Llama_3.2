"""
ATO-Enhanced Document Handlers for Llama-3.2

This module provides specialized handlers for Australian tax documents,
integrating extraction, validation, and confidence scoring.
"""

from .accommodation_handler import AccommodationHandler
from .bank_statement_handler import BankStatementHandler
from .base_ato_handler import ATOProcessingResult, BaseATOHandler
from .business_receipt_handler import BusinessReceiptHandler
from .fuel_receipt_handler import FuelReceiptHandler
from .meal_receipt_handler import MealReceiptHandler
from .other_document_handler import OtherDocumentHandler
from .tax_invoice_handler import TaxInvoiceHandler

__all__ = [
    "BaseATOHandler",
    "ATOProcessingResult",
    "FuelReceiptHandler",
    "TaxInvoiceHandler",
    "BusinessReceiptHandler",
    "BankStatementHandler",
    "MealReceiptHandler",
    "AccommodationHandler",
    "OtherDocumentHandler",
]
