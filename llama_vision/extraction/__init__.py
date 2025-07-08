"""Extraction module for Llama-3.2-Vision package."""

# Legacy extractors (backward compatibility)
# Specific document handlers
from .bank_statement_handler import BankStatementHandler

# Modern architecture (recommended)
from .document_handlers import DocumentTypeHandler, DocumentTypeRegistry
from .extraction_engine import DocumentExtractionEngine
from .fuel_receipt_handler import FuelReceiptHandler
from .fuel_receipt_parser import FuelReceiptParser
from .key_value_extraction import KeyValueExtractor
from .receipt_handler import ReceiptHandler
from .registry import get_initialized_registry, initialize_document_handlers
from .tax_authority_parser import TaxAuthorityParser
from .tax_invoice_handler import TaxInvoiceHandler

__all__ = [
    # Legacy (backward compatibility)
    "KeyValueExtractor",
    "TaxAuthorityParser",
    "FuelReceiptParser",
    # Modern architecture
    "DocumentTypeHandler",
    "DocumentTypeRegistry",
    "DocumentExtractionEngine",
    "get_initialized_registry",
    "initialize_document_handlers",
    # Document handlers
    "BankStatementHandler",
    "FuelReceiptHandler",
    "ReceiptHandler",
    "TaxInvoiceHandler",
]
