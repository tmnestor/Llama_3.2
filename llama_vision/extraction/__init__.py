"""Extraction module for Llama-3.2-Vision package."""

# Modern architecture (Registry + Strategy + Director pattern)
# Document handlers
from .bank_statement_handler import BankStatementHandler
from .document_handlers import DocumentTypeHandler, DocumentTypeRegistry
from .extraction_engine import DocumentExtractionEngine
from .fuel_receipt_handler import FuelReceiptHandler

# Legacy extractors (backward compatibility)
from .json_extraction import JSONExtractor
from .key_value_extraction import KeyValueExtractor
from .receipt_handler import ReceiptHandler
from .registry import get_initialized_registry, initialize_document_handlers
from .tax_authority_parser import TaxAuthorityParser
from .tax_invoice_handler import TaxInvoiceHandler

__all__ = [
    # Modern architecture (recommended)
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
    # Legacy (backward compatibility)
    "JSONExtractor",
    "KeyValueExtractor",
    "TaxAuthorityParser",
]
