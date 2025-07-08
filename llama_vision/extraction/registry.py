"""Document type registry initialization and auto-registration."""

from .bank_statement_handler import BankStatementHandler
from .document_handlers import get_registry, register_document_handler
from .fuel_receipt_handler import FuelReceiptHandler
from .receipt_handler import ReceiptHandler
from .tax_invoice_handler import TaxInvoiceHandler


def initialize_document_handlers():
    """Initialize and register all document type handlers."""
    
    # Register all built-in handlers
    handlers = [
        FuelReceiptHandler(),
        BankStatementHandler(), 
        ReceiptHandler(),
        TaxInvoiceHandler(),
    ]
    
    registry = get_registry()
    
    for handler in handlers:
        register_document_handler(handler)
    
    return registry


def get_initialized_registry():
    """Get registry with all handlers pre-registered."""
    registry = get_registry()
    
    # Only initialize if no handlers registered yet
    if not registry.list_document_types():
        initialize_document_handlers()
    
    return registry