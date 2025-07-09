"""Document type registry initialization and auto-registration."""

from .accommodation_handler import AccommodationHandler
from .bank_statement_handler import BankStatementHandler
from .document_handlers import get_registry, register_document_handler
from .equipment_supplies_handler import EquipmentSuppliesHandler
from .fuel_receipt_handler import FuelReceiptHandler
from .meal_receipt_handler import MealReceiptHandler
from .other_document_handler import OtherDocumentHandler
from .parking_toll_handler import ParkingTollHandler
from .professional_services_handler import ProfessionalServicesHandler
from .receipt_handler import ReceiptHandler
from .tax_invoice_handler import TaxInvoiceHandler
from .travel_document_handler import TravelDocumentHandler


def initialize_document_handlers():
    """Initialize and register all document type handlers."""

    # Register all built-in handlers
    handlers = [
        FuelReceiptHandler(),
        BankStatementHandler(),
        ReceiptHandler(),
        TaxInvoiceHandler(),
        MealReceiptHandler(),
        AccommodationHandler(),
        TravelDocumentHandler(),
        ParkingTollHandler(),
        EquipmentSuppliesHandler(),
        ProfessionalServicesHandler(),
        OtherDocumentHandler(),  # Keep as fallback - should be last
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
