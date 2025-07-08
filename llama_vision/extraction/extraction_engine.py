"""Modern extraction engine using document type registry pattern."""

from typing import Any, Dict, Optional

from ..config import PromptManager
from ..utils import setup_logging
from .document_handlers import ClassificationResult, ExtractionResult
from .registry import get_initialized_registry


class DocumentExtractionEngine:
    """Modern extraction engine using registry pattern for scalable document processing."""
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize extraction engine.
        
        Args:
            log_level: Logging level
        """
        self.log_level = log_level
        self.logger = setup_logging(log_level)
        self.registry = get_initialized_registry()
        self.prompt_manager = PromptManager()
        
        self.logger.info(
            f"Extraction engine initialized with {len(self.registry.list_document_types())} "
            f"document types: {', '.join(self.registry.list_document_types())}"
        )
    
    def classify_document(self, ocr_text: str) -> ClassificationResult:
        """Classify document type using registered handlers.
        
        Args:
            ocr_text: OCR text from document
            
        Returns:
            Classification result with document type and confidence
        """
        return self.registry.classify_document(ocr_text)
    
    def get_prompt_for_document_type(
        self, 
        document_type: str, 
        classification_response: str = ""
    ) -> Optional[str]:
        """Get extraction prompt for specific document type.
        
        Args:
            document_type: Document type identifier
            classification_response: Classification response for content-aware selection
            
        Returns:
            Prompt text or None if document type not supported
        """
        # First check if we have a registered handler
        handler = self.registry.get_handler(document_type)
        if not handler:
            self.logger.warning(f"No handler found for document type: {document_type}")
            return None
        
        # Get prompt name from handler
        prompt_name = handler.get_prompt_name()
        
        # Handle special cases for content-aware selection (legacy compatibility)
        if document_type == "tax_invoice" and classification_response:
            # Check if this tax invoice contains fuel indicators
            fuel_indicators = [
                "costco", "ulp", "unleaded", "diesel", "litre", " l ", "fuel", "petrol"
            ]
            response_lower = classification_response.lower()
            
            if any(indicator in response_lower for indicator in fuel_indicators):
                # This is a fuel tax invoice - use fuel-specific prompt
                fuel_handler = self.registry.get_handler("fuel_receipt")
                if fuel_handler:
                    prompt_name = fuel_handler.get_prompt_name()
                    self.logger.info("Using fuel receipt prompt for fuel tax invoice")
        
        try:
            return self.prompt_manager.get_prompt(prompt_name)
        except KeyError:
            self.logger.error(f"Prompt not found: {prompt_name}")
            return None
    
    def extract_fields(
        self, 
        document_type: str, 
        model_response: str
    ) -> Optional[ExtractionResult]:
        """Extract fields from model response using document type handler.
        
        Args:
            document_type: Document type identifier
            model_response: Raw model response text
            
        Returns:
            Extraction result or None if document type not supported
        """
        return self.registry.extract_from_document(document_type, model_response)
    
    def get_supported_document_types(self) -> Dict[str, str]:
        """Get dictionary of supported document types and their display names.
        
        Returns:
            Dictionary mapping document_type -> display_name
        """
        result = {}
        for doc_type in self.registry.list_document_types():
            handler = self.registry.get_handler(doc_type)
            if handler:
                result[doc_type] = handler.display_name
        return result
    
    def get_document_type_info(self, document_type: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a document type.
        
        Args:
            document_type: Document type identifier
            
        Returns:
            Dictionary with document type information or None if not found
        """
        handler = self.registry.get_handler(document_type)
        if not handler:
            return None
        
        return {
            "document_type": handler.document_type,
            "display_name": handler.display_name,
            "prompt_name": handler.get_prompt_name(),
            "classification_indicators": handler.get_classification_indicators(),
            "field_count": len(handler.get_field_patterns()),
            "required_fields": [
                p.field_name for p in handler.get_field_patterns() if p.required
            ],
            "supported_field_types": list(set(
                p.field_type for p in handler.get_field_patterns()
            )),
        }
    
    def process_document(
        self, 
        ocr_text: str, 
        model_response: str
    ) -> Dict[str, Any]:
        """Complete document processing pipeline: classify and extract.
        
        Args:
            ocr_text: OCR text from document (for classification)
            model_response: Model response text (for extraction)
            
        Returns:
            Complete processing result
        """
        # Step 1: Classify document
        classification = self.classify_document(ocr_text)
        
        # Step 2: Extract fields if document type is supported
        extraction = None
        if classification.document_type != "unknown":
            extraction = self.extract_fields(
                classification.document_type, 
                model_response
            )
        
        # Step 3: Compile results
        result = {
            "classification": {
                "document_type": classification.document_type,
                "confidence": classification.confidence,
                "is_business_document": classification.is_business_document,
                "indicators_found": classification.indicators_found,
            },
            "extraction": None,
            "processing_successful": extraction is not None,
        }
        
        if extraction:
            result["extraction"] = {
                "fields": extraction.fields,
                "extraction_method": extraction.extraction_method,
                "compliance_score": extraction.compliance_score,
                "field_count": extraction.field_count,
            }
        
        return result