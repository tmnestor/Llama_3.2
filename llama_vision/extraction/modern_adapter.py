"""Adapter to integrate modern extraction engine with existing interfaces."""

from typing import Any, Dict, Optional

from ..config import PromptManager
from ..utils import setup_logging
from .extraction_engine import DocumentExtractionEngine


class ModernExtractionAdapter:
    """Adapter to integrate modern extraction engine with existing interfaces.
    
    This adapter provides backward compatibility while using the new registry pattern.
    """
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize adapter.
        
        Args:
            log_level: Logging level
        """
        self.log_level = log_level
        self.logger = setup_logging(log_level)
        self.extraction_engine = DocumentExtractionEngine(log_level)
        self.prompt_manager = PromptManager()
    
    def classify_document_modern(self, ocr_text: str) -> Dict[str, Any]:
        """Classify document using modern registry system.
        
        Args:
            ocr_text: OCR text from document
            
        Returns:
            Classification result in legacy format for compatibility
        """
        classification = self.extraction_engine.classify_document(ocr_text)
        
        # Convert to legacy format
        return {
            "document_type": classification.document_type,
            "confidence": classification.confidence,
            "classification_response": classification.classification_response,
            "is_business_document": classification.is_business_document,
            "indicators_found": classification.indicators_found,  # New field
        }
    
    def get_prompt_for_document_type_modern(
        self, 
        document_type: str, 
        classification_response: str = ""
    ) -> Optional[str]:
        """Get prompt using modern registry system.
        
        Args:
            document_type: Document type identifier
            classification_response: Classification response for content-aware selection
            
        Returns:
            Prompt text or None if not found
        """
        return self.extraction_engine.get_prompt_for_document_type(
            document_type, classification_response
        )
    
    def extract_fields_modern(
        self, 
        document_type: str, 
        model_response: str
    ) -> Dict[str, Any]:
        """Extract fields using modern registry system.
        
        Args:
            document_type: Document type identifier
            model_response: Raw model response text
            
        Returns:
            Extracted fields in legacy format for compatibility
        """
        extraction = self.extraction_engine.extract_fields(document_type, model_response)
        
        if not extraction:
            # Fallback to empty result
            return {
                "_compliance_score": 0.0,
                "_extraction_method": "modern_adapter_fallback",
                "currency": "AUD",
                "country": "Australia",
            }
        
        # Add legacy compatibility fields
        result = extraction.fields.copy()
        result["_compliance_score"] = extraction.compliance_score
        result["_extraction_method"] = extraction.extraction_method
        
        return result
    
    def get_supported_document_types(self) -> Dict[str, str]:
        """Get supported document types.
        
        Returns:
            Dictionary mapping document_type -> display_name
        """
        return self.extraction_engine.get_supported_document_types()
    
    def process_complete_document(
        self, 
        ocr_text: str, 
        model_response: str
    ) -> Dict[str, Any]:
        """Complete document processing pipeline.
        
        Args:
            ocr_text: OCR text from document (for classification)
            model_response: Model response text (for extraction)
            
        Returns:
            Complete processing result
        """
        return self.extraction_engine.process_document(ocr_text, model_response)


# Global adapter instance for easy access
_adapter = None


def get_modern_adapter() -> ModernExtractionAdapter:
    """Get the global modern extraction adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = ModernExtractionAdapter()
    return _adapter