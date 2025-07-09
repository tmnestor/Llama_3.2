"""
Hybrid Extraction Manager for Australian Tax Document Processing

This manager coordinates multiple extraction methods and handlers to provide
comprehensive document processing with ATO compliance.
"""

import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..handlers.accommodation_handler import AccommodationHandler
from ..handlers.bank_statement_handler import BankStatementHandler
from ..handlers.base_ato_handler import ATOProcessingResult, BaseATOHandler
from ..handlers.business_receipt_handler import BusinessReceiptHandler
from ..handlers.fuel_receipt_handler import FuelReceiptHandler
from ..handlers.meal_receipt_handler import MealReceiptHandler
from ..handlers.other_document_handler import OtherDocumentHandler
from ..handlers.tax_invoice_handler import TaxInvoiceHandler
from ..utils import setup_logging
from .australian_tax_classifier import DocumentType, classify_australian_tax_document

logger = setup_logging()


@dataclass
class HybridExtractionResult:
    """Result of hybrid extraction processing."""

    success: bool
    document_type: DocumentType
    processing_result: ATOProcessingResult
    extraction_method: str
    processing_time: float
    handler_used: str
    confidence_summary: Dict[str, Any]
    error_message: Optional[str] = None


class HybridExtractionManager:
    """Manager for hybrid extraction with multiple handlers and methods."""

    def __init__(self):
        """Initialize hybrid extraction manager with all handlers."""

        # Initialize document handlers
        self.handlers: Dict[DocumentType, BaseATOHandler] = {
            DocumentType.FUEL_RECEIPT: FuelReceiptHandler(),
            DocumentType.TAX_INVOICE: TaxInvoiceHandler(),
            DocumentType.BUSINESS_RECEIPT: BusinessReceiptHandler(),
            DocumentType.BANK_STATEMENT: BankStatementHandler(),
            DocumentType.MEAL_RECEIPT: MealReceiptHandler(),
            DocumentType.ACCOMMODATION: AccommodationHandler(),
            DocumentType.OTHER: OtherDocumentHandler(),
        }

        # Fallback handler for unknown document types
        self.fallback_handler = OtherDocumentHandler()

        # Processing configuration
        self.processing_config = {
            "enable_classification": True,
            "enable_awk_fallback": True,
            "enable_confidence_scoring": True,
            "enable_ato_validation": True,
            "confidence_threshold": 0.7,
            "max_processing_time": 30.0,  # seconds
        }

        logger.info("HybridExtractionManager initialized with all document handlers")

    def process_document(
        self,
        document_text: str,
        image_path: Optional[str] = None,
        force_document_type: Optional[DocumentType] = None,
    ) -> HybridExtractionResult:
        """
        Process document using hybrid extraction approach.

        Args:
            document_text: Document text content
            image_path: Optional path to document image
            force_document_type: Optional document type to force (skip classification)

        Returns:
            HybridExtractionResult with comprehensive processing results
        """
        start_time = self._get_current_time()

        try:
            logger.info("Starting hybrid document processing")

            # Step 1: Document classification (unless forced)
            if force_document_type:
                document_type = force_document_type
                logger.info(f"Document type forced to: {document_type.value}")
            else:
                document_type = self._classify_document(document_text)
                logger.info(f"Document classified as: {document_type.value}")

            # Step 2: Select appropriate handler
            handler = self._get_handler(document_type)
            logger.info(f"Selected handler: {handler.__class__.__name__}")

            # Step 3: Process document with selected handler
            processing_result = handler.process_document(document_text, image_path)

            # Step 4: Calculate processing time
            processing_time = self._get_current_time() - start_time

            # Step 5: Extract confidence summary
            confidence_summary = self._extract_confidence_summary(processing_result)

            # Step 6: Determine extraction method used
            extraction_method = self._determine_extraction_method(processing_result)

            # Step 7: Create result
            result = HybridExtractionResult(
                success=processing_result.success,
                document_type=document_type,
                processing_result=processing_result,
                extraction_method=extraction_method,
                processing_time=processing_time,
                handler_used=handler.__class__.__name__,
                confidence_summary=confidence_summary,
            )

            logger.info(
                f"Document processing completed in {processing_time:.2f}s with {confidence_summary.get('quality_grade', 'Unknown')} quality"
            )
            return result

        except Exception as e:
            processing_time = self._get_current_time() - start_time
            logger.error(f"Document processing failed: {str(e)}")
            logger.error(traceback.format_exc())

            return HybridExtractionResult(
                success=False,
                document_type=DocumentType.OTHER,
                processing_result=self._create_error_result(str(e)),
                extraction_method="failed",
                processing_time=processing_time,
                handler_used="none",
                confidence_summary={
                    "quality_grade": "Failed",
                    "overall_confidence": 0.0,
                },
                error_message=str(e),
            )

    def process_batch(
        self, documents: List[Dict[str, Any]]
    ) -> List[HybridExtractionResult]:
        """
        Process multiple documents in batch.

        Args:
            documents: List of document dictionaries with 'text' and optional 'image_path'

        Returns:
            List of HybridExtractionResult objects
        """
        results = []

        logger.info(f"Starting batch processing of {len(documents)} documents")

        for i, doc in enumerate(documents):
            try:
                document_text = doc.get("text", "")
                image_path = doc.get("image_path")
                force_type = doc.get("force_document_type")

                logger.info(f"Processing document {i + 1}/{len(documents)}")

                result = self.process_document(document_text, image_path, force_type)
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing document {i + 1}: {str(e)}")
                error_result = HybridExtractionResult(
                    success=False,
                    document_type=DocumentType.OTHER,
                    processing_result=self._create_error_result(str(e)),
                    extraction_method="failed",
                    processing_time=0.0,
                    handler_used="none",
                    confidence_summary={
                        "quality_grade": "Failed",
                        "overall_confidence": 0.0,
                    },
                    error_message=str(e),
                )
                results.append(error_result)

        logger.info(
            f"Batch processing completed: {sum(1 for r in results if r.success)}/{len(results)} successful"
        )
        return results

    def get_processing_statistics(
        self, results: List[HybridExtractionResult]
    ) -> Dict[str, Any]:
        """
        Generate processing statistics from results.

        Args:
            results: List of HybridExtractionResult objects

        Returns:
            Dictionary with processing statistics
        """
        if not results:
            return {}

        # Basic statistics
        total_documents = len(results)
        successful_documents = sum(1 for r in results if r.success)
        failed_documents = total_documents - successful_documents

        # Document type distribution
        document_types = {}
        for result in results:
            doc_type = result.document_type.value
            document_types[doc_type] = document_types.get(doc_type, 0) + 1

        # Handler usage
        handler_usage = {}
        for result in results:
            handler = result.handler_used
            handler_usage[handler] = handler_usage.get(handler, 0) + 1

        # Extraction method distribution
        extraction_methods = {}
        for result in results:
            method = result.extraction_method
            extraction_methods[method] = extraction_methods.get(method, 0) + 1

        # Quality grades
        quality_grades = {}
        for result in results:
            grade = result.confidence_summary.get("quality_grade", "Unknown")
            quality_grades[grade] = quality_grades.get(grade, 0) + 1

        # Average processing time
        avg_processing_time = sum(r.processing_time for r in results) / total_documents

        # Average confidence
        confidences = [
            r.confidence_summary.get("overall_confidence", 0.0)
            for r in results
            if r.success
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "total_documents": total_documents,
            "successful_documents": successful_documents,
            "failed_documents": failed_documents,
            "success_rate": successful_documents / total_documents * 100,
            "document_type_distribution": document_types,
            "handler_usage": handler_usage,
            "extraction_method_distribution": extraction_methods,
            "quality_grade_distribution": quality_grades,
            "average_processing_time": avg_processing_time,
            "average_confidence": avg_confidence,
        }

    def _classify_document(self, document_text: str) -> DocumentType:
        """Classify document type."""
        if not self.processing_config["enable_classification"]:
            return DocumentType.OTHER

        try:
            classification_result = classify_australian_tax_document(document_text)
            return classification_result.document_type
        except Exception as e:
            logger.warning(f"Classification failed: {str(e)}")
            return DocumentType.OTHER

    def _get_handler(self, document_type: DocumentType) -> BaseATOHandler:
        """Get appropriate handler for document type."""
        handler = self.handlers.get(document_type)
        if handler is None:
            logger.warning(
                f"No specific handler for {document_type.value}, using fallback"
            )
            return self.fallback_handler

        return handler

    def _extract_confidence_summary(
        self, processing_result: ATOProcessingResult
    ) -> Dict[str, Any]:
        """Extract confidence summary from processing result."""
        if not processing_result.success:
            return {
                "quality_grade": "Failed",
                "overall_confidence": 0.0,
                "is_production_ready": False,
            }

        confidence_score = processing_result.confidence_score

        return {
            "quality_grade": confidence_score.get("quality_grade", "Unknown"),
            "overall_confidence": confidence_score.get("overall_confidence", 0.0),
            "is_production_ready": confidence_score.get("is_production_ready", False),
            "classification_confidence": confidence_score.get(
                "classification_confidence", 0.0
            ),
            "extraction_confidence": confidence_score.get("extraction_confidence", 0.0),
            "ato_compliance_confidence": confidence_score.get(
                "ato_compliance_confidence", 0.0
            ),
            "australian_business_confidence": confidence_score.get(
                "australian_business_confidence", 0.0
            ),
        }

    def _determine_extraction_method(
        self, processing_result: ATOProcessingResult
    ) -> str:
        """Determine extraction method used."""
        if not processing_result.success:
            return "failed"

        return processing_result.processing_method

    def _create_error_result(self, error_message: str) -> ATOProcessingResult:
        """Create error processing result."""
        return ATOProcessingResult(
            success=False,
            document_type=DocumentType.OTHER,
            extracted_fields={},
            ato_compliance={},
            confidence_score={},
            processing_method="error",
            extraction_quality="Failed",
            recommendations=["Document processing failed - manual review required"],
            raw_extraction={},
            error_message=error_message,
        )

    def _get_current_time(self) -> float:
        """Get current time for timing measurements."""
        import time

        return time.time()

    def get_supported_document_types(self) -> List[str]:
        """Get list of supported document types."""
        return [doc_type.value for doc_type in self.handlers.keys()]

    def get_handler_info(self, document_type: DocumentType) -> Dict[str, Any]:
        """Get information about a specific handler."""
        handler = self._get_handler(document_type)

        return {
            "handler_class": handler.__class__.__name__,
            "document_type": document_type.value,
            "requirements": handler.get_document_requirements(),
        }

    def update_configuration(self, config_updates: Dict[str, Any]) -> None:
        """Update processing configuration."""
        self.processing_config.update(config_updates)
        logger.info(f"Processing configuration updated: {config_updates}")

    def get_configuration(self) -> Dict[str, Any]:
        """Get current processing configuration."""
        return self.processing_config.copy()


# Global instance for easy access
hybrid_extraction_manager = HybridExtractionManager()


def process_australian_tax_document(
    document_text: str,
    image_path: Optional[str] = None,
    force_document_type: Optional[DocumentType] = None,
) -> HybridExtractionResult:
    """
    Process Australian tax document using hybrid extraction.

    Args:
        document_text: Document text content
        image_path: Optional path to document image
        force_document_type: Optional document type to force

    Returns:
        HybridExtractionResult with comprehensive processing results
    """
    return hybrid_extraction_manager.process_document(
        document_text, image_path, force_document_type
    )


def process_document_batch(
    documents: List[Dict[str, Any]],
) -> List[HybridExtractionResult]:
    """
    Process multiple documents in batch.

    Args:
        documents: List of document dictionaries

    Returns:
        List of HybridExtractionResult objects
    """
    return hybrid_extraction_manager.process_batch(documents)


def get_processing_statistics(results: List[HybridExtractionResult]) -> Dict[str, Any]:
    """
    Generate processing statistics from results.

    Args:
        results: List of HybridExtractionResult objects

    Returns:
        Dictionary with processing statistics
    """
    return hybrid_extraction_manager.get_processing_statistics(results)
