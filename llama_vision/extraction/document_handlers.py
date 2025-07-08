"""Document type handlers using Strategy pattern for scalable document processing."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern

from ..utils import setup_logging


@dataclass
class ClassificationResult:
    """Result of document classification."""

    document_type: str
    confidence: float
    classification_response: str
    is_business_document: bool
    indicators_found: List[str]


@dataclass
class ExtractionResult:
    """Result of field extraction."""

    fields: Dict[str, Any]
    extraction_method: str
    compliance_score: float
    field_count: int


@dataclass
class DocumentPattern:
    """Pattern definition for document field extraction."""

    pattern: str
    field_name: str
    field_type: str = "string"  # string, number, date, list
    required: bool = False
    normalize_func: Optional[str] = None


class DocumentTypeHandler(ABC):
    """Abstract base class for document type handlers."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize document handler.

        Args:
            log_level: Logging level
        """
        self.log_level = log_level
        self.logger = setup_logging(log_level)

    @property
    @abstractmethod
    def document_type(self) -> str:
        """Get the document type identifier."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Get the human-readable document type name."""
        pass

    @abstractmethod
    def get_classification_indicators(self) -> List[str]:
        """Get list of text indicators for classifying this document type."""
        pass

    @abstractmethod
    def get_classification_patterns(self) -> List[Pattern]:
        """Get regex patterns for advanced classification."""
        pass

    @abstractmethod
    def get_prompt_name(self) -> str:
        """Get the prompt name for this document type."""
        pass

    @abstractmethod
    def get_field_patterns(self) -> List[DocumentPattern]:
        """Get field extraction patterns for this document type."""
        pass

    @abstractmethod
    def get_field_mappings(self) -> Dict[str, List[str]]:
        """Get field mappings from extracted fields to standardized fields."""
        pass

    def classify(
        self, ocr_text: str, base_confidence: float = 0.5
    ) -> ClassificationResult:
        """Classify if this document matches this handler's type.

        Args:
            ocr_text: OCR text from document
            base_confidence: Base confidence if type matches

        Returns:
            Classification result
        """
        ocr_lower = ocr_text.lower()
        indicators = self.get_classification_indicators()
        patterns = self.get_classification_patterns()

        # Check text indicators
        found_indicators = [
            indicator for indicator in indicators if indicator.lower() in ocr_lower
        ]

        # Check regex patterns
        pattern_matches = []
        for pattern in patterns:
            if pattern.search(ocr_text):
                pattern_matches.append(pattern.pattern)

        # Calculate confidence based on indicators found
        indicator_ratio = len(found_indicators) / len(indicators) if indicators else 0
        pattern_ratio = len(pattern_matches) / len(patterns) if patterns else 0

        # Weighted confidence calculation
        confidence = base_confidence + (indicator_ratio * 0.3) + (pattern_ratio * 0.2)
        confidence = min(confidence, 0.99)  # Cap at 99%

        # Determine if this is the document type
        is_match = (
            len(found_indicators) >= 2  # At least 2 indicators
            or len(pattern_matches) >= 1  # At least 1 pattern match
            or indicator_ratio >= 0.5  # At least 50% of indicators
        )

        if is_match:
            self.logger.debug(
                f"{self.document_type} classification: {confidence:.2f} confidence "
                f"(indicators: {found_indicators}, patterns: {len(pattern_matches)})"
            )

        return ClassificationResult(
            document_type=self.document_type if is_match else "unknown",
            confidence=confidence if is_match else 0.1,
            classification_response=ocr_text,
            is_business_document=is_match and confidence > 0.7,
            indicators_found=found_indicators + pattern_matches,
        )

    def extract_fields(self, response: str) -> ExtractionResult:
        """Extract fields from model response.

        Args:
            response: Model response text

        Returns:
            Extraction result with fields and metadata
        """
        # Debug logging for bank statements
        if self.document_type == "bank_statement":
            self.logger.info("=== BANK STATEMENT RAW RESPONSE (first 500 chars) ===")
            self.logger.info(f"{response[:500]}...")
            self.logger.info("=== END RAW RESPONSE ===")

        extracted = {}
        patterns = self.get_field_patterns()

        # Extract fields using patterns
        for pattern_def in patterns:
            pattern = re.compile(pattern_def.pattern, re.IGNORECASE)
            match = pattern.search(response)

            # Debug logging for bank statements
            if self.document_type == "bank_statement":
                if match:
                    self.logger.info(
                        f"✓ Pattern matched: {pattern_def.field_name} = '{match.group(1).strip()}'"
                    )
                else:
                    self.logger.info(
                        f"✗ Pattern failed: {pattern_def.field_name} (pattern: {pattern_def.pattern})"
                    )

            if match:
                value = match.group(1).strip()
                if value and value not in ["", "N/A", "Not visible", "Not available"]:
                    # Apply type conversion
                    if pattern_def.field_type == "number":
                        try:
                            # Extract numeric value from currency strings
                            numeric_match = re.search(r"[\d.]+", value.replace("$", ""))
                            if numeric_match:
                                value = float(numeric_match.group(0))
                        except ValueError:
                            pass
                    elif pattern_def.field_type == "list":
                        # Split by pipe separator
                        if "|" in value:
                            value = [
                                item.strip()
                                for item in value.split("|")
                                if item.strip()
                            ]
                        else:
                            value = [value]

                    extracted[pattern_def.field_name] = value

        # Apply field mappings to create standardized fields
        mappings = self.get_field_mappings()
        normalized = self._apply_field_mappings(extracted, mappings)

        # Calculate compliance score
        required_patterns = [p for p in patterns if p.required]
        required_found = sum(1 for p in required_patterns if p.field_name in normalized)
        compliance_score = (
            required_found / len(required_patterns) if required_patterns else 1.0
        )

        return ExtractionResult(
            fields=normalized,
            extraction_method=f"{self.document_type}_handler",
            compliance_score=compliance_score,
            field_count=len(normalized),
        )

    def _apply_field_mappings(
        self, extracted: Dict[str, Any], mappings: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Apply field mappings to normalize extracted fields.

        Args:
            extracted: Raw extracted fields
            mappings: Field mapping dictionary

        Returns:
            Normalized fields dictionary
        """
        normalized = extracted.copy()

        for target_field, source_fields in mappings.items():
            for source_field in source_fields:
                if source_field in extracted and target_field not in normalized:
                    normalized[target_field] = extracted[source_field]
                    break

        # Add common fields
        normalized["currency"] = "AUD"
        normalized["country"] = "Australia"
        normalized["_extraction_method"] = f"{self.document_type}_handler"

        return normalized


class DocumentTypeRegistry:
    """Registry for managing document type handlers using Director pattern."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize document registry.

        Args:
            log_level: Logging level
        """
        self.handlers: Dict[str, DocumentTypeHandler] = {}
        self.log_level = log_level
        self.logger = setup_logging(log_level)

    def register_handler(self, handler: DocumentTypeHandler) -> None:
        """Register a document type handler.

        Args:
            handler: Document type handler instance
        """
        self.handlers[handler.document_type] = handler
        self.logger.info(
            f"Registered handler for document type: {handler.document_type}"
        )

    def get_handler(self, document_type: str) -> Optional[DocumentTypeHandler]:
        """Get handler for specific document type.

        Args:
            document_type: Document type identifier

        Returns:
            Handler instance or None if not found
        """
        return self.handlers.get(document_type)

    def list_document_types(self) -> List[str]:
        """Get list of registered document types."""
        return list(self.handlers.keys())

    def classify_document(self, ocr_text: str) -> ClassificationResult:
        """Classify document using all registered handlers.

        Args:
            ocr_text: OCR text from document

        Returns:
            Best classification result
        """
        if not self.handlers:
            self.logger.warning("No document handlers registered")
            return ClassificationResult(
                document_type="unknown",
                confidence=0.0,
                classification_response=ocr_text,
                is_business_document=False,
                indicators_found=[],
            )

        # Get classification from all handlers
        results = []
        for handler in self.handlers.values():
            result = handler.classify(ocr_text)
            if result.document_type != "unknown":
                results.append(result)

        if not results:
            # No matches found
            return ClassificationResult(
                document_type="unknown",
                confidence=0.0,
                classification_response=ocr_text,
                is_business_document=False,
                indicators_found=[],
            )

        # Return highest confidence result
        best_result = max(results, key=lambda r: r.confidence)

        self.logger.info(
            f"Document classified as {best_result.document_type} "
            f"with {best_result.confidence:.2f} confidence"
        )

        return best_result

    def extract_from_document(
        self, document_type: str, response: str
    ) -> Optional[ExtractionResult]:
        """Extract fields using specific document type handler.

        Args:
            document_type: Document type identifier
            response: Model response text

        Returns:
            Extraction result or None if handler not found
        """
        handler = self.get_handler(document_type)
        if not handler:
            self.logger.error(f"No handler found for document type: {document_type}")
            return None

        return handler.extract_fields(response)

    def get_prompt_name(self, document_type: str) -> Optional[str]:
        """Get prompt name for specific document type.

        Args:
            document_type: Document type identifier

        Returns:
            Prompt name or None if handler not found
        """
        handler = self.get_handler(document_type)
        return handler.get_prompt_name() if handler else None


# Registry singleton instance
_registry = None


def get_registry() -> DocumentTypeRegistry:
    """Get the global document type registry instance."""
    global _registry
    if _registry is None:
        _registry = DocumentTypeRegistry()
    return _registry


def register_document_handler(handler: DocumentTypeHandler) -> None:
    """Register a document handler with the global registry."""
    get_registry().register_handler(handler)
