"""
Base ATO Handler for Australian Tax Document Processing

This module provides the base class for all Australian tax document handlers,
integrating extraction, validation, and confidence scoring.
"""

import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..extraction.ato_compliance_handler import assess_ato_compliance_enhanced
from ..extraction.australian_tax_classifier import (
    DocumentType,
    classify_australian_tax_document,
)
from ..extraction.australian_tax_confidence_scorer import (
    score_australian_tax_document_processing,
)
from ..extraction.australian_tax_prompts import get_document_extraction_prompt
from ..extraction.awk_extractor import AwkExtractor
from ..utils import setup_logging

logger = setup_logging()


@dataclass
class ATOProcessingResult:
    """Result of ATO document processing."""

    success: bool
    document_type: DocumentType
    extracted_fields: Dict[str, Any]
    ato_compliance: Dict[str, Any]
    confidence_score: Dict[str, Any]
    processing_method: str
    extraction_quality: str
    recommendations: List[str]
    raw_extraction: Dict[str, Any]
    error_message: Optional[str] = None


class BaseATOHandler(ABC):
    """Base class for Australian Tax Office document handlers."""

    def __init__(self, document_type: DocumentType):
        self.document_type = document_type
        self.awk_extractor = AwkExtractor()
        self.extraction_prompt = get_document_extraction_prompt(document_type.value)

        # Document-specific configuration
        self.required_fields = self._get_required_fields()
        self.optional_fields = self._get_optional_fields()
        self.validation_rules = self._get_validation_rules()
        self.ato_thresholds = self._get_ato_thresholds()

        logger.info(f"Initialized {self.__class__.__name__} for {document_type.value}")

    def process_document(
        self, document_text: str, _image_path: Optional[str] = None
    ) -> ATOProcessingResult:
        """
        Process document with comprehensive ATO compliance pipeline.

        Args:
            document_text: Document text content
            image_path: Optional path to document image

        Returns:
            ATOProcessingResult with complete processing details
        """
        try:
            logger.info(f"Processing {self.document_type.value} document")

            # Step 1: Classify document to verify type
            classification_result = classify_australian_tax_document(document_text)

            # Step 2: Extract fields using primary method
            extracted_fields = self._extract_fields_primary(document_text)

            # Step 3: Apply AWK fallback if needed
            if self._needs_awk_fallback(extracted_fields):
                logger.info("Applying AWK fallback extraction")
                awk_fields = self._extract_fields_awk(document_text)
                extracted_fields = self._merge_extractions(extracted_fields, awk_fields)

            # Step 4: Validate extracted fields
            validated_fields = self._validate_fields(extracted_fields)

            # Step 5: Assess ATO compliance
            ato_compliance = self._assess_ato_compliance(validated_fields)

            # Step 6: Score confidence
            confidence_score = self._score_confidence(
                document_text, classification_result, validated_fields, ato_compliance
            )

            # Step 7: Generate recommendations
            recommendations = self._generate_recommendations(
                validated_fields, ato_compliance, confidence_score
            )

            # Determine processing method
            processing_method = self._determine_processing_method(extracted_fields)

            # Get extraction quality
            extraction_quality = confidence_score.get("quality_grade", "Unknown")

            result = ATOProcessingResult(
                success=True,
                document_type=self.document_type,
                extracted_fields=validated_fields,
                ato_compliance=ato_compliance,
                confidence_score=confidence_score,
                processing_method=processing_method,
                extraction_quality=extraction_quality,
                recommendations=recommendations,
                raw_extraction=extracted_fields,
            )

            logger.info(
                f"Document processing completed with {extraction_quality} quality"
            )
            return result

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            logger.error(traceback.format_exc())

            return ATOProcessingResult(
                success=False,
                document_type=self.document_type,
                extracted_fields={},
                ato_compliance={},
                confidence_score={},
                processing_method="error",
                extraction_quality="Failed",
                recommendations=["Document processing failed - manual review required"],
                raw_extraction={},
                error_message=str(e),
            )

    @abstractmethod
    def _extract_fields_primary(self, document_text: str) -> Dict[str, Any]:
        """Extract fields using primary extraction method."""
        pass

    @abstractmethod
    def _get_required_fields(self) -> List[str]:
        """Get list of required fields for this document type."""
        pass

    @abstractmethod
    def _get_optional_fields(self) -> List[str]:
        """Get list of optional fields for this document type."""
        pass

    @abstractmethod
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this document type."""
        pass

    @abstractmethod
    def _get_ato_thresholds(self) -> Dict[str, Any]:
        """Get ATO-specific thresholds for this document type."""
        pass

    def _extract_fields_awk(self, document_text: str) -> Dict[str, Any]:
        """Extract fields using AWK fallback method."""
        try:
            # Get document-specific AWK rules
            awk_rules = self._get_awk_rules()

            # Apply AWK extraction
            awk_fields = self.awk_extractor.extract_fields(document_text, awk_rules)

            logger.debug(f"AWK extraction yielded {len(awk_fields)} fields")
            return awk_fields

        except Exception as e:
            logger.error(f"AWK extraction failed: {str(e)}")
            return {}

    def _get_awk_rules(self) -> List[Dict[str, Any]]:
        """Get AWK extraction rules for this document type."""
        # Default AWK rules - can be overridden by subclasses
        return [
            {
                "field": "date",
                "pattern": r"\d{1,2}/\d{1,2}/\d{4}",
                "line_filter": lambda line: any(
                    word in line.lower() for word in ["date", "dated", "day"]
                ),
                "transform": lambda x: x.strip(),
            },
            {
                "field": "total_amount",
                "pattern": r"\$\d+\.\d{2}",
                "line_filter": lambda line: any(
                    word in line.lower() for word in ["total", "amount", "pay"]
                ),
                "transform": lambda x: x.replace("$", "").strip(),
            },
            {
                "field": "gst_amount",
                "pattern": r"\$\d+\.\d{2}",
                "line_filter": lambda line: "gst" in line.lower(),
                "transform": lambda x: x.replace("$", "").strip(),
            },
        ]

    def _needs_awk_fallback(self, extracted_fields: Dict[str, Any]) -> bool:
        """Determine if AWK fallback is needed."""
        if not extracted_fields:
            return True

        # Check if we have minimum required fields
        required_present = sum(
            1
            for field in self.required_fields
            if field in extracted_fields and extracted_fields[field]
        )

        # Apply AWK fallback if less than 70% of required fields present
        threshold = max(1, len(self.required_fields) * 0.7)
        return required_present < threshold

    def _merge_extractions(
        self, primary_fields: Dict[str, Any], awk_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge primary and AWK extractions."""
        merged_fields = primary_fields.copy()

        # Add AWK fields where primary fields are missing or empty
        for field, value in awk_fields.items():
            if field not in merged_fields or not merged_fields[field]:
                merged_fields[field] = value
                logger.debug(f"Added AWK field: {field} = {value}")

        return merged_fields

    def _validate_fields(self, extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted fields against document requirements."""
        validated_fields = {}

        for field, value in extracted_fields.items():
            if field in self.validation_rules:
                validation_rule = self.validation_rules[field]

                try:
                    # Apply validation rule
                    if callable(validation_rule):
                        validated_value = validation_rule(value)
                    else:
                        validated_value = value

                    validated_fields[field] = validated_value

                except Exception as e:
                    logger.warning(f"Validation failed for field {field}: {str(e)}")
                    validated_fields[field] = value
            else:
                validated_fields[field] = value

        return validated_fields

    def _assess_ato_compliance(
        self, validated_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess ATO compliance for extracted fields."""
        try:
            # Use the enhanced ATO compliance assessment
            compliance_result = assess_ato_compliance_enhanced(
                validated_fields, self.document_type.value
            )

            return compliance_result

        except Exception as e:
            logger.error(f"ATO compliance assessment failed: {str(e)}")
            return {"success": False, "compliance_score": 0, "error": str(e)}

    def _score_confidence(
        self,
        document_text: str,
        classification_result,
        validated_fields: Dict[str, Any],
        ato_compliance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Score confidence for document processing."""
        try:
            # Use the Australian tax confidence scorer
            confidence_result = score_australian_tax_document_processing(
                document_text, classification_result, validated_fields, ato_compliance
            )

            return {
                "overall_confidence": confidence_result.overall_confidence,
                "classification_confidence": confidence_result.classification_confidence,
                "extraction_confidence": confidence_result.extraction_confidence,
                "ato_compliance_confidence": confidence_result.ato_compliance_confidence,
                "australian_business_confidence": confidence_result.australian_business_confidence,
                "quality_grade": confidence_result.quality_grade,
                "is_production_ready": confidence_result.is_production_ready,
                "evidence": confidence_result.evidence,
            }

        except Exception as e:
            logger.error(f"Confidence scoring failed: {str(e)}")
            return {
                "overall_confidence": 0.0,
                "quality_grade": "Failed",
                "is_production_ready": False,
                "error": str(e),
            }

    def _generate_recommendations(
        self,
        validated_fields: Dict[str, Any],
        ato_compliance: Dict[str, Any],
        confidence_score: Dict[str, Any],
    ) -> List[str]:
        """Generate processing recommendations."""
        recommendations = []

        # Field completeness recommendations
        missing_required = [
            field
            for field in self.required_fields
            if field not in validated_fields or not validated_fields[field]
        ]
        if missing_required:
            recommendations.append(
                f"Missing required fields: {', '.join(missing_required)}"
            )

        # ATO compliance recommendations
        if (
            ato_compliance.get("success")
            and ato_compliance.get("compliance_score", 0) < 80
        ):
            recommendations.append(
                "ATO compliance below 80% - additional documentation may be required"
            )

        # Confidence recommendations
        overall_confidence = confidence_score.get("overall_confidence", 0)
        if overall_confidence < 0.7:
            recommendations.append("Low confidence score - manual review recommended")

        # Quality grade recommendations
        quality_grade = confidence_score.get("quality_grade", "Unknown")
        if quality_grade in ["Poor", "Very Poor"]:
            recommendations.append(
                "Poor extraction quality - consider re-scanning document"
            )

        return recommendations

    def _determine_processing_method(self, extracted_fields: Dict[str, Any]) -> str:
        """Determine which processing method was used."""
        if not extracted_fields:
            return "failed"

        # Check if AWK fallback was used
        primary_fields = self._extract_fields_primary("")
        if len(extracted_fields) > len(primary_fields):
            return "hybrid_primary_awk"
        else:
            return "primary_extraction"

    def get_document_requirements(self) -> Dict[str, Any]:
        """Get document requirements summary."""
        return {
            "document_type": self.document_type.value,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "ato_thresholds": self.ato_thresholds,
            "validation_rules": list(self.validation_rules.keys()),
        }
