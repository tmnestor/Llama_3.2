"""
Confidence Integration Manager for Australian Tax Document Processing

This manager integrates confidence scoring throughout the processing pipeline
to provide production-ready assessment and quality control.
"""

import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from ..utils import setup_logging
from .hybrid_extraction_manager import HybridExtractionResult, hybrid_extraction_manager

logger = setup_logging()


class ProductionReadinessLevel(Enum):
    """Production readiness levels based on confidence scores."""

    EXCELLENT = "excellent"  # 90%+ confidence
    GOOD = "good"  # 70-89% confidence
    FAIR = "fair"  # 50-69% confidence
    POOR = "poor"  # 30-49% confidence
    VERY_POOR = "very_poor"  # <30% confidence


@dataclass
class ConfidenceIntegrationResult:
    """Result of confidence integration analysis."""

    overall_confidence: float
    production_readiness: ProductionReadinessLevel
    quality_grade: str
    component_scores: Dict[str, float]
    recommendations: List[str]
    processing_decision: str
    quality_flags: List[str]
    ato_compliance_status: Dict[str, Any]
    australian_business_context: Dict[str, Any]
    extraction_quality_metrics: Dict[str, Any]


class ConfidenceIntegrationManager:
    """Manager for integrated confidence scoring and production readiness assessment."""

    def __init__(self):
        """Initialize confidence integration manager."""

        # Production readiness thresholds
        self.readiness_thresholds = {
            ProductionReadinessLevel.EXCELLENT: 0.90,
            ProductionReadinessLevel.GOOD: 0.70,
            ProductionReadinessLevel.FAIR: 0.50,
            ProductionReadinessLevel.POOR: 0.30,
            ProductionReadinessLevel.VERY_POOR: 0.0,
        }

        # Component weight configuration
        self.component_weights = {
            "classification": 0.25,  # 25% - Document type classification
            "extraction": 0.35,  # 35% - Field extraction quality
            "ato_compliance": 0.25,  # 25% - ATO compliance indicators
            "australian_business": 0.15,  # 15% - Australian business recognition
        }

        # Quality control thresholds
        self.quality_thresholds = {
            "minimum_production_confidence": 0.70,
            "minimum_extraction_fields": 3,
            "minimum_ato_compliance": 0.60,
            "minimum_classification_confidence": 0.60,
        }

        # Processing decision rules
        self.processing_rules = {
            "auto_approve_threshold": 0.90,
            "manual_review_threshold": 0.70,
            "reject_threshold": 0.30,
            "ato_compliance_required": 0.80,
        }

        logger.info("ConfidenceIntegrationManager initialized")

    def assess_document_confidence(
        self,
        document_text: str,
        extraction_result: Optional[HybridExtractionResult] = None,
        image_path: Optional[str] = None,
    ) -> ConfidenceIntegrationResult:
        """
        Assess document confidence with integrated scoring.

        Args:
            document_text: Document text content
            extraction_result: Optional pre-computed extraction result
            image_path: Optional path to document image

        Returns:
            ConfidenceIntegrationResult with comprehensive assessment
        """
        try:
            logger.info("Starting integrated confidence assessment")

            # Step 1: Get extraction result if not provided
            if extraction_result is None:
                extraction_result = hybrid_extraction_manager.process_document(
                    document_text, image_path
                )

            # Step 2: Extract confidence components
            confidence_components = self._extract_confidence_components(
                extraction_result
            )

            # Step 3: Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                confidence_components
            )

            # Step 4: Determine production readiness
            production_readiness = self._determine_production_readiness(
                overall_confidence
            )

            # Step 5: Generate quality flags
            quality_flags = self._generate_quality_flags(
                extraction_result, confidence_components
            )

            # Step 6: Make processing decision
            processing_decision = self._make_processing_decision(
                overall_confidence, confidence_components, quality_flags
            )

            # Step 7: Generate recommendations
            recommendations = self._generate_integrated_recommendations(
                extraction_result, confidence_components, quality_flags
            )

            # Step 8: Extract detailed metrics
            ato_compliance_status = self._extract_ato_compliance_status(
                extraction_result
            )
            australian_business_context = self._extract_australian_business_context(
                extraction_result
            )
            extraction_quality_metrics = self._extract_extraction_quality_metrics(
                extraction_result
            )

            # Step 9: Create integrated result
            result = ConfidenceIntegrationResult(
                overall_confidence=overall_confidence,
                production_readiness=production_readiness,
                quality_grade=extraction_result.confidence_summary.get(
                    "quality_grade", "Unknown"
                ),
                component_scores=confidence_components,
                recommendations=recommendations,
                processing_decision=processing_decision,
                quality_flags=quality_flags,
                ato_compliance_status=ato_compliance_status,
                australian_business_context=australian_business_context,
                extraction_quality_metrics=extraction_quality_metrics,
            )

            logger.info(
                f"Confidence assessment completed: {overall_confidence:.2f} ({production_readiness.value})"
            )
            return result

        except Exception as e:
            logger.error(f"Confidence assessment failed: {str(e)}")
            logger.error(traceback.format_exc())

            return ConfidenceIntegrationResult(
                overall_confidence=0.0,
                production_readiness=ProductionReadinessLevel.VERY_POOR,
                quality_grade="Failed",
                component_scores={},
                recommendations=[
                    "Confidence assessment failed - manual review required"
                ],
                processing_decision="reject",
                quality_flags=["assessment_failed"],
                ato_compliance_status={},
                australian_business_context={},
                extraction_quality_metrics={},
            )

    def assess_batch_confidence(
        self, documents: List[Dict[str, Any]]
    ) -> List[ConfidenceIntegrationResult]:
        """
        Assess confidence for multiple documents in batch.

        Args:
            documents: List of document dictionaries

        Returns:
            List of ConfidenceIntegrationResult objects
        """
        results = []

        logger.info(
            f"Starting batch confidence assessment for {len(documents)} documents"
        )

        for i, doc in enumerate(documents):
            try:
                document_text = doc.get("text", "")
                image_path = doc.get("image_path")

                logger.info(
                    f"Assessing confidence for document {i + 1}/{len(documents)}"
                )

                result = self.assess_document_confidence(
                    document_text, None, image_path
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error assessing document {i + 1}: {str(e)}")
                error_result = ConfidenceIntegrationResult(
                    overall_confidence=0.0,
                    production_readiness=ProductionReadinessLevel.VERY_POOR,
                    quality_grade="Failed",
                    component_scores={},
                    recommendations=["Assessment failed - manual review required"],
                    processing_decision="reject",
                    quality_flags=["assessment_failed"],
                    ato_compliance_status={},
                    australian_business_context={},
                    extraction_quality_metrics={},
                )
                results.append(error_result)

        logger.info("Batch confidence assessment completed")
        return results

    def generate_production_report(
        self, results: List[ConfidenceIntegrationResult]
    ) -> Dict[str, Any]:
        """
        Generate production readiness report from confidence results.

        Args:
            results: List of ConfidenceIntegrationResult objects

        Returns:
            Dictionary with production readiness report
        """
        if not results:
            return {}

        # Production readiness distribution
        readiness_distribution = {}
        for result in results:
            readiness = result.production_readiness.value
            readiness_distribution[readiness] = (
                readiness_distribution.get(readiness, 0) + 1
            )

        # Processing decision distribution
        decision_distribution = {}
        for result in results:
            decision = result.processing_decision
            decision_distribution[decision] = decision_distribution.get(decision, 0) + 1

        # Quality flags summary
        quality_flags_summary = {}
        for result in results:
            for flag in result.quality_flags:
                quality_flags_summary[flag] = quality_flags_summary.get(flag, 0) + 1

        # Average confidence scores
        avg_overall_confidence = sum(r.overall_confidence for r in results) / len(
            results
        )

        # Component averages
        component_averages = {}
        for component in self.component_weights.keys():
            scores = [r.component_scores.get(component, 0.0) for r in results]
            component_averages[component] = sum(scores) / len(scores)

        # Production ready count
        production_ready = sum(
            1
            for r in results
            if r.production_readiness
            in [ProductionReadinessLevel.EXCELLENT, ProductionReadinessLevel.GOOD]
        )

        # ATO compliance summary
        ato_compliant = sum(
            1
            for r in results
            if r.ato_compliance_status.get("compliance_score", 0) >= 80
        )

        return {
            "total_documents": len(results),
            "production_ready": production_ready,
            "production_ready_percentage": (production_ready / len(results)) * 100,
            "ato_compliant": ato_compliant,
            "ato_compliance_percentage": (ato_compliant / len(results)) * 100,
            "average_overall_confidence": avg_overall_confidence,
            "component_averages": component_averages,
            "readiness_distribution": readiness_distribution,
            "decision_distribution": decision_distribution,
            "quality_flags_summary": quality_flags_summary,
            "quality_thresholds": self.quality_thresholds,
            "processing_rules": self.processing_rules,
        }

    def _extract_confidence_components(
        self, extraction_result: HybridExtractionResult
    ) -> Dict[str, float]:
        """Extract confidence components from extraction result."""
        confidence_summary = extraction_result.confidence_summary

        return {
            "classification": confidence_summary.get("classification_confidence", 0.0),
            "extraction": confidence_summary.get("extraction_confidence", 0.0),
            "ato_compliance": confidence_summary.get("ato_compliance_confidence", 0.0),
            "australian_business": confidence_summary.get(
                "australian_business_confidence", 0.0
            ),
        }

    def _calculate_overall_confidence(self, components: Dict[str, float]) -> float:
        """Calculate weighted overall confidence score."""
        overall_confidence = 0.0

        for component, weight in self.component_weights.items():
            component_score = components.get(component, 0.0)
            overall_confidence += component_score * weight

        return min(overall_confidence, 1.0)

    def _determine_production_readiness(
        self, overall_confidence: float
    ) -> ProductionReadinessLevel:
        """Determine production readiness level."""
        for level, threshold in self.readiness_thresholds.items():
            if overall_confidence >= threshold:
                return level

        return ProductionReadinessLevel.VERY_POOR

    def _generate_quality_flags(
        self, extraction_result: HybridExtractionResult, components: Dict[str, float]
    ) -> List[str]:
        """Generate quality flags based on analysis."""
        flags = []

        # Extraction quality flags
        if not extraction_result.success:
            flags.append("extraction_failed")

        if (
            components.get("extraction", 0.0)
            < self.quality_thresholds["minimum_extraction_fields"]
        ):
            flags.append("insufficient_fields")

        if (
            components.get("classification", 0.0)
            < self.quality_thresholds["minimum_classification_confidence"]
        ):
            flags.append("low_classification_confidence")

        if (
            components.get("ato_compliance", 0.0)
            < self.quality_thresholds["minimum_ato_compliance"]
        ):
            flags.append("low_ato_compliance")

        # Processing method flags
        if extraction_result.extraction_method == "hybrid_primary_awk":
            flags.append("awk_fallback_used")

        # Confidence flags
        if (
            extraction_result.confidence_summary.get("overall_confidence", 0.0)
            < self.quality_thresholds["minimum_production_confidence"]
        ):
            flags.append("below_production_threshold")

        return flags

    def _make_processing_decision(
        self,
        overall_confidence: float,
        _components: Dict[str, float],
        quality_flags: List[str],
    ) -> str:
        """Make processing decision based on confidence and flags."""

        # Auto-approve if high confidence
        if overall_confidence >= self.processing_rules["auto_approve_threshold"]:
            return "auto_approve"

        # Reject if very low confidence
        if overall_confidence < self.processing_rules["reject_threshold"]:
            return "reject"

        # Check for critical flags
        critical_flags = ["extraction_failed", "assessment_failed"]
        if any(flag in quality_flags for flag in critical_flags):
            return "reject"

        # Manual review for medium confidence
        if overall_confidence >= self.processing_rules["manual_review_threshold"]:
            return "manual_review"

        return "reject"

    def _generate_integrated_recommendations(
        self,
        extraction_result: HybridExtractionResult,
        components: Dict[str, float],
        quality_flags: List[str],
    ) -> List[str]:
        """Generate integrated recommendations."""
        recommendations = []

        # Add extraction result recommendations
        if extraction_result.success:
            recommendations.extend(extraction_result.processing_result.recommendations)

        # Add component-specific recommendations
        if components.get("classification", 0.0) < 0.6:
            recommendations.append("Consider manual document type verification")

        if components.get("extraction", 0.0) < 0.6:
            recommendations.append("Review extracted fields for completeness")

        if components.get("ato_compliance", 0.0) < 0.6:
            recommendations.append("Verify ATO compliance requirements")

        # Add quality flag recommendations
        if "awk_fallback_used" in quality_flags:
            recommendations.append("Primary extraction failed - AWK fallback was used")

        if "below_production_threshold" in quality_flags:
            recommendations.append("Document confidence below production threshold")

        return list(set(recommendations))  # Remove duplicates

    def _extract_ato_compliance_status(
        self, extraction_result: HybridExtractionResult
    ) -> Dict[str, Any]:
        """Extract ATO compliance status."""
        if not extraction_result.success:
            return {}

        ato_compliance = extraction_result.processing_result.ato_compliance

        return {
            "compliance_score": ato_compliance.get("compliance_score", 0),
            "required_fields_present": ato_compliance.get(
                "required_fields_present", []
            ),
            "missing_fields": ato_compliance.get("missing_fields", []),
            "validation_errors": ato_compliance.get("validation_errors", []),
            "recommendations": ato_compliance.get("recommendations", []),
        }

    def _extract_australian_business_context(
        self, extraction_result: HybridExtractionResult
    ) -> Dict[str, Any]:
        """Extract Australian business context."""
        if not extraction_result.success:
            return {}

        return {
            "document_type": extraction_result.document_type.value,
            "business_detected": extraction_result.confidence_summary.get(
                "australian_business_confidence", 0.0
            )
            > 0.5,
            "business_confidence": extraction_result.confidence_summary.get(
                "australian_business_confidence", 0.0
            ),
            "handler_used": extraction_result.handler_used,
        }

    def _extract_extraction_quality_metrics(
        self, extraction_result: HybridExtractionResult
    ) -> Dict[str, Any]:
        """Extract extraction quality metrics."""
        if not extraction_result.success:
            return {}

        extracted_fields = extraction_result.processing_result.extracted_fields

        return {
            "fields_extracted": len(extracted_fields),
            "extraction_method": extraction_result.extraction_method,
            "processing_time": extraction_result.processing_time,
            "quality_grade": extraction_result.confidence_summary.get(
                "quality_grade", "Unknown"
            ),
            "is_production_ready": extraction_result.confidence_summary.get(
                "is_production_ready", False
            ),
        }

    def update_thresholds(self, threshold_updates: Dict[str, Any]) -> None:
        """Update quality and processing thresholds."""
        if "quality_thresholds" in threshold_updates:
            self.quality_thresholds.update(threshold_updates["quality_thresholds"])

        if "processing_rules" in threshold_updates:
            self.processing_rules.update(threshold_updates["processing_rules"])

        if "component_weights" in threshold_updates:
            self.component_weights.update(threshold_updates["component_weights"])

        logger.info(f"Thresholds updated: {threshold_updates}")

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "readiness_thresholds": {
                level.value: threshold
                for level, threshold in self.readiness_thresholds.items()
            },
            "component_weights": self.component_weights,
            "quality_thresholds": self.quality_thresholds,
            "processing_rules": self.processing_rules,
        }


# Global instance for easy access
confidence_integration_manager = ConfidenceIntegrationManager()


def assess_document_confidence(
    document_text: str,
    extraction_result: Optional[HybridExtractionResult] = None,
    image_path: Optional[str] = None,
) -> ConfidenceIntegrationResult:
    """
    Assess document confidence with integrated scoring.

    Args:
        document_text: Document text content
        extraction_result: Optional pre-computed extraction result
        image_path: Optional path to document image

    Returns:
        ConfidenceIntegrationResult with comprehensive assessment
    """
    return confidence_integration_manager.assess_document_confidence(
        document_text, extraction_result, image_path
    )


def generate_production_report(
    results: List[ConfidenceIntegrationResult],
) -> Dict[str, Any]:
    """
    Generate production readiness report.

    Args:
        results: List of ConfidenceIntegrationResult objects

    Returns:
        Dictionary with production readiness report
    """
    return confidence_integration_manager.generate_production_report(results)
