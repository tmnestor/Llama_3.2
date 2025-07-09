"""
Australian Tax Confidence Scorer for Llama-3.2

This module provides confidence scoring for Australian tax document processing,
ported from the InternVL system to ensure domain expertise parity for fair comparison.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..utils import setup_logging
from .australian_tax_classifier import ClassificationResult, DocumentType

logger = setup_logging()


@dataclass
class ConfidenceScoreResult:
    """Result of confidence scoring analysis."""

    overall_confidence: float
    classification_confidence: float
    extraction_confidence: float
    ato_compliance_confidence: float
    australian_business_confidence: float
    quality_grade: str
    evidence: Dict[str, Any]
    recommendations: List[str]
    is_production_ready: bool


class AustralianTaxConfidenceScorer:
    """Confidence scorer for Australian tax document processing."""

    def __init__(self):
        """Initialize confidence scorer with Australian tax expertise."""

        # Confidence thresholds
        self.confidence_thresholds = {
            "excellent": 0.9,
            "good": 0.7,
            "fair": 0.5,
            "poor": 0.3,
            "production_ready": 0.7,
        }

        # Scoring weights
        self.scoring_weights = {
            "classification": 0.25,  # 25% - Document type classification
            "extraction": 0.35,  # 35% - Field extraction quality
            "ato_compliance": 0.25,  # 25% - ATO compliance indicators
            "australian_business": 0.15,  # 15% - Australian business recognition
        }

        # Australian business confidence indicators
        self.australian_business_indicators = {
            "major_retailers": {
                "names": [
                    "woolworths",
                    "coles",
                    "aldi",
                    "target",
                    "kmart",
                    "bunnings",
                    "officeworks",
                ],
                "confidence_bonus": 0.3,
            },
            "fuel_stations": {
                "names": ["bp", "shell", "caltex", "ampol", "mobil", "7-eleven"],
                "confidence_bonus": 0.25,
            },
            "banks": {
                "names": [
                    "anz",
                    "commonwealth bank",
                    "westpac",
                    "nab",
                    "ing",
                    "macquarie",
                ],
                "confidence_bonus": 0.25,
            },
            "airlines": {
                "names": ["qantas", "jetstar", "virgin australia", "tigerair"],
                "confidence_bonus": 0.2,
            },
            "hotels": {
                "names": ["hilton", "marriott", "hyatt", "ibis", "mercure", "novotel"],
                "confidence_bonus": 0.2,
            },
            "professional_services": {
                "names": ["deloitte", "pwc", "kpmg", "ey", "bdo", "rsm"],
                "confidence_bonus": 0.3,
            },
        }

        # ATO compliance indicators with weights
        self.ato_compliance_indicators = {
            "abn_present": {
                "pattern": r"abn:?\s*\d{2}\s?\d{3}\s?\d{3}\s?\d{3}",
                "weight": 0.2,
                "description": "Valid ABN format detected",
            },
            "gst_breakdown": {
                "pattern": r"gst:?\s*\$?\d+\.\d{2}",
                "weight": 0.15,
                "description": "GST amount breakdown present",
            },
            "tax_invoice_header": {
                "pattern": r"tax invoice",
                "weight": 0.15,
                "description": "Tax invoice header present",
            },
            "australian_date": {
                "pattern": r"\d{1,2}/\d{1,2}/\d{4}",
                "weight": 0.1,
                "description": "Australian date format (DD/MM/YYYY)",
            },
            "business_name_format": {
                "pattern": r"[A-Z][A-Z\s&]+(?:PTY\s+LTD|LIMITED|COMPANY)",
                "weight": 0.1,
                "description": "Proper business name format",
            },
            "currency_format": {
                "pattern": r"\$\d+\.\d{2}",
                "weight": 0.1,
                "description": "Australian currency format",
            },
            "bsb_code": {
                "pattern": r"bsb:?\s*\d{2,3}-\d{3}",
                "weight": 0.1,
                "description": "Australian BSB code format",
            },
            "fuel_specifics": {
                "pattern": r"\d+\.\d{3}l|\d+\.\d{2}l|unleaded|diesel|premium",
                "weight": 0.1,
                "description": "Fuel-specific indicators",
            },
        }

        # Field extraction quality indicators
        self.extraction_quality_indicators = {
            "required_fields_present": 0.4,
            "field_format_valid": 0.3,
            "field_completeness": 0.2,
            "field_consistency": 0.1,
        }

        # Document type specific requirements
        self.document_requirements = {
            DocumentType.BUSINESS_RECEIPT: {
                "critical_fields": ["date", "store_name", "total_amount"],
                "important_fields": ["gst_amount", "items"],
                "abn_required_threshold": 82.50,
            },
            DocumentType.FUEL_RECEIPT: {
                "critical_fields": [
                    "date",
                    "station_name",
                    "fuel_type",
                    "litres",
                    "total_amount",
                ],
                "important_fields": ["price_per_litre", "gst_amount"],
                "abn_required_threshold": 82.50,
            },
            DocumentType.TAX_INVOICE: {
                "critical_fields": [
                    "date",
                    "supplier_name",
                    "supplier_abn",
                    "gst_amount",
                    "total_amount",
                ],
                "important_fields": ["invoice_number", "description"],
                "abn_required_threshold": 0.0,  # Always required
            },
            DocumentType.BANK_STATEMENT: {
                "critical_fields": ["bank_name", "account_holder", "statement_period"],
                "important_fields": ["bsb", "account_number", "transactions"],
                "abn_required_threshold": float("inf"),  # Not applicable
            },
            DocumentType.MEAL_RECEIPT: {
                "critical_fields": ["date", "restaurant_name", "total_amount"],
                "important_fields": ["gst_amount", "meal_type", "items"],
                "abn_required_threshold": 82.50,
            },
            DocumentType.ACCOMMODATION: {
                "critical_fields": ["date", "hotel_name", "total_amount", "nights"],
                "important_fields": ["gst_amount", "room_type"],
                "abn_required_threshold": 82.50,
            },
            DocumentType.PROFESSIONAL_SERVICES: {
                "critical_fields": [
                    "date",
                    "firm_name",
                    "firm_abn",
                    "total_amount",
                    "description",
                ],
                "important_fields": ["hours", "rate", "service_period"],
                "abn_required_threshold": 0.0,  # Always required
            },
        }

        logger.info("Australian Tax Confidence Scorer initialized")

    def score_document_processing(
        self,
        text: str,
        classification_result: ClassificationResult,
        extracted_fields: Dict[str, Any],
        ato_compliance_result: Optional[Dict[str, Any]] = None,
    ) -> ConfidenceScoreResult:
        """
        Score confidence for Australian tax document processing.

        Args:
            text: Original document text
            classification_result: Document classification result
            extracted_fields: Dictionary of extracted fields
            ato_compliance_result: ATO compliance assessment result

        Returns:
            ConfidenceScoreResult with comprehensive confidence assessment
        """
        logger.debug("Scoring document processing confidence")

        # Score each component
        classification_score = self._score_classification(classification_result)
        extraction_score = self._score_extraction(
            extracted_fields, classification_result.document_type
        )
        ato_compliance_score = self._score_ato_compliance(
            text, extracted_fields, ato_compliance_result
        )
        australian_business_score = self._score_australian_business(
            text, classification_result
        )

        # Calculate weighted overall confidence
        overall_confidence = (
            classification_score * self.scoring_weights["classification"]
            + extraction_score * self.scoring_weights["extraction"]
            + ato_compliance_score * self.scoring_weights["ato_compliance"]
            + australian_business_score * self.scoring_weights["australian_business"]
        )

        # Get quality grade
        quality_grade = self._get_quality_grade(overall_confidence)

        # Determine if production ready
        is_production_ready = (
            overall_confidence >= self.confidence_thresholds["production_ready"]
            and classification_score >= 0.6
            and extraction_score >= 0.6
        )

        # Collect evidence
        evidence = {
            "classification_score": classification_score,
            "extraction_score": extraction_score,
            "ato_compliance_score": ato_compliance_score,
            "australian_business_score": australian_business_score,
            "classification_evidence": classification_result.evidence,
            "australian_business_detected": classification_result.australian_business_detected,
            "ato_indicators": classification_result.ato_compliance_indicators,
            "scoring_weights": self.scoring_weights,
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_confidence,
            classification_score,
            extraction_score,
            ato_compliance_score,
            australian_business_score,
            classification_result,
            extracted_fields,
        )

        result = ConfidenceScoreResult(
            overall_confidence=overall_confidence,
            classification_confidence=classification_score,
            extraction_confidence=extraction_score,
            ato_compliance_confidence=ato_compliance_score,
            australian_business_confidence=australian_business_score,
            quality_grade=quality_grade,
            evidence=evidence,
            recommendations=recommendations,
            is_production_ready=is_production_ready,
        )

        logger.info(
            f"Document processing confidence: {overall_confidence:.2f} ({quality_grade})"
        )
        return result

    def _score_classification(
        self, classification_result: ClassificationResult
    ) -> float:
        """Score classification confidence."""

        base_score = classification_result.confidence_score

        # Bonus for high confidence classification
        if classification_result.confidence.value == "High":
            base_score += 0.1
        elif classification_result.confidence.value == "Medium":
            base_score += 0.05

        # Penalty for competing types
        if classification_result.secondary_type:
            base_score *= 0.9

        # Bonus for Australian business detection
        if classification_result.australian_business_detected:
            base_score += 0.1

        # Bonus for ATO compliance indicators
        if classification_result.ato_compliance_indicators:
            indicator_bonus = min(
                len(classification_result.ato_compliance_indicators) * 0.05, 0.2
            )
            base_score += indicator_bonus

        return min(base_score, 1.0)

    def _score_extraction(
        self, extracted_fields: Dict[str, Any], document_type: DocumentType
    ) -> float:
        """Score extraction quality."""

        requirements = self.document_requirements.get(document_type, {})
        critical_fields = requirements.get("critical_fields", [])
        important_fields = requirements.get("important_fields", [])

        if not critical_fields and not important_fields:
            return 0.7  # Default score for unknown document types

        # Score critical fields
        critical_score = 0.0
        critical_present = 0

        for field in critical_fields:
            if field in extracted_fields and extracted_fields[field]:
                critical_present += 1
                # Bonus for valid field format
                if self._is_field_format_valid(field, extracted_fields[field]):
                    critical_score += 1.0
                else:
                    critical_score += (
                        0.7  # Partial credit for present but invalid format
                    )

        if critical_fields:
            critical_score /= len(critical_fields)

        # Score important fields
        important_score = 0.0
        important_present = 0

        for field in important_fields:
            if field in extracted_fields and extracted_fields[field]:
                important_present += 1
                if self._is_field_format_valid(field, extracted_fields[field]):
                    important_score += 1.0
                else:
                    important_score += 0.7

        if important_fields:
            important_score /= len(important_fields)

        # Weighted combination (critical fields are more important)
        extraction_score = critical_score * 0.7 + important_score * 0.3

        # Bonus for field completeness
        total_fields = len(critical_fields) + len(important_fields)
        total_present = critical_present + important_present

        if total_fields > 0:
            completeness_bonus = (total_present / total_fields) * 0.1
            extraction_score += completeness_bonus

        return min(extraction_score, 1.0)

    def _score_ato_compliance(
        self,
        text: str,
        extracted_fields: Dict[str, Any],
        ato_compliance_result: Optional[Dict[str, Any]],
    ) -> float:
        """Score ATO compliance indicators."""

        compliance_score = 0.0

        # Use ATO compliance result if available
        if ato_compliance_result and ato_compliance_result.get("success"):
            compliance_score = (
                ato_compliance_result.get("compliance_score", 0.0) / 100.0
            )
        else:
            # Fallback to pattern-based scoring
            for _indicator, config in self.ato_compliance_indicators.items():
                if re.search(config["pattern"], text, re.IGNORECASE):
                    compliance_score += config["weight"]

        # Additional scoring based on extracted fields
        if extracted_fields:
            # ABN validation
            abn_field = extracted_fields.get("supplier_abn") or extracted_fields.get(
                "abn"
            )
            if abn_field and self._is_valid_abn(abn_field):
                compliance_score += 0.15

            # GST calculation validation
            if self._validate_gst_calculation(extracted_fields):
                compliance_score += 0.1

            # Date format validation
            date_field = extracted_fields.get("date")
            if date_field and self._is_valid_australian_date(date_field):
                compliance_score += 0.05

        return min(compliance_score, 1.0)

    def _score_australian_business(
        self, text: str, classification_result: ClassificationResult
    ) -> float:
        """Score Australian business recognition."""

        business_score = 0.0
        text_lower = text.lower()

        # Score based on business categories
        for _category, config in self.australian_business_indicators.items():
            for business_name in config["names"]:
                if business_name in text_lower:
                    business_score += config["confidence_bonus"]
                    break  # Only count once per category

        # Bonus from classification result
        if classification_result.australian_business_detected:
            business_score += 0.2

        # Bonus for business format indicators
        if re.search(r"pty\s+ltd|limited|company", text_lower):
            business_score += 0.1

        # Bonus for Australian-specific terms
        australian_terms = [
            "australia",
            "australian",
            "sydney",
            "melbourne",
            "brisbane",
            "perth",
            "adelaide",
        ]
        for term in australian_terms:
            if term in text_lower:
                business_score += 0.05
                break

        return min(business_score, 1.0)

    def _is_field_format_valid(self, field: str, value: str) -> bool:
        """Check if field format is valid."""

        if not value:
            return False

        value_str = str(value).strip()

        # Field-specific validation
        if field in ["date"]:
            return self._is_valid_australian_date(value_str)
        elif field in ["supplier_abn", "abn"]:
            return self._is_valid_abn(value_str)
        elif field in ["total_amount", "gst_amount", "subtotal"]:
            return self._is_valid_currency_amount(value_str)
        elif field in ["bsb"]:
            return self._is_valid_bsb(value_str)
        elif field in ["litres"]:
            return self._is_valid_fuel_quantity(value_str)
        else:
            return len(value_str) > 0

    def _validate_gst_calculation(self, extracted_fields: Dict[str, Any]) -> bool:
        """Validate GST calculation."""

        try:
            subtotal = extracted_fields.get("subtotal", "0")
            gst = extracted_fields.get("gst_amount", "0")
            total = extracted_fields.get("total_amount", "0")

            if not all([subtotal, gst, total]):
                return False

            subtotal_val = float(re.sub(r"[^\d.]", "", str(subtotal)))
            gst_val = float(re.sub(r"[^\d.]", "", str(gst)))

            # Check if GST is approximately 10% of subtotal
            expected_gst = subtotal_val * 0.10
            gst_difference = abs(gst_val - expected_gst)

            return gst_difference <= 0.02  # 2 cent tolerance

        except (ValueError, TypeError):
            return False

    def _is_valid_australian_date(self, date_str: str) -> bool:
        """Check if date is in valid Australian format."""
        patterns = [
            r"^\d{1,2}/\d{1,2}/\d{4}$",
            r"^\d{1,2}-\d{1,2}-\d{4}$",
            r"^\d{1,2}\.\d{1,2}\.\d{4}$",
        ]

        for pattern in patterns:
            if re.match(pattern, date_str.strip()):
                return True

        return False

    def _is_valid_abn(self, abn_str: str) -> bool:
        """Check if ABN is in valid format."""
        clean_abn = re.sub(r"[^\d]", "", abn_str.strip())
        return len(clean_abn) == 11

    def _is_valid_currency_amount(self, amount_str: str) -> bool:
        """Check if currency amount is valid."""
        try:
            clean_amount = re.sub(r"[^\d.]", "", amount_str.strip())
            amount = float(clean_amount)
            return 0.01 <= amount <= 100000.0
        except (ValueError, TypeError):
            return False

    def _is_valid_bsb(self, bsb_str: str) -> bool:
        """Check if BSB is in valid format."""
        return bool(re.match(r"^\d{2,3}-\d{3}$", bsb_str.strip()))

    def _is_valid_fuel_quantity(self, qty_str: str) -> bool:
        """Check if fuel quantity is valid."""
        patterns = [
            r"^\d+\.\d{3}L?$",  # 45.230L
            r"^\d+\.\d{2}L?$",  # 45.23L
            r"^\d+L?$",  # 45L
        ]

        for pattern in patterns:
            if re.match(pattern, qty_str.strip(), re.IGNORECASE):
                return True

        return False

    def _get_quality_grade(self, confidence: float) -> str:
        """Get quality grade based on confidence score."""
        if confidence >= self.confidence_thresholds["excellent"]:
            return "Excellent"
        elif confidence >= self.confidence_thresholds["good"]:
            return "Good"
        elif confidence >= self.confidence_thresholds["fair"]:
            return "Fair"
        elif confidence >= self.confidence_thresholds["poor"]:
            return "Poor"
        else:
            return "Very Poor"

    def _generate_recommendations(
        self,
        overall_confidence: float,
        classification_score: float,
        extraction_score: float,
        ato_compliance_score: float,
        australian_business_score: float,
        classification_result: ClassificationResult,
        extracted_fields: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations for improving confidence."""

        recommendations = []

        # Overall confidence recommendations
        if overall_confidence < self.confidence_thresholds["production_ready"]:
            recommendations.append(
                "Document processing confidence below production threshold"
            )

        # Classification recommendations
        if classification_score < 0.6:
            recommendations.append(
                "Low classification confidence - consider manual review"
            )
            if not classification_result.australian_business_detected:
                recommendations.append("Australian business not clearly identified")

        # Extraction recommendations
        if extraction_score < 0.6:
            recommendations.append(
                "Low extraction confidence - check field completeness"
            )
            requirements = self.document_requirements.get(
                classification_result.document_type, {}
            )
            critical_fields = requirements.get("critical_fields", [])
            for field in critical_fields:
                if field not in extracted_fields or not extracted_fields[field]:
                    recommendations.append(f"Missing critical field: {field}")

        # ATO compliance recommendations
        if ato_compliance_score < 0.6:
            recommendations.append(
                "Low ATO compliance indicators - verify document authenticity"
            )
            if not any(
                "abn" in str(field).lower() for field in extracted_fields.keys()
            ):
                recommendations.append(
                    "ABN not detected - required for business expense claims"
                )

        # Australian business recommendations
        if australian_business_score < 0.4:
            recommendations.append(
                "Australian business context unclear - verify document origin"
            )

        # Success recommendations
        if overall_confidence >= self.confidence_thresholds["excellent"]:
            recommendations.append(
                "Excellent processing confidence - suitable for automated processing"
            )
        elif overall_confidence >= self.confidence_thresholds["good"]:
            recommendations.append(
                "Good processing confidence - suitable for production use"
            )

        return recommendations


# Global instance for easy access
australian_tax_confidence_scorer = AustralianTaxConfidenceScorer()


def score_australian_tax_document_processing(
    text: str,
    classification_result: ClassificationResult,
    extracted_fields: Dict[str, Any],
    ato_compliance_result: Optional[Dict[str, Any]] = None,
) -> ConfidenceScoreResult:
    """
    Score confidence for Australian tax document processing.

    Args:
        text: Original document text
        classification_result: Document classification result
        extracted_fields: Dictionary of extracted fields
        ato_compliance_result: ATO compliance assessment result

    Returns:
        ConfidenceScoreResult with comprehensive confidence assessment
    """
    return australian_tax_confidence_scorer.score_document_processing(
        text, classification_result, extracted_fields, ato_compliance_result
    )


def get_confidence_thresholds() -> Dict[str, float]:
    """Get confidence thresholds for quality assessment."""
    return australian_tax_confidence_scorer.confidence_thresholds


def get_scoring_weights() -> Dict[str, float]:
    """Get scoring weights for confidence components."""
    return australian_tax_confidence_scorer.scoring_weights
