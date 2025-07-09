"""
Australian Tax Document Classifier for Llama-3.2

This module provides document classification capabilities specifically for Australian
tax documents, ported from the InternVL system to ensure domain expertise parity.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..utils import setup_logging

logger = setup_logging()


class DocumentType(Enum):
    """Australian tax document types."""

    BUSINESS_RECEIPT = "business_receipt"
    TAX_INVOICE = "tax_invoice"
    BANK_STATEMENT = "bank_statement"
    FUEL_RECEIPT = "fuel_receipt"
    MEAL_RECEIPT = "meal_receipt"
    ACCOMMODATION = "accommodation"
    TRAVEL_DOCUMENT = "travel_document"
    PARKING_TOLL = "parking_toll"
    EQUIPMENT_SUPPLIES = "equipment_supplies"
    PROFESSIONAL_SERVICES = "professional_services"
    OTHER = "other"


class ConfidenceLevel(Enum):
    """Confidence levels for classification."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class ClassificationResult:
    """Result of document classification."""

    document_type: DocumentType
    confidence: ConfidenceLevel
    confidence_score: float
    reasoning: str
    secondary_type: Optional[DocumentType] = None
    evidence: Dict[str, Any] = None
    australian_business_detected: bool = False
    ato_compliance_indicators: List[str] = None


class AustralianTaxDocumentClassifier:
    """Classifier for Australian tax documents with domain expertise."""

    def __init__(self):
        """Initialize classifier with Australian business knowledge."""

        # Australian business classification keywords
        self.classification_keywords = {
            DocumentType.BUSINESS_RECEIPT: [
                "woolworths",
                "coles",
                "aldi",
                "target",
                "kmart",
                "bunnings",
                "officeworks",
                "harvey norman",
                "jb hi-fi",
                "big w",
                "myer",
                "david jones",
                "ikea",
                "spotlight",
                "rebel sport",
                "chemist warehouse",
                "priceline",
                "terry white",
                "dan murphy",
                "bws",
                "liquorland",
            ],
            DocumentType.FUEL_RECEIPT: [
                "bp",
                "shell",
                "caltex",
                "ampol",
                "mobil",
                "7-eleven",
                "united petroleum",
                "liberty",
                "metro petroleum",
                "speedway",
                "fuel",
                "petrol",
                "diesel",
                "unleaded",
                "premium",
                "litres",
                "pump",
                "station",
            ],
            DocumentType.TAX_INVOICE: [
                "tax invoice",
                "gst invoice",
                "invoice",
                "abn",
                "tax invoice number",
                "invoice number",
                "supplier",
                "customer",
                "subtotal",
                "gst amount",
                "professional services",
                "consulting",
                "advisory",
            ],
            DocumentType.BANK_STATEMENT: [
                "anz",
                "commonwealth bank",
                "westpac",
                "nab",
                "ing",
                "macquarie",
                "bendigo bank",
                "suncorp",
                "bank of queensland",
                "credit union",
                "account statement",
                "transaction history",
                "bsb",
                "account number",
                "opening balance",
                "closing balance",
                "statement period",
            ],
            DocumentType.MEAL_RECEIPT: [
                "restaurant",
                "cafe",
                "bistro",
                "bar",
                "pub",
                "club",
                "hotel",
                "mcdonald's",
                "kfc",
                "subway",
                "domino's",
                "pizza hut",
                "hungry jack's",
                "red rooster",
                "nando's",
                "guzman y gomez",
                "zambrero",
                "starbucks",
                "gloria jean's",
                "coffee",
                "breakfast",
                "lunch",
                "dinner",
            ],
            DocumentType.ACCOMMODATION: [
                "hilton",
                "marriott",
                "hyatt",
                "ibis",
                "mercure",
                "novotel",
                "crowne plaza",
                "holiday inn",
                "radisson",
                "sheraton",
                "hotel",
                "motel",
                "resort",
                "accommodation",
                "booking",
                "check-in",
                "check-out",
                "room",
                "suite",
                "nights",
            ],
            DocumentType.TRAVEL_DOCUMENT: [
                "qantas",
                "jetstar",
                "virgin australia",
                "tigerair",
                "rex airlines",
                "flight",
                "airline",
                "boarding pass",
                "ticket",
                "travel",
                "departure",
                "arrival",
                "gate",
                "seat",
                "passenger",
            ],
            DocumentType.PARKING_TOLL: [
                "secure parking",
                "wilson parking",
                "ace parking",
                "care park",
                "parking australia",
                "premium parking",
                "toll",
                "citylink",
                "eastlink",
                "westlink",
                "parking",
                "meter",
                "space",
                "duration",
            ],
            DocumentType.EQUIPMENT_SUPPLIES: [
                "computer",
                "laptop",
                "tablet",
                "printer",
                "software",
                "hardware",
                "equipment",
                "supplies",
                "stationery",
                "office supplies",
                "tools",
                "machinery",
                "furniture",
                "electronics",
            ],
            DocumentType.PROFESSIONAL_SERVICES: [
                "deloitte",
                "pwc",
                "kpmg",
                "ey",
                "bdo",
                "rsm",
                "pitcher partners",
                "allens",
                "ashurst",
                "clayton utz",
                "corrs",
                "herbert smith freehills",
                "legal",
                "accounting",
                "consulting",
                "advisory",
                "professional",
                "solicitor",
                "barrister",
                "accountant",
                "consultant",
            ],
        }

        # Document format indicators
        self.format_indicators = {
            DocumentType.TAX_INVOICE: [
                r"tax invoice",
                r"gst invoice",
                r"invoice number",
                r"abn",
                r"supplier",
                r"customer",
                r"due date",
                r"terms",
            ],
            DocumentType.BANK_STATEMENT: [
                r"account statement",
                r"transaction history",
                r"bsb",
                r"opening balance",
                r"closing balance",
                r"statement period",
            ],
            DocumentType.FUEL_RECEIPT: [
                r"pump \d+",
                r"litres?",
                r"fuel type",
                r"unleaded",
                r"diesel",
                r"premium",
                r"cents?/litre",
                r"total fuel",
            ],
            DocumentType.BUSINESS_RECEIPT: [
                r"receipt",
                r"purchase",
                r"total",
                r"gst",
                r"subtotal",
                r"items?",
                r"quantity",
                r"price",
            ],
            DocumentType.MEAL_RECEIPT: [
                r"table \d+",
                r"covers?",
                r"dine in",
                r"take away",
                r"breakfast",
                r"lunch",
                r"dinner",
                r"beverage",
            ],
            DocumentType.ACCOMMODATION: [
                r"check.?in",
                r"check.?out",
                r"room \d+",
                r"nights?",
                r"guest",
                r"booking",
                r"reservation",
            ],
            DocumentType.TRAVEL_DOCUMENT: [
                r"flight \w+",
                r"gate \w+",
                r"seat \w+",
                r"departure",
                r"arrival",
                r"passenger",
                r"boarding",
            ],
            DocumentType.PARKING_TOLL: [
                r"entry time",
                r"exit time",
                r"duration",
                r"space \d+",
                r"plate",
                r"registration",
                r"toll",
            ],
            DocumentType.EQUIPMENT_SUPPLIES: [
                r"model",
                r"serial",
                r"warranty",
                r"qty",
                r"unit price",
                r"description",
                r"part number",
            ],
            DocumentType.PROFESSIONAL_SERVICES: [
                r"hours?",
                r"rate",
                r"time",
                r"service period",
                r"matter",
                r"file",
                r"professional",
            ],
        }

        # Australian business names for detection
        self.australian_businesses = {
            "major_retailers": [
                "woolworths",
                "coles",
                "aldi",
                "target",
                "kmart",
                "bunnings",
                "officeworks",
                "harvey norman",
                "jb hi-fi",
                "big w",
            ],
            "fuel_stations": ["bp", "shell", "caltex", "ampol", "mobil", "7-eleven"],
            "banks": ["anz", "commonwealth bank", "westpac", "nab", "ing", "macquarie"],
            "airlines": ["qantas", "jetstar", "virgin australia", "tigerair"],
            "hotels": ["hilton", "marriott", "hyatt", "ibis", "mercure", "novotel"],
        }

        # ATO compliance indicators
        self.ato_compliance_indicators = {
            "abn_present": r"abn:?\s*\d{2}\s?\d{3}\s?\d{3}\s?\d{3}",
            "gst_breakdown": r"gst:?\s*\$?\d+\.\d{2}",
            "tax_invoice_header": r"tax invoice",
            "australian_date": r"\d{1,2}/\d{1,2}/\d{4}",
            "business_name": r"[A-Z][A-Z\s&]+(?:PTY\s+LTD|LIMITED|COMPANY)",
            "receipt_total": r"total:?\s*\$?\d+\.\d{2}",
            "bsb_code": r"bsb:?\s*\d{2,3}-\d{3}",
            "fuel_litres": r"\d+\.\d{3}l|\d+\.\d{2}l",
            "professional_hours": r"\d+\.?\d*\s*hours?",
            "accommodation_nights": r"\d+\s*nights?",
        }

        logger.info("Australian Tax Document Classifier initialized")

    def classify_document(self, text: str) -> ClassificationResult:
        """
        Classify Australian tax document based on content.

        Args:
            text: Document text content

        Returns:
            ClassificationResult with classification details
        """
        logger.debug(f"Classifying document with {len(text)} characters")

        text_lower = text.lower()

        # Score each document type
        type_scores = {}
        evidence_by_type = {}

        for doc_type in DocumentType:
            score, evidence = self._score_document_type(text_lower, doc_type)
            type_scores[doc_type] = score
            evidence_by_type[doc_type] = evidence

        # Find best match
        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]

        # Find second best for comparison
        remaining_types = [t for t in type_scores if t != best_type]
        second_best_type = (
            max(remaining_types, key=type_scores.get) if remaining_types else None
        )
        second_best_score = type_scores[second_best_type] if second_best_type else 0

        # Determine confidence
        confidence_score = best_score
        if best_score >= 0.8:
            confidence = ConfidenceLevel.HIGH
        elif best_score >= 0.6:
            confidence = ConfidenceLevel.MEDIUM
        elif best_score >= 0.4:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.LOW
            best_type = DocumentType.OTHER

        # Reduce confidence if there's strong competition
        if second_best_score > best_score * 0.8:
            if confidence == ConfidenceLevel.HIGH:
                confidence = ConfidenceLevel.MEDIUM
            elif confidence == ConfidenceLevel.MEDIUM:
                confidence = ConfidenceLevel.LOW

        # Generate reasoning
        reasoning = self._generate_reasoning(
            best_type, evidence_by_type[best_type], confidence_score
        )

        # Detect Australian business
        australian_business = self._detect_australian_business(text_lower)

        # Find ATO compliance indicators
        ato_indicators = self._find_ato_compliance_indicators(text)

        result = ClassificationResult(
            document_type=best_type,
            confidence=confidence,
            confidence_score=confidence_score,
            reasoning=reasoning,
            secondary_type=second_best_type if second_best_score > 0.3 else None,
            evidence=evidence_by_type[best_type],
            australian_business_detected=australian_business,
            ato_compliance_indicators=ato_indicators,
        )

        logger.info(
            f"Document classified as {best_type.value} with {confidence.value} confidence"
        )
        return result

    def _score_document_type(
        self, text: str, doc_type: DocumentType
    ) -> Tuple[float, Dict[str, Any]]:
        """Score how well text matches a document type."""

        evidence = {
            "keyword_matches": [],
            "format_matches": [],
            "business_matches": [],
            "total_score": 0.0,
        }

        total_score = 0.0
        max_possible_score = 0.0

        # Score keyword matches
        keywords = self.classification_keywords.get(doc_type, [])
        keyword_score = 0.0

        for keyword in keywords:
            if keyword in text:
                # Weight by keyword specificity
                if len(keyword) > 15:  # Very specific business names
                    weight = 1.0
                elif len(keyword) > 10:  # Specific business names
                    weight = 0.8
                elif len(keyword) > 6:  # Industry terms
                    weight = 0.6
                else:  # General terms
                    weight = 0.4

                keyword_score += weight
                evidence["keyword_matches"].append(
                    {"keyword": keyword, "weight": weight}
                )

        # Normalize keyword score
        if keywords:
            keyword_score = min(keyword_score / len(keywords), 1.0)
            total_score += keyword_score * 0.5  # 50% weight
            max_possible_score += 0.5

        # Score format indicators
        format_patterns = self.format_indicators.get(doc_type, [])
        format_score = 0.0

        for pattern in format_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                format_score += 0.2
                evidence["format_matches"].append(pattern)

        # Normalize format score
        if format_patterns:
            format_score = min(format_score, 1.0)
            total_score += format_score * 0.3  # 30% weight
            max_possible_score += 0.3

        # Score business name matches
        business_score = 0.0
        for category, businesses in self.australian_businesses.items():
            for business in businesses:
                if business in text:
                    business_score += 0.3
                    evidence["business_matches"].append(
                        {"business": business, "category": category}
                    )

        # Normalize business score
        business_score = min(business_score, 1.0)
        total_score += business_score * 0.2  # 20% weight
        max_possible_score += 0.2

        # Calculate final score
        if max_possible_score > 0:
            final_score = total_score / max_possible_score
        else:
            final_score = 0.0

        evidence["total_score"] = final_score

        return final_score, evidence

    def _generate_reasoning(
        self, _doc_type: DocumentType, evidence: Dict[str, Any], score: float
    ) -> str:
        """Generate reasoning for classification decision."""

        reasons = []

        # Keyword evidence
        if evidence["keyword_matches"]:
            top_keywords = sorted(
                evidence["keyword_matches"], key=lambda x: x["weight"], reverse=True
            )[:3]
            keyword_list = [kw["keyword"] for kw in top_keywords]
            reasons.append(f"Strong keyword matches: {', '.join(keyword_list)}")

        # Format evidence
        if evidence["format_matches"]:
            reasons.append(
                f"Document format indicators: {len(evidence['format_matches'])} matches"
            )

        # Business evidence
        if evidence["business_matches"]:
            business_names = [bm["business"] for bm in evidence["business_matches"]]
            reasons.append(
                f"Australian business detected: {', '.join(business_names[:2])}"
            )

        # Score-based reasoning
        if score >= 0.8:
            reasons.append("High confidence classification")
        elif score >= 0.6:
            reasons.append("Medium confidence classification")
        else:
            reasons.append("Low confidence classification")

        return "; ".join(reasons)

    def _detect_australian_business(self, text: str) -> bool:
        """Detect if text contains Australian business names."""

        for _category, businesses in self.australian_businesses.items():
            for business in businesses:
                if business in text:
                    return True

        return False

    def _find_ato_compliance_indicators(self, text: str) -> List[str]:
        """Find ATO compliance indicators in text."""

        indicators = []

        for indicator, pattern in self.ato_compliance_indicators.items():
            if re.search(pattern, text, re.IGNORECASE):
                indicators.append(indicator)

        return indicators

    def get_document_type_requirements(self, doc_type: DocumentType) -> Dict[str, Any]:
        """Get ATO requirements for specific document type."""

        requirements = {
            DocumentType.BUSINESS_RECEIPT: {
                "required_fields": ["date", "business_name", "total_amount"],
                "high_value_fields": ["abn", "gst_amount"],
                "threshold": 82.50,
                "description": "General business receipt for goods/services",
            },
            DocumentType.FUEL_RECEIPT: {
                "required_fields": [
                    "date",
                    "station_name",
                    "fuel_type",
                    "litres",
                    "total_amount",
                ],
                "high_value_fields": ["abn", "gst_amount"],
                "threshold": 82.50,
                "description": "Fuel receipt for vehicle expense claims",
            },
            DocumentType.TAX_INVOICE: {
                "required_fields": [
                    "date",
                    "supplier_name",
                    "supplier_abn",
                    "gst_amount",
                    "total_amount",
                ],
                "high_value_fields": ["invoice_number", "customer_abn"],
                "threshold": 82.50,
                "description": "Formal tax invoice with GST breakdown",
            },
            DocumentType.BANK_STATEMENT: {
                "required_fields": [
                    "date",
                    "bank_name",
                    "account_holder",
                    "transaction_details",
                ],
                "high_value_fields": ["bsb", "account_number"],
                "threshold": 0.0,
                "description": "Bank statement for expense verification",
            },
            DocumentType.MEAL_RECEIPT: {
                "required_fields": ["date", "restaurant_name", "total_amount"],
                "high_value_fields": ["abn", "gst_amount", "business_purpose"],
                "threshold": 82.50,
                "description": "Meal receipt for entertainment expenses",
            },
            DocumentType.ACCOMMODATION: {
                "required_fields": ["date", "hotel_name", "total_amount", "nights"],
                "high_value_fields": ["abn", "gst_amount", "business_purpose"],
                "threshold": 82.50,
                "description": "Accommodation receipt for travel expenses",
            },
            DocumentType.TRAVEL_DOCUMENT: {
                "required_fields": [
                    "date",
                    "airline_name",
                    "total_amount",
                    "destination",
                ],
                "high_value_fields": ["abn", "gst_amount", "business_purpose"],
                "threshold": 82.50,
                "description": "Travel document for travel expenses",
            },
            DocumentType.PARKING_TOLL: {
                "required_fields": ["date", "operator_name", "total_amount"],
                "high_value_fields": ["location", "duration", "gst_amount"],
                "threshold": 82.50,
                "description": "Parking/toll receipt for vehicle expenses",
            },
            DocumentType.EQUIPMENT_SUPPLIES: {
                "required_fields": [
                    "date",
                    "supplier_name",
                    "total_amount",
                    "description",
                ],
                "high_value_fields": ["abn", "gst_amount", "warranty"],
                "threshold": 82.50,
                "description": "Equipment/supplies receipt for business expenses",
            },
            DocumentType.PROFESSIONAL_SERVICES: {
                "required_fields": [
                    "date",
                    "firm_name",
                    "firm_abn",
                    "total_amount",
                    "description",
                ],
                "high_value_fields": ["hours", "rate", "service_period"],
                "threshold": 82.50,
                "description": "Professional services invoice for business expenses",
            },
            DocumentType.OTHER: {
                "required_fields": ["date", "supplier_name", "total_amount"],
                "high_value_fields": ["abn", "gst_amount"],
                "threshold": 82.50,
                "description": "Other business-related document",
            },
        }

        return requirements.get(doc_type, requirements[DocumentType.OTHER])

    def get_classification_keywords(self, doc_type: DocumentType) -> List[str]:
        """Get classification keywords for document type."""
        return self.classification_keywords.get(doc_type, [])


# Global instance for easy access
australian_tax_classifier = AustralianTaxDocumentClassifier()


def classify_australian_tax_document(text: str) -> ClassificationResult:
    """
    Classify Australian tax document.

    Args:
        text: Document text content

    Returns:
        ClassificationResult with classification details
    """
    return australian_tax_classifier.classify_document(text)


def get_document_type_requirements(doc_type: DocumentType) -> Dict[str, Any]:
    """
    Get ATO requirements for specific document type.

    Args:
        doc_type: Document type

    Returns:
        Dictionary with ATO requirements
    """
    return australian_tax_classifier.get_document_type_requirements(doc_type)


def get_classification_keywords(doc_type: DocumentType) -> List[str]:
    """
    Get classification keywords for document type.

    Args:
        doc_type: Document type

    Returns:
        List of classification keywords
    """
    return australian_tax_classifier.get_classification_keywords(doc_type)
