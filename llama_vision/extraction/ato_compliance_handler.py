"""
ATO Compliance Handler for Australian Tax Office Work-Related Expense Claims

This module provides comprehensive Australian tax compliance validation and assessment
functionality ported from the InternVL system to ensure the Llama-3.2 system has
identical domain expertise for fair comparison.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..utils import setup_logging

logger = setup_logging()


@dataclass
class ATOComplianceResult:
    """Result of ATO compliance assessment."""

    compliance_score: float
    ato_ready: bool
    required_fields_present: List[str]
    missing_fields: List[str]
    invalid_fields: List[str]
    field_assessment: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    document_type: str
    claim_category: str


class ATOComplianceHandler:
    """Australian Tax Office compliance handler for work-related expense claims."""

    def __init__(self):
        """Initialize ATO compliance handler with Australian tax rules."""

        # ATO compliance thresholds
        self.ato_thresholds = {
            "receipt_required_amount": 82.50,  # ATO requires receipt for claims over $82.50
            "maximum_without_receipt": 300.00,  # Maximum claim without receipt
            "minimum_compliance_score": 80.0,  # Minimum score for ATO readiness
            "gst_rate": 0.10,  # Australian GST rate (10%)
            "gst_calculation_tolerance": 0.02,  # 2c tolerance for GST calculation
        }

        # Required fields by document type for ATO compliance
        self.required_fields_by_type = {
            "business_receipt": ["date", "supplier_name", "total_amount"],
            "fuel_receipt": [
                "date",
                "station_name",
                "total_amount",
                "fuel_type",
                "litres",
            ],
            "tax_invoice": [
                "date",
                "supplier_name",
                "supplier_abn",
                "gst_amount",
                "total_amount",
            ],
            "meal_receipt": ["date", "restaurant_name", "total_amount", "meal_type"],
            "accommodation": [
                "date",
                "supplier_name",
                "total_amount",
                "accommodation_type",
            ],
            "travel_document": ["date", "supplier_name", "total_amount", "travel_type"],
            "parking_toll": ["date", "supplier_name", "total_amount", "location"],
            "equipment_supplies": [
                "date",
                "supplier_name",
                "total_amount",
                "description",
            ],
            "professional_services": [
                "date",
                "supplier_name",
                "supplier_abn",
                "total_amount",
                "description",
            ],
            "bank_statement": [
                "date",
                "bank_name",
                "account_holder",
                "transaction_amount",
            ],
            "other": ["date", "supplier_name", "total_amount"],
        }

        # Additional required fields for high-value claims
        self.high_value_required_fields = {
            "supplier_abn": 82.50,  # ABN required for claims over $82.50
            "gst_amount": 82.50,  # GST breakdown required for claims over $82.50
            "description": 300.00,  # Description required for claims over $300
        }

        # Australian business names for validation
        self.australian_business_names = {
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
            "bp",
            "shell",
            "caltex",
            "ampol",
            "mobil",
            "7-eleven",
            "united",
            "anz",
            "commonwealth bank",
            "westpac",
            "nab",
            "ing",
            "macquarie",
            "qantas",
            "jetstar",
            "virgin australia",
            "tigerair",
            "avis",
            "hertz",
            "mcdonald's",
            "kfc",
            "subway",
            "domino's",
            "pizza hut",
            "hungry jack's",
            "hilton",
            "marriott",
            "hyatt",
            "ibis",
            "mercure",
            "novotel",
        }

        # Work-related expense categories
        self.work_expense_categories = {
            "vehicle_expenses": [
                "fuel",
                "parking",
                "tolls",
                "car_repairs",
                "registration",
            ],
            "travel_expenses": [
                "accommodation",
                "flights",
                "meals_travel",
                "transport",
            ],
            "office_expenses": [
                "office_supplies",
                "equipment",
                "software",
                "stationery",
            ],
            "professional_development": ["training", "conferences", "courses", "books"],
            "professional_services": ["accounting", "legal", "consulting", "advisory"],
            "communication": ["phone", "internet", "mobile", "data"],
            "entertainment": ["client_meals", "business_entertainment", "functions"],
            "other": ["insurance", "repairs", "maintenance", "subscriptions"],
        }

        logger.info("ATO Compliance Handler initialized with Australian tax rules")

    def assess_ato_compliance(
        self,
        extracted_fields: Dict[str, Any],
        document_type: str = "other",
        claim_category: str = "General",
    ) -> ATOComplianceResult:
        """
        Assess extracted fields for ATO compliance.

        Args:
            extracted_fields: Dictionary of extracted field values
            document_type: Type of document being assessed
            claim_category: Category of work-related expense

        Returns:
            ATOComplianceResult with comprehensive assessment
        """
        logger.debug(f"Assessing ATO compliance for {document_type} document")

        # Get required fields for this document type
        required_fields = self.required_fields_by_type.get(
            document_type, self.required_fields_by_type["other"]
        )

        # Assess each field
        field_assessment = {}
        present_fields = []
        missing_fields = []
        invalid_fields = []

        # Check basic required fields
        for field in required_fields:
            field_value = extracted_fields.get(field, "")
            assessment = self._assess_field_validity(field, field_value, document_type)
            field_assessment[field] = assessment

            if assessment["present"]:
                present_fields.append(field)
                if not assessment["valid"]:
                    invalid_fields.append(field)
            else:
                missing_fields.append(field)

        # Check high-value requirements
        total_amount = self._extract_numeric_value(
            extracted_fields.get("total_amount", "0")
        )
        for field, threshold in self.high_value_required_fields.items():
            if total_amount > threshold:
                field_value = extracted_fields.get(field, "")
                assessment = self._assess_field_validity(
                    field, field_value, document_type
                )
                field_assessment[field] = assessment

                if field not in required_fields:  # Don't double-count
                    if assessment["present"]:
                        present_fields.append(field)
                        if not assessment["valid"]:
                            invalid_fields.append(field)
                    else:
                        missing_fields.append(field)

        # Calculate compliance score
        total_fields = len(field_assessment)
        valid_fields = sum(
            1
            for assessment in field_assessment.values()
            if assessment["present"] and assessment["valid"]
        )

        if total_fields > 0:
            compliance_score = (valid_fields / total_fields) * 100
        else:
            compliance_score = 0.0

        # Determine ATO readiness
        ato_ready = self._determine_ato_readiness(
            compliance_score, total_amount, field_assessment
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            field_assessment,
            total_amount,
            document_type,
            missing_fields,
            invalid_fields,
        )

        result = ATOComplianceResult(
            compliance_score=compliance_score,
            ato_ready=ato_ready,
            required_fields_present=present_fields,
            missing_fields=missing_fields,
            invalid_fields=invalid_fields,
            field_assessment=field_assessment,
            recommendations=recommendations,
            document_type=document_type,
            claim_category=claim_category,
        )

        logger.info(
            f"ATO compliance assessment complete: {compliance_score:.1f}% compliance, "
            f"ATO ready: {ato_ready}"
        )

        return result

    def _assess_field_validity(
        self, field: str, value: str, _document_type: str
    ) -> Dict[str, Any]:
        """Assess the validity of a specific field value."""

        is_present = bool(value and str(value).strip())
        is_valid = False
        validation_message = ""

        if not is_present:
            validation_message = f"Field '{field}' is missing"
            return {
                "present": False,
                "valid": False,
                "value": value,
                "message": validation_message,
            }

        value_str = str(value).strip()

        # Field-specific validation
        if field == "date":
            is_valid = self._is_valid_australian_date(value_str)
            validation_message = (
                "Valid Australian date format"
                if is_valid
                else "Invalid date format (use DD/MM/YYYY)"
            )

        elif field == "supplier_abn":
            is_valid = self._is_valid_abn(value_str)
            validation_message = (
                "Valid ABN format"
                if is_valid
                else "Invalid ABN format (use XX XXX XXX XXX)"
            )

        elif field in ["total_amount", "gst_amount", "subtotal"]:
            is_valid = self._is_valid_currency_amount(value_str)
            validation_message = (
                "Valid currency amount" if is_valid else "Invalid currency format"
            )

        elif field in ["supplier_name", "station_name", "restaurant_name", "bank_name"]:
            is_valid = self._is_valid_business_name(value_str)
            validation_message = (
                "Valid business name"
                if is_valid
                else "Business name should be in uppercase"
            )

        elif field == "fuel_type":
            is_valid = self._is_valid_fuel_type(value_str)
            validation_message = "Valid fuel type" if is_valid else "Invalid fuel type"

        elif field == "litres":
            is_valid = self._is_valid_quantity(value_str)
            validation_message = (
                "Valid quantity" if is_valid else "Invalid quantity format"
            )

        else:
            # Generic validation for other fields
            is_valid = len(value_str) > 0
            validation_message = "Field present" if is_valid else "Field is empty"

        return {
            "present": is_present,
            "valid": is_valid,
            "value": value,
            "message": validation_message,
        }

    def _determine_ato_readiness(
        self,
        compliance_score: float,
        total_amount: float,
        field_assessment: Dict[str, Dict[str, Any]],
    ) -> bool:
        """Determine if the document is ready for ATO submission."""

        # Basic compliance score check
        if compliance_score < self.ato_thresholds["minimum_compliance_score"]:
            return False

        # Check mandatory fields based on amount
        if total_amount > self.ato_thresholds["receipt_required_amount"]:
            # High-value claims require ABN and GST
            abn_valid = field_assessment.get("supplier_abn", {}).get("valid", False)
            gst_valid = field_assessment.get("gst_amount", {}).get("valid", False)

            if not (abn_valid and gst_valid):
                return False

        # Check that essential fields are present and valid
        essential_fields = ["date", "supplier_name", "total_amount"]
        for field in essential_fields:
            if field in field_assessment:
                if not (
                    field_assessment[field]["present"]
                    and field_assessment[field]["valid"]
                ):
                    return False

        return True

    def _generate_recommendations(
        self,
        field_assessment: Dict[str, Dict[str, Any]],
        total_amount: float,
        document_type: str,
        missing_fields: List[str],
        invalid_fields: List[str],
    ) -> List[str]:
        """Generate recommendations for improving ATO compliance."""

        recommendations = []

        # Missing field recommendations
        if missing_fields:
            recommendations.append(
                f"Missing required fields: {', '.join(missing_fields)}"
            )

        # Invalid field recommendations
        for field in invalid_fields:
            message = field_assessment.get(field, {}).get("message", "")
            if message:
                recommendations.append(f"{field}: {message}")

        # Amount-specific recommendations
        if total_amount > self.ato_thresholds["receipt_required_amount"]:
            if "supplier_abn" in missing_fields:
                recommendations.append("ABN required for claims over $82.50")
            if "gst_amount" in missing_fields:
                recommendations.append("GST breakdown required for claims over $82.50")

        # Document type specific recommendations
        if document_type == "fuel_receipt":
            if "litres" in missing_fields:
                recommendations.append(
                    "Fuel quantity in litres required for vehicle expense claims"
                )
            if "fuel_type" in missing_fields:
                recommendations.append(
                    "Fuel type required for accurate vehicle expense claims"
                )

        elif document_type == "meal_receipt":
            if "meal_type" in missing_fields:
                recommendations.append(
                    "Meal type required for entertainment expense claims"
                )
            if total_amount > 300:
                recommendations.append(
                    "Business purpose documentation required for meal claims over $300"
                )

        # General recommendations
        if not recommendations:
            recommendations.append("Document meets basic ATO compliance requirements")

        return recommendations

    def _is_valid_australian_date(self, date_str: str) -> bool:
        """Check if date string matches Australian DD/MM/YYYY format."""
        if not date_str:
            return False

        # Australian date patterns
        patterns = [
            r"^\d{1,2}/\d{1,2}/\d{4}$",  # DD/MM/YYYY or D/M/YYYY
            r"^\d{1,2}-\d{1,2}-\d{4}$",  # DD-MM-YYYY
            r"^\d{1,2}\.\d{1,2}\.\d{4}$",  # DD.MM.YYYY
        ]

        for pattern in patterns:
            if re.match(pattern, date_str.strip()):
                return True

        return False

    def _is_valid_abn(self, abn_str: str) -> bool:
        """Check if ABN string matches Australian Business Number format."""
        if not abn_str:
            return False

        # Remove all non-digits
        clean_abn = re.sub(r"[^\d]", "", abn_str.strip())

        # ABN must be exactly 11 digits
        if len(clean_abn) != 11:
            return False

        # Check for valid format patterns
        patterns = [
            r"^\d{2}\s\d{3}\s\d{3}\s\d{3}$",  # XX XXX XXX XXX
            r"^\d{11}$",  # XXXXXXXXXXX
            r"^\d{2}\s\d{9}$",  # XX XXXXXXXXX
            r"^\d{2}-\d{3}-\d{3}-\d{3}$",  # XX-XXX-XXX-XXX
        ]

        for pattern in patterns:
            if re.match(pattern, abn_str.strip()):
                return True

        return len(clean_abn) == 11

    def _is_valid_currency_amount(self, amount_str: str) -> bool:
        """Check if currency amount is valid for Australian context."""
        if not amount_str:
            return False

        try:
            # Remove currency symbols and clean
            clean_amount = re.sub(r"[$AUD\s,]", "", amount_str.strip())

            # Should be a valid decimal
            if re.match(r"^\d+(\.\d{1,2})?$", clean_amount):
                amount = float(clean_amount)
                # Reasonable amount range (0.01 to 100000 AUD)
                return 0.01 <= amount <= 100000.0

            return False
        except (ValueError, TypeError):
            return False

    def _is_valid_business_name(self, name_str: str) -> bool:
        """Check if business name is valid (should be uppercase for receipts)."""
        if not name_str:
            return False

        # Check if it's a known Australian business
        name_lower = name_str.lower()
        if any(business in name_lower for business in self.australian_business_names):
            return True

        # Should have at least some uppercase letters for receipt format
        return any(c.isupper() for c in name_str)

    def _is_valid_fuel_type(self, fuel_str: str) -> bool:
        """Check if fuel type is valid."""
        if not fuel_str:
            return False

        valid_fuel_types = [
            "unleaded",
            "premium",
            "diesel",
            "ulp",
            "e10",
            "e85",
            "u91",
            "u95",
            "u98",
            "premium unleaded",
            "super premium",
        ]

        return fuel_str.lower() in valid_fuel_types

    def _is_valid_quantity(self, qty_str: str) -> bool:
        """Check if quantity string is valid."""
        if not qty_str:
            return False

        try:
            # Allow integers, floats, and units
            patterns = [
                r"^\d+$",  # Simple integer
                r"^\d+\.\d+$",  # Simple decimal
                r"^\d+(\.\d+)?\s*[a-zA-Z]+$",  # With units
            ]

            for pattern in patterns:
                if re.match(pattern, qty_str.strip()):
                    return True

            return False
        except Exception:
            return False

    def _extract_numeric_value(self, value_str: str) -> float:
        """Extract numeric value from string."""
        if not value_str:
            return 0.0

        try:
            # Remove currency symbols and clean
            clean_value = re.sub(r"[$AUD\s,]", "", str(value_str).strip())
            return float(clean_value)
        except (ValueError, TypeError):
            return 0.0

    def validate_gst_calculation(
        self, subtotal: str, gst: str, total: str
    ) -> Tuple[bool, str]:
        """Validate GST calculation (10% in Australia)."""

        try:
            subtotal_val = self._extract_numeric_value(subtotal)
            gst_val = self._extract_numeric_value(gst)
            total_val = self._extract_numeric_value(total)

            if subtotal_val <= 0 or gst_val <= 0 or total_val <= 0:
                return False, "Invalid amounts for GST calculation"

            # Check if GST is approximately 10% of subtotal
            expected_gst = subtotal_val * self.ato_thresholds["gst_rate"]
            gst_difference = abs(gst_val - expected_gst)

            if gst_difference > self.ato_thresholds["gst_calculation_tolerance"]:
                return (
                    False,
                    f"GST calculation error: {gst_val} should be {expected_gst:.2f}",
                )

            # Check if total = subtotal + GST
            expected_total = subtotal_val + gst_val
            total_difference = abs(total_val - expected_total)

            if total_difference > self.ato_thresholds["gst_calculation_tolerance"]:
                return (
                    False,
                    f"Total calculation error: {total_val} should be {expected_total:.2f}",
                )

            return True, "GST calculation is correct"

        except Exception as e:
            return False, f"GST validation error: {e}"

    def categorize_work_expense(
        self, extracted_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Categorize work-related expense for ATO purposes."""

        supplier_name = extracted_fields.get("supplier_name", "").lower()
        description = extracted_fields.get("description", "").lower()
        document_type = extracted_fields.get("document_type", "other")

        # Determine category based on supplier and description
        category = "other"
        subcategory = "general"
        deductibility = "medium"

        # Vehicle expenses
        if any(
            fuel_station in supplier_name
            for fuel_station in ["bp", "shell", "caltex", "ampol"]
        ):
            category = "vehicle_expenses"
            subcategory = "fuel"
            deductibility = "high"
        elif "parking" in description or "toll" in description:
            category = "vehicle_expenses"
            subcategory = "parking_tolls"
            deductibility = "high"

        # Office expenses
        elif "officeworks" in supplier_name or "staples" in supplier_name:
            category = "office_expenses"
            subcategory = "supplies"
            deductibility = "high"
        elif any(
            tech in supplier_name for tech in ["jb hi-fi", "harvey norman", "apple"]
        ):
            category = "office_expenses"
            subcategory = "equipment"
            deductibility = "medium"

        # Travel expenses
        elif any(
            airline in supplier_name for airline in ["qantas", "jetstar", "virgin"]
        ):
            category = "travel_expenses"
            subcategory = "flights"
            deductibility = "high"
        elif any(hotel in supplier_name for hotel in ["hilton", "marriott", "hyatt"]):
            category = "travel_expenses"
            subcategory = "accommodation"
            deductibility = "high"

        # Professional services
        elif any(
            service in description for service in ["accounting", "legal", "consulting"]
        ):
            category = "professional_services"
            subcategory = "advisory"
            deductibility = "high"

        # Entertainment (limited deductibility)
        elif document_type == "meal_receipt":
            category = "entertainment"
            subcategory = "meals"
            deductibility = "low"  # Often limited deductibility

        return {
            "category": category,
            "subcategory": subcategory,
            "deductibility": deductibility,
            "ato_category": self._map_to_ato_category(category),
            "documentation_required": self._get_documentation_requirements(
                category, subcategory
            ),
        }

    def _map_to_ato_category(self, category: str) -> str:
        """Map internal category to ATO expense category."""

        ato_mapping = {
            "vehicle_expenses": "Car expenses",
            "travel_expenses": "Travel expenses",
            "office_expenses": "Office expenses",
            "professional_development": "Self-education expenses",
            "professional_services": "Professional services",
            "communication": "Phone and internet expenses",
            "entertainment": "Entertainment expenses",
            "other": "Other work-related expenses",
        }

        return ato_mapping.get(category, "Other work-related expenses")

    def _get_documentation_requirements(
        self, category: str, _subcategory: str
    ) -> List[str]:
        """Get documentation requirements for ATO category."""

        requirements = ["Receipt or invoice", "Date and amount", "Business purpose"]

        if category == "vehicle_expenses":
            requirements.extend(["Logbook for business use", "Odometer readings"])
        elif category == "travel_expenses":
            requirements.extend(["Travel purpose", "Destination and dates"])
        elif category == "entertainment":
            requirements.extend(["Business purpose", "Attendees", "Venue"])
        elif category == "professional_services":
            requirements.extend(["Service description", "Business benefit"])

        return requirements


def assess_ato_compliance_enhanced(
    extracted_fields: Dict[str, Any],
    document_type: str = "other",
    claim_category: str = "General",
) -> Dict[str, Any]:
    """
    Enhanced ATO compliance assessment function.

    Args:
        extracted_fields: Dictionary of extracted field values
        document_type: Type of document being assessed
        claim_category: Category of work-related expense

    Returns:
        Dictionary with comprehensive ATO compliance assessment
    """
    handler = ATOComplianceHandler()

    try:
        # Assess ATO compliance
        compliance_result = handler.assess_ato_compliance(
            extracted_fields, document_type, claim_category
        )

        # Categorize work expense
        expense_category = handler.categorize_work_expense(extracted_fields)

        # Validate GST if applicable
        gst_valid = True
        gst_message = "GST validation not applicable"

        if (
            "subtotal" in extracted_fields
            and "gst_amount" in extracted_fields
            and "total_amount" in extracted_fields
        ):
            gst_valid, gst_message = handler.validate_gst_calculation(
                extracted_fields["subtotal"],
                extracted_fields["gst_amount"],
                extracted_fields["total_amount"],
            )

        return {
            "success": True,
            "compliance_result": compliance_result,
            "expense_category": expense_category,
            "gst_validation": {"valid": gst_valid, "message": gst_message},
            "ato_ready": compliance_result.ato_ready,
            "compliance_score": compliance_result.compliance_score,
            "recommendations": compliance_result.recommendations,
            "handler_version": "ato_compliance_v1.0_australian_focus",
        }

    except Exception as e:
        logger.error(f"ATO compliance assessment failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "compliance_result": None,
            "expense_category": None,
            "gst_validation": {"valid": False, "message": f"Error: {e}"},
            "ato_ready": False,
            "compliance_score": 0.0,
            "recommendations": [
                "Error in compliance assessment - manual review required"
            ],
            "handler_version": "ato_compliance_v1.0_australian_focus",
        }
