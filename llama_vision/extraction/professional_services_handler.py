"""Professional services document type handler."""

import re
from typing import Any, Dict, List, Pattern

from .document_handlers import DocumentPattern, DocumentTypeHandler, ExtractionResult


class ProfessionalServicesHandler(DocumentTypeHandler):
    """Handler for professional services documents (legal, accounting, consulting)."""

    @property
    def document_type(self) -> str:
        return "professional_services"

    @property
    def display_name(self) -> str:
        return "Professional Services"

    def get_classification_indicators(self) -> List[str]:
        """Get text indicators for professional services."""
        return [
            "legal",
            "lawyer",
            "solicitor",
            "barrister",
            "attorney",
            "law firm",
            "legal services",
            "accounting",
            "accountant",
            "bookkeeper",
            "audit",
            "tax preparation",
            "tax return",
            "consulting",
            "consultant",
            "advisory",
            "professional services",
            "chartered accountant",
            "cpa",
            "tax agent",
            "business advisor",
            "financial advisor",
            "valuation",
            "appraisal",
            "notary",
            "conveyancing",
            "contract",
            "legal advice",
            "professional fee",
            "hourly rate",
            "retainer",
            "disbursement",
            "professional consultation",
            "expert witness",
            "litigation",
            "settlement",
            "mediation",
            "arbitration",
            "compliance",
            "due diligence",
            "audit report",
            "financial statement",
            "tax compliance",
            "legal document",
            "legal representation",
            "court",
            "tribunal",
            "regulatory",
            "corporate",
            "partnership",
            "trust",
            "estate",
            "probate",
            "will",
            "power of attorney",
            "intellectual property",
            "trademark",
            "patent",
            "copyright",
            "business registration",
            "asic",
            "ato",
            "australian taxation office",
            "australian securities",
        ]

    def get_classification_patterns(self) -> List[Pattern]:
        """Get regex patterns for professional services classification."""
        return [
            re.compile(r"legal|lawyer|solicitor|barrister|attorney", re.IGNORECASE),
            re.compile(r"accounting|accountant|bookkeeper|audit", re.IGNORECASE),
            re.compile(r"consulting|consultant|advisory", re.IGNORECASE),
            re.compile(r"professional\s+services", re.IGNORECASE),
            re.compile(r"chartered\s+accountant|cpa|tax\s+agent", re.IGNORECASE),
            re.compile(r"law\s+firm|legal\s+services", re.IGNORECASE),
            re.compile(r"hourly\s+rate|retainer|disbursement", re.IGNORECASE),
            re.compile(r"professional\s+fee|consultation\s+fee", re.IGNORECASE),
            re.compile(r"expert\s+witness|litigation|settlement", re.IGNORECASE),
            re.compile(r"mediation|arbitration|compliance", re.IGNORECASE),
            re.compile(r"due\s+diligence|audit\s+report", re.IGNORECASE),
            re.compile(r"financial\s+statement|tax\s+compliance", re.IGNORECASE),
            re.compile(r"legal\s+document|legal\s+representation", re.IGNORECASE),
            re.compile(r"court|tribunal|regulatory", re.IGNORECASE),
            re.compile(r"corporate|partnership|trust|estate", re.IGNORECASE),
            re.compile(r"probate|will|power\s+of\s+attorney", re.IGNORECASE),
            re.compile(r"intellectual\s+property|trademark|patent", re.IGNORECASE),
            re.compile(r"business\s+registration|asic|ato", re.IGNORECASE),
            re.compile(r"australian\s+taxation\s+office", re.IGNORECASE),
        ]

    def get_prompt_name(self) -> str:
        """Get prompt name for professional services."""
        return "professional_services_extraction_prompt"

    def get_field_patterns(self) -> List[DocumentPattern]:
        """Get field extraction patterns for professional services."""
        return [
            DocumentPattern(
                pattern=r"DATE:\s*([^\n\r]+)",
                field_name="DATE",
                field_type="date",
                required=True,
            ),
            DocumentPattern(
                pattern=r"FIRM:\s*([^\n\r]+)",
                field_name="FIRM",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"PROFESSIONAL:\s*([^\n\r]+)",
                field_name="PROFESSIONAL",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"ABN:\s*([^\n\r]+)",
                field_name="ABN",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"ADDRESS:\s*([^\n\r]+)",
                field_name="ADDRESS",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"TOTAL:\s*([^\n\r]+)",
                field_name="TOTAL",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"GST:\s*([^\n\r]+)",
                field_name="GST",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"SERVICES:\s*([^\n\r]+)",
                field_name="SERVICES",
                field_type="list",
                required=True,
            ),
            DocumentPattern(
                pattern=r"HOURS:\s*([^\n\r]+)",
                field_name="HOURS",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"RATE:\s*([^\n\r]+)",
                field_name="RATE",
                field_type="number",
                required=False,
            ),
            DocumentPattern(
                pattern=r"PAYMENT_METHOD:\s*([^\n\r]+)",
                field_name="PAYMENT_METHOD",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"INVOICE_NUMBER:\s*([^\n\r]+)",
                field_name="INVOICE_NUMBER",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"MATTER:\s*([^\n\r]+)",
                field_name="MATTER",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"CLIENT:\s*([^\n\r]+)",
                field_name="CLIENT",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"REFERENCE:\s*([^\n\r]+)",
                field_name="REFERENCE",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"DISBURSEMENTS:\s*([^\n\r]+)",
                field_name="DISBURSEMENTS",
                field_type="number",
                required=False,
            ),
            DocumentPattern(
                pattern=r"PROFESSIONAL_FEES:\s*([^\n\r]+)",
                field_name="PROFESSIONAL_FEES",
                field_type="number",
                required=False,
            ),
            DocumentPattern(
                pattern=r"CONSULTATION_TYPE:\s*([^\n\r]+)",
                field_name="CONSULTATION_TYPE",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"BILLING_PERIOD:\s*([^\n\r]+)",
                field_name="BILLING_PERIOD",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"RETAINER:\s*([^\n\r]+)",
                field_name="RETAINER",
                field_type="number",
                required=False,
            ),
        ]

    def get_field_mappings(self) -> Dict[str, List[str]]:
        """Get field mappings for standardization."""
        return {
            # Standard compliance fields
            "supplier_name": ["FIRM"],
            "total_amount": ["TOTAL"],
            "gst_amount": ["GST"],
            "invoice_date": ["DATE"],
            "supplier_abn": ["ABN"],
            "items": ["SERVICES"],
            "payment_method": ["PAYMENT_METHOD"],
            "invoice_number": ["INVOICE_NUMBER"],
            # Professional services specific fields
            "firm_name": ["FIRM"],
            "professional_name": ["PROFESSIONAL"],
            "service_date": ["DATE"],
            "services_provided": ["SERVICES"],
            "hourly_rate": ["RATE"],
            "hours_worked": ["HOURS"],
            "matter_reference": ["MATTER"],
            "client_name": ["CLIENT"],
            "reference_number": ["REFERENCE"],
            "disbursements": ["DISBURSEMENTS"],
            "professional_fees": ["PROFESSIONAL_FEES"],
            "consultation_type": ["CONSULTATION_TYPE"],
            "billing_period": ["BILLING_PERIOD"],
            "retainer_amount": ["RETAINER"],
            "firm_address": ["ADDRESS"],
            "service_type": ["CONSULTATION_TYPE"],
        }

    def extract_fields(self, response: str) -> ExtractionResult:
        """Extract fields with fallback pattern matching.

        Args:
            response: Model response text

        Returns:
            Extraction result with fields and metadata
        """
        # First try the standard KEY-VALUE approach
        result = super().extract_fields(response)

        # If we found less than 4 meaningful fields, use fallback pattern matching
        # We expect 5+ fields for professional services, so <4 indicates KEY-VALUE parsing failed
        if result.field_count < 4:
            self.logger.debug(
                f"KEY-VALUE parsing found only {result.field_count} fields, trying fallback pattern matching..."
            )
            fallback_fields = self._extract_from_raw_text(response)

            # Merge fallback fields with any successful KEY-VALUE fields
            combined_fields = result.fields.copy()
            combined_fields.update(fallback_fields)

            # Apply field mappings
            mappings = self.get_field_mappings()
            normalized = self._apply_field_mappings(combined_fields, mappings)

            # Recalculate compliance score
            required_patterns = [p for p in self.get_field_patterns() if p.required]
            required_found = sum(
                1 for p in required_patterns if p.field_name in normalized
            )
            compliance_score = (
                required_found / len(required_patterns) if required_patterns else 1.0
            )

            return ExtractionResult(
                fields=normalized,
                extraction_method=f"{self.document_type}_handler_with_fallback",
                compliance_score=compliance_score,
                field_count=len(normalized),
            )
        else:
            self.logger.debug(
                f"KEY-VALUE parsing successful with {result.field_count} fields, skipping fallback"
            )
            return result

    def _extract_from_raw_text(self, response: str) -> Dict[str, Any]:
        """Extract fields from raw OCR text using AWK-style processing.

        Args:
            response: Raw model response text

        Returns:
            Extracted fields dictionary
        """
        # Use AWK-style extractor for cleaner, more maintainable extraction
        from .awk_extractor import ProfessionalServicesAwkExtractor

        awk_extractor = ProfessionalServicesAwkExtractor(self.log_level)
        extracted = awk_extractor.extract_professional_services_fields(response)

        self.logger.debug(
            f"AWK fallback extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted
