"""Other document type handler for miscellaneous business documents."""

import re
from typing import Any, Dict, List, Pattern

from .document_handlers import DocumentPattern, DocumentTypeHandler, ExtractionResult


class OtherDocumentHandler(DocumentTypeHandler):
    """Handler for other miscellaneous business documents."""

    @property
    def document_type(self) -> str:
        return "other"

    @property
    def display_name(self) -> str:
        return "Other Document"

    def get_classification_indicators(self) -> List[str]:
        """Get text indicators for other business documents."""
        return [
            "invoice",
            "receipt",
            "bill",
            "statement",
            "quote",
            "quotation",
            "estimate",
            "proposal",
            "contract",
            "agreement",
            "order",
            "purchase",
            "delivery",
            "shipping",
            "freight",
            "logistics",
            "courier",
            "postal",
            "postage",
            "subscription",
            "membership",
            "license",
            "permit",
            "registration",
            "fee",
            "fine",
            "penalty",
            "refund",
            "credit",
            "adjustment",
            "discount",
            "promotion",
            "voucher",
            "coupon",
            "gift card",
            "warranty",
            "insurance",
            "premium",
            "claim",
            "policy",
            "certificate",
            "document",
            "business",
            "commercial",
            "transaction",
            "payment",
            "remittance",
            "advice",
            "notice",
            "notification",
            "reminder",
            "overdue",
            "due",
            "payable",
            "receivable",
            "balance",
            "outstanding",
            "account",
            "customer",
            "client",
            "vendor",
            "supplier",
            "service",
            "product",
            "item",
            "goods",
            "services",
            "consultation",
            "meeting",
            "appointment",
            "booking",
            "reservation",
            "cancellation",
            "refund",
            "exchange",
            "return",
            "replacement",
            "repair",
            "maintenance",
            "support",
            "help",
            "assistance",
            "training",
            "education",
            "course",
            "workshop",
            "seminar",
            "conference",
            "event",
            "ticket",
            "admission",
            "entry",
            "pass",
            "permit",
            "authorization",
            "approval",
            "clearance",
            "compliance",
            "audit",
            "inspection",
            "verification",
            "validation",
            "confirmation",
            "acknowledgment",
            "acceptance",
            "rejection",
            "cancellation",
            "termination",
            "suspension",
            "renewal",
            "extension",
            "modification",
            "amendment",
            "update",
            "revision",
            "version",
            "edition",
            "copy",
            "duplicate",
            "original",
            "certified",
            "official",
            "authorized",
            "approved",
            "stamped",
            "signed",
            "witnessed",
            "notarized",
        ]

    def get_classification_patterns(self) -> List[Pattern]:
        """Get regex patterns for other document classification."""
        return [
            re.compile(r"invoice|receipt|bill|statement", re.IGNORECASE),
            re.compile(r"quote|quotation|estimate|proposal", re.IGNORECASE),
            re.compile(r"contract|agreement|order", re.IGNORECASE),
            re.compile(r"delivery|shipping|freight|courier", re.IGNORECASE),
            re.compile(r"subscription|membership|license", re.IGNORECASE),
            re.compile(r"registration|fee|fine|penalty", re.IGNORECASE),
            re.compile(r"refund|credit|adjustment|discount", re.IGNORECASE),
            re.compile(r"warranty|insurance|premium|claim", re.IGNORECASE),
            re.compile(r"certificate|document|business", re.IGNORECASE),
            re.compile(r"payment|remittance|advice|notice", re.IGNORECASE),
            re.compile(r"customer|client|vendor|supplier", re.IGNORECASE),
            re.compile(r"service|product|consultation", re.IGNORECASE),
            re.compile(r"booking|reservation|appointment", re.IGNORECASE),
            re.compile(r"training|education|course|workshop", re.IGNORECASE),
            re.compile(r"conference|event|ticket|admission", re.IGNORECASE),
            re.compile(r"permit|authorization|approval", re.IGNORECASE),
            re.compile(r"compliance|audit|inspection", re.IGNORECASE),
            re.compile(r"confirmation|acknowledgment", re.IGNORECASE),
            re.compile(r"official|authorized|certified", re.IGNORECASE),
        ]

    def get_prompt_name(self) -> str:
        """Get prompt name for other documents."""
        return "other_document_extraction_prompt"

    def get_field_patterns(self) -> List[DocumentPattern]:
        """Get field extraction patterns for other documents."""
        return [
            DocumentPattern(
                pattern=r"DATE:\s*([^\n\r]+)",
                field_name="DATE",
                field_type="date",
                required=True,
            ),
            DocumentPattern(
                pattern=r"BUSINESS:\s*([^\n\r]+)",
                field_name="BUSINESS",
                field_type="string",
                required=True,
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
                required=False,
            ),
            DocumentPattern(
                pattern=r"GST:\s*([^\n\r]+)",
                field_name="GST",
                field_type="number",
                required=False,
            ),
            DocumentPattern(
                pattern=r"DESCRIPTION:\s*([^\n\r]+)",
                field_name="DESCRIPTION",
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
                pattern=r"CUSTOMER:\s*([^\n\r]+)",
                field_name="CUSTOMER",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"CONTACT:\s*([^\n\r]+)",
                field_name="CONTACT",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"PHONE:\s*([^\n\r]+)",
                field_name="PHONE",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"EMAIL:\s*([^\n\r]+)",
                field_name="EMAIL",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"WEBSITE:\s*([^\n\r]+)",
                field_name="WEBSITE",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"PAYMENT_METHOD:\s*([^\n\r]+)",
                field_name="PAYMENT_METHOD",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"DOCUMENT_TYPE:\s*([^\n\r]+)",
                field_name="DOCUMENT_TYPE",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"DOCUMENT_NUMBER:\s*([^\n\r]+)",
                field_name="DOCUMENT_NUMBER",
                field_type="string",
                required=False,
            ),
        ]

    def get_field_mappings(self) -> Dict[str, List[str]]:
        """Get field mappings for standardization."""
        return {
            # Standard compliance fields
            "supplier_name": ["BUSINESS"],
            "total_amount": ["TOTAL"],
            "gst_amount": ["GST"],
            "invoice_date": ["DATE"],
            "supplier_abn": ["ABN"],
            "payment_method": ["PAYMENT_METHOD"],
            "receipt_number": ["DOCUMENT_NUMBER", "REFERENCE"],
            "invoice_number": ["DOCUMENT_NUMBER", "REFERENCE"],
            # Other document-specific fields
            "business_name": ["BUSINESS"],
            "document_description": ["DESCRIPTION"],
            "document_reference": ["REFERENCE"],
            "customer_name": ["CUSTOMER"],
            "contact_person": ["CONTACT"],
            "phone_number": ["PHONE"],
            "email_address": ["EMAIL"],
            "website_url": ["WEBSITE"],
            "document_type": ["DOCUMENT_TYPE"],
            "document_number": ["DOCUMENT_NUMBER"],
            "business_address": ["ADDRESS"],
            "transaction_date": ["DATE"],
            "service_description": ["DESCRIPTION"],
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

        # If we found less than 3 meaningful fields, use fallback pattern matching
        # We expect fewer fields for "other" documents, so <3 indicates KEY-VALUE parsing failed
        if result.field_count < 3:
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
        from .awk_extractor import OtherDocumentAwkExtractor

        awk_extractor = OtherDocumentAwkExtractor(self.log_level)
        extracted = awk_extractor.extract_other_document_fields(response)

        self.logger.debug(
            f"AWK fallback extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted
