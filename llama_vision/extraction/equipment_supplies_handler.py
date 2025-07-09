"""Equipment supplies document type handler."""

import re
from typing import Any, Dict, List, Pattern

from .document_handlers import DocumentPattern, DocumentTypeHandler, ExtractionResult


class EquipmentSuppliesHandler(DocumentTypeHandler):
    """Handler for equipment and supplies documents."""

    @property
    def document_type(self) -> str:
        return "equipment_supplies"

    @property
    def display_name(self) -> str:
        return "Equipment Supplies"

    def get_classification_indicators(self) -> List[str]:
        """Get text indicators for equipment and supplies documents."""
        return [
            "computer",
            "laptop",
            "desktop",
            "monitor",
            "keyboard",
            "mouse",
            "printer",
            "scanner",
            "software",
            "hardware",
            "electronics",
            "equipment",
            "supplies",
            "office supplies",
            "stationery",
            "paper",
            "ink",
            "toner",
            "cartridge",
            "cable",
            "adapter",
            "charger",
            "battery",
            "memory",
            "storage",
            "hard drive",
            "ssd",
            "usb",
            "hdmi",
            "ethernet",
            "wifi",
            "router",
            "modem",
            "switch",
            "hub",
            "webcam",
            "headset",
            "microphone",
            "speaker",
            "phone",
            "tablet",
            "ipad",
            "iphone",
            "android",
            "samsung",
            "apple",
            "microsoft",
            "dell",
            "hp",
            "lenovo",
            "asus",
            "acer",
            "canon",
            "epson",
            "brother",
            "xerox",
            "officeworks",
            "harvey norman",
            "jb hi-fi",
            "centrecom",
            "mwave",
            "pccasegear",
            "scorptec",
            "umart",
            "warranty",
            "license",
            "subscription",
            "renewal",
            "upgrade",
            "installation",
            "setup",
            "configuration",
            "maintenance",
            "repair",
            "replacement",
            "parts",
            "accessories",
        ]

    def get_classification_patterns(self) -> List[Pattern]:
        """Get regex patterns for equipment supplies classification."""
        return [
            re.compile(r"computer|laptop|desktop|monitor", re.IGNORECASE),
            re.compile(r"software|hardware|electronics", re.IGNORECASE),
            re.compile(r"equipment|supplies|stationery", re.IGNORECASE),
            re.compile(r"printer|scanner|ink|toner", re.IGNORECASE),
            re.compile(r"cable|adapter|charger|battery", re.IGNORECASE),
            re.compile(r"memory|storage|hard drive|ssd", re.IGNORECASE),
            re.compile(r"usb|hdmi|ethernet|wifi", re.IGNORECASE),
            re.compile(r"webcam|headset|microphone|speaker", re.IGNORECASE),
            re.compile(r"phone|tablet|ipad|iphone", re.IGNORECASE),
            re.compile(r"apple|microsoft|dell|hp|lenovo", re.IGNORECASE),
            re.compile(r"canon|epson|brother|xerox", re.IGNORECASE),
            re.compile(r"officeworks|harvey\s+norman|jb\s+hi-fi", re.IGNORECASE),
            re.compile(r"warranty|license|subscription", re.IGNORECASE),
            re.compile(r"installation|setup|configuration", re.IGNORECASE),
            re.compile(r"maintenance|repair|replacement", re.IGNORECASE),
        ]

    def get_prompt_name(self) -> str:
        """Get prompt name for equipment supplies documents."""
        return "equipment_supplies_extraction_prompt"

    def get_field_patterns(self) -> List[DocumentPattern]:
        """Get field extraction patterns for equipment supplies documents."""
        return [
            DocumentPattern(
                pattern=r"DATE:\s*([^\n\r]+)",
                field_name="DATE",
                field_type="date",
                required=True,
            ),
            DocumentPattern(
                pattern=r"SUPPLIER:\s*([^\n\r]+)",
                field_name="SUPPLIER",
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
                required=True,
            ),
            DocumentPattern(
                pattern=r"GST:\s*([^\n\r]+)",
                field_name="GST",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"ITEMS:\s*([^\n\r]+)",
                field_name="ITEMS",
                field_type="list",
                required=True,
            ),
            DocumentPattern(
                pattern=r"QUANTITIES:\s*([^\n\r]+)",
                field_name="QUANTITIES",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"PRICES:\s*([^\n\r]+)",
                field_name="PRICES",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"CATEGORIES:\s*([^\n\r]+)",
                field_name="CATEGORIES",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"BRANDS:\s*([^\n\r]+)",
                field_name="BRANDS",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"MODELS:\s*([^\n\r]+)",
                field_name="MODELS",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"SKU:\s*([^\n\r]+)",
                field_name="SKU",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"WARRANTY:\s*([^\n\r]+)",
                field_name="WARRANTY",
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
                pattern=r"INVOICE:\s*([^\n\r]+)",
                field_name="INVOICE",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"ORDER:\s*([^\n\r]+)",
                field_name="ORDER",
                field_type="string",
                required=False,
            ),
        ]

    def get_field_mappings(self) -> Dict[str, List[str]]:
        """Get field mappings for standardization."""
        return {
            # Standard compliance fields
            "supplier_name": ["SUPPLIER"],
            "total_amount": ["TOTAL"],
            "gst_amount": ["GST"],
            "invoice_date": ["DATE"],
            "supplier_abn": ["ABN"],
            "items": ["ITEMS"],
            "quantities": ["QUANTITIES"],
            "prices": ["PRICES"],
            "payment_method": ["PAYMENT_METHOD"],
            "receipt_number": ["INVOICE", "ORDER"],
            "invoice_number": ["INVOICE", "ORDER"],
            # Equipment/supplies-specific fields
            "equipment_supplier": ["SUPPLIER"],
            "equipment_items": ["ITEMS"],
            "equipment_categories": ["CATEGORIES"],
            "equipment_brands": ["BRANDS"],
            "equipment_models": ["MODELS"],
            "product_sku": ["SKU"],
            "warranty_period": ["WARRANTY"],
            "order_number": ["ORDER"],
            "invoice_reference": ["INVOICE"],
            "supplier_address": ["ADDRESS"],
            "purchase_date": ["DATE"],
            "item_prices": ["PRICES"],
            "item_quantities": ["QUANTITIES"],
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

        # If we found less than 5 meaningful fields, use fallback pattern matching
        # We expect 6+ fields for equipment/supplies, so <5 indicates KEY-VALUE parsing failed
        if result.field_count < 5:
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
        from .awk_extractor import EquipmentSuppliesAwkExtractor

        awk_extractor = EquipmentSuppliesAwkExtractor(self.log_level)
        extracted = awk_extractor.extract_equipment_supplies_fields(response)

        self.logger.debug(
            f"AWK fallback extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted
