"""Fuel receipt document type handler."""

import re
from typing import Any, Dict, List, Pattern

from .document_handlers import DocumentPattern, DocumentTypeHandler, ExtractionResult


class FuelReceiptHandler(DocumentTypeHandler):
    """Handler for fuel receipt documents."""

    @property
    def document_type(self) -> str:
        return "fuel_receipt"

    @property
    def display_name(self) -> str:
        return "Fuel Receipt"

    def get_classification_indicators(self) -> List[str]:
        """Get text indicators for fuel receipts."""
        return [
            "13ulp",
            "ulp",
            "unleaded",
            "diesel",
            "litre",
            " l ",
            ".l ",
            "price/l",
            "per litre",
            "fuel",
            "petrol",
            "costco",
            "shell",
            "bp",
            "coles express",
            "7-eleven",
        ]

    def get_classification_patterns(self) -> List[Pattern]:
        """Get regex patterns for fuel receipt classification."""
        return [
            re.compile(r"\d+\.\d{2,3}\s*l\b", re.IGNORECASE),  # Fuel quantity pattern
            re.compile(r"\d{3}/l", re.IGNORECASE),  # Price per litre in cents
            re.compile(r"\$\d+\.\d{2}/l", re.IGNORECASE),  # Price per litre in dollars
            re.compile(r"(13ulp|u91|u95|u98|e10)", re.IGNORECASE),  # Fuel type codes
        ]

    def get_prompt_name(self) -> str:
        """Get prompt name for fuel receipts."""
        return "fuel_receipt_extraction_prompt"

    def get_field_patterns(self) -> List[DocumentPattern]:
        """Get field extraction patterns for fuel receipts."""
        return [
            DocumentPattern(
                pattern=r"DATE:\s*([^\n\r]+)",
                field_name="DATE",
                field_type="date",
                required=True,
            ),
            DocumentPattern(
                pattern=r"STORE:\s*([^\n\r]+)",
                field_name="STORE",
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
                pattern=r"PAYER:\s*([^\n\r]+)",
                field_name="PAYER",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"TAX:\s*([^\n\r]+)",
                field_name="TAX",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"TOTAL:\s*([^\n\r]+)",
                field_name="TOTAL",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"PRODUCTS:\s*([^\n\r]+)",
                field_name="PRODUCTS",
                field_type="list",
                required=True,
            ),
            DocumentPattern(
                pattern=r"QUANTITIES:\s*([^\n\r]+)",
                field_name="QUANTITIES",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"PRICES:\s*([^\n\r]+)",
                field_name="PRICES",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"PAYMENT_METHOD:\s*([^\n\r]+)",
                field_name="PAYMENT_METHOD",
                field_type="string",
                required=False,
            ),
            DocumentPattern(
                pattern=r"RECEIPT:\s*([^\n\r]+)",
                field_name="RECEIPT",
                field_type="string",
                required=False,
            ),
        ]

    def get_field_mappings(self) -> Dict[str, List[str]]:
        """Get field mappings for standardization."""
        return {
            # Standard compliance fields
            "supplier_name": ["STORE"],
            "total_amount": ["TOTAL"],
            "gst_amount": ["TAX"],
            "invoice_date": ["DATE"],
            "supplier_abn": ["ABN"],
            "items": ["PRODUCTS"],
            "quantities": ["QUANTITIES"],
            "prices": ["PRICES"],
            "payment_method": ["PAYMENT_METHOD"],
            "receipt_number": ["RECEIPT"],
            "invoice_number": ["RECEIPT"],
            # Fuel-specific fields
            "fuel_type": ["PRODUCTS"],
            "fuel_quantity": ["QUANTITIES"],
            "price_per_litre": ["PRICES"],
            "fuel_station": ["STORE"],
            "member_number": ["PAYER"],
        }

    def extract_fields(self, response: str) -> ExtractionResult:
        """Extract fields with fallback pattern matching like TaxAuthorityParser.

        Args:
            response: Model response text

        Returns:
            Extraction result with fields and metadata
        """
        # First try the standard KEY-VALUE approach
        result = super().extract_fields(response)

        # If we found less than 6 meaningful fields, use fallback pattern matching
        # We expect 8+ fields for fuel receipts, so <6 indicates KEY-VALUE parsing failed
        if result.field_count < 6:
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
        """Extract fields from raw OCR text when KEY-VALUE format fails.

        This implements the same fallback mechanism that makes TaxAuthorityParser successful.

        Args:
            response: Raw model response text

        Returns:
            Extracted fields dictionary
        """
        extracted = {}

        # Extract date patterns
        date_patterns = [
            r"(\d{2}/\d{2}/\d{4})",  # DD/MM/YYYY
            r"(\d{1,2}/\d{1,2}/\d{4})",  # D/M/YYYY or DD/M/YYYY
            r"(\d{4}-\d{2}-\d{2})",  # YYYY-MM-DD
        ]

        for pattern in date_patterns:
            match = re.search(pattern, response)
            if match:
                extracted["DATE"] = match.group(1)
                break

        # Extract store/business name - look for common fuel retailers
        store_patterns = [
            r"(COSTCO\s+WHOLESALE\s+AUSTRALIA)",
            r"(COSTCO)",
            r"(SHELL)",
            r"(BP)",
            r"(COLES\s+EXPRESS)",
            r"(7-ELEVEN)",
            r"(WOOLWORTHS\s+PETROL)",
            r"(AMPOL)",
            r"(MOBIL)",
            r"(UNITED\s+PETROLEUM)",
        ]

        for pattern in store_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted["STORE"] = match.group(1).upper()
                break

        # Extract ABN
        abn_patterns = [
            r"ABN:?\s*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
            r"(\d{2}\s\d{3}\s\d{3}\s\d{3})",  # Standard ABN format
        ]

        for pattern in abn_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted["ABN"] = match.group(1)
                break

        # Extract fuel quantity
        quantity_patterns = [
            r"(\d+\.\d{3})L",  # Costco: 32.230L
            r"(\d+\.\d{2})L",  # Shell/BP: 45.67L
            r"(\d+\.\d{1})L",  # Some retailers: 32.2L
        ]

        for pattern in quantity_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted["QUANTITIES"] = f"{match.group(1)}L"
                break

        # Extract price per litre
        price_patterns = [
            r"\$(\d+\.\d{3})/L",  # $1.827/L
            r"(\d{3})/L",  # 827/L (cents)
            r"\$(\d+\.\d{2})/L",  # $1.85/L
        ]

        for i, pattern in enumerate(price_patterns):
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                if i == 1:  # Cents format
                    price_dollars = float(match.group(1)) / 100
                    extracted["PRICES"] = f"${price_dollars:.3f}/L"
                else:
                    extracted["PRICES"] = f"${match.group(1)}/L"
                break

        # Extract fuel type
        fuel_type_patterns = [
            r"(13ULP)",  # Costco
            r"(U91|ULP|UNLEADED)",  # Standard unleaded
            r"(U95|PREMIUM\s*ULP)",  # Premium unleaded
            r"(U98|SUPER\s*PREMIUM)",  # Super premium
            r"(DIESEL|DSL)",  # Diesel
            r"(E10)",  # Ethanol blend
        ]

        for pattern in fuel_type_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted["PRODUCTS"] = match.group(1).upper()
                break

        # Extract total amount
        total_patterns = [
            r"TOTAL[^\d]*\$(\d+\.\d{2})",
            r"\$(\d+\.\d{2})\s*TOTAL",
            r"AMOUNT[^\d]*\$(\d+\.\d{2})",
        ]

        for pattern in total_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted["TOTAL"] = f"${match.group(1)}"
                break

        # Extract GST/Tax amount
        tax_patterns = [
            r"GST[^\d]*\$(\d+\.\d{2})",
            r"Tax[^\d]*\$(\d+\.\d{2})",
            r"\$(\d+\.\d{2})\s*GST",
            r"\$(\d+\.\d{2})\s*Tax",
        ]

        for pattern in tax_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted["TAX"] = f"${match.group(1)}"
                break

        # Extract payment method
        payment_patterns = [
            r"(CREDIT|DEBIT|EFTPOS|CASH)",
            r"CARD[^\w]*(\w+)",
            r"VISA|MASTERCARD|AMEX",
        ]

        for pattern in payment_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted["PAYMENT_METHOD"] = match.group(0).upper()
                break

        # Extract receipt/transaction number
        receipt_patterns = [
            r"Receipt[^\d]*(\d+)",
            r"Transaction[^\d]*(\d+)",
            r"Auth[^\d]*(\d+)",
            r"(\d{6,})",  # Generic long number
        ]

        for pattern in receipt_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted["RECEIPT"] = match.group(1)
                break

        # Extract member/loyalty number
        member_patterns = [
            r"Member[^\d]*(\d+)",
            r"Card[^\d]*(\d+)",
            r"Account[^\d]*(\d+)",
        ]

        for pattern in member_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted["PAYER"] = match.group(1)
                break

        self.logger.debug(
            f"Fallback extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted
