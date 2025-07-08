"""AWK-style field extraction for maintainable text processing."""

import re
from typing import Any, Dict, List


class AwkExtractor:
    """AWK-style text processor for field extraction.

    Much more maintainable than complex regex patterns.
    """

    def __init__(self, log_level: str = "INFO"):
        """Initialize AWK-style extractor."""
        from ..utils import setup_logging

        self.logger = setup_logging(log_level)

    def extract_fields(
        self, text: str, field_rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract fields using AWK-style rules.

        Args:
            text: Input text to process
            field_rules: List of extraction rules

        Returns:
            Extracted fields dictionary
        """
        extracted = {}
        lines = text.split("\n")

        for rule in field_rules:
            field_name = rule["field"]
            patterns = rule.get("patterns", [])
            line_filters = rule.get("line_filters", [])
            transformers = rule.get("transform", [])

            # Find matching lines
            matching_lines = self._filter_lines(lines, line_filters)

            # Extract from patterns
            for line in matching_lines:
                for pattern in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        value = match.group(1) if match.groups() else match.group(0)

                        # Apply transformations
                        for transform in transformers:
                            value = self._apply_transform(value, transform)

                        extracted[field_name] = value
                        self.logger.debug(f"Extracted {field_name}: {value}")
                        break

                if field_name in extracted:
                    break

        return extracted

    def _filter_lines(self, lines: List[str], filters: List[str]) -> List[str]:
        """Filter lines based on AWK-style conditions."""
        if not filters:
            return lines

        matching_lines = []
        for line in lines:
            for line_filter in filters:
                if self._line_matches_filter(line, line_filter):
                    matching_lines.append(line)
                    break

        return matching_lines

    def _line_matches_filter(self, line: str, filter_expr: str) -> bool:
        """Check if line matches AWK-style filter."""
        # Simple AWK-like patterns
        if filter_expr.startswith("/") and filter_expr.endswith("/"):
            # Regex pattern
            pattern = filter_expr[1:-1]
            return bool(re.search(pattern, line, re.IGNORECASE))
        elif "NF" in filter_expr:
            # Number of fields
            fields = line.split()
            nf = len(fields)
            try:
                return eval(filter_expr.replace("NF", str(nf)))
            except (ValueError, NameError, SyntaxError):
                return False
        elif "$" in filter_expr:
            # Field references
            fields = line.split()
            # Simple $1, $2 etc. support
            for i, field in enumerate(fields, 1):
                filter_expr = filter_expr.replace(f"${i}", f'"{field}"')
            try:
                return eval(filter_expr)
            except (ValueError, NameError, SyntaxError):
                return False
        else:
            # Simple substring match
            return filter_expr.lower() in line.lower()

    def _apply_transform(self, value: str, transform: str) -> str:
        """Apply transformation to extracted value."""
        if transform == "upper":
            return value.upper()
        elif transform == "lower":
            return value.lower()
        elif transform == "strip":
            return value.strip()
        elif transform == "remove_spaces":
            return re.sub(r"\s+", "", value)
        elif transform == "normalize_spaces":
            return " ".join(value.split())
        elif transform.startswith("prefix:"):
            prefix = transform[7:]
            return f"{prefix}{value}"
        elif transform.startswith("suffix:"):
            suffix = transform[7:]
            return f"{value}{suffix}"
        elif transform.startswith("replace:"):
            # Format: replace:old,new
            parts = transform[8:].split(",", 1)
            if len(parts) == 2:
                return value.replace(parts[0], parts[1])

        return value


class FuelReceiptAwkExtractor(AwkExtractor):
    """Fuel receipt specific AWK-style extractor."""

    def get_fuel_extraction_rules(self) -> List[Dict[str, Any]]:
        """Get AWK-style extraction rules for fuel receipts."""
        return [
            {
                "field": "STORE",
                "line_filters": [
                    "/costco|shell|bp|coles|7-eleven|woolworths|ampol|mobil/"
                ],
                "patterns": [
                    r"(COSTCO\s+WHOLESALE\s+AUSTRALIA)",
                    r"(COSTCO)",
                    r"(SHELL)",
                    r"(BP)",
                    r"(COLES\s+EXPRESS)",
                    r"(7-ELEVEN)",
                    r"(WOOLWORTHS\s+PETROL)",
                    r"(AMPOL)",
                    r"(MOBIL)",
                ],
                "transform": ["upper", "normalize_spaces"],
            },
            {
                "field": "ABN",
                "line_filters": ["/abn|\d{2}\s\d{3}\s\d{3}\s\d{3}/"],
                "patterns": [
                    r"ABN:?\s*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
                    r"(\d{2}\s\d{3}\s\d{3}\s\d{3})",
                ],
                "transform": ["normalize_spaces"],
            },
            {
                "field": "QUANTITIES",
                "line_filters": ["/.+L\b/"],
                "patterns": [
                    r"(\d+\.\d{3})L",  # 32.230L
                    r"(\d+\.\d{2})L",  # 45.67L
                    r"(\d+\.\d{1})L",  # 32.2L
                ],
                "transform": ["suffix:L"],
            },
            {
                "field": "PRICES",
                "line_filters": ["/\$/L|/L/"],
                "patterns": [
                    r"\$(\d+\.\d{3})/L",  # $1.827/L
                    r"(\d{3})/L",  # 827/L (cents)
                    r"\$(\d+\.\d{2})/L",  # $1.85/L
                ],
                "transform": ["prefix:$", "suffix:/L"],
            },
            {
                "field": "PRODUCTS",
                "line_filters": ["/13ulp|u91|u95|u98|ulp|unleaded|diesel|e10/"],
                "patterns": [
                    r"(13ULP)",
                    r"(U91|ULP|UNLEADED)",
                    r"(U95|PREMIUM\s*ULP)",
                    r"(U98|SUPER\s*PREMIUM)",
                    r"(DIESEL|DSL)",
                    r"(E10)",
                ],
                "transform": ["upper"],
            },
            {
                "field": "TOTAL",
                "line_filters": ["/total|amount/"],
                "patterns": [
                    r"TOTAL[^\d]*\$(\d+\.\d{2})",
                    r"\$(\d+\.\d{2})\s*TOTAL",
                    r"AMOUNT[^\d]*\$(\d+\.\d{2})",
                ],
                "transform": ["prefix:$"],
            },
            {
                "field": "TAX",
                "line_filters": ["/gst|tax/"],
                "patterns": [
                    r"GST[^\d]*\$(\d+\.\d{2})",
                    r"Tax[^\d]*\$(\d+\.\d{2})",
                    r"\$(\d+\.\d{2})\s*GST",
                    r"\$(\d+\.\d{2})\s*Tax",
                ],
                "transform": ["prefix:$"],
            },
            {
                "field": "PAYMENT_METHOD",
                "line_filters": ["/credit|debit|eftpos|cash|visa|mastercard/"],
                "patterns": [
                    r"(CREDIT|DEBIT|EFTPOS|CASH)",
                    r"(VISA|MASTERCARD|AMEX)",
                ],
                "transform": ["upper"],
            },
            {
                "field": "RECEIPT",
                "line_filters": ["/auth|receipt|transaction|\d{6,}/"],
                "patterns": [
                    r"Auth[^\d]*(\d+)",
                    r"Receipt[^\d]*(\d+)",
                    r"Transaction[^\d]*(\d+)",
                    r"(\d{8,})",  # Long numbers
                ],
                "transform": ["strip"],
            },
            {
                "field": "DATE",
                "line_filters": ["NF > 2"],  # Lines with multiple fields
                "patterns": [
                    r"(\d{2}/\d{2}/\d{4})",
                    r"(\d{1,2}/\d{1,2}/\d{4})",
                    r"(\d{4}-\d{2}-\d{2})",
                ],
                "transform": ["strip"],
            },
            {
                "field": "PAYER",
                "line_filters": ["/member|card|account/"],
                "patterns": [
                    r"Member[^\d]*(\d+)",
                    r"Card[^\d]*(\d+)",
                    r"Account[^\d]*(\d+)",
                ],
                "transform": ["strip"],
            },
        ]

    def extract_fuel_fields(self, response: str) -> Dict[str, Any]:
        """Extract fuel receipt fields using AWK-style rules."""
        rules = self.get_fuel_extraction_rules()
        extracted = self.extract_fields(response, rules)

        # Post-process specific to fuel receipts
        if "PRICES" in extracted and "/L" not in extracted["PRICES"]:
            # Handle cents format
            if extracted["PRICES"].isdigit() and len(extracted["PRICES"]) == 3:
                price_dollars = float(extracted["PRICES"]) / 100
                extracted["PRICES"] = f"${price_dollars:.3f}/L"

        self.logger.debug(
            f"AWK extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted


class BankStatementAwkExtractor(AwkExtractor):
    """Bank statement specific AWK-style extractor."""

    def get_bank_extraction_rules(self) -> List[Dict[str, Any]]:
        """Get AWK-style extraction rules for bank statements."""
        return [
            {
                "field": "ACCOUNT_NUMBER",
                "line_filters": ["/account|acc/"],
                "patterns": [
                    r"Account\s+Number[:\s]*(\d+)",
                    r"Account[:\s]*(\d+)",
                    r"Acc[:\s]*(\d+)",
                ],
                "transform": ["strip"],
            },
            {
                "field": "BSB",
                "line_filters": ["/bsb|\d{2,3}-\d{3}/"],
                "patterns": [
                    r"BSB[:\s]*(\d{2,3}-\d{3})",
                    r"(\d{2,3}-\d{3})",
                ],
                "transform": ["strip"],
            },
            {
                "field": "ACCOUNT_HOLDER",
                "line_filters": ["/account.*[a-z]{2,}/"],
                "patterns": [
                    r"Account\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
                    r"Account\s+Holder[:\s]*([A-Z][a-zA-Z\s]+)",
                ],
                "transform": ["upper", "normalize_spaces"],
            },
            {
                "field": "BANK_NAME",
                "line_filters": ["/anz|commonwealth|westpac|nab/"],
                "patterns": [
                    r"(ANZ\s+BANK)",
                    r"(COMMONWEALTH\s+BANK)",
                    r"(WESTPAC)",
                    r"(NAB)",
                    r"(ANZ)",
                ],
                "transform": ["upper"],
            },
            # Add more bank-specific rules...
        ]

    def extract_bank_fields(self, response: str) -> Dict[str, Any]:
        """Extract bank statement fields using AWK-style rules."""
        rules = self.get_bank_extraction_rules()
        return self.extract_fields(response, rules)
