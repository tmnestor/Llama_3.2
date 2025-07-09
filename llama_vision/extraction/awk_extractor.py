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


class MealReceiptAwkExtractor(AwkExtractor):
    """Meal receipt specific AWK-style extractor."""

    def get_meal_extraction_rules(self) -> List[Dict[str, Any]]:
        """Get AWK-style extraction rules for meal receipts."""
        return [
            {
                "field": "RESTAURANT",
                "line_filters": [
                    "/restaurant|cafe|catering|kitchen|bistro|diner|eatery|grill|pub|bar/"
                ],
                "patterns": [
                    r"(.*RESTAURANT.*)",
                    r"(.*CAFE.*)",
                    r"(.*CATERING.*)",
                    r"(.*KITCHEN.*)",
                    r"(.*BISTRO.*)",
                    r"(.*DINER.*)",
                    r"(.*EATERY.*)",
                    r"(.*GRILL.*)",
                    r"(.*PUB.*)",
                    r"(.*BAR.*)",
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
                "field": "TOTAL",
                "line_filters": ["/total|amount|balance/"],
                "patterns": [
                    r"TOTAL[^\d]*\$(\d+\.\d{2})",
                    r"\$(\d+\.\d{2})\s*TOTAL",
                    r"AMOUNT[^\d]*\$(\d+\.\d{2})",
                    r"BALANCE[^\d]*\$(\d+\.\d{2})",
                ],
                "transform": ["prefix:$"],
            },
            {
                "field": "GST",
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
                "field": "ITEMS",
                "line_filters": [
                    "/\d+\s+x\s+|qty\s+\d+|\d+\.\d{2}\s*$|@\s*\d+\.\d{2}/"
                ],
                "patterns": [
                    r"(\d+\s+x\s+.+)",  # Quantity format: "2 x Coffee"
                    r"(.*\s+@\s*\$?\d+\.\d{2})",  # Price format: "Coffee @ $4.50"
                    r"([A-Z][a-z]+.*\s+\$?\d+\.\d{2})",  # Item with price
                ],
                "transform": ["normalize_spaces"],
            },
            {
                "field": "PAYMENT_METHOD",
                "line_filters": [
                    "/credit|debit|eftpos|cash|visa|mastercard|amex|paypal/"
                ],
                "patterns": [
                    r"(CREDIT|DEBIT|EFTPOS|CASH)",
                    r"(VISA|MASTERCARD|AMEX)",
                    r"(PAYPAL|APPLE\s+PAY|GOOGLE\s+PAY)",
                ],
                "transform": ["upper"],
            },
            {
                "field": "TABLE",
                "line_filters": ["/table\s+\d+/"],
                "patterns": [
                    r"Table\s+(\d+)",
                    r"TABLE\s+(\d+)",
                    r"Tbl\s+(\d+)",
                ],
                "transform": ["upper"],
            },
            {
                "field": "ORDER",
                "line_filters": ["/order\s+\d+|order\s+no/"],
                "patterns": [
                    r"Order\s+(\d+)",
                    r"ORDER\s+(\d+)",
                    r"Order\s+No[:\s]*(\d+)",
                    r"(\d{4,})",  # Long order numbers
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
                "field": "ADDRESS",
                "line_filters": [
                    "/street|road|ave|avenue|drive|lane|place|suburb|city/"
                ],
                "patterns": [
                    r"(\d+\s+.*(?:Street|Road|Ave|Avenue|Drive|Lane|Place))",
                    r"(.*(?:Street|Road|Ave|Avenue|Drive|Lane|Place).*)",
                    r"([A-Z][a-z]+\s+[A-Z][a-z]+.*\d{4})",  # Suburb State PostCode
                ],
                "transform": ["normalize_spaces"],
            },
        ]

    def extract_meal_fields(self, response: str) -> Dict[str, Any]:
        """Extract meal receipt fields using AWK-style rules."""
        rules = self.get_meal_extraction_rules()
        extracted = self.extract_fields(response, rules)

        # Post-process specific to meal receipts
        if "ITEMS" in extracted:
            # Clean up item descriptions
            items = extracted["ITEMS"]
            if isinstance(items, str):
                # Remove price information from item descriptions
                items = re.sub(r"\s*@\s*\$?\d+\.\d{2}", "", items)
                items = re.sub(r"\s*\$?\d+\.\d{2}\s*$", "", items)
                extracted["ITEMS"] = items.strip()

        self.logger.debug(
            f"AWK meal extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted


class AccommodationAwkExtractor(AwkExtractor):
    """Accommodation specific AWK-style extractor."""

    def get_accommodation_extraction_rules(self) -> List[Dict[str, Any]]:
        """Get AWK-style extraction rules for accommodation documents."""
        return [
            {
                "field": "HOTEL",
                "line_filters": [
                    "/hotel|motel|accommodation|booking|airbnb|lodging|resort|inn|hostel/"
                ],
                "patterns": [
                    r"(.*HOTEL.*)",
                    r"(.*MOTEL.*)",
                    r"(.*ACCOMMODATION.*)",
                    r"(.*RESORT.*)",
                    r"(.*INN.*)",
                    r"(.*HOSTEL.*)",
                    r"(.*LODGE.*)",
                    r"(.*GUEST\s+HOUSE.*)",
                    r"(.*B&B.*)",
                    r"(.*HILTON.*)",
                    r"(.*MARRIOTT.*)",
                    r"(.*HYATT.*)",
                    r"(.*IBIS.*)",
                    r"(.*NOVOTEL.*)",
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
                "field": "TOTAL",
                "line_filters": ["/total|amount|balance|charged/"],
                "patterns": [
                    r"TOTAL[^\d]*\$(\d+\.\d{2})",
                    r"\$(\d+\.\d{2})\s*TOTAL",
                    r"AMOUNT[^\d]*\$(\d+\.\d{2})",
                    r"CHARGED[^\d]*\$(\d+\.\d{2})",
                ],
                "transform": ["prefix:$"],
            },
            {
                "field": "GST",
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
                "field": "GUEST",
                "line_filters": ["/guest|name/"],
                "patterns": [
                    r"Guest\s+Name[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"Guest[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"Name[:\s]*([A-Z][a-zA-Z\s]+)",
                ],
                "transform": ["upper", "normalize_spaces"],
            },
            {
                "field": "ROOM",
                "line_filters": ["/room\s+\d+|suite\s+\d+/"],
                "patterns": [
                    r"Room\s+(\d+)",
                    r"ROOM\s+(\d+)",
                    r"Suite\s+(\d+)",
                    r"SUITE\s+(\d+)",
                ],
                "transform": ["upper"],
            },
            {
                "field": "NIGHTS",
                "line_filters": ["/\d+\s+nights?/"],
                "patterns": [
                    r"(\d+)\s+nights?",
                    r"(\d+)\s+NIGHTS?",
                ],
                "transform": ["strip"],
            },
            {
                "field": "CHECK_IN",
                "line_filters": ["/check.?in|arrival/"],
                "patterns": [
                    r"Check.?in[:\s]*(\d{2}/\d{2}/\d{4})",
                    r"Check.?in[:\s]*(\d{1,2}/\d{1,2}/\d{4})",
                    r"Arrival[:\s]*(\d{2}/\d{2}/\d{4})",
                    r"Arrival[:\s]*(\d{1,2}/\d{1,2}/\d{4})",
                ],
                "transform": ["strip"],
            },
            {
                "field": "CHECK_OUT",
                "line_filters": ["/check.?out|departure/"],
                "patterns": [
                    r"Check.?out[:\s]*(\d{2}/\d{2}/\d{4})",
                    r"Check.?out[:\s]*(\d{1,2}/\d{1,2}/\d{4})",
                    r"Departure[:\s]*(\d{2}/\d{2}/\d{4})",
                    r"Departure[:\s]*(\d{1,2}/\d{1,2}/\d{4})",
                ],
                "transform": ["strip"],
            },
            {
                "field": "PAYMENT_METHOD",
                "line_filters": ["/credit|debit|eftpos|cash|visa|mastercard|amex/"],
                "patterns": [
                    r"(CREDIT|DEBIT|EFTPOS|CASH)",
                    r"(VISA|MASTERCARD|AMEX)",
                ],
                "transform": ["upper"],
            },
            {
                "field": "BOOKING",
                "line_filters": ["/booking|reservation|confirmation/"],
                "patterns": [
                    r"Booking[:\s]*([A-Z0-9]+)",
                    r"Reservation[:\s]*([A-Z0-9]+)",
                    r"Confirmation[:\s]*([A-Z0-9]+)",
                    r"(\d{6,})",  # Long booking numbers
                ],
                "transform": ["upper"],
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
                "field": "ADDRESS",
                "line_filters": [
                    "/street|road|ave|avenue|drive|lane|place|suburb|city/"
                ],
                "patterns": [
                    r"(\d+\s+.*(?:Street|Road|Ave|Avenue|Drive|Lane|Place))",
                    r"(.*(?:Street|Road|Ave|Avenue|Drive|Lane|Place).*)",
                    r"([A-Z][a-z]+\s+[A-Z][a-z]+.*\d{4})",  # Suburb State PostCode
                ],
                "transform": ["normalize_spaces"],
            },
        ]

    def extract_accommodation_fields(self, response: str) -> Dict[str, Any]:
        """Extract accommodation fields using AWK-style rules."""
        rules = self.get_accommodation_extraction_rules()
        extracted = self.extract_fields(response, rules)

        # Post-process specific to accommodation
        if "NIGHTS" in extracted:
            # Ensure nights is a number
            nights = extracted["NIGHTS"]
            if isinstance(nights, str) and nights.isdigit():
                extracted["NIGHTS"] = int(nights)

        self.logger.debug(
            f"AWK accommodation extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted


class TravelDocumentAwkExtractor(AwkExtractor):
    """Travel document specific AWK-style extractor."""

    def get_travel_extraction_rules(self) -> List[Dict[str, Any]]:
        """Get AWK-style extraction rules for travel documents."""
        return [
            {
                "field": "CARRIER",
                "line_filters": [
                    "/qantas|jetstar|virgin|tigerair|airline|airways|transport|bus|train|railway/"
                ],
                "patterns": [
                    r"(QANTAS.*)",
                    r"(JETSTAR.*)",
                    r"(VIRGIN.*)",
                    r"(TIGERAIR.*)",
                    r"(.*AIRLINES?.*)",
                    r"(.*AIRWAYS.*)",
                    r"(.*TRANSPORT.*)",
                    r"(.*BUS.*)",
                    r"(.*RAILWAY.*)",
                    r"(.*TRAIN.*)",
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
                "field": "TOTAL",
                "line_filters": ["/total|amount|fare|price/"],
                "patterns": [
                    r"TOTAL[^\d]*\$(\d+\.\d{2})",
                    r"\$(\d+\.\d{2})\s*TOTAL",
                    r"AMOUNT[^\d]*\$(\d+\.\d{2})",
                    r"FARE[^\d]*\$(\d+\.\d{2})",
                    r"PRICE[^\d]*\$(\d+\.\d{2})",
                ],
                "transform": ["prefix:$"],
            },
            {
                "field": "GST",
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
                "field": "PASSENGER",
                "line_filters": ["/passenger|name/"],
                "patterns": [
                    r"Passenger[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"Name[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"MR\s+([A-Z][a-zA-Z\s]+)",
                    r"MS\s+([A-Z][a-zA-Z\s]+)",
                    r"MRS\s+([A-Z][a-zA-Z\s]+)",
                ],
                "transform": ["upper", "normalize_spaces"],
            },
            {
                "field": "ORIGIN",
                "line_filters": [
                    "/from|depart|origin|sydney|melbourne|brisbane|perth|adelaide/"
                ],
                "patterns": [
                    r"From[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"Depart[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"Origin[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"(SYDNEY|MELBOURNE|BRISBANE|PERTH|ADELAIDE|DARWIN|CAIRNS|GOLD COAST|CANBERRA|HOBART)",
                    r"(SYD|MEL|BNE|PER|ADL|DRW|CNS|OOL|CBR|HBA)",  # Airport codes
                ],
                "transform": ["upper", "normalize_spaces"],
            },
            {
                "field": "DESTINATION",
                "line_filters": [
                    "/to|arrive|destination|sydney|melbourne|brisbane|perth|adelaide/"
                ],
                "patterns": [
                    r"To[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"Arrive[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"Destination[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"(SYDNEY|MELBOURNE|BRISBANE|PERTH|ADELAIDE|DARWIN|CAIRNS|GOLD COAST|CANBERRA|HOBART)",
                    r"(SYD|MEL|BNE|PER|ADL|DRW|CNS|OOL|CBR|HBA)",  # Airport codes
                ],
                "transform": ["upper", "normalize_spaces"],
            },
            {
                "field": "FLIGHT",
                "line_filters": ["/flight|qf|jq|va|tt/"],
                "patterns": [
                    r"Flight[:\s]*([A-Z]{2}\d+)",
                    r"(QF\d+)",  # Qantas
                    r"(JQ\d+)",  # Jetstar
                    r"(VA\d+)",  # Virgin
                    r"(TT\d+)",  # TigerAir
                ],
                "transform": ["upper"],
            },
            {
                "field": "SEAT",
                "line_filters": ["/seat\s+\d+[a-z]/"],
                "patterns": [
                    r"Seat[:\s]*(\d+[A-Z])",
                    r"SEAT[:\s]*(\d+[A-Z])",
                    r"(\d+[A-Z])",  # Direct seat format
                ],
                "transform": ["upper"],
            },
            {
                "field": "GATE",
                "line_filters": ["/gate\s+[a-z]?\d+/"],
                "patterns": [
                    r"Gate[:\s]*([A-Z]?\d+)",
                    r"GATE[:\s]*([A-Z]?\d+)",
                ],
                "transform": ["upper"],
            },
            {
                "field": "DEPARTURE",
                "line_filters": ["/departure|depart/"],
                "patterns": [
                    r"Departure[:\s]*(\d{1,2}:\d{2})",
                    r"Depart[:\s]*(\d{1,2}:\d{2})",
                    r"(\d{1,2}:\d{2})",  # Time format
                ],
                "transform": ["strip"],
            },
            {
                "field": "ARRIVAL",
                "line_filters": ["/arrival|arrive/"],
                "patterns": [
                    r"Arrival[:\s]*(\d{1,2}:\d{2})",
                    r"Arrive[:\s]*(\d{1,2}:\d{2})",
                    r"(\d{1,2}:\d{2})",  # Time format
                ],
                "transform": ["strip"],
            },
            {
                "field": "BOOKING",
                "line_filters": ["/booking|confirmation|reference/"],
                "patterns": [
                    r"Booking[:\s]*([A-Z0-9]+)",
                    r"Confirmation[:\s]*([A-Z0-9]+)",
                    r"Reference[:\s]*([A-Z0-9]+)",
                    r"([A-Z0-9]{6,})",  # Long booking codes
                ],
                "transform": ["upper"],
            },
            {
                "field": "TICKET",
                "line_filters": ["/ticket|e-ticket/"],
                "patterns": [
                    r"Ticket[:\s]*([A-Z0-9]+)",
                    r"E-ticket[:\s]*([A-Z0-9]+)",
                    r"(\d{10,})",  # Long ticket numbers
                ],
                "transform": ["upper"],
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
                "field": "PAYMENT_METHOD",
                "line_filters": ["/credit|debit|eftpos|cash|visa|mastercard|amex/"],
                "patterns": [
                    r"(CREDIT|DEBIT|EFTPOS|CASH)",
                    r"(VISA|MASTERCARD|AMEX)",
                ],
                "transform": ["upper"],
            },
        ]

    def extract_travel_fields(self, response: str) -> Dict[str, Any]:
        """Extract travel document fields using AWK-style rules."""
        rules = self.get_travel_extraction_rules()
        extracted = self.extract_fields(response, rules)

        # Post-process specific to travel documents
        if "ORIGIN" in extracted and "DESTINATION" in extracted:
            # Create route field
            extracted["ROUTE"] = f"{extracted['ORIGIN']} - {extracted['DESTINATION']}"

        self.logger.debug(
            f"AWK travel extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted


class ParkingTollAwkExtractor(AwkExtractor):
    """Parking toll specific AWK-style extractor."""

    def get_parking_toll_extraction_rules(self) -> List[Dict[str, Any]]:
        """Get AWK-style extraction rules for parking and toll documents."""
        return [
            {
                "field": "OPERATOR",
                "line_filters": [
                    "/wilson|secure|care|parking|citylink|eastlink|westlink|toll/"
                ],
                "patterns": [
                    r"(WILSON\s+PARKING)",
                    r"(SECURE\s+PARKING)",
                    r"(CARE\s+PARK)",
                    r"(.*PARKING.*)",
                    r"(CITYLINK)",
                    r"(EASTLINK)",
                    r"(WESTLINK)",
                    r"(.*TOLL.*)",
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
                "field": "TOTAL",
                "line_filters": ["/total|amount|charged|fee|cost/"],
                "patterns": [
                    r"TOTAL[^\d]*\$(\d+\.\d{2})",
                    r"\$(\d+\.\d{2})\s*TOTAL",
                    r"AMOUNT[^\d]*\$(\d+\.\d{2})",
                    r"CHARGED[^\d]*\$(\d+\.\d{2})",
                    r"FEE[^\d]*\$(\d+\.\d{2})",
                    r"COST[^\d]*\$(\d+\.\d{2})",
                ],
                "transform": ["prefix:$"],
            },
            {
                "field": "GST",
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
                "field": "VEHICLE",
                "line_filters": ["/vehicle|registration|rego|plate/"],
                "patterns": [
                    r"Vehicle[:\s]*([A-Z0-9]{3,8})",
                    r"Registration[:\s]*([A-Z0-9]{3,8})",
                    r"Rego[:\s]*([A-Z0-9]{3,8})",
                    r"Plate[:\s]*([A-Z0-9]{3,8})",
                    r"([A-Z]{3}\s?\d{3})",  # NSW format
                    r"(\d{3}\s?[A-Z]{3})",  # VIC format
                ],
                "transform": ["upper", "normalize_spaces"],
            },
            {
                "field": "LOCATION",
                "line_filters": ["/street|road|ave|level|zone|car park|parking/"],
                "patterns": [
                    r"(.*(?:Street|Road|Ave|Avenue|Drive|Lane|Place).*)",
                    r"([A-Z][a-z]+\s+Car\s+Park)",
                    r"([A-Z][a-z]+\s+Parking)",
                    r"(Level\s+\d+.*)",
                    r"(Zone\s+\d+.*)",
                    r"([A-Z][a-z]+\s+[A-Z][a-z]+.*\d{4})",  # Suburb State PostCode
                ],
                "transform": ["normalize_spaces"],
            },
            {
                "field": "ENTRY",
                "line_filters": ["/entry|enter|arrival|start/"],
                "patterns": [
                    r"Entry[:\s]*(\d{1,2}:\d{2})",
                    r"Enter[:\s]*(\d{1,2}:\d{2})",
                    r"Arrival[:\s]*(\d{1,2}:\d{2})",
                    r"Start[:\s]*(\d{1,2}:\d{2})",
                    r"(\d{1,2}:\d{2}.*AM|PM)",
                ],
                "transform": ["strip"],
            },
            {
                "field": "EXIT",
                "line_filters": ["/exit|leave|departure|end/"],
                "patterns": [
                    r"Exit[:\s]*(\d{1,2}:\d{2})",
                    r"Leave[:\s]*(\d{1,2}:\d{2})",
                    r"Departure[:\s]*(\d{1,2}:\d{2})",
                    r"End[:\s]*(\d{1,2}:\d{2})",
                    r"(\d{1,2}:\d{2}.*AM|PM)",
                ],
                "transform": ["strip"],
            },
            {
                "field": "DURATION",
                "line_filters": ["/duration|hours|minutes|time/"],
                "patterns": [
                    r"Duration[:\s]*(\d+[hm]?\s?\d*[hm]?)",
                    r"(\d+\s+hours?)",
                    r"(\d+\s+minutes?)",
                    r"(\d+[hm]\s?\d*[hm]?)",  # 2h 30m format
                ],
                "transform": ["normalize_spaces"],
            },
            {
                "field": "ZONE",
                "line_filters": ["/zone\s+\d+/"],
                "patterns": [
                    r"Zone[:\s]*(\d+)",
                    r"ZONE[:\s]*(\d+)",
                    r"Zone\s+([A-Z]\d*)",
                ],
                "transform": ["upper"],
            },
            {
                "field": "LEVEL",
                "line_filters": ["/level\s+\d+|floor\s+\d+/"],
                "patterns": [
                    r"Level[:\s]*(\d+)",
                    r"LEVEL[:\s]*(\d+)",
                    r"Floor[:\s]*(\d+)",
                    r"FLOOR[:\s]*(\d+)",
                ],
                "transform": ["upper"],
            },
            {
                "field": "SPACE",
                "line_filters": ["/space\s+\d+|bay\s+\d+/"],
                "patterns": [
                    r"Space[:\s]*(\d+)",
                    r"SPACE[:\s]*(\d+)",
                    r"Bay[:\s]*(\d+)",
                    r"BAY[:\s]*(\d+)",
                ],
                "transform": ["upper"],
            },
            {
                "field": "RATE",
                "line_filters": ["/rate|per hour|hourly|\$\d+\.\d{2}/"],
                "patterns": [
                    r"Rate[:\s]*\$(\d+\.\d{2})",
                    r"Per\s+Hour[:\s]*\$(\d+\.\d{2})",
                    r"Hourly[:\s]*\$(\d+\.\d{2})",
                    r"\$(\d+\.\d{2})\s*per\s+hour",
                ],
                "transform": ["prefix:$"],
            },
            {
                "field": "TICKET",
                "line_filters": ["/ticket|receipt|validation/"],
                "patterns": [
                    r"Ticket[:\s]*([A-Z0-9]+)",
                    r"Receipt[:\s]*([A-Z0-9]+)",
                    r"Validation[:\s]*([A-Z0-9]+)",
                    r"(\d{6,})",  # Long ticket numbers
                ],
                "transform": ["upper"],
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
                "field": "PAYMENT_METHOD",
                "line_filters": [
                    "/credit|debit|eftpos|cash|visa|mastercard|amex|contactless/"
                ],
                "patterns": [
                    r"(CREDIT|DEBIT|EFTPOS|CASH)",
                    r"(VISA|MASTERCARD|AMEX)",
                    r"(CONTACTLESS|TAP)",
                ],
                "transform": ["upper"],
            },
        ]

    def extract_parking_toll_fields(self, response: str) -> Dict[str, Any]:
        """Extract parking toll fields using AWK-style rules."""
        rules = self.get_parking_toll_extraction_rules()
        extracted = self.extract_fields(response, rules)

        # Post-process specific to parking/toll documents
        if "DURATION" in extracted:
            # Normalize duration format
            duration = extracted["DURATION"]
            if isinstance(duration, str):
                # Convert various formats to standard
                duration = duration.replace("hours", "h").replace("minutes", "m")
                duration = duration.replace("hour", "h").replace("minute", "m")
                extracted["DURATION"] = duration.strip()

        if "VEHICLE" in extracted:
            # Clean up vehicle registration
            vehicle = extracted["VEHICLE"]
            if isinstance(vehicle, str):
                # Remove common prefixes
                vehicle = vehicle.replace("Registration:", "").replace("Rego:", "")
                extracted["VEHICLE"] = vehicle.strip()

        self.logger.debug(
            f"AWK parking toll extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted


class EquipmentSuppliesAwkExtractor(AwkExtractor):
    """Equipment supplies specific AWK-style extractor."""

    def get_equipment_supplies_extraction_rules(self) -> List[Dict[str, Any]]:
        """Get AWK-style extraction rules for equipment and supplies documents."""
        return [
            {
                "field": "SUPPLIER",
                "line_filters": [
                    "/officeworks|harvey norman|jb hi-fi|centrecom|mwave|pccasegear|scorptec|umart|apple|microsoft|dell|hp|lenovo/"
                ],
                "patterns": [
                    r"(OFFICEWORKS)",
                    r"(HARVEY\s+NORMAN)",
                    r"(JB\s+HI-FI)",
                    r"(CENTRECOM)",
                    r"(MWAVE)",
                    r"(PCCASEGEAR)",
                    r"(SCORPTEC)",
                    r"(UMART)",
                    r"(APPLE)",
                    r"(MICROSOFT)",
                    r"(DELL)",
                    r"(HP)",
                    r"(LENOVO)",
                    r"(ASUS)",
                    r"(ACER)",
                    r"(CANON)",
                    r"(EPSON)",
                    r"(BROTHER)",
                    r"(XEROX)",
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
                "field": "TOTAL",
                "line_filters": ["/total|amount|balance|subtotal/"],
                "patterns": [
                    r"TOTAL[^\d]*\$(\d+\.\d{2})",
                    r"\$(\d+\.\d{2})\s*TOTAL",
                    r"AMOUNT[^\d]*\$(\d+\.\d{2})",
                    r"BALANCE[^\d]*\$(\d+\.\d{2})",
                    r"SUBTOTAL[^\d]*\$(\d+\.\d{2})",
                ],
                "transform": ["prefix:$"],
            },
            {
                "field": "GST",
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
                "field": "ITEMS",
                "line_filters": [
                    "/computer|laptop|software|hardware|printer|monitor|keyboard|mouse|cable|adapter/"
                ],
                "patterns": [
                    r"(.*COMPUTER.*)",
                    r"(.*LAPTOP.*)",
                    r"(.*SOFTWARE.*)",
                    r"(.*HARDWARE.*)",
                    r"(.*PRINTER.*)",
                    r"(.*MONITOR.*)",
                    r"(.*KEYBOARD.*)",
                    r"(.*MOUSE.*)",
                    r"(.*CABLE.*)",
                    r"(.*ADAPTER.*)",
                    r"(.*CHARGER.*)",
                    r"(.*BATTERY.*)",
                    r"(.*MEMORY.*)",
                    r"(.*STORAGE.*)",
                    r"(.*WEBCAM.*)",
                    r"(.*HEADSET.*)",
                    r"(.*SPEAKER.*)",
                ],
                "transform": ["normalize_spaces"],
            },
            {
                "field": "CATEGORIES",
                "line_filters": [
                    "/electronics|computers|software|hardware|accessories|supplies/"
                ],
                "patterns": [
                    r"(ELECTRONICS)",
                    r"(COMPUTERS)",
                    r"(SOFTWARE)",
                    r"(HARDWARE)",
                    r"(ACCESSORIES)",
                    r"(SUPPLIES)",
                    r"(OFFICE\s+SUPPLIES)",
                    r"(STATIONERY)",
                ],
                "transform": ["upper", "normalize_spaces"],
            },
            {
                "field": "BRANDS",
                "line_filters": [
                    "/apple|microsoft|dell|hp|lenovo|asus|acer|canon|epson|brother|samsung/"
                ],
                "patterns": [
                    r"(APPLE)",
                    r"(MICROSOFT)",
                    r"(DELL)",
                    r"(HP)",
                    r"(LENOVO)",
                    r"(ASUS)",
                    r"(ACER)",
                    r"(CANON)",
                    r"(EPSON)",
                    r"(BROTHER)",
                    r"(SAMSUNG)",
                    r"(SONY)",
                    r"(LG)",
                ],
                "transform": ["upper"],
            },
            {
                "field": "MODELS",
                "line_filters": ["/model|part|series/"],
                "patterns": [
                    r"Model[:\s]*([A-Z0-9\-]+)",
                    r"Part[:\s]*([A-Z0-9\-]+)",
                    r"Series[:\s]*([A-Z0-9\-]+)",
                    r"([A-Z0-9\-]{5,})",  # Model number format
                ],
                "transform": ["upper"],
            },
            {
                "field": "SKU",
                "line_filters": ["/sku|product code|item code/"],
                "patterns": [
                    r"SKU[:\s]*([A-Z0-9\-]+)",
                    r"Product\s+Code[:\s]*([A-Z0-9\-]+)",
                    r"Item\s+Code[:\s]*([A-Z0-9\-]+)",
                    r"([A-Z0-9\-]{6,})",  # SKU format
                ],
                "transform": ["upper"],
            },
            {
                "field": "QUANTITIES",
                "line_filters": ["/qty|quantity|\d+\s+x\s+/"],
                "patterns": [
                    r"Qty[:\s]*(\d+)",
                    r"Quantity[:\s]*(\d+)",
                    r"(\d+)\s+x\s+",
                    r"(\d+)\s+units?",
                ],
                "transform": ["strip"],
            },
            {
                "field": "PRICES",
                "line_filters": ["/\$\d+\.\d{2}|price|cost/"],
                "patterns": [
                    r"Price[:\s]*\$(\d+\.\d{2})",
                    r"Cost[:\s]*\$(\d+\.\d{2})",
                    r"\$(\d+\.\d{2})\s*each",
                    r"\$(\d+\.\d{2})",
                ],
                "transform": ["prefix:$"],
            },
            {
                "field": "WARRANTY",
                "line_filters": ["/warranty|guarantee|\d+\s+years?|\d+\s+months?/"],
                "patterns": [
                    r"Warranty[:\s]*(\d+\s+years?)",
                    r"Warranty[:\s]*(\d+\s+months?)",
                    r"Guarantee[:\s]*(\d+\s+years?)",
                    r"(\d+\s+year\s+warranty)",
                    r"(\d+\s+month\s+warranty)",
                ],
                "transform": ["normalize_spaces"],
            },
            {
                "field": "INVOICE",
                "line_filters": ["/invoice|inv\s+\d+/"],
                "patterns": [
                    r"Invoice[:\s]*([A-Z0-9\-]+)",
                    r"INV[:\s]*([A-Z0-9\-]+)",
                    r"Invoice\s+Number[:\s]*([A-Z0-9\-]+)",
                    r"(\d{6,})",  # Invoice number format
                ],
                "transform": ["upper"],
            },
            {
                "field": "ORDER",
                "line_filters": ["/order|po\s+\d+/"],
                "patterns": [
                    r"Order[:\s]*([A-Z0-9\-]+)",
                    r"PO[:\s]*([A-Z0-9\-]+)",
                    r"Order\s+Number[:\s]*([A-Z0-9\-]+)",
                    r"Purchase\s+Order[:\s]*([A-Z0-9\-]+)",
                ],
                "transform": ["upper"],
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
                "field": "PAYMENT_METHOD",
                "line_filters": [
                    "/credit|debit|eftpos|cash|visa|mastercard|amex|paypal/"
                ],
                "patterns": [
                    r"(CREDIT|DEBIT|EFTPOS|CASH)",
                    r"(VISA|MASTERCARD|AMEX)",
                    r"(PAYPAL|APPLE\s+PAY|GOOGLE\s+PAY)",
                ],
                "transform": ["upper"],
            },
            {
                "field": "ADDRESS",
                "line_filters": [
                    "/street|road|ave|avenue|drive|lane|place|suburb|city/"
                ],
                "patterns": [
                    r"(\d+\s+.*(?:Street|Road|Ave|Avenue|Drive|Lane|Place))",
                    r"(.*(?:Street|Road|Ave|Avenue|Drive|Lane|Place).*)",
                    r"([A-Z][a-z]+\s+[A-Z][a-z]+.*\d{4})",  # Suburb State PostCode
                ],
                "transform": ["normalize_spaces"],
            },
        ]

    def extract_equipment_supplies_fields(self, response: str) -> Dict[str, Any]:
        """Extract equipment supplies fields using AWK-style rules."""
        rules = self.get_equipment_supplies_extraction_rules()
        extracted = self.extract_fields(response, rules)

        # Post-process specific to equipment/supplies
        if "ITEMS" in extracted:
            # Clean up item descriptions
            items = extracted["ITEMS"]
            if isinstance(items, str):
                # Remove common prefixes and suffixes
                items = items.replace("Item:", "").replace("Product:", "")
                extracted["ITEMS"] = items.strip()

        if "QUANTITIES" in extracted:
            # Ensure quantities is a number
            qty = extracted["QUANTITIES"]
            if isinstance(qty, str) and qty.isdigit():
                extracted["QUANTITIES"] = int(qty)

        self.logger.debug(
            f"AWK equipment supplies extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted


class OtherDocumentAwkExtractor(AwkExtractor):
    """Other document specific AWK-style extractor for miscellaneous business documents."""

    def get_other_document_extraction_rules(self) -> List[Dict[str, Any]]:
        """Get AWK-style extraction rules for other business documents."""
        return [
            {
                "field": "BUSINESS",
                "line_filters": [
                    "/pty ltd|proprietary|limited|company|business|corp|corporation|enterprise|group|services|solutions/"
                ],
                "patterns": [
                    r"(.*PTY\s+LTD.*)",
                    r"(.*PROPRIETARY.*)",
                    r"(.*LIMITED.*)",
                    r"(.*COMPANY.*)",
                    r"(.*BUSINESS.*)",
                    r"(.*CORP.*)",
                    r"(.*CORPORATION.*)",
                    r"(.*ENTERPRISE.*)",
                    r"(.*GROUP.*)",
                    r"(.*SERVICES.*)",
                    r"(.*SOLUTIONS.*)",
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
                "field": "TOTAL",
                "line_filters": ["/total|amount|balance|due|payable|outstanding/"],
                "patterns": [
                    r"TOTAL[^\d]*\$(\d+\.\d{2})",
                    r"\$(\d+\.\d{2})\s*TOTAL",
                    r"AMOUNT[^\d]*\$(\d+\.\d{2})",
                    r"BALANCE[^\d]*\$(\d+\.\d{2})",
                    r"DUE[^\d]*\$(\d+\.\d{2})",
                    r"PAYABLE[^\d]*\$(\d+\.\d{2})",
                    r"OUTSTANDING[^\d]*\$(\d+\.\d{2})",
                ],
                "transform": ["prefix:$"],
            },
            {
                "field": "GST",
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
                "field": "DESCRIPTION",
                "line_filters": [
                    "/description|service|product|item|goods|consultation/"
                ],
                "patterns": [
                    r"Description[:\s]*([^\n\r]+)",
                    r"Service[:\s]*([^\n\r]+)",
                    r"Product[:\s]*([^\n\r]+)",
                    r"Item[:\s]*([^\n\r]+)",
                    r"Goods[:\s]*([^\n\r]+)",
                    r"Consultation[:\s]*([^\n\r]+)",
                ],
                "transform": ["normalize_spaces"],
            },
            {
                "field": "REFERENCE",
                "line_filters": ["/reference|ref|number|invoice|receipt|document/"],
                "patterns": [
                    r"Reference[:\s]*([A-Z0-9\-]+)",
                    r"Ref[:\s]*([A-Z0-9\-]+)",
                    r"Number[:\s]*([A-Z0-9\-]+)",
                    r"Invoice[:\s]*([A-Z0-9\-]+)",
                    r"Receipt[:\s]*([A-Z0-9\-]+)",
                    r"Document[:\s]*([A-Z0-9\-]+)",
                    r"([A-Z0-9\-]{6,})",  # Generic reference format
                ],
                "transform": ["upper"],
            },
            {
                "field": "CUSTOMER",
                "line_filters": ["/customer|client|to:|bill to|ship to/"],
                "patterns": [
                    r"Customer[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"Client[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"To[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"Bill\s+To[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"Ship\s+To[:\s]*([A-Z][a-zA-Z\s]+)",
                ],
                "transform": ["upper", "normalize_spaces"],
            },
            {
                "field": "CONTACT",
                "line_filters": ["/contact|attention|attn/"],
                "patterns": [
                    r"Contact[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"Attention[:\s]*([A-Z][a-zA-Z\s]+)",
                    r"Attn[:\s]*([A-Z][a-zA-Z\s]+)",
                ],
                "transform": ["upper", "normalize_spaces"],
            },
            {
                "field": "PHONE",
                "line_filters": ["/phone|tel|telephone|mobile|mob/"],
                "patterns": [
                    r"Phone[:\s]*([0-9\s\-\+\(\)]+)",
                    r"Tel[:\s]*([0-9\s\-\+\(\)]+)",
                    r"Telephone[:\s]*([0-9\s\-\+\(\)]+)",
                    r"Mobile[:\s]*([0-9\s\-\+\(\)]+)",
                    r"Mob[:\s]*([0-9\s\-\+\(\)]+)",
                    r"(\+61\s?[0-9\s\-\(\)]+)",  # Australian format
                    r"(\([0-9]{2}\)\s?[0-9\s\-]+)",  # (02) format
                ],
                "transform": ["normalize_spaces"],
            },
            {
                "field": "EMAIL",
                "line_filters": ["/email|e-mail|@/"],
                "patterns": [
                    r"Email[:\s]*([a-zA-Z0-9\.\-_]+@[a-zA-Z0-9\.\-_]+)",
                    r"E-mail[:\s]*([a-zA-Z0-9\.\-_]+@[a-zA-Z0-9\.\-_]+)",
                    r"([a-zA-Z0-9\.\-_]+@[a-zA-Z0-9\.\-_]+)",
                ],
                "transform": ["lower"],
            },
            {
                "field": "WEBSITE",
                "line_filters": ["/website|web|www|http|https/"],
                "patterns": [
                    r"Website[:\s]*(www\.[a-zA-Z0-9\.\-_]+)",
                    r"Web[:\s]*(www\.[a-zA-Z0-9\.\-_]+)",
                    r"(www\.[a-zA-Z0-9\.\-_]+)",
                    r"(https?://[a-zA-Z0-9\.\-_/]+)",
                ],
                "transform": ["lower"],
            },
            {
                "field": "DOCUMENT_TYPE",
                "line_filters": [
                    "/invoice|receipt|quote|estimate|proposal|contract|agreement/"
                ],
                "patterns": [
                    r"(INVOICE)",
                    r"(RECEIPT)",
                    r"(QUOTE)",
                    r"(QUOTATION)",
                    r"(ESTIMATE)",
                    r"(PROPOSAL)",
                    r"(CONTRACT)",
                    r"(AGREEMENT)",
                    r"(STATEMENT)",
                    r"(BILL)",
                    r"(NOTICE)",
                    r"(CERTIFICATE)",
                ],
                "transform": ["upper"],
            },
            {
                "field": "DOCUMENT_NUMBER",
                "line_filters": ["/invoice|receipt|quote|document|number/"],
                "patterns": [
                    r"Invoice\s+Number[:\s]*([A-Z0-9\-]+)",
                    r"Receipt\s+Number[:\s]*([A-Z0-9\-]+)",
                    r"Quote\s+Number[:\s]*([A-Z0-9\-]+)",
                    r"Document\s+Number[:\s]*([A-Z0-9\-]+)",
                    r"Number[:\s]*([A-Z0-9\-]+)",
                    r"([A-Z0-9\-]{6,})",  # Generic number format
                ],
                "transform": ["upper"],
            },
            {
                "field": "DATE",
                "line_filters": ["NF > 2"],  # Lines with multiple fields
                "patterns": [
                    r"(\d{2}/\d{2}/\d{4})",
                    r"(\d{1,2}/\d{1,2}/\d{4})",
                    r"(\d{4}-\d{2}-\d{2})",
                    r"(\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
                ],
                "transform": ["strip"],
            },
            {
                "field": "PAYMENT_METHOD",
                "line_filters": [
                    "/credit|debit|eftpos|cash|visa|mastercard|amex|paypal|cheque|bank transfer/"
                ],
                "patterns": [
                    r"(CREDIT|DEBIT|EFTPOS|CASH)",
                    r"(VISA|MASTERCARD|AMEX)",
                    r"(PAYPAL|APPLE\s+PAY|GOOGLE\s+PAY)",
                    r"(CHEQUE|CHECK)",
                    r"(BANK\s+TRANSFER)",
                    r"(DIRECT\s+DEPOSIT)",
                ],
                "transform": ["upper"],
            },
            {
                "field": "ADDRESS",
                "line_filters": [
                    "/street|road|ave|avenue|drive|lane|place|suburb|city|state|postcode/"
                ],
                "patterns": [
                    r"(\d+\s+.*(?:Street|Road|Ave|Avenue|Drive|Lane|Place))",
                    r"(.*(?:Street|Road|Ave|Avenue|Drive|Lane|Place).*)",
                    r"([A-Z][a-z]+\s+[A-Z][a-z]+.*\d{4})",  # Suburb State PostCode
                    r"(PO\s+Box\s+\d+)",  # PO Box
                ],
                "transform": ["normalize_spaces"],
            },
        ]

    def extract_other_document_fields(self, response: str) -> Dict[str, Any]:
        """Extract other document fields using AWK-style rules."""
        rules = self.get_other_document_extraction_rules()
        extracted = self.extract_fields(response, rules)

        # Post-process specific to other documents
        if "PHONE" in extracted:
            # Clean up phone number
            phone = extracted["PHONE"]
            if isinstance(phone, str):
                # Remove common prefixes and normalize
                phone = phone.replace("Phone:", "").replace("Tel:", "")
                phone = phone.replace("Mobile:", "").replace("Mob:", "")
                extracted["PHONE"] = phone.strip()

        if "EMAIL" in extracted:
            # Clean up email
            email = extracted["EMAIL"]
            if isinstance(email, str):
                # Remove common prefixes
                email = email.replace("Email:", "").replace("E-mail:", "")
                extracted["EMAIL"] = email.strip().lower()

        if "WEBSITE" in extracted:
            # Clean up website
            website = extracted["WEBSITE"]
            if isinstance(website, str):
                # Remove common prefixes
                website = website.replace("Website:", "").replace("Web:", "")
                extracted["WEBSITE"] = website.strip().lower()

        self.logger.debug(
            f"AWK other document extraction found {len(extracted)} fields: {list(extracted.keys())}"
        )
        return extracted
