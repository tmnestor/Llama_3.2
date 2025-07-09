"""
Business Receipt Handler for Australian Tax Document Processing

This handler specializes in processing business receipts for general expense claims
with ATO compliance validation.
"""

import re
from typing import Any, Dict, List

from ..extraction.australian_tax_classifier import DocumentType
from ..utils import setup_logging
from .base_ato_handler import BaseATOHandler

logger = setup_logging()


class BusinessReceiptHandler(BaseATOHandler):
    """Handler for Australian business receipt processing with ATO compliance."""

    def __init__(self):
        super().__init__(DocumentType.BUSINESS_RECEIPT)
        logger.info(
            "BusinessReceiptHandler initialized for general business expense claims"
        )

    def _extract_fields_primary(self, document_text: str) -> Dict[str, Any]:
        """Extract fields using primary extraction method for business receipts."""
        extracted_fields = {}

        # Extract store name
        store_patterns = [
            r"(WOOLWORTHS|COLES|ALDI|TARGET|KMART|BUNNINGS|OFFICEWORKS|HARVEY NORMAN|JB HI-FI|BIG W)",
            r"([A-Z][A-Z\s&]+(?:PTY\s+LTD|LIMITED|COMPANY|SUPERMARKET|STORE))",
            r"([A-Z\s]+(?:SUPERMARKET|STORE|SHOP|RETAIL))",
        ]

        for pattern in store_patterns:
            match = re.search(pattern, document_text, re.MULTILINE)
            if match:
                extracted_fields["store_name"] = match.group(1).strip()
                break

        # Extract ABN
        abn_patterns = [
            r"abn[\s:]*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
            r"australian business number[\s:]*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
            r"(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
        ]

        for pattern in abn_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["abn"] = match.group(1).strip()
                break

        # Extract date
        date_patterns = [
            r"(\d{1,2}/\d{1,2}/\d{4})",
            r"(\d{1,2}-\d{1,2}-\d{4})",
            r"(\d{4}-\d{1,2}-\d{1,2})",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, document_text)
            if match:
                extracted_fields["date"] = match.group(1)
                break

        # Extract GST amount
        gst_patterns = [
            r"gst[\s:]*\$?(\d+\.\d{2})",
            r"(?:goods\s+services\s+tax|g\.s\.t\.)[\s:]*\$?(\d+\.\d{2})",
            r"tax[\s:]*\$?(\d+\.\d{2})",
        ]

        for pattern in gst_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["gst_amount"] = match.group(1)
                break

        # Extract total amount
        total_patterns = [
            r"total[\s:]*\$?(\d+\.\d{2})",
            r"(?:total amount|amount)[\s:]*\$?(\d+\.\d{2})",
            r"pay[\s:]*\$?(\d+\.\d{2})",
        ]

        for pattern in total_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["total_amount"] = match.group(1)
                break

        # Extract subtotal
        subtotal_patterns = [
            r"subtotal[\s:]*\$?(\d+\.\d{2})",
            r"(?:sub total|net)[\s:]*\$?(\d+\.\d{2})",
        ]

        for pattern in subtotal_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["subtotal"] = match.group(1)
                break

        # Extract items - this is more complex for receipts
        items_list = []
        quantities_list = []
        prices_list = []

        # Pattern to match item lines (item name, quantity, price)
        item_patterns = [
            r"([A-Za-z\s]+\w)\s+(\d+)\s+\$?(\d+\.\d{2})",
            r"([A-Za-z\s]+\w)\s+\$?(\d+\.\d{2})",
        ]

        lines = document_text.split("\n")
        for line in lines:
            line = line.strip()

            # Skip header lines and totals
            if any(
                word in line.upper()
                for word in [
                    "TOTAL",
                    "SUBTOTAL",
                    "GST",
                    "RECEIPT",
                    "STORE",
                    "ABN",
                    "DATE",
                ]
            ):
                continue

            # Try to match item patterns
            for pattern in item_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 3:  # item, quantity, price
                        item_name = match.group(1).strip().title()
                        quantity = match.group(2)
                        price = match.group(3)
                    else:  # item, price (quantity assumed 1)
                        item_name = match.group(1).strip().title()
                        quantity = "1"
                        price = match.group(2)

                    items_list.append(item_name)
                    quantities_list.append(quantity)
                    prices_list.append(price)
                    break

        # Add items to extracted fields
        if items_list:
            extracted_fields["items"] = " | ".join(items_list)
            extracted_fields["quantities"] = " | ".join(quantities_list)
            extracted_fields["prices"] = " | ".join(prices_list)

        # Extract payer/customer name
        payer_patterns = [
            r"(?:customer|member)[\s:]*([A-Za-z\s]+)",
            r"(?:name|cardholder)[\s:]*([A-Za-z\s]+)",
        ]

        for pattern in payer_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["payer"] = match.group(1).strip()
                break

        # Extract payment method
        payment_patterns = [
            r"(cash|card|eftpos|credit|debit|visa|mastercard|amex)",
            r"payment[\s:]*([A-Za-z\s]+)",
        ]

        for pattern in payment_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["payment_method"] = match.group(1).strip().upper()
                break

        # Extract receipt number
        receipt_patterns = [
            r"receipt[\s#:]*([A-Za-z0-9\-]+)",
            r"transaction[\s#:]*([A-Za-z0-9\-]+)",
            r"ref[\s#:]*([A-Za-z0-9\-]+)",
        ]

        for pattern in receipt_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["receipt_number"] = match.group(1).strip()
                break

        logger.debug(
            f"Primary extraction yielded {len(extracted_fields)} fields for business receipt"
        )
        return extracted_fields

    def _get_required_fields(self) -> List[str]:
        """Get required fields for business receipt processing."""
        return ["date", "store_name", "total_amount"]

    def _get_optional_fields(self) -> List[str]:
        """Get optional fields for business receipt processing."""
        return [
            "abn",
            "gst_amount",
            "subtotal",
            "items",
            "quantities",
            "prices",
            "payer",
            "payment_method",
            "receipt_number",
        ]

    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for business receipt fields."""
        return {
            "date": self._validate_australian_date,
            "abn": self._validate_abn_format,
            "total_amount": self._validate_currency_amount,
            "gst_amount": self._validate_currency_amount,
            "subtotal": self._validate_currency_amount,
            "store_name": self._validate_store_name,
            "items": self._validate_items_list,
            "quantities": self._validate_quantities_list,
            "prices": self._validate_prices_list,
            "receipt_number": self._validate_receipt_number,
        }

    def _get_ato_thresholds(self) -> Dict[str, Any]:
        """Get ATO-specific thresholds for business receipts."""
        return {
            "receipt_required_threshold": 82.50,
            "abn_required_threshold": 82.50,
            "gst_rate": 0.10,
            "gst_tolerance": 0.02,
            "maximum_cash_claim": 300.00,
        }

    def _get_awk_rules(self) -> List[Dict[str, Any]]:
        """Get AWK extraction rules for business receipts."""
        return [
            {
                "field": "store_name",
                "pattern": r"(WOOLWORTHS|COLES|ALDI|TARGET|KMART|BUNNINGS|OFFICEWORKS)",
                "line_filter": lambda line: any(
                    store in line.upper()
                    for store in [
                        "WOOLWORTHS",
                        "COLES",
                        "ALDI",
                        "TARGET",
                        "KMART",
                        "BUNNINGS",
                    ]
                ),
                "transform": lambda x: x.strip().upper(),
            },
            {
                "field": "abn",
                "pattern": r"(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
                "line_filter": lambda line: "abn" in line.lower()
                or "australian business number" in line.lower(),
                "transform": lambda x: x.strip(),
            },
            {
                "field": "date",
                "pattern": r"\d{1,2}/\d{1,2}/\d{4}",
                "line_filter": lambda line: any(
                    word in line.lower() for word in ["date", "dated", "day"]
                ),
                "transform": lambda x: x.strip(),
            },
            {
                "field": "gst_amount",
                "pattern": r"\$?(\d+\.\d{2})",
                "line_filter": lambda line: "gst" in line.lower()
                or "tax" in line.lower(),
                "transform": lambda x: x.replace("$", "").strip(),
            },
            {
                "field": "total_amount",
                "pattern": r"\$?(\d+\.\d{2})",
                "line_filter": lambda line: any(
                    word in line.lower() for word in ["total", "amount", "pay"]
                ),
                "transform": lambda x: x.replace("$", "").strip(),
            },
            {
                "field": "subtotal",
                "pattern": r"\$?(\d+\.\d{2})",
                "line_filter": lambda line: any(
                    word in line.lower() for word in ["subtotal", "sub total", "net"]
                ),
                "transform": lambda x: x.replace("$", "").strip(),
            },
            {
                "field": "items",
                "pattern": r"([A-Za-z\s]+\w)",
                "line_filter": lambda line: self._is_item_line(line),
                "transform": lambda x: x.strip().title(),
            },
        ]

    def _is_item_line(self, line: str) -> bool:
        """Check if line contains item information."""
        line_lower = line.lower()

        # Skip header/footer lines
        skip_terms = [
            "total",
            "subtotal",
            "gst",
            "tax",
            "receipt",
            "store",
            "abn",
            "date",
            "thank you",
        ]
        if any(term in line_lower for term in skip_terms):
            return False

        # Check if line has item characteristics
        item_indicators = [
            re.search(
                r"[A-Za-z\s]+\w\s+\d+\s+\$?\d+\.\d{2}", line
            ),  # item quantity price
            re.search(r"[A-Za-z\s]+\w\s+\$?\d+\.\d{2}", line),  # item price
            any(
                food in line_lower
                for food in ["bread", "milk", "egg", "meat", "vegetable", "fruit"]
            ),
            any(
                item in line_lower
                for item in ["paper", "pen", "book", "tool", "equipment"]
            ),
        ]

        return any(item_indicators)

    def _validate_australian_date(self, date_str: str) -> str:
        """Validate Australian date format."""
        if not date_str:
            return date_str

        # Convert to DD/MM/YYYY format
        date_patterns = [
            (r"(\d{1,2})/(\d{1,2})/(\d{4})", r"\1/\2/\3"),
            (r"(\d{1,2})-(\d{1,2})-(\d{4})", r"\1/\2/\3"),
            (r"(\d{4})-(\d{1,2})-(\d{1,2})", r"\2/\3/\1"),
        ]

        for pattern, replacement in date_patterns:
            match = re.match(pattern, date_str.strip())
            if match:
                return re.sub(pattern, replacement, date_str.strip())

        return date_str

    def _validate_abn_format(self, abn_str: str) -> str:
        """Validate ABN format."""
        if not abn_str:
            return abn_str

        # Remove non-digits and format as XX XXX XXX XXX
        clean_abn = re.sub(r"[^\d]", "", abn_str)

        if len(clean_abn) == 11:
            return (
                f"{clean_abn[:2]} {clean_abn[2:5]} {clean_abn[5:8]} {clean_abn[8:11]}"
            )
        else:
            logger.warning(f"Invalid ABN format: {abn_str}")
            return abn_str

    def _validate_currency_amount(self, amount_str: str) -> str:
        """Validate currency amount format."""
        if not amount_str:
            return amount_str

        # Remove currency symbols and validate
        clean_amount = re.sub(r"[^\d.]", "", amount_str)

        try:
            amount = float(clean_amount)
            if 0.01 <= amount <= 10000.0:  # Reasonable receipt range
                return f"{amount:.2f}"
            else:
                logger.warning(f"Receipt amount {amount} outside reasonable range")
                return clean_amount
        except ValueError:
            logger.warning(f"Invalid currency amount format: {amount_str}")
            return amount_str

    def _validate_store_name(self, store_str: str) -> str:
        """Validate store name format."""
        if not store_str:
            return store_str

        # Clean and format store name
        clean_store = store_str.strip().upper()

        # Known Australian retailers
        known_stores = [
            "WOOLWORTHS",
            "COLES",
            "ALDI",
            "TARGET",
            "KMART",
            "BUNNINGS",
            "OFFICEWORKS",
            "HARVEY NORMAN",
            "JB HI-FI",
            "BIG W",
            "MYER",
        ]

        for store in known_stores:
            if store in clean_store:
                return store

        return clean_store

    def _validate_items_list(self, items_str: str) -> str:
        """Validate items list format."""
        if not items_str:
            return items_str

        # Clean and format items
        items = [item.strip().title() for item in items_str.split("|")]

        # Filter out empty items
        valid_items = [item for item in items if item and len(item) > 1]

        return " | ".join(valid_items)

    def _validate_quantities_list(self, quantities_str: str) -> str:
        """Validate quantities list format."""
        if not quantities_str:
            return quantities_str

        # Clean and validate quantities
        quantities = [qty.strip() for qty in quantities_str.split("|")]

        valid_quantities = []
        for qty in quantities:
            try:
                qty_num = int(qty)
                if 1 <= qty_num <= 100:  # Reasonable quantity range
                    valid_quantities.append(str(qty_num))
                else:
                    valid_quantities.append(qty)
            except ValueError:
                valid_quantities.append(qty)

        return " | ".join(valid_quantities)

    def _validate_prices_list(self, prices_str: str) -> str:
        """Validate prices list format."""
        if not prices_str:
            return prices_str

        # Clean and validate prices
        prices = [price.strip() for price in prices_str.split("|")]

        valid_prices = []
        for price in prices:
            clean_price = re.sub(r"[^\d.]", "", price)
            try:
                price_num = float(clean_price)
                if 0.01 <= price_num <= 1000.0:  # Reasonable price range
                    valid_prices.append(f"{price_num:.2f}")
                else:
                    valid_prices.append(clean_price)
            except ValueError:
                valid_prices.append(price)

        return " | ".join(valid_prices)

    def _validate_receipt_number(self, receipt_str: str) -> str:
        """Validate receipt number format."""
        if not receipt_str:
            return receipt_str

        # Clean receipt number
        clean_receipt = re.sub(r"[^\w\-]", "", receipt_str)

        if len(clean_receipt) >= 3:
            return clean_receipt.upper()
        else:
            logger.warning(f"Invalid receipt number format: {receipt_str}")
            return receipt_str
