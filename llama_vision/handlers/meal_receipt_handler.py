"""
Meal Receipt Handler for Australian Tax Document Processing

This handler specializes in processing meal receipts for business entertainment claims
with ATO compliance validation.
"""

import re
from typing import Any, Dict, List

from ..extraction.australian_tax_classifier import DocumentType
from ..utils import setup_logging
from .base_ato_handler import BaseATOHandler

logger = setup_logging()


class MealReceiptHandler(BaseATOHandler):
    """Handler for Australian meal receipt processing with ATO compliance."""

    def __init__(self):
        super().__init__(DocumentType.MEAL_RECEIPT)
        logger.info("MealReceiptHandler initialized for business entertainment claims")

    def _extract_fields_primary(self, document_text: str) -> Dict[str, Any]:
        """Extract fields using primary extraction method for meal receipts."""
        extracted_fields = {}

        # Extract restaurant name
        restaurant_patterns = [
            r"([A-Z][A-Za-z\s&]+(?:RESTAURANT|CAFE|BAR|BISTRO|HOTEL))",
            r"(MCDONALD\'S|KFC|SUBWAY|DOMINO\'S|PIZZA HUT|HUNGRY JACK\'S)",
            r"(STARBUCKS|GLORIA JEAN\'S|COFFEE CLUB)",
        ]

        for pattern in restaurant_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["restaurant_name"] = match.group(1).strip()
                break

        # Extract date and time
        date_patterns = [r"(\d{1,2}/\d{1,2}/\d{4})", r"(\d{1,2}-\d{1,2}-\d{4})"]

        for pattern in date_patterns:
            match = re.search(pattern, document_text)
            if match:
                extracted_fields["date"] = match.group(1)
                break

        time_patterns = [r"(\d{1,2}:\d{2}(?::\d{2})?)", r"time[\s:]*(\d{1,2}:\d{2})"]

        for pattern in time_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["time"] = match.group(1)
                break

        # Extract meal type
        meal_type_patterns = [
            r"(breakfast|lunch|dinner|brunch|supper)",
            r"(coffee|beverage|drink)",
            r"(snack|appetizer|dessert)",
        ]

        for pattern in meal_type_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["meal_type"] = match.group(1).title()
                break

        # Extract covers (number of people)
        covers_patterns = [
            r"covers?[\s:]*(\d+)",
            r"persons?[\s:]*(\d+)",
            r"guests?[\s:]*(\d+)",
        ]

        for pattern in covers_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["covers"] = match.group(1)
                break

        # Extract items and prices
        items_list = []
        prices_list = []

        lines = document_text.split("\n")
        for line in lines:
            line = line.strip()

            # Skip header/footer lines
            if any(
                word in line.upper()
                for word in [
                    "TOTAL",
                    "SUBTOTAL",
                    "GST",
                    "RECEIPT",
                    "RESTAURANT",
                    "ABN",
                    "DATE",
                ]
            ):
                continue

            # Match food/drink items
            item_patterns = [
                r"([A-Za-z\s]+(?:salad|burger|pizza|coffee|tea|beer|wine|steak|chicken|fish|pasta))\s+\$?(\d+\.\d{2})",
                r"([A-Za-z\s]+)\s+\$?(\d+\.\d{2})",
            ]

            for pattern in item_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    item_name = match.group(1).strip().title()
                    price = match.group(2)

                    # Filter out obvious non-food items
                    food_indicators = [
                        "salad",
                        "burger",
                        "pizza",
                        "coffee",
                        "tea",
                        "beer",
                        "wine",
                        "steak",
                        "chicken",
                        "fish",
                        "pasta",
                        "soup",
                        "sandwich",
                    ]
                    if (
                        any(food in item_name.lower() for food in food_indicators)
                        or len(item_name) > 3
                    ):
                        items_list.append(item_name)
                        prices_list.append(price)
                    break

        if items_list:
            extracted_fields["items"] = " | ".join(items_list)
            extracted_fields["prices"] = " | ".join(prices_list)

        # Extract GST and total
        gst_patterns = [r"gst[\s:]*\$?(\d+\.\d{2})", r"tax[\s:]*\$?(\d+\.\d{2})"]

        for pattern in gst_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["gst_amount"] = match.group(1)
                break

        total_patterns = [r"total[\s:]*\$?(\d+\.\d{2})", r"amount[\s:]*\$?(\d+\.\d{2})"]

        for pattern in total_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["total_amount"] = match.group(1)
                break

        logger.debug(
            f"Primary extraction yielded {len(extracted_fields)} fields for meal receipt"
        )
        return extracted_fields

    def _get_required_fields(self) -> List[str]:
        """Get required fields for meal receipt processing."""
        return ["date", "restaurant_name", "total_amount"]

    def _get_optional_fields(self) -> List[str]:
        """Get optional fields for meal receipt processing."""
        return [
            "time",
            "meal_type",
            "items",
            "prices",
            "gst_amount",
            "subtotal",
            "covers",
            "payment_method",
            "restaurant_abn",
        ]

    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for meal receipt fields."""
        return {
            "date": self._validate_australian_date,
            "time": self._validate_time_format,
            "total_amount": self._validate_currency_amount,
            "gst_amount": self._validate_currency_amount,
            "covers": self._validate_covers_count,
            "meal_type": self._validate_meal_type,
        }

    def _get_ato_thresholds(self) -> Dict[str, Any]:
        """Get ATO-specific thresholds for meal receipts."""
        return {
            "receipt_required_threshold": 82.50,
            "entertainment_deduction_rate": 0.5,  # 50% deductible
            "business_purpose_required": True,
            "gst_rate": 0.10,
        }

    def _get_awk_rules(self) -> List[Dict[str, Any]]:
        """Get AWK extraction rules for meal receipts."""
        return [
            {
                "field": "restaurant_name",
                "pattern": r"([A-Z][A-Za-z\s&]+(?:RESTAURANT|CAFE|BAR|BISTRO))",
                "line_filter": lambda line: any(
                    word in line.upper()
                    for word in ["RESTAURANT", "CAFE", "BAR", "BISTRO"]
                ),
                "transform": lambda x: x.strip(),
            },
            {
                "field": "date",
                "pattern": r"\d{1,2}/\d{1,2}/\d{4}",
                "line_filter": lambda line: any(
                    word in line.lower() for word in ["date", "dated"]
                ),
                "transform": lambda x: x.strip(),
            },
            {
                "field": "total_amount",
                "pattern": r"\$?(\d+\.\d{2})",
                "line_filter": lambda line: any(
                    word in line.lower() for word in ["total", "amount"]
                ),
                "transform": lambda x: x.replace("$", "").strip(),
            },
        ]

    def _validate_australian_date(self, date_str: str) -> str:
        """Validate Australian date format."""
        if not date_str:
            return date_str

        date_patterns = [
            (r"(\d{1,2})/(\d{1,2})/(\d{4})", r"\1/\2/\3"),
            (r"(\d{1,2})-(\d{1,2})-(\d{4})", r"\1/\2/\3"),
        ]

        for pattern, replacement in date_patterns:
            match = re.match(pattern, date_str.strip())
            if match:
                return re.sub(pattern, replacement, date_str.strip())

        return date_str

    def _validate_time_format(self, time_str: str) -> str:
        """Validate time format."""
        if not time_str:
            return time_str

        time_pattern = r"^\d{1,2}:\d{2}(?::\d{2})?$"
        if re.match(time_pattern, time_str.strip()):
            return time_str.strip()

        return time_str

    def _validate_currency_amount(self, amount_str: str) -> str:
        """Validate currency amount format."""
        if not amount_str:
            return amount_str

        clean_amount = re.sub(r"[^\d.]", "", amount_str)

        try:
            amount = float(clean_amount)
            if 0.01 <= amount <= 500.0:  # Reasonable meal range
                return f"{amount:.2f}"
            return clean_amount
        except ValueError:
            return amount_str

    def _validate_covers_count(self, covers_str: str) -> str:
        """Validate covers count."""
        if not covers_str:
            return covers_str

        try:
            covers = int(covers_str)
            if 1 <= covers <= 20:  # Reasonable range
                return str(covers)
            return covers_str
        except ValueError:
            return covers_str

    def _validate_meal_type(self, meal_str: str) -> str:
        """Validate meal type."""
        if not meal_str:
            return meal_str

        valid_types = [
            "Breakfast",
            "Lunch",
            "Dinner",
            "Brunch",
            "Coffee",
            "Beverage",
            "Snack",
        ]
        meal_title = meal_str.title()

        for meal_type in valid_types:
            if meal_type in meal_title:
                return meal_type

        return meal_str.title()
