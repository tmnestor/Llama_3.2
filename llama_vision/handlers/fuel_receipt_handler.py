"""
Fuel Receipt Handler for Australian Tax Document Processing

This handler specializes in processing fuel receipts for vehicle expense claims
with ATO compliance validation.
"""

import re
from typing import Any, Dict, List

from ..extraction.australian_tax_classifier import DocumentType
from ..utils import setup_logging
from .base_ato_handler import BaseATOHandler

logger = setup_logging()


class FuelReceiptHandler(BaseATOHandler):
    """Handler for Australian fuel receipt processing with ATO compliance."""

    def __init__(self):
        super().__init__(DocumentType.FUEL_RECEIPT)
        logger.info("FuelReceiptHandler initialized for vehicle expense claims")

    def _extract_fields_primary(self, document_text: str) -> Dict[str, Any]:
        """Extract fields using primary extraction method for fuel receipts."""
        extracted_fields = {}

        # Extract station name
        station_patterns = [
            r"(bp|shell|caltex|ampol|mobil|7-eleven|united petroleum|liberty|metro petroleum|speedway)",
            r"([A-Z\s]+(?:petroleum|fuel|service station))",
            r"([A-Z\s]+(?:bp|shell|caltex|ampol|mobil))",
        ]

        for pattern in station_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["station_name"] = match.group(1).strip().upper()
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

        # Extract fuel type
        fuel_type_patterns = [
            r"(unleaded|premium|diesel|e10|e85|lpg)",
            r"(petrol|gasoline)",
            r"(ul|prem|dies)",
        ]

        for pattern in fuel_type_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["fuel_type"] = match.group(1).title()
                break

        # Extract litres
        litres_patterns = [
            r"(\d+\.\d{2,3})\s*l",
            r"(\d+\.\d{2,3})\s*litres?",
            r"qty:?\s*(\d+\.\d{2,3})",
            r"volume:?\s*(\d+\.\d{2,3})",
        ]

        for pattern in litres_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["litres"] = match.group(1)
                break

        # Extract price per litre
        price_per_litre_patterns = [
            r"(\d+\.\d{1,3})\s*c/l",
            r"(\d+\.\d{1,3})\s*cents/litre",
            r"rate:?\s*(\d+\.\d{1,3})",
            r"price:?\s*(\d+\.\d{1,3})",
        ]

        for pattern in price_per_litre_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["price_per_litre"] = match.group(1)
                break

        # Extract total amount
        total_patterns = [
            r"total:?\s*\$?(\d+\.\d{2})",
            r"amount:?\s*\$?(\d+\.\d{2})",
            r"pay:?\s*\$?(\d+\.\d{2})",
        ]

        for pattern in total_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["total_amount"] = match.group(1)
                break

        # Extract GST amount
        gst_patterns = [
            r"gst:?\s*\$?(\d+\.\d{2})",
            r"tax:?\s*\$?(\d+\.\d{2})",
            r"goods\s+services\s+tax:?\s*\$?(\d+\.\d{2})",
        ]

        for pattern in gst_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["gst_amount"] = match.group(1)
                break

        # Extract pump number
        pump_patterns = [r"pump:?\s*(\d+)", r"p(\d+)", r"#(\d+)"]

        for pattern in pump_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["pump_number"] = match.group(1)
                break

        # Extract time
        time_patterns = [r"(\d{1,2}:\d{2}(?::\d{2})?)", r"time:?\s*(\d{1,2}:\d{2})"]

        for pattern in time_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["time"] = match.group(1)
                break

        # Extract station address
        address_patterns = [
            r"(\d+\s+[A-Za-z\s]+(?:street|st|road|rd|avenue|ave|drive|dr|lane|ln)[A-Za-z\s,]*(?:NSW|VIC|QLD|WA|SA|TAS|NT|ACT))",
            r"([A-Za-z\s]+(?:NSW|VIC|QLD|WA|SA|TAS|NT|ACT)\s+\d{4})",
        ]

        for pattern in address_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["station_address"] = match.group(1).strip()
                break

        # Extract vehicle odometer
        odometer_patterns = [r"odometer:?\s*(\d+)", r"km:?\s*(\d+)", r"kms:?\s*(\d+)"]

        for pattern in odometer_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["vehicle_km"] = match.group(1)
                break

        logger.debug(
            f"Primary extraction yielded {len(extracted_fields)} fields for fuel receipt"
        )
        return extracted_fields

    def _get_required_fields(self) -> List[str]:
        """Get required fields for fuel receipt processing."""
        return ["date", "station_name", "fuel_type", "litres", "total_amount"]

    def _get_optional_fields(self) -> List[str]:
        """Get optional fields for fuel receipt processing."""
        return [
            "price_per_litre",
            "gst_amount",
            "pump_number",
            "time",
            "station_address",
            "vehicle_km",
            "subtotal",
        ]

    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for fuel receipt fields."""
        return {
            "date": self._validate_australian_date,
            "litres": self._validate_fuel_quantity,
            "total_amount": self._validate_currency_amount,
            "gst_amount": self._validate_currency_amount,
            "price_per_litre": self._validate_fuel_price,
            "pump_number": self._validate_pump_number,
            "time": self._validate_time_format,
            "vehicle_km": self._validate_odometer,
        }

    def _get_ato_thresholds(self) -> Dict[str, Any]:
        """Get ATO-specific thresholds for fuel receipts."""
        return {
            "receipt_required_threshold": 82.50,
            "gst_rate": 0.10,
            "gst_tolerance": 0.02,
            "minimum_business_use_percentage": 10,
            "logbook_required_threshold": 5000,  # Annual business km
        }

    def _get_awk_rules(self) -> List[Dict[str, Any]]:
        """Get AWK extraction rules for fuel receipts."""
        return [
            {
                "field": "station_name",
                "pattern": r"(BP|SHELL|CALTEX|AMPOL|MOBIL|7-ELEVEN)",
                "line_filter": lambda line: any(
                    station in line.upper()
                    for station in [
                        "BP",
                        "SHELL",
                        "CALTEX",
                        "AMPOL",
                        "MOBIL",
                        "7-ELEVEN",
                    ]
                ),
                "transform": lambda x: x.strip().upper(),
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
                "field": "fuel_type",
                "pattern": r"(unleaded|premium|diesel|e10|e85|lpg)",
                "line_filter": lambda line: any(
                    fuel in line.lower()
                    for fuel in ["unleaded", "premium", "diesel", "fuel"]
                ),
                "transform": lambda x: x.strip().title(),
            },
            {
                "field": "litres",
                "pattern": r"\d+\.\d{2,3}",
                "line_filter": lambda line: any(
                    unit in line.lower() for unit in ["l", "litre", "ltr"]
                ),
                "transform": lambda x: x.strip(),
            },
            {
                "field": "total_amount",
                "pattern": r"\$?\d+\.\d{2}",
                "line_filter": lambda line: any(
                    word in line.lower() for word in ["total", "amount", "pay"]
                ),
                "transform": lambda x: x.replace("$", "").strip(),
            },
            {
                "field": "gst_amount",
                "pattern": r"\$?\d+\.\d{2}",
                "line_filter": lambda line: "gst" in line.lower()
                or "tax" in line.lower(),
                "transform": lambda x: x.replace("$", "").strip(),
            },
            {
                "field": "price_per_litre",
                "pattern": r"\d+\.\d{1,3}",
                "line_filter": lambda line: any(
                    unit in line.lower()
                    for unit in ["c/l", "cents/litre", "rate", "price"]
                ),
                "transform": lambda x: x.strip(),
            },
            {
                "field": "pump_number",
                "pattern": r"\d+",
                "line_filter": lambda line: "pump" in line.lower()
                or line.startswith("P")
                or line.startswith("#"),
                "transform": lambda x: x.strip(),
            },
        ]

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

    def _validate_fuel_quantity(self, quantity_str: str) -> str:
        """Validate fuel quantity format."""
        if not quantity_str:
            return quantity_str

        # Remove units and validate number
        clean_qty = re.sub(r"[^\d.]", "", quantity_str)

        try:
            qty = float(clean_qty)
            if 0.1 <= qty <= 200.0:  # Reasonable fuel quantity range
                return f"{qty:.3f}"
            else:
                logger.warning(f"Fuel quantity {qty} outside reasonable range")
                return clean_qty
        except ValueError:
            logger.warning(f"Invalid fuel quantity format: {quantity_str}")
            return quantity_str

    def _validate_currency_amount(self, amount_str: str) -> str:
        """Validate currency amount format."""
        if not amount_str:
            return amount_str

        # Remove currency symbols and validate
        clean_amount = re.sub(r"[^\d.]", "", amount_str)

        try:
            amount = float(clean_amount)
            if 0.01 <= amount <= 1000.0:  # Reasonable fuel cost range
                return f"{amount:.2f}"
            else:
                logger.warning(f"Fuel cost {amount} outside reasonable range")
                return clean_amount
        except ValueError:
            logger.warning(f"Invalid currency amount format: {amount_str}")
            return amount_str

    def _validate_fuel_price(self, price_str: str) -> str:
        """Validate fuel price per litre format."""
        if not price_str:
            return price_str

        # Remove units and validate
        clean_price = re.sub(r"[^\d.]", "", price_str)

        try:
            price = float(clean_price)
            if 80.0 <= price <= 300.0:  # Reasonable price range in cents
                return f"{price:.1f}"
            else:
                logger.warning(f"Fuel price {price} outside reasonable range")
                return clean_price
        except ValueError:
            logger.warning(f"Invalid fuel price format: {price_str}")
            return price_str

    def _validate_pump_number(self, pump_str: str) -> str:
        """Validate pump number format."""
        if not pump_str:
            return pump_str

        # Extract numeric part
        clean_pump = re.sub(r"[^\d]", "", pump_str)

        try:
            pump_num = int(clean_pump)
            if 1 <= pump_num <= 50:  # Reasonable pump number range
                return str(pump_num)
            else:
                logger.warning(f"Pump number {pump_num} outside reasonable range")
                return clean_pump
        except ValueError:
            logger.warning(f"Invalid pump number format: {pump_str}")
            return pump_str

    def _validate_time_format(self, time_str: str) -> str:
        """Validate time format."""
        if not time_str:
            return time_str

        # Validate HH:MM format
        time_pattern = r"^\d{1,2}:\d{2}(?::\d{2})?$"

        if re.match(time_pattern, time_str.strip()):
            return time_str.strip()
        else:
            logger.warning(f"Invalid time format: {time_str}")
            return time_str

    def _validate_odometer(self, odometer_str: str) -> str:
        """Validate odometer reading format."""
        if not odometer_str:
            return odometer_str

        # Extract numeric part
        clean_odometer = re.sub(r"[^\d]", "", odometer_str)

        try:
            odometer = int(clean_odometer)
            if 0 <= odometer <= 999999:  # Reasonable odometer range
                return str(odometer)
            else:
                logger.warning(f"Odometer reading {odometer} outside reasonable range")
                return clean_odometer
        except ValueError:
            logger.warning(f"Invalid odometer format: {odometer_str}")
            return odometer_str
