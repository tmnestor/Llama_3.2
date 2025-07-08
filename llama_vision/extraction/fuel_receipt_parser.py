"""Fuel receipt parsing for Australian fuel retailers."""

import re
from typing import Any, Dict


class FuelReceiptParser:
    """Parser for fuel receipts from any Australian fuel retailer.

    Supports: Costco, Shell, BP, Coles Express, Woolworths, 7-Eleven, and others.
    """

    def __init__(self, log_level: str = "INFO"):
        """Initialize fuel receipt parser.

        Args:
            log_level: Logging level
        """
        from ..utils import setup_logging

        self.logger = setup_logging(log_level)

    def extract_fuel_fields(
        self, response: str, parsed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract fuel-specific fields from any Australian fuel retailer.

        Args:
            response: Model response text
            parsed: Current parsed data to update

        Returns:
            Updated parsed data with fuel-specific fields
        """
        # Check if this is a fuel receipt
        if not self._is_fuel_receipt(response):
            return parsed

        # Extract fuel quantity
        self._extract_fuel_quantity(response, parsed)

        # Extract price per litre
        self._extract_price_per_litre(response, parsed)

        # Extract fuel type
        self._extract_fuel_type(response, parsed)

        # Extract loyalty/member numbers
        self._extract_member_number(response, parsed)

        # Calculate fuel costs if possible
        self._calculate_fuel_costs(parsed)

        # Calculate GST for fuel
        self._calculate_fuel_gst(parsed)

        return parsed

    def _is_fuel_receipt(self, response: str) -> bool:
        """Check if response indicates a fuel receipt.

        Args:
            response: Model response text

        Returns:
            True if this appears to be a fuel receipt
        """
        fuel_indicators = [
            "ULP",
            "UNLEADED",
            "DIESEL",
            "PETROL",
            "FUEL",
            "LITRE",
            "L/100",
            "/L",
            "OCTANE",
            "E10",
            "LPG",
        ]

        return any(indicator in response.upper() for indicator in fuel_indicators)

    def _extract_fuel_quantity(self, response: str, parsed: Dict[str, Any]) -> None:
        """Extract fuel quantity from response.

        Args:
            response: Model response text
            parsed: Current parsed data to update
        """
        if "fuel_quantity" in parsed:
            return

        quantity_patterns = [
            r"(\d+\.\d{3})L",  # Costco: 32.230L
            r"(\d+\.\d{2})L",  # Shell/BP: 45.67L
            r"(\d+\.\d{1})L",  # Some retailers: 32.2L
            r"(\d+)\.(\d+)\s*L",  # Spaced: 32.230 L
            r"LITRES?:\s*(\d+\.\d+)",  # "Litres: 32.230"
            r"QTY:\s*(\d+\.\d+)L",  # "Qty: 32.230L"
            r"VOLUME:\s*(\d+\.\d+)L",  # "Volume: 32.230L"
        ]

        for pattern in quantity_patterns:
            quantity_match = re.search(pattern, response, re.IGNORECASE)
            if quantity_match:
                if "." in pattern and len(quantity_match.groups()) == 1:
                    quantity = quantity_match.group(1)
                else:
                    quantity = f"{quantity_match.group(1)}.{quantity_match.group(2)}"
                parsed["fuel_quantity"] = f"{quantity}L"
                self.logger.debug(f"Extracted fuel quantity: {quantity}L")
                break

    def _extract_price_per_litre(self, response: str, parsed: Dict[str, Any]) -> None:
        """Extract price per litre from response.

        Args:
            response: Model response text
            parsed: Current parsed data to update
        """
        if "price_per_litre" in parsed:
            return

        price_patterns = [
            r"(\d{3})/L",  # Costco: 827/L (cents)
            r"\$(\d+\.\d{2})/L",  # Shell: $1.85/L
            r"(\d+\.\d{3})\s*¢/L",  # BP: 184.5¢/L
            r"PRICE/L:\s*\$?(\d+\.\d{2})",  # "Price/L: $1.84"
            r"PER\s*LITRE:\s*\$?(\d+\.\d{2})",  # "Per Litre: $1.84"
            r"UNIT\s*PRICE:\s*\$?(\d+\.\d{2})",  # "Unit Price: $1.84"
            r"RATE:\s*\$?(\d+\.\d{2})",  # "Rate: $1.84"
        ]

        for i, pattern in enumerate(price_patterns):
            price_match = re.search(pattern, response, re.IGNORECASE)
            if price_match:
                price_value = price_match.group(1)
                if i == 0:  # Costco cents format
                    price_dollars = float(price_value) / 100
                    parsed["price_per_litre"] = f"${price_dollars:.2f}/L"
                elif i == 2:  # BP cents format
                    price_dollars = float(price_value) / 100
                    parsed["price_per_litre"] = f"${price_dollars:.2f}/L"
                else:  # Already in dollars
                    parsed["price_per_litre"] = f"${price_value}/L"

                self.logger.debug(
                    f"Extracted price per litre: {parsed['price_per_litre']}"
                )
                break

    def _extract_fuel_type(self, response: str, parsed: Dict[str, Any]) -> None:
        """Extract fuel type from response.

        Args:
            response: Model response text
            parsed: Current parsed data to update
        """
        if "fuel_type" in parsed:
            return

        fuel_type_patterns = [
            r"(13ULP)",  # Costco
            r"(U91|ULP|UNLEADED)",  # Standard unleaded
            r"(U95|PREMIUM\s*ULP)",  # Premium unleaded
            r"(U98|SUPER\s*PREMIUM)",  # Super premium
            r"(DIESEL|DSL)",  # Diesel
            r"(E10)",  # Ethanol blend
            r"(LPG)",  # LPG
            r"(PREMIUM\s*UNLEADED)",  # Premium unleaded
            r"(REGULAR\s*UNLEADED)",  # Regular unleaded
        ]

        for pattern in fuel_type_patterns:
            fuel_match = re.search(pattern, response, re.IGNORECASE)
            if fuel_match:
                parsed["fuel_type"] = fuel_match.group(1).upper()
                self.logger.debug(f"Extracted fuel type: {parsed['fuel_type']}")
                break

    def _extract_member_number(self, response: str, parsed: Dict[str, Any]) -> None:
        """Extract member/loyalty numbers from response.

        Args:
            response: Model response text
            parsed: Current parsed data to update
        """
        if "member_number" in parsed:
            return

        member_patterns = [
            r"Member\s*#?:?\s*(\d+)",  # Costco: "Member #1234"
            r"FlyBuys\s*#?:?\s*(\d+)",  # Coles: "FlyBuys: 1234"
            r"Everyday\s*Rewards\s*#?:?\s*(\d+)",  # Woolworths: "Everyday Rewards: 1234"
            r"Shell\s*Card\s*#?:?\s*(\d+)",  # Shell: "Shell Card: 1234"
            r"BP\s*Card\s*#?:?\s*(\d+)",  # BP: "BP Card: 1234"
            r"Card\s*#?:?\s*(\d+)",  # Generic: "Card: 1234"
            r"Loyalty\s*#?:?\s*(\d+)",  # Generic: "Loyalty: 1234"
            r"Account\s*#?:?\s*(\d+)",  # Generic: "Account: 1234"
        ]

        for pattern in member_patterns:
            member_match = re.search(pattern, response, re.IGNORECASE)
            if member_match:
                parsed["member_number"] = member_match.group(1)
                self.logger.debug(f"Extracted member number: {parsed['member_number']}")
                break

    def _calculate_fuel_costs(self, parsed: Dict[str, Any]) -> None:
        """Calculate fuel costs if quantity and price are available.

        Args:
            parsed: Current parsed data to update
        """
        if "fuel_quantity" not in parsed or "price_per_litre" not in parsed:
            return

        if "total_amount" in parsed:
            return  # Don't override existing total

        try:
            # Extract numeric values
            quantity_str = parsed["fuel_quantity"].replace("L", "")
            price_str = parsed["price_per_litre"].replace("$", "").replace("/L", "")

            quantity = float(quantity_str)
            price_per_litre = float(price_str)

            # Calculate total (quantity × price per litre)
            calculated_total = quantity * price_per_litre
            parsed["calculated_total"] = round(calculated_total, 2)

            # If we don't have total_amount yet, use calculated
            if "total_amount" not in parsed:
                parsed["total_amount"] = calculated_total
                self.logger.debug(f"Calculated fuel total: ${calculated_total:.2f}")

        except (ValueError, KeyError) as e:
            self.logger.debug(f"Could not calculate fuel costs: {e}")

    def _calculate_fuel_gst(self, parsed: Dict[str, Any]) -> None:
        """Calculate GST for fuel (10% of total in Australia).

        Args:
            parsed: Current parsed data to update
        """
        if "gst_amount" in parsed or "total_amount" not in parsed:
            return

        try:
            total = float(parsed["total_amount"])
            # GST component of inclusive amount (total includes GST)
            gst = total * 0.1 / 1.1
            parsed["gst_amount"] = round(gst, 2)
            self.logger.debug(f"Calculated fuel GST: ${gst:.2f}")
        except (ValueError, KeyError) as e:
            self.logger.debug(f"Could not calculate fuel GST: {e}")

    def extract_fuel_items(self, response: str) -> list:
        """Extract fuel item descriptions from response.

        Args:
            response: Model response text

        Returns:
            List of fuel item descriptions
        """
        fuel_patterns = [
            r"13ULP\s+[\d.]+L",  # Costco specific
            r"U91\s+[\d.]+L",  # Shell/BP Unleaded 91
            r"U95\s+[\d.]+L",  # Shell/BP Premium 95
            r"U98\s+[\d.]+L",  # Shell/BP Super Premium 98
            r"ULP\s+[\d.]+L",  # Generic unleaded
            r"UNLEADED\s+[\d.]+L",  # Generic unleaded
            r"PREMIUM\s+ULP\s+[\d.]+L",  # Premium unleaded
            r"E10\s+[\d.]+L",  # Ethanol blend
            r"DIESEL\s+[\d.]+L",  # Diesel
            r"DSL\s+[\d.]+L",  # Diesel abbreviation
            r"FUEL\s+[\d.]+L",  # Generic fuel
            r"PETROL\s+[\d.]+L",  # Generic petrol
            r"LPG\s+[\d.]+L",  # LPG
        ]

        found_fuel = []
        for pattern in fuel_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                found_fuel.append(match.strip())

        return found_fuel
