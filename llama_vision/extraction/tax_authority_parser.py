"""Tax authority specific parser for Llama-3.2-Vision package."""

import re
from datetime import datetime
from typing import Any, Dict, List

from ..utils import setup_logging


class TaxAuthorityParser:
    """Parse responses for tax authority requirements (national taxation office)."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize tax authority parser.

        Args:
            log_level: Logging level
        """
        self.log_level = log_level
        self.logger = setup_logging(log_level)

    def parse_receipt_response(self, response: str) -> Dict[str, Any]:
        """Parse KEY-VALUE format responses from Llama for tax data extraction.

        This method handles Llama's KEY-VALUE response format as specified in prompts.

        Args:
            response: Raw model response in KEY-VALUE format

        Returns:
            Parsed tax authority data
        """
        self.logger.info(f"Raw model response: {response}")
        self.logger.debug(f"Parsing receipt response: {response[:100]}...")

        parsed = {}

        # First try to parse KEY-VALUE format (modern approach)
        parsed = self._parse_key_value_format(response)

        # If KEY-VALUE parsing didn't find much, fall back to pattern matching
        if len(parsed) < 3:
            self.logger.debug(
                "KEY-VALUE parsing found limited data, trying pattern matching..."
            )
            parsed = self._parse_with_patterns(response)
        else:
            self.logger.debug(
                f"KEY-VALUE parsing successful with {len(parsed)} fields, skipping pattern matching"
            )
            # Normalize KEY-VALUE fields to tax authority standards (replace duplicates)
            parsed = self._normalize_key_value_fields(parsed)

        # Extract product items (for detailed expense tracking)
        parsed = self._extract_product_items(response, parsed)

        # Add compliance validation
        parsed = self._add_compliance_fields(parsed)

        # Calculate tax authority compliance score
        compliance_score = self._calculate_compliance_score(parsed)
        parsed["_compliance_score"] = compliance_score
        parsed["_extraction_method"] = "TAX_AUTHORITY_PARSER"

        self.logger.info(
            f"Tax authority parsing extracted {len(parsed)} fields (compliance: {compliance_score:.2f})"
        )

        return parsed

    def _parse_key_value_format(self, response: str) -> Dict[str, Any]:
        """Parse KEY-VALUE format response.

        Args:
            response: Model response text

        Returns:
            Parsed data dictionary
        """
        parsed = {}

        # Standard KEY-VALUE patterns (case insensitive)
        kv_patterns = [
            (r"DATE:\s*([^\n\r]+)", "DATE"),
            (r"STORE:\s*([^\n\r]+)", "STORE"),
            (r"ABN:\s*([^\n\r]+)", "ABN"),
            (r"PAYER:\s*([^\n\r]+)", "PAYER"),
            (r"TAX:\s*([^\n\r]+)", "TAX"),
            (r"GST:\s*([^\n\r]+)", "GST"),
            (r"TOTAL:\s*([^\n\r]+)", "TOTAL"),
            (r"SUBTOTAL:\s*([^\n\r]+)", "SUBTOTAL"),
            (r"PRODUCTS:\s*([^\n\r]+)", "PRODUCTS"),
            (r"QUANTITIES:\s*([^\n\r]+)", "QUANTITIES"),
            (r"PRICES:\s*([^\n\r]+)", "PRICES"),
            (r"RECEIPT:\s*([^\n\r]+)", "RECEIPT"),
            (r"INVOICE_NUMBER:\s*([^\n\r]+)", "INVOICE_NUMBER"),
            (r"PAYMENT_METHOD:\s*([^\n\r]+)", "PAYMENT_METHOD"),
            # Bank statement specific patterns
            (r"ACCOUNT_NUMBER:\s*([^\n\r]+)", "ACCOUNT_NUMBER"),
            (r"BSB:\s*([^\n\r]+)", "BSB"),
            (r"ACCOUNT_HOLDER:\s*([^\n\r]+)", "ACCOUNT_HOLDER"),
            (r"STATEMENT_PERIOD:\s*([^\n\r]+)", "STATEMENT_PERIOD"),
            (r"OPENING_BALANCE:\s*([^\n\r]+)", "OPENING_BALANCE"),
            (r"CLOSING_BALANCE:\s*([^\n\r]+)", "CLOSING_BALANCE"),
            (r"TOTAL_CREDITS:\s*([^\n\r]+)", "TOTAL_CREDITS"),
            (r"TOTAL_DEBITS:\s*([^\n\r]+)", "TOTAL_DEBITS"),
            (r"BANK_NAME:\s*([^\n\r]+)", "BANK_NAME"),
            (r"TRANSACTION_COUNT:\s*([^\n\r]+)", "TRANSACTION_COUNT"),
            (r"STATEMENT_DATE:\s*([^\n\r]+)", "STATEMENT_DATE"),
        ]

        for pattern, key in kv_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value and value not in ["", "N/A", "Not visible", "Not available"]:
                    parsed[key] = value

                    # Add tax authority specific mappings
                    if key == "STORE":
                        parsed["supplier_name"] = value
                        parsed["BUSINESS_NAME"] = value
                    elif key == "DATE":
                        if self._validate_australian_date(value):
                            parsed["invoice_date"] = value
                            parsed["transaction_date"] = value
                    elif key == "ABN":
                        parsed["supplier_abn"] = value
                    elif key in ["TAX", "GST"]:
                        # Extract numeric amount
                        numeric_value = re.search(r"[\d.]+", value.replace("$", ""))
                        if numeric_value:
                            amount = numeric_value.group(0)
                            parsed["gst_amount"] = amount
                            parsed["tax_amount"] = amount
                    elif key == "TOTAL":
                        # Extract numeric amount
                        numeric_value = re.search(r"[\d.]+", value.replace("$", ""))
                        if numeric_value:
                            amount = numeric_value.group(0)
                            parsed["total_amount"] = amount
                    elif key == "PRODUCTS":
                        # Split by pipe separator
                        if "|" in value:
                            items = [
                                item.strip()
                                for item in value.split("|")
                                if item.strip()
                            ]
                            parsed["items"] = items
                            parsed["ITEMS"] = items
                        else:
                            parsed["items"] = [value]
                            parsed["ITEMS"] = [value]
                    elif key == "RECEIPT":
                        parsed["receipt_number"] = value
                        parsed["invoice_number"] = value
                    elif key == "PAYMENT_METHOD":
                        parsed["payment_method"] = value

                    # Bank statement specific mappings
                    elif key == "BANK_NAME":
                        parsed["supplier_name"] = value  # Map to standard field
                        parsed["institution_name"] = value
                    elif key == "ACCOUNT_HOLDER":
                        parsed["account_holder_name"] = value
                        parsed["customer_name"] = value
                    elif key == "STATEMENT_DATE":
                        if self._validate_australian_date(value):
                            parsed["invoice_date"] = value  # Map to standard field
                            parsed["transaction_date"] = value
                            parsed["statement_date"] = value
                    elif key in [
                        "OPENING_BALANCE",
                        "CLOSING_BALANCE",
                        "TOTAL_CREDITS",
                        "TOTAL_DEBITS",
                    ]:
                        # Extract numeric amount
                        numeric_value = re.search(r"[\d.]+", value.replace("$", ""))
                        if numeric_value:
                            amount = numeric_value.group(0)
                            parsed[f"{key.lower()}"] = amount
                            # Map closing balance to total_amount for compliance scoring
                            if key == "CLOSING_BALANCE":
                                parsed["total_amount"] = amount

        self.logger.debug(f"KEY-VALUE parsing extracted {len(parsed)} fields")
        return parsed

    def _normalize_key_value_fields(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize KEY-VALUE fields to tax authority standards without duplication.

        Args:
            parsed: Dictionary with KEY-VALUE extracted fields

        Returns:
            Normalized dictionary with standardized field names
        """
        normalized = {}

        # Define field mappings (target_field: list_of_possible_sources)
        field_mappings = {
            "supplier_name": ["STORE", "BUSINESS_NAME", "PAYER", "BANK_NAME"],
            "total_amount": ["TOTAL", "CLOSING_BALANCE"],
            "gst_amount": ["GST", "TAX"],
            "invoice_date": ["DATE", "STATEMENT_DATE"],
            "supplier_abn": ["ABN"],
            "items": ["PRODUCTS", "ITEMS"],
            "quantities": ["QUANTITIES"],
            "prices": ["PRICES"],
            "invoice_number": ["INVOICE_NUMBER", "RECEIPT"],
            "payment_method": ["PAYMENT_METHOD"],
            # Bank statement specific fields
            "account_number": ["ACCOUNT_NUMBER"],
            "bsb": ["BSB"],
            "account_holder": ["ACCOUNT_HOLDER"],
            "statement_period": ["STATEMENT_PERIOD"],
            "opening_balance": ["OPENING_BALANCE"],
            "closing_balance": ["CLOSING_BALANCE"],
            "total_credits": ["TOTAL_CREDITS"],
            "total_debits": ["TOTAL_DEBITS"],
            "transaction_count": ["TRANSACTION_COUNT"],
        }

        # Apply field mappings and remove source duplicates
        sources_to_remove = set()
        for target_field, source_fields in field_mappings.items():
            for source_field in source_fields:
                if source_field in parsed and target_field not in normalized:
                    value = parsed[source_field]
                    # Clean currency symbols and formatting
                    if target_field in ["total_amount", "gst_amount"] and isinstance(
                        value, str
                    ):
                        # Extract numeric value from currency strings like "$1.82"
                        import re

                        amount_match = re.search(r"[\d.]+", value.replace("$", ""))
                        if amount_match:
                            normalized[target_field] = float(amount_match.group())
                        else:
                            normalized[target_field] = value
                    else:
                        normalized[target_field] = value

                    # Mark all source fields for removal to avoid duplicates
                    sources_to_remove.update(source_fields)
                    break  # Use first match only

        # Remove the original source fields to eliminate duplicates
        for source_field in sources_to_remove:
            if source_field in parsed:
                del parsed[source_field]

        # Add normalized fields to the parsed dictionary
        parsed.update(normalized)

        # Add essential fields for compliance calculation
        if "supplier_name" in parsed:
            parsed["taxpayer_name"] = parsed["supplier_name"]

        # Add currency and country for Australian compliance
        parsed["currency"] = "AUD"
        parsed["country"] = "Australia"

        # Calculate GST compliance if amounts available
        if "total_amount" in parsed and "gst_amount" in parsed:
            try:
                total = float(parsed["total_amount"])
                gst = float(parsed["gst_amount"])
                if total > 0:
                    gst_rate = (gst / total) * 100
                    parsed["calculated_gst_rate"] = f"{gst_rate:.1f}%"
                    # Australian GST is 10%
                    parsed["gst_compliant"] = abs(gst_rate - 10.0) < 1.0
            except (ValueError, ZeroDivisionError):
                pass

        # Add expense category
        parsed["expense_category"] = "General Business Expenses"

        self.logger.debug(
            f"Normalized KEY-VALUE fields, removed duplicates. Final count: {len(parsed)} fields"
        )
        return parsed

    def _parse_with_patterns(self, response: str) -> Dict[str, Any]:
        """Fallback pattern-based parsing for non-KEY-VALUE responses.

        Args:
            response: Model response text

        Returns:
            Parsed data dictionary
        """
        parsed = {}

        # Extract business name (critical for tax authority)
        business_patterns = [
            r"COSTCO",
            r"WOOLWORTHS",
            r"COLES",
            r"ALDI",
            r"BUNNINGS",
            r"([A-Z][A-Z\s&]+[A-Z])\s+(?:PTY|LTD|LIMITED|WHOLESALE|SUPERMARKET)",
            r"Business[:\s]+([A-Z][A-Za-z\s&]+)",
        ]

        for pattern in business_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                if pattern == r"COSTCO":
                    business_name = "COSTCO"
                else:
                    business_name = match.group(1) if "(" in pattern else match.group(0)

                parsed["supplier_name"] = business_name
                parsed["STORE"] = business_name
                parsed["BUSINESS_NAME"] = business_name
                break

        # Extract transaction date (Australian format DD/MM/YYYY)
        date_patterns = [
            r"(\d{2}/\d{2}/\d{4})",
            r"Date[:\s]+(\d{2}/\d{2}/\d{4})",
            r"Transaction[:\s]+(\d{2}/\d{2}/\d{4})",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, response)
            if match:
                date = match.group(1)
                # Validate Australian date format
                if self._validate_australian_date(date):
                    parsed["invoice_date"] = date
                    parsed["DATE"] = date
                    parsed["transaction_date"] = date
                break

        # Extract financial amounts (critical for tax calculations)
        amount_patterns = [
            (r"Total[:\s]*\$?(\d+\.\d{2})", "TOTAL"),
            (r"GST[:\s]*\$?(\d+\.\d{2})", "GST"),
            (r"Tax[:\s]*\$?(\d+\.\d{2})", "TAX"),
            (r"Subtotal[:\s]*\$?(\d+\.\d{2})", "SUBTOTAL"),
        ]

        for pattern, field in amount_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                amount = match.group(1)
                parsed[field] = f"${amount}"
                parsed[f"{field.lower()}_amount"] = amount

                # Add tax authority specific fields
                if field == "TOTAL":
                    parsed["total_amount"] = amount
                elif field in ["GST", "TAX"]:
                    parsed["gst_amount"] = amount
                    parsed["tax_amount"] = amount

        # Extract Australian Business Number (ABN)
        abn_patterns = [
            r"ABN[:\s]*(\d{2}\s\d{3}\s\d{3}\s\d{3})",
            r"ABN[:\s]*(\d{11})",
            r"A\.?B\.?N\.?[:\s]*(\d{2}\s\d{3}\s\d{3}\s\d{3})",
        ]

        for pattern in abn_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                abn = match.group(1)
                parsed["supplier_abn"] = abn
                parsed["ABN"] = abn
                break

        # Extract receipt/invoice number
        receipt_patterns = [
            r"Receipt[:\s]*#?(\d+)",
            r"Invoice[:\s]*#?(\d+)",
            r"#(\d{6,})",
        ]

        for pattern in receipt_patterns:
            match = re.search(pattern, response)
            if match:
                receipt_num = match.group(1)
                parsed["receipt_number"] = receipt_num
                parsed["RECEIPT"] = receipt_num
                parsed["invoice_number"] = receipt_num
                break

        # Extract payment method
        payment_patterns = [
            r"CASH",
            r"CREDIT",
            r"DEBIT",
            r"EFTPOS",
            r"CARD",
            r"MASTERCARD",
            r"VISA",
        ]

        for pattern in payment_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                parsed["payment_method"] = pattern.upper()
                parsed["PAYMENT_METHOD"] = pattern.upper()
                break

        return parsed

    def _extract_product_items(
        self, response: str, parsed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract product items from response.

        Args:
            response: Model response text
            parsed: Current parsed data

        Returns:
            Updated parsed data with product information
        """
        # If products already extracted from KEY-VALUE format, don't override
        if "items" in parsed or "PRODUCTS" in parsed:
            return parsed

        # Look for fuel-specific items (generic patterns for all Australian retailers)
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

        # Use dedicated fuel parser for fuel receipts
        fuel_parser = self._get_fuel_parser()
        if fuel_parser._is_fuel_receipt(response):
            parsed = fuel_parser.extract_fuel_fields(response, parsed)

            # Also extract fuel items for the items field
            fuel_items = fuel_parser.extract_fuel_items(response)
            if fuel_items and "items" not in parsed:
                parsed["items"] = fuel_items

        if found_fuel:
            # Only add normalized field, avoid duplicates
            if "items" not in parsed:
                parsed["items"] = found_fuel
            return parsed

        # Common product keywords from receipts (fallback)
        product_keywords = [
            "Ice Cream",
            "Beer",
            "Water",
            "Coffee",
            "Chips",
            "Biscuits",
            "Milk",
            "Bread",
            "Eggs",
            "Chicken",
            "Beef",
            "Fish",
            "Vegetables",
            "Fruit",
            "Cheese",
            "Yogurt",
            "Butter",
        ]

        found_items = []
        for item in product_keywords:
            if item.lower() in response.lower():
                found_items.append(item)

        if found_items and "items" not in parsed:
            # Only add normalized field, avoid duplicates
            parsed["items"] = found_items

        return parsed

    def _extract_generic_fuel_fields(
        self, response: str, parsed: Dict[str, Any]
    ) -> None:
        """Extract fuel-specific fields from any Australian fuel retailer without creating duplicates.

        Works for: Costco, Shell, BP, Coles Express, Woolworths, 7-Eleven, etc.

        Args:
            response: Model response text
            parsed: Current parsed data to update
        """
        # Extract fuel quantity - multiple patterns for different retailers
        quantity_patterns = [
            r"(\d+\.\d{3})L",  # Costco: 32.230L
            r"(\d+\.\d{2})L",  # Shell/BP: 45.67L
            r"(\d+\.\d{1})L",  # Some retailers: 32.2L
            r"(\d+)\.(\d+)\s*L",  # Spaced: 32.230 L
            r"LITRES?:\s*(\d+\.\d+)",  # "Litres: 32.230"
            r"QTY:\s*(\d+\.\d+)L",  # "Qty: 32.230L"
        ]

        for pattern in quantity_patterns:
            quantity_match = re.search(pattern, response, re.IGNORECASE)
            if quantity_match and "fuel_quantity" not in parsed:
                if "." in pattern and len(quantity_match.groups()) == 1:
                    quantity = quantity_match.group(1)
                else:
                    quantity = f"{quantity_match.group(1)}.{quantity_match.group(2)}"
                parsed["fuel_quantity"] = f"{quantity}L"
                break

        # Extract price per litre - multiple formats for different retailers
        price_patterns = [
            r"(\d{3})/L",  # Costco: 827/L (cents)
            r"\$(\d+\.\d{2})/L",  # Shell: $1.85/L
            r"(\d+\.\d{3})\s*¢/L",  # BP: 184.5¢/L
            r"PRICE/L:\s*\$?(\d+\.\d{2})",  # "Price/L: $1.84"
            r"PER\s*LITRE:\s*\$?(\d+\.\d{2})",  # "Per Litre: $1.84"
            r"UNIT\s*PRICE:\s*\$?(\d+\.\d{2})",  # "Unit Price: $1.84"
        ]

        for i, pattern in enumerate(price_patterns):
            price_match = re.search(pattern, response, re.IGNORECASE)
            if price_match and "price_per_litre" not in parsed:
                price_value = price_match.group(1)
                if i == 0:  # Costco cents format
                    price_dollars = float(price_value) / 100
                    parsed["price_per_litre"] = f"${price_dollars:.2f}/L"
                elif i == 2:  # BP cents format
                    price_dollars = float(price_value) / 100
                    parsed["price_per_litre"] = f"${price_dollars:.2f}/L"
                else:  # Already in dollars
                    parsed["price_per_litre"] = f"${price_value}/L"
                break

        # Extract fuel type - common Australian fuel types
        fuel_type_patterns = [
            r"(13ULP)",  # Costco
            r"(U91|ULP|UNLEADED)",  # Standard unleaded
            r"(U95|PREMIUM\s*ULP)",  # Premium unleaded
            r"(U98|SUPER\s*PREMIUM)",  # Super premium
            r"(DIESEL|DSL)",  # Diesel
            r"(E10)",  # Ethanol blend
            r"(LPG)",  # LPG
        ]

        for pattern in fuel_type_patterns:
            fuel_match = re.search(pattern, response, re.IGNORECASE)
            if fuel_match and "fuel_type" not in parsed:
                parsed["fuel_type"] = fuel_match.group(1).upper()
                break

        # Extract member/loyalty numbers - works for different programs
        member_patterns = [
            r"Member\s*#?:?\s*(\d+)",  # Costco: "Member #1234"
            r"FlyBuys\s*#?:?\s*(\d+)",  # Coles: "FlyBuys: 1234"
            r"Everyday\s*Rewards\s*#?:?\s*(\d+)",  # Woolworths: "Everyday Rewards: 1234"
            r"Shell\s*Card\s*#?:?\s*(\d+)",  # Shell: "Shell Card: 1234"
            r"Card\s*#?:?\s*(\d+)",  # Generic: "Card: 1234"
            r"Loyalty\s*#?:?\s*(\d+)",  # Generic: "Loyalty: 1234"
        ]

        for pattern in member_patterns:
            member_match = re.search(pattern, response, re.IGNORECASE)
            if member_match and "member_number" not in parsed:
                parsed["member_number"] = member_match.group(1)
                break

        # Calculate fuel volume and costs if we have the data
        if (
            "fuel_quantity" in parsed
            and "price_per_litre" in parsed
            and "total_amount" not in parsed
        ):
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

            except (ValueError, KeyError):
                pass

        # Calculate GST for fuel (10% of total) - only if not already calculated
        if "total_amount" in parsed and "gst_amount" not in parsed:
            try:
                total = float(parsed["total_amount"])
                gst = total * 0.1 / 1.1  # GST component of inclusive amount
                parsed["gst_amount"] = round(gst, 2)
            except (ValueError, KeyError):
                pass

    def _add_compliance_fields(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Add Australian tax compliance specific fields.

        Args:
            parsed: Current parsed data

        Returns:
            Data with compliance fields added
        """
        # Add currency designation
        parsed["currency"] = "AUD"
        parsed["country"] = "Australia"

        # Validate GST rate (should be 10% in Australia)
        if "total_amount" in parsed and "gst_amount" in parsed:
            try:
                total = float(parsed["total_amount"])
                gst = float(parsed["gst_amount"])

                if total > 0:
                    # Calculate GST rate
                    gst_rate = (gst / (total - gst)) * 100
                    parsed["calculated_gst_rate"] = f"{gst_rate:.1f}%"
                    parsed["gst_compliant"] = abs(gst_rate - 10.0) < 1.0  # 10% ± 1%

            except (ValueError, ZeroDivisionError):
                pass

        # Add taxpayer identification fields
        if "supplier_name" in parsed:
            parsed["taxpayer_name"] = parsed["supplier_name"]

        # Add expense category (can be enhanced based on items)
        if "items" in parsed:
            parsed["expense_category"] = self._categorize_expense(parsed["items"])

        return parsed

    def _categorize_expense(self, items: List[str]) -> str:
        """Categorize expense based on items purchased.

        Args:
            items: List of purchased items

        Returns:
            Expense category string
        """
        # Simple categorization logic
        food_keywords = ["milk", "bread", "meat", "fruit", "vegetable", "cheese"]
        office_keywords = ["paper", "pen", "printer", "computer", "software"]
        fuel_keywords = ["fuel", "petrol", "gas", "diesel"]

        items_lower = [item.lower() for item in items]

        if any(keyword in " ".join(items_lower) for keyword in fuel_keywords):
            return "Vehicle Expenses"
        elif any(keyword in " ".join(items_lower) for keyword in office_keywords):
            return "Office Expenses"
        elif any(keyword in " ".join(items_lower) for keyword in food_keywords):
            return "Meals & Entertainment"
        else:
            return "General Business Expenses"

    def _validate_australian_date(self, date_str: str) -> bool:
        """Validate if date is in Australian format DD/MM/YYYY.

        Args:
            date_str: Date string to validate

        Returns:
            True if valid Australian date format
        """
        try:
            # Parse as DD/MM/YYYY
            datetime.strptime(date_str, "%d/%m/%Y")
            return True
        except ValueError:
            return False

    def _calculate_compliance_score(self, parsed: Dict[str, Any]) -> float:
        """Calculate tax authority compliance score.

        Args:
            parsed: Parsed data dictionary

        Returns:
            Compliance score between 0.0 and 1.0
        """
        required_fields = [
            "supplier_name",  # Business name
            "invoice_date",  # Transaction date
            "total_amount",  # Total amount
        ]

        compliance_fields = [
            "supplier_abn",  # ABN for tax validation
            "gst_amount",  # GST for tax calculations
            "payment_method",  # Payment verification
        ]

        # Check required fields (70% weight)
        required_score = sum(1 for field in required_fields if field in parsed) / len(
            required_fields
        )

        # Check compliance fields (30% weight)
        compliance_score = sum(
            1 for field in compliance_fields if field in parsed
        ) / len(compliance_fields)

        # Calculate weighted score
        total_score = (required_score * 0.7) + (compliance_score * 0.3)

        return total_score

    def validate_for_tax_authority(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parsed data for tax authority requirements.

        Args:
            parsed_data: Parsed receipt data

        Returns:
            Validation results with recommendations
        """
        validation_result = {
            "is_tax_compliant": False,
            "compliance_score": 0.0,
            "validation_errors": [],
            "recommendations": [],
            "required_fields_present": [],
            "missing_fields": [],
        }

        # Required fields for tax authority
        required_fields = {
            "supplier_name": "Business/Supplier Name",
            "invoice_date": "Transaction Date",
            "total_amount": "Total Amount",
        }

        # Check required fields
        missing_fields = []
        present_fields = []

        for field, description in required_fields.items():
            if field in parsed_data and parsed_data[field]:
                present_fields.append(description)
            else:
                missing_fields.append(description)
                validation_result["validation_errors"].append(f"Missing {description}")

        # Check Australian specific requirements
        if "supplier_abn" not in parsed_data:
            validation_result["recommendations"].append(
                "ABN required for business expense claims over $82.50"
            )

        if "gst_amount" not in parsed_data:
            validation_result["recommendations"].append(
                "GST amount required for tax calculations"
            )

        # Validate date format
        if "invoice_date" in parsed_data:
            if not self._validate_australian_date(parsed_data["invoice_date"]):
                validation_result["validation_errors"].append(
                    "Date should be in DD/MM/YYYY format"
                )

        # Calculate compliance score
        compliance_score = len(present_fields) / len(required_fields)

        validation_result.update(
            {
                "compliance_score": compliance_score,
                "is_tax_compliant": compliance_score >= 0.8
                and len(validation_result["validation_errors"]) == 0,
                "required_fields_present": present_fields,
                "missing_fields": missing_fields,
            }
        )

        return validation_result
