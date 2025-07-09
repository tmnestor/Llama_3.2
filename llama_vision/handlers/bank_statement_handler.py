"""
Bank Statement Handler for Australian Tax Document Processing

This handler specializes in processing bank statements for expense verification
with ATO compliance validation.
"""

import re
from typing import Any, Dict, List

from ..extraction.australian_tax_classifier import DocumentType
from ..utils import setup_logging
from .base_ato_handler import BaseATOHandler

logger = setup_logging()


class BankStatementHandler(BaseATOHandler):
    """Handler for Australian bank statement processing with ATO compliance."""

    def __init__(self):
        super().__init__(DocumentType.BANK_STATEMENT)
        logger.info("BankStatementHandler initialized for expense verification")

    def _extract_fields_primary(self, document_text: str) -> Dict[str, Any]:
        """Extract fields using primary extraction method for bank statements."""
        extracted_fields = {}

        # Extract bank name
        bank_patterns = [
            r"(ANZ|COMMONWEALTH BANK|WESTPAC|NAB|ING|MACQUARIE|BENDIGO BANK|SUNCORP)",
            r"([A-Z\s]+BANK)",
            r"([A-Z\s]+CREDIT UNION)",
        ]

        for pattern in bank_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["bank_name"] = match.group(1).strip().upper()
                break

        # Extract account holder name
        holder_patterns = [
            r"(?:account holder|customer|name)[\s:]*([A-Za-z\s]+)",
            r"(?:mr|ms|mrs|dr)\s+([A-Za-z\s]+)",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+)",
        ]

        for pattern in holder_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["account_holder"] = match.group(1).strip()
                break

        # Extract account number (masked)
        account_patterns = [
            r"account[\s#:]*(\d{3,4}[\*x]{3,6}\d{3,4})",
            r"(\d{3,4}[\*x]{3,6}\d{3,4})",
            r"account[\s#:]*(\d{6,12})",
        ]

        for pattern in account_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                account_num = match.group(1)
                # Mask middle digits for security
                if len(account_num) > 6 and "*" not in account_num:
                    masked = account_num[:3] + "***" + account_num[-3:]
                    extracted_fields["account_number"] = masked
                else:
                    extracted_fields["account_number"] = account_num
                break

        # Extract BSB
        bsb_patterns = [
            r"bsb[\s:]*(\d{2,3}-\d{3})",
            r"(\d{2,3}-\d{3})",
            r"bsb[\s:]*(\d{6})",
        ]

        for pattern in bsb_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                bsb = match.group(1)
                if "-" not in bsb and len(bsb) == 6:
                    bsb = f"{bsb[:3]}-{bsb[3:]}"
                extracted_fields["bsb"] = bsb
                break

        # Extract statement period
        period_patterns = [
            r"(?:statement period|period)[\s:]*(\d{1,2}/\d{1,2}/\d{4})\s*(?:to|-)\s*(\d{1,2}/\d{1,2}/\d{4})",
            r"(\d{1,2}/\d{1,2}/\d{4})\s*(?:to|-)\s*(\d{1,2}/\d{1,2}/\d{4})",
        ]

        for pattern in period_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                start_date = match.group(1)
                end_date = match.group(2)
                extracted_fields["statement_period"] = f"{start_date} - {end_date}"
                break

        # Extract opening balance
        opening_patterns = [
            r"(?:opening balance|opening|balance forward)[\s:]*\$?(\d+\.\d{2})",
            r"(?:brought forward|b/f)[\s:]*\$?(\d+\.\d{2})",
        ]

        for pattern in opening_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["opening_balance"] = match.group(1)
                break

        # Extract closing balance
        closing_patterns = [
            r"(?:closing balance|closing|balance)[\s:]*\$?(\d+\.\d{2})",
            r"(?:carried forward|c/f)[\s:]*\$?(\d+\.\d{2})",
        ]

        for pattern in closing_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_fields["closing_balance"] = match.group(1)
                break

        # Extract transactions
        transactions = self._extract_transactions(document_text)
        if transactions:
            extracted_fields["transactions"] = transactions

        # Extract work-related transactions
        work_transactions = self._extract_work_related_transactions(document_text)
        if work_transactions:
            extracted_fields["work_transactions"] = work_transactions

        logger.debug(
            f"Primary extraction yielded {len(extracted_fields)} fields for bank statement"
        )
        return extracted_fields

    def _extract_transactions(self, document_text: str) -> List[Dict[str, Any]]:
        """Extract transaction details from bank statement."""
        transactions = []
        lines = document_text.split("\n")

        # Transaction patterns
        transaction_patterns = [
            r"(\d{1,2}/\d{1,2}/\d{4})\s+([A-Za-z\s\-\*]+)\s+(\d+\.\d{2})\s+(\d+\.\d{2})",  # date, desc, amount, balance
            r"(\d{1,2}/\d{1,2})\s+([A-Za-z\s\-\*]+)\s+(\d+\.\d{2})\s+(\d+\.\d{2})",  # date, desc, amount, balance
        ]

        for line in lines:
            line = line.strip()

            for pattern in transaction_patterns:
                match = re.search(pattern, line)
                if match:
                    date = match.group(1)
                    description = match.group(2).strip()
                    amount = match.group(3)
                    balance = match.group(4)

                    # Determine if debit or credit
                    transaction_type = self._determine_transaction_type(
                        description, amount
                    )

                    # Assess work relevance
                    work_relevance = self._assess_work_relevance(description)

                    transaction = {
                        "date": date,
                        "description": description,
                        "amount": amount,
                        "balance": balance,
                        "type": transaction_type,
                        "work_relevance": work_relevance,
                    }

                    transactions.append(transaction)
                    break

        return transactions

    def _extract_work_related_transactions(
        self, document_text: str
    ) -> List[Dict[str, Any]]:
        """Extract work-related transactions specifically."""
        work_transactions = []

        # Work-related merchant patterns
        work_merchants = [
            r"(OFFICEWORKS|HARVEY NORMAN|JB HI-FI)",  # Office supplies
            r"(BP|SHELL|CALTEX|AMPOL|MOBIL|7-ELEVEN)",  # Fuel
            r"(QANTAS|JETSTAR|VIRGIN|TIGERAIR)",  # Airlines
            r"(HILTON|MARRIOTT|HYATT|IBIS|MERCURE)",  # Hotels
            r"(UBER|TAXI|PARKING|TOLL)",  # Transport
            r"(DELOITTE|PWC|KPMG|LEGAL|ACCOUNTING)",  # Professional services
        ]

        lines = document_text.split("\n")

        for line in lines:
            line_upper = line.upper()

            for pattern in work_merchants:
                if re.search(pattern, line_upper):
                    # Extract transaction details
                    transaction_match = re.search(
                        r"(\d{1,2}/\d{1,2}/\d{4})\s+([A-Za-z\s\-\*]+)\s+(\d+\.\d{2})",
                        line,
                    )
                    if transaction_match:
                        work_transaction = {
                            "date": transaction_match.group(1),
                            "description": transaction_match.group(2).strip(),
                            "amount": transaction_match.group(3),
                            "work_relevance": "High",
                            "category": self._categorize_work_expense(
                                transaction_match.group(2)
                            ),
                        }
                        work_transactions.append(work_transaction)
                    break

        return work_transactions

    def _determine_transaction_type(self, description: str, _amount: str) -> str:
        """Determine if transaction is debit or credit."""
        desc_lower = description.lower()

        # Credit indicators
        credit_indicators = [
            "deposit",
            "credit",
            "salary",
            "interest",
            "refund",
            "transfer in",
        ]
        if any(indicator in desc_lower for indicator in credit_indicators):
            return "credit"

        # Debit indicators
        debit_indicators = [
            "withdrawal",
            "debit",
            "purchase",
            "payment",
            "fee",
            "transfer out",
        ]
        if any(indicator in desc_lower for indicator in debit_indicators):
            return "debit"

        return "debit"  # Default assumption

    def _assess_work_relevance(self, description: str) -> str:
        """Assess work relevance of transaction."""
        desc_lower = description.lower()

        # High relevance
        high_relevance = [
            "officeworks",
            "fuel",
            "petrol",
            "hotel",
            "airline",
            "parking",
            "taxi",
            "uber",
            "legal",
            "accounting",
        ]
        if any(term in desc_lower for term in high_relevance):
            return "High"

        # Medium relevance
        medium_relevance = [
            "restaurant",
            "coffee",
            "meal",
            "training",
            "conference",
            "equipment",
            "supplies",
        ]
        if any(term in desc_lower for term in medium_relevance):
            return "Medium"

        # Low relevance
        low_relevance = ["purchase", "retail", "shopping", "subscription", "service"]
        if any(term in desc_lower for term in low_relevance):
            return "Low"

        return "None"

    def _categorize_work_expense(self, description: str) -> str:
        """Categorize work expense type."""
        desc_lower = description.lower()

        # Expense categories
        if any(
            term in desc_lower for term in ["fuel", "petrol", "bp", "shell", "caltex"]
        ):
            return "vehicle_expenses"
        elif any(
            term in desc_lower
            for term in ["hotel", "accommodation", "travel", "airline"]
        ):
            return "travel_expenses"
        elif any(
            term in desc_lower
            for term in ["officeworks", "supplies", "equipment", "computer"]
        ):
            return "office_expenses"
        elif any(
            term in desc_lower for term in ["restaurant", "meal", "coffee", "catering"]
        ):
            return "meal_entertainment"
        elif any(
            term in desc_lower
            for term in ["legal", "accounting", "professional", "consultant"]
        ):
            return "professional_services"
        elif any(term in desc_lower for term in ["parking", "toll", "taxi", "uber"]):
            return "transport"
        elif any(
            term in desc_lower for term in ["training", "conference", "education"]
        ):
            return "training_development"
        else:
            return "other"

    def _get_required_fields(self) -> List[str]:
        """Get required fields for bank statement processing."""
        return ["bank_name", "account_holder", "statement_period"]

    def _get_optional_fields(self) -> List[str]:
        """Get optional fields for bank statement processing."""
        return [
            "account_number",
            "bsb",
            "opening_balance",
            "closing_balance",
            "transactions",
            "work_transactions",
        ]

    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for bank statement fields."""
        return {
            "statement_period": self._validate_date_range,
            "account_number": self._validate_account_number,
            "bsb": self._validate_bsb_format,
            "opening_balance": self._validate_currency_amount,
            "closing_balance": self._validate_currency_amount,
            "bank_name": self._validate_bank_name,
            "account_holder": self._validate_account_holder,
        }

    def _get_ato_thresholds(self) -> Dict[str, Any]:
        """Get ATO-specific thresholds for bank statements."""
        return {
            "work_expense_threshold": 300.00,
            "substantiation_required": 82.50,
            "high_value_transaction": 1000.00,
            "business_use_percentage": 10,
        }

    def _get_awk_rules(self) -> List[Dict[str, Any]]:
        """Get AWK extraction rules for bank statements."""
        return [
            {
                "field": "bank_name",
                "pattern": r"(ANZ|COMMONWEALTH BANK|WESTPAC|NAB|ING|MACQUARIE)",
                "line_filter": lambda line: any(
                    bank in line.upper()
                    for bank in [
                        "ANZ",
                        "COMMONWEALTH",
                        "WESTPAC",
                        "NAB",
                        "ING",
                        "MACQUARIE",
                    ]
                ),
                "transform": lambda x: x.strip().upper(),
            },
            {
                "field": "account_holder",
                "pattern": r"([A-Za-z\s]+)",
                "line_filter": lambda line: any(
                    word in line.lower()
                    for word in ["account holder", "customer", "name"]
                ),
                "transform": lambda x: x.strip().title(),
            },
            {
                "field": "bsb",
                "pattern": r"(\d{2,3}-\d{3})",
                "line_filter": lambda line: "bsb" in line.lower(),
                "transform": lambda x: x.strip(),
            },
            {
                "field": "statement_period",
                "pattern": r"(\d{1,2}/\d{1,2}/\d{4})\s*(?:to|-)\s*(\d{1,2}/\d{1,2}/\d{4})",
                "line_filter": lambda line: any(
                    word in line.lower()
                    for word in ["statement period", "period", "to"]
                ),
                "transform": lambda x: x.strip(),
            },
        ]

    def _validate_date_range(self, date_range_str: str) -> str:
        """Validate date range format."""
        if not date_range_str:
            return date_range_str

        # Extract start and end dates
        date_match = re.search(
            r"(\d{1,2}/\d{1,2}/\d{4})\s*(?:to|-)\s*(\d{1,2}/\d{1,2}/\d{4})",
            date_range_str,
        )
        if date_match:
            start_date = date_match.group(1)
            end_date = date_match.group(2)
            return f"{start_date} - {end_date}"

        return date_range_str

    def _validate_account_number(self, account_str: str) -> str:
        """Validate account number format."""
        if not account_str:
            return account_str

        # Ensure account number is masked for security
        if len(account_str) > 6 and "*" not in account_str:
            masked = account_str[:3] + "***" + account_str[-3:]
            return masked

        return account_str

    def _validate_bsb_format(self, bsb_str: str) -> str:
        """Validate BSB format."""
        if not bsb_str:
            return bsb_str

        # Format as XXX-XXX
        clean_bsb = re.sub(r"[^\d]", "", bsb_str)
        if len(clean_bsb) == 6:
            return f"{clean_bsb[:3]}-{clean_bsb[3:]}"

        return bsb_str

    def _validate_currency_amount(self, amount_str: str) -> str:
        """Validate currency amount format."""
        if not amount_str:
            return amount_str

        # Remove currency symbols and validate
        clean_amount = re.sub(r"[^\d.\-]", "", amount_str)

        try:
            amount = float(clean_amount)
            return f"{amount:.2f}"
        except ValueError:
            logger.warning(f"Invalid currency amount format: {amount_str}")
            return amount_str

    def _validate_bank_name(self, bank_str: str) -> str:
        """Validate bank name format."""
        if not bank_str:
            return bank_str

        # Known Australian banks
        known_banks = [
            "ANZ",
            "COMMONWEALTH BANK",
            "WESTPAC",
            "NAB",
            "ING",
            "MACQUARIE",
            "BENDIGO BANK",
            "SUNCORP",
            "BANK OF QUEENSLAND",
        ]

        bank_upper = bank_str.upper()
        for bank in known_banks:
            if bank in bank_upper:
                return bank

        return bank_str.strip().upper()

    def _validate_account_holder(self, holder_str: str) -> str:
        """Validate account holder name format."""
        if not holder_str:
            return holder_str

        # Clean and format name
        clean_holder = re.sub(r"[^\w\s]", "", holder_str)
        return clean_holder.strip().title()
