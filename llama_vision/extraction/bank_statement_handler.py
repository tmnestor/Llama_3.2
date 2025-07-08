"""Bank statement document type handler."""

import re
from typing import Dict, List, Pattern

from .document_handlers import DocumentPattern, DocumentTypeHandler


class BankStatementHandler(DocumentTypeHandler):
    """Handler for bank statement documents."""

    @property
    def document_type(self) -> str:
        return "bank_statement"

    @property
    def display_name(self) -> str:
        return "Bank Statement"

    def get_classification_indicators(self) -> List[str]:
        """Get text indicators for bank statements."""
        return [
            "account",
            "balance",
            "transaction",
            "deposit",
            "withdrawal",
            "bsb",
            "opening balance",
            "closing balance",
            "statement period",
            "account number",
            "sort code",
            "debit",
            "credit",
            "available balance",
            "current balance",
            "commonwealth bank",
            "westpac",
            "anz",
            "nab",
            "statement",
        ]

    def get_classification_patterns(self) -> List[Pattern]:
        """Get regex patterns for bank statement classification."""
        return [
            re.compile(
                r"\d{3}-\d{3}\s+\d{4,10}", re.IGNORECASE
            ),  # BSB + Account format
            re.compile(r"\bbsb\b", re.IGNORECASE),  # BSB code
            re.compile(r"\baccount\s+number\b", re.IGNORECASE),  # Account number
            re.compile(r"opening\s+balance", re.IGNORECASE),  # Opening balance
            re.compile(r"closing\s+balance", re.IGNORECASE),  # Closing balance
            re.compile(r"statement\s+period", re.IGNORECASE),  # Statement period
        ]

    def get_prompt_name(self) -> str:
        """Get prompt name for bank statements."""
        return "bank_statement_extraction_prompt"

    def get_field_patterns(self) -> List[DocumentPattern]:
        """Get field extraction patterns for bank statements."""
        return [
            DocumentPattern(
                pattern=r"ACCOUNT_NUMBER:\s*([\d]+)",
                field_name="ACCOUNT_NUMBER",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"BSB:\s*([\d-]+)",
                field_name="BSB",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"ACCOUNT_HOLDER:\s*([A-Z][A-Z\s]*?)(?=\n|$|[A-Z_]+:)",
                field_name="ACCOUNT_HOLDER",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"STATEMENT_PERIOD:\s*([\d/]+\s+to\s+[\d/]+)",
                field_name="STATEMENT_PERIOD",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"OPENING_BALANCE:\s*([\d.]+)",
                field_name="OPENING_BALANCE",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"CLOSING_BALANCE:\s*([\d.]+)",
                field_name="CLOSING_BALANCE",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"TOTAL_CREDITS:\s*([\d.]+)",
                field_name="TOTAL_CREDITS",
                field_type="number",
                required=False,
            ),
            DocumentPattern(
                pattern=r"TOTAL_DEBITS:\s*([\d.]+)",
                field_name="TOTAL_DEBITS",
                field_type="number",
                required=False,
            ),
            DocumentPattern(
                pattern=r"BANK_NAME:\s*([A-Z][A-Z\s&]*?)(?=\n|$|[A-Z_]+:)",
                field_name="BANK_NAME",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"TRANSACTION_COUNT:\s*([\d]+)",
                field_name="TRANSACTION_COUNT",
                field_type="number",
                required=False,
            ),
            DocumentPattern(
                pattern=r"STATEMENT_DATE:\s*([\d/]+)",
                field_name="STATEMENT_DATE",
                field_type="date",
                required=True,
            ),
        ]

    def get_field_mappings(self) -> Dict[str, List[str]]:
        """Get field mappings for standardization."""
        return {
            # Standard compliance fields (map to tax authority standards)
            "supplier_name": ["BANK_NAME"],
            "total_amount": ["CLOSING_BALANCE"],
            "invoice_date": ["STATEMENT_DATE"],
            "payment_method": [],  # Not applicable for bank statements
            # Bank-specific fields
            "account_number": ["ACCOUNT_NUMBER"],
            "bsb": ["BSB"],
            "account_holder": ["ACCOUNT_HOLDER"],
            "account_holder_name": ["ACCOUNT_HOLDER"],
            "customer_name": ["ACCOUNT_HOLDER"],
            "statement_period": ["STATEMENT_PERIOD"],
            "opening_balance": ["OPENING_BALANCE"],
            "closing_balance": ["CLOSING_BALANCE"],
            "total_credits": ["TOTAL_CREDITS"],
            "total_debits": ["TOTAL_DEBITS"],
            "transaction_count": ["TRANSACTION_COUNT"],
            "statement_date": ["STATEMENT_DATE"],
            "institution_name": ["BANK_NAME"],
            # Transaction date for compliance
            "transaction_date": ["STATEMENT_DATE"],
        }
