"""Bank statement document type handler."""

import re
from typing import Any, Dict, List, Pattern

from .document_handlers import DocumentPattern, DocumentTypeHandler, ExtractionResult


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
                pattern=r"ACCOUNT_NUMBER:\s*([^\n\r]+)",
                field_name="ACCOUNT_NUMBER",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"BSB:\s*([^\n\r]+)",
                field_name="BSB",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"ACCOUNT_HOLDER:\s*([^\n\r]+)",
                field_name="ACCOUNT_HOLDER",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"STATEMENT_PERIOD:\s*([^\n\r]+)",
                field_name="STATEMENT_PERIOD",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"OPENING_BALANCE:\s*([^\n\r]+)",
                field_name="OPENING_BALANCE",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"CLOSING_BALANCE:\s*([^\n\r]+)",
                field_name="CLOSING_BALANCE",
                field_type="number",
                required=True,
            ),
            DocumentPattern(
                pattern=r"TOTAL_CREDITS:\s*([^\n\r]+)",
                field_name="TOTAL_CREDITS",
                field_type="number",
                required=False,
            ),
            DocumentPattern(
                pattern=r"TOTAL_DEBITS:\s*([^\n\r]+)",
                field_name="TOTAL_DEBITS",
                field_type="number",
                required=False,
            ),
            DocumentPattern(
                pattern=r"BANK_NAME:\s*([^\n\r]+)",
                field_name="BANK_NAME",
                field_type="string",
                required=True,
            ),
            DocumentPattern(
                pattern=r"TRANSACTION_COUNT:\s*([^\n\r]+)",
                field_name="TRANSACTION_COUNT",
                field_type="number",
                required=False,
            ),
            DocumentPattern(
                pattern=r"STATEMENT_DATE:\s*([^\n\r]+)",
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

    def extract_fields(self, response: str) -> ExtractionResult:
        """Extract fields with fallback pattern matching like TaxAuthorityParser.

        Args:
            response: Model response text

        Returns:
            Extraction result with fields and metadata
        """
        # First try the standard KEY-VALUE approach
        result = super().extract_fields(response)
        
        self.logger.info(f"KEY-VALUE extraction found {result.field_count} fields: {list(result.fields.keys())}")
        
        # If we found less than 7 fields (most of our 11 required fields), use fallback pattern matching
        if result.field_count < 7:
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
            required_found = sum(1 for p in required_patterns if p.field_name in normalized)
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
        
        This is the fallback mechanism that makes TaxAuthorityParser successful.
        
        Args:
            response: Raw model response text
            
        Returns:
            Extracted fields dictionary
        """
        extracted = {}
        
        # Extract account number - look for patterns like "Account Number: 435073466"
        account_patterns = [
            r"Account Number:\s*([0-9]+)",
            r"account number:\s*([0-9]+)", 
            r"Account:\s*([0-9]+)",
            r"Acc(?:ount)?[:\s]*([0-9]+)",
        ]
        
        for pattern in account_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted["ACCOUNT_NUMBER"] = match.group(1)
                break
        
        # Extract BSB - look for patterns like "BSB: 14-870"
        bsb_patterns = [
            r"BSB:\s*([0-9]{2,3}-[0-9]{3})",
            r"bsb:\s*([0-9]{2,3}-[0-9]{3})",
            r"BSB[:\s]*([0-9]{2,3}-[0-9]{3})",
        ]
        
        for pattern in bsb_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted["BSB"] = match.group(1)
                break
        
        # Extract account holder name - look for patterns like "Account Jennifer Liu"
        name_patterns = [
            r"Account\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+BSB",
            r"Account\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"Account Holder[:\s]*([A-Z][a-zA-Z\s]+)",
            r"Name[:\s]*([A-Z][a-zA-Z\s]+)",
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 3:  # Reasonable name length
                    extracted["ACCOUNT_HOLDER"] = name.upper()
                    break
        
        # Extract statement period from dates like "01/05/2025 31/05/2025"
        period_patterns = [
            r"([0-9]{2}/[0-9]{2}/[0-9]{4})\s+([0-9]{2}/[0-9]{2}/[0-9]{4})",
            r"from\s+([0-9]{2}/[0-9]{2}/[0-9]{4})\s+to\s+([0-9]{2}/[0-9]{2}/[0-9]{4})",
        ]
        
        for pattern in period_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted["STATEMENT_PERIOD"] = f"{match.group(1)} to {match.group(2)}"
                extracted["STATEMENT_DATE"] = match.group(2)  # End date as statement date
                break
        
        # Extract balances from patterns like "$4409.49" 
        balance_amounts = re.findall(r"\$([0-9]+\.[0-9]{2})", response)
        if len(balance_amounts) >= 2:
            # First balance found is likely opening, last is likely closing
            extracted["OPENING_BALANCE"] = balance_amounts[0]
            extracted["CLOSING_BALANCE"] = balance_amounts[-1]
        
        # Extract bank name
        bank_patterns = [
            r"(ANZ)\s+BANK",
            r"(COMMONWEALTH\s+BANK)",
            r"(WESTPAC)",
            r"(NAB)",
            r"(ANZ)",
        ]
        
        for pattern in bank_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted["BANK_NAME"] = match.group(1).upper()
                break
        
        self.logger.debug(f"Fallback extraction found {len(extracted)} fields: {list(extracted.keys())}")
        return extracted
