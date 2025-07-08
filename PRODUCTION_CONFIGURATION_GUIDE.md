# Production Configuration Guide: Key-Value Pairs and Named Entities

This guide explains how to customize the named entity keys and corresponding prompt configurations for your production environment.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Key Components to Modify](#key-components-to-modify)
3. [Step-by-Step Configuration](#step-by-step-configuration)
4. [Entity Key Mapping](#entity-key-mapping)
5. [Prompt File Modifications](#prompt-file-modifications)
6. [Parser Configuration](#parser-configuration)
7. [Testing and Validation](#testing-and-validation)
8. [Common Patterns](#common-patterns)
9. [Troubleshooting](#troubleshooting)

## 🎯 Overview

The system uses three main components for entity extraction:
1. **YAML Prompts** (`prompts.yaml`) - Define what the model should extract
2. **Tax Authority Parser** (`tax_authority_parser.py`) - Maps model responses to structured data
3. **Key-Value Extractor** (`key_value_extraction.py`) - Handles KEY:VALUE format parsing

## 🔧 Key Components to Modify

### Files You'll Need to Edit:
- `prompts.yaml` - Model prompts and expected output format
- `llama_vision/extraction/tax_authority_parser.py` - Response parsing logic
- `llama_vision/extraction/key_value_extraction.py` - KEY:VALUE pattern matching

## 📝 Step-by-Step Configuration

### Step 1: Define Your Production Entity Schema

First, decide on your production entity names. Common patterns:

```yaml
# Example Production Schema
VENDOR_NAME          # Instead of: STORE, BUSINESS_NAME, supplier_name
INVOICE_DATE         # Instead of: DATE, transaction_date
TOTAL_AMOUNT         # Instead of: TOTAL, total_amount
TAX_AMOUNT           # Instead of: GST, TAX, gst_amount
INVOICE_NUMBER       # Instead of: RECEIPT, receipt_number
PAYMENT_TYPE         # Instead of: PAYMENT_METHOD
PRODUCT_LIST         # Instead of: ITEMS, PRODUCTS
CUSTOMER_ID          # Instead of: PAYER, member_number
```

### Step 2: Update YAML Prompts

Edit `prompts.yaml` to use your production entity names:

#### Before (Current):
```yaml
key_value_receipt_prompt: |
  <|image|>
  Extract information from this Australian receipt and return in KEY-VALUE format.
  
  Use this exact format:
  DATE: [purchase date in DD/MM/YYYY format]
  STORE: [store name in capitals]
  ABN: [Australian Business Number - XX XXX XXX XXX format]
  PAYER: [customer/member name if visible]
  TAX: [GST amount]
  TOTAL: [total amount including GST]
  PRODUCTS: [item1 | item2 | item3]
```

#### After (Production Example):
```yaml
production_invoice_prompt: |
  <|image|>
  Extract information from this business invoice and return in KEY-VALUE format.
  
  Use this exact format:
  INVOICE_DATE: [invoice date in DD/MM/YYYY format]
  VENDOR_NAME: [supplier business name in capitals]
  ABN: [Australian Business Number - XX XXX XXX XXX format]
  CUSTOMER_ID: [customer reference if visible]
  TAX_AMOUNT: [GST/tax amount]
  TOTAL_AMOUNT: [total amount including tax]
  PRODUCT_LIST: [item1 | item2 | item3]
  INVOICE_NUMBER: [invoice reference number]
  PAYMENT_TYPE: [payment method used]
```

### Step 3: Update Tax Authority Parser

Edit `llama_vision/extraction/tax_authority_parser.py`:

#### A. Update KEY-VALUE Patterns

Find the `_parse_key_value_format` method and update the patterns:

```python
# In _parse_key_value_format method, update kv_patterns:
kv_patterns = [
    # OLD PATTERNS:
    # (r"DATE:\s*([^\n\r]+)", "DATE"),
    # (r"STORE:\s*([^\n\r]+)", "STORE"),
    # (r"TAX:\s*([^\n\r]+)", "TAX"),
    # (r"TOTAL:\s*([^\n\r]+)", "TOTAL"),
    # (r"PRODUCTS:\s*([^\n\r]+)", "PRODUCTS"),
    
    # NEW PRODUCTION PATTERNS:
    (r"INVOICE_DATE:\s*([^\n\r]+)", "INVOICE_DATE"),
    (r"VENDOR_NAME:\s*([^\n\r]+)", "VENDOR_NAME"),
    (r"TAX_AMOUNT:\s*([^\n\r]+)", "TAX_AMOUNT"),
    (r"TOTAL_AMOUNT:\s*([^\n\r]+)", "TOTAL_AMOUNT"),
    (r"PRODUCT_LIST:\s*([^\n\r]+)", "PRODUCT_LIST"),
    (r"INVOICE_NUMBER:\s*([^\n\r]+)", "INVOICE_NUMBER"),
    (r"PAYMENT_TYPE:\s*([^\n\r]+)", "PAYMENT_TYPE"),
    (r"CUSTOMER_ID:\s*([^\n\r]+)", "CUSTOMER_ID"),
    (r"ABN:\s*([^\n\r]+)", "ABN"),
]
```

#### B. Update Field Mappings

Update the field mapping logic in the same method:

```python
# Add production field mappings
if key == "VENDOR_NAME":
    parsed["supplier_name"] = value
    parsed["business_name"] = value
elif key == "INVOICE_DATE":
    if self._validate_australian_date(value):
        parsed["invoice_date"] = value
        parsed["transaction_date"] = value
elif key == "TAX_AMOUNT":
    # Extract numeric amount
    numeric_value = re.search(r"[\d.]+", value.replace("$", ""))
    if numeric_value:
        amount = numeric_value.group(0)
        parsed["gst_amount"] = amount
        parsed["tax_amount"] = amount
elif key == "TOTAL_AMOUNT":
    # Extract numeric amount  
    numeric_value = re.search(r"[\d.]+", value.replace("$", ""))
    if numeric_value:
        amount = numeric_value.group(0)
        parsed["total_amount"] = amount
elif key == "PRODUCT_LIST":
    # Split by pipe separator
    if "|" in value:
        items = [item.strip() for item in value.split("|") if item.strip()]
        parsed["items"] = items
        parsed["products"] = items
    else:
        parsed["items"] = [value]
        parsed["products"] = [value]
elif key == "INVOICE_NUMBER":
    parsed["receipt_number"] = value
    parsed["invoice_number"] = value
elif key == "PAYMENT_TYPE":
    parsed["payment_method"] = value
elif key == "CUSTOMER_ID":
    parsed["customer_reference"] = value
    parsed["payer_name"] = value
```

#### C. Update Compliance Fields

Update the `_calculate_compliance_score` method:

```python
def _calculate_compliance_score(self, parsed: Dict[str, Any]) -> float:
    """Calculate tax authority compliance score."""
    
    # Update required fields for your production environment
    required_fields = [
        "supplier_name",    # Maps from VENDOR_NAME
        "invoice_date",     # Maps from INVOICE_DATE  
        "total_amount",     # Maps from TOTAL_AMOUNT
    ]

    compliance_fields = [
        "supplier_abn",     # Maps from ABN
        "gst_amount",       # Maps from TAX_AMOUNT
        "payment_method",   # Maps from PAYMENT_TYPE
    ]
    
    # Rest of the method remains the same...
```

### Step 4: Update Key-Value Extractor

Edit `llama_vision/extraction/key_value_extraction.py`:

#### Update the patterns in the `extract` method:

```python
# Standard KEY-VALUE patterns (update for production)
kv_patterns = [
    # OLD:
    # (r"DATE:\s*([^\n\r]+)", "DATE"),
    # (r"STORE:\s*([^\n\r]+)", "STORE"),
    
    # NEW PRODUCTION:
    (r"INVOICE_DATE:\s*([^\n\r]+)", "INVOICE_DATE"),
    (r"VENDOR_NAME:\s*([^\n\r]+)", "VENDOR_NAME"),
    (r"TAX_AMOUNT:\s*([^\n\r]+)", "TAX_AMOUNT"),
    (r"TOTAL_AMOUNT:\s*([^\n\r]+)", "TOTAL_AMOUNT"),
    (r"PRODUCT_LIST:\s*([^\n\r]+)", "PRODUCT_LIST"),
    (r"INVOICE_NUMBER:\s*([^\n\r]+)", "INVOICE_NUMBER"),
    (r"PAYMENT_TYPE:\s*([^\n\r]+)", "PAYMENT_TYPE"),
    (r"CUSTOMER_ID:\s*([^\n\r]+)", "CUSTOMER_ID"),
    (r"ABN:\s*([^\n\r]+)", "ABN"),
]
```

#### Update the field aliases in `_add_derived_fields`:

```python
# Update field aliases for production
field_aliases = {
    "VENDOR_NAME": ["supplier_name", "business_name", "vendor"],
    "INVOICE_DATE": ["invoice_date", "transaction_date", "date"],
    "TOTAL_AMOUNT": ["total_amount", "total", "amount_total"],
    "TAX_AMOUNT": ["gst_amount", "tax_amount", "tax"],
    "ABN": ["supplier_abn", "business_abn", "abn"],
    "CUSTOMER_ID": ["customer_reference", "payer_name", "customer"],
    "PAYMENT_TYPE": ["payment_method", "payment"],
    "PRODUCT_LIST": ["items", "products"],
    "INVOICE_NUMBER": ["invoice_number", "receipt_number"],
}
```

## 🔗 Entity Key Mapping Reference

### Current System → Production Mapping Examples

| Current Key | Production Key | Internal Mapping | Description |
|-------------|----------------|------------------|-------------|
| `STORE` | `VENDOR_NAME` | `supplier_name` | Business/supplier name |
| `DATE` | `INVOICE_DATE` | `invoice_date` | Transaction date |
| `TOTAL` | `TOTAL_AMOUNT` | `total_amount` | Total amount including tax |
| `TAX/GST` | `TAX_AMOUNT` | `gst_amount` | Tax/GST amount |
| `PRODUCTS` | `PRODUCT_LIST` | `items` | List of products/services |
| `RECEIPT` | `INVOICE_NUMBER` | `invoice_number` | Document reference |
| `PAYMENT_METHOD` | `PAYMENT_TYPE` | `payment_method` | Payment method |
| `PAYER` | `CUSTOMER_ID` | `customer_reference` | Customer identifier |

## 📂 Common Production Patterns

### Enterprise Pattern:
```yaml
SUPPLIER_NAME:       # Business entity
INVOICE_REF:         # Document reference
INVOICE_DATE:        # Document date
LINE_ITEMS:          # Product details
NET_AMOUNT:          # Amount before tax
TAX_AMOUNT:          # Tax component
GROSS_AMOUNT:        # Total amount
PAYMENT_TERMS:       # Payment method/terms
CUSTOMER_REF:        # Customer reference
```

### Government/Compliance Pattern:
```yaml
ENTITY_NAME:         # Legal entity name
ENTITY_ABN:          # Australian Business Number
DOCUMENT_DATE:       # Official date
DOCUMENT_NUMBER:     # Reference number
TAXABLE_AMOUNT:      # Amount subject to tax
GST_AMOUNT:          # GST component
TOTAL_AMOUNT:        # Total inclusive amount
PAYMENT_METHOD:      # Payment type
RECIPIENT_ID:        # Recipient identifier
```

### Simple Business Pattern:
```yaml
BUSINESS:            # Business name
DATE:                # Transaction date
AMOUNT:              # Total amount
TAX:                 # Tax amount
REFERENCE:           # Document reference
CUSTOMER:            # Customer details
```

## 🧪 Testing and Validation

### Step 1: Test Configuration Loading
```bash
# Test that prompts load correctly
python -c "
from llama_vision.config import PromptManager
pm = PromptManager()
print('Available prompts:', pm.list_prompts())
print('Production prompt:', pm.get_prompt('production_invoice_prompt')[:100])
"
```

### Step 2: Test Entity Extraction
```bash
# Test with your production prompt
python -m llama_vision.cli.llama_single extract test_invoice.jpg \
  --prompt-name production_invoice_prompt \
  --extraction-method tax_authority \
  --verbose
```

### Step 3: Verify Field Mapping
```python
# Test script to verify mapping
from llama_vision.extraction import TaxAuthorityParser

parser = TaxAuthorityParser()
test_response = """
VENDOR_NAME: ACME CORPORATION
INVOICE_DATE: 15/06/2024
TOTAL_AMOUNT: 150.00
TAX_AMOUNT: 13.64
"""

result = parser.parse_receipt_response(test_response)
print("Extracted fields:", list(result.keys()))
print("Vendor name:", result.get('supplier_name'))
print("Total amount:", result.get('total_amount'))
```

## ⚠️ Common Issues and Solutions

### Issue 1: Prompts Not Found
**Error**: `KeyError: 'production_invoice_prompt'`
**Solution**: Check prompt name spelling in `prompts.yaml` and CLI command

### Issue 2: No Fields Extracted
**Error**: Only compliance fields returned (currency, country, etc.)
**Solution**: Check regex patterns match your prompt format exactly

### Issue 3: Wrong Field Names in Output
**Error**: Getting old field names instead of production names
**Solution**: Update the field mapping section in `_parse_key_value_format`

### Issue 4: Date Validation Fails
**Error**: Dates not being extracted
**Solution**: Check `_validate_australian_date` method accepts your date format

## 🔍 Debugging Tips

### 1. Enable Verbose Logging
```bash
python -m llama_vision.cli.llama_single extract invoice.jpg --verbose
```

### 2. Check Raw Model Response
Add debug printing in `tax_authority_parser.py`:
```python
def parse_receipt_response(self, response: str) -> Dict[str, Any]:
    print("DEBUG: Raw model response:")
    print(response)
    print("=" * 50)
    # ... rest of method
```

### 3. Test Pattern Matching
```python
import re
response = "VENDOR_NAME: ACME CORP"
pattern = r"VENDOR_NAME:\s*([^\n\r]+)"
match = re.search(pattern, response, re.IGNORECASE)
print("Match found:", match.group(1) if match else "No match")
```

## 📋 Quick Reference Checklist

- [ ] Define production entity schema
- [ ] Update prompt in `prompts.yaml`
- [ ] Update KEY-VALUE patterns in `tax_authority_parser.py`
- [ ] Update field mappings in `tax_authority_parser.py`
- [ ] Update compliance fields calculation
- [ ] Update patterns in `key_value_extraction.py`
- [ ] Update field aliases in `key_value_extraction.py`
- [ ] Test with sample documents
- [ ] Verify field names in output
- [ ] Test compliance scoring

## 🚀 Deployment Notes

1. **Backup Current Configuration**: Make copies of working files before changes
2. **Test Incrementally**: Change one component at a time and test
3. **Document Changes**: Keep notes of your specific production mappings
4. **Validate Compliance**: Ensure tax compliance requirements are met
5. **Performance Test**: Verify speed and accuracy with production data

This guide should provide everything you need to configure the system for your production environment. Keep this document as a reference for future modifications and troubleshooting.