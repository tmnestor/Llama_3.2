# Document Type Classification and Extraction Guide

## Overview

This guide documents the successful implementation of document type-based extraction following the InternVL PoC architecture pattern. The system first classifies documents, then selects optimal prompts for extraction, resulting in significantly improved field extraction and compliance scores.

## Architecture Pattern

### 1. Classification-First Approach
```
Document → Classify Type → Select Optimal Prompt → Extract with Type-Specific Logic → Parse Results
```

This replaces the previous single-prompt approach with intelligent document-aware processing.

### 2. Core Components

#### Document Classification (`classify_document`)
- **Location**: `llama_vision/model/inference.py`
- **Prompt**: `document_classification_prompt` in `prompts.yaml`
- **Output**: Document type + confidence score + classification response

#### Smart Prompt Selection (`get_prompt_for_document_type`)
- **Location**: `llama_vision/config/prompts.py`
- **Logic**: Content-aware selection based on document type and OCR content
- **Special Handling**: Australian tax invoices with fuel content detection

#### Type-Specific Extraction
- **Location**: Various parsers in `llama_vision/extraction/`
- **Method**: Uses document-optimized prompts with standardized field names

## Implementation Lessons Learned

### 1. Australian Business Document Context

**Critical Understanding**: In Australia, "Tax Invoice" and "Tax Receipt" are synonymous. Many business documents (including fuel receipts) are technically tax invoices but require specialized extraction.

**Solution**: Content-aware prompt selection that detects fuel indicators in tax invoices and routes to fuel-specific extraction.

```python
# Smart routing example
if document_type == "tax_invoice" and classification_response:
    fuel_indicators = ['costco', 'ulp', 'unleaded', 'diesel', 'litre', ' l ', 'fuel', 'petrol']
    if any(indicator in response_lower for indicator in fuel_indicators):
        return self.get_prompt("fuel_receipt_extraction_prompt")
```

### 2. Prompt Design Patterns

#### Classification Prompt Requirements
- **Focus on structure analysis** rather than business content to avoid privacy concerns
- **Specific indicators** for each document type
- **Technical format analysis** rather than company information

```yaml
document_classification_prompt: |
  <|image|>Analyze the document structure and format. Classify based on layout patterns:
  
  - fuel_receipt: Contains fuel quantities (L, litres), price per unit, fuel product codes
  - tax_invoice: Contains formal invoice formatting, line items, tax calculations 
  - receipt: Contains product lists, quantities, prices in retail format
```

#### Extraction Prompt Success Pattern
Based on successful `key_value_receipt_prompt`, effective prompts require:

1. **Concrete Examples**: Real data examples prevent repetitive generation
2. **Clear Format**: "Use this exact format:" with field specifications
3. **Detailed Requirements**: Specific formatting rules and expectations
4. **Clean Termination**: "Return ONLY the key-value pairs above. No explanations."

```yaml
fuel_receipt_extraction_prompt: |
  <|image|>
  Extract information from this Australian fuel receipt and return in KEY-VALUE format.
  
  Use this exact format:
  DATE: [purchase date in DD/MM/YYYY format]
  STORE: [fuel station name in capitals]
  # ... field definitions
  
  Example for fuel receipt:
  DATE: 08/06/2024
  STORE: COSTCO WHOLESALE AUSTRALIA
  # ... real example data
  
  FORMATTING REQUIREMENTS:
  - Store names: Use ALL CAPITALS
  # ... specific rules
  
  Return ONLY the key-value pairs above. No explanations.
```

### 3. Common Pitfalls and Solutions

#### Privacy Blocking
**Problem**: Model refuses to analyze business documents due to privacy concerns
**Solution**: Frame as "document structure analysis" rather than business content analysis

#### Repetitive Generation
**Problem**: Model generates repetitive "if available" text instead of extraction
**Solution**: Provide concrete examples and clear termination instructions

#### Field Name Incompatibility
**Problem**: Prompt uses different field names than parser expects
**Solution**: Standardize field names across all prompts to match parser expectations

#### Poor Classification
**Problem**: Documents misclassified due to formal appearance
**Solution**: OCR content analysis to override initial classification

## Document Type Extensions

### Template for New Document Types

#### 1. Add Classification Logic
```python
# In inference.py classify_document method
elif "document_indicator" in response_lower:
    doc_type = "new_document_type"
    confidence = 0.85
```

#### 2. Create Type-Specific Prompt
```yaml
new_document_extraction_prompt: |
  <|image|>
  Extract information from this [document type] and return in KEY-VALUE format.
  
  Use this exact format:
  DATE: [transaction date]
  # ... type-specific fields
  
  Example:
  DATE: 01/01/2024
  # ... realistic example data
  
  FORMATTING REQUIREMENTS:
  # ... specific rules
  
  Return ONLY the key-value pairs above. No explanations.
```

#### 3. Add Document Type Mapping
```yaml
# In prompts.yaml metadata section
document_type_mapping:
  new_document_type: new_document_extraction_prompt
```

#### 4. Add Content-Aware Logic (if needed)
```python
# In prompts.py get_prompt_for_document_type
if document_type == "tax_invoice" and classification_response:
    indicators = ['specific', 'keywords']
    if any(indicator in response_lower for indicator in indicators):
        return self.get_prompt("specific_extraction_prompt")
```

### Planned Document Types

#### Travel Receipts
**Indicators**: hotel, airline, travel, booking, accommodation
**Key Fields**: DATE, VENDOR, TOTAL, TAX, TRAVEL_TYPE, DESTINATION, DURATION
**Special Logic**: Detect travel vs general receipts

#### Parking Receipts
**Indicators**: parking, meter, garage, hourly, zone
**Key Fields**: DATE, LOCATION, DURATION, RATE, TOTAL, VEHICLE_REG
**Special Logic**: Time-based vs flat-rate parking

#### Bank Statements
**Indicators**: account, balance, transaction, deposit, withdrawal
**Key Fields**: ACCOUNT_NUMBER, PERIOD, OPENING_BALANCE, CLOSING_BALANCE, TRANSACTIONS
**Special Logic**: Multi-transaction parsing

#### Meal/Entertainment Receipts
**Indicators**: restaurant, cafe, meal, dining, entertainment
**Key Fields**: DATE, VENUE, TOTAL, TAX, ATTENDEES, PURPOSE
**Special Logic**: Business meal compliance requirements

#### Office Supply Receipts
**Indicators**: stationery, office, supplies, equipment
**Key Fields**: DATE, SUPPLIER, ITEMS, QUANTITIES, PRICES, TOTAL
**Special Logic**: Bulk item processing

## CLI Usage

### Smart Extraction (Always Classifies First)
```bash
python -m llama_vision.cli.llama_single smart-extract /path/to/document.jpg
```

### Manual Extraction with Classification
```bash
python -m llama_vision.cli.llama_single extract /path/to/document.jpg --use-document-classification
```

### Test Classification Only
```bash
python -m llama_vision.cli.llama_single classify /path/to/document.jpg
```

## Performance Results

### Before Document Type Classification
- **Fields Extracted**: 4 basic fields
- **Compliance Score**: 0.00
- **Method**: Single generic prompt for all documents

### After Document Type Classification
- **Fields Extracted**: 22 comprehensive fields
- **Compliance Score**: 0.77
- **Method**: Document-aware prompt selection with type-specific extraction

## Best Practices

### 1. Prompt Development
- Always include concrete examples with realistic data
- Use "Return ONLY key-value pairs" to prevent repetitive generation
- Standardize field names across all document types
- Test with verbose logging to see raw model responses

### 2. Classification Tuning
- Focus on structural indicators rather than business content
- Use OCR content analysis for complex cases
- Implement content-aware routing for ambiguous documents

### 3. Parser Compatibility
- Ensure all prompts use compatible field names
- Test extraction with TaxAuthorityParser or relevant parser
- Validate compliance scoring works correctly

### 4. Testing Workflow
1. Test classification accuracy first
2. Verify prompt generates clean KEY-VALUE output
3. Confirm parser extracts expected fields
4. Validate compliance scoring

## Future Enhancements

### 1. Machine Learning Classification
Replace rule-based classification with ML model trained on document features.

### 2. Dynamic Prompt Selection
Use confidence scores to select between multiple prompt options per document type.

### 3. Multi-Language Support
Extend classification and extraction to support multiple languages and regions.

### 4. Template Learning
Automatically learn new document templates from user corrections.

---

**Success Metrics**: The document type classification system successfully increased field extraction from 4 to 22 fields and compliance scores from 0.00 to 0.77 for fuel receipts, demonstrating the effectiveness of the classification-first approach.