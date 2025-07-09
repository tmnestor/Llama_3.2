# AWK Extractor Guide: llama_vision/extraction/awk_extractor.py

## Overview

The `awk_extractor.py` script implements an **AWK-style text processing system in Python** that provides maintainable field extraction from documents. It simulates AWK's line-by-line processing paradigm while offering more integration with the Python ecosystem.

## Architecture

### Core Classes

1. **`AwkExtractor`** - Base class implementing AWK-style processing
2. **`FuelReceiptAwkExtractor`** - Specialized fuel receipt processor
3. **`BankStatementAwkExtractor`** - Specialized bank statement processor

## How It Works

### 1. AWK-Style Processing Flow

```
Input Text → Split into Lines → Filter Lines → Apply Patterns → Transform Values → Extract Fields
```

### 2. Rule-Based Extraction

Each extraction rule contains:
- **`field`**: Target field name
- **`line_filters`**: AWK-style conditions to select lines
- **`patterns`**: Regex patterns to extract values
- **`transform`**: Value transformations to apply

## Base AwkExtractor Class

### Core Method: `extract_fields()`

```python
def extract_fields(self, text: str, field_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract fields using AWK-style rules."""
    extracted = {}
    lines = text.split("\n")  # Split into lines (AWK paradigm)
    
    for rule in field_rules:
        field_name = rule["field"]
        patterns = rule.get("patterns", [])
        line_filters = rule.get("line_filters", [])
        transformers = rule.get("transform", [])
        
        # Step 1: Find matching lines
        matching_lines = self._filter_lines(lines, line_filters)
        
        # Step 2: Extract from patterns
        for line in matching_lines:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    value = match.group(1) if match.groups() else match.group(0)
                    
                    # Step 3: Apply transformations
                    for transform in transformers:
                        value = self._apply_transform(value, transform)
                    
                    extracted[field_name] = value
                    break
```

### Line Filtering: `_filter_lines()`

Implements AWK-style line filtering with three types of conditions:

#### 1. Regex Patterns
```python
# AWK syntax: /pattern/
"/costco|shell|bp/"  # Matches lines containing fuel station names
"/\$/L|/L/"          # Matches lines with price per litre indicators
```

#### 2. Field Count Conditions
```python
# AWK syntax: NF > 2
"NF > 2"   # Lines with more than 2 fields (words)
"NF == 5"  # Lines with exactly 5 fields
```

#### 3. Field Reference Conditions
```python
# AWK syntax: $1, $2, etc.
'$1 == "TOTAL"'     # First field equals "TOTAL"
'$2 == "AMOUNT"'    # Second field equals "AMOUNT"
```

### Implementation Details

```python
def _line_matches_filter(self, line: str, filter_expr: str) -> bool:
    """Check if line matches AWK-style filter."""
    if filter_expr.startswith("/") and filter_expr.endswith("/"):
        # Regex pattern: /pattern/
        pattern = filter_expr[1:-1]
        return bool(re.search(pattern, line, re.IGNORECASE))
    
    elif "NF" in filter_expr:
        # Number of fields: NF > 2
        fields = line.split()
        nf = len(fields)
        return eval(filter_expr.replace("NF", str(nf)))
    
    elif "$" in filter_expr:
        # Field references: $1 == "TOTAL"
        fields = line.split()
        for i, field in enumerate(fields, 1):
            filter_expr = filter_expr.replace(f"${i}", f'"{field}"')
        return eval(filter_expr)
    
    else:
        # Simple substring match
        return filter_expr.lower() in line.lower()
```

### Value Transformations: `_apply_transform()`

Supports various text transformations:

```python
def _apply_transform(self, value: str, transform: str) -> str:
    """Apply transformation to extracted value."""
    if transform == "upper":
        return value.upper()
    elif transform == "lower":
        return value.lower()
    elif transform == "strip":
        return value.strip()
    elif transform == "remove_spaces":
        return re.sub(r"\s+", "", value)
    elif transform == "normalize_spaces":
        return " ".join(value.split())
    elif transform.startswith("prefix:"):
        prefix = transform[7:]
        return f"{prefix}{value}"
    elif transform.startswith("suffix:"):
        suffix = transform[7:]
        return f"{value}{suffix}"
    elif transform.startswith("replace:"):
        # Format: replace:old,new
        parts = transform[8:].split(",", 1)
        if len(parts) == 2:
            return value.replace(parts[0], parts[1])
    
    return value
```

## FuelReceiptAwkExtractor

### Fuel Receipt Extraction Rules

The `FuelReceiptAwkExtractor` defines 10 specialized extraction rules:

#### 1. STORE Field
```python
{
    "field": "STORE",
    "line_filters": ["/costco|shell|bp|coles|7-eleven|woolworths|ampol|mobil/"],
    "patterns": [
        r"(COSTCO\s+WHOLESALE\s+AUSTRALIA)",
        r"(COSTCO)",
        r"(SHELL)",
        r"(BP)",
        r"(COLES\s+EXPRESS)",
        # ... more patterns
    ],
    "transform": ["upper", "normalize_spaces"],
}
```

**How it works:**
1. **Line filter**: Only process lines containing fuel station names
2. **Patterns**: Match specific store name formats
3. **Transform**: Convert to uppercase and normalize spaces

#### 2. QUANTITIES Field
```python
{
    "field": "QUANTITIES",
    "line_filters": ["/.+L\b/"],  # Lines ending with 'L'
    "patterns": [
        r"(\d+\.\d{3})L",  # 32.230L
        r"(\d+\.\d{2})L",  # 45.67L
        r"(\d+\.\d{1})L",  # 32.2L
    ],
    "transform": ["suffix:L"],
}
```

**How it works:**
1. **Line filter**: Only process lines containing litre quantities
2. **Patterns**: Match various quantity formats with decimal places
3. **Transform**: Ensure 'L' suffix is present

#### 3. PRICES Field
```python
{
    "field": "PRICES",
    "line_filters": ["/\$/L|/L/"],  # Lines with price per litre
    "patterns": [
        r"\$(\d+\.\d{3})/L",  # $1.827/L
        r"(\d{3})/L",         # 827/L (cents)
        r"\$(\d+\.\d{2})/L",  # $1.85/L
    ],
    "transform": ["prefix:$", "suffix:/L"],
}
```

#### 4. DATE Field
```python
{
    "field": "DATE",
    "line_filters": ["NF > 2"],  # Lines with multiple fields
    "patterns": [
        r"(\d{2}/\d{2}/\d{4})",      # DD/MM/YYYY
        r"(\d{1,2}/\d{1,2}/\d{4})",  # D/M/YYYY
        r"(\d{4}-\d{2}-\d{2})",      # YYYY-MM-DD
    ],
    "transform": ["strip"],
}
```

### Post-Processing Logic

The `extract_fuel_fields()` method includes post-processing:

```python
def extract_fuel_fields(self, response: str) -> Dict[str, Any]:
    """Extract fuel receipt fields using AWK-style rules."""
    rules = self.get_fuel_extraction_rules()
    extracted = self.extract_fields(response, rules)
    
    # Post-process specific to fuel receipts
    if "PRICES" in extracted and "/L" not in extracted["PRICES"]:
        # Handle cents format: 827 → $0.827/L
        if extracted["PRICES"].isdigit() and len(extracted["PRICES"]) == 3:
            price_dollars = float(extracted["PRICES"]) / 100
            extracted["PRICES"] = f"${price_dollars:.3f}/L"
    
    return extracted
```

## BankStatementAwkExtractor

### Bank Statement Extraction Rules

#### 1. ACCOUNT_NUMBER Field
```python
{
    "field": "ACCOUNT_NUMBER",
    "line_filters": ["/account|acc/"],
    "patterns": [
        r"Account\s+Number[:\s]*(\d+)",
        r"Account[:\s]*(\d+)",
        r"Acc[:\s]*(\d+)",
    ],
    "transform": ["strip"],
}
```

#### 2. BSB Field
```python
{
    "field": "BSB",
    "line_filters": ["/bsb|\d{2,3}-\d{3}/"],
    "patterns": [
        r"BSB[:\s]*(\d{2,3}-\d{3})",
        r"(\d{2,3}-\d{3})",
    ],
    "transform": ["strip"],
}
```

#### 3. BANK_NAME Field
```python
{
    "field": "BANK_NAME",
    "line_filters": ["/anz|commonwealth|westpac|nab/"],
    "patterns": [
        r"(ANZ\s+BANK)",
        r"(COMMONWEALTH\s+BANK)",
        r"(WESTPAC)",
        r"(NAB)",
        r"(ANZ)",
    ],
    "transform": ["upper"],
}
```

## Usage Examples

### Basic Usage
```python
from llama_vision.extraction.awk_extractor import FuelReceiptAwkExtractor

extractor = FuelReceiptAwkExtractor()
extracted_fields = extractor.extract_fuel_fields(ocr_text)
```

### Custom Rules
```python
from llama_vision.extraction.awk_extractor import AwkExtractor

extractor = AwkExtractor()

custom_rules = [
    {
        "field": "INVOICE_NUMBER",
        "line_filters": ["/invoice|inv/"],
        "patterns": [r"Invoice\s+Number[:\s]*(\d+)"],
        "transform": ["strip", "upper"]
    }
]

extracted = extractor.extract_fields(text, custom_rules)
```

## AWK Equivalents

### Traditional AWK vs Python Implementation

#### AWK Script
```bash
# Traditional AWK for fuel quantity extraction
awk '/L$/ { 
    if (match($0, /([0-9]+\.[0-9]{3})L/)) {
        print substr($0, RSTART, RLENGTH)
    }
}' input.txt
```

#### Python AWK Extractor Equivalent
```python
{
    "field": "QUANTITIES",
    "line_filters": ["/.+L$/"],  # Lines ending with 'L'
    "patterns": [r"(\d+\.\d{3})L"],
    "transform": []
}
```

### Field Processing Comparison

#### AWK Field Processing
```bash
# AWK: Process fields in line
awk 'NF > 2 { print $1, $2 }' input.txt
```

#### Python Implementation
```python
# Python: Field reference in line filter
"line_filters": ["NF > 2"]
# Then use $1, $2 in patterns or field references
```

## Extending the System

### Adding New Document Types

1. **Create new extractor class:**
```python
class InvoiceAwkExtractor(AwkExtractor):
    def get_invoice_extraction_rules(self) -> List[Dict[str, Any]]:
        return [
            {
                "field": "INVOICE_NUMBER",
                "line_filters": ["/invoice|inv/"],
                "patterns": [r"Invoice\s+Number[:\s]*(\d+)"],
                "transform": ["strip"]
            },
            # More rules...
        ]
    
    def extract_invoice_fields(self, response: str) -> Dict[str, Any]:
        rules = self.get_invoice_extraction_rules()
        return self.extract_fields(response, rules)
```

2. **Register with handler system:**
```python
# In handler class
def get_awk_extractor(self):
    return InvoiceAwkExtractor()
```

### Adding New Transformations

Extend the `_apply_transform()` method:

```python
def _apply_transform(self, value: str, transform: str) -> str:
    # ... existing transforms ...
    elif transform.startswith("regex:"):
        # Format: regex:pattern,replacement
        parts = transform[6:].split(",", 1)
        if len(parts) == 2:
            return re.sub(parts[0], parts[1], value)
    elif transform == "currency":
        # Format numbers as currency
        return f"${float(value):.2f}"
    # ... more custom transforms ...
```

### Adding New Filter Types

Extend the `_line_matches_filter()` method:

```python
def _line_matches_filter(self, line: str, filter_expr: str) -> bool:
    # ... existing filters ...
    elif filter_expr.startswith("length:"):
        # Format: length:>20 or length:<10
        condition = filter_expr[7:]
        line_length = len(line)
        return eval(f"{line_length} {condition}")
    elif filter_expr.startswith("word_count:"):
        # Format: word_count:>5
        condition = filter_expr[11:]
        word_count = len(line.split())
        return eval(f"{word_count} {condition}")
```

## Debugging and Troubleshooting

### Enable Debug Logging
```python
extractor = FuelReceiptAwkExtractor(log_level="DEBUG")
```

### Common Issues

1. **No matches found:**
   - Check line filters are not too restrictive
   - Verify regex patterns are correct
   - Test patterns individually

2. **Wrong values extracted:**
   - Check regex capture groups
   - Verify transformation order
   - Test with sample data

3. **Performance issues:**
   - Optimize regex patterns
   - Use more specific line filters
   - Consider caching compiled patterns

### Testing Individual Rules

```python
# Test single rule
test_rule = {
    "field": "TOTAL",
    "line_filters": ["/total/"],
    "patterns": [r"TOTAL[^\d]*\$(\d+\.\d{2})"],
    "transform": ["prefix:$"]
}

extractor = AwkExtractor()
result = extractor.extract_fields(test_text, [test_rule])
print(result)
```

## Best Practices

### 1. Rule Design
- Use specific line filters to reduce processing
- Order patterns from most specific to most general
- Use appropriate capture groups in regex

### 2. Pattern Optimization
- Compile frequently used patterns
- Use non-greedy matching when appropriate
- Test patterns with real data

### 3. Transformation Strategy
- Apply transformations in logical order
- Use normalize_spaces before other transforms
- Consider validation transforms

### 4. Error Handling
- Use try-catch in eval() calls
- Validate extracted values
- Provide fallback patterns

## Integration with Document Handlers

The AWK extractor integrates with the document handler system as a fallback mechanism:

```python
# In fuel_receipt_handler.py
def extract_fields(self, response: str) -> Dict[str, Any]:
    # Primary extraction
    primary_fields = self.primary_extraction(response)
    
    # Fallback to AWK extraction if insufficient fields
    if len(primary_fields) < 6:
        awk_extractor = FuelReceiptAwkExtractor()
        awk_fields = awk_extractor.extract_fuel_fields(response)
        primary_fields.update(awk_fields)
    
    return primary_fields
```

This provides a robust extraction system that combines structured KEY-VALUE parsing with flexible AWK-style pattern matching for maximum field extraction coverage.