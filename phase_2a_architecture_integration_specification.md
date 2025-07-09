# Phase 2A: Architecture Integration Specification

## Overview

**Phase**: 2A - Architecture Integration  
**Duration**: Week 5 of hybrid system implementation  
**Prerequisites**: Phase 1A (AWK Parity) and Phase 1B (Domain Expertise) completed  
**Objective**: Integrate Australian tax domain expertise into the Llama-3.2 extraction pipeline

## Implementation Goals

### Primary Objectives
1. **Create ATO-enhanced document handlers** for each Australian tax document type
2. **Integrate validation layer** into the extraction pipeline
3. **Implement hybrid extraction** with AWK fallback and ATO validation
4. **Add Australian tax confidence scoring** to processing pipeline
5. **Ensure architectural parity** between Llama-3.2 and InternVL systems

### Success Criteria
- All 11 Australian tax document types have dedicated handlers
- ATO compliance validation is integrated into extraction pipeline
- Confidence scoring provides Australian tax-specific assessment
- AWK fallback works seamlessly with ATO validation
- Processing pipeline maintains compatibility with existing Llama-3.2 architecture

## Technical Architecture

### 2A.1: ATO-Enhanced Document Handlers

**Purpose**: Create specialized handlers for each Australian tax document type that combine extraction with ATO compliance validation.

#### Implementation Structure
```
llama_vision/
├── handlers/
│   ├── __init__.py
│   ├── base_ato_handler.py
│   ├── business_receipt_handler.py
│   ├── fuel_receipt_handler.py
│   ├── tax_invoice_handler.py
│   ├── bank_statement_handler.py
│   ├── meal_receipt_handler.py
│   ├── accommodation_handler.py
│   ├── travel_document_handler.py
│   ├── parking_toll_handler.py
│   ├── equipment_supplies_handler.py
│   ├── professional_services_handler.py
│   └── other_document_handler.py
```

#### A. Base ATO Handler (`base_ato_handler.py`)

```python
"""
Base ATO Handler for Australian Tax Document Processing

This module provides the base class for all Australian tax document handlers,
integrating extraction, validation, and confidence scoring.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from ..extraction.ato_compliance_handler import assess_ato_compliance_enhanced
from ..extraction.australian_tax_confidence_scorer import score_australian_tax_document_processing
from ..extraction.australian_tax_classifier import classify_australian_tax_document, DocumentType
from ..extraction.australian_tax_prompts import get_document_extraction_prompt
from ..extraction.awk_extractor import AwkExtractor
from ..utils import setup_logging

logger = setup_logging()


@dataclass
class ATOProcessingResult:
    """Result of ATO document processing."""
    success: bool
    document_type: DocumentType
    extracted_fields: Dict[str, Any]
    ato_compliance: Dict[str, Any]
    confidence_score: Dict[str, Any]
    processing_method: str
    extraction_quality: str
    recommendations: List[str]
    raw_extraction: Dict[str, Any]
    error_message: Optional[str] = None


class BaseATOHandler(ABC):
    """Base class for Australian Tax Office document handlers."""
    
    def __init__(self, document_type: DocumentType):
        self.document_type = document_type
        self.awk_extractor = AwkExtractor()
        self.extraction_prompt = get_document_extraction_prompt(document_type.value)
        
        # Document-specific configuration
        self.required_fields = self._get_required_fields()
        self.optional_fields = self._get_optional_fields()
        self.validation_rules = self._get_validation_rules()
        
        logger.info(f"Initialized {self.__class__.__name__} for {document_type.value}")
    
    def process_document(self, document_text: str, image_path: Optional[str] = None) -> ATOProcessingResult:
        """
        Process Australian tax document with full ATO compliance.
        
        Args:
            document_text: Extracted text from document
            image_path: Optional path to document image
            
        Returns:
            ATOProcessingResult with comprehensive processing information
        """
        logger.info(f"Processing {self.document_type.value} document")
        
        try:
            # Step 1: Classify document to verify type
            classification_result = classify_australian_tax_document(document_text)
            
            # Step 2: Extract fields using document-specific method
            extracted_fields = self._extract_fields(document_text)
            
            # Step 3: Apply AWK fallback if needed
            if self._should_use_awk_fallback(extracted_fields):
                logger.info("Applying AWK fallback extraction")
                awk_fields = self._extract_with_awk_fallback(document_text)
                extracted_fields = self._merge_extraction_results(extracted_fields, awk_fields)
            
            # Step 4: Apply document-specific validation
            validated_fields = self._validate_extracted_fields(extracted_fields)
            
            # Step 5: Assess ATO compliance
            ato_compliance = assess_ato_compliance_enhanced(
                validated_fields,
                self.document_type.value,
                self._get_expense_category()
            )
            
            # Step 6: Score confidence
            confidence_score = score_australian_tax_document_processing(
                document_text,
                classification_result,
                validated_fields,
                ato_compliance
            )
            
            # Step 7: Generate processing result
            processing_method = self._determine_processing_method(
                extracted_fields, awk_fields if 'awk_fields' in locals() else {}
            )
            
            result = ATOProcessingResult(
                success=True,
                document_type=self.document_type,
                extracted_fields=validated_fields,
                ato_compliance=ato_compliance,
                confidence_score=confidence_score.__dict__,
                processing_method=processing_method,
                extraction_quality=confidence_score.quality_grade,
                recommendations=confidence_score.recommendations,
                raw_extraction={
                    'classification': classification_result.__dict__,
                    'original_extraction': extracted_fields,
                    'awk_fallback_used': 'awk_fields' in locals()
                }
            )
            
            logger.info(f"Successfully processed {self.document_type.value} document "
                       f"with {confidence_score.quality_grade} quality")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {self.document_type.value} document: {e}")
            return ATOProcessingResult(
                success=False,
                document_type=self.document_type,
                extracted_fields={},
                ato_compliance={'success': False, 'error': str(e)},
                confidence_score={'overall_confidence': 0.0, 'quality_grade': 'Failed'},
                processing_method='failed',
                extraction_quality='Failed',
                recommendations=['Document processing failed - manual review required'],
                raw_extraction={},
                error_message=str(e)
            )
    
    @abstractmethod
    def _extract_fields(self, document_text: str) -> Dict[str, Any]:
        """Extract fields using document-specific extraction method."""
        pass
    
    @abstractmethod
    def _get_required_fields(self) -> List[str]:
        """Get list of required fields for this document type."""
        pass
    
    @abstractmethod
    def _get_optional_fields(self) -> List[str]:
        """Get list of optional fields for this document type."""
        pass
    
    @abstractmethod
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this document type."""
        pass
    
    @abstractmethod
    def _get_expense_category(self) -> str:
        """Get expense category for ATO compliance."""
        pass
    
    def _should_use_awk_fallback(self, extracted_fields: Dict[str, Any]) -> bool:
        """Determine if AWK fallback should be used."""
        
        # Count meaningful fields
        meaningful_fields = sum(1 for value in extracted_fields.values() 
                              if value and str(value).strip())
        
        # Use AWK fallback if insufficient fields extracted
        return meaningful_fields < len(self.required_fields)
    
    def _extract_with_awk_fallback(self, document_text: str) -> Dict[str, Any]:
        """Extract fields using AWK fallback."""
        
        # Get AWK extraction rules for this document type
        awk_rules = self._get_awk_extraction_rules()
        
        # Extract using AWK
        awk_fields = self.awk_extractor.extract_fields(document_text, awk_rules)
        
        # Map AWK fields to standard field names
        return self._map_awk_fields(awk_fields)
    
    def _merge_extraction_results(
        self, 
        primary_fields: Dict[str, Any], 
        fallback_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge primary and fallback extraction results."""
        
        merged = primary_fields.copy()
        
        # Use fallback values for missing or empty primary fields
        for field, value in fallback_fields.items():
            if field not in merged or not merged[field]:
                merged[field] = value
        
        return merged
    
    def _validate_extracted_fields(self, extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Apply document-specific validation to extracted fields."""
        
        validated = extracted_fields.copy()
        
        # Apply validation rules
        for field, rules in self.validation_rules.items():
            if field in validated:
                validated[field] = self._apply_field_validation(validated[field], rules)
        
        return validated
    
    def _apply_field_validation(self, field_value: Any, rules: Dict[str, Any]) -> Any:
        """Apply validation rules to a field value."""
        
        if not field_value:
            return field_value
        
        value = str(field_value).strip()
        
        # Apply transformations
        if 'transform' in rules:
            for transform in rules['transform']:
                if transform == 'upper':
                    value = value.upper()
                elif transform == 'lower':
                    value = value.lower()
                elif transform == 'strip':
                    value = value.strip()
                elif transform == 'normalize_spaces':
                    value = ' '.join(value.split())
        
        # Apply format validation
        if 'format' in rules:
            format_pattern = rules['format']
            import re
            if not re.match(format_pattern, value):
                logger.warning(f"Field value '{value}' does not match format '{format_pattern}'")
        
        return value
    
    def _determine_processing_method(
        self, 
        primary_fields: Dict[str, Any], 
        awk_fields: Dict[str, Any]
    ) -> str:
        """Determine which processing method was used."""
        
        if awk_fields:
            return "hybrid_with_awk_fallback"
        else:
            return "primary_extraction_only"
    
    @abstractmethod
    def _get_awk_extraction_rules(self) -> List[Dict[str, Any]]:
        """Get AWK extraction rules for this document type."""
        pass
    
    @abstractmethod
    def _map_awk_fields(self, awk_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Map AWK field names to standard field names."""
        pass
```

#### B. Fuel Receipt Handler (`fuel_receipt_handler.py`)

```python
"""
Fuel Receipt Handler for Australian Tax Document Processing

Specialized handler for fuel receipts with vehicle expense claim support.
"""

from typing import Dict, Any, List
from .base_ato_handler import BaseATOHandler
from ..extraction.australian_tax_classifier import DocumentType
from ..model.llama_interface import LlamaInterface
from ..utils import setup_logging

logger = setup_logging()


class FuelReceiptHandler(BaseATOHandler):
    """Handler for Australian fuel receipts with ATO compliance."""
    
    def __init__(self):
        super().__init__(DocumentType.FUEL_RECEIPT)
        self.llama_interface = LlamaInterface()
    
    def _extract_fields(self, document_text: str) -> Dict[str, Any]:
        """Extract fuel receipt fields using Llama model."""
        
        try:
            # Use specialized fuel receipt prompt
            response = self.llama_interface.generate_response(
                self.extraction_prompt,
                document_text
            )
            
            # Parse KEY-VALUE response
            extracted_fields = self._parse_key_value_response(response)
            
            # Apply fuel-specific post-processing
            extracted_fields = self._apply_fuel_specific_processing(extracted_fields)
            
            return extracted_fields
            
        except Exception as e:
            logger.error(f"Error extracting fuel receipt fields: {e}")
            return {}
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for fuel receipts."""
        return [
            'date',
            'station_name',
            'fuel_type',
            'litres',
            'price_per_litre',
            'total_amount'
        ]
    
    def _get_optional_fields(self) -> List[str]:
        """Get optional fields for fuel receipts."""
        return [
            'station_address',
            'time',
            'pump_number',
            'vehicle_km',
            'gst_amount',
            'payment_method',
            'receipt_number'
        ]
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for fuel receipts."""
        return {
            'date': {
                'format': r'^\d{1,2}/\d{1,2}/\d{4}$',
                'transform': ['strip']
            },
            'station_name': {
                'transform': ['upper', 'normalize_spaces']
            },
            'fuel_type': {
                'transform': ['upper'],
                'valid_values': ['UNLEADED', 'PREMIUM', 'DIESEL', 'ULP', 'E10', 'E85']
            },
            'litres': {
                'format': r'^\d+\.\d{1,3}$',
                'transform': ['strip']
            },
            'price_per_litre': {
                'format': r'^\d+\.\d{1,3}$',
                'transform': ['strip']
            },
            'total_amount': {
                'format': r'^\d+\.\d{2}$',
                'transform': ['strip']
            }
        }
    
    def _get_expense_category(self) -> str:
        """Get expense category for fuel receipts."""
        return "Vehicle Expenses"
    
    def _get_awk_extraction_rules(self) -> List[Dict[str, Any]]:
        """Get AWK extraction rules for fuel receipts."""
        return [
            {
                "field": "STATION",
                "line_filters": ["/bp|shell|caltex|ampol|mobil|7-eleven/"],
                "patterns": [
                    r"(BP|SHELL|CALTEX|AMPOL|MOBIL|7-ELEVEN)",
                    r"(.*FUEL.*|.*PETROL.*|.*SERVICE.*STATION.*)"
                ],
                "transform": ["upper", "normalize_spaces"]
            },
            {
                "field": "DATE",
                "line_filters": ["NF > 1"],
                "patterns": [
                    r"(\d{1,2}/\d{1,2}/\d{4})",
                    r"(\d{1,2}-\d{1,2}-\d{4})"
                ],
                "transform": ["strip"]
            },
            {
                "field": "FUEL_TYPE",
                "line_filters": ["/ulp|unleaded|premium|diesel|e10/"],
                "patterns": [
                    r"(ULP|UNLEADED|PREMIUM|DIESEL|E10)",
                    r"(.*UNLEADED.*|.*PREMIUM.*|.*DIESEL.*)"
                ],
                "transform": ["upper"]
            },
            {
                "field": "LITRES",
                "line_filters": ["/\d+\.\d+L/"],
                "patterns": [
                    r"(\d+\.\d{3})L",
                    r"(\d+\.\d{2})L"
                ],
                "transform": ["strip"]
            },
            {
                "field": "PRICE_PER_LITRE",
                "line_filters": ["/\d+\.\d+.*\/L/"],
                "patterns": [
                    r"(\d+\.\d+)\/L",
                    r"(\d+\.\d+)c\/L"
                ],
                "transform": ["strip"]
            },
            {
                "field": "TOTAL",
                "line_filters": ["/total|amount/"],
                "patterns": [
                    r"TOTAL[^\d]*\$(\d+\.\d{2})",
                    r"\$(\d+\.\d{2})\s*TOTAL"
                ],
                "transform": ["strip"]
            }
        ]
    
    def _map_awk_fields(self, awk_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Map AWK field names to standard field names."""
        
        field_mapping = {
            'STATION': 'station_name',
            'DATE': 'date',
            'FUEL_TYPE': 'fuel_type',
            'LITRES': 'litres',
            'PRICE_PER_LITRE': 'price_per_litre',
            'TOTAL': 'total_amount'
        }
        
        mapped_fields = {}
        for awk_field, standard_field in field_mapping.items():
            if awk_field in awk_fields:
                mapped_fields[standard_field] = awk_fields[awk_field]
        
        return mapped_fields
    
    def _parse_key_value_response(self, response: str) -> Dict[str, Any]:
        """Parse KEY-VALUE response from Llama model."""
        
        fields = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key and value:
                    fields[key] = value
        
        return fields
    
    def _apply_fuel_specific_processing(self, extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fuel-specific post-processing."""
        
        processed_fields = extracted_fields.copy()
        
        # Calculate missing fields if possible
        if ('litres' in processed_fields and 
            'price_per_litre' in processed_fields and 
            'total_amount' not in processed_fields):
            
            try:
                litres = float(processed_fields['litres'])
                price_per_litre = float(processed_fields['price_per_litre'])
                calculated_total = litres * price_per_litre
                processed_fields['total_amount'] = f"{calculated_total:.2f}"
                logger.info(f"Calculated total amount: ${calculated_total:.2f}")
            except (ValueError, TypeError):
                pass
        
        # Validate fuel station names
        if 'station_name' in processed_fields:
            station_name = processed_fields['station_name'].upper()
            known_stations = ['BP', 'SHELL', 'CALTEX', 'AMPOL', 'MOBIL', '7-ELEVEN']
            
            for known_station in known_stations:
                if known_station in station_name:
                    processed_fields['station_name'] = known_station
                    break
        
        return processed_fields
```

#### C. Tax Invoice Handler (`tax_invoice_handler.py`)

```python
"""
Tax Invoice Handler for Australian Tax Document Processing

Specialized handler for tax invoices with GST compliance.
"""

from typing import Dict, Any, List
from .base_ato_handler import BaseATOHandler
from ..extraction.australian_tax_classifier import DocumentType
from ..model.llama_interface import LlamaInterface
from ..utils import setup_logging

logger = setup_logging()


class TaxInvoiceHandler(BaseATOHandler):
    """Handler for Australian tax invoices with ATO compliance."""
    
    def __init__(self):
        super().__init__(DocumentType.TAX_INVOICE)
        self.llama_interface = LlamaInterface()
    
    def _extract_fields(self, document_text: str) -> Dict[str, Any]:
        """Extract tax invoice fields using Llama model."""
        
        try:
            response = self.llama_interface.generate_response(
                self.extraction_prompt,
                document_text
            )
            
            extracted_fields = self._parse_key_value_response(response)
            extracted_fields = self._apply_tax_invoice_processing(extracted_fields)
            
            return extracted_fields
            
        except Exception as e:
            logger.error(f"Error extracting tax invoice fields: {e}")
            return {}
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for tax invoices."""
        return [
            'document_type',
            'supplier_name',
            'supplier_abn',
            'invoice_number',
            'date',
            'description',
            'subtotal',
            'gst_amount',
            'total_amount'
        ]
    
    def _get_optional_fields(self) -> List[str]:
        """Get optional fields for tax invoices."""
        return [
            'supplier_address',
            'customer_name',
            'customer_abn',
            'due_date',
            'payment_terms',
            'invoice_reference'
        ]
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for tax invoices."""
        return {
            'document_type': {
                'valid_values': ['TAX INVOICE', 'INVOICE'],
                'transform': ['upper']
            },
            'supplier_name': {
                'transform': ['upper', 'normalize_spaces']
            },
            'supplier_abn': {
                'format': r'^\d{2}\s?\d{3}\s?\d{3}\s?\d{3}$',
                'transform': ['strip']
            },
            'date': {
                'format': r'^\d{1,2}/\d{1,2}/\d{4}$',
                'transform': ['strip']
            },
            'subtotal': {
                'format': r'^\d+\.\d{2}$',
                'transform': ['strip']
            },
            'gst_amount': {
                'format': r'^\d+\.\d{2}$',
                'transform': ['strip']
            },
            'total_amount': {
                'format': r'^\d+\.\d{2}$',
                'transform': ['strip']
            }
        }
    
    def _get_expense_category(self) -> str:
        """Get expense category for tax invoices."""
        return "Professional Services"
    
    def _get_awk_extraction_rules(self) -> List[Dict[str, Any]]:
        """Get AWK extraction rules for tax invoices."""
        return [
            {
                "field": "DOCUMENT_TYPE",
                "line_filters": ["/tax invoice|invoice/"],
                "patterns": [
                    r"(TAX INVOICE)",
                    r"(INVOICE)"
                ],
                "transform": ["upper"]
            },
            {
                "field": "SUPPLIER",
                "line_filters": ["/pty ltd|company|business/"],
                "patterns": [
                    r"(.*PTY\s+LTD.*)",
                    r"(.*COMPANY.*)"
                ],
                "transform": ["upper", "normalize_spaces"]
            },
            {
                "field": "ABN",
                "line_filters": ["/abn|\d{2}\s\d{3}\s\d{3}\s\d{3}/"],
                "patterns": [
                    r"ABN:?\s*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
                    r"(\d{2}\s\d{3}\s\d{3}\s\d{3})"
                ],
                "transform": ["normalize_spaces"]
            },
            {
                "field": "INVOICE_NUMBER",
                "line_filters": ["/invoice|inv|number/"],
                "patterns": [
                    r"Invoice[:\s]*#?(\w+)",
                    r"Inv[:\s]*#?(\w+)"
                ],
                "transform": ["strip"]
            },
            {
                "field": "DATE",
                "line_filters": ["NF > 1"],
                "patterns": [
                    r"(\d{1,2}/\d{1,2}/\d{4})",
                    r"(\d{1,2}-\d{1,2}-\d{4})"
                ],
                "transform": ["strip"]
            },
            {
                "field": "SUBTOTAL",
                "line_filters": ["/subtotal|sub-total/"],
                "patterns": [
                    r"SUBTOTAL[^\d]*\$(\d+\.\d{2})",
                    r"\$(\d+\.\d{2})\s*SUBTOTAL"
                ],
                "transform": ["strip"]
            },
            {
                "field": "GST",
                "line_filters": ["/gst|tax/"],
                "patterns": [
                    r"GST[^\d]*\$(\d+\.\d{2})",
                    r"\$(\d+\.\d{2})\s*GST"
                ],
                "transform": ["strip"]
            },
            {
                "field": "TOTAL",
                "line_filters": ["/total|amount/"],
                "patterns": [
                    r"TOTAL[^\d]*\$(\d+\.\d{2})",
                    r"\$(\d+\.\d{2})\s*TOTAL"
                ],
                "transform": ["strip"]
            }
        ]
    
    def _map_awk_fields(self, awk_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Map AWK field names to standard field names."""
        
        field_mapping = {
            'DOCUMENT_TYPE': 'document_type',
            'SUPPLIER': 'supplier_name',
            'ABN': 'supplier_abn',
            'INVOICE_NUMBER': 'invoice_number',
            'DATE': 'date',
            'SUBTOTAL': 'subtotal',
            'GST': 'gst_amount',
            'TOTAL': 'total_amount'
        }
        
        mapped_fields = {}
        for awk_field, standard_field in field_mapping.items():
            if awk_field in awk_fields:
                mapped_fields[standard_field] = awk_fields[awk_field]
        
        return mapped_fields
    
    def _parse_key_value_response(self, response: str) -> Dict[str, Any]:
        """Parse KEY-VALUE response from Llama model."""
        
        fields = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key and value:
                    fields[key] = value
        
        return fields
    
    def _apply_tax_invoice_processing(self, extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Apply tax invoice specific post-processing."""
        
        processed_fields = extracted_fields.copy()
        
        # Validate GST calculation (10% in Australia)
        if ('subtotal' in processed_fields and 
            'gst_amount' in processed_fields and 
            'total_amount' in processed_fields):
            
            try:
                subtotal = float(processed_fields['subtotal'])
                gst_amount = float(processed_fields['gst_amount'])
                total_amount = float(processed_fields['total_amount'])
                
                # Validate GST is 10% of subtotal
                expected_gst = subtotal * 0.10
                if abs(gst_amount - expected_gst) > 0.02:
                    logger.warning(f"GST calculation mismatch: {gst_amount} vs expected {expected_gst}")
                
                # Validate total = subtotal + GST
                expected_total = subtotal + gst_amount
                if abs(total_amount - expected_total) > 0.02:
                    logger.warning(f"Total calculation mismatch: {total_amount} vs expected {expected_total}")
                
            except (ValueError, TypeError):
                logger.warning("Could not validate GST calculation - non-numeric values")
        
        # Ensure document type is properly identified
        if 'document_type' not in processed_fields:
            processed_fields['document_type'] = 'TAX INVOICE'
        
        return processed_fields
```

### 2A.2: Validation Layer Integration

**Purpose**: Integrate ATO compliance validation into the extraction pipeline.

#### Implementation: Enhanced Extraction Pipeline (`enhanced_extraction_pipeline.py`)

```python
"""
Enhanced Extraction Pipeline with ATO Validation

This module integrates ATO compliance validation into the Llama-3.2 extraction pipeline.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from .handlers.base_ato_handler import ATOProcessingResult
from .handlers.business_receipt_handler import BusinessReceiptHandler
from .handlers.fuel_receipt_handler import FuelReceiptHandler
from .handlers.tax_invoice_handler import TaxInvoiceHandler
from .handlers.bank_statement_handler import BankStatementHandler
from .handlers.meal_receipt_handler import MealReceiptHandler
from .handlers.accommodation_handler import AccommodationHandler
from .handlers.travel_document_handler import TravelDocumentHandler
from .handlers.parking_toll_handler import ParkingTollHandler
from .handlers.equipment_supplies_handler import EquipmentSuppliesHandler
from .handlers.professional_services_handler import ProfessionalServicesHandler
from .handlers.other_document_handler import OtherDocumentHandler
from .extraction.australian_tax_classifier import classify_australian_tax_document, DocumentType
from .utils import setup_logging

logger = setup_logging()


@dataclass
class PipelineResult:
    """Result of enhanced extraction pipeline."""
    success: bool
    processing_result: ATOProcessingResult
    pipeline_method: str
    processing_time: float
    error_message: Optional[str] = None


class EnhancedExtractionPipeline:
    """Enhanced extraction pipeline with ATO validation."""
    
    def __init__(self):
        """Initialize pipeline with document handlers."""
        
        self.handlers = {
            DocumentType.BUSINESS_RECEIPT: BusinessReceiptHandler(),
            DocumentType.FUEL_RECEIPT: FuelReceiptHandler(),
            DocumentType.TAX_INVOICE: TaxInvoiceHandler(),
            DocumentType.BANK_STATEMENT: BankStatementHandler(),
            DocumentType.MEAL_RECEIPT: MealReceiptHandler(),
            DocumentType.ACCOMMODATION: AccommodationHandler(),
            DocumentType.TRAVEL_DOCUMENT: TravelDocumentHandler(),
            DocumentType.PARKING_TOLL: ParkingTollHandler(),
            DocumentType.EQUIPMENT_SUPPLIES: EquipmentSuppliesHandler(),
            DocumentType.PROFESSIONAL_SERVICES: ProfessionalServicesHandler(),
            DocumentType.OTHER: OtherDocumentHandler()
        }
        
        logger.info("Enhanced extraction pipeline initialized with ATO validation")
    
    def process_document(
        self, 
        document_text: str, 
        image_path: Optional[str] = None,
        document_type_hint: Optional[str] = None
    ) -> PipelineResult:
        """
        Process document through enhanced pipeline with ATO validation.
        
        Args:
            document_text: Extracted text from document
            image_path: Optional path to document image
            document_type_hint: Optional document type hint
            
        Returns:
            PipelineResult with comprehensive processing information
        """
        
        import time
        start_time = time.time()
        
        try:
            # Step 1: Classify document type
            if document_type_hint:
                document_type = DocumentType(document_type_hint)
                logger.info(f"Using provided document type hint: {document_type.value}")
            else:
                classification_result = classify_australian_tax_document(document_text)
                document_type = classification_result.document_type
                logger.info(f"Classified document as: {document_type.value}")
            
            # Step 2: Get appropriate handler
            handler = self.handlers.get(document_type)
            if not handler:
                logger.warning(f"No handler found for {document_type.value}, using OTHER handler")
                handler = self.handlers[DocumentType.OTHER]
            
            # Step 3: Process document with handler
            processing_result = handler.process_document(document_text, image_path)
            
            # Step 4: Create pipeline result
            processing_time = time.time() - start_time
            
            result = PipelineResult(
                success=processing_result.success,
                processing_result=processing_result,
                pipeline_method='enhanced_with_ato_validation',
                processing_time=processing_time
            )
            
            logger.info(f"Pipeline processing completed in {processing_time:.2f}s "
                       f"with {processing_result.extraction_quality} quality")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Pipeline processing failed: {e}")
            
            return PipelineResult(
                success=False,
                processing_result=ATOProcessingResult(
                    success=False,
                    document_type=DocumentType.OTHER,
                    extracted_fields={},
                    ato_compliance={'success': False, 'error': str(e)},
                    confidence_score={'overall_confidence': 0.0, 'quality_grade': 'Failed'},
                    processing_method='failed',
                    extraction_quality='Failed',
                    recommendations=['Pipeline processing failed - manual review required'],
                    raw_extraction={},
                    error_message=str(e)
                ),
                pipeline_method='enhanced_with_ato_validation',
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def get_handler(self, document_type: DocumentType) -> Optional[Any]:
        """Get handler for specific document type."""
        return self.handlers.get(document_type)
    
    def get_supported_document_types(self) -> List[DocumentType]:
        """Get list of supported document types."""
        return list(self.handlers.keys())
```

### 2A.3: Hybrid Extraction Implementation

**Purpose**: Implement hybrid extraction that combines primary extraction with AWK fallback and ATO validation.

#### Implementation: Hybrid Extraction Manager (`hybrid_extraction_manager.py`)

```python
"""
Hybrid Extraction Manager for Australian Tax Documents

This module manages the hybrid extraction process combining multiple extraction
methods with ATO validation for optimal results.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .enhanced_extraction_pipeline import EnhancedExtractionPipeline, PipelineResult
from .extraction.awk_extractor import AwkExtractor
from .extraction.ato_compliance_handler import assess_ato_compliance_enhanced
from .extraction.australian_tax_confidence_scorer import score_australian_tax_document_processing
from .extraction.australian_tax_classifier import classify_australian_tax_document
from .utils import setup_logging

logger = setup_logging()


class ExtractionMethod(Enum):
    """Extraction method types."""
    PRIMARY_ONLY = "primary_only"
    AWK_FALLBACK = "awk_fallback"
    HYBRID_COMBINED = "hybrid_combined"
    VALIDATION_ENHANCED = "validation_enhanced"


@dataclass
class HybridExtractionResult:
    """Result of hybrid extraction process."""
    success: bool
    extraction_method: ExtractionMethod
    primary_result: Dict[str, Any]
    awk_result: Dict[str, Any]
    combined_result: Dict[str, Any]
    ato_compliance: Dict[str, Any]
    confidence_score: Dict[str, Any]
    processing_pipeline: PipelineResult
    extraction_quality: str
    recommendations: List[str]
    performance_metrics: Dict[str, Any]
    error_message: Optional[str] = None


class HybridExtractionManager:
    """Manager for hybrid extraction with ATO validation."""
    
    def __init__(self):
        """Initialize hybrid extraction manager."""
        
        self.extraction_pipeline = EnhancedExtractionPipeline()
        self.awk_extractor = AwkExtractor()
        
        # Performance thresholds
        self.performance_thresholds = {
            'minimum_fields': 4,
            'minimum_confidence': 0.6,
            'awk_fallback_threshold': 0.5,
            'validation_enhancement_threshold': 0.8
        }
        
        logger.info("Hybrid extraction manager initialized")
    
    def extract_document(
        self, 
        document_text: str, 
        image_path: Optional[str] = None,
        document_type_hint: Optional[str] = None,
        extraction_strategy: str = "adaptive"
    ) -> HybridExtractionResult:
        """
        Extract document using hybrid approach with ATO validation.
        
        Args:
            document_text: Extracted text from document
            image_path: Optional path to document image
            document_type_hint: Optional document type hint
            extraction_strategy: Extraction strategy (adaptive/aggressive/conservative)
            
        Returns:
            HybridExtractionResult with comprehensive extraction information
        """
        
        import time
        start_time = time.time()
        
        try:
            # Step 1: Primary extraction through pipeline
            logger.info("Starting primary extraction through enhanced pipeline")
            pipeline_result = self.extraction_pipeline.process_document(
                document_text, image_path, document_type_hint
            )
            
            primary_result = pipeline_result.processing_result.extracted_fields
            primary_confidence = pipeline_result.processing_result.confidence_score
            
            # Step 2: Determine if AWK fallback is needed
            needs_awk_fallback = self._should_use_awk_fallback(
                primary_result, primary_confidence, extraction_strategy
            )
            
            awk_result = {}
            if needs_awk_fallback:
                logger.info("Applying AWK fallback extraction")
                awk_result = self._extract_with_awk_fallback(
                    document_text, 
                    pipeline_result.processing_result.document_type
                )
            
            # Step 3: Combine results
            combined_result = self._combine_extraction_results(
                primary_result, awk_result, extraction_strategy
            )
            
            # Step 4: Enhanced ATO validation
            ato_compliance = assess_ato_compliance_enhanced(
                combined_result,
                pipeline_result.processing_result.document_type.value,
                pipeline_result.processing_result.ato_compliance.get('claim_category', 'General')
            )
            
            # Step 5: Comprehensive confidence scoring
            classification_result = classify_australian_tax_document(document_text)
            confidence_score = score_australian_tax_document_processing(
                document_text,
                classification_result,
                combined_result,
                ato_compliance
            )
            
            # Step 6: Determine extraction method used
            extraction_method = self._determine_extraction_method(
                primary_result, awk_result, extraction_strategy
            )
            
            # Step 7: Generate performance metrics
            performance_metrics = self._generate_performance_metrics(
                start_time, primary_result, awk_result, combined_result,
                pipeline_result, confidence_score
            )
            
            # Step 8: Generate recommendations
            recommendations = self._generate_hybrid_recommendations(
                extraction_method, confidence_score, ato_compliance, performance_metrics
            )
            
            result = HybridExtractionResult(
                success=True,
                extraction_method=extraction_method,
                primary_result=primary_result,
                awk_result=awk_result,
                combined_result=combined_result,
                ato_compliance=ato_compliance,
                confidence_score=confidence_score.__dict__,
                processing_pipeline=pipeline_result,
                extraction_quality=confidence_score.quality_grade,
                recommendations=recommendations,
                performance_metrics=performance_metrics
            )
            
            logger.info(f"Hybrid extraction completed using {extraction_method.value} "
                       f"with {confidence_score.quality_grade} quality")
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid extraction failed: {e}")
            
            return HybridExtractionResult(
                success=False,
                extraction_method=ExtractionMethod.PRIMARY_ONLY,
                primary_result={},
                awk_result={},
                combined_result={},
                ato_compliance={'success': False, 'error': str(e)},
                confidence_score={'overall_confidence': 0.0, 'quality_grade': 'Failed'},
                processing_pipeline=None,
                extraction_quality='Failed',
                recommendations=['Hybrid extraction failed - manual review required'],
                performance_metrics={'processing_time': time.time() - start_time},
                error_message=str(e)
            )
    
    def _should_use_awk_fallback(
        self, 
        primary_result: Dict[str, Any], 
        primary_confidence: Dict[str, Any], 
        strategy: str
    ) -> bool:
        """Determine if AWK fallback should be used."""
        
        # Count meaningful fields
        meaningful_fields = sum(1 for value in primary_result.values() 
                              if value and str(value).strip())
        
        # Get confidence score
        confidence = primary_confidence.get('overall_confidence', 0.0)
        
        # Strategy-based decision
        if strategy == "aggressive":
            return meaningful_fields < self.performance_thresholds['minimum_fields'] * 1.5
        elif strategy == "conservative":
            return meaningful_fields < self.performance_thresholds['minimum_fields'] * 0.5
        else:  # adaptive
            return (meaningful_fields < self.performance_thresholds['minimum_fields'] or
                   confidence < self.performance_thresholds['awk_fallback_threshold'])
    
    def _extract_with_awk_fallback(
        self, 
        document_text: str, 
        document_type: Any
    ) -> Dict[str, Any]:
        """Extract using AWK fallback."""
        
        # Get document handler for AWK rules
        handler = self.extraction_pipeline.get_handler(document_type)
        if not handler:
            logger.warning(f"No handler found for {document_type.value}")
            return {}
        
        # Get AWK rules from handler
        awk_rules = handler._get_awk_extraction_rules()
        
        # Extract using AWK
        awk_fields = self.awk_extractor.extract_fields(document_text, awk_rules)
        
        # Map AWK fields to standard format
        return handler._map_awk_fields(awk_fields)
    
    def _combine_extraction_results(
        self, 
        primary_result: Dict[str, Any], 
        awk_result: Dict[str, Any], 
        strategy: str
    ) -> Dict[str, Any]:
        """Combine primary and AWK extraction results."""
        
        combined = primary_result.copy()
        
        # Strategy-based combination
        if strategy == "aggressive":
            # Prefer AWK results when available
            for field, value in awk_result.items():
                if value and str(value).strip():
                    combined[field] = value
        elif strategy == "conservative":
            # Only use AWK results for missing fields
            for field, value in awk_result.items():
                if field not in combined or not combined[field]:
                    if value and str(value).strip():
                        combined[field] = value
        else:  # adaptive
            # Use AWK results for missing or low-confidence fields
            for field, value in awk_result.items():
                if value and str(value).strip():
                    if (field not in combined or 
                        not combined[field] or 
                        len(str(combined[field]).strip()) < 3):
                        combined[field] = value
        
        return combined
    
    def _determine_extraction_method(
        self, 
        primary_result: Dict[str, Any], 
        awk_result: Dict[str, Any], 
        strategy: str
    ) -> ExtractionMethod:
        """Determine which extraction method was used."""
        
        if not awk_result:
            return ExtractionMethod.PRIMARY_ONLY
        elif not primary_result:
            return ExtractionMethod.AWK_FALLBACK
        else:
            return ExtractionMethod.HYBRID_COMBINED
    
    def _generate_performance_metrics(
        self, 
        start_time: float, 
        primary_result: Dict[str, Any], 
        awk_result: Dict[str, Any], 
        combined_result: Dict[str, Any],
        pipeline_result: PipelineResult,
        confidence_score: Any
    ) -> Dict[str, Any]:
        """Generate performance metrics."""
        
        import time
        total_time = time.time() - start_time
        
        return {
            'processing_time': total_time,
            'pipeline_time': pipeline_result.processing_time if pipeline_result else 0.0,
            'primary_fields_count': len([v for v in primary_result.values() if v]),
            'awk_fields_count': len([v for v in awk_result.values() if v]),
            'combined_fields_count': len([v for v in combined_result.values() if v]),
            'confidence_improvement': confidence_score.overall_confidence - 
                                   pipeline_result.processing_result.confidence_score.get('overall_confidence', 0.0)
                                   if pipeline_result and pipeline_result.processing_result else 0.0,
            'awk_fallback_used': bool(awk_result),
            'extraction_efficiency': len([v for v in combined_result.values() if v]) / total_time
        }
    
    def _generate_hybrid_recommendations(
        self, 
        extraction_method: ExtractionMethod, 
        confidence_score: Any, 
        ato_compliance: Dict[str, Any], 
        performance_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for hybrid extraction."""
        
        recommendations = []
        
        # Method-specific recommendations
        if extraction_method == ExtractionMethod.PRIMARY_ONLY:
            recommendations.append("Primary extraction sufficient - no fallback needed")
        elif extraction_method == ExtractionMethod.HYBRID_COMBINED:
            recommendations.append("Hybrid extraction used - improved field completeness")
        elif extraction_method == ExtractionMethod.AWK_FALLBACK:
            recommendations.append("AWK fallback required - primary extraction insufficient")
        
        # Performance recommendations
        if performance_metrics['processing_time'] > 10.0:
            recommendations.append("Long processing time - consider optimization")
        
        if performance_metrics['confidence_improvement'] > 0.2:
            recommendations.append("Significant confidence improvement from hybrid approach")
        
        # ATO compliance recommendations
        if ato_compliance.get('success') and ato_compliance.get('ato_ready'):
            recommendations.append("Document meets ATO compliance requirements")
        else:
            recommendations.append("ATO compliance issues detected - manual review recommended")
        
        # Quality recommendations
        if confidence_score.overall_confidence >= 0.9:
            recommendations.append("Excellent extraction quality - suitable for automated processing")
        elif confidence_score.overall_confidence >= 0.7:
            recommendations.append("Good extraction quality - suitable for production use")
        else:
            recommendations.append("Low extraction quality - manual review required")
        
        return recommendations
```

### 2A.4: Australian Tax Confidence Scoring Integration

**Purpose**: Integrate Australian tax confidence scoring into the processing pipeline.

#### Implementation: Confidence Integration Manager (`confidence_integration_manager.py`)

```python
"""
Confidence Integration Manager for Australian Tax Processing

This module integrates Australian tax confidence scoring throughout the processing pipeline.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .extraction.australian_tax_confidence_scorer import (
    score_australian_tax_document_processing,
    ConfidenceScoreResult
)
from .extraction.australian_tax_classifier import classify_australian_tax_document
from .utils import setup_logging

logger = setup_logging()


@dataclass
class IntegratedConfidenceResult:
    """Result of integrated confidence assessment."""
    document_confidence: ConfidenceScoreResult
    processing_confidence: float
    production_readiness: bool
    quality_assurance: Dict[str, Any]
    recommendations: List[str]
    monitoring_alerts: List[str]


class ConfidenceIntegrationManager:
    """Manager for integrating confidence scoring throughout the pipeline."""
    
    def __init__(self):
        """Initialize confidence integration manager."""
        
        self.confidence_thresholds = {
            'production_minimum': 0.7,
            'automatic_processing': 0.9,
            'manual_review': 0.5,
            'rejection_threshold': 0.3
        }
        
        self.monitoring_rules = {
            'low_confidence_alert': 0.6,
            'ato_compliance_alert': 0.7,
            'australian_business_alert': 0.4,
            'extraction_quality_alert': 0.6
        }
        
        logger.info("Confidence integration manager initialized")
    
    def assess_integrated_confidence(
        self,
        document_text: str,
        extracted_fields: Dict[str, Any],
        ato_compliance_result: Dict[str, Any],
        processing_metadata: Dict[str, Any]
    ) -> IntegratedConfidenceResult:
        """
        Assess integrated confidence across all processing aspects.
        
        Args:
            document_text: Original document text
            extracted_fields: Extracted field values
            ato_compliance_result: ATO compliance assessment
            processing_metadata: Processing metadata
            
        Returns:
            IntegratedConfidenceResult with comprehensive confidence assessment
        """
        
        try:
            # Step 1: Classify document
            classification_result = classify_australian_tax_document(document_text)
            
            # Step 2: Score document processing confidence
            document_confidence = score_australian_tax_document_processing(
                document_text,
                classification_result,
                extracted_fields,
                ato_compliance_result
            )
            
            # Step 3: Assess processing confidence
            processing_confidence = self._assess_processing_confidence(
                document_confidence,
                processing_metadata
            )
            
            # Step 4: Determine production readiness
            production_readiness = self._determine_production_readiness(
                document_confidence,
                processing_confidence,
                ato_compliance_result
            )
            
            # Step 5: Generate quality assurance metrics
            quality_assurance = self._generate_quality_assurance_metrics(
                document_confidence,
                processing_confidence,
                ato_compliance_result,
                processing_metadata
            )
            
            # Step 6: Generate integrated recommendations
            recommendations = self._generate_integrated_recommendations(
                document_confidence,
                processing_confidence,
                production_readiness,
                quality_assurance
            )
            
            # Step 7: Generate monitoring alerts
            monitoring_alerts = self._generate_monitoring_alerts(
                document_confidence,
                processing_confidence,
                ato_compliance_result
            )
            
            result = IntegratedConfidenceResult(
                document_confidence=document_confidence,
                processing_confidence=processing_confidence,
                production_readiness=production_readiness,
                quality_assurance=quality_assurance,
                recommendations=recommendations,
                monitoring_alerts=monitoring_alerts
            )
            
            logger.info(f"Integrated confidence assessment completed: "
                       f"document={document_confidence.overall_confidence:.2f}, "
                       f"processing={processing_confidence:.2f}, "
                       f"production_ready={production_readiness}")
            
            return result
            
        except Exception as e:
            logger.error(f"Integrated confidence assessment failed: {e}")
            
            return IntegratedConfidenceResult(
                document_confidence=ConfidenceScoreResult(
                    overall_confidence=0.0,
                    classification_confidence=0.0,
                    extraction_confidence=0.0,
                    ato_compliance_confidence=0.0,
                    australian_business_confidence=0.0,
                    quality_grade="Failed",
                    evidence={},
                    recommendations=[f"Confidence assessment failed: {e}"],
                    is_production_ready=False
                ),
                processing_confidence=0.0,
                production_readiness=False,
                quality_assurance={'status': 'failed', 'error': str(e)},
                recommendations=['Manual review required due to confidence assessment failure'],
                monitoring_alerts=[f'CRITICAL: Confidence assessment failed - {e}']
            )
    
    def _assess_processing_confidence(
        self,
        document_confidence: ConfidenceScoreResult,
        processing_metadata: Dict[str, Any]
    ) -> float:
        """Assess processing confidence based on metadata."""
        
        base_confidence = document_confidence.overall_confidence
        
        # Processing time factor
        processing_time = processing_metadata.get('processing_time', 0.0)
        if processing_time > 10.0:
            base_confidence *= 0.9  # Penalty for slow processing
        elif processing_time < 2.0:
            base_confidence *= 1.05  # Bonus for fast processing
        
        # Extraction method factor
        extraction_method = processing_metadata.get('extraction_method', '')
        if extraction_method == 'hybrid_combined':
            base_confidence *= 1.1  # Bonus for hybrid approach
        elif extraction_method == 'awk_fallback':
            base_confidence *= 0.9  # Penalty for fallback only
        
        # Field completeness factor
        field_count = processing_metadata.get('field_count', 0)
        if field_count >= 8:
            base_confidence *= 1.05  # Bonus for complete extraction
        elif field_count < 4:
            base_confidence *= 0.85  # Penalty for incomplete extraction
        
        return min(base_confidence, 1.0)
    
    def _determine_production_readiness(
        self,
        document_confidence: ConfidenceScoreResult,
        processing_confidence: float,
        ato_compliance_result: Dict[str, Any]
    ) -> bool:
        """Determine if document processing is production ready."""
        
        # Check confidence thresholds
        if document_confidence.overall_confidence < self.confidence_thresholds['production_minimum']:
            return False
        
        if processing_confidence < self.confidence_thresholds['production_minimum']:
            return False
        
        # Check ATO compliance
        if not ato_compliance_result.get('success', False):
            return False
        
        if not ato_compliance_result.get('ato_ready', False):
            return False
        
        # Check Australian business context
        if document_confidence.australian_business_confidence < 0.3:
            return False
        
        return True
    
    def _generate_quality_assurance_metrics(
        self,
        document_confidence: ConfidenceScoreResult,
        processing_confidence: float,
        ato_compliance_result: Dict[str, Any],
        processing_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate quality assurance metrics."""
        
        return {
            'overall_quality_score': (
                document_confidence.overall_confidence * 0.6 +
                processing_confidence * 0.4
            ),
            'component_scores': {
                'document_confidence': document_confidence.overall_confidence,
                'processing_confidence': processing_confidence,
                'ato_compliance': ato_compliance_result.get('compliance_score', 0.0) / 100.0,
                'australian_business': document_confidence.australian_business_confidence
            },
            'quality_indicators': {
                'extraction_completeness': processing_metadata.get('field_count', 0) / 10.0,
                'processing_efficiency': min(10.0 / processing_metadata.get('processing_time', 10.0), 1.0),
                'confidence_consistency': min(abs(document_confidence.overall_confidence - processing_confidence) < 0.2, 1.0),
                'ato_readiness': ato_compliance_result.get('ato_ready', False)
            },
            'risk_assessment': {
                'low_confidence_risk': document_confidence.overall_confidence < 0.7,
                'ato_compliance_risk': not ato_compliance_result.get('ato_ready', False),
                'processing_risk': processing_confidence < 0.7,
                'business_context_risk': document_confidence.australian_business_confidence < 0.5
            }
        }
    
    def _generate_integrated_recommendations(
        self,
        document_confidence: ConfidenceScoreResult,
        processing_confidence: float,
        production_readiness: bool,
        quality_assurance: Dict[str, Any]
    ) -> List[str]:
        """Generate integrated recommendations."""
        
        recommendations = []
        
        # Production readiness recommendations
        if production_readiness:
            recommendations.append("Document processing is production ready")
        else:
            recommendations.append("Manual review required before production use")
        
        # Quality-specific recommendations
        overall_quality = quality_assurance['overall_quality_score']
        if overall_quality >= 0.9:
            recommendations.append("Excellent quality - suitable for automated processing")
        elif overall_quality >= 0.7:
            recommendations.append("Good quality - monitor for consistency")
        else:
            recommendations.append("Quality concerns - increase manual oversight")
        
        # Component-specific recommendations
        if document_confidence.australian_business_confidence < 0.5:
            recommendations.append("Australian business context unclear - verify document authenticity")
        
        if processing_confidence < 0.7:
            recommendations.append("Processing confidence low - review extraction methods")
        
        # Risk-based recommendations
        risks = quality_assurance['risk_assessment']
        if risks['ato_compliance_risk']:
            recommendations.append("ATO compliance risk - ensure all required fields are validated")
        
        if risks['processing_risk']:
            recommendations.append("Processing risk detected - consider hybrid extraction approach")
        
        return recommendations
    
    def _generate_monitoring_alerts(
        self,
        document_confidence: ConfidenceScoreResult,
        processing_confidence: float,
        ato_compliance_result: Dict[str, Any]
    ) -> List[str]:
        """Generate monitoring alerts."""
        
        alerts = []
        
        # Confidence alerts
        if document_confidence.overall_confidence < self.monitoring_rules['low_confidence_alert']:
            alerts.append(f"LOW_CONFIDENCE: Document confidence {document_confidence.overall_confidence:.2f}")
        
        if processing_confidence < self.monitoring_rules['low_confidence_alert']:
            alerts.append(f"LOW_PROCESSING_CONFIDENCE: Processing confidence {processing_confidence:.2f}")
        
        # ATO compliance alerts
        ato_score = ato_compliance_result.get('compliance_score', 0.0) / 100.0
        if ato_score < self.monitoring_rules['ato_compliance_alert']:
            alerts.append(f"ATO_COMPLIANCE_RISK: Compliance score {ato_score:.2f}")
        
        # Australian business alerts
        if document_confidence.australian_business_confidence < self.monitoring_rules['australian_business_alert']:
            alerts.append(f"BUSINESS_CONTEXT_RISK: Australian business confidence {document_confidence.australian_business_confidence:.2f}")
        
        # Extraction quality alerts
        if document_confidence.extraction_confidence < self.monitoring_rules['extraction_quality_alert']:
            alerts.append(f"EXTRACTION_QUALITY_RISK: Extraction confidence {document_confidence.extraction_confidence:.2f}")
        
        return alerts
```

## Implementation Timeline

### Week 5: Phase 2A Implementation Schedule

**Day 1-2: Document Handlers**
- Implement `BaseATOHandler` class
- Create `FuelReceiptHandler` and `TaxInvoiceHandler`
- Create `BusinessReceiptHandler` and `BankStatementHandler`
- Test handler integration

**Day 3-4: Pipeline Integration**
- Implement `EnhancedExtractionPipeline`
- Create `HybridExtractionManager`
- Integrate ATO validation into pipeline
- Test pipeline functionality

**Day 5: Confidence Integration**
- Implement `ConfidenceIntegrationManager`
- Integrate confidence scoring throughout pipeline
- Add monitoring and alerting
- Final testing and validation

## Testing Strategy

### Unit Testing
- Individual handler testing with document samples
- AWK fallback testing with challenging documents
- ATO compliance validation testing
- Confidence scoring accuracy testing

### Integration Testing
- End-to-end pipeline testing
- Hybrid extraction testing
- Performance benchmarking
- Error handling testing

### Validation Testing
- Australian tax document validation
- ATO compliance verification
- Confidence score accuracy validation
- Production readiness assessment

## Success Metrics

### Functionality Metrics
- All 11 document types have working handlers
- AWK fallback operates correctly
- ATO compliance validation works
- Confidence scoring provides accurate assessment

### Performance Metrics
- Processing time < 10 seconds per document
- Confidence score accuracy > 85%
- ATO compliance detection > 90%
- Production readiness assessment > 80% accuracy

### Quality Metrics
- Extraction completeness > 85%
- Field validation accuracy > 95%
- Australian business detection > 90%
- Error handling coverage > 95%

## Deliverables

### Code Deliverables
1. **Document Handlers** (12 files)
2. **Enhanced Extraction Pipeline** (1 file)
3. **Hybrid Extraction Manager** (1 file)
4. **Confidence Integration Manager** (1 file)
5. **Unit Tests** (15 files)
6. **Integration Tests** (5 files)

### Documentation Deliverables
1. **Implementation Guide** (this document)
2. **API Documentation** (auto-generated)
3. **Testing Documentation** (testing results)
4. **Performance Benchmarks** (performance analysis)

### Validation Deliverables
1. **ATO Compliance Report** (compliance verification)
2. **Confidence Scoring Analysis** (accuracy assessment)
3. **Production Readiness Report** (deployment readiness)
4. **Architecture Parity Report** (comparison with InternVL)

## Conclusion

Phase 2A: Architecture Integration will successfully integrate all Australian tax domain expertise into the Llama-3.2 extraction pipeline, creating a production-ready system with:

- **11 Specialized Document Handlers** with ATO compliance
- **Hybrid Extraction Pipeline** with AWK fallback
- **Integrated Confidence Scoring** throughout the process
- **Comprehensive Validation Layer** for quality assurance
- **Production-Ready Architecture** with monitoring and alerting

Upon completion, the Llama-3.2 system will have full architectural parity with the InternVL PoC system, ensuring fair comparison of core model capabilities rather than system architecture differences.

**Next Phase**: Phase 2B - System Testing (Week 6)