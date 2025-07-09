"""
Test Phase 2A: Architecture Integration

This test script verifies the complete Phase 2A implementation including:
- ATO-enhanced document handlers
- Hybrid extraction manager
- Confidence integration manager
- End-to-end processing pipeline
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from llama_vision.extraction.australian_tax_classifier import DocumentType
from llama_vision.extraction.confidence_integration_manager import (
    assess_document_confidence,
)
from llama_vision.extraction.hybrid_extraction_manager import (
    process_australian_tax_document,
)
from llama_vision.handlers.business_receipt_handler import BusinessReceiptHandler
from llama_vision.handlers.fuel_receipt_handler import FuelReceiptHandler
from llama_vision.handlers.tax_invoice_handler import TaxInvoiceHandler


def test_fuel_receipt_handler():
    """Test fuel receipt handler with sample data."""
    print("=== Testing Fuel Receipt Handler ===")

    sample_fuel_receipt = """
    BP AUSTRALIA
    123 MAIN STREET, MELBOURNE VIC 3000
    ABN: 12 345 678 901
    
    Date: 15/06/2024
    Time: 14:35
    
    Fuel Type: Unleaded 91
    Pump: 3
    Litres: 45.230
    Rate: 189.9 c/L
    
    Fuel Total: $85.85
    GST: $7.81
    Total: $85.85
    
    Thank you for your business
    """

    handler = FuelReceiptHandler()
    result = handler.process_document(sample_fuel_receipt)

    print(f"Success: {result.success}")
    print(f"Document Type: {result.document_type.value}")
    print(f"Extraction Quality: {result.extraction_quality}")
    print(f"Processing Method: {result.processing_method}")
    print(f"Extracted Fields: {len(result.extracted_fields)}")
    print(f"Sample Fields: {dict(list(result.extracted_fields.items())[:5])}")
    print(f"Recommendations: {result.recommendations[:3]}")
    print()


def test_tax_invoice_handler():
    """Test tax invoice handler with sample data."""
    print("=== Testing Tax Invoice Handler ===")

    sample_tax_invoice = """
    TAX INVOICE
    
    ACME CONSULTING PTY LTD
    ABN: 12 345 678 901
    456 Business Street, Sydney NSW 2000
    
    To: CLIENT COMPANY PTY LTD
    Customer ABN: 98 765 432 109
    
    Invoice Number: INV-2024-0156
    Date: 15/06/2024
    Due Date: 15/07/2024
    
    Description: Professional consulting services
    
    Subtotal: $500.00
    GST: $50.00
    Total: $550.00
    
    Payment Terms: Net 30 days
    """

    handler = TaxInvoiceHandler()
    result = handler.process_document(sample_tax_invoice)

    print(f"Success: {result.success}")
    print(f"Document Type: {result.document_type.value}")
    print(f"Extraction Quality: {result.extraction_quality}")
    print(f"Processing Method: {result.processing_method}")
    print(f"Extracted Fields: {len(result.extracted_fields)}")
    print(f"Sample Fields: {dict(list(result.extracted_fields.items())[:5])}")
    print(f"Recommendations: {result.recommendations[:3]}")
    print()


def test_business_receipt_handler():
    """Test business receipt handler with sample data."""
    print("=== Testing Business Receipt Handler ===")

    sample_business_receipt = """
    WOOLWORTHS SUPERMARKETS
    ABN: 88 000 014 675
    
    Date: 15/06/2024
    
    Bread White                1    $3.50
    Milk 2L                    1    $5.20
    Eggs Free Range 12pk       1    $8.95
    
    Subtotal: $16.05
    GST: $1.60
    Total: $17.65
    
    Payment: EFTPOS
    Thank you for shopping with us
    """

    handler = BusinessReceiptHandler()
    result = handler.process_document(sample_business_receipt)

    print(f"Success: {result.success}")
    print(f"Document Type: {result.document_type.value}")
    print(f"Extraction Quality: {result.extraction_quality}")
    print(f"Processing Method: {result.processing_method}")
    print(f"Extracted Fields: {len(result.extracted_fields)}")
    print(f"Sample Fields: {dict(list(result.extracted_fields.items())[:5])}")
    print(f"Recommendations: {result.recommendations[:3]}")
    print()


def test_hybrid_extraction_manager():
    """Test hybrid extraction manager with multiple document types."""
    print("=== Testing Hybrid Extraction Manager ===")

    # Test documents
    test_documents = [
        {
            "text": """
            BP AUSTRALIA
            Date: 15/06/2024
            Fuel Type: Unleaded 91
            Litres: 45.230
            Total: $85.85
            """,
            "expected_type": DocumentType.FUEL_RECEIPT,
        },
        {
            "text": """
            TAX INVOICE
            ACME CONSULTING PTY LTD
            ABN: 12 345 678 901
            Date: 15/06/2024
            Total: $550.00
            """,
            "expected_type": DocumentType.TAX_INVOICE,
        },
        {
            "text": """
            WOOLWORTHS SUPERMARKETS
            ABN: 88 000 014 675
            Date: 15/06/2024
            Total: $17.65
            """,
            "expected_type": DocumentType.BUSINESS_RECEIPT,
        },
    ]

    for i, doc in enumerate(test_documents):
        print(f"--- Document {i + 1} ---")

        result = process_australian_tax_document(doc["text"])

        print(f"Success: {result.success}")
        print(f"Document Type: {result.document_type.value}")
        print(f"Expected Type: {doc['expected_type'].value}")
        print(f"Type Match: {result.document_type == doc['expected_type']}")
        print(f"Handler Used: {result.handler_used}")
        print(f"Processing Time: {result.processing_time:.3f}s")
        print(f"Confidence Summary: {result.confidence_summary}")
        print()


def test_confidence_integration_manager():
    """Test confidence integration manager."""
    print("=== Testing Confidence Integration Manager ===")

    sample_document = """
    BP AUSTRALIA
    ABN: 12 345 678 901
    123 MAIN STREET, MELBOURNE VIC 3000
    
    Date: 15/06/2024
    Time: 14:35
    
    Fuel Type: Unleaded 91
    Pump: 3
    Litres: 45.230
    Rate: 189.9 c/L
    
    Fuel Total: $85.85
    GST: $7.81
    Total: $85.85
    """

    confidence_result = assess_document_confidence(sample_document)

    print(f"Overall Confidence: {confidence_result.overall_confidence:.3f}")
    print(f"Production Readiness: {confidence_result.production_readiness.value}")
    print(f"Quality Grade: {confidence_result.quality_grade}")
    print(f"Processing Decision: {confidence_result.processing_decision}")
    print(f"Component Scores: {confidence_result.component_scores}")
    print(f"Quality Flags: {confidence_result.quality_flags}")
    print(f"Recommendations: {confidence_result.recommendations[:3]}")
    print(f"ATO Compliance Status: {confidence_result.ato_compliance_status}")
    print()


def test_awk_fallback_integration():
    """Test AWK fallback integration."""
    print("=== Testing AWK Fallback Integration ===")

    # Minimal document that should trigger AWK fallback
    minimal_document = """
    Some Business Name
    15/06/2024
    Total: $45.50
    """

    result = process_australian_tax_document(minimal_document)

    print(f"Success: {result.success}")
    print(f"Document Type: {result.document_type.value}")
    print(f"Extraction Method: {result.extraction_method}")
    print(f"AWK Fallback Used: {'hybrid' in result.extraction_method}")
    print(f"Handler Used: {result.handler_used}")
    print(f"Extracted Fields: {len(result.processing_result.extracted_fields)}")
    print(
        f"Sample Fields: {dict(list(result.processing_result.extracted_fields.items())[:3])}"
    )
    print()


def test_end_to_end_pipeline():
    """Test complete end-to-end processing pipeline."""
    print("=== Testing End-to-End Pipeline ===")

    documents = [
        """
        BP AUSTRALIA
        ABN: 12 345 678 901
        Date: 15/06/2024
        Fuel Type: Unleaded 91
        Litres: 45.230
        Total: $85.85
        GST: $7.81
        """,
        """
        TAX INVOICE
        ACME CONSULTING PTY LTD
        ABN: 12 345 678 901
        Date: 15/06/2024
        Subtotal: $500.00
        GST: $50.00
        Total: $550.00
        """,
        """
        WOOLWORTHS SUPERMARKETS
        ABN: 88 000 014 675
        Date: 15/06/2024
        Bread White: $3.50
        Milk 2L: $5.20
        Total: $17.65
        GST: $1.60
        """,
    ]

    # Process all documents
    extraction_results = []
    confidence_results = []

    for i, doc_text in enumerate(documents):
        print(f"--- Processing Document {i + 1} ---")

        # Step 1: Extract with hybrid manager
        extraction_result = process_australian_tax_document(doc_text)
        extraction_results.append(extraction_result)

        # Step 2: Assess confidence
        confidence_result = assess_document_confidence(doc_text, extraction_result)
        confidence_results.append(confidence_result)

        print(f"Document Type: {extraction_result.document_type.value}")
        print(f"Extraction Success: {extraction_result.success}")
        print(f"Confidence: {confidence_result.overall_confidence:.3f}")
        print(f"Production Ready: {confidence_result.production_readiness.value}")
        print(f"Processing Decision: {confidence_result.processing_decision}")
        print()

    # Summary statistics
    print("--- Pipeline Summary ---")
    successful_extractions = sum(1 for r in extraction_results if r.success)
    production_ready = sum(
        1
        for r in confidence_results
        if r.production_readiness.value in ["excellent", "good"]
    )
    avg_confidence = sum(r.overall_confidence for r in confidence_results) / len(
        confidence_results
    )

    print(f"Total Documents: {len(documents)}")
    print(f"Successful Extractions: {successful_extractions}/{len(documents)}")
    print(f"Production Ready: {production_ready}/{len(documents)}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print()


def main():
    """Run all Phase 2A integration tests."""
    print("🧪 Phase 2A Architecture Integration Tests")
    print("=" * 50)

    try:
        # Test individual handlers
        test_fuel_receipt_handler()
        test_tax_invoice_handler()
        test_business_receipt_handler()

        # Test hybrid extraction manager
        test_hybrid_extraction_manager()

        # Test confidence integration manager
        test_confidence_integration_manager()

        # Test AWK fallback integration
        test_awk_fallback_integration()

        # Test end-to-end pipeline
        test_end_to_end_pipeline()

        print("✅ All Phase 2A integration tests completed successfully!")

    except Exception as e:
        print(f"❌ Phase 2A integration test failed: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
