"""
Data handling module for Llama-Vision receipt extractor.
"""

# Import the data generators
try:
    from data.generators import (
        create_multimodal_data,
        receipt_generator,
        tax_document_generator,
    )
    
    __all__ = [
        "receipt_generator",
        "tax_document_generator", 
        "create_multimodal_data"
    ]
except ImportError:
    # Keep empty if generators aren't available
    __all__ = []