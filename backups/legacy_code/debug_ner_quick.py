#!/usr/bin/env python3
"""Quick debug script for NER without full CLI overhead."""

import os
import sys
from pathlib import Path


# Set environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from models.extractors.llama_vision_extractor import LlamaVisionExtractor


def main():
    print("Quick NER debug test")

    # Initialize extractor with very short timeout
    print("Loading model...")
    extractor = LlamaVisionExtractor(
        model_path="/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision",
        device="mps",
        max_new_tokens=50  # Very short response
    )
    print("Model loaded")

    # Test simple extraction
    print("Testing simple store name extraction...")
    try:
        result = extractor.extract_field("test_receipt.png", "store_name")
        print(f"Store name result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    print("Done")

if __name__ == "__main__":
    main()
