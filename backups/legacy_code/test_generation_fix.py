#!/usr/bin/env python3
"""Test generation with timeout fix."""

import os
import sys
from pathlib import Path


# Set environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from models.extractors.llama_vision_extractor import LlamaVisionExtractor


def main():
    print("Testing generation fix...")

    try:
        # Initialize with working model and MPS
        extractor = LlamaVisionExtractor(
            model_path="/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision",
            device="mps",
            max_new_tokens=20  # Very short response
        )
        print("Model loaded successfully")

        # Test store name extraction (simplest case)
        print("Testing store name extraction...")
        result = extractor.extract_field("test_receipt.png", "store_name")
        print(f"Store name result: '{result}'")

        if result and result.strip():
            print("SUCCESS! Generation is working")
            return True
        else:
            print("No result returned")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Ready to test NER system")
    else:
        print("\n❌ Still has generation issues")
