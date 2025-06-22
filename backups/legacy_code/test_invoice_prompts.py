#!/usr/bin/env python3
"""
Test script for invoice-specific prompts with extended token generation.
Tests various prompts and 256 token generation capability.
"""

import logging
import time

from models.extractors.llama_vision_extractor import LlamaVisionExtractor


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_invoice_prompts():
    """Test various invoice/receipt specific prompts with 256 tokens."""

    logger.info("🧾 TESTING INVOICE-SPECIFIC PROMPTS WITH 256 TOKENS")

    # Invoice-specific prompts to test
    test_prompts = {
        "store_name_simple": "What is the store name on this receipt?",
        "store_name_detailed": "Extract the business name from this receipt image. Provide only the business name.",
        "full_extraction": "Extract all information from this receipt including store name, date, total amount, items, and any other details you can see.",
        "structured_json": "Analyze this receipt and provide a JSON structure with: store_name, date, time, total_amount, payment_method, items_list, and tax_info.",
        "itemized_list": "List all individual items purchased on this receipt with their prices and quantities.",
        "receipt_summary": "Summarize this receipt in detail, including who purchased what, when, where, and how much was spent.",
        "business_analysis": "Describe what type of business this receipt is from and what categories of items were purchased.",
        "date_and_amount": "What is the date and total amount on this receipt?",
    }

    try:
        # Load model with extended token generation
        logger.info("📥 Loading model for extended token generation...")
        start_time = time.time()

        extractor = LlamaVisionExtractor(
            model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            use_8bit=False,
            max_new_tokens=256,  # Extended token generation
            device="cpu",
        )

        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded in {load_time:.1f} seconds")

        # Test each prompt
        results = {}

        for prompt_name, prompt_text in test_prompts.items():
            logger.info(f"\n🔍 Testing prompt: {prompt_name}")
            logger.info(f"📝 Prompt: {prompt_text}")

            test_start = time.time()

            # Use the internal _generate_response method for custom prompts
            image = extractor._preprocess_image("test_receipt.png")
            response = extractor._generate_response(prompt_text, image)

            test_time = time.time() - test_start

            # Log results
            logger.info(f"⏱️  Generation time: {test_time:.1f} seconds")
            logger.info(f"📄 Response ({len(response.split())} words):")
            logger.info(f"    {response}")

            # Store results
            results[prompt_name] = {
                "prompt": prompt_text,
                "response": response,
                "time": test_time,
                "word_count": len(response.split()),
                "char_count": len(response)
            }

            # Brief pause between tests
            time.sleep(2)

        # Summary analysis
        logger.info("\n" + "="*80)
        logger.info("📊 PROMPT TESTING SUMMARY")
        logger.info("="*80)

        for prompt_name, result in results.items():
            logger.info(f"\n🏷️  {prompt_name}:")
            logger.info(f"   ⏱️  Time: {result['time']:.1f}s")
            logger.info(f"   📝 Words: {result['word_count']}")
            logger.info(f"   📊 Chars: {result['char_count']}")
            logger.info(f"   📄 Response: {result['response'][:100]}...")

        # Find best prompts
        logger.info("\n🎯 BEST PROMPTS ANALYSIS:")

        # Most detailed response
        most_detailed = max(results.items(), key=lambda x: x[1]['word_count'])
        logger.info(f"📚 Most detailed: {most_detailed[0]} ({most_detailed[1]['word_count']} words)")

        # Fastest response
        fastest = min(results.items(), key=lambda x: x[1]['time'])
        logger.info(f"⚡ Fastest: {fastest[0]} ({fastest[1]['time']:.1f}s)")

        # Check for structured output
        structured_responses = {name: result for name, result in results.items()
                              if '{' in result['response'] or 'store' in result['response'].lower()}

        if structured_responses:
            logger.info(f"🏗️  Structured outputs: {list(structured_responses.keys())}")

        return results

    except Exception as e:
        logger.error(f"❌ Error during testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_field_extraction_extended():
    """Test the built-in field extraction methods with extended tokens."""

    logger.info("\n🔧 TESTING BUILT-IN FIELD EXTRACTION (256 TOKENS)")

    try:
        extractor = LlamaVisionExtractor(
            model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
            use_8bit=False,
            max_new_tokens=256,
            device="cpu",
        )

        # Test individual field extractions
        fields_to_test = ["store_name", "date", "time", "total", "items"]

        for field in fields_to_test:
            logger.info(f"\n🏷️  Testing field: {field}")
            start_time = time.time()

            result = extractor.extract_field("test_receipt.png", field)

            elapsed = time.time() - start_time
            logger.info(f"⏱️  Time: {elapsed:.1f}s")
            logger.info(f"📄 Result: {result}")

        # Test full extraction
        logger.info("\n🏗️  Testing full extraction...")
        start_time = time.time()

        full_result = extractor.extract_all_fields("test_receipt.png")

        elapsed = time.time() - start_time
        logger.info(f"⏱️  Time: {elapsed:.1f}s")
        logger.info("📄 Full result:")
        for key, value in full_result.items():
            logger.info(f"    {key}: {value}")

        return full_result

    except Exception as e:
        logger.error(f"❌ Error in field extraction: {e}")
        return None

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("🧾 INVOICE PROMPT TESTING WITH 256 TOKENS")
    logger.info("="*80)

    # Test 1: Custom prompts
    custom_results = test_invoice_prompts()

    # Test 2: Built-in field extraction
    field_results = test_field_extraction_extended()

    logger.info("\n" + "="*80)
    logger.info("🎉 TESTING COMPLETE")

    if custom_results and field_results:
        logger.info("✅ All tests completed successfully")
        logger.info("🎯 Model can now generate detailed invoice responses")
    else:
        logger.info("⚠️  Some tests failed - check logs above")

    logger.info("="*80)
