#!/usr/bin/env python3
"""
Test optimized prompts for the kadirnar/Llama-3.2-1B-Vision model
"""

from models.extractors.llama_vision_extractor import LlamaVisionExtractor
from PIL import Image


def test_different_prompts():
    print("🔄 Loading model...")
    extractor = LlamaVisionExtractor(
        model_path="/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision",
        device="auto",
        max_new_tokens=100
    )

    print(f"✅ Model loaded on device: {extractor.device}")

    # Load the receipt image
    image = Image.open("/Users/tod/Desktop/Llama_3.2/test_receipt.png")
    print(f"✅ Image loaded: {image.size}")

    # Test different prompt styles
    prompts_to_test = [
        {
            "name": "Simple Question",
            "prompt": "What is the store name on this receipt?"
        },
        {
            "name": "Direct Instruction",
            "prompt": "Look at this receipt image. Tell me the store name."
        },
        {
            "name": "Descriptive",
            "prompt": "Describe what you see in this receipt image."
        },
        {
            "name": "Store Name Only",
            "prompt": "Store name:"
        },
        {
            "name": "Receipt Analysis",
            "prompt": "Analyze this receipt and extract: store name, date, total amount"
        }
    ]

    for i, test in enumerate(prompts_to_test, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {test['name']}")
        print(f"Prompt: {test['prompt']}")
        print(f"{'='*50}")

        try:
            # Test with image encoding first
            print("🔄 Encoding image...")
            image_embeds = extractor.model.encode_image(image)
            print(f"✅ Image encoded: {image_embeds.shape}")

            # Test generation
            print("🔄 Generating response...")
            output_ids = extractor.model.generate(
                image_embeds=image_embeds,
                prompt=test['prompt'],
                tokenizer=extractor.tokenizer,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.1,
            )

            # Decode response
            response = extractor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Clean up response
            if test['prompt'] in response:
                response = response.replace(test['prompt'], "").strip()

            print(f"✅ Response: '{response}'")

            if response and response.strip():
                print(f"🎉 SUCCESS! Got response: {response[:100]}...")
                break
            else:
                print("❌ Empty response")

        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n{'='*50}")
    print("Test completed!")

if __name__ == "__main__":
    test_different_prompts()
