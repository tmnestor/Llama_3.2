#!/usr/bin/env python3
"""
Minimal test bypassing answer_question method
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def minimal_test():
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    model_path = "/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=None,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    )

    if torch.backends.mps.is_available():
        model = model.to("mps")
        print("Model on MPS")

    # Load and encode image
    image = Image.open("/Users/tod/Desktop/Llama_3.2/test_receipt.png")
    print(f"Image loaded: {image.size}")

    # Try to encode image
    if hasattr(model, 'encode_image'):
        print("Encoding image...")
        image_embeds = model.encode_image(image)
        print(f"Image encoded: {image_embeds.shape}")

        # Test generation with a simple prompt
        prompt = "Store name:"
        print(f"Testing prompt: '{prompt}'")

        try:
            # Use the generate method directly
            output_ids = model.generate(
                image_embeds=image_embeds,
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=5,  # Very short
                do_sample=False,
                temperature=0.1,
            )

            if isinstance(output_ids, list) and len(output_ids) > 0:
                response = output_ids[0]  # It returns decoded text directly
                print(f"SUCCESS: '{response}'")
            else:
                print(f"Got output_ids: {output_ids}")

        except Exception as e:
            print(f"Error in generate: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No encode_image method found")

if __name__ == "__main__":
    minimal_test()
