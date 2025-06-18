#!/usr/bin/env python3
"""
Test the answer_question method directly with optimized prompts
"""

from PIL import Image

from models.extractors.llama_vision_extractor import LlamaVisionExtractor


def test_answer_question():
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
    
    # Test simple prompts with answer_question
    test_prompts = [
        "What is the store name?",
        "What store is this receipt from?", 
        "What is the name of the business on this receipt?",
        "Tell me the store name on this receipt.",
        "Look at this receipt. What is the store name?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {prompt}")
        print(f"{'='*50}")
        
        try:
            print("🔄 Generating response...")
            
            # Use answer_question directly
            if hasattr(extractor.model, 'answer_question'):
                response = extractor.model.answer_question(
                    image=image,
                    question=prompt,
                    tokenizer=extractor.tokenizer,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.1,
                )
                
                print(f"✅ Response: '{response}'")
                
                if response and response.strip() and len(response.strip()) > 2:
                    print(f"🎉 SUCCESS! Got meaningful response: {response}")
                    break
                else:
                    print("❌ Empty or very short response")
            else:
                print("❌ No answer_question method found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*50}")
    print("Test completed!")

if __name__ == "__main__":
    test_answer_question()