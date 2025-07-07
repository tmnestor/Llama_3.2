#!/usr/bin/env python3
"""
Debug script for Llama-3.2-Vision model integrity and vision processing.
"""

import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration


def check_model_integrity(model_path: str):
    """Check model files and configuration integrity."""
    model_path = Path(model_path)

    print("🔍 MODEL INTEGRITY CHECK")
    print("=" * 40)

    # Check model files
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    model_files = list(model_path.glob("*.safetensors")) + list(
        model_path.glob("*.bin")
    )

    print(f"📁 Model directory: {model_path}")
    print(f"📄 Config files: {[f.name for f in model_path.glob('*.json')]}")
    print(f"🔧 Model files: {len(model_files)} found")
    print(f"📝 Tokenizer files: {[f.name for f in model_path.glob('tokenizer*')]}")

    for file in required_files:
        if (model_path / file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING")

    return len(model_files) > 0


def test_model_loading(model_path: str):
    """Test model loading without inference."""
    print("\n🚀 MODEL LOADING TEST")
    print("=" * 40)

    try:
        # Load processor first
        print("📋 Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        print("   ✅ Processor loaded")

        # Check processor components
        print(f"   📝 Tokenizer vocab size: {len(processor.tokenizer)}")
        print(f"   🖼️  Image processor: {type(processor.image_processor).__name__}")

        # Test tokenizer on vision tokens
        test_text = "<|image|>What is in this image?"
        tokens = processor.tokenizer(test_text, return_tensors="pt")
        print(f"   🔤 Test tokenization: {tokens['input_ids'].shape}")

        # Load model with minimal config
        print("\n🤖 Loading model...")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            device_map=None,  # Load to CPU first
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        print("   ✅ Model loaded to CPU")

        # Check model components
        print(f"   🧠 Model type: {type(model).__name__}")
        print(
            f"   📊 Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B"
        )

        # Check vision model
        if hasattr(model, "vision_model"):
            print("   👁️  Vision model: Present")
        else:
            print("   ❌ Vision model: Missing")

        # Check language model
        if hasattr(model, "language_model"):
            print("   💬 Language model: Present")
        else:
            print("   ❌ Language model: Missing")

        return model, processor

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, None


def test_cpu_inference(model, processor, image_path: str):
    """Test inference on CPU to isolate CUDA issues."""
    print("\n🖥️  CPU INFERENCE TEST")
    print("=" * 40)

    if model is None or processor is None:
        print("❌ Model/processor not available")
        return False

    try:
        # Load a simple test image
        if not Path(image_path).exists():
            print(f"❌ Test image not found: {image_path}")
            return False

        image = Image.open(image_path).convert("RGB")
        print(f"🖼️  Image loaded: {image.size}")

        # Simple prompt
        prompt = "<|image|>What is in this image?"

        # Process on CPU
        print("🔄 Processing inputs on CPU...")
        inputs = processor(text=prompt, images=image, return_tensors="pt")

        print(f"   📥 Input IDs shape: {inputs['input_ids'].shape}")
        print(f"   🖼️  Pixel values shape: {inputs['pixel_values'].shape}")

        # Try CPU generation with minimal tokens
        print("🧮 Testing CPU generation...")
        model.eval()

        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,  # Very short generation
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

                response = processor.decode(
                    outputs[0][inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                )

                print(f"   ✅ CPU generation successful: '{response}'")
                return True

            except Exception as e:
                print(f"   ❌ CPU generation failed: {e}")
                return False

    except Exception as e:
        print(f"❌ CPU test failed: {e}")
        return False


def main():
    """Main diagnostic function."""
    # Use environment variable for model path
    model_path = os.getenv(
        "TAX_INVOICE_NER_MODEL_PATH",
        "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision",
    )
    image_path = (
        os.getenv(
            "TAX_INVOICE_NER_IMAGE_PATH", "/home/jovyan/nfs_share/tod/data/examples"
        )
        + "/test_receipt.png"
    )

    print("🩺 LLAMA-3.2-VISION DIAGNOSTIC")
    print("=" * 50)
    print(f"🎯 Model path: {model_path}")
    print(f"🖼️  Test image: {image_path}")

    # Step 1: Check model integrity
    if not check_model_integrity(model_path):
        print("\n❌ Model files incomplete - cannot proceed")
        return

    # Step 2: Test model loading
    model, processor = test_model_loading(model_path)

    # Step 3: Test CPU inference
    if model and processor:
        cpu_success = test_cpu_inference(model, processor, image_path)

        if cpu_success:
            print("\n✅ CPU inference works - CUDA issue is device-specific")
            print("💡 Recommendations:")
            print("   1. Try different CUDA device mapping")
            print("   2. Check GPU memory constraints")
            print("   3. Use CPU inference as fallback")
        else:
            print("\n❌ CPU inference also fails - model checkpoint issue")
            print("💡 Recommendations:")
            print("   1. Re-download model checkpoint")
            print("   2. Verify transformers version compatibility")
            print("   3. Try different model variant")

    print("\n🏁 Diagnostic complete")


if __name__ == "__main__":
    main()
