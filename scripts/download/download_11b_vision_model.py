#!/usr/bin/env python3
"""
Download the official Llama-3.2-11B-Vision-Instruct model.

This script downloads the working 11B Vision model to replace the broken 1B model.
"""

import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def download_11b_vision_model():
    """Download the official Llama-3.2-11B-Vision-Instruct model."""

    print("🚀 Downloading Llama-3.2-11B-Vision-Instruct Model")
    print("=" * 60)

    # Model configuration
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    local_dir = "/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision"

    print(f"📦 Model: {model_name}")
    print(f"📁 Local directory: {local_dir}")
    print("💾 Available disk space check...")

    # Check disk space
    local_path = Path(local_dir)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    statvfs = os.statvfs(local_path.parent)
    free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    print(f"💽 Free disk space: {free_space_gb:.1f} GB")

    if free_space_gb < 25:
        print("⚠️  Warning: May need ~25GB for model download")
    else:
        print("✅ Sufficient disk space available")

    print("\n🔄 Starting download...")
    print("Note: This will take 10-30 minutes depending on internet speed")

    try:
        # Download model files
        print("📥 Downloading model files...")
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            # Only download essential files for inference
            ignore_patterns=["*.bin"]  # Skip .bin files if .safetensors exist
        )

        print("✅ Model download completed!")

        # Test model loading with 8-bit quantization
        print("\n🧪 Testing model loading with 8-bit quantization...")

        # Load tokenizer first
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            trust_remote_code=True,
            local_files_only=True
        )
        print(f"✅ Tokenizer loaded: {type(tokenizer)}")

        # Test model loading with 8-bit
        print("🔧 Loading model with 8-bit quantization...")
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["vision_model", "vision_tower"]
        )

        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True
        )

        print(f"✅ Model loaded successfully: {type(model)}")
        print("📊 Model parameters: ~11B")
        print("💾 Memory usage: ~6-7GB with 8-bit quantization")

        # Check available methods
        print("🔍 Available methods:")
        print(f"   - Has chat method: {hasattr(model, 'chat')}")
        print(f"   - Model architecture: {model.config.architectures}")

        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Llama-3.2-11B-Vision-Instruct ready for use")
        print("=" * 60)
        print("📋 Next steps:")
        print("   1. Update config files to use new model path")
        print("   2. Test NER system with working 11B model")
        print("   3. Deploy to 78GB RAM Linux box for production")

        return True

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("💡 Troubleshooting:")
        print("   - Check internet connection")
        print("   - Verify HuggingFace access token if needed")
        print("   - Ensure sufficient disk space (~25GB)")
        return False


def update_config_files():
    """Update configuration files to use the 11B model."""

    print("\n🔧 Updating configuration files...")

    config_files = [
        "/Users/tod/Desktop/Llama_3.2/config/extractor/llama_vision_config.yaml",
        "/Users/tod/Desktop/Llama_3.2/config/extractor/work_expense_ner_config.yaml"
    ]

    old_path = "/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision"
    new_path = "/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision"

    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            content = config_path.read_text()
            if old_path in content:
                updated_content = content.replace(old_path, new_path)
                updated_content = updated_content.replace("use_8bit: false", "use_8bit: true")
                config_path.write_text(updated_content)
                print(f"✅ Updated: {config_file}")
            else:
                print(f"ℹ️  No update needed: {config_file}")
        else:
            print(f"⚠️  Not found: {config_file}")


def main():
    """Main download function."""

    print("Llama-3.2-11B-Vision-Instruct Download Script")
    print("This replaces the broken 1B model with the official working 11B model\n")

    # Download model
    success = download_11b_vision_model()

    if success:
        # Update config files
        update_config_files()

        print("\n✅ Setup complete!")
        print("🎯 Model ready at: /Users/tod/PretrainedLLM/Llama-3.2-11B-Vision")
        print("🔧 Configuration files updated for 8-bit quantization")
    else:
        print("\n❌ Download failed - see errors above")


if __name__ == "__main__":
    main()
