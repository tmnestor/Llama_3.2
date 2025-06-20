#!/bin/bash
# Download with token as environment variable

echo "🔐 To download, first set your token:"
echo "export HF_TOKEN='your_huggingface_token_here'"
echo "Then run: ./download_with_token.sh"
echo ""

if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN environment variable not set"
    echo "💡 Set it like this:"
    echo "export HF_TOKEN='hf_your_token_here'"
    echo "Then run this script again"
    exit 1
fi

echo "✅ Token found, starting download..."

# Activate conda environment
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate llama_vision_env

# Create directory
mkdir -p /Users/tod/PretrainedLLM/Llama-3.2-11B-Vision

# Login with token
echo $HF_TOKEN | huggingface-cli login --token $HF_TOKEN

# Download
echo "📥 Downloading Llama-3.2-11B-Vision-Instruct..."
huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
  --local-dir /Users/tod/PretrainedLLM/Llama-3.2-11B-Vision \
  --resume

echo "✅ Download complete!"