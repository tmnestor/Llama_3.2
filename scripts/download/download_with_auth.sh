#!/bin/bash
# Download Llama-3.2-11B-Vision-Instruct with authentication

echo "🔐 Setting up authentication for Llama model download"

# Activate conda environment
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate llama_vision_env

echo "📋 You'll need to provide your HuggingFace token"
echo "💡 Get it from: https://huggingface.co/settings/tokens"
echo ""

# Login to HuggingFace (will prompt for token)
huggingface-cli login

echo ""
echo "🚀 Starting download of Llama-3.2-11B-Vision-Instruct..."
echo "📁 Destination: /Users/tod/PretrainedLLM/Llama-3.2-11B-Vision"
echo "⏱️  This will take 20-40 minutes for ~22GB..."
echo ""

# Create directory
mkdir -p /Users/tod/PretrainedLLM/Llama-3.2-11B-Vision

# Download with progress
huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
  --local-dir /Users/tod/PretrainedLLM/Llama-3.2-11B-Vision \
  --resume

echo ""
echo "✅ Download complete!"
echo "📁 Model location: /Users/tod/PretrainedLLM/Llama-3.2-11B-Vision"
echo "🎯 Ready to test NER system with working 11B model!"