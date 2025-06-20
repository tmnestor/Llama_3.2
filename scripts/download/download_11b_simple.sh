#!/bin/bash
# Simple download script for Llama-3.2-11B-Vision-Instruct

echo "🚀 Downloading Llama-3.2-11B-Vision-Instruct"
echo "This will take 20-40 minutes..."

# Activate conda environment
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate llama_vision_env

# Create directory
mkdir -p /Users/tod/PretrainedLLM/Llama-3.2-11B-Vision

# Use huggingface-cli to download
echo "📥 Starting download with huggingface-cli..."
huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
  --local-dir /Users/tod/PretrainedLLM/Llama-3.2-11B-Vision \
  --resume

echo "✅ Download complete!"
echo "📁 Model location: /Users/tod/PretrainedLLM/Llama-3.2-11B-Vision"