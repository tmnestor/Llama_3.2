# HuggingFace Model Download Guide

This guide explains how to use `huggingface_model_download.py` to download Llama-3.2-1B-Vision-Instruct and other HuggingFace models for offline use in the receipt extraction system.

## Overview

The `huggingface_model_download.py` script downloads HuggingFace models using git LFS, enabling offline model usage. This is essential for:

- **Remote GPU servers** without internet access
- **Reproducible deployments** with fixed model versions
- **Faster loading** by avoiding repeated downloads
- **Local development** with cached models

## Prerequisites

### 1. Environment Setup

Activate the conda environment:
```bash
conda activate llama_vision_env
```

### 2. Git LFS Installation

Install git Large File Storage (required for model weights):
```bash
# Install git-lfs via conda
conda install -c conda-forge git-lfs

# Initialize git-lfs
git lfs install
```

### 3. HuggingFace Authentication (if required)

For gated models like Llama-3.2, you may need authentication:
```bash
# Install HuggingFace CLI if not available
pip install huggingface_hub

# Login with your HuggingFace token
huggingface-cli login
```

## Usage

### Basic Syntax

```bash
python huggingface_model_download.py [MODEL_NAME] [OPTIONS]
```

### Download Llama-3.2-1B-Vision-Instruct

```bash
# Download to local PretrainedLLM directory (recommended for Mac M1/M2)
python huggingface_model_download.py meta-llama/Llama-3.2-1B-Vision-Instruct \
  --output-dir /Users/tod/PretrainedLLM/Llama-3.2-1B-Vision-Instruct

# Download to default cache location
python huggingface_model_download.py meta-llama/Llama-3.2-1B-Vision-Instruct

# Download without environment check (for automation)
python huggingface_model_download.py meta-llama/Llama-3.2-1B-Vision-Instruct \
  --no-check-env
```

### Available Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output-dir` | `-o` | Directory to save model files | `~/.cache/huggingface/models/` |
| `--check-env/--no-check-env` | | Check conda environment | `True` |
| `--help` | | Show help message | |

## Model Variants

### Llama-3.2 Vision Models

```bash
# 1B Vision-Instruct (recommended for Mac M1/M2 with 16GB RAM)
python huggingface_model_download.py meta-llama/Llama-3.2-1B-Vision-Instruct \
  --output-dir /Users/tod/PretrainedLLM/Llama-3.2-1B-Vision-Instruct

# 11B Vision-Instruct (requires 32GB+ RAM)
python huggingface_model_download.py meta-llama/Llama-3.2-11B-Vision-Instruct

# 90B Vision-Instruct (requires 256GB+ RAM)
python huggingface_model_download.py meta-llama/Llama-3.2-90B-Vision-Instruct
```

### Other Compatible Models

```bash
# LLaVA models
python huggingface_model_download.py llava-hf/llava-1.5-7b-hf
python huggingface_model_download.py llava-hf/llava-1.5-13b-hf

# InternVL2 models  
python huggingface_model_download.py OpenGVLab/InternVL2-8B
python huggingface_model_download.py OpenGVLab/InternVL2-26B
```

## Example Workflows

### 1. Local Mac Development Setup

```bash
# Download 1B model for Mac M1/M2 (16GB RAM compatible)
python huggingface_model_download.py meta-llama/Llama-3.2-1B-Vision-Instruct \
  --output-dir /Users/tod/PretrainedLLM/Llama-3.2-1B-Vision-Instruct

# Test the downloaded model
PYTHONPATH=. python scripts/test_llama_vision_extractor.py \
  --model-path /Users/tod/PretrainedLLM/Llama-3.2-1B-Vision-Instruct \
  test_receipt.png
```

### 2. Remote GPU Server Setup

```bash
# On local machine: Download and sync to remote
python huggingface_model_download.py meta-llama/Llama-3.2-11B-Vision-Instruct \
  --output-dir ./models/Llama-3.2-11B-Vision

# Sync to remote server
rsync -av ./models/ user@gpu-server:/efs/shared/models/

# On remote server: Use the synced model
python scripts/batch_extract_receipts.py \
  --model-path /efs/shared/models/Llama-3.2-11B-Vision \
  --use-8bit \
  receipt_images/
```

### 3. Evaluation Pipeline

```bash
# Download model
python huggingface_model_download.py meta-llama/Llama-3.2-11B-Vision-Instruct \
  --output-dir /shared/models/Llama-3.2-11B-Vision

# Run evaluation
python scripts/evaluation/evaluate_receipt_extractor.py \
  --model-path /shared/models/Llama-3.2-11B-Vision \
  --ground-truth datasets/synthetic_receipts/metadata.json \
  --sample-size 100 \
  --use-8bit
```

## Output and File Structure

### Downloaded Model Structure

```
model_directory/
├── config.json              # Model configuration
├── generation_config.json   # Generation parameters
├── model-*.safetensors      # Model weights (chunked)
├── preprocessor_config.json # Image preprocessing config
├── special_tokens_map.json  # Tokenizer special tokens
├── tokenizer.json          # Tokenizer
├── tokenizer_config.json   # Tokenizer configuration
└── .git/                   # Git metadata
```

### Model Size Information

| Model | Size | RAM Requirements |
|-------|------|------------------|
| Llama-3.2-1B-Vision | ~3 GB | 16GB (Mac M1/M2 compatible) |
| Llama-3.2-11B-Vision | ~22 GB | 32+ GB recommended |
| Llama-3.2-90B-Vision | ~180 GB | 256+ GB required |

## Troubleshooting

### Common Issues

#### 1. Git LFS Not Found
```bash
# Error: git lfs version command not found
conda install -c conda-forge git-lfs
git lfs install
```

#### 2. Authentication Required
```bash
# Error: Repository not found or access denied
huggingface-cli login
# Enter your HuggingFace token when prompted
```

#### 3. Insufficient Disk Space
```bash
# Check available space
df -h

# Clean up old downloads if needed
rm -rf ~/.cache/huggingface/transformers/
```

#### 4. Network Issues
```bash
# Retry with verbose output
GIT_TRACE=1 python huggingface_model_download.py model-name

# Use different mirror (if available)
export HF_ENDPOINT=https://hf-mirror.com
```

### Environment Verification

```bash
# Check conda environment
conda list | grep -E "(torch|transformers|git)"

# Verify git LFS
git lfs version

# Test HuggingFace access
huggingface-cli whoami
```

## Usage in Scripts

After downloading, use the model path in your scripts:

```python
# In Python scripts
from models.extractors.llama_vision_extractor import LlamaVisionExtractor

extractor = LlamaVisionExtractor(
    model_path="/path/to/downloaded/model",
    device="cuda",
    use_8bit=True
)
```

```bash
# In CLI scripts
python scripts/test_llama_vision_extractor.py \
  --model-path /path/to/downloaded/model \
  --use-8bit \
  receipt.jpg
```

## Integration with Project Scripts

The downloaded models work seamlessly with all project scripts:

- **`scripts/test_llama_vision_extractor.py`** - Single image testing
- **`scripts/batch_extract_receipts.py`** - Batch processing
- **`scripts/evaluation/evaluate_receipt_extractor.py`** - Model evaluation

Simply use the `--model-path` option to specify your downloaded model location.

## Best Practices

1. **Use specific output directories** for different model versions
2. **Verify downloads** by checking file sizes and running quick tests
3. **Document model versions** in your deployment configs
4. **Use 8-bit quantization** (`--use-8bit`) for memory efficiency
5. **Monitor disk space** as vision models are large (20+ GB)
6. **Test locally first** before deploying to remote servers

## Support

For issues specific to:
- **Model download**: Check git LFS installation and HuggingFace authentication
- **Model usage**: Refer to individual script documentation
- **Memory issues**: Use `--use-8bit` flag and ensure sufficient RAM
- **Permission errors**: Verify HuggingFace account has access to gated models