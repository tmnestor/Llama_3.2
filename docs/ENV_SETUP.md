# Environment Configuration Guide

This project supports multiple environment configurations for different hardware setups.

## Environment Files

### `.env` - Remote Multi-GPU Setup (Current)
- **Hardware**: Multi-GPU remote machine with 40GB+ VRAM
- **Model Size**: ~20GB (fp16 precision)
- **Configuration**: `TAX_INVOICE_NER_USE_8BIT=false`

### `.env.v100` - V100 16GB VRAM Setup
- **Hardware**: Single V100 GPU with 16GB VRAM
- **Model Size**: ~10GB (8-bit quantization)
- **Configuration**: `TAX_INVOICE_NER_USE_8BIT=true`

## Quick Setup

### For V100 Work Computer:
```bash
# Copy V100 configuration
cp .env.v100 .env

# Update paths in .env for your work computer:
# TAX_INVOICE_NER_BASE_PATH=/path/to/work/Llama_3.2
# TAX_INVOICE_NER_MODEL_PATH=/path/to/work/models/Llama-3.2-11B-Vision
```

### For Remote Multi-GPU:
```bash
# Keep current .env file
# Or ensure: TAX_INVOICE_NER_USE_8BIT=false
```

## Model Memory Usage

| Configuration | Memory Usage | Compatible Hardware |
|---------------|--------------|-------------------|
| fp32 (default) | ~40GB | High-end multi-GPU |
| fp16 | ~20GB | A100, H100, Multi-GPU |
| fp16 + 8-bit | ~10GB | V100, RTX 3090, RTX 4090 |

## Environment Variables

### Key Settings
- `TAX_INVOICE_NER_DEVICE`: `auto` (recommended), `cuda`, `cpu`, `mps`
- `TAX_INVOICE_NER_USE_8BIT`: `true` for V100, `false` for multi-GPU
- `TAX_INVOICE_NER_MODEL_PATH`: Path to Llama-3.2-11B-Vision model

### Path Configuration
Update these paths based on your setup:
- `TAX_INVOICE_NER_BASE_PATH`: Project root directory
- `TAX_INVOICE_NER_MODEL_PATH`: Model files location
- `TAX_INVOICE_NER_IMAGE_PATH`: Test images directory

## Performance Impact

### 8-bit Quantization vs fp16:
- **Memory**: 50% reduction (20GB → 10GB)
- **Speed**: Minimal impact on V100
- **Quality**: <5% degradation in most cases
- **V100 Compatibility**: ✅ Fits in 16GB VRAM

## Testing

Run the notebook cell 2 to verify your configuration:
- ✅ Model loads successfully
- ✅ Memory usage within GPU limits
- ✅ Device detection works correctly