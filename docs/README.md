# Llama-Vision Receipt Extractor

Zero-shot receipt information extraction using Llama-3.2-11B-Vision. This system extracts structured information from receipts without requiring any receipt-specific training, operating purely through prompt engineering and the pre-trained capabilities of Llama-Vision.

## Features

- **Zero-shot extraction**: No training required, works out of the box
- **Comprehensive field extraction**: Store name, date, total, items, payment method, and more
- **Batch processing**: Process multiple receipts efficiently
- **Evaluation framework**: Comprehensive metrics for accuracy assessment
- **Memory efficient**: 8-bit quantization support for limited GPU memory

## Quick Start

### Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate llama_vision_env

# For remote GPU host with CUDA, also install:
# pip install bitsandbytes  # For 8-bit quantization
```

### Download Llama-3.2-11B-Vision Model

```bash
# Download model for offline use (requires ~22GB disk space)
python huggingface_model_download.py meta-llama/Llama-3.2-11B-Vision-Instruct \
  --output-dir /path/to/models/Llama-3.2-11B-Vision

# See HUGGINGFACE_MODEL_DOWNLOAD.md for detailed instructions
```

### Generate Synthetic Data (Local)

```bash
# Generate synthetic receipts for evaluation (can be done locally)
PYTHONPATH=. python scripts/data_generation/generate_data.py \
  --output_dir datasets/synthetic_receipts \
  --num_collages 100 \
  --seed 42

# Enhance metadata for extraction evaluation
PYTHONPATH=. python scripts/enhance_metadata_for_evaluation.py \
  --input datasets/synthetic_receipts/metadata.csv \
  --output datasets/synthetic_receipts/metadata.json \
  --seed 42
```

### Model Extraction (Remote GPU Host Only)

```bash
# Extract from single receipt
PYTHONPATH=. python scripts/test_llama_vision_extractor.py \
  --model-path /efs/shared/models/Llama-3.2-11B-Vision \
  --image-path receipt.jpg \
  --use-8bit

# Batch process multiple receipts
PYTHONPATH=. python scripts/batch_extract_receipts.py \
  --model-path /efs/shared/models/Llama-3.2-11B-Vision \
  --input-dir receipts/ \
  --output-dir results/ \
  --use-8bit

# Evaluate on synthetic dataset
PYTHONPATH=. python scripts/evaluation/evaluate_receipt_extractor.py \
  --model-path /efs/shared/models/Llama-3.2-11B-Vision \
  --ground-truth datasets/synthetic_receipts/metadata.json \
  --use-8bit
```

## Extracted Fields

The system extracts the following information from receipts:

- **Store/Business Name**
- **Date of Purchase** (YYYY-MM-DD format)
- **Time of Purchase** (HH:MM format)
- **Total Amount** (with currency)
- **Payment Method**
- **Receipt Number/ID**
- **Individual Items** (with prices and quantities)
- **Tax Information**
- **Discounts/Promotions**

## Requirements

- **GPU Memory**: 20GB+ with 8-bit quantization, 40GB+ without
- **CUDA**: Version 11.8 or higher
- **Python**: 3.11+
- **PyTorch**: 2.0+

## Project Structure

```
├── models/extractors/           # Llama-Vision extraction models
├── evaluation/                  # Evaluation framework
├── scripts/
│   ├── evaluation/             # Evaluation scripts
│   ├── data_generation/        # Synthetic data generation
│   ├── test_llama_vision_extractor.py
│   └── batch_extract_receipts.py
├── config/extractor/           # Configuration files
├── data/generators/            # Data generation utilities
└── utils/                      # Utility functions
```

## Performance

- **Processing time**: 2-5 seconds per receipt
- **Throughput**: ~15,000-20,000 receipts per day (single GPU)
- **Accuracy**: Varies by receipt quality and field type
- **Memory usage**: 20GB VRAM with 8-bit quantization

## Configuration

The system uses YAML configuration files in `config/extractor/` for:
- Model settings (path, quantization, device)
- Extraction parameters (fields, prompts, preprocessing)
- Post-processing options (format standardization, validation)
- Evaluation metrics and thresholds

## Contributing

This project focuses on zero-shot extraction capabilities. To contribute:

1. Test on new receipt types and formats
2. Improve prompt engineering for better accuracy
3. Enhance post-processing and validation
4. Optimize memory usage and performance

## License

This project is for research and educational purposes.