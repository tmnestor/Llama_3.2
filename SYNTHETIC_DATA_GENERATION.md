# Synthetic Data Generation for Receipt Extraction

This document describes the synthetic data generation process for evaluating the Llama-Vision zero-shot receipt extractor.

## Overview

We've successfully generated synthetic receipt datasets that can be used to evaluate the zero-shot extraction performance of the Llama-Vision model. This data generation runs locally and does not require the GPU or the actual model.

## Generated Datasets

### 1. Test Dataset (`datasets/test_synthetic_receipts/`)
- **Size**: 10 images
- **Purpose**: Quick testing and validation
- **Images with receipts**: 7
- **Images without receipts**: 3 (tax documents)
- **Total items**: 37 across all receipts

### 2. Evaluation Dataset (`datasets/synthetic_receipts/`)
- **Size**: 100 images
- **Purpose**: Comprehensive evaluation
- **Images with receipts**: 66
- **Images without receipts**: 34 (tax documents)
- **Total items**: 328 across all receipts
- **Distribution**: Stratified split (70% train, 15% val, 15% test)

## Dataset Structure

Each dataset contains:

```
datasets/synthetic_receipts/
├── images/                     # Receipt images (ignored by git)
│   ├── receipt_collage_00000.png
│   ├── receipt_collage_00001.png
│   └── ...
├── metadata.csv               # Basic metadata
└── metadata.json             # Enhanced metadata for extraction evaluation
```

## Metadata Format

### Enhanced Metadata (`metadata.json`)

Each entry contains detailed extraction fields:

```json
{
  "filename": "receipt_collage_00001.png",
  "image_path": "images/receipt_collage_00001.png", 
  "receipt_count": 4,
  "is_stapled": false,
  "store_name": "SPOTLIGHT",
  "date": "2025-06-05",
  "time": "16:26", 
  "total_amount": "$22.04",
  "payment_method": "VISA",
  "receipt_id": "6514-207175",
  "items": [
    {
      "item_name": "Pasta",
      "quantity": 3,
      "price": "$2.56"
    }
  ],
  "tax_info": "GST $2.00",
  "discounts": null
}
```

## Generation Commands

### Generate Images
```bash
# Activate environment
conda activate llama_vision_env

# Generate test dataset (10 images)
PYTHONPATH=. python scripts/data_generation/generate_data.py \
  --output_dir datasets/test_synthetic_receipts \
  --num_collages 10 \
  --seed 42

# Generate evaluation dataset (100 images)  
PYTHONPATH=. python scripts/data_generation/generate_data.py \
  --output_dir datasets/synthetic_receipts \
  --num_collages 100 \
  --seed 42
```

### Enhance Metadata
```bash
# Enhance test dataset metadata
PYTHONPATH=. python scripts/enhance_metadata_for_evaluation.py \
  --input datasets/test_synthetic_receipts/metadata.csv \
  --output datasets/test_synthetic_receipts/metadata.json \
  --seed 42

# Enhance evaluation dataset metadata
PYTHONPATH=. python scripts/enhance_metadata_for_evaluation.py \
  --input datasets/synthetic_receipts/metadata.csv \
  --output datasets/synthetic_receipts/metadata.json \
  --seed 42
```

## Features

### Realistic Receipt Content
- **Australian stores**: Woolworths, Coles, Bunnings, etc.
- **Australian items**: Tim Tams, Vegemite, etc. with realistic pricing
- **Payment methods**: VISA, EFTPOS, PAYWAVE, etc.
- **Tax information**: 10% GST calculations
- **Receipt formats**: Various layouts and styles

### Evaluation Fields
- **Store names**: From major Australian retailers
- **Dates**: Realistic dates within last 2 years
- **Times**: Business hours (8 AM - 9 PM)
- **Items**: 2-8 items per receipt with quantities and prices
- **Totals**: Calculated with GST
- **Payment methods**: Common Australian payment types
- **Receipt IDs**: Realistic transaction identifiers
- **Discounts**: 20% chance of discounts

### Image Characteristics
- **Resolution**: 2048x2048 pixels (high-resolution)
- **Receipt counts**: 0-4 receipts per image
- **Stapled receipts**: 10-20% are stapled
- **Tax documents**: ~30% are empty (ATO documents)

## Git Integration

The `.gitignore` file is configured to:
- **Include**: Metadata files (`.json`, `.csv`)
- **Exclude**: Image files (large, can be regenerated)
- **Sync**: Code and metadata via GitHub
- **Regenerate**: Images on remote host if needed

## Next Steps

1. **Commit metadata**: The metadata files will be synced to GitHub
2. **Remote generation**: Images can be regenerated on remote host using same commands
3. **Evaluation**: Use metadata to evaluate Llama-Vision extraction performance
4. **Analysis**: Compare extracted vs. ground truth data for accuracy metrics

## Usage for Evaluation

Once the remote GPU host has the Llama-Vision model, the evaluation can be run:

```bash
# On remote host with GPU
PYTHONPATH=. python scripts/evaluation/evaluate_receipt_extractor.py \
  --model-path /efs/shared/models/Llama-3.2-11B-Vision \
  --ground-truth datasets/synthetic_receipts/metadata.json \
  --output-dir evaluation_results \
  --use-8bit
```

This will provide comprehensive metrics on the zero-shot extraction accuracy across all receipt fields.