#!/bin/bash

# Quick test with improved settings
echo "🔧 Testing Improved NER Configuration"
echo "===================================="

# Activate conda environment
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate llama_vision_env

# Create results directory
mkdir -p results

# Quick test on invoice sample only
echo "💰 Quick Test: Invoice Sample with Key Entities"
python -m tax_invoice_ner.cli extract invoice_sample.png \
  --entity BUSINESS_NAME \
  --entity TOTAL_AMOUNT \
  --entity INVOICE_DATE \
  --entity ABN \
  --entity INVOICE_NUMBER \
  --output results/quick_test_invoice.json \
  --device auto \
  --verbose

echo ""
echo "✅ Quick test completed! Check results/quick_test_invoice.json"
echo "📊 Should be much faster and more accurate now."