#!/bin/bash

# GPU-Optimized Entity Extraction Test Script
# Run this script to test the new GPU optimization on sample images

echo "🚀 Testing GPU-Optimized NER Extraction System"
echo "=============================================="

# Activate conda environment
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate llama_vision_env

# Create results directory
mkdir -p results

# Test 1: Bank Statement Sample with All Entities
echo "📊 Test 1: Bank Statement - Full Entity Extraction"
python -m tax_invoice_ner.cli extract bank_statement_sample.png \
  --output results/bank_statement_full_extraction.json \
  --device auto \
  --verbose

echo ""

# Test 2: Invoice Sample with All Entities  
echo "🧾 Test 2: Invoice - Full Entity Extraction"
python -m tax_invoice_ner.cli extract invoice_sample.png \
  --output results/invoice_full_extraction.json \
  --device auto \
  --verbose

echo ""

# Test 3: Bank Statement - Banking Entities Only
echo "🏦 Test 3: Bank Statement - Banking Entities Focus"
python -m tax_invoice_ner.cli extract bank_statement_sample.png \
  --entity ACCOUNT_NUMBER \
  --entity BSB \
  --entity BANK_NAME \
  --entity ACCOUNT_BALANCE \
  --entity WITHDRAWAL_AMOUNT \
  --entity DEPOSIT_AMOUNT \
  --entity TRANSACTION_DATE \
  --output results/bank_statement_banking_entities.json \
  --device auto \
  --verbose

echo ""

# Test 4: Invoice Sample - Financial Entities Only
echo "💰 Test 4: Invoice - Financial Entities Focus"
python -m tax_invoice_ner.cli extract invoice_sample.png \
  --entity TOTAL_AMOUNT \
  --entity SUBTOTAL \
  --entity TAX_AMOUNT \
  --entity TAX_RATE \
  --entity BUSINESS_NAME \
  --entity ABN \
  --entity INVOICE_DATE \
  --output results/invoice_financial_entities.json \
  --device auto \
  --verbose

echo ""

# Test 5: Configuration Validation
echo "✅ Test 5: Configuration Validation"
python -m tax_invoice_ner.cli validate-config

echo ""

# Test 6: List Available Entities
echo "📋 Test 6: Available Entity Types"
python -m tax_invoice_ner.cli list-entities

echo ""

# Test 7: Demo Mode
echo "🎪 Test 7: Demo Mode"
python -m tax_invoice_ner.cli demo --image invoice_sample.png

echo ""
echo "🎉 All tests completed! Check the results/ directory for output files."
echo "📊 With device: auto - will use CPU on Mac, CUDA when moved to KFP Discovery."