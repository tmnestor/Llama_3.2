#!/bin/bash

# Quick test for bank statement with improved settings
echo "🏦 Testing Improved NER Configuration - Bank Statement"
echo "=================================================="

# Activate conda environment
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate llama_vision_env

# Create results directory
mkdir -p results

# Quick test on bank statement sample with banking-specific entities
echo "💳 Quick Test: Bank Statement Sample with Key Banking Entities"
python -m tax_invoice_ner.cli extract bank_statement_sample.png \
  --entity BANK_NAME \
  --entity ACCOUNT_NUMBER \
  --entity BSB \
  --entity ACCOUNT_HOLDER \
  --entity ACCOUNT_BALANCE \
  --entity WITHDRAWAL_AMOUNT \
  --entity DEPOSIT_AMOUNT \
  --entity TRANSACTION_DATE \
  --entity STATEMENT_PERIOD \
  --output results/quick_test_bank_statement.csv \
  --device auto \
  --verbose

echo ""
echo "✅ Bank statement quick test completed! Check results/quick_test_bank_statement.csv"
echo "📊 Should extract banking information much more accurately now."