"""
Tax Invoice Named Entity Recognition (NER) System.

A configurable NER system for extracting entities from tax invoices using
Llama-3.2-Vision with YAML-based entity configuration.
"""

from tax_invoice_ner.config.config_manager import ConfigManager
from tax_invoice_ner.extractors.work_expense_ner_extractor import WorkExpenseNERExtractor


__version__ = "1.0.0"
__author__ = "Tax Invoice NER Team"

__all__ = [
    "WorkExpenseNERExtractor",
    "ConfigManager",
]
