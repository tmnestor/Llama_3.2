"""
Llama-3.2-Vision Package for Australian Tax Document Processing

A professional, modular package for processing business receipts and tax documents
using the Llama-3.2-Vision model. Designed for national taxation office requirements
with fair comparison capabilities against InternVL.
"""

__version__ = "0.1.0"

from .config import LlamaConfig, PromptManager, load_config
from .evaluation import InternVLComparison, PerformanceMetrics
from .extraction import KeyValueExtractor, TaxAuthorityParser
from .image import ImageLoader, preprocess_image_for_llama
from .model import LlamaInferenceEngine, LlamaModelLoader

__all__ = [
    "load_config",
    "LlamaConfig",
    "PromptManager",
    "LlamaModelLoader",
    "LlamaInferenceEngine",
    "KeyValueExtractor",
    "TaxAuthorityParser",
    "ImageLoader",
    "preprocess_image_for_llama",
    "InternVLComparison",
    "PerformanceMetrics",
]
