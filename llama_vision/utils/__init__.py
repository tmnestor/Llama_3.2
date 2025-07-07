"""Utility module for Llama-3.2-Vision package."""

from .device import detect_device
from .logging import setup_logging

__all__ = ["detect_device", "setup_logging"]
