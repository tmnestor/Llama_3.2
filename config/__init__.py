"""
Configuration module for Llama-Vision receipt extractor.
"""
from pathlib import Path

import yaml


def load_config(config_path):
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    with config_file.open("r") as f:
        config = yaml.safe_load(f)
    return config
