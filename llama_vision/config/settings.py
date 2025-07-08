"""Configuration settings for Llama-3.2-Vision package."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env file in the project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from: {env_path}")
    else:
        print(f"⚠️  No .env file found at: {env_path}")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment variables only")


@dataclass
class LlamaConfig:
    """Llama-3.2-Vision configuration following InternVL pattern."""

    # Model configuration
    model_path: str
    device: str = "cuda"
    use_quantization: bool = False
    quantization_type: str = "int8"  # Options: "int8", "int4", "mixed_int8_int4"

    # Generation parameters
    max_tokens: int = 1024
    temperature: float = 0.3
    do_sample: bool = True
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    pad_token_id: int = -1

    # Path configuration
    base_path: str = ""
    image_path: str = ""
    output_path: str = ""
    config_path: str = ""

    # Processing settings
    classification_max_tokens: int = 20
    extraction_max_tokens: int = 1024
    memory_cleanup_enabled: bool = True
    process_batch_size: int = 1
    memory_cleanup_delay: int = 1

    # Australian compliance settings
    enable_abn_validation: bool = True
    enable_gst_validation: bool = True
    default_currency: str = "AUD"
    date_format: str = "DD/MM/YYYY"

    # Environment settings
    environment: str = "local"
    log_level: str = "INFO"
    enable_metrics: bool = True


def load_config() -> LlamaConfig:
    """Load configuration from environment variables following .env pattern."""

    # Get environment variables with defaults
    config = LlamaConfig(
        # Model configuration
        model_path=os.getenv(
            "TAX_INVOICE_NER_MODEL_PATH",
            "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision",
        ),
        device=os.getenv("TAX_INVOICE_NER_DEVICE", "cuda"),
        use_quantization=os.getenv("TAX_INVOICE_NER_USE_8BIT", "false").lower()
        == "true",
        quantization_type=os.getenv("TAX_INVOICE_NER_QUANTIZATION_TYPE", "int8"),
        # Generation parameters
        max_tokens=int(os.getenv("TAX_INVOICE_NER_MAX_TOKENS", "1024")),
        temperature=float(os.getenv("TAX_INVOICE_NER_TEMPERATURE", "0.3")),
        do_sample=os.getenv("TAX_INVOICE_NER_DO_SAMPLE", "true").lower() == "true",
        top_p=float(os.getenv("TAX_INVOICE_NER_TOP_P", "0.95")),
        top_k=int(os.getenv("TAX_INVOICE_NER_TOP_K", "50")),
        repetition_penalty=float(
            os.getenv("TAX_INVOICE_NER_REPETITION_PENALTY", "1.1")
        ),
        pad_token_id=int(os.getenv("TAX_INVOICE_NER_PAD_TOKEN_ID", "-1")),
        # Path configuration
        base_path=os.getenv(
            "TAX_INVOICE_NER_BASE_PATH", "/home/jovyan/nfs_share/tod/Llama_3.2"
        ),
        image_path=os.getenv(
            "TAX_INVOICE_NER_IMAGE_PATH", "/home/jovyan/nfs_share/tod/data/examples"
        ),
        output_path=os.getenv(
            "TAX_INVOICE_NER_OUTPUT_PATH", "/home/jovyan/nfs_share/tod/Llama_3.2/output"
        ),
        config_path=os.getenv(
            "TAX_INVOICE_NER_CONFIG_PATH",
            "/home/jovyan/nfs_share/tod/Llama_3.2/config/extractor/work_expense_ner_config.yaml",
        ),
        # Processing settings
        classification_max_tokens=int(
            os.getenv("TAX_INVOICE_NER_CLASSIFICATION_MAX_TOKENS", "20")
        ),
        extraction_max_tokens=int(
            os.getenv("TAX_INVOICE_NER_EXTRACTION_MAX_TOKENS", "1024")
        ),
        memory_cleanup_enabled=os.getenv(
            "TAX_INVOICE_NER_MEMORY_CLEANUP_ENABLED", "true"
        ).lower()
        == "true",
        process_batch_size=int(os.getenv("TAX_INVOICE_NER_PROCESS_BATCH_SIZE", "1")),
        memory_cleanup_delay=int(
            os.getenv("TAX_INVOICE_NER_MEMORY_CLEANUP_DELAY", "1")
        ),
        # Australian compliance settings
        enable_abn_validation=os.getenv(
            "TAX_INVOICE_NER_ENABLE_ABN_VALIDATION", "true"
        ).lower()
        == "true",
        enable_gst_validation=os.getenv(
            "TAX_INVOICE_NER_ENABLE_GST_VALIDATION", "true"
        ).lower()
        == "true",
        default_currency=os.getenv("TAX_INVOICE_NER_DEFAULT_CURRENCY", "AUD"),
        date_format=os.getenv("TAX_INVOICE_NER_DATE_FORMAT", "DD/MM/YYYY"),
        # Environment settings
        environment=os.getenv("TAX_INVOICE_NER_ENVIRONMENT", "local"),
        log_level=os.getenv("TAX_INVOICE_NER_LOG_LEVEL", "INFO"),
        enable_metrics=os.getenv("TAX_INVOICE_NER_ENABLE_METRICS", "true").lower()
        == "true",
    )

    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def resolve_path(path: str | Path, base_path: Optional[str] = None) -> Path:
    """Resolve a path relative to the project root or base path."""
    path = Path(path)

    if path.is_absolute():
        return path

    if base_path:
        return Path(base_path) / path

    return get_project_root() / path
