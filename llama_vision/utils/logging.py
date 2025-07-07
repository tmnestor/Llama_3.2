"""Logging configuration utilities."""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", format_style: str = "rich") -> logging.Logger:
    """Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_style: Format style ("rich" or "simple")

    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger("llama_vision")

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(numeric_level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    # Choose format based on style
    if format_style == "rich":
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
        )
    else:
        formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Get a logger instance with optional level override.

    Args:
        name: Logger name (usually module name)
        level: Optional logging level override

    Returns:
        Logger instance
    """
    logger = logging.getLogger(f"llama_vision.{name}")

    if level:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

    return logger


def log_system_info(logger: logging.Logger) -> None:
    """Log system information for debugging.

    Args:
        logger: Logger instance to use
    """
    import platform

    import torch

    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info(f"  CUDA: {torch.version.cuda}")
        logger.info(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    elif torch.backends.mps.is_available():
        logger.info("  MPS: Available")
    else:
        logger.info("  Acceleration: CPU only")


def log_model_info(logger: logging.Logger, model_path: str, config: dict) -> None:
    """Log model configuration information.

    Args:
        logger: Logger instance to use
        model_path: Path to model
        config: Model configuration dictionary
    """
    logger.info("Model Configuration:")
    logger.info(f"  Model Path: {model_path}")
    logger.info(f"  Device: {config.get('device', 'auto')}")
    logger.info(f"  Quantization: {config.get('use_quantization', False)}")
    logger.info(f"  Max Tokens: {config.get('max_tokens', 1024)}")
    logger.info(f"  Temperature: {config.get('temperature', 0.3)}")
    logger.info(f"  Memory Cleanup: {config.get('memory_cleanup_enabled', True)}")


def log_inference_metrics(
    logger: logging.Logger, image_path: str, response_length: int, inference_time: float
) -> None:
    """Log inference metrics.

    Args:
        logger: Logger instance to use
        image_path: Path to processed image
        response_length: Length of model response
        inference_time: Time taken for inference
    """
    logger.info("Inference Metrics:")
    logger.info(f"  Image: {image_path}")
    logger.info(f"  Response Length: {response_length} characters")
    logger.info(f"  Inference Time: {inference_time:.2f} seconds")

    if inference_time > 0:
        chars_per_second = response_length / inference_time
        logger.info(f"  Generation Speed: {chars_per_second:.1f} chars/second")
