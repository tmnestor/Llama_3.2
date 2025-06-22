"""
CLI utilities and configuration for the Llama-Vision receipt extractor.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console


@dataclass
class RichConfig:
    """Configuration for rich console output."""

    console: Console = Console()
    success_style: str = "[bold green]\u2705[/bold green]"
    fail_style: str = "[bold red]\u274C[/bold red]"
    warning_style: str = "[bold yellow]\u26A0[/bold yellow]"
    info_style: str = "[bold blue]ℹ[/bold blue]"


def print_success(rich_config: RichConfig, message: str) -> None:
    """Print success message with standardized styling."""
    rich_config.console.print(f"{rich_config.success_style} {message}")


def print_error(rich_config: RichConfig, message: str) -> None:
    """Print error message with standardized styling."""
    rich_config.console.print(f"{rich_config.fail_style} {message}")


def print_warning(rich_config: RichConfig, message: str) -> None:
    """Print warning message with standardized styling."""
    rich_config.console.print(f"{rich_config.warning_style} {message}")


def print_info(rich_config: RichConfig, message: str) -> None:
    """Print info message with standardized styling."""
    rich_config.console.print(f"{rich_config.info_style} {message}")


def validate_model_path(model_path: str | None) -> Path:
    """
    Validate and return model path as Path object.
    
    Args:
        model_path: Model path string or None
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If model path is None or doesn't exist
    """
    if model_path is None:
        raise ValueError("Model path is required")

    path = Path(model_path)
    if not path.exists():
        raise ValueError(f"Model path does not exist: {path}")

    return path


def validate_input_path(input_path: str) -> Path:
    """
    Validate and return input path as Path object.
    
    Args:
        input_path: Input path string
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If input path doesn't exist
    """
    path = Path(input_path)
    if not path.exists():
        raise ValueError(f"Input path does not exist: {path}")

    return path


def ensure_output_dir(output_path: str) -> Path:
    """
    Ensure output directory exists and return as Path object.
    
    Args:
        output_path: Output directory path string
        
    Returns:
        Path object for output directory
    """
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_system_args(rich_config: RichConfig, args: dict[str, Any]) -> None:
    """
    Log system arguments and options using rich formatting.
    
    Args:
        rich_config: Rich configuration object
        args: Dictionary of arguments and their values
    """
    rich_config.console.print("\n[bold cyan]Configuration:[/bold cyan]")
    for key, value in args.items():
        if value is not None:
            rich_config.console.print(f"  {key}: [green]{value}[/green]")
        else:
            rich_config.console.print(f"  {key}: [dim]None[/dim]")
    rich_config.console.print()
