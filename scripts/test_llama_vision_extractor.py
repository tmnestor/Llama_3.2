#!/usr/bin/env python3
"""
Test script for Llama-Vision receipt extractor.

This script tests the zero-shot receipt extraction with sample images.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path - must be done before local imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Local imports after path modification
from models.extractors.llama_vision_extractor import LlamaVisionExtractor  # noqa: E402
from utils.cli import (  # noqa: E402
    RichConfig,
    ensure_output_dir,
    log_system_args,
    print_error,
    print_info,
    print_success,
    validate_input_path,
    validate_model_path,
)

app = typer.Typer(help="Test Llama-Vision receipt extractor")
rich_config = RichConfig()


def configure_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


@app.command()
def extract(
    image_path: str = typer.Argument(..., help="Path to receipt image"),
    model_path: str = typer.Option(
        "/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision",
        "--model-path", "-m",
        help="Path to Llama-Vision model"
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output-file", "-o",
        help="Path to save extraction results (JSON format)"
    ),
    field: Optional[str] = typer.Option(
        None,
        "--field", "-f",
        help="Extract specific field only (e.g., store_name, date, total)"
    ),
    use_8bit: bool = typer.Option(
        False,
        "--use-8bit",
        help="Use 8-bit quantization for memory efficiency"
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "--device", "-d",
        help="Device to run extraction on"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
):
    """Test Llama-Vision receipt extractor on a single image."""
    # Configure logging
    configure_logging(verbose)
    logging.getLogger(__name__)
    
    # Log system arguments
    args_dict = {
        "image_path": image_path,
        "model_path": model_path,
        "output_file": output_file,
        "field": field,
        "use_8bit": use_8bit,
        "device": device,
        "verbose": verbose,
    }
    log_system_args(rich_config, args_dict)
    
    # Log system info
    print_info(rich_config, f"Device: {device}")
    print_info(rich_config, f"PyTorch version: {torch.__version__}")
    print_info(rich_config, f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        # Validate paths
        model_path_obj = validate_model_path(model_path)
        image_path_obj = validate_input_path(image_path)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=rich_config.console,
        ) as progress:
            # Initialize extractor
            task = progress.add_task("Initializing Llama-Vision extractor...", total=None)
            print_info(rich_config, f"Loading model from: {model_path_obj}")
            
            extractor = LlamaVisionExtractor(
                model_path=str(model_path_obj),
                device=device,
                use_8bit=use_8bit
            )
            progress.update(task, description="Model loaded successfully")
            
            # Perform extraction
            progress.update(task, description=f"Processing image: {image_path_obj.name}")
            
            if field:
                # Extract specific field
                print_info(rich_config, f"Extracting field: {field}")
                result = extractor.extract_field(str(image_path_obj), field)
                
                rich_config.console.print(f"\n[bold]Extracted {field}:[/bold] [green]{result}[/green]")
                
                output_data = {
                    "image_path": str(image_path_obj),
                    "field": field,
                    "extracted_value": result
                }
            else:
                # Extract all fields
                print_info(rich_config, "Extracting all fields")
                result = extractor.extract_all_fields(str(image_path_obj))
                
                # Print results with rich formatting
                rich_config.console.print("\n[bold cyan]EXTRACTION RESULTS[/bold cyan]")
                rich_config.console.print("=" * 50)
                
                for field_name, value in result.items():
                    if field_name == "raw_extraction":
                        continue  # Skip raw extraction in summary
                    if field_name == "items" and isinstance(value, list):
                        rich_config.console.print(f"[bold]{field_name:15}:[/bold] {len(value)} items")
                        for _i, item in enumerate(value[:5]):  # Show first 5 items
                            item_name = item.get("item_name", "Unknown")
                            item_price = item.get("price", "Unknown")
                            rich_config.console.print(f"{'':17}- [green]{item_name}[/green]: [yellow]{item_price}[/yellow]")
                        if len(value) > 5:
                            rich_config.console.print(f"{'':17}... and {len(value) - 5} more items")
                    else:
                        color = "green" if value else "dim"
                        rich_config.console.print(f"[bold]{field_name:15}:[/bold] [{color}]{value}[/{color}]")
                
                output_data = result.copy()
                output_data["image_path"] = str(image_path_obj)
            
            progress.update(task, description="Extraction completed")
        
        # Save results if requested
        if output_file:
            output_path = ensure_output_dir(Path(output_file).parent) / Path(output_file).name
            
            output_path.write_text(json.dumps(output_data, indent=2, default=str))
            
            print_success(rich_config, f"Results saved to: {output_path}")
        
        print_success(rich_config, "Extraction completed successfully")
        
    except Exception as e:
        print_error(rich_config, f"Extraction failed: {e}")
        if verbose:
            rich_config.console.print_exception()
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()