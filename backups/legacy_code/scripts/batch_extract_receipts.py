#!/usr/bin/env python3
"""
Batch processing script for receipt information extraction using Llama-Vision.

This script processes multiple receipt images and extracts structured information
from each one, saving results in various formats.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import typer
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn


# Add project root to path - must be done before local imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Local imports after path modification
from models.extractors.llama_vision_extractor import LlamaVisionExtractor  # noqa: E402
from utils.cli import RichConfig  # noqa: E402
from utils.cli import ensure_output_dir  # noqa: E402
from utils.cli import log_system_args  # noqa: E402
from utils.cli import print_error  # noqa: E402
from utils.cli import print_info  # noqa: E402
from utils.cli import print_success  # noqa: E402
from utils.cli import validate_input_path  # noqa: E402
from utils.cli import validate_model_path  # noqa: E402


app = typer.Typer(help="Batch process receipts with Llama-Vision extractor")
rich_config = RichConfig()

# Default file patterns - module level constant to avoid B008
DEFAULT_FILE_PATTERNS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]

# Typer option definitions - module level to avoid B008
FILE_PATTERNS_OPTION = typer.Option(
    None,
    help="File patterns to match for image files"
)


def configure_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def find_image_files(input_dir: Path, patterns: list[str]) -> list[Path]:
    """Find all image files matching the given patterns.
    
    Args:
        input_dir: Directory to search
        patterns: List of file patterns to match
        
    Returns:
        List of image file paths
    """
    image_files = []
    for pattern in patterns:
        image_files.extend(input_dir.glob(pattern))
        # Also search recursively
        image_files.extend(input_dir.glob(f"**/{pattern}"))

    # Remove duplicates and sort
    return sorted(list(set(image_files)))


@app.command()
def process(
    input_dir: str = typer.Argument(..., help="Directory containing receipt images"),
    model_path: str = typer.Option(
        "/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision",
        "--model-path", "-m",
        help="Path to Llama-Vision model"
    ),
    output_dir: str = typer.Option(
        "batch_extraction_results",
        "--output-dir", "-o",
        help="Directory to save extraction results"
    ),
    file_patterns: list[str] | None = FILE_PATTERNS_OPTION,
    use_8bit: bool = typer.Option(
        False,
        "--use-8bit",
        help="Use 8-bit quantization for memory efficiency"
    ),
    device: str = typer.Option(
        default="auto",
        help="Device to run extraction on"
    ),
    max_files: int | None = typer.Option(
        None,
        "--max-files",
        help="Maximum number of files to process (for testing)"
    ),
    save_individual: bool = typer.Option(
        False,
        "--save-individual",
        help="Save individual JSON files for each extraction"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
):
    """Batch process multiple receipt images for information extraction."""
    # Set default file patterns if not provided
    if file_patterns is None:
        file_patterns = DEFAULT_FILE_PATTERNS

    # Configure logging
    configure_logging(verbose)
    logger = logging.getLogger(__name__)

    # Log system arguments
    args_dict = {
        "input_dir": input_dir,
        "model_path": model_path,
        "output_dir": output_dir,
        "file_patterns": file_patterns,
        "use_8bit": use_8bit,
        "device": device,
        "max_files": max_files,
        "save_individual": save_individual,
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
        input_dir_obj = validate_input_path(input_dir)
        output_dir_obj = ensure_output_dir(output_dir)

        # Find image files
        print_info(rich_config, f"Searching for image files in {input_dir_obj}")
        image_files = find_image_files(input_dir_obj, file_patterns)

        if not image_files:
            print_error(rich_config, f"No image files found in {input_dir_obj} matching patterns {file_patterns}")
            raise typer.Exit(1)

        # Limit files if specified
        if max_files and max_files < len(image_files):
            image_files = image_files[:max_files]
            print_info(rich_config, f"Limited to {max_files} files for processing")

        print_info(rich_config, f"Found {len(image_files)} image files to process")

        # Initialize extractor
        print_info(rich_config, f"Initializing Llama-Vision extractor from {model_path_obj}")
        extractor = LlamaVisionExtractor(
            model_path=str(model_path_obj),
            device=device,
            use_8bit=use_8bit
        )

        # Process each image
        results = []
        failed_files = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=rich_config.console,
        ) as progress:
            task = progress.add_task("Processing receipts...", total=len(image_files))

            for image_file in image_files:
                progress.update(task, description=f"Processing: {image_file.name}")

                try:
                    # Extract information
                    extraction = extractor.extract_all_fields(str(image_file))

                    # Add metadata
                    extraction["image_filename"] = image_file.name
                    extraction["image_path"] = str(image_file.relative_to(input_dir_obj))
                    extraction["processing_status"] = "success"

                    results.append(extraction)

                    # Save individual file if requested
                    if save_individual:
                        individual_output = output_dir_obj / "individual" / f"{image_file.stem}.json"
                        individual_output.parent.mkdir(parents=True, exist_ok=True)
                        individual_output.write_text(json.dumps(extraction, indent=2, default=str))

                    logger.debug(f"Successfully processed {image_file.name}")

                except Exception as e:
                    error_msg = f"Failed to process {image_file.name}: {e}"
                    logger.error(error_msg)
                    failed_files.append({
                        "filename": image_file.name,
                        "path": str(image_file.relative_to(input_dir_obj)),
                        "error": str(e)
                    })

                    # Add failed entry to results for completeness
                    results.append({
                        "image_filename": image_file.name,
                        "image_path": str(image_file.relative_to(input_dir_obj)),
                        "processing_status": "failed",
                        "error": str(e),
                        **dict.fromkeys(["store_name", "date", "time", "total_amount", "payment_method", "receipt_id", "tax_info", "discounts"]),
                        "items": [],
                    })

                progress.advance(task)

        # Save consolidated results
        print_info(rich_config, "Saving consolidated results")

        # Save as JSON
        (output_dir_obj / "batch_results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )

        # Save as CSV for easy analysis
        results_df = pd.json_normalize(results)
        results_df.to_csv(output_dir_obj / "batch_results.csv", index=False)

        # Save summary statistics
        successful_extractions = len([r for r in results if r.get("processing_status") == "success"])
        failed_extractions = len([r for r in results if r.get("processing_status") == "failed"])

        summary = {
            "total_files": len(image_files),
            "successful_extractions": successful_extractions,
            "failed_extractions": failed_extractions,
            "success_rate": successful_extractions / len(image_files) if image_files else 0,
            "failed_files": failed_files
        }

        (output_dir_obj / "processing_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        # Display summary with rich formatting
        rich_config.console.print("\n[bold cyan]BATCH PROCESSING SUMMARY[/bold cyan]")
        rich_config.console.print("=" * 50)
        rich_config.console.print(f"[bold]Total files processed:[/bold] {summary['total_files']}")
        rich_config.console.print(f"[bold green]Successful extractions:[/bold green] {summary['successful_extractions']}")
        rich_config.console.print(f"[bold red]Failed extractions:[/bold red] {summary['failed_extractions']}")
        rich_config.console.print(f"[bold]Success rate:[/bold] {summary['success_rate']:.2%}")

        if failed_files:
            rich_config.console.print("\n[bold red]Failed files:[/bold red]")
            for failed in failed_files:
                rich_config.console.print(f"  - [red]{failed['filename']}[/red]: {failed['error']}")

        rich_config.console.print("\n[bold]Results saved to:[/bold]")
        rich_config.console.print(f"  - JSON: {output_dir_obj}/batch_results.json")
        rich_config.console.print(f"  - CSV: {output_dir_obj}/batch_results.csv")
        rich_config.console.print(f"  - Summary: {output_dir_obj}/processing_summary.json")

        if save_individual:
            rich_config.console.print(f"  - Individual files: {output_dir_obj}/individual/")

        print_success(rich_config, "Batch processing completed successfully")

    except Exception as e:
        print_error(rich_config, f"Batch processing failed: {e}")
        if verbose:
            rich_config.console.print_exception()
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
