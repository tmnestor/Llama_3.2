#!/usr/bin/env python3
"""
Generate synthetic receipt dataset for training.

This script generates a dataset of synthetic receipt images and creates
appropriate train/val/test splits for model training.
"""

# NOTE: This file has been updated to use the new ab initio implementation
# of receipt and tax document generation. The old implementation in
# data.data_generators has been replaced with data.data_generators_new.
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path - must be done before local imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Local imports after path modification
from data.generators.receipt_generator import (  # noqa: E402
    generate_dataset as create_receipts,
)
from utils.cli import RichConfig, print_error, print_info, print_success  # noqa: E402

app = typer.Typer(help="Generate synthetic receipt datasets")
rich_config = RichConfig()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate that split ratios sum to 1.0."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-10:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@app.command()
def generate(
    output_dir: str = typer.Option(
        "datasets/synthetic_receipts",
        "--output-dir", "-o",
        help="Output directory for generated dataset"
    ),
    temp_dir: str = typer.Option(
        "data/raw/temp_receipts",
        "--temp-dir", "-t",
        help="Temporary directory for intermediate files (will be cleaned up)"
    ),
    num_collages: int = typer.Option(
        300,
        "--num-collages", "-n",
        help="Number of collages to generate"
    ),
    count_probs: str = typer.Option(
        "0.3,0.3,0.2,0.1,0.1,0",
        "--count-probs", "-p",
        help="Probability distribution for receipt counts (comma-separated)"
    ),
    stapled_ratio: float = typer.Option(
        0.3,
        "--stapled-ratio", "-s",
        help="Ratio of images that should have stapled receipts (0.0-1.0)"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility"
    ),
    image_size: int = typer.Option(
        2048,
        "--image-size", "-i",
        help="Output image size (default: 2048 for high-resolution)"
    ),
    train_ratio: float = typer.Option(
        0.7,
        "--train-ratio",
        help="Proportion for training set"
    ),
    val_ratio: float = typer.Option(
        0.15,
        "--val-ratio",
        help="Proportion for validation set"
    ),
    test_ratio: float = typer.Option(
        0.15,
        "--test-ratio",
        help="Proportion for test set"
    ),
    keep_temp: bool = typer.Option(
        False,
        "--keep-temp",
        help="Keep temporary files after generation"
    ),
):
    """Generate synthetic receipt dataset with specified parameters."""
    try:
        # Validate split ratios
        validate_split_ratios(train_ratio, val_ratio, test_ratio)
        
        # Create output directories
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory for intermediate files
        temp_dir_path = Path(temp_dir)
        temp_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set the random seed
        set_seed(seed)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=rich_config.console,
        ) as progress:
            # Generate receipts in temp directory first
            task = progress.add_task("Generating synthetic receipts...", total=None)
            
            # Parse probability distribution
            count_probs_list = [float(p) for p in count_probs.split(',')]
            
            # Generate the dataset using our optimized module
            create_receipts(
                output_dir=temp_dir_path,  # Use temp directory for initial generation
                num_collages=num_collages,
                count_probs=count_probs_list,
                image_size=image_size,
                stapled_ratio=stapled_ratio,
                seed=seed
            )
            
            # Copy final files from temp directory to output directory
            progress.update(task, description="Moving files to output directory...")
            
            # Move metadata.csv file
            if (temp_dir_path / "metadata.csv").exists():
                shutil.copy2(temp_dir_path / "metadata.csv", output_dir_path / "metadata.csv")
            
            # Move images directory
            if (temp_dir_path / "images").exists():
                # Create images directory in output if it doesn't exist
                (output_dir_path / "images").mkdir(exist_ok=True)
                
                # Copy all image files
                for img_file in (temp_dir_path / "images").glob("*.png"):
                    shutil.copy2(img_file, output_dir_path / "images" / img_file.name)
            
            # Clean up temporary files if not keeping them
            if not keep_temp:
                progress.update(task, description="Cleaning up temporary files...")
                try:
                    shutil.rmtree(temp_dir_path)
                    print_info(rich_config, f"Removed temporary directory: {temp_dir_path}")
                    
                    # Also try to remove parent directory if empty
                    parent_dir = temp_dir_path.parent
                    if parent_dir.exists() and not any(parent_dir.iterdir()):
                        parent_dir.rmdir()
                        print_info(rich_config, f"Removed empty parent directory: {parent_dir}")
                except Exception as e:
                    rich_config.console.print(f"[yellow]Warning: Failed to remove temporary files: {e}[/yellow]")
            
            progress.update(task, description="Dataset generation completed")
        
        # Display summary
        rich_config.console.print("\n[bold cyan]DATASET GENERATION SUMMARY[/bold cyan]")
        rich_config.console.print("=" * 50)
        rich_config.console.print(f"[bold]Images created:[/bold] {image_size}×{image_size} resolution")
        rich_config.console.print(f"[bold]Number of collages:[/bold] {num_collages}")
        rich_config.console.print(f"[bold]Output directory:[/bold] {output_dir_path}")
        
        print_success(rich_config, "Dataset generation completed successfully")
        
    except Exception as e:
        print_error(rich_config, f"Dataset generation failed: {e}")
        rich_config.console.print_exception()
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()