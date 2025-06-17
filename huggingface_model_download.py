#!/usr/bin/env python3
"""
Utility script to pre-download model weights and configuration from HuggingFace.
This allows any supported model to be used in offline mode.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from utils.cli import RichConfig, print_error, print_info, print_success

app = typer.Typer(help="Download HuggingFace models for offline use")
rich_config = RichConfig()

# Set environment variable to disable NumPy 2.x compatibility warnings
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"


@app.command()
def download(
    model_name: str = typer.Argument(
        ..., 
        help="HuggingFace model name to download (e.g., meta-llama/Llama-3.2-11B-Vision-Instruct)"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Directory to save model files (default: ~/.cache/huggingface/models/)"
    ),
    check_env: bool = typer.Option(
        True,
        "--check-env/--no-check-env",
        help="Check if running in conda environment"
    ),
):
    """Download HuggingFace model for offline use with git LFS."""
    
    # Check conda environment
    if check_env and "CONDA_PREFIX" not in os.environ:
        print_error(rich_config, "Not running in a conda environment")
        rich_config.console.print("Please activate the llama_vision_env conda environment first:")
        rich_config.console.print("[dim]conda activate llama_vision_env[/dim]")
        raise typer.Exit(1)
    
    # Determine output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        # Set default location in cache
        output_path = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "models"
            / model_name.replace("/", "_")
        )
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Git URL for the model
    git_url = f"https://huggingface.co/{model_name}"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=rich_config.console,
    ) as progress:
        # Check git LFS
        task = progress.add_task("Checking git LFS installation...", total=None)
        try:
            subprocess.check_call("git lfs version", shell=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print_error(rich_config, "git-lfs not found")
            rich_config.console.print("Install git-lfs with: [dim]conda install -c conda-forge git-lfs[/dim]")
            raise typer.Exit(1)
        
        # Download model
        progress.update(task, description=f"Downloading {model_name}...")
        print_info(rich_config, f"Cloning from: {git_url}")
        print_info(rich_config, f"Saving to: {output_path}")
        
        cmd = f"git lfs install && git clone {git_url} {output_path}"
        
        try:
            subprocess.check_call(cmd, shell=True)
            progress.update(task, description="Download completed")
            
        except subprocess.CalledProcessError as e:
            print_error(rich_config, f"Download failed: {e}")
            rich_config.console.print("[yellow]Troubleshooting tips:[/yellow]")
            rich_config.console.print("  1. Check internet connection")
            rich_config.console.print("  2. Verify model name is correct")
            rich_config.console.print("  3. Ensure you have HuggingFace access to the model")
            rich_config.console.print("  4. Try: [dim]huggingface-cli login[/dim]")
            raise typer.Exit(1)
    
    # Display success and usage instructions
    rich_config.console.print("\n[bold cyan]MODEL DOWNLOAD SUMMARY[/bold cyan]")
    rich_config.console.print("=" * 50)
    rich_config.console.print(f"[bold]Model:[/bold] {model_name}")
    rich_config.console.print(f"[bold]Location:[/bold] {output_path}")
    
    # Check model size
    try:
        total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        rich_config.console.print(f"[bold]Size:[/bold] {size_gb:.1f} GB")
    except Exception:
        pass
    
    rich_config.console.print("\n[bold]Usage Instructions:[/bold]")
    rich_config.console.print(f"Use this path in your scripts: [green]{output_path}[/green]")
    rich_config.console.print("\n[bold]Example commands:[/bold]")
    rich_config.console.print("[dim]# Test extraction[/dim]")
    rich_config.console.print(f"python scripts/test_llama_vision_extractor.py --model-path {output_path} image.jpg")
    rich_config.console.print("\n[dim]# Batch processing[/dim]")
    rich_config.console.print(f"python scripts/batch_extract_receipts.py --model-path {output_path} images/")
    rich_config.console.print("\n[dim]# Evaluation[/dim]")
    rich_config.console.print(f"python scripts/evaluation/evaluate_receipt_extractor.py --model-path {output_path} metadata.json")
    
    print_success(rich_config, "Model download completed successfully")
    
    return str(output_path)


if __name__ == "__main__":
    app()
