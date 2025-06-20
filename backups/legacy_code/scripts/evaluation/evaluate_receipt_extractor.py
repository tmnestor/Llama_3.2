#!/usr/bin/env python3
"""
Evaluation script for receipt information extractor using Llama-Vision.

This script evaluates the zero-shot receipt extraction performance on synthetic datasets.
"""

import logging
import sys
from pathlib import Path

import torch
import typer
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn


# Add project root to path - must be done before local imports
project_root = Path(__file__).resolve().parent.parent.parent
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

from evaluation.receipt_extractor_evaluator import ReceiptExtractorEvaluator  # noqa: E402


app = typer.Typer(help="Evaluate Llama-Vision receipt extractor")
rich_config = RichConfig()


def configure_logging(log_level: str) -> None:
    """Configure logging based on log level."""
    numeric_level = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


@app.command()
def evaluate(
    ground_truth: str = typer.Argument(..., help="Path to ground truth data file or directory"),
    model_path: str = typer.Option(
        "/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision",
        "--model-path", "-m",
        help="Path to Llama-Vision model"
    ),
    output_dir: str = typer.Option(
        "evaluation_results",
        "--output-dir", "-o",
        help="Directory to save evaluation results"
    ),
    sample_size: int | None = typer.Option(
        None,
        "--sample-size", "-s",
        help="Number of samples to evaluate (default: all)"
    ),
    use_8bit: bool = typer.Option(
        False,
        "--use-8bit",
        help="Use 8-bit quantization for memory efficiency"
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "--device", "-d",
        help="Device to run evaluation on"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level", "-l",
        help="Logging level"
    ),
):
    """Evaluate Llama-Vision receipt extractor on ground truth data."""
    # Configure logging
    configure_logging(log_level)
    logger = logging.getLogger(__name__)

    # Log system arguments
    args_dict = {
        "ground_truth": ground_truth,
        "model_path": model_path,
        "output_dir": output_dir,
        "sample_size": sample_size,
        "use_8bit": use_8bit,
        "device": device,
        "log_level": log_level,
    }
    log_system_args(rich_config, args_dict)

    # Log system info
    print_info(rich_config, f"Device: {device}")
    print_info(rich_config, f"PyTorch version: {torch.__version__}")
    print_info(rich_config, f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print_info(rich_config, f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            print_info(rich_config, f"GPU {i}: {torch.cuda.get_device_name(i)}")

    try:
        # Validate paths
        model_path_obj = validate_model_path(model_path)
        gt_path_obj = validate_input_path(ground_truth)
        output_dir_obj = ensure_output_dir(output_dir)

        # Setup file logging
        file_handler = logging.FileHandler(output_dir_obj / "evaluation.log")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=rich_config.console,
        ) as progress:
            # Initialize receipt extractor
            task = progress.add_task("Initializing Llama-Vision extractor...", total=None)
            print_info(rich_config, f"Loading model from: {model_path_obj}")

            extractor = LlamaVisionExtractor(
                model_path=str(model_path_obj),
                device=device,
                use_8bit=use_8bit
            )
            progress.update(task, description="Model loaded successfully")

            # Initialize evaluator
            progress.update(task, description="Initializing evaluator...")
            print_info(rich_config, f"Loading ground truth from: {gt_path_obj}")

            evaluator = ReceiptExtractorEvaluator(
                extractor=extractor,
                ground_truth_path=str(gt_path_obj),
                output_dir=str(output_dir_obj)
            )

            # Run evaluation
            progress.update(task, description="Running evaluation...")
            print_info(rich_config, "Starting evaluation")

            metrics = evaluator.evaluate(sample_size=sample_size)

            progress.update(task, description="Evaluation completed")

        # Display results with rich formatting
        rich_config.console.print("\n[bold cyan]EVALUATION RESULTS[/bold cyan]")
        rich_config.console.print("=" * 50)

        if "overall" in metrics and metrics["overall"]:
            overall = metrics["overall"]
            rich_config.console.print(f"[bold]Overall Accuracy:[/bold] [green]{overall['accuracy']:.4f}[/green]")
            rich_config.console.print(f"[bold]Overall Match Rate:[/bold] [green]{overall['match_rate']:.4f}[/green]")
            rich_config.console.print(f"[bold]Error Rate:[/bold] [red]{overall['error_rate']:.4f}[/red]")
            rich_config.console.print(f"[bold]Total Fields Evaluated:[/bold] {overall['total_fields_evaluated']}")

        if "fields" in metrics and metrics["fields"]:
            rich_config.console.print("\n[bold]Per-Field Results:[/bold]")
            rich_config.console.print("-" * 30)
            for field, field_metrics in metrics["fields"].items():
                accuracy = field_metrics['accuracy']
                count = field_metrics['count']
                color = "green" if accuracy > 0.8 else "yellow" if accuracy > 0.5 else "red"
                rich_config.console.print(f"{field:15}: [{color}]{accuracy:.4f}[/{color}] (n={count})")

        rich_config.console.print("\n[bold]Results saved to:[/bold]")
        rich_config.console.print(f"  - Detailed results: {output_dir_obj}/detailed_results.json")
        rich_config.console.print(f"  - Metrics summary: {output_dir_obj}/metrics_summary.json")
        rich_config.console.print(f"  - Evaluation log: {output_dir_obj}/evaluation.log")

        print_success(rich_config, "Evaluation completed successfully")

    except Exception as e:
        print_error(rich_config, f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Full traceback:")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
