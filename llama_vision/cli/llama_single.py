"""Single image processing CLI for Llama-3.2-Vision package."""

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import PromptManager, load_config
from ..evaluation import InternVLComparison
from ..extraction import KeyValueExtractor, TaxAuthorityParser
from ..image import ImageLoader
from ..model import LlamaInferenceEngine, LlamaModelLoader

app = typer.Typer(help="Single image processing with Llama-3.2-Vision")
console = Console()


@app.command()
def extract(
    image_path: str = typer.Argument(..., help="Path to image file"),
    prompt_name: str = typer.Option(
        "key_value_receipt_prompt", help="Prompt to use from prompts.yaml"
    ),
    output_file: Optional[str] = typer.Option(None, help="Output file path (JSON)"),
    extraction_method: str = typer.Option(
        "tax_authority", help="Extraction method: key_value, tax_authority, or json"
    ),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
    compare_internvl: bool = typer.Option(False, help="Run InternVL comparison test"),
):
    """Extract information from a single image."""

    # Setup
    log_level = "DEBUG" if verbose else "INFO"
    config = load_config()
    console.print("[blue]Loading Llama-3.2-Vision model...[/blue]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Load model
            load_task = progress.add_task("Loading model and processor...", total=None)
            loader = LlamaModelLoader(config)
            model, processor = loader.load_model()
            progress.update(load_task, description="✅ Model loaded")

            # Initialize components
            init_task = progress.add_task("Initializing components...", total=None)
            inference_engine = LlamaInferenceEngine(model, processor, config)
            prompt_manager = PromptManager()
            progress.update(init_task, description="✅ Components initialized")

            # Validate image
            validate_task = progress.add_task("Validating image...", total=None)
            image_loader = ImageLoader(log_level)
            if not Path(image_path).exists():
                console.print(f"[red]Error: Image file not found: {image_path}[/red]")
<<<<<<< HEAD
                raise typer.Exit(1)
=======
                raise typer.Exit(1) from None
>>>>>>> 53cbe49 (✨ feat: Add comprehensive llama_vision package with CLI tools and document extraction capabilities)
            progress.update(validate_task, description="✅ Image validated")

            # Get prompt
            prompt_task = progress.add_task("Loading prompt...", total=None)
            try:
                prompt = prompt_manager.get_prompt(prompt_name)
            except KeyError:
                available_prompts = prompt_manager.list_prompts()
                console.print(f"[red]Error: Prompt '{prompt_name}' not found.[/red]")
                console.print(
                    f"Available prompts: {', '.join(available_prompts[:5])}..."
                )
<<<<<<< HEAD
                raise typer.Exit(1)
=======
                raise typer.Exit(1) from None
>>>>>>> 53cbe49 (✨ feat: Add comprehensive llama_vision package with CLI tools and document extraction capabilities)
            progress.update(prompt_task, description="✅ Prompt loaded")

            # Run inference
            inference_task = progress.add_task("Running inference...", total=None)
            start_time = time.time()
            response = inference_engine.predict(image_path, prompt)
            inference_time = time.time() - start_time
            progress.update(inference_task, description="✅ Inference complete")

            # Extract data
            extract_task = progress.add_task("Extracting data...", total=None)

            if extraction_method == "key_value":
                extractor = KeyValueExtractor(log_level)
                extracted_data = extractor.extract(response)
            elif extraction_method == "tax_authority":
                parser = TaxAuthorityParser(log_level)
                extracted_data = parser.parse_receipt_response(response)
            elif extraction_method == "json":
                from ..extraction import JSONExtractor

                extractor = JSONExtractor(log_level)
                extracted_data = extractor.extract(response)
            else:
                console.print(
                    f"[red]Error: Unknown extraction method: {extraction_method}[/red]"
                )
<<<<<<< HEAD
                raise typer.Exit(1)
=======
                raise typer.Exit(1) from None
>>>>>>> 53cbe49 (✨ feat: Add comprehensive llama_vision package with CLI tools and document extraction capabilities)

            progress.update(extract_task, description="✅ Data extracted")

        # Display results
        console.print("\n[green]🎉 Extraction Complete![/green]")
        console.print(f"[dim]Image:[/dim] {Path(image_path).name}")
        console.print(f"[dim]Prompt:[/dim] {prompt_name}")
        console.print(f"[dim]Method:[/dim] {extraction_method}")
        console.print(f"[dim]Time:[/dim] {inference_time:.2f} seconds")
        console.print(f"[dim]Fields:[/dim] {len(extracted_data)}")

        # Show extracted data
        console.print("\n[yellow]📋 Extracted Data:[/yellow]")
        for key, value in extracted_data.items():
            if isinstance(value, list):
                console.print(
                    f"  [cyan]{key}:[/cyan] {', '.join(str(v) for v in value)}"
                )
            else:
                console.print(f"  [cyan]{key}:[/cyan] {value}")

        # Run InternVL comparison if requested
        if compare_internvl:
            console.print("\n[blue]🔄 Running InternVL Comparison...[/blue]")
            comparison = InternVLComparison(model, processor, prompt_manager, log_level)
            results = comparison.run_comparison(image_path)

            console.print("\n[yellow]📊 Comparison Results:[/yellow]")
            for result in results[:3]:  # Show top 3
                console.print(f"  {result.summary}")

        # Save output if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "image_path": str(Path(image_path).absolute()),
                "prompt_name": prompt_name,
                "extraction_method": extraction_method,
                "inference_time_seconds": inference_time,
                "extracted_data": extracted_data,
                "response_length": len(response),
                "timestamp": time.time(),
            }

            if compare_internvl and "results" in locals():
                output_data["internvl_comparison"] = [
                    {
                        "prompt_name": r.prompt_name,
                        "compatibility_score": r.metrics["internvl_compatibility"],
                        "performance_rating": r.metrics["performance_rating"],
                    }
                    for r in results
                ]

            with output_path.open("w") as f:
                json.dump(output_data, f, indent=2)

            console.print(f"\n[green]💾 Results saved to: {output_file}[/green]")

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        if verbose:
            import traceback

            console.print(f"[red]{traceback.format_exc()}[/red]")
<<<<<<< HEAD
        raise typer.Exit(1)
=======
        raise typer.Exit(1) from None
>>>>>>> 53cbe49 (✨ feat: Add comprehensive llama_vision package with CLI tools and document extraction capabilities)


@app.command()
def classify(
    image_path: str = typer.Argument(..., help="Path to image file"),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
):
    """Classify document type of an image."""

    log_level = "DEBUG" if verbose else "INFO"
    config = load_config()
<<<<<<< HEAD
=======
    
    # Configure logging level
    import logging
    logging.basicConfig(level=getattr(logging, log_level))
>>>>>>> 53cbe49 (✨ feat: Add comprehensive llama_vision package with CLI tools and document extraction capabilities)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Load model
            load_task = progress.add_task("Loading model...", total=None)
            loader = LlamaModelLoader(config)
            model, processor = loader.load_model()
            inference_engine = LlamaInferenceEngine(model, processor, config)
            progress.update(load_task, description="✅ Model loaded")

            # Classify document
            classify_task = progress.add_task("Classifying document...", total=None)
            result = inference_engine.classify_document(image_path)
            progress.update(classify_task, description="✅ Classification complete")

        # Display results
        console.print("\n[green]📋 Document Classification:[/green]")
        console.print(f"  [cyan]Type:[/cyan] {result['document_type']}")
        console.print(f"  [cyan]Confidence:[/cyan] {result['confidence']:.2f}")
        console.print(
            f"  [cyan]Business Document:[/cyan] {'Yes' if result['is_business_document'] else 'No'}"
        )

        if verbose:
            console.print(
                f"  [dim]Response:[/dim] {result['classification_response'][:100]}..."
            )

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        if verbose:
            import traceback

            console.print(f"[red]{traceback.format_exc()}[/red]")
<<<<<<< HEAD
        raise typer.Exit(1)
=======
        raise typer.Exit(1) from None
>>>>>>> 53cbe49 (✨ feat: Add comprehensive llama_vision package with CLI tools and document extraction capabilities)


@app.command()
def list_prompts():
    """List available prompts from prompts.yaml."""

    try:
        prompt_manager = PromptManager()
        prompts = prompt_manager.list_prompts()

        console.print(f"[green]📝 Available Prompts ({len(prompts)}):[/green]")

        # Show recommended prompts first
        recommended = prompt_manager.get_recommended_prompts()
        if recommended:
            console.print("\n[yellow]⭐ Recommended:[/yellow]")
            for prompt in recommended:
                if prompt in prompts:
                    console.print(f"  • {prompt}")

        # Show all prompts
        console.print("\n[blue]📋 All Prompts:[/blue]")
        for i, prompt in enumerate(sorted(prompts), 1):
            status = "⭐" if prompt in recommended else "  "
            console.print(f"{status} {i:2d}. {prompt}")

    except Exception as e:
        console.print(f"[red]❌ Error loading prompts: {e}[/red]")
<<<<<<< HEAD
        raise typer.Exit(1)
=======
        raise typer.Exit(1) from None
>>>>>>> 53cbe49 (✨ feat: Add comprehensive llama_vision package with CLI tools and document extraction capabilities)


@app.command()
def validate_config():
    """Validate configuration and environment setup."""

    try:
        console.print("[blue]🔧 Validating Configuration...[/blue]")

        # Load config
        config = load_config()
        console.print("✅ Configuration loaded from environment")

        # Check paths
        model_path = Path(config.model_path)
        if model_path.exists():
            console.print(f"✅ Model path exists: {model_path}")
        else:
            console.print(f"❌ Model path not found: {model_path}")

        image_path = Path(config.image_path)
        if image_path.exists():
            console.print(f"✅ Image path exists: {image_path}")
        else:
            console.print(f"❌ Image path not found: {image_path}")

        # Check prompts
        try:
            prompt_manager = PromptManager()
            prompts = prompt_manager.list_prompts()
            console.print(f"✅ Prompts loaded: {len(prompts)} prompts available")
        except Exception as e:
            console.print(f"❌ Prompts loading failed: {e}")

        # Check device
        from ..utils import detect_device

        device_info = detect_device()
        console.print(
            f"✅ Device detected: {device_info['type']} ({device_info['name']})"
        )

        if device_info["type"] == "cuda":
            console.print(f"   GPU Memory: {device_info['memory_gb']:.1f}GB")

        console.print("\n[green]🎉 Configuration validation complete![/green]")

    except Exception as e:
        console.print(f"[red]❌ Validation failed: {e}[/red]")
<<<<<<< HEAD
        raise typer.Exit(1)
=======
        raise typer.Exit(1) from None
>>>>>>> 53cbe49 (✨ feat: Add comprehensive llama_vision package with CLI tools and document extraction capabilities)


if __name__ == "__main__":
    app()
