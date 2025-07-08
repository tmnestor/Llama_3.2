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
from ..image import ImageLoader
from ..model import LlamaInferenceEngine, LlamaModelLoader

app = typer.Typer(
    help="Single image processing with Llama-3.2-Vision", rich_markup_mode="rich"
)
console = Console()


@app.command()
def extract(
    image_path: str = typer.Argument(..., help="Path to image file"),
    prompt: Optional[str] = typer.Option(
        None, help="Manual prompt selection (disables smart classification)"
    ),
    output_file: Optional[str] = typer.Option(None, help="Output file path (JSON)"),
    classify_only: bool = typer.Option(
        False, help="Only classify document type, don't extract fields"
    ),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
    compare_internvl: bool = typer.Option(False, help="Run InternVL comparison test"),
):
    """Extract information from images with smart document classification (default) or manual prompt selection."""

    # Setup
    log_level = "DEBUG" if verbose else "INFO"
    config = load_config()

    # Determine operation mode
    use_smart_classification = prompt is None
    use_manual_prompt = prompt is not None

    mode_description = (
        "Smart classification" if use_smart_classification else "Manual prompt"
    )
    console.print(f"[blue]Extraction mode: {mode_description}[/blue]")

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
                raise typer.Exit(1)
            progress.update(validate_task, description="✅ Image validated")

            # STEP 1: Classification (if needed)
            document_type = None
            classification_result = None
            confidence = 0.0
            selected_prompt_name = prompt or "manual"

            if use_smart_classification or classify_only:
                classify_task = progress.add_task(
                    "Classifying document type...", total=None
                )
                classification_result = inference_engine.classify_document(image_path)
                document_type = classification_result["document_type"]
                confidence = classification_result["confidence"]
                progress.update(
                    classify_task,
                    description=f"✅ Classified as: {document_type} ({confidence:.2f})",
                )

                if classify_only:
                    # Just show classification and exit
                    console.print("\n[green]📋 Document Classification:[/green]")
                    console.print(f"  Type: {document_type}")
                    console.print(f"  Confidence: {confidence:.2f}")
                    console.print(
                        f"  Business Document: {classification_result.get('is_business_document', False)}"
                    )
                    return

            # STEP 2: Prompt Selection
            if use_smart_classification:
                prompt_task = progress.add_task(
                    "Selecting optimal prompt...", total=None
                )
                try:
                    selected_prompt = prompt_manager.get_prompt_for_document_type(
                        document_type,
                        classification_result.get("classification_response", ""),
                    )
                    # Get the actual prompt name from document type mapping
                    type_mapping = prompt_manager.metadata.get(
                        "document_type_mapping", {}
                    )
                    selected_prompt_name = type_mapping.get(
                        document_type, f"{document_type}_prompt"
                    )
                except KeyError:
                    console.print(
                        f"[red]Error: No prompt configured for document type '{document_type}'[/red]"
                    )
                    available_types = list(
                        prompt_manager.metadata.get("document_type_mapping", {}).keys()
                    )
                    console.print(
                        f"Supported document types: {', '.join(available_types)}"
                    )
                    raise typer.Exit(1) from None
                progress.update(
                    prompt_task, description=f"✅ Selected prompt for {document_type}"
                )
            elif use_manual_prompt:
                prompt_task = progress.add_task("Loading manual prompt...", total=None)
                try:
                    selected_prompt = prompt_manager.get_prompt(prompt)
                    selected_prompt_name = prompt
                except KeyError:
                    available_prompts = prompt_manager.list_prompts()
                    console.print(f"[red]Error: Prompt '{prompt}' not found.[/red]")
                    console.print(
                        f"Available prompts: {', '.join(available_prompts[:5])}..."
                    )
                    raise typer.Exit(1) from None
                progress.update(prompt_task, description="✅ Manual prompt loaded")

            # STEP 3: Run inference
            inference_task = progress.add_task(
                "Running optimized extraction...", total=None
            )
            start_time = time.time()
            response = inference_engine.predict(image_path, selected_prompt)
            inference_time = time.time() - start_time
            progress.update(inference_task, description="✅ Extraction complete")

            # STEP 4: Parse extracted data using modern registry
            parse_task = progress.add_task("Parsing extracted data...", total=None)
            from ..extraction.modern_adapter import get_modern_adapter

            adapter = get_modern_adapter()
            extracted_data = adapter.extract_fields_modern(document_type, response)
            progress.update(parse_task, description="✅ Data parsed")

        # Display results
        console.print("\n[green]🎉 Extraction Complete![/green]")
        console.print(f"[dim]Image:[/dim] {Path(image_path).name}")
        if use_smart_classification:
            console.print(
                f"[dim]Document Type:[/dim] {document_type} (confidence: {confidence:.2f})"
            )
        console.print(f"[dim]Selected Prompt:[/dim] {selected_prompt_name}")
        console.print("[dim]Extraction Method:[/dim] modern_registry")
        console.print(f"[dim]Time:[/dim] {inference_time:.2f} seconds")
        console.print(f"[dim]Fields Extracted:[/dim] {len(extracted_data)}")

        # Show compliance score if available
        if "_compliance_score" in extracted_data:
            score = extracted_data["_compliance_score"]
            console.print(f"[dim]Compliance Score:[/dim] {score:.2f}")

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
                "document_classification": classification_result,
                "selected_prompt": selected_prompt_name,
                "extraction_method": "modern_registry",
                "inference_time_seconds": inference_time,
                "extracted_data": extracted_data,
                "response_length": len(response),
                "extraction_mode": "smart_classification"
                if use_smart_classification
                else "manual_prompt",
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
        raise typer.Exit(1) from None


@app.command()
def classify(
    image_path: str = typer.Argument(..., help="Path to image file"),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
):
    """Classify document type of an image."""

    log_level = "DEBUG" if verbose else "INFO"
    config = load_config()

    # Configure logging level
    import logging

    logging.basicConfig(level=getattr(logging, log_level))

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
        raise typer.Exit(1) from None


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

        # Show document type mappings
        console.print("\n[blue]📄 Document Type Mappings:[/blue]")
        try:
            type_mappings = prompt_manager.metadata.get("document_type_mapping", {})
            for doc_type, prompt_name in type_mappings.items():
                console.print(f"  {doc_type}: {prompt_name}")
        except Exception:
            console.print("  (No document type mappings found)")

        # Show all prompts
        console.print("\n[blue]📋 All Prompts:[/blue]")
        for i, prompt in enumerate(sorted(prompts), 1):
            status = "⭐" if prompt in recommended else "  "
            console.print(f"{status} {i:2d}. {prompt}")

    except Exception as e:
        console.print(f"[red]❌ Error loading prompts: {e}[/red]")
        raise typer.Exit(1) from None


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

        # Check prompts and document type mappings
        try:
            prompt_manager = PromptManager()
            prompts = prompt_manager.list_prompts()
            console.print(f"✅ Prompts loaded: {len(prompts)} prompts available")

            # Validate document type mappings
            type_mappings = prompt_manager.metadata.get("document_type_mapping", {})
            console.print(
                f"✅ Document type mappings: {len(type_mappings)} types configured"
            )

            # Check if all mapped prompts exist
            missing_prompts = []
            for doc_type, prompt_name in type_mappings.items():
                if prompt_name not in prompts:
                    missing_prompts.append(f"{doc_type} -> {prompt_name}")

            if missing_prompts:
                console.print(
                    f"⚠️ Warning: Missing prompts for document types: {missing_prompts}"
                )
            else:
                console.print("✅ All document type mappings have valid prompts")

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
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
