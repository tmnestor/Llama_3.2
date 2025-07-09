"""Single image processing CLI for Llama-3.2-Vision package."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import PromptManager, load_config
from ..evaluation import InternVLComparison
from ..model import LlamaInferenceEngine, LlamaModelLoader

app = typer.Typer(
    help="Single image processing with Llama-3.2-Vision", rich_markup_mode="rich"
)
console = Console()


def process_single_image_core(
    image_path: str,
    inference_engine: LlamaInferenceEngine,
    prompt_manager: PromptManager,
    prompt: Optional[str] = None,
    classify_only: bool = False,
    _verbose: bool = False,
) -> Dict[str, Any]:
    """Core single image processing logic that can be reused by batch processing.

    Args:
        image_path: Path to the image file
        inference_engine: Initialized inference engine
        prompt_manager: Initialized prompt manager
        prompt: Manual prompt (if None, uses smart classification)
        classify_only: If True, only classify document type
        verbose: Enable verbose logging

    Returns:
        Dictionary containing processing results
    """
    # Determine operation mode
    use_smart_classification = prompt is None
    use_manual_prompt = prompt is not None

    # Validate image
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # STEP 1: Classification (if needed)
    document_type = None
    classification_result = None
    confidence = 0.0
    selected_prompt_name = prompt or "manual"

    if use_smart_classification or classify_only:
        classification_result = inference_engine.classify_document(image_path)
        document_type = classification_result["document_type"]
        confidence = classification_result["confidence"]

        if classify_only:
            return {
                "success": True,
                "classify_only": True,
                "document_type": document_type,
                "confidence": confidence,
                "is_business_document": classification_result.get(
                    "is_business_document", False
                ),
                "classification_result": classification_result,
            }

    # STEP 2: Prompt Selection
    if use_smart_classification:
        try:
            selected_prompt = prompt_manager.get_prompt_for_document_type(
                document_type,
                classification_result.get("classification_response", ""),
            )
            # Get the actual prompt name from document type mapping
            type_mapping = prompt_manager.metadata.get("document_type_mapping", {})
            selected_prompt_name = type_mapping.get(
                document_type, f"{document_type}_prompt"
            )
        except KeyError as e:
            raise KeyError(
                f"No prompt configured for document type '{document_type}'"
            ) from e
    elif use_manual_prompt:
        try:
            selected_prompt = prompt_manager.get_prompt(prompt)
            selected_prompt_name = prompt
        except KeyError as e:
            raise KeyError(f"Prompt '{prompt}' not found") from e

    # STEP 3: Run inference
    start_time = time.time()
    response = inference_engine.predict(image_path, selected_prompt)
    inference_time = time.time() - start_time

    # STEP 4: Parse extracted data using modern registry
    from ..extraction.extraction_engine import DocumentExtractionEngine

    engine = DocumentExtractionEngine()

    # For manual prompts, use "receipt" as default document type for extraction
    extraction_document_type = document_type or "receipt"
    extraction_result = engine.extract_fields(extraction_document_type, response)
    extracted_data = extraction_result.fields if extraction_result else {}

    # Return comprehensive result
    return {
        "success": True,
        "classify_only": False,
        "image_path": image_path,
        "image_name": Path(image_path).name,
        "document_type": extraction_document_type,  # Use the document type used for extraction
        "confidence": confidence,
        "is_business_document": classification_result.get("is_business_document", False)
        if classification_result
        else False,
        "classification_result": classification_result,
        "selected_prompt": selected_prompt_name,
        "extraction_method": "modern_registry",
        "inference_time_seconds": inference_time,
        "response_length": len(response),
        "field_count": len(extracted_data),
        "extracted_data": extracted_data,
        "raw_response": response,
    }


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

            # Process image using core function
            process_task = progress.add_task("Processing image...", total=None)
            result = process_single_image_core(
                image_path=image_path,
                inference_engine=inference_engine,
                prompt_manager=prompt_manager,
                prompt=prompt,
                classify_only=classify_only,
                verbose=verbose,
            )
            progress.update(process_task, description="✅ Processing complete")

        # Handle classify_only mode
        if result.get("classify_only", False):
            console.print("\n[green]📋 Document Classification:[/green]")
            console.print(f"  Type: {result['document_type']}")
            console.print(f"  Confidence: {result['confidence']:.2f}")
            console.print(f"  Business Document: {result['is_business_document']}")
            return

        # Extract variables from result
        document_type = result["document_type"]
        confidence = result["confidence"]
        selected_prompt_name = result["selected_prompt"]
        inference_time = result["inference_time_seconds"]
        extracted_data = result["extracted_data"]
        classification_result = result["classification_result"]

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
                "response_length": result["response_length"],
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
