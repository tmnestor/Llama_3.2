"""Batch processing CLI for Llama-3.2-Vision package."""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from ..config import PromptManager, load_config
from ..extraction import KeyValueExtractor, TaxAuthorityParser
from ..image import ImageLoader
from ..model import LlamaInferenceEngine, LlamaModelLoader

app = typer.Typer(help="Batch processing with Llama-3.2-Vision")
console = Console()


@app.command()
def extract(
    image_folder: str = typer.Argument(..., help="Folder containing images"),
    output_file: str = typer.Option("batch_results.csv", help="Output CSV file"),
    prompt_name: str = typer.Option("key_value_receipt_prompt", help="Prompt to use"),
    extraction_method: str = typer.Option("tax_authority", help="Extraction method"),
    max_workers: int = typer.Option(
        1, help="Number of parallel workers (1 = sequential)"
    ),
    max_images: Optional[int] = typer.Option(
        None, help="Maximum number of images to process"
    ),
    file_pattern: str = typer.Option("*", help="File pattern to match (e.g., '*.jpg')"),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
):
    """Process multiple images in batch."""

    log_level = "DEBUG" if verbose else "INFO"
    config = load_config()

    try:
        # Setup
        console.print(f"[blue]🔍 Discovering images in: {image_folder}[/blue]")

        image_loader = ImageLoader(log_level)
        discovered_images = image_loader.discover_images(image_folder)

        # Collect all images
        all_images = []
        for _category, images in discovered_images.items():
            all_images.extend(images)

        # Filter by pattern if specified
        if file_pattern != "*":
            all_images = [img for img in all_images if img.match(file_pattern)]

        # Limit number of images if specified
        if max_images:
            all_images = all_images[:max_images]

        if not all_images:
            console.print(f"[red]❌ No images found in {image_folder}[/red]")
            raise typer.Exit(1)

        console.print(f"[green]📊 Found {len(all_images)} images to process[/green]")

        # Load model and components
        console.print("[blue]🚀 Loading Llama-3.2-Vision model...[/blue]")
        loader = LlamaModelLoader(config)
        model, processor = loader.load_model()

        inference_engine = LlamaInferenceEngine(model, processor, config)
        prompt_manager = PromptManager()

        # Get prompt
        try:
            prompt = prompt_manager.get_prompt(prompt_name)
        except KeyError:
            available_prompts = prompt_manager.list_prompts()
            console.print(f"[red]Error: Prompt '{prompt_name}' not found.[/red]")
            console.print(f"Available prompts: {', '.join(available_prompts[:5])}...")
            raise typer.Exit(1) from None

        # Setup extractor
        if extraction_method == "key_value":
            extractor = KeyValueExtractor(log_level)
        elif extraction_method == "tax_authority":
            extractor = TaxAuthorityParser(log_level)
        elif extraction_method == "json":
            from ..extraction import JSONExtractor

            extractor = JSONExtractor(log_level)
        else:
            console.print(
                f"[red]Error: Unknown extraction method: {extraction_method}[/red]"
            )
            raise typer.Exit(1) from None

        # Process images
        console.print(
            f"[blue]⚡ Processing {len(all_images)} images with {max_workers} worker(s)...[/blue]"
        )

        results = []
        start_time = time.time()

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing images...", total=len(all_images))

            if max_workers == 1:
                # Sequential processing
                for i, image_path in enumerate(all_images):
                    result = process_single_image(
                        str(image_path),
                        inference_engine,
                        extractor,
                        prompt,
                        extraction_method,
                        i + 1,
                        len(all_images),
                    )
                    results.append(result)
                    progress.update(task, advance=1)
            else:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = {
                        executor.submit(
                            process_single_image,
                            str(image_path),
                            inference_engine,
                            extractor,
                            prompt,
                            extraction_method,
                            i + 1,
                            len(all_images),
                        ): i
                        for i, image_path in enumerate(all_images)
                    }

                    # Collect results in order
                    results = [None] * len(all_images)
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            result = future.result()
                            results[index] = result
                        except Exception as e:
                            console.print(
                                f"[red]Error processing image {index + 1}: {e}[/red]"
                            )
                            results[index] = {
                                "image_path": str(all_images[index]),
                                "error": str(e),
                                "success": False,
                            }
                        progress.update(task, advance=1)

        # Display classification results after processing
        console.print("\n[yellow]📋 Classification Results:[/yellow]")
        for i, result in enumerate(results):
            if result and result.get("success", False):
                doc_type = result.get("document_type", "unknown")
                confidence = result.get("classification_confidence", 0.0)
                image_name = result.get("image_name", "unknown")
                console.print(
                    f"  {i + 1:3d}. {image_name} → {doc_type} (confidence: {confidence:.2f})"
                )

        total_time = time.time() - start_time
        successful_results = [r for r in results if r and r.get("success", False)]

        console.print("\n[green]🎉 Batch processing complete![/green]")
        console.print(f"[dim]Total time:[/dim] {total_time:.1f} seconds")
        console.print(
            f"[dim]Successful:[/dim] {len(successful_results)}/{len(results)}"
        )
        console.print(
            f"[dim]Average time per image:[/dim] {total_time / len(results):.2f} seconds"
        )
        throughput_per_minute = (
            60 / (total_time / len(results)) if total_time > 0 else 0
        )
        console.print(
            f"[dim]Throughput:[/dim] {throughput_per_minute:.1f} images/minute"
        )

        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_file.endswith(".csv"):
            save_results_csv(results, output_path)
        else:
            save_results_json(results, output_path)

        console.print(f"[green]💾 Results saved to: {output_file}[/green]")

        # Show summary statistics
        if successful_results:
            show_batch_summary(successful_results)

    except Exception as e:
        console.print(f"[red]❌ Batch processing failed: {e}[/red]")
        if verbose:
            import traceback

            console.print(f"[red]{traceback.format_exc()}[/red]")
        raise typer.Exit(1) from None


def process_single_image(
    image_path: str,
    inference_engine: LlamaInferenceEngine,
    extractor,
    prompt: str,
    extraction_method: str,
    current: int,
    total: int,
) -> dict:
    """Process a single image and return results."""

    try:
        start_time = time.time()
        print(f"DEBUG: Starting processing for {Path(image_path).name}")

        # Run inference for extraction
        response = inference_engine.predict(image_path, prompt)
        print(
            f"DEBUG: Inference completed for {Path(image_path).name}, response length: {len(response)}"
        )

        # Extract data and get classification from extraction engine
        print(f"DEBUG: About to extract using method: {extraction_method}")
        try:
            if extraction_method == "tax_authority":
                print(
                    f"DEBUG: Calling extractor.parse_receipt_response() for {Path(image_path).name}"
                )
                extracted_data = extractor.parse_receipt_response(response)
            else:
                print(f"DEBUG: Calling extractor.extract() for {Path(image_path).name}")
                extracted_data = extractor.extract(response)
        except Exception as e:
            print(f"DEBUG: Extraction failed for {Path(image_path).name}: {e}")
            import traceback

            traceback.print_exc()
            # Use empty extraction data as fallback
            extracted_data = {}
        print(
            f"DEBUG: Extraction completed for {Path(image_path).name}, extracted {len(extracted_data)} fields"
        )

        # Get classification from the extraction engine to avoid duplicate inference
        print(f"DEBUG: Starting classification for {Path(image_path).name}")
        try:
            from ..extraction.extraction_engine import DocumentExtractionEngine

            engine = DocumentExtractionEngine()
            print("DEBUG: DocumentExtractionEngine created, calling classify_document")
            classification_result = engine.classify_document(response)
            print(
                f"DEBUG: Classification completed: {classification_result.document_type} (confidence: {classification_result.confidence})"
            )

            # Convert to dict format
            classification_dict = {
                "document_type": classification_result.document_type,
                "confidence": classification_result.confidence,
                "is_business_document": classification_result.is_business_document,
            }
        except Exception as e:
            # Fallback classification if extraction engine fails
            print(f"DEBUG: Classification failed with error: {e}")
            import traceback

            traceback.print_exc()
            classification_dict = {
                "document_type": "unknown",
                "confidence": 0.0,
                "is_business_document": False,
            }

        inference_time = time.time() - start_time

        # Prepare result
        result = {
            "image_path": image_path,
            "image_name": Path(image_path).name,
            "inference_time_seconds": inference_time,
            "response_length": len(response),
            "processed_order": current,
            "total_images": total,
            "field_count": len(extracted_data),
            "success": True,
            "extraction_method": extraction_method,
            "timestamp": time.time(),
            # Add classification information
            "document_type": classification_dict["document_type"],
            "classification_confidence": classification_dict["confidence"],
            "is_business_document": classification_dict["is_business_document"],
        }

        # Add extracted fields
        result.update(extracted_data)

        print(
            f"DEBUG: Result prepared for {Path(image_path).name}, returning success=True"
        )
        return result

    except Exception as e:
        return {
            "image_path": image_path,
            "image_name": Path(image_path).name,
            "error": str(e),
            "success": False,
            "timestamp": time.time(),
        }


def save_results_csv(results: List[dict], output_path: Path) -> None:
    """Save results to CSV file."""

    if not results:
        return

    # Get all possible field names
    all_fields = set()
    for result in results:
        all_fields.update(result.keys())

    # Standard fields first, then alphabetical
    standard_fields = [
        "image_name",
        "image_path",
        "success",
        "document_type",
        "classification_confidence",
        "is_business_document",
        "inference_time_seconds",
        "field_count",
        "extraction_method",
        "error",
    ]

    extracted_fields = sorted([f for f in all_fields if f not in standard_fields])
    fieldnames = [f for f in standard_fields if f in all_fields] + extracted_fields

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            # Convert lists to strings for CSV
            csv_result = {}
            for key, value in result.items():
                if isinstance(value, list):
                    csv_result[key] = ", ".join(str(v) for v in value)
                else:
                    csv_result[key] = value
            writer.writerow(csv_result)


def save_results_json(results: List[dict], output_path: Path) -> None:
    """Save results to JSON file."""

    output_data = {
        "batch_summary": {
            "total_images": len(results),
            "successful": len([r for r in results if r.get("success", False)]),
            "timestamp": time.time(),
        },
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


def show_batch_summary(successful_results: List[dict]) -> None:
    """Show summary statistics for batch processing."""

    console.print("\n[yellow]📊 Batch Summary:[/yellow]")

    # Timing statistics
    times = [r["inference_time_seconds"] for r in successful_results]
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    console.print(
        f"  [cyan]Timing:[/cyan] avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s"
    )

    # Field count statistics
    field_counts = [r["field_count"] for r in successful_results]
    avg_fields = sum(field_counts) / len(field_counts)
    min_fields = min(field_counts)
    max_fields = max(field_counts)

    console.print(
        f"  [cyan]Fields:[/cyan] avg={avg_fields:.1f}, min={min_fields}, max={max_fields}"
    )

    # Classification statistics
    doc_types = {}
    confidences = []
    business_docs = 0

    for result in successful_results:
        doc_type = result.get("document_type", "unknown")
        confidence = result.get("classification_confidence", 0.0)
        is_business = result.get("is_business_document", False)

        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        confidences.append(confidence)
        if is_business:
            business_docs += 1

    if doc_types:
        console.print("  [cyan]Document types:[/cyan]")
        sorted_types = sorted(doc_types.items(), key=lambda x: x[1], reverse=True)
        for doc_type, count in sorted_types:
            percentage = (count / len(successful_results)) * 100
            console.print(f"    • {doc_type}: {count} ({percentage:.1f}%)")

    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        console.print(
            f"  [cyan]Average classification confidence:[/cyan] {avg_confidence:.2f}"
        )
        console.print(
            f"  [cyan]Business documents:[/cyan] {business_docs}/{len(successful_results)} ({(business_docs / len(successful_results)) * 100:.1f}%)"
        )

    # Common extracted fields
    field_frequency = {}
    for result in successful_results:
        for field in result.keys():
            if field not in [
                "image_path",
                "image_name",
                "success",
                "inference_time_seconds",
                "field_count",
                "extraction_method",
                "timestamp",
                "response_length",
                "document_type",
                "classification_confidence",
                "is_business_document",
            ]:
                field_frequency[field] = field_frequency.get(field, 0) + 1

    if field_frequency:
        console.print("  [cyan]Common fields:[/cyan]")
        sorted_fields = sorted(
            field_frequency.items(), key=lambda x: x[1], reverse=True
        )
        for field, count in sorted_fields[:5]:
            percentage = (count / len(successful_results)) * 100
            console.print(
                f"    • {field}: {count}/{len(successful_results)} ({percentage:.1f}%)"
            )


@app.command()
def analyze(
    input_file: str = typer.Argument(..., help="CSV/JSON file with batch results"),
    field_analysis: bool = typer.Option(True, help="Show field analysis"),
    performance_analysis: bool = typer.Option(True, help="Show performance analysis"),
):
    """Analyze batch processing results."""

    try:
        input_path = Path(input_file)

        if not input_path.exists():
            console.print(f"[red]❌ Input file not found: {input_file}[/red]")
            raise typer.Exit(1) from None

        # Load results
        if input_file.endswith(".csv"):
            results = load_results_csv(input_path)
        else:
            results = load_results_json(input_path)

        console.print(
            f"[green]📊 Analyzing {len(results)} results from {input_file}[/green]"
        )

        # Filter successful results
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]

        console.print(
            f"[blue]Success rate:[/blue] {len(successful)}/{len(results)} ({len(successful) / len(results) * 100:.1f}%)"
        )

        if failed:
            console.print("[red]Failed images:[/red]")
            for result in failed[:5]:
                console.print(
                    f"  • {result.get('image_name', 'Unknown')}: {result.get('error', 'Unknown error')}"
                )

        if successful and performance_analysis:
            show_performance_analysis(successful)

        if successful and field_analysis:
            show_field_analysis(successful)

    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")
        raise typer.Exit(1) from None


def load_results_csv(input_path: Path) -> List[dict]:
    """Load results from CSV file."""
    results = []
    with input_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert string fields back to appropriate types
            if "inference_time_seconds" in row:
                try:
                    row["inference_time_seconds"] = float(row["inference_time_seconds"])
                except (ValueError, TypeError):
                    pass

            if "field_count" in row:
                try:
                    row["field_count"] = int(row["field_count"])
                except (ValueError, TypeError):
                    pass

            if "success" in row:
                row["success"] = row["success"].lower() in ["true", "1", "yes"]

            results.append(row)

    return results


def load_results_json(input_path: Path) -> List[dict]:
    """Load results from JSON file."""
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        return data["results"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(
            "Invalid JSON format: expected list or dict with 'results' key"
        )


def show_performance_analysis(results: List[dict]) -> None:
    """Show performance analysis of batch results."""
    console.print("\n[yellow]⚡ Performance Analysis:[/yellow]")

    # Timing analysis
    times = [
        r.get("inference_time_seconds", 0)
        for r in results
        if "inference_time_seconds" in r
    ]
    if times:
        avg_time = sum(times) / len(times)
        console.print(f"  [cyan]Average inference time:[/cyan] {avg_time:.2f} seconds")
        console.print(
            f"  [cyan]Estimated throughput:[/cyan] {3600 / avg_time:.0f} images/hour"
        )

    # Field extraction rates
    field_counts = [r.get("field_count", 0) for r in results if "field_count" in r]
    if field_counts:
        avg_fields = sum(field_counts) / len(field_counts)
        console.print(f"  [cyan]Average fields extracted:[/cyan] {avg_fields:.1f}")


def show_field_analysis(results: List[dict]) -> None:
    """Show field extraction analysis."""
    console.print("\n[yellow]📋 Field Extraction Analysis:[/yellow]")

    # Count field frequency
    field_stats = {}
    total_docs = len(results)

    for result in results:
        for field, value in result.items():
            if field in [
                "image_path",
                "image_name",
                "success",
                "inference_time_seconds",
                "field_count",
                "extraction_method",
                "timestamp",
                "response_length",
                "error",
                "document_type",
                "classification_confidence",
                "is_business_document",
            ]:
                continue

            if field not in field_stats:
                field_stats[field] = {"count": 0, "non_empty": 0}

            field_stats[field]["count"] += 1
            if value and str(value).strip() and str(value) != "[]":
                field_stats[field]["non_empty"] += 1

    # Show top fields by extraction rate
    sorted_fields = sorted(
        field_stats.items(), key=lambda x: x[1]["non_empty"], reverse=True
    )

    console.print("  [cyan]Top extracted fields:[/cyan]")
    for field, stats in sorted_fields[:10]:
        rate = (stats["non_empty"] / total_docs) * 100
        console.print(f"    • {field}: {stats['non_empty']}/{total_docs} ({rate:.1f}%)")


if __name__ == "__main__":
    app()
