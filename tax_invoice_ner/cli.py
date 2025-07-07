"""
Command-line interface for tax invoice NER system.

Provides a typer-based CLI for extracting entities from tax invoices
using configurable YAML entity definitions.
"""

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from tax_invoice_ner.config.config_manager import ConfigManager
from tax_invoice_ner.extractors.work_expense_ner_extractor import (
    WorkExpenseNERExtractor,
)
<<<<<<< HEAD
=======

# Module-level typer options to avoid B008 errors
ENTITY_TYPES_OPTION = typer.Option(
    None, "--entity", "-e", help="Specific entity types to extract (default: all)"
)
>>>>>>> 53cbe49 (✨ feat: Add comprehensive llama_vision package with CLI tools and document extraction capabilities)


@dataclass
class RichConfig:
    """Configuration for rich console output."""

    console: Console = Console()
    success_style: str = "[bold green]✅[/bold green]"
    fail_style: str = "[bold red]❌[/bold red]"
    warning_style: str = "[bold yellow]⚠️[/bold yellow]"
    info_style: str = "[bold blue]ℹ️[/bold blue]"


# Initialize CLI app and rich config
app = typer.Typer(
    name="tax-invoice-ner",
    help="Tax Invoice Named Entity Recognition System",
    add_completion=False,
)
rich_config = RichConfig()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=rich_config.console, rich_tracebacks=True)],
    )


@app.command()
def extract(
    image_path: str = typer.Argument(..., help="Path to tax invoice image"),
    config_path: str = typer.Option(
        "config/extractor/work_expense_ner_config.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
    entity_types: list[str] | None = ENTITY_TYPES_OPTION,
    output_path: str | None = typer.Option(
        None, "--output", "-o", help="Output JSON file path"
    ),
    model_path: str | None = typer.Option(
        None, "--model", "-m", help="Override model path from config"
    ),
    device: str | None = typer.Option(
        None, "--device", "-d", help="Device to use (cpu, cuda, mps)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
<<<<<<< HEAD
    output_path: str | None = typer.Option(
        None, "--output", "-o", help="Output JSON file path"
    ),
    model_path: str | None = typer.Option(
        None, "--model", "-m", help="Override model path from config"
    ),
    device: str | None = typer.Option(
        None, "--device", "-d", help="Device to use (cpu, cuda, mps)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
=======
>>>>>>> 53cbe49 (✨ feat: Add comprehensive llama_vision package with CLI tools and document extraction capabilities)
) -> None:
    """Extract entities from tax invoice image."""

    setup_logging(verbose)
    console = rich_config.console

    console.print(f"{rich_config.info_style} Starting tax invoice NER extraction...")

    try:
        # Validate input image
        image_file = Path(image_path)
        if not image_file.exists():
            console.print(
                f"{rich_config.fail_style} Image file not found: {image_path}"
            )
<<<<<<< HEAD
            raise typer.Exit(1)
=======
            raise typer.Exit(1) from None
>>>>>>> 53cbe49 (✨ feat: Add comprehensive llama_vision package with CLI tools and document extraction capabilities)

        # Initialize extractor
        console.print(f"{rich_config.info_style} Loading NER extractor...")
        extractor = WorkExpenseNERExtractor(
            config_path=config_path, model_path=model_path, device=device
        )

        console.print(
            f"{rich_config.success_style} Model loaded with {len(extractor.get_available_entities())} entity types"
        )

        # Extract entities
        console.print(
            f"{rich_config.info_style} Extracting entities from {image_path}..."
        )
        result = extractor.extract_entities(image_path, entity_types=entity_types)

        # Display results
        console.print(
            f"{rich_config.success_style} Extracted {len(result['entities'])} entities"
        )

        if result["entities"]:
            table = Table(title="Extracted Entities")
            table.add_column("Entity Type", style="cyan")
            table.add_column("Text", style="white")
            table.add_column("Confidence", style="green")

            for entity in result["entities"]:
                table.add_row(
                    entity["label"],
                    entity["text"],
                    f"{entity['confidence']:.2f}",
                )

            console.print(table)

        # Save results if output path specified
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Determine output format from file extension or config
            if output_path.endswith(".csv"):
                _save_csv_results(result, output_file)
            else:
                with output_file.open("w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

            console.print(f"{rich_config.success_style} Results saved to {output_path}")

        # Print summary
        console.print(f"\n{rich_config.info_style} Extraction Summary:")
        console.print(f"  • Document type: {result.get('document_type', 'unknown')}")
        console.print(f"  • Entities extracted: {len(result['entities'])}")
        console.print(f"  • Config entities: {result.get('config_entities', 0)}")
        console.print(f"  • Timestamp: {result.get('extraction_timestamp', 'unknown')}")

        if "error" in result:
            console.print(f"{rich_config.fail_style} Error occurred: {result['error']}")
            raise typer.Exit(1) from None

    except Exception as e:
        console.print(f"{rich_config.fail_style} Extraction failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from None


@app.command()
def list_entities(
    config_path: str = typer.Option(
        "config/extractor/work_expense_ner_config.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
) -> None:
    """List available entity types from configuration."""

    console = rich_config.console

    try:
        config_manager = ConfigManager(config_path)
        entities = config_manager.get_entities()

        console.print(
            f"{rich_config.info_style} Available Entity Types ({len(entities)} total):"
        )

        # Group entities by category
        categories = {
            "Business": ["BUSINESS_NAME", "VENDOR_NAME", "CLIENT_NAME"],
            "Financial": ["TOTAL_AMOUNT", "SUBTOTAL", "TAX_AMOUNT", "TAX_RATE"],
            "Dates": ["INVOICE_DATE", "DUE_DATE"],
            "Identification": ["INVOICE_NUMBER", "ABN", "GST_NUMBER", "PURCHASE_ORDER"],
            "Items": ["ITEM_DESCRIPTION", "ITEM_QUANTITY", "UNIT_PRICE", "LINE_TOTAL"],
            "Contact": ["CONTACT_PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"],
            "Address": ["BUSINESS_ADDRESS", "BILLING_ADDRESS"],
            "Payment": ["PAYMENT_METHOD", "PAYMENT_TERMS"],
        }

        for category, entity_types in categories.items():
            console.print(f"\n[bold cyan]{category}:[/bold cyan]")
            for entity_type in entity_types:
                if entity_type in entities:
                    description = entities[entity_type].get(
                        "description", "No description"
                    )
                    console.print(f"  • [green]{entity_type}[/green]: {description}")

        # Show any uncategorized entities
        categorized = set()
        for entity_list in categories.values():
            categorized.update(entity_list)

        uncategorized = set(entities.keys()) - categorized
        if uncategorized:
            console.print("\n[bold cyan]Other:[/bold cyan]")
            for entity_type in sorted(uncategorized):
                description = entities[entity_type].get("description", "No description")
                console.print(f"  • [green]{entity_type}[/green]: {description}")

    except Exception as e:
        console.print(f"{rich_config.fail_style} Failed to load configuration: {e}")
        raise typer.Exit(1) from None


@app.command()
def validate_config(
    config_path: str = typer.Option(
        "config/extractor/work_expense_ner_config.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
) -> None:
    """Validate YAML configuration file."""

    console = rich_config.console

    try:
        config_manager = ConfigManager(config_path)

        console.print(f"{rich_config.success_style} Configuration file is valid")
        console.print(f"{rich_config.info_style} Configuration summary:")

        model_config = config_manager.get_model_config()
        console.print(f"  • Model path: {model_config['model_path']}")
        console.print(f"  • Device: {model_config['device']}")
        console.print(f"  • Max tokens: {model_config['max_new_tokens']}")
        console.print(f"  • Entity types: {len(config_manager.get_entity_types())}")
        console.print(
            f"  • Confidence threshold: {config_manager.get_confidence_threshold()}"
        )

        # Check if model path exists
        model_path = Path(model_config["model_path"])
        if model_path.exists():
            console.print(f"{rich_config.success_style} Model path exists")
        else:
            console.print(
                f"{rich_config.warning_style} Model path does not exist: {model_path}"
            )

    except Exception as e:
        console.print(f"{rich_config.fail_style} Configuration validation failed: {e}")
        raise typer.Exit(1) from None


@app.command()
def demo(
    config_path: str = typer.Option(
        "config/extractor/work_expense_ner_config.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
    image_path: str = typer.Option(
        "test_receipt.png", "--image", "-i", help="Path to test image"
    ),
) -> None:
    """Run a demonstration of the NER system."""

    setup_logging(False)
    console = rich_config.console

    console.print("[bold blue]🚀 TAX INVOICE NER DEMONSTRATION[/bold blue]")
    console.print("=" * 60)

    # Test different entity groups
    entity_groups = {
        "Business entities": ["BUSINESS_NAME", "VENDOR_NAME", "ABN"],
        "Financial entities": ["TOTAL_AMOUNT", "SUBTOTAL", "TAX_AMOUNT"],
        "Date entities": ["INVOICE_DATE", "DUE_DATE"],
    }

    try:
        extractor = WorkExpenseNERExtractor(config_path=config_path)

        for group_name, entities in entity_groups.items():
            console.print(f"\n[bold cyan]Testing {group_name}:[/bold cyan]")

            result = extractor.extract_entities(image_path, entity_types=entities)

            if result["entities"]:
                for entity in result["entities"]:
                    confidence_emoji = (
                        "🟢"
                        if entity["confidence"] > 0.8
                        else "🟡"
                        if entity["confidence"] > 0.6
                        else "🔴"
                    )
                    console.print(
                        f"  {confidence_emoji} {entity['label']}: '{entity['text']}' ({entity['confidence']:.2f})"
                    )
            else:
                console.print(f"  {rich_config.warning_style} No entities extracted")

        console.print(f"\n{rich_config.success_style} Demonstration completed!")

    except Exception as e:
        console.print(f"{rich_config.fail_style} Demo failed: {e}")
        raise typer.Exit(1) from None


def _save_csv_results(result: dict, output_file: Path) -> None:
    """Save extraction results to CSV format.

    Args:
        result: Extraction result dictionary
        output_file: Path to save CSV file
    """
    entities = result.get("entities", [])

    if not entities:
        # Create empty CSV with headers
        with output_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "entity_type",
                    "text",
                    "confidence",
                    "start_pos",
                    "end_pos",
                    "source_snippet",
                ]
            )
        return

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(
            [
                "entity_type",
                "text",
                "confidence",
                "start_pos",
                "end_pos",
                "source_snippet",
            ]
        )

        # Write entity data
        for entity in entities:
            writer.writerow(
                [
                    entity.get("label", ""),
                    entity.get("text", ""),
                    entity.get("confidence", ""),
                    entity.get("start_pos", ""),
                    entity.get("end_pos", ""),
                    entity.get("source_snippet", "")
                    .replace("\n", " ")
                    .replace("\r", " ")
                    if entity.get("source_snippet")
                    else "",
                ]
            )


if __name__ == "__main__":
    app()
