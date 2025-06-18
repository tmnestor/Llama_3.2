#!/usr/bin/env python3
"""
Enhance metadata for receipt extraction evaluation.

This script takes the basic metadata from receipt generation and enhances it
with detailed extraction fields for zero-shot evaluation purposes.
"""

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

# Add project root to path - must be done before local imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Local imports after path modification
from utils.cli import (  # noqa: E402
    RichConfig,
    ensure_output_dir,
    print_error,
    print_info,
    print_success,
    validate_input_path,
)

app = typer.Typer(help="Enhance metadata for receipt extraction evaluation")
rich_config = RichConfig()

# Australian store data for realistic metadata
STORE_NAMES = [
    "WOOLWORTHS", "COLES", "ALDI", "IGA", "BUNNINGS", "KMART", "TARGET", 
    "OFFICEWORKS", "BIG W", "DAN MURPHY'S", "BWS", "CHEMIST WAREHOUSE",
    "JB HI-FI", "HARVEY NORMAN", "REBEL", "SUPERCHEAP AUTO", "LIQUORLAND",
    "PRICELINE", "DAVID JONES", "MYER", "SPOTLIGHT", "THE GOOD GUYS"
]

PAYMENT_METHODS = ["VISA", "MASTERCARD", "EFTPOS", "CASH", "AMEX", "PAYWAVE", "PAYPASS"]

# Generate realistic items with prices
ITEMS_DATABASE = [
    {"name": "Milk 2L", "price_range": (3.50, 5.50)},
    {"name": "Bread", "price_range": (2.80, 4.50)},
    {"name": "Free Range Eggs", "price_range": (4.00, 7.00)},
    {"name": "Bananas", "price_range": (2.50, 4.00), "unit": "kg"},
    {"name": "Apples", "price_range": (3.50, 5.50), "unit": "kg"},
    {"name": "Tim Tams", "price_range": (3.50, 5.00)},
    {"name": "Coffee", "price_range": (8.00, 15.00)},
    {"name": "Pasta", "price_range": (2.00, 4.00)},
    {"name": "Toilet Paper", "price_range": (7.00, 12.00)},
    {"name": "Chicken Breast", "price_range": (9.00, 15.00), "unit": "kg"},
]


def generate_receipt_metadata(filename: str, receipt_count: int, is_stapled: bool) -> dict:
    """Generate realistic receipt extraction metadata."""
    
    if receipt_count == 0:
        # Tax document or empty image
        return {
            "filename": filename,
            "image_path": f"images/{filename}",
            "receipt_count": receipt_count,
            "is_stapled": is_stapled,
            "store_name": None,
            "date": None,
            "time": None,
            "total_amount": None,
            "payment_method": None,
            "receipt_id": None,
            "items": [],
            "tax_info": None,
            "discounts": None
        }
    
    # Generate realistic receipt data
    store_name = random.choice(STORE_NAMES)
    
    # Generate date within last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    random_date = start_date + timedelta(days=random.randint(0, 730))
    date = random_date.strftime("%Y-%m-%d")
    time = f"{random.randint(8, 21):02d}:{random.randint(0, 59):02d}"
    
    # Generate items
    num_items = random.randint(2, 8)
    items = []
    total = 0.0
    
    for _ in range(num_items):
        item_data = random.choice(ITEMS_DATABASE)
        quantity = random.randint(1, 3)
        price = round(random.uniform(*item_data["price_range"]), 2)
        
        item = {
            "item_name": item_data["name"],
            "quantity": quantity,
            "price": f"${price:.2f}"
        }
        
        if "unit" in item_data:
            item["unit"] = item_data["unit"]
        
        items.append(item)
        total += price * quantity
    
    # Add tax (10% GST in Australia)
    tax_amount = round(total * 0.1, 2)
    total_with_tax = round(total + tax_amount, 2)
    
    # Generate receipt ID
    receipt_id = f"{random.randint(1000, 9999)}-{random.randint(100000, 999999)}"
    
    # Payment method
    payment_method = random.choice(PAYMENT_METHODS)
    
    # Discount (sometimes)
    discount = None
    if random.random() < 0.2:  # 20% chance of discount
        discount_amount = round(random.uniform(1.0, total * 0.15), 2)
        discount = f"${discount_amount:.2f}"
        total_with_tax -= discount_amount
        total_with_tax = max(total_with_tax, 0.50)  # Minimum total
    
    return {
        "filename": filename,
        "image_path": f"images/{filename}",
        "receipt_count": receipt_count,
        "is_stapled": is_stapled,
        "store_name": store_name,
        "date": date,
        "time": time,
        "total_amount": f"${total_with_tax:.2f}",
        "payment_method": payment_method,
        "receipt_id": receipt_id,
        "items": items,
        "tax_info": f"GST ${tax_amount:.2f}",
        "discounts": discount
    }


@app.command()
def enhance(
    input_file: str = typer.Argument(..., help="Input CSV metadata file"),
    output_file: str = typer.Argument(..., help="Output JSON metadata file"),
    seed: int = typer.Option(
        42,
        "--seed", "-s",
        help="Random seed for reproducibility"
    ),
):
    """Enhance basic metadata with detailed extraction fields for evaluation."""
    try:
        # Validate input file
        input_path = validate_input_path(input_file)
        
        # Ensure output directory exists
        output_path = Path(output_file)
        ensure_output_dir(str(output_path.parent))
        
        # Set random seed
        random.seed(seed)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=rich_config.console,
        ) as progress:
            # Load basic metadata
            task = progress.add_task("Loading metadata...", total=None)
            print_info(rich_config, f"Loading metadata from: {input_path}")
            
            metadata_df = pd.read_csv(input_path)
            progress.update(task, total=len(metadata_df), completed=0, description="Enhancing metadata...")
            
            # Generate enhanced metadata
            enhanced_data = []
            
            for _i, (_, row) in enumerate(metadata_df.iterrows()):
                enhanced = generate_receipt_metadata(
                    filename=row['filename'],
                    receipt_count=row['receipt_count'],
                    is_stapled=row['is_stapled']
                )
                enhanced_data.append(enhanced)
                progress.advance(task)
            
            # Save enhanced metadata
            progress.update(task, description="Saving enhanced metadata...")
            output_path.write_text(json.dumps(enhanced_data, indent=2))
        
        # Display statistics with rich formatting
        rich_config.console.print("\n[bold cyan]METADATA ENHANCEMENT SUMMARY[/bold cyan]")
        rich_config.console.print("=" * 50)
        rich_config.console.print(f"[bold]Total images processed:[/bold] {len(enhanced_data)}")
        
        with_receipts = [item for item in enhanced_data if item['receipt_count'] > 0]
        without_receipts = len(enhanced_data) - len(with_receipts)
        
        rich_config.console.print(f"[bold green]Images with receipts:[/bold green] {len(with_receipts)}")
        rich_config.console.print(f"[bold blue]Images without receipts:[/bold blue] {without_receipts}")
        
        if with_receipts:
            total_items = sum(len(item['items']) for item in with_receipts)
            rich_config.console.print(f"[bold]Total items across all receipts:[/bold] {total_items}")
        
        rich_config.console.print(f"\n[bold]Enhanced metadata saved to:[/bold] {output_path}")
        
        print_success(rich_config, "Metadata enhancement completed successfully")
        
    except Exception as e:
        print_error(rich_config, f"Metadata enhancement failed: {e}")
        rich_config.console.print_exception()
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()