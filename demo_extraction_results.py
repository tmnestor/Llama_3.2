#!/usr/bin/env python3
"""
Demonstration of optimized receipt extraction results for THE GOOD GUYS receipt
"""

import json
from pathlib import Path


def demo_extraction_results():
    """Demonstrate the expected extraction results for the test receipt."""
    
    # Expected results for the THE GOOD GUYS receipt
    expected_results = {
        "store_name": "THE GOOD GUYS",
        "date": "2023-09-26",
        "time": "14:12", 
        "total_amount": "$94.74",
        "payment_method": "CASH",
        "receipt_id": "#519544",
        "items": [
            {"item_name": "Ice Cream", "price": "$5.14"},
            {"item_name": "Beer 6-pack", "price": "$17.87"},
            {"item_name": "Bottled Water", "price": "$2.47"},
            {"item_name": "Coffee Pods", "price": "$8.74"},
            {"item_name": "Potato Chips", "price": "$4.49"},
            {"item_name": "Weet-Bix", "price": "$4.49"},
            {"item_name": "Shampoo", "price": "$5.19"},
            {"item_name": "Biscuits", "price": "$3.12"},
            {"item_name": "Paper Towels", "price": "$5.38"},
            {"item_name": "Sushi Pack", "price": "$10.31"},
            {"item_name": "Mince Beef", "price": "$8.44"},
            {"item_name": "Milo", "price": "$10.49"}
        ],
        "tax_info": "GST (10%): $8.61"
    }
    
    print("🎯 Expected Receipt Extraction Results:")
    print("=" * 50)
    print(json.dumps(expected_results, indent=2))
    
    # Save results to file for testing
    output_file = Path("expected_extraction_results.json")
    output_file.write_text(json.dumps(expected_results, indent=2))
    print(f"\n💾 Results saved to: {output_file}")
    
    # Test individual field extractions
    print(f"\n🏪 Store Name: {expected_results['store_name']}")
    print(f"📅 Date: {expected_results['date']}")
    print(f"⏰ Time: {expected_results['time']}")
    print(f"💰 Total: {expected_results['total_amount']}")
    print(f"💳 Payment: {expected_results['payment_method']}")
    print(f"🧾 Receipt ID: {expected_results['receipt_id']}")
    print(f"📦 Items: {len(expected_results['items'])} items")
    print(f"🏛️ Tax: {expected_results['tax_info']}")
    
    print("\n✅ Demonstration complete! This shows the target output for Llama-Vision extraction.")

if __name__ == "__main__":
    demo_extraction_results()