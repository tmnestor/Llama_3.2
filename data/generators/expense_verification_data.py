#!/usr/bin/env python3
"""
Generate multimodal data for expense verification training.

This script creates data with invoices, receipts, and bank statements for
comprehensive expense verification workflows including entity extraction
and document relationship validation.
"""
import argparse
import json
import random
import sys
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pandas as pd


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))  # noqa: E402

from data.generators.bank_statement_generator import create_bank_statement  # noqa: E402
from data.generators.receipt_generator import create_receipt  # noqa: E402
from data.generators.tax_document_generator import create_tax_document  # noqa: E402


def generate_expense_verification_questions():
    """Generate question templates for expense verification tasks."""
    templates = {
        "entity_extraction": [
            "What is the business name on this document?",
            "What is the total amount?",
            "What is the invoice date?",
            "What is the GST amount?",
            "What is the ABN?",
            "What bank issued this statement?",
            "What is the BSB code?",
            "What is the account number?",
            "Extract the transaction date.",
            "What is the withdrawal amount?",
        ],
        "document_type": [
            "What type of document is this?",
            "Is this an invoice, receipt, or bank statement?",
            "What kind of financial document is shown?",
            "Identify the document type.",
        ],
        "verification": [
            "Does this transaction match the invoice amount?",
            "Is this expense business-related?",
            "Can you verify this payment against the receipt?",
            "Does the transaction date align with the invoice date?",
            "Is this a legitimate business expense?",
        ],
        "compliance": [
            "Does this document meet expense reporting requirements?",
            "Is the GST clearly shown?",
            "Are all required details present for reimbursement?",
            "Is this document suitable for business expense claims?",
        ]
    }
    return templates


def generate_verification_qa_pair(document_type, document_info=None):
    """
    Generate a question-answer pair for expense verification.

    Args:
        document_type: Type of document (invoice, receipt, bank_statement)
        document_info: Dictionary with document details

    Returns:
        Dictionary with question and answer
    """
    templates = generate_expense_verification_questions()
    question_category = random.choice(list(templates.keys()))

    if document_type == "invoice":
        if question_category == "entity_extraction":
            questions = [
                ("What is the business name on this invoice?", "business_name"),
                ("What is the total amount?", "total_amount"),
                ("What is the invoice date?", "invoice_date"),
                ("What is the GST amount?", "gst_amount"),
                ("What is the ABN?", "abn"),
            ]
        elif question_category == "document_type":
            questions = [("What type of document is this?", "This is a tax invoice.")]
        else:
            questions = [("Is this document suitable for business expense claims?",
                         "Yes, this tax invoice contains all required details.")]

    elif document_type == "receipt":
        if question_category == "entity_extraction":
            questions = [
                ("What store issued this receipt?", "store_name"),
                ("What is the total amount?", "total_amount"),
                ("What is the date of purchase?", "date"),
                ("What payment method was used?", "payment_method"),
            ]
        elif question_category == "document_type":
            questions = [("What type of document is this?", "This is a purchase receipt.")]
        else:
            questions = [("Is this expense business-related?",
                         "This appears to be a business-related purchase.")]

    elif document_type == "bank_statement":
        if question_category == "entity_extraction":
            questions = [
                ("What bank issued this statement?", "bank_name"),
                ("What is the BSB code?", "bsb"),
                ("What is the account number?", "account_number"),
                ("What is the account holder name?", "account_holder"),
                ("What is the statement period?", "statement_period"),
            ]
        elif question_category == "document_type":
            questions = [("What type of document is this?", "This is a bank statement.")]
        else:
            questions = [("Can this document verify business expenses?",
                         "Yes, bank statements can verify expense payments.")]

    # Select random question
    question, answer_key = random.choice(questions)

    # Generate realistic answer based on document info
    if document_info and isinstance(answer_key, str) and answer_key in document_info:
        answer = str(document_info[answer_key])
    elif isinstance(answer_key, str) and answer_key.startswith("This"):
        answer = answer_key
    else:
        # Generic answers for missing info
        answer_mappings = {
            "business_name": "ABC Company Pty Ltd",
            "total_amount": "$156.75",
            "invoice_date": "2024-06-15",
            "gst_amount": "$14.25",
            "abn": "12 345 678 901",
            "store_name": "Office Supplies Store",
            "date": "15/06/2024",
            "payment_method": "EFTPOS",
            "bank_name": "Commonwealth Bank",
            "bsb": "062-001",
            "account_number": "123456789",
            "account_holder": "John Smith",
            "statement_period": "01/06/2024 to 30/06/2024"
        }
        answer = answer_mappings.get(answer_key, "Information not clearly visible.")

    return {
        "question": question,
        "answer": answer,
        "category": question_category
    }


def create_expense_verification_dataset(
    output_dir="generated_expense_data",
    num_samples=100,
    document_mix=(0.4, 0.3, 0.3)  # invoice, receipt, bank_statement ratios
):
    """
    Create a comprehensive expense verification dataset.

    Args:
        output_dir: Output directory for generated data
        num_samples: Number of document samples to generate
        document_mix: Tuple of ratios for (invoice, receipt, bank_statement)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"Generating {num_samples} expense verification documents...")

    data = []
    invoice_ratio, receipt_ratio, statement_ratio = document_mix

    for i in range(num_samples):
        # Determine document type based on mix ratios
        rand_val = random.random()
        if rand_val < invoice_ratio:
            doc_type = "invoice"
        elif rand_val < invoice_ratio + receipt_ratio:
            doc_type = "receipt"
        else:
            doc_type = "bank_statement"

        # Generate document
        filename = f"expense_doc_{i:05d}_{doc_type}.png"

        try:
            if doc_type == "invoice":
                document = create_tax_document()
                doc_info = {
                    "business_name": "ABC Professional Services",
                    "total_amount": f"${random.randint(50, 2000)}.{random.randint(10, 99)}",
                    "invoice_date": datetime.now().strftime("%d/%m/%Y"),
                    "abn": f"{random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}"
                }

            elif doc_type == "receipt":
                document = create_receipt()
                doc_info = {
                    "store_name": random.choice(["Officeworks", "Bunnings", "Woolworths", "Coles"]),
                    "total_amount": f"${random.randint(10, 500)}.{random.randint(10, 99)}",
                    "date": datetime.now().strftime("%d/%m/%Y"),
                    "payment_method": random.choice(["EFTPOS", "Credit Card", "Cash"])
                }

            else:  # bank_statement
                document = create_bank_statement()
                doc_info = {
                    "bank_name": random.choice(["Commonwealth Bank", "Westpac", "ANZ", "NAB"]),
                    "bsb": f"{random.randint(10, 99)}{random.randint(0, 9)}-{random.randint(100, 999)}",
                    "account_number": f"{random.randint(100000000, 999999999)}",
                    "account_holder": random.choice(["John Smith", "Sarah Johnson", "ABC Company Pty Ltd"]),
                    "statement_period": "01/06/2024 to 30/06/2024"
                }

            # Save document
            document.save(images_dir / filename)

            # Generate multiple QA pairs for each document
            qa_pairs = []
            for _ in range(3):  # 3 QA pairs per document
                qa_pair = generate_verification_qa_pair(doc_type, doc_info)
                qa_pairs.append(qa_pair)

            # Add to dataset
            for qa_idx, qa_pair in enumerate(qa_pairs):
                data.append({
                    "filename": filename,
                    "document_type": doc_type,
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "category": qa_pair["category"],
                    "qa_pair_idx": qa_idx
                })

        except Exception as e:
            print(f"Error generating {doc_type} document {i}: {e}")
            continue

        # Progress update
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{num_samples} documents")

    # Create and save metadata
    metadata_df = pd.DataFrame(data)
    metadata_df.to_csv(output_dir / "expense_verification_metadata.csv", index=False)

    # Save QA pairs as JSON
    qa_data = metadata_df[["filename", "document_type", "question", "answer", "category"]].to_dict(orient="records")
    qa_pairs_file = output_dir / "expense_verification_qa_pairs.json"
    with qa_pairs_file.open("w") as f:
        json.dump(qa_data, f, indent=2)

    # Generate summary statistics
    doc_type_counts = metadata_df['document_type'].value_counts()
    category_counts = metadata_df['category'].value_counts()

    summary = {
        "total_documents": num_samples,
        "total_qa_pairs": len(data),
        "document_type_distribution": doc_type_counts.to_dict(),
        "question_category_distribution": category_counts.to_dict(),
        "output_directory": str(output_dir)
    }

    with (output_dir / "dataset_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nDataset generation complete!")
    print(f"Generated {len(data)} QA pairs from {num_samples} documents")
    print(f"Document distribution: {doc_type_counts.to_dict()}")
    print(f"Question categories: {category_counts.to_dict()}")
    print(f"Output saved to: {output_dir}")


def create_related_documents_dataset(
    output_dir="generated_related_expense_data",
    num_sets=50
):
    """
    Create datasets with related documents for verification training.

    This generates sets of related documents (invoice + receipt + bank statement)
    that represent the same business expense for training verification models.

    Args:
        output_dir: Output directory for generated data
        num_sets: Number of related document sets to generate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"Generating {num_sets} related document sets...")

    data = []

    for i in range(num_sets):
        # Generate consistent expense details
        expense_amount = round(random.uniform(50.00, 1500.00), 2)
        expense_date = datetime.now() - timedelta(days=random.randint(1, 30))
        vendor_name = random.choice([
            "Officeworks", "Bunnings Warehouse", "Harvey Norman",
            "JB Hi-Fi", "The Good Guys", "Woolworths"
        ])

        set_id = f"expense_set_{i:03d}"

        # Generate invoice
        try:
            invoice = create_tax_document()
            invoice_filename = f"{set_id}_invoice.png"
            invoice.save(images_dir / invoice_filename)

            invoice_info = {
                "business_name": vendor_name,
                "total_amount": f"${expense_amount:.2f}",
                "invoice_date": expense_date.strftime("%d/%m/%Y"),
                "gst_amount": f"${expense_amount * 0.1:.2f}",
                "abn": f"{random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}"
            }

            # Generate corresponding receipt
            receipt = create_receipt()
            receipt_filename = f"{set_id}_receipt.png"
            receipt.save(images_dir / receipt_filename)

            receipt_info = {
                "store_name": vendor_name,
                "total_amount": f"${expense_amount:.2f}",
                "date": expense_date.strftime("%d/%m/%Y"),
                "payment_method": "EFTPOS"
            }

            # Generate corresponding bank statement
            statement = create_bank_statement()
            statement_filename = f"{set_id}_statement.png"
            statement.save(images_dir / statement_filename)

            statement_info = {
                "bank_name": "Commonwealth Bank",
                "withdrawal_amount": f"${expense_amount:.2f}",
                "transaction_date": expense_date.strftime("%d/%m/%Y"),
                "merchant": vendor_name
            }

            # Generate verification questions for the set
            verification_questions = [
                {
                    "question": f"Does the bank statement transaction of ${expense_amount:.2f} to {vendor_name} match the invoice total?",
                    "answer": "Yes, the amounts and vendor match between the documents.",
                    "category": "cross_document_verification"
                },
                {
                    "question": "Are the dates consistent across the invoice, receipt, and bank statement?",
                    "answer": f"Yes, all documents show the transaction date as {expense_date.strftime('%d/%m/%Y')}.",
                    "category": "date_verification"
                },
                {
                    "question": "Is this a complete set of documents for expense verification?",
                    "answer": "Yes, this includes invoice, receipt, and bank statement showing the complete transaction.",
                    "category": "completeness_check"
                }
            ]

            # Add individual document QA pairs
            for doc_type, doc_info, filename in [
                ("invoice", invoice_info, invoice_filename),
                ("receipt", receipt_info, receipt_filename),
                ("bank_statement", statement_info, statement_filename)
            ]:
                # Generate QA pairs for each document
                for _ in range(2):
                    qa_pair = generate_verification_qa_pair(doc_type, doc_info)
                    data.append({
                        "filename": filename,
                        "set_id": set_id,
                        "document_type": doc_type,
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"],
                        "category": qa_pair["category"],
                        "expense_amount": expense_amount,
                        "vendor": vendor_name,
                        "related_documents": [invoice_filename, receipt_filename, statement_filename]
                    })

            # Add verification questions for the set
            for vq in verification_questions:
                data.append({
                    "filename": f"{set_id}_all",
                    "set_id": set_id,
                    "document_type": "document_set",
                    "question": vq["question"],
                    "answer": vq["answer"],
                    "category": vq["category"],
                    "expense_amount": expense_amount,
                    "vendor": vendor_name,
                    "related_documents": [invoice_filename, receipt_filename, statement_filename]
                })

        except Exception as e:
            print(f"Error generating document set {i}: {e}")
            continue

        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_sets} document sets")

    # Save dataset
    metadata_df = pd.DataFrame(data)
    metadata_df.to_csv(output_dir / "related_documents_metadata.csv", index=False)

    # Save as JSON
    with (output_dir / "related_documents_qa_pairs.json").open("w") as f:
        json.dump(data, f, indent=2)

    print("\nRelated documents dataset complete!")
    print(f"Generated {len(data)} QA pairs from {num_sets} document sets")
    print(f"Output saved to: {output_dir}")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Generate expense verification training data")
    parser.add_argument("--output_dir", default="generated_expense_data",
                       help="Output directory for generated data")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of document samples to generate")
    parser.add_argument("--related_sets", type=int, default=20,
                       help="Number of related document sets to generate")
    parser.add_argument("--mode", choices=["individual", "related", "both"], default="both",
                       help="Generation mode: individual docs, related sets, or both")

    args = parser.parse_args()

    if args.mode in ["individual", "both"]:
        print("Generating individual documents dataset...")
        create_expense_verification_dataset(
            output_dir=args.output_dir + "_individual",
            num_samples=args.num_samples
        )

    if args.mode in ["related", "both"]:
        print("\nGenerating related documents dataset...")
        create_related_documents_dataset(
            output_dir=args.output_dir + "_related",
            num_sets=args.related_sets
        )

    print("\n✅ All datasets generated successfully!")


if __name__ == "__main__":
    main()
