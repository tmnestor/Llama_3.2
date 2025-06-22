#!/usr/bin/env python3
"""
Bank statement generator for expense verification training data.

This module creates synthetic bank statements with realistic
variations in layout, content, and styling for comprehensive
expense verification workflows.
"""
import random
from datetime import datetime
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


# Only import numpy if available, otherwise use fallbacks
try:
    import numpy as np
except ImportError:
    # Fallback implementations for numpy functions we use
    class NumpyFallback:
        def random(self):
            class Random:
                def uniform(self, low, high):
                    return random.uniform(low, high)
                def choice(self, arr, p=None, size=None):  # noqa: ARG002
                    if p is None:
                        return random.choice(arr)
                    # Simple weighted random choice
                    import bisect
                    cumulative_sum = []
                    cumsum = 0
                    for item in p:
                        cumsum += item
                        cumulative_sum.append(cumsum)
                    x = random.random()
                    i = bisect.bisect(cumulative_sum, x)
                    return arr[i]
                def randint(self, low, high):
                    return random.randint(low, high)
                def seed(self, seed):
                    random.seed(seed)
            return Random()
    np = NumpyFallback()

# Bank statement parameters
STATEMENT_WIDTHS = range(700, 900)  # A4 width proportions
STATEMENT_HEIGHTS = range(1000, 1400)  # Variable height based on transactions
TRANSACTION_COUNTS = range(8, 25)  # Typical monthly transaction counts
STATEMENT_MARGIN = 50  # Base margin in pixels

FONT_SIZES = {
    "bank_name": 24,
    "header": 18,
    "subheader": 14,
    "normal": 12,
    "small": 10,
    "account_info": 13,
    "transaction": 11,
}

# Major banks with realistic formatting
BANKS = {
    "Commonwealth Bank": {
        "colors": {"primary": "#FFD700", "text": "#000000", "bg": "#FFFFFF"},
        "bsb_range": ("062", "066"),
        "logo_text": "CommBank",
        "phone": "13 2221",
        "website": "www.commbank.com.au"
    },
    "Westpac": {
        "colors": {"primary": "#DA1E37", "text": "#000000", "bg": "#FFFFFF"},
        "bsb_range": ("032", "034"),
        "logo_text": "Westpac",
        "phone": "132 032",
        "website": "www.westpac.com.au"
    },
    "ANZ": {
        "colors": {"primary": "#005A9C", "text": "#000000", "bg": "#FFFFFF"},
        "bsb_range": ("013", "015"),
        "logo_text": "ANZ",
        "phone": "13 13 14",
        "website": "www.anz.com.au"
    },
    "NAB": {
        "colors": {"primary": "#E31E24", "text": "#000000", "bg": "#FFFFFF"},
        "bsb_range": ("083", "085"),
        "logo_text": "NAB",
        "phone": "13 22 65",
        "website": "www.nab.com.au"
    },
    "Bendigo Bank": {
        "colors": {"primary": "#7B2D8E", "text": "#000000", "bg": "#FFFFFF"},
        "bsb_range": ("633", "635"),
        "logo_text": "Bendigo Bank",
        "phone": "1300 236 344",
        "website": "www.bendigobank.com.au"
    }
}

# Transaction types with realistic descriptions
TRANSACTION_TYPES = {
    "business_expenses": [
        ("EFTPOS PURCHASE", "OFFICEWORKS", "Office supplies"),
        ("DIRECT DEBIT", "MICROSOFT", "Software subscription"),
        ("ONLINE PURCHASE", "AMAZON AU", "Business equipment"),
        ("EFTPOS PURCHASE", "WOOLWORTHS", "Client meeting catering"),
        ("DIRECT DEBIT", "TELSTRA", "Business phone"),
        ("ONLINE PURCHASE", "BUNNINGS", "Workshop materials"),
        ("ATM WITHDRAWAL", "ANZ ATM", "Petty cash"),
        ("PAYPAL PAYMENT", "CANVA", "Design software"),
        ("EFTPOS PURCHASE", "COLES EXPRESS", "Fuel for work vehicle"),
        ("DIRECT DEBIT", "AIRTABLE", "Project management"),
    ],
    "personal_expenses": [
        ("EFTPOS PURCHASE", "MYER", "Personal shopping"),
        ("ONLINE PURCHASE", "NETFLIX", "Entertainment"),
        ("DIRECT DEBIT", "GYM MEMBERSHIP", "Personal fitness"),
        ("ATM WITHDRAWAL", "CBA ATM", "Cash withdrawal"),
        ("EFTPOS PURCHASE", "MCDONALD'S", "Personal meal"),
        ("ONLINE PURCHASE", "SPOTIFY", "Music streaming"),
        ("DIRECT DEBIT", "CAR INSURANCE", "Personal vehicle"),
        ("EFTPOS PURCHASE", "CHEMIST WAREHOUSE", "Personal items"),
    ],
    "income": [
        ("SALARY CREDIT", "EMPLOYER PTY LTD", "Monthly salary"),
        ("TRANSFER CREDIT", "TAX REFUND", "Government refund"),
        ("INTEREST CREDIT", "SAVINGS INTEREST", "Account interest"),
        ("TRANSFER CREDIT", "DIVIDEND PAYMENT", "Investment income"),
    ]
}

# Common account holder names
ACCOUNT_HOLDERS = [
    "John Smith", "Sarah Johnson", "Michael Brown", "Emma Wilson",
    "David Lee", "Lisa Zhang", "James Taylor", "Michelle Chen",
    "Robert Davis", "Amanda Thompson", "Christopher Wang", "Jennifer Liu",
    "ABC Company Pty Ltd", "XYZ Consulting", "Smith & Associates",
    "Professional Services Ltd", "Tech Solutions Pty Ltd"
]


def generate_bsb(bank_info):
    """Generate realistic BSB for the bank."""
    bsb_start, bsb_end = bank_info["bsb_range"]
    branch = random.randint(100, 999)
    return f"{random.randint(int(bsb_start), int(bsb_end))}-{branch:03d}"


def generate_account_number():
    """Generate realistic account number."""
    return f"{random.randint(100000000, 999999999)}"


def generate_transaction_date(start_date, end_date):
    """Generate random date within range."""
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randint(0, days_between)
    return start_date + timedelta(days=random_days)


def generate_transaction_amount(transaction_type):
    """Generate realistic transaction amounts based on type."""
    if transaction_type == "business_expenses":
        return round(random.uniform(15.50, 850.00), 2)
    elif transaction_type == "personal_expenses":
        return round(random.uniform(8.95, 450.00), 2)
    elif transaction_type == "income":
        return round(random.uniform(1000.00, 8500.00), 2)
    else:
        return round(random.uniform(20.00, 500.00), 2)


def generate_transactions(count, start_date, end_date, opening_balance=5000.00):
    """Generate realistic transaction list."""
    transactions = []
    current_balance = Decimal(str(opening_balance))

    # Generate transaction dates
    dates = []
    for _ in range(count):
        dates.append(generate_transaction_date(start_date, end_date))
    dates.sort()

    for date in dates:
        # Choose transaction type (weighted towards expenses)
        transaction_type = random.choices(
            ["business_expenses", "personal_expenses", "income"],
            weights=[40, 40, 20],
            k=1
        )[0]

        # Get transaction details
        method, merchant, description = random.choice(TRANSACTION_TYPES[transaction_type])
        amount = Decimal(str(generate_transaction_amount(transaction_type)))

        # Determine if withdrawal or deposit
        if transaction_type == "income":
            withdrawal = None
            deposit = amount
            current_balance += amount
        else:
            withdrawal = amount
            deposit = None
            current_balance -= amount

        transactions.append({
            "date": date,
            "description": f"{method} {merchant}",
            "withdrawal": withdrawal,
            "deposit": deposit,
            "balance": current_balance
        })

    return transactions


def get_font_path():
    """Get system font path for cross-platform compatibility."""
    possible_fonts = [
        "/System/Library/Fonts/Arial.ttf",  # macOS
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:/Windows/Fonts/arial.ttf",  # Windows
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/usr/share/fonts/TTF/arial.ttf",  # Linux alternative
    ]

    for font_path in possible_fonts:
        if Path(font_path).exists():
            return font_path

    return None  # Fall back to default font


def load_font(size):
    """Load font with fallback to default."""
    font_path = get_font_path()
    try:
        if font_path:
            return ImageFont.truetype(font_path, size)
        else:
            return ImageFont.load_default()
    except OSError:
        return ImageFont.load_default()


def create_standard_statement(width, height, transaction_count=None):
    """Create a standard bank statement layout."""
    if transaction_count is None:
        transaction_count = random.choice(TRANSACTION_COUNTS)

    # Choose bank
    bank_name, bank_info = random.choice(list(BANKS.items()))

    # Create image with bank colors
    img = Image.new('RGB', (width, height), bank_info["colors"]["bg"])
    draw = ImageDraw.Draw(img)

    # Load fonts
    fonts = {}
    for size_name, size in FONT_SIZES.items():
        fonts[size_name] = load_font(size)

    y_offset = STATEMENT_MARGIN

    # Bank header with logo area
    draw.rectangle([0, 0, width, 80], fill=bank_info["colors"]["primary"])
    draw.text((STATEMENT_MARGIN, 25), bank_info["logo_text"],
              font=fonts["bank_name"], fill=bank_info["colors"]["bg"])

    y_offset = 100

    # Statement title
    draw.text((STATEMENT_MARGIN, y_offset), "BANK STATEMENT",
              font=fonts["header"], fill=bank_info["colors"]["text"])
    y_offset += 40

    # Account information
    account_holder = random.choice(ACCOUNT_HOLDERS)
    bsb = generate_bsb(bank_info)
    account_number = generate_account_number()

    draw.text((STATEMENT_MARGIN, y_offset), f"Account Holder: {account_holder}",
              font=fonts["account_info"], fill=bank_info["colors"]["text"])
    y_offset += 25

    draw.text((STATEMENT_MARGIN, y_offset), f"BSB: {bsb}",
              font=fonts["account_info"], fill=bank_info["colors"]["text"])
    y_offset += 25

    draw.text((STATEMENT_MARGIN, y_offset), f"Account Number: {account_number}",
              font=fonts["account_info"], fill=bank_info["colors"]["text"])
    y_offset += 25

    # Statement period
    end_date = datetime.now().replace(day=1) - timedelta(days=1)  # Last day of previous month
    start_date = end_date.replace(day=1)  # First day of previous month

    draw.text((STATEMENT_MARGIN, y_offset),
              f"Statement Period: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}",
              font=fonts["normal"], fill=bank_info["colors"]["text"])
    y_offset += 40

    # Transaction header
    draw.line([STATEMENT_MARGIN, y_offset, width - STATEMENT_MARGIN, y_offset],
              fill=bank_info["colors"]["text"], width=2)
    y_offset += 10

    header_y = y_offset
    draw.text((STATEMENT_MARGIN, header_y), "Date", font=fonts["subheader"], fill=bank_info["colors"]["text"])
    draw.text((STATEMENT_MARGIN + 100, header_y), "Description", font=fonts["subheader"], fill=bank_info["colors"]["text"])
    draw.text((width - 250, header_y), "Withdrawal", font=fonts["subheader"], fill=bank_info["colors"]["text"])
    draw.text((width - 150, header_y), "Deposit", font=fonts["subheader"], fill=bank_info["colors"]["text"])
    draw.text((width - 80, header_y), "Balance", font=fonts["subheader"], fill=bank_info["colors"]["text"])

    y_offset += 30
    draw.line([STATEMENT_MARGIN, y_offset, width - STATEMENT_MARGIN, y_offset],
              fill=bank_info["colors"]["text"], width=1)
    y_offset += 15

    # Generate and display transactions
    transactions = generate_transactions(transaction_count, start_date, end_date)

    for transaction in transactions:
        if y_offset > height - 100:  # Prevent overflow
            break

        # Date
        date_str = transaction["date"].strftime("%d/%m/%Y")
        draw.text((STATEMENT_MARGIN, y_offset), date_str,
                  font=fonts["transaction"], fill=bank_info["colors"]["text"])

        # Description (truncate if too long)
        description = transaction["description"][:35]
        draw.text((STATEMENT_MARGIN + 100, y_offset), description,
                  font=fonts["transaction"], fill=bank_info["colors"]["text"])

        # Withdrawal
        if transaction["withdrawal"]:
            withdrawal_str = f"${transaction['withdrawal']:.2f}"
            draw.text((width - 250, y_offset), withdrawal_str,
                      font=fonts["transaction"], fill=bank_info["colors"]["text"])

        # Deposit
        if transaction["deposit"]:
            deposit_str = f"${transaction['deposit']:.2f}"
            draw.text((width - 150, y_offset), deposit_str,
                      font=fonts["transaction"], fill=bank_info["colors"]["text"])

        # Balance
        balance_str = f"${transaction['balance']:.2f}"
        draw.text((width - 100, y_offset), balance_str,
                  font=fonts["transaction"], fill=bank_info["colors"]["text"])

        y_offset += 20

    # Footer with bank contact info
    y_offset = height - 80
    draw.line([STATEMENT_MARGIN, y_offset, width - STATEMENT_MARGIN, y_offset],
              fill=bank_info["colors"]["text"], width=1)
    y_offset += 15

    draw.text((STATEMENT_MARGIN, y_offset), f"Phone: {bank_info['phone']}",
              font=fonts["small"], fill=bank_info["colors"]["text"])
    draw.text((width - 200, y_offset), bank_info["website"],
              font=fonts["small"], fill=bank_info["colors"]["text"])

    return img


def create_detailed_statement(width, height, transaction_count=None):
    """Create a detailed bank statement with additional information."""
    if transaction_count is None:
        transaction_count = random.choice(TRANSACTION_COUNTS)

    # Choose bank
    bank_name, bank_info = random.choice(list(BANKS.items()))

    # Create image
    img = Image.new('RGB', (width, height), bank_info["colors"]["bg"])
    draw = ImageDraw.Draw(img)

    # Load fonts
    fonts = {}
    for size_name, size in FONT_SIZES.items():
        fonts[size_name] = load_font(size)

    y_offset = STATEMENT_MARGIN

    # Detailed bank header
    draw.rectangle([0, 0, width, 100], fill=bank_info["colors"]["primary"])
    draw.text((STATEMENT_MARGIN, 15), bank_info["logo_text"],
              font=fonts["bank_name"], fill=bank_info["colors"]["bg"])
    draw.text((STATEMENT_MARGIN, 50), "Personal Banking",
              font=fonts["normal"], fill=bank_info["colors"]["bg"])
    draw.text((STATEMENT_MARGIN, 70), "Transaction Account",
              font=fonts["small"], fill=bank_info["colors"]["bg"])

    y_offset = 120

    # Statement details with more information
    draw.text((STATEMENT_MARGIN, y_offset), "ACCOUNT STATEMENT",
              font=fonts["header"], fill=bank_info["colors"]["text"])
    y_offset += 35

    # Customer and account details in two columns
    account_holder = random.choice(ACCOUNT_HOLDERS)
    bsb = generate_bsb(bank_info)
    account_number = generate_account_number()

    # Left column
    draw.text((STATEMENT_MARGIN, y_offset), "Account Holder:",
              font=fonts["normal"], fill=bank_info["colors"]["text"])
    draw.text((STATEMENT_MARGIN + 120, y_offset), account_holder,
              font=fonts["normal"], fill=bank_info["colors"]["text"])
    y_offset += 20

    draw.text((STATEMENT_MARGIN, y_offset), "BSB:",
              font=fonts["normal"], fill=bank_info["colors"]["text"])
    draw.text((STATEMENT_MARGIN + 120, y_offset), bsb,
              font=fonts["normal"], fill=bank_info["colors"]["text"])
    y_offset += 20

    draw.text((STATEMENT_MARGIN, y_offset), "Account Number:",
              font=fonts["normal"], fill=bank_info["colors"]["text"])
    draw.text((STATEMENT_MARGIN + 120, y_offset), account_number,
              font=fonts["normal"], fill=bank_info["colors"]["text"])
    y_offset += 20

    # Statement period and page info
    end_date = datetime.now().replace(day=1) - timedelta(days=1)
    start_date = end_date.replace(day=1)

    draw.text((STATEMENT_MARGIN, y_offset), "Statement Period:",
              font=fonts["normal"], fill=bank_info["colors"]["text"])
    draw.text((STATEMENT_MARGIN + 120, y_offset),
              f"{start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}",
              font=fonts["normal"], fill=bank_info["colors"]["text"])

    # Right side - account summary
    draw.text((width - 250, y_offset - 60), "Account Summary",
              font=fonts["subheader"], fill=bank_info["colors"]["text"])

    opening_balance = random.uniform(2000, 15000)
    draw.text((width - 250, y_offset - 35), f"Opening Balance: ${opening_balance:.2f}",
              font=fonts["small"], fill=bank_info["colors"]["text"])

    y_offset += 40

    # Enhanced transaction table
    draw.rectangle([STATEMENT_MARGIN, y_offset, width - STATEMENT_MARGIN, y_offset + 25],
                   fill="#F0F0F0")
    y_offset += 5

    # Headers with better spacing
    draw.text((STATEMENT_MARGIN + 5, y_offset), "Date", font=fonts["subheader"], fill=bank_info["colors"]["text"])
    draw.text((STATEMENT_MARGIN + 80, y_offset), "Description", font=fonts["subheader"], fill=bank_info["colors"]["text"])
    draw.text((STATEMENT_MARGIN + 350, y_offset), "Reference", font=fonts["subheader"], fill=bank_info["colors"]["text"])
    draw.text((width - 200, y_offset), "Debit", font=fonts["subheader"], fill=bank_info["colors"]["text"])
    draw.text((width - 130, y_offset), "Credit", font=fonts["subheader"], fill=bank_info["colors"]["text"])
    draw.text((width - 70, y_offset), "Balance", font=fonts["subheader"], fill=bank_info["colors"]["text"])

    y_offset += 25

    # Generate and display detailed transactions
    transactions = generate_transactions(transaction_count, start_date, end_date, opening_balance)

    for i, transaction in enumerate(transactions):
        if y_offset > height - 100:
            break

        # Alternate row colors
        if i % 2 == 0:
            draw.rectangle([STATEMENT_MARGIN, y_offset - 2, width - STATEMENT_MARGIN, y_offset + 18],
                          fill="#F8F8F8")

        # Date
        date_str = transaction["date"].strftime("%d/%m")
        draw.text((STATEMENT_MARGIN + 5, y_offset), date_str,
                  font=fonts["transaction"], fill=bank_info["colors"]["text"])

        # Description
        description = transaction["description"][:25]
        draw.text((STATEMENT_MARGIN + 80, y_offset), description,
                  font=fonts["transaction"], fill=bank_info["colors"]["text"])

        # Reference number
        ref_num = f"REF{random.randint(100000, 999999)}"
        draw.text((STATEMENT_MARGIN + 350, y_offset), ref_num,
                  font=fonts["small"], fill=bank_info["colors"]["text"])

        # Debit/Credit with color coding
        if transaction["withdrawal"]:
            debit_str = f"${transaction['withdrawal']:.2f}"
            draw.text((width - 200, y_offset), debit_str,
                      font=fonts["transaction"], fill="#CC0000")  # Red for debits

        if transaction["deposit"]:
            credit_str = f"${transaction['deposit']:.2f}"
            draw.text((width - 130, y_offset), credit_str,
                      font=fonts["transaction"], fill="#008000")  # Green for credits

        # Balance
        balance_str = f"${transaction['balance']:.2f}"
        draw.text((width - 85, y_offset), balance_str,
                  font=fonts["transaction"], fill=bank_info["colors"]["text"])

        y_offset += 20

    # Footer with detailed contact information
    y_offset = height - 60
    draw.line([STATEMENT_MARGIN, y_offset, width - STATEMENT_MARGIN, y_offset],
              fill=bank_info["colors"]["text"], width=1)
    y_offset += 10

    draw.text((STATEMENT_MARGIN, y_offset), f"{bank_name} - Customer Service: {bank_info['phone']}",
              font=fonts["small"], fill=bank_info["colors"]["text"])
    y_offset += 15
    draw.text((STATEMENT_MARGIN, y_offset), f"Internet Banking: {bank_info['website']}",
              font=fonts["small"], fill=bank_info["colors"]["text"])

    return img


def create_bank_statement(image_size=1200):
    """
    Create a bank statement image by randomly selecting a style.

    Args:
        image_size: Base size for the image

    Returns:
        PIL Image of the bank statement
    """
    # Choose statement style
    statement_styles = ["standard", "detailed"]
    style = random.choice(statement_styles)

    # Determine realistic dimensions
    if style == "standard":
        width = random.randint(700, 800)
        height = random.randint(1000, 1300)
    else:  # detailed
        width = random.randint(750, 850)
        height = random.randint(1100, 1400)

    # Generate transaction count
    transaction_count = random.randint(10, 22)

    # Create statement based on style
    if style == "standard":
        return create_standard_statement(width, height, transaction_count)
    else:
        return create_detailed_statement(width, height, transaction_count)


def save_bank_statement(filename, image_size=1200):
    """
    Generate and save a bank statement image.

    Args:
        filename: Output filename
        image_size: Base image size
    """
    statement = create_bank_statement(image_size)
    statement.save(filename)
    return statement


if __name__ == "__main__":
    # Generate sample bank statements
    for i in range(3):
        filename = f"sample_bank_statement_{i+1}.png"
        print(f"Generating {filename}...")
        save_bank_statement(filename)

    print("Bank statement generation complete!")
