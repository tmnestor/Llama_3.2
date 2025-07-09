"""
Australian Tax Office Receipt Extraction Prompts for Llama-3.2

This module contains specialized prompts for Australian tax document processing,
ported from the InternVL system to ensure domain expertise parity for fair comparison.
"""

from typing import Any, Dict, List

from ..utils import setup_logging

logger = setup_logging()


class AustralianTaxPrompts:
    """Collection of Australian tax-specific prompts for document processing."""

    def __init__(self):
        """Initialize with Australian tax prompts."""
        self.prompts = self._load_prompts()
        logger.info("Australian tax prompts loaded for Llama-3.2 system")

    def _load_prompts(self) -> Dict[str, str]:
        """Load all Australian tax prompts."""
        return {
            # === DOCUMENT CLASSIFICATION ===
            "document_classification": """
Analyze this Australian work-related expense document and classify its type.

DOCUMENT_TYPES:
1. business_receipt - General retail receipt (Woolworths, Coles, Target, etc.)
2. tax_invoice - GST tax invoice with ABN (formal business invoice)
3. bank_statement - Bank account statement
4. fuel_receipt - Petrol/diesel station receipt (BP, Shell, Caltex, etc.)
5. meal_receipt - Restaurant/cafe/catering receipt
6. accommodation - Hotel/motel/Airbnb receipt
7. travel_document - Flight/train/bus ticket or travel booking
8. parking_toll - Parking meter/garage or toll road receipt
9. equipment_supplies - Office supplies/tools/equipment receipt
10. professional_services - Legal/accounting/consulting invoice
11. other - Any other work-related document

CLASSIFICATION_CRITERIA:
- Look for business names, logos, and document layout
- Identify specific industry indicators (fuel company logos, hotel chains, etc.)
- Check for formal invoice elements (ABN, tax invoice headers)
- Consider document structure and typical content

RESPONSE_FORMAT:
DOCUMENT_TYPE: [type from list above]
CONFIDENCE: [High/Medium/Low]
REASONING: [Brief explanation of classification decision]
SECONDARY_TYPE: [Alternative type if confidence is not High]

Focus on Australian businesses and document formats.
""",
            # === KEY-VALUE RECEIPT PROMPT ===
            "key_value_receipt": """
Extract information from this Australian receipt and return in KEY-VALUE format.

Use this exact format:
DATE: [purchase date in DD/MM/YYYY format]
STORE: [store name in capitals]
ABN: [Australian Business Number - XX XXX XXX XXX format]
PAYER: [customer/member name if visible]
TAX: [GST amount]
TOTAL: [total amount including GST]
PRODUCTS: [item1 | item2 | item3]
QUANTITIES: [qty1 | qty2 | qty3]
PRICES: [price1 | price2 | price3]

Example:
DATE: 16/03/2023
STORE: WOOLWORTHS
TAX: 3.82
TOTAL: 42.08
PRODUCTS: Milk 2L | Bread Multigrain | Eggs Free Range 12pk
QUANTITIES: 1 | 2 | 1
PRICES: 4.50 | 8.00 | 7.60

FORMATTING REQUIREMENTS:
- Product names: Use Title Case (Milk 2L, not MILK 2L or milk 2l)
- Prices: Read carefully from receipt, match exact amounts shown
- Store names: Use ALL CAPITALS (WOOLWORTHS, COLES, ALDI)
- Dates: DD/MM/YYYY format (Australian standard)
- Use pipe (|) to separate multiple items in lists
- Extract ALL products from the receipt
- Ensure products, quantities, and prices lists have same length

CRITICAL: 
- Product names must be in Title Case format: "Chicken Breast" not "CHICKEN BREAST"
- Read prices carefully from receipt - accuracy is essential
- GST (Goods and Services Tax) is 10% in Australia

Return ONLY the key-value pairs above. No explanations.
""",
            # === BUSINESS RECEIPT EXTRACTION ===
            "business_receipt_extraction": """
Extract information from this Australian business receipt in KEY-VALUE format.

CRITICAL: Use EXACT KEY-VALUE format below - NO JSON, NO other formats.

STORE: [Business name in CAPITALS]
ABN: [Australian Business Number if visible - XX XXX XXX XXX format]
DATE: [DD/MM/YYYY format]
GST: [GST amount - 10% component]
TOTAL: [Total amount including GST]
ITEMS: [Product1 | Product2 | Product3]
QUANTITIES: [Qty1 | Qty2 | Qty3]  
PRICES: [Price1 | Price2 | Price3]

EXAMPLE OUTPUT:
STORE: WOOLWORTHS SUPERMARKETS
ABN: 88 000 014 675
DATE: 15/06/2024
GST: 4.25
TOTAL: 46.75
ITEMS: Bread White | Milk 2L | Eggs Free Range 12pk
QUANTITIES: 1 | 1 | 1
PRICES: 3.50 | 5.20 | 8.95

ATO_COMPLIANCE_REQUIREMENTS:
- Business name and date are mandatory
- GST component required for claims over $82.50
- ABN validates legitimate business expense
- Use pipe (|) separator for multiple items
""",
            # === FUEL RECEIPT EXTRACTION ===
            "fuel_receipt_extraction": """
Extract information from this Australian fuel receipt for work vehicle expense claims.

CRITICAL: Use EXACT KEY-VALUE format below - NO JSON, NO other formats.

STATION: [Fuel station name - BP, Shell, Caltex, Ampol, etc.]
STATION_ADDRESS: [Station location if visible]
DATE: [DD/MM/YYYY]
TIME: [HH:MM if visible]
FUEL_TYPE: [Unleaded, Premium, Diesel, etc.]
LITRES: [Fuel quantity in litres]
PRICE_PER_LITRE: [Rate per litre - cents format]
TOTAL_FUEL_COST: [Total fuel amount before other items]
GST: [GST component]
TOTAL: [Total amount including GST]
PUMP_NUMBER: [Pump number if visible]
VEHICLE_KM: [Odometer reading if visible]

EXAMPLE OUTPUT:
STATION: BP AUSTRALIA
STATION_ADDRESS: 123 Main Street, Melbourne VIC
DATE: 15/06/2024
TIME: 14:35
FUEL_TYPE: Unleaded 91
LITRES: 45.20
PRICE_PER_LITRE: 189.9
TOTAL_FUEL_COST: 85.85
GST: 7.81
TOTAL: 85.85
PUMP_NUMBER: 3
VEHICLE_KM: 45230

ATO_FUEL_REQUIREMENTS:
- Date, station name, and total amount are mandatory
- Litres and rate per litre support logbook method claims
- GST breakdown essential for business vehicle deductions
- Vehicle odometer helps verify business vs personal use
""",
            # === TAX INVOICE EXTRACTION ===
            "tax_invoice_extraction": """
Extract information from this Australian GST tax invoice for business expense claims.

CRITICAL: Use EXACT KEY-VALUE format below - NO JSON, NO other formats.

DOCUMENT_TYPE: [Must contain "TAX INVOICE" or "INVOICE"]
SUPPLIER: [Business/company name]
SUPPLIER_ABN: [Supplier's ABN - XX XXX XXX XXX format]
SUPPLIER_ADDRESS: [Supplier's business address]
CUSTOMER: [Customer/client name]
CUSTOMER_ABN: [Customer's ABN if visible]
INVOICE_NUMBER: [Invoice reference number]
DATE: [Invoice date DD/MM/YYYY]
DUE_DATE: [Payment due date if specified]
DESCRIPTION: [Services/goods description]
SUBTOTAL: [Amount before GST]
GST: [GST amount - must be specified separately]
TOTAL: [Total amount including GST]

EXAMPLE OUTPUT:
DOCUMENT_TYPE: TAX INVOICE
SUPPLIER: ACME CONSULTING PTY LTD
SUPPLIER_ABN: 12 345 678 901
SUPPLIER_ADDRESS: 456 Business Street, Sydney NSW 2000
CUSTOMER: CLIENT COMPANY PTY LTD
INVOICE_NUMBER: INV-2024-0156
DATE: 15/06/2024
DUE_DATE: 15/07/2024
DESCRIPTION: Professional consulting services
SUBTOTAL: 500.00
GST: 50.00
TOTAL: 550.00

TAX_INVOICE_REQUIREMENTS:
- Must contain "TAX INVOICE" text on document
- Supplier ABN mandatory for invoices over $82.50
- GST amount must be specified separately from subtotal
- Essential for business expense claims and BAS reporting
""",
            # === BANK STATEMENT PROCESSING ===
            "bank_statement_ato_compliance": """
Extract bank statement information for Australian Tax Office work-related expense claims.

ATO REQUIREMENTS for bank statement evidence:
1. Transaction date and description
2. Amount of expense
3. Business purpose (if determinable from description)
4. Account holder name matching taxpayer

EXTRACTION_PRIORITIES:
1. HIGHLIGHTED TRANSACTIONS (user-marked as work expenses)
2. Business-relevant merchants (Officeworks, petrol stations, airlines)
3. Professional services (accounting, legal, consulting)
4. Travel and transport expenses
5. Equipment and supply purchases

BANK STATEMENT FORMAT:
BANK: [Financial institution name]
ACCOUNT_HOLDER: [Customer name]
ACCOUNT_NUMBER: [Account number - mask middle digits]
BSB: [Bank State Branch code if visible]
STATEMENT_PERIOD: [Start date - End date in DD/MM/YYYY format]
OPENING_BALANCE: [Starting balance]
CLOSING_BALANCE: [Ending balance]

TRANSACTIONS:
DATE: [DD/MM/YYYY] | DESCRIPTION: [Transaction description] | DEBIT: [Amount withdrawn] | CREDIT: [Amount deposited] | BALANCE: [Balance after transaction] | WORK_RELEVANCE: [High/Medium/Low/None]

WORK_RELEVANCE_CRITERIA:
- High: Clear work expenses (fuel, office supplies, professional services)
- Medium: Potentially work-related (meals, equipment, training)
- Low: Possibly work-related (general purchases, subscriptions)
- None: Personal expenses (groceries, entertainment, personal items)

COMPLIANCE_ASSESSMENT:
- Rate each transaction's ATO compliance (0-100%)
- Identify missing information for full deductibility
- Suggest additional documentation needed

Extract ALL visible transactions and assess their work-related potential.

CRITICAL: Use ONLY the exact format specified above. Do NOT use markdown, bullets, or natural language formatting.
""",
            # === MEAL RECEIPT EXTRACTION ===
            "meal_receipt_extraction": """
Extract information from this Australian meal receipt for business entertainment claims.

CRITICAL: Use EXACT KEY-VALUE format below - NO JSON, NO other formats.

RESTAURANT: [Restaurant/cafe name]
RESTAURANT_ABN: [ABN if visible]
DATE: [DD/MM/YYYY]
TIME: [HH:MM if visible]
MEAL_TYPE: [Breakfast/Lunch/Dinner/Coffee/etc.]
ITEMS: [Food item1 | Drink item1 | Food item2]
PRICES: [Price1 | Price2 | Price3]
SUBTOTAL: [Amount before GST]
GST: [GST amount]
TOTAL: [Total amount including GST]
PAYMENT_METHOD: [Cash/Card/EFTPOS]
COVERS: [Number of people if visible]

EXAMPLE OUTPUT:
RESTAURANT: CAFE MELBOURNE
RESTAURANT_ABN: 23 456 789 012
DATE: 15/06/2024
TIME: 12:30
MEAL_TYPE: Lunch
ITEMS: Chicken Caesar Salad | Coffee Latte | Sparkling Water
PRICES: 18.50 | 4.50 | 3.50
SUBTOTAL: 24.09
GST: 2.41
TOTAL: 26.50
PAYMENT_METHOD: EFTPOS
COVERS: 2

ATO_MEAL_REQUIREMENTS:
- Business purpose documentation required
- Entertainment expenses have limited deductibility
- GST breakdown essential for business claims
- Number of attendees affects deductibility
""",
            # === ACCOMMODATION RECEIPT ===
            "accommodation_extraction": """
Extract information from this Australian accommodation receipt for business travel claims.

CRITICAL: Use EXACT KEY-VALUE format below - NO JSON, NO other formats.

HOTEL: [Hotel/accommodation name]
HOTEL_ABN: [ABN if visible]
ADDRESS: [Hotel address]
DATE_CHECKIN: [Check-in date DD/MM/YYYY]
DATE_CHECKOUT: [Check-out date DD/MM/YYYY]
NIGHTS: [Number of nights]
ROOM_TYPE: [Room type/description]
ROOM_RATE: [Rate per night]
SUBTOTAL: [Amount before GST]
GST: [GST amount]
TOTAL: [Total amount including GST]
GUEST_NAME: [Guest name]
PAYMENT_METHOD: [Cash/Card/EFTPOS]

EXAMPLE OUTPUT:
HOTEL: HILTON SYDNEY
HOTEL_ABN: 12 345 678 901
ADDRESS: 488 George Street, Sydney NSW 2000
DATE_CHECKIN: 15/06/2024
DATE_CHECKOUT: 17/06/2024
NIGHTS: 2
ROOM_TYPE: Standard King Room
ROOM_RATE: 250.00
SUBTOTAL: 454.55
GST: 45.45
TOTAL: 500.00
GUEST_NAME: JOHN SMITH
PAYMENT_METHOD: CREDIT CARD

ATO_ACCOMMODATION_REQUIREMENTS:
- Business purpose documentation required
- Travel dates and destination mandatory
- GST breakdown essential for business claims
- Reasonable accommodation standards apply
""",
            # === PARKING/TOLL RECEIPT ===
            "parking_toll_extraction": """
Extract information from this Australian parking or toll receipt for vehicle expense claims.

CRITICAL: Use EXACT KEY-VALUE format below - NO JSON, NO other formats.

OPERATOR: [Parking/toll operator name]
LOCATION: [Parking location or toll road]
DATE: [DD/MM/YYYY]
TIME_ENTRY: [Entry time if visible]
TIME_EXIT: [Exit time if visible]
DURATION: [Parking duration if visible]
VEHICLE_PLATE: [Vehicle registration if visible]
RATE: [Hourly rate or toll amount]
GST: [GST amount if applicable]
TOTAL: [Total amount]
PAYMENT_METHOD: [Cash/Card/EFTPOS]
RECEIPT_NUMBER: [Receipt number]

EXAMPLE OUTPUT:
OPERATOR: SECURE PARKING
LOCATION: 123 Collins Street, Melbourne VIC
DATE: 15/06/2024
TIME_ENTRY: 09:00
TIME_EXIT: 17:30
DURATION: 8.5 hours
VEHICLE_PLATE: ABC123
RATE: 12.00
GST: 9.09
TOTAL: 109.00
PAYMENT_METHOD: CREDIT CARD
RECEIPT_NUMBER: 789456123

ATO_PARKING_REQUIREMENTS:
- Business purpose documentation required
- Vehicle registration helps verify business use
- GST breakdown for claims over $82.50
- Duration supports business activity verification
""",
            # === EQUIPMENT/SUPPLIES RECEIPT ===
            "equipment_supplies_extraction": """
Extract information from this Australian equipment/supplies receipt for business expense claims.

CRITICAL: Use EXACT KEY-VALUE format below - NO JSON, NO other formats.

STORE: [Store name - Officeworks, JB Hi-Fi, etc.]
STORE_ABN: [ABN if visible]
DATE: [DD/MM/YYYY]
ITEMS: [Item1 | Item2 | Item3]
DESCRIPTIONS: [Description1 | Description2 | Description3]
QUANTITIES: [Qty1 | Qty2 | Qty3]
PRICES: [Price1 | Price2 | Price3]
SUBTOTAL: [Amount before GST]
GST: [GST amount]
TOTAL: [Total amount including GST]
PAYMENT_METHOD: [Cash/Card/EFTPOS]
WARRANTY: [Warranty information if visible]

EXAMPLE OUTPUT:
STORE: OFFICEWORKS
STORE_ABN: 88 000 014 675
DATE: 15/06/2024
ITEMS: Laptop Computer | Wireless Mouse | USB Cable
DESCRIPTIONS: Dell Inspiron 15 | Logitech MX Master | USB-C to USB-A
QUANTITIES: 1 | 1 | 2
PRICES: 899.00 | 129.00 | 15.00
SUBTOTAL: 947.27
GST: 94.73
TOTAL: 1042.00
PAYMENT_METHOD: CREDIT CARD
WARRANTY: 1 year manufacturer warranty

ATO_EQUIPMENT_REQUIREMENTS:
- Business purpose documentation required
- Depreciation may apply for expensive items
- GST breakdown essential for business claims
- Warranty information supports asset tracking
""",
            # === PROFESSIONAL SERVICES INVOICE ===
            "professional_services_extraction": """
Extract information from this Australian professional services invoice for business expense claims.

CRITICAL: Use EXACT KEY-VALUE format below - NO JSON, NO other formats.

DOCUMENT_TYPE: [INVOICE or TAX INVOICE]
FIRM: [Professional firm name]
FIRM_ABN: [Firm's ABN - XX XXX XXX XXX format]
FIRM_ADDRESS: [Firm's business address]
PROFESSIONAL: [Professional's name if visible]
CLIENT: [Client name]
INVOICE_NUMBER: [Invoice reference number]
DATE: [Invoice date DD/MM/YYYY]
SERVICE_PERIOD: [Service period if specified]
DESCRIPTION: [Services description]
HOURS: [Hours worked if visible]
RATE: [Hourly rate if visible]
SUBTOTAL: [Amount before GST]
GST: [GST amount]
TOTAL: [Total amount including GST]
PAYMENT_TERMS: [Payment terms if visible]

EXAMPLE OUTPUT:
DOCUMENT_TYPE: TAX INVOICE
FIRM: SMITH & ASSOCIATES LEGAL
FIRM_ABN: 12 345 678 901
FIRM_ADDRESS: 789 Legal Street, Sydney NSW 2000
PROFESSIONAL: JANE SMITH
CLIENT: BUSINESS PTY LTD
INVOICE_NUMBER: INV-2024-0789
DATE: 15/06/2024
SERVICE_PERIOD: 01/06/2024 - 15/06/2024
DESCRIPTION: Legal advice and contract review
HOURS: 8.5
RATE: 350.00
SUBTOTAL: 2727.27
GST: 272.73
TOTAL: 3000.00
PAYMENT_TERMS: Net 30 days

ATO_PROFESSIONAL_SERVICES_REQUIREMENTS:
- Business purpose documentation required
- Service description essential for deductibility
- GST breakdown mandatory for business claims
- Professional firm ABN validates legitimate expense
""",
        }

    def get_prompt(self, prompt_type: str) -> str:
        """
        Get a specific prompt by type.

        Args:
            prompt_type: Type of prompt to retrieve

        Returns:
            Prompt string or empty string if not found
        """
        prompt = self.prompts.get(prompt_type, "")
        if not prompt:
            logger.warning(f"Prompt type '{prompt_type}' not found")
        return prompt

    def get_document_classification_prompt(self) -> str:
        """Get document classification prompt."""
        return self.get_prompt("document_classification")

    def get_extraction_prompt(self, document_type: str) -> str:
        """
        Get extraction prompt for specific document type.

        Args:
            document_type: Type of document

        Returns:
            Appropriate extraction prompt
        """
        prompt_mapping = {
            "business_receipt": "business_receipt_extraction",
            "fuel_receipt": "fuel_receipt_extraction",
            "tax_invoice": "tax_invoice_extraction",
            "bank_statement": "bank_statement_ato_compliance",
            "meal_receipt": "meal_receipt_extraction",
            "accommodation": "accommodation_extraction",
            "parking_toll": "parking_toll_extraction",
            "equipment_supplies": "equipment_supplies_extraction",
            "professional_services": "professional_services_extraction",
            "other": "key_value_receipt",
        }

        prompt_type = prompt_mapping.get(document_type, "key_value_receipt")
        return self.get_prompt(prompt_type)

    def get_all_prompt_types(self) -> List[str]:
        """Get list of all available prompt types."""
        return list(self.prompts.keys())

    def get_australian_business_keywords(self) -> List[str]:
        """Get list of Australian business keywords for classification."""
        return [
            # Major retailers
            "woolworths",
            "coles",
            "aldi",
            "target",
            "kmart",
            "bunnings",
            "officeworks",
            "harvey norman",
            "jb hi-fi",
            "big w",
            "myer",
            "david jones",
            "ikea",
            # Fuel stations
            "bp",
            "shell",
            "caltex",
            "ampol",
            "mobil",
            "7-eleven",
            "united petroleum",
            # Banks
            "anz",
            "commonwealth bank",
            "westpac",
            "nab",
            "ing",
            "macquarie",
            "bendigo bank",
            "suncorp",
            "bank of queensland",
            "credit union",
            # Airlines
            "qantas",
            "jetstar",
            "virgin australia",
            "tigerair",
            "rex airlines",
            # Car rental
            "avis",
            "hertz",
            "budget",
            "thrifty",
            "europcar",
            "redspot",
            # Fast food
            "mcdonald's",
            "kfc",
            "subway",
            "domino's",
            "pizza hut",
            "hungry jack's",
            "red rooster",
            "nando's",
            "guzman y gomez",
            "zambrero",
            # Hotels
            "hilton",
            "marriott",
            "hyatt",
            "ibis",
            "mercure",
            "novotel",
            "crowne plaza",
            "holiday inn",
            "radisson",
            "sheraton",
            # Professional services
            "deloitte",
            "pwc",
            "kpmg",
            "ey",
            "bdo",
            "rsm",
            "pitcher partners",
            "allens",
            "ashurst",
            "clayton utz",
            "corrs",
            "herbert smith freehills",
            # Parking
            "secure parking",
            "wilson parking",
            "ace parking",
            "care park",
            "parking australia",
            "premium parking",
        ]

    def get_gst_validation_rules(self) -> Dict[str, Any]:
        """Get GST validation rules for Australian tax compliance."""
        return {
            "gst_rate": 0.10,  # 10% GST in Australia
            "gst_tolerance": 0.02,  # 2 cent tolerance for rounding
            "receipt_threshold": 82.50,  # ATO requires receipt for claims over $82.50
            "abn_threshold": 82.50,  # ABN required for claims over $82.50
            "gst_free_items": [
                "basic food items",
                "milk",
                "bread",
                "meat",
                "vegetables",
                "fruit",
                "medical services",
                "education",
                "childcare",
                "health services",
            ],
            "gst_applicable_items": [
                "processed food",
                "restaurant meals",
                "alcohol",
                "fuel",
                "clothing",
                "electronics",
                "services",
                "entertainment",
                "accommodation",
            ],
        }


# Global instance for easy access
australian_tax_prompts = AustralianTaxPrompts()


def get_australian_tax_prompt(prompt_type: str) -> str:
    """
    Get Australian tax prompt by type.

    Args:
        prompt_type: Type of prompt to retrieve

    Returns:
        Prompt string
    """
    return australian_tax_prompts.get_prompt(prompt_type)


def get_document_extraction_prompt(document_type: str) -> str:
    """
    Get document extraction prompt for specific document type.

    Args:
        document_type: Type of document

    Returns:
        Appropriate extraction prompt
    """
    return australian_tax_prompts.get_extraction_prompt(document_type)


def get_classification_prompt() -> str:
    """Get document classification prompt."""
    return australian_tax_prompts.get_document_classification_prompt()
