#!/usr/bin/env python3
"""
Work-related expense entity definitions for NER system.

This module defines the entities, confidence thresholds, and validation rules
for work-related expense processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class WorkExpenseCategory(Enum):
    """Standard work-related expense categories."""

    # Transport and Travel
    CAR_EXPENSES = "car_expenses"
    PUBLIC_TRANSPORT = "public_transport"
    TAXI_RIDESHARE = "taxi_rideshare"
    PARKING_TOLLS = "parking_tolls"
    ACCOMMODATION = "accommodation"
    MEALS_TRAVEL = "meals_travel"

    # Professional Development
    TRAINING_COURSES = "training_courses"
    CONFERENCES_SEMINARS = "conferences_seminars"
    PROFESSIONAL_MEMBERSHIPS = "professional_memberships"
    SUBSCRIPTIONS = "subscriptions"

    # Equipment and Tools
    TOOLS_EQUIPMENT = "tools_equipment"
    PROTECTIVE_CLOTHING = "protective_clothing"
    UNIFORMS = "uniforms"
    COMPUTER_SOFTWARE = "computer_software"
    MOBILE_PHONE = "mobile_phone"

    # Home Office
    HOME_OFFICE_RUNNING = "home_office_running"
    INTERNET_PHONE = "internet_phone"
    STATIONERY = "stationery"

    # Other Deductible
    UNION_FEES = "union_fees"
    INCOME_PROTECTION = "income_protection"
    WORK_RELATED_INSURANCE = "work_related_insurance"

    # Non-Deductible (for validation)
    PERSONAL_EXPENSES = "personal_expenses"
    PRIVATE_USE = "private_use"


class EntityType(Enum):
    """Core entity types for work expense extraction."""

    # Business Information
    BUSINESS_NAME = "business_name"
    BUSINESS_ID = "business_id"  # ABN/ACN equivalent
    BUSINESS_ADDRESS = "business_address"

    # Transaction Details
    TRANSACTION_DATE = "transaction_date"
    TRANSACTION_TIME = "transaction_time"
    TRANSACTION_ID = "transaction_id"

    # Financial Information
    TOTAL_AMOUNT = "total_amount"
    TAX_AMOUNT = "tax_amount"
    TAX_RATE = "tax_rate"
    SUBTOTAL = "subtotal"
    CURRENCY = "currency"

    # Payment Information
    PAYMENT_METHOD = "payment_method"
    CARD_NUMBER = "card_number"  # Last 4 digits only

    # Line Items
    ITEM_NAME = "item_name"
    ITEM_QUANTITY = "item_quantity"
    ITEM_UNIT_PRICE = "item_unit_price"
    ITEM_TOTAL_PRICE = "item_total_price"
    ITEM_CATEGORY = "item_category"

    # Work Expense Classification
    EXPENSE_CATEGORY = "expense_category"
    DEDUCTIBLE_PERCENTAGE = "deductible_percentage"
    BUSINESS_PURPOSE = "business_purpose"


@dataclass
class Entity:
    """Represents a single extracted entity."""

    entity_type: EntityType
    value: Any
    confidence: float
    text_span: str | None = None
    validation_status: str = "pending"  # pending, valid, invalid, requires_review
    validation_message: str | None = None


@dataclass
class LineItem:
    """Represents a single line item from a receipt/invoice."""

    name: str
    quantity: float | None = None
    unit_price: float | None = None
    total_price: float | None = None
    category: WorkExpenseCategory | None = None
    confidence: float = 0.0


@dataclass
class WorkExpenseExtraction:
    """Complete work expense extraction result."""

    # Document metadata
    document_type: str  # receipt, invoice, etc.
    processing_date: str
    confidence_score: float

    # Core entities
    entities: dict[EntityType, Entity]

    # Structured data
    business_info: dict[str, Any]
    transaction_info: dict[str, Any]
    financial_info: dict[str, Any]
    line_items: list[LineItem]

    # Classification
    expense_category: WorkExpenseCategory | None = None
    deductible_amount: float | None = None
    deductible_percentage: float | None = None

    # Validation
    validation_status: str = "pending"
    validation_errors: list[str] = None
    requires_human_review: bool = False

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []


# Confidence thresholds for different entity types
CONFIDENCE_THRESHOLDS = {
    EntityType.TOTAL_AMOUNT: 0.9,
    EntityType.TRANSACTION_DATE: 0.85,
    EntityType.BUSINESS_NAME: 0.8,
    EntityType.TAX_AMOUNT: 0.9,
    EntityType.PAYMENT_METHOD: 0.7,
    EntityType.ITEM_NAME: 0.6,
    EntityType.ITEM_TOTAL_PRICE: 0.8,
}

# Default confidence threshold
DEFAULT_CONFIDENCE_THRESHOLD = 0.75

# Business purpose keywords for automatic categorization
BUSINESS_PURPOSE_KEYWORDS = {
    WorkExpenseCategory.CAR_EXPENSES: [
        "fuel", "petrol", "diesel", "parking", "toll", "car wash", "mechanic", "service"
    ],
    WorkExpenseCategory.MEALS_TRAVEL: [
        "restaurant", "cafe", "hotel", "accommodation", "meal", "lunch", "dinner"
    ],
    WorkExpenseCategory.COMPUTER_SOFTWARE: [
        "software", "license", "subscription", "cloud", "saas", "microsoft", "adobe"
    ],
    WorkExpenseCategory.STATIONERY: [
        "office", "supplies", "paper", "pen", "notebook", "printer", "ink"
    ],
    WorkExpenseCategory.TRAINING_COURSES: [
        "training", "course", "workshop", "seminar", "conference", "education"
    ],
}
