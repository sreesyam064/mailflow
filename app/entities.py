"""
Lightweight regex-based entity extraction.

Deliberately NOT an NER model. Each extractor returns None when it can't 
confidently match rather than guessing, per spec: "If an entity cannot be confidently
extracted, return null rather than hallucinating it."
"""

import re

_ORDER_ID_RE = re.compile(r"\b(?:order|ord)\b[\s#:-]*([A-Z0-9]{4,12})\b", re.IGNORECASE)
_INVOICE_ID_RE = re.compile(r"\b(?:invoice|inv)\b[\s#:-]*([A-Z0-9]{4,12})\b", re.IGNORECASE)
_TICKET_ID_RE = re.compile(r"\b(?:ticket)\b[\s#:-]*([A-Z0-9]{4,12})\b", re.IGNORECASE)

# matches $999, ₹999, €999.50, 999 USD, 999.99
_AMOUNT_RE = re.compile(
    r"(?:[$₹€£]\s?\d[\d,]*(?:\.\d{1,2})?)|(?:\d[\d,]*(?:\.\d{1,2})?\s?(?:USD|INR|EUR|dollars|rupees))",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Common date formats: 2024-01-15, 01/15/2024. 15 Jan 2024, January 15, 2024
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
    re.IGNORECASE,
)


def _first_match(pattern: re.Pattern, text: str, group: int = 0):
    m = pattern.search(text)
    if not m:
        return None
    return m.group(group).strip()


def extract_entities(text: str) -> dict:
    # Extract order ID, invoice ID, ticket ID, amount, email, date.
    # Missing entities are returned as None, never guessed.
    return {
        "order_id": _first_match(_ORDER_ID_RE, text, group=1),
        "invoice_id": _first_match(_INVOICE_ID_RE, text, group=1),
        "ticket_id": _first_match(_TICKET_ID_RE, text, group=1),
        "amount": _first_match(_AMOUNT_RE, text),
        "email_address": _first_match(_EMAIL_RE, text),
        "date": _first_match(_DATE_RE, text),
    }
    