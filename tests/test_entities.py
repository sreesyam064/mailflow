import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.entities import extract_entities


def test_order_id_extraction():
    result = extract_entities("My order ORD-10482 has not shipped yet.")
    assert result["order_id"] == "10482"


def test_invoice_id_extraction():
    result = extract_entities("Please review invoice INV-99213 for discrepancies.")
    assert result["invoice_id"] == "99213"


def test_amount_extraction_dollar():
    result = extract_entities("I was charged $999 for this order.")
    assert result["amount"] == "$999"


def test_amount_extraction_rupee():
    result = extract_entities("I was charged \u20b9999 yesterday for my subscription.")
    assert result["amount"] == "\u20b9999"


def test_email_address_extraction():
    result = extract_entities("Please contact me at jane.doe@example.com for updates.")
    assert result["email_address"] == "jane.doe@example.com"


def test_missing_entities_return_none():
    result = extract_entities("I have a general question about your product.")
    assert result["order_id"] is None
    assert result["invoice_id"] is None
    assert result["amount"] is None
    assert result["email_address"] is None


def test_date_extraction_iso_format():
    result = extract_entities("The charge occurred on 2024-01-15 according to my statement.")
    assert result["date"] == "2024-01-15"


def test_never_hallucinates_when_absent():
    # Regression guard: entity extraction must return None, not a guessed
    # value, when no confident match exists.
    result = extract_entities("hello")
    assert all(v is None for v in result.values())
