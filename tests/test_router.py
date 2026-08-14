import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.router import route


def test_billing_routes_to_finance():
    result = route("Billing")
    assert result["department"] == "Finance"


def test_technical_support_routes_to_technical_team():
    result = route("Technical Support")
    assert result["department"] == "Technical Support Team"


def test_sales_routes_to_sales():
    result = route("Sales")
    assert result["department"] == "Sales"


def test_general_inquiry_routes_to_general_support():
    result = route("General Inquiry")
    assert result["department"] == "General Support"


def test_unknown_category_falls_back_to_general_support():
    result = route("Some Unexpected Category")
    assert result["department"] == "General Support"


def test_route_always_returns_explanation():
    for category in ["Billing", "Technical Support", "Sales", "General Inquiry", "Unknown"]:
        result = route(category)
        assert "explanation" in result
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0
