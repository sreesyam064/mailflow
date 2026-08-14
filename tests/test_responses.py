import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.responses import get_recommended_action, get_suggested_response


def test_recommended_action_varies_by_category_and_priority():
    high_billing = get_recommended_action("Billing", "HIGH")
    low_billing = get_recommended_action("Billing", "LOW")
    assert high_billing != low_billing


def test_recommended_action_unknown_combination_has_fallback():
    result = get_recommended_action("Nonexistent Category", "HIGH")
    assert isinstance(result, str) and len(result) > 0


def test_suggested_response_structure():
    result = get_suggested_response("Billing", "HIGH", "Negative")
    assert "subject" in result
    assert "body" in result
    assert isinstance(result["subject"], str)
    assert isinstance(result["body"], str)


def test_suggested_response_varies_by_category():
    billing_resp = get_suggested_response("Billing", "MEDIUM", "Neutral")
    sales_resp = get_suggested_response("Sales", "MEDIUM", "Neutral")
    assert billing_resp["body"] != sales_resp["body"]
    assert billing_resp["subject"] != sales_resp["subject"]


def test_suggested_response_varies_by_sentiment():
    negative_resp = get_suggested_response("Technical Support", "HIGH", "Negative")
    positive_resp = get_suggested_response("Technical Support", "HIGH", "Positive")
    assert negative_resp["body"] != positive_resp["body"]


def test_suggested_response_unknown_category_falls_back():
    result = get_suggested_response("Unknown Category", "MEDIUM", "Neutral")
    assert result["subject"] and result["body"]


def test_suggested_response_unknown_sentiment_falls_back_to_neutral():
    result = get_suggested_response("Billing", "MEDIUM", "SomeUnknownSentiment")
    neutral_result = get_suggested_response("Billing", "MEDIUM", "Neutral")
    assert result["body"] == neutral_result["body"]
