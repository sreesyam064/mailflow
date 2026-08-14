import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.predictor import predict_category, predict_priority
from app.sentiment import analyze_sentiment


def test_models_load_without_error():
    # If this module imported successfully, the singleton models loaded.
    # Explicit call here forces a prediction to prove the loaded objects work.
    result = predict_category("test email")
    assert result is not None


def test_predict_category_returns_expected_structure():
    result = predict_category("I was charged twice for my subscription this month.")
    assert "label" in result
    assert "confidence" in result
    assert "requires_human_review" in result
    assert isinstance(result["confidence"], float)


def test_predict_category_confidence_in_valid_range():
    result = predict_category("Our server is completely down and throwing errors.")
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_category_label_is_known_class():
    result = predict_category("What is the pricing for your enterprise plan?")
    assert result["label"] in ("Billing", "Technical Support", "Sales", "General Inquiry")


def test_predict_priority_returns_expected_structure():
    sentiment = analyze_sentiment("This is fine, no rush.")
    result = predict_priority("This is fine, no rush at all.", sentiment["compound"])
    assert "label" in result
    assert "raw_model_label" in result
    assert "confidence" in result
    assert "sentiment_adjusted" in result
    assert result["label"] in ("LOW", "MEDIUM", "HIGH")


def test_predict_priority_confidence_in_valid_range():
    sentiment = analyze_sentiment("Please help when you get a chance.")
    result = predict_priority("Please help when you get a chance.", sentiment["compound"])
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_priority_sentiment_boost_bumps_non_high_up_one_level():
    # A strongly negative compound score should trigger the boost if the
    # raw model prediction isn't already HIGH.
    text = "This is a routine question about my account settings."
    result = predict_priority(text, sentiment_compound=-0.95)
    if result["raw_model_label"] != "HIGH":
        assert result["sentiment_adjusted"] is True
        order = ["LOW", "MEDIUM", "HIGH"]
        assert order.index(result["label"]) == order.index(result["raw_model_label"]) + 1


def test_predict_priority_no_boost_when_sentiment_is_neutral():
    text = "This is a routine question about my account settings."
    result = predict_priority(text, sentiment_compound=0.0)
    assert result["sentiment_adjusted"] is False
    assert result["label"] == result["raw_model_label"]


def test_sentiment_empty_text_returns_neutral():
    result = analyze_sentiment("")
    assert result["label"] == "Neutral"
    assert result["compound"] == 0.0


def test_sentiment_negative_text():
    result = analyze_sentiment("This is terrible, I am furious and disgusted.")
    assert result["label"] == "Negative"


def test_sentiment_positive_text():
    result = analyze_sentiment("This is wonderful, thank you so much, great work!")
    assert result["label"] == "Positive"
