import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.features import FEATURE_NAMES, extract_numeric_features, features_to_array


def test_returns_all_expected_keys():
    result = extract_numeric_features("Subject", "This is a body.")
    assert set(result.keys()) == set(FEATURE_NAMES)


def test_missing_subject_flag():
    result = extract_numeric_features("", "Some body text here.")
    assert result["missing_subject"] == 1

    result2 = extract_numeric_features("Real subject", "Some body text here.")
    assert result2["missing_subject"] == 0


def test_low_information_body_flag():
    result = extract_numeric_features("Help", "please help")
    assert result["low_information_body"] == 1

    result2 = extract_numeric_features(
        "Payment issue", "I was charged twice for my subscription this month and need a refund."
    )
    assert result2["low_information_body"] == 0


def test_exclamation_count():
    result = extract_numeric_features("Urgent!!!", "This is not working!!!")
    assert result["exclamation_count"] == 6


def test_urgency_keyword_count():
    result = extract_numeric_features("Server down", "This is urgent, please help immediately, critical issue.")
    assert result["urgency_keyword_count"] >= 3


def test_urgency_keyword_count_zero_for_calm_text():
    result = extract_numeric_features("Question", "I was wondering about your pricing plans.")
    assert result["urgency_keyword_count"] == 0


def test_all_caps_ratio_detects_shouting():
    result = extract_numeric_features("THIS IS BROKEN", "PLEASE FIX THIS NOW IT IS COMPLETELY BROKEN")
    assert result["all_caps_ratio"] > 0.5


def test_all_caps_ratio_zero_for_normal_text():
    result = extract_numeric_features("Question about billing", "I have a question about my recent invoice.")
    assert result["all_caps_ratio"] == 0.0


def test_features_to_array_shape_and_order():
    result = extract_numeric_features("Subject", "Body text")
    arr = features_to_array(result)
    assert arr.shape == (1, len(FEATURE_NAMES))


def test_handles_nan_float_subject_without_crashing():
    # Regression guard: a NaN float (e.g. read back from a CSV where an
    # empty subject round-tripped through pandas) must not crash. `nan or
    # ""` silently returns nan (NaN is truthy in Python), so this only
    # passes if the isinstance(str) guard is in place.
    result = extract_numeric_features(float("nan"), "This is a real body.")
    assert result["missing_subject"] == 1
    assert isinstance(result["char_count"], int)


def test_handles_nan_float_body_without_crashing():
    result = extract_numeric_features("Real subject", float("nan"))
    assert result["low_information_body"] == 1


def test_handles_none_subject_and_body():
    result = extract_numeric_features(None, None)
    assert result["missing_subject"] == 1
    assert result["low_information_body"] == 1
