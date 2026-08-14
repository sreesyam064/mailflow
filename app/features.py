"""
Engineeried numeric features, shared by training and inference so there is 
zero drift between the two.

"""

import re

import numpy as np

_URGENCY_WORDS = {
    "urgent", "immediately", "asap", "critical", "emergency", "down",
    "breach", "broken", "failing", "failed", "crash", "crashed", "outage",
    "unacceptable", "furious", "escalate", "immediate",
}


def _word_count(text: str) -> int:
    return len(text.split())


def _exclamation_count(text: str) -> int:
    return text.count("!")


def _all_caps_ratio(text: str) -> float:
    words = [w for w in re.findall(r"[A-Za-z]+", text) if len(w) > 2]
    if not words:
        return 0.0
    caps = sum(1 for w in words if w.isupper())
    return caps / len(words)


def _has_question_mark(text: str) -> int:
    return int("?" in text)


def _urgency_keyword_count(text: str) -> int:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & _URGENCY_WORDS)

def extract_numeric_features(subject: str, body: str) -> dict:
    # Returns flat dict of numeric features derivable from subject+body alone 
    # (no dataset-only metadata like tags). Used identically at training time
    # and inference time.
    subject = subject if isinstance(subject, str) else ""
    body = body if isinstance(body, str) else ""
    combined = f"{subject} {body}"
    
    return {
        "char_count": len(combined),
        "word_count": _word_count(combined),
        "exclamation_count": _exclamation_count(combined),
        "all_caps_ratio": round(_all_caps_ratio(combined), 4),
        "has_question_mark": _has_question_mark(combined),
        "urgency_keyword_count": _urgency_keyword_count(combined),
        "missing_subject": int(not subject.strip()),
        "low_information_body": int(len(body.strip()) < 20),
    }
    
    
FEATURE_NAMES = [
    "char_count", "word_count", "exclamation_count", "all_caps_ratio",
    "has_question_mark", "urgency_keyword_count", "missing_subject",
    "low_information_body",
]


def features_to_array(feature_dict: dict) -> np.ndarray:
    # Deterministic ordering so array matches what scaler/model was fit on.
    return np.array([[feature_dict[name] for name in FEATURE_NAMES]], dtype=float)
