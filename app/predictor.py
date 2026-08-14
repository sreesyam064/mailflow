"""
Loads the trained models once (singleton) and exposes prediction functions
for intent (category) and priority.

Confidence handling: LinearSVC doesn't natively output probabilities, so we use
`decision_function` margins converted to a pseudo-probability via softmax over 
class margins. This is a documented, honest approximation — not a true calibrated
probability.
"""

import joblib
import numpy as np

from app.config import (
    CONFIDENCE_THRESHOLD,
    MODELS_DIR,
    NEGATIVE_SENTIMENT_PRIORITY_BOOST_THRESHOLD,
)

_intent_model = joblib.load(MODELS_DIR / "intent_model.joblib")
_intent_vectorizer = joblib.load(MODELS_DIR / "intent_vectorizer.joblib")
_priority_model = joblib.load(MODELS_DIR / "priority_model.joblib")
_priority_vectorizer = joblib.load(MODELS_DIR / "priority_vectorizer.joblib")

_PRIORITY_ORDER = ["LOW", "MEDIUM", "HIGH"]


def _margins_to_confidence(margins: np.ndarray) -> tuple:
    # Converts LinearSVC decision_function margins to a pseudo-probability distribution
    # via softmax, then returns (predicted_index, confidence).
    # Documented approximation — LinearSVC has no native predict_proba.
    margins = np.atleast_2d(margins)
    exp = np.exp(margins - margins.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    pred_idx = int(np.argmax(probs[0]))
    confidence = float(probs[0][pred_idx])
    return pred_idx, confidence


def predict_category(text: str) -> dict:
    X = _intent_vectorizer.transform([text])
    margins = _intent_model.decision_function(X)
    pred_idx, confidence = _margins_to_confidence(margins)
    label = _intent_model.classes_[pred_idx]
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "requires_human_review": confidence < CONFIDENCE_THRESHOLD,
    }
    
    
def predict_priority(text: str, sentiment_compound: float) -> dict:
    """Predicts priority from text, then applies a documented, bounded sentiment
    nudge: if sentiment is very negative (below configured boost threshold) AND 
    model's raw prediction isn't already HIGH, bump priority up exactly one level.
    Sentiment never fully determines priority on its own — it only adjusts model's
    own prediction by one step, and adjustment is reported in response so it's never
    silently applied.
    """
    X = _priority_vectorizer.transform([text])
    margins = _priority_model.decision_function(X)
    pred_idx, confidence = _margins_to_confidence(margins)
    raw_label = _priority_model.classes_[pred_idx]
    
    final_label = raw_label
    sentiment_adjusted = False
    if (
        sentiment_compound <= NEGATIVE_SENTIMENT_PRIORITY_BOOST_THRESHOLD
        and raw_label != "HIGH"
    ):
        current_idx = _PRIORITY_ORDER.index(raw_label)
        final_label = _PRIORITY_ORDER[current_idx + 1]
        sentiment_adjusted = True
        
    return {
        "label": final_label,
        "raw_model_label": raw_label,
        "confidence": round(confidence, 4),
        "requires_human_review": confidence < CONFIDENCE_THRESHOLD,
        "sentiment_adjusted": sentiment_adjusted,
    }
    