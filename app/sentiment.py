"""
Lightweight sentiment analysis via VADER.

VADER is a pure lexicorn+rule-based analyzewr (no model file, no training, loads instantly) — 
deliberately chosen over a transformer. It's well-suited to short, informal text like emails/tickets.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.config import SENTIMENT_NEGATIVE_THRESHOLD, SENTIMENT_POSITIVE_THRESHOLD

_analyzer = SentimentIntensityAnalyzer()    # loaded once at import time (singleton)


def analyze_sentiment(text: str) -> dict:
    """Returns {'label': 'Positive'|'Neutral'|'Negative', 'compound': float}.
    
    'compound' is VADER's normalized score in [-1, 1] and is exposed so
    predictor.py can use its magnitude (not just the label) to decide
    whether sentiment should nudge priority up a level.
    """
    if not text or not text.strip():
        return {"label": "Neutral", "compound": 0.0}
    
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]
    
    if compound >= SENTIMENT_POSITIVE_THRESHOLD:
        label = "Positive"
    elif compound <= SENTIMENT_NEGATIVE_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"
        
    return {"label": label, "compound": round(compound, 4)}
