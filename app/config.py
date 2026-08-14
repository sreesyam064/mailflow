# Centralized configuration via environment variables

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

# Confidence below this -> requires_human_review = True
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))

# Max characters accepted for subject + body combined (input validation / abuse guard)
MAX_EMAIL_LENGTH = int(os.getenv("MAX_EMAIL_LENGTH", "20000"))
MIN_EMAIL_LENGTH = int(os.getenv("MIN_EMAIL_LENGTH", "3"))

# Senttiment score below this (VADER compound, [-1, 1]) count as negative
# above this ccounts as positives, between is neutral.
SENTIMENT_NEGATIVE_THRESHOLD = float(os.getenv("SENTIMENT_NEGATIVE_THRESHOLD", "-0.05"))
SENTIMENT_POSITIVE_THRESHOLD = float(os.getenv("SENTIMENT_POSITIVE_THRESHOLD", "0.05"))

# Sentiment can bump priority up by one level when very negative, but never fully
# determines priority on its own 
NEGATIVE_SENTIMENT_PRIORITY_BOOST_THRESHOLD = float(os.getenv("NEGATIVE_SENTIMENT_PRIORITY_BOOST_THRESHOLD", "-0.5"))

HOST = os.getenv("HOST", "0.0.0.0")
PORT = os.getenv("PORT", "5000")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
