"""
MailFlow AI data preparation

Loads the raw multilingual ticket dataset, filters to English rows,
applies the documented queue -> category mapping, cleans text fields,
and writes a single clean CSV consumed by both training scripts.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.features import FEATURE_NAMES, extract_numeric_features

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw_tickets.csv"
CLEAN_PATH = Path(__file__).resolve().parent.parent / "data" / "clean_tickets.csv"

RANDOM_SEED = 42

# Many characters, a body carries almost no signal (e.g. "Help", "Can you assist?") — 
# confirmed by manual inspection: only 25 of 11,923 English rows fall below this threshold, 
# spread thinly across every category, so dropping them from training does not starve any class.
# NOTE: this threshold only affects which historical rows we TRAIN on.
# A real incoming email this short is still handled at inference time (flagged via low_information_body 
# feature, not rejected)
MIN_BODY_LENGTH_FOR_TRAINING = 20

# Documented queue -> category mapping

# NOTE on taxonomy (data-driven decision, see training/eval_intent.txt and training/error_analysis.md):
# an initial 6-class taxonomy (keeping Product Support / Account / General Inquiry separate) produced 
# heavy confusion btw exactly those 3 classes. Manual inspection of misclassified examples showed
# *true labels themselves* were often inconsistent for these 3 categories in underlying synthetic 
# dataset (e.g. clear bug reports labeled "General Inquiry"). Billing, Technical Support and Sales
# were textually distinct and classified well. 
# Decision: merge Product Support + Account + General Inquiry into one General Inquiry bucket rather
# than let the model be penalized for unreliable upstream labels.
QUEUE_TO_CATEGORY = {
    "Technical Support": "Technical Support",
    "IT Support": "Technical Support",
    "Service Outages and Maintenance": "Technical Support",
    "Billing and Payments": "Billing",
    "Sales and Pre-Sales": "Sales",
    "Product Support": "General Inquiry",
    "Customer Service": "General Inquiry",
    "General Inquiry": "General Inquiry",
    "Returns and Exchanges": "General Inquiry",
    # "Human Resources" intentionally dropped: internal HR tickets are out
    # of scope for a customer-facing router and the class is tiny (205 rows).
}

CATEGORY_TO_DEPARTMENT = {
    "Billing": "Finance",
    "Technical Support": "Technical Support Team",
    "Sales": "Sales",
    "General Inquiry": "General Support",
}


def clean_text(text: str) -> str:
    # Strips generator placeholder tokens & normalizes whitespace.
    # Keeps punctuation since TF-IDF bigrams benefit from phrases like 'not working'
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)    # remove <tel_num>, <url>, etc placeholders
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    if not RAW_PATH.exists():
        print(f"ERROR: raw dataset not found at {RAW_PATH}", file=sys.stderr)
        sys.exit(1)
        
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} total rows")
    
    # Filter to English only
    df = df[df['language'] == 'en'].copy()
    print(f"English rows: {len(df)}")
    
    # Drop rows with no queue/priority label (can't train without them)
    df = df.dropna(subset=["queue", "priority"])
    
    # Apply category mapping, drop unmapped (HR)
    df['category'] = df['queue'].map(QUEUE_TO_CATEGORY)
    dropped = df['category'].isna().sum()
    print(f"Dropping {dropped} rows with unmapped queue (Human Resources)")
    df = df.dropna(subset=['category'])
    
    # Clean subject/body, handle nulls
    df['subject_missing'] = df['subject'].isna().astype(int)
    df['subject'] = df['subject'].apply(clean_text)
    df['body'] = df['body'].apply(clean_text)
    
    print(f"\nRows with missing subject: {df['subject_missing'].sum()}")
    
    # Drop near-empty bodies from TRAINING data only
    before = len(df)
    df = df[df['body'].str.len() >= MIN_BODY_LENGTH_FOR_TRAINING]
    print(f"Dropped {before - len(df)} near-empty-body rows (< {MIN_BODY_LENGTH_FOR_TRAINING} chars) from training set")
    
    # Subject length outlier check (not filtered, just surfaced — a long
    # subject is unusual but not necessarily invalid signal)
    subj_len = df['subject'].str.len()
    print(f"Subject length: median={subj_len.median():.0f}, max={subj_len.max():.0f}, "
          f"rows over 150 chars: {(subj_len > 150).sum()}")
    
        # Combined text field used for TF-IDF (subject weighted by repeating it once —
    # subjects are short but high-signal for intent)
    df["text"] = (df["subject"] + " " + df["subject"] + " " + df["body"]).str.strip()
    
    # Engineered numeric features — computed identically here and at inference time (app/features.py) 
    # so there is zero train/serve skew. Deliberately does NOT use tag_1/tag_2/tag_3: those are dataset
    # annotations added after human triage and would not exist for a real incoming email — 
    # including them would be data leakage.
    feature_rows = [extract_numeric_features(s, b) for s, b in zip(df["subject"], df["body"])]
    feature_df = pd.DataFrame(feature_rows, index=df.index)
    df = pd.concat([df, feature_df], axis=1)
    
    # Normalize priority casing
    df['priority'] = df['priority'].str.upper()
    
    # Dept (deterministic, not model output — added here for sample emails file / sanity checks,
    # NOT used as a training feature)
    df["department"] = df["category"].map(CATEGORY_TO_DEPARTMENT)

    keep_cols = ['subject', 'body', 'text', 'category', 'priority', 'type', 'department'] + FEATURE_NAMES
    df = df[keep_cols].reset_index(drop=True)

    print("\nFinal category distribution:")
    print(df["category"].value_counts())
    print("\nFinal priority distribution:")
    print(df["priority"].value_counts())

    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved clean dataset to {CLEAN_PATH} ({len(df)} rows)")
    

if __name__ == "__main__":
    main()
    