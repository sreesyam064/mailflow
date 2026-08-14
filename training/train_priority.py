"""
Model B: Ticket priority classification

Pipeline: TF-IDF -> {LogisticRegression, LinearSVC, MultinomialNB}
Trained on REAL priority labels (LOW/MEDIUM/HIGH) from dataset — not fabricated.
A separate vectorizer from Model A is used because the
vocabulary that signals urgency (e.g. "urgent", "immediately", "critical",
"asap", "down", "breach") is largely disjoint from the vocabulary that
signals intent category.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

RANDOM_SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "clean_tickets.csv"
MODELS_DIR = ROOT / "models"
EVAL_PATH = ROOT / "training" / "eval_priority.txt"

MODELS_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
        
    df["subject"] = df["subject"].fillna("")
        
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["priority"], 
        test_size=0.2, random_state=RANDOM_SEED, stratify=df["priority"]
    )
    
    # Same tuned config as Model A as a starting point
    vectorizer = TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=1,
            stop_words='english', 
            sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    candidates = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED),
        'LinearSVC': LinearSVC(class_weight='balanced', C=1.5, random_state=RANDOM_SEED, max_iter=5000),
        'MultinomialNB': MultinomialNB(),
    }
    
    results = {}
    fitted = {}
    for name, clf in candidates.items():
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)
        score = f1_score(y_test, preds, average='macro')
        results[name] = score
        fitted[name] = (clf, preds)
        print(f"{name}: macro-F1 = {score:.4f}")
        
    best_name = max(results, key=results.get)
    best_clf, best_preds = fitted[best_name]
    print(f"\nBest: {best_name} ({results[best_name]:.4f})")
    
    labels_order = sorted(df["priority"].unique(), key=lambda x: ["LOW", "MEDIUM", "HIGH"].index(x))
    report = classification_report(y_test, best_preds)
    cm = confusion_matrix(y_test, best_preds, labels=labels_order)
    
    with open(EVAL_PATH, "w") as f:
        f.write("MailFlow AI — Model B (Priority Classification) Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write("NOTE: trained on REAL priority labels from the source dataset,\n")
        f.write("not rule-derived. See README for dataset disclosure.\n\n")
        f.write("Model comparison (macro-F1):\n")
        for name, score in sorted(results.items(), key=lambda x: -x[1]):
            f.write(f"  {name}: {score:.4f}\n")
        f.write("Classification report (held-out 20% test set):\n")
        f.write(report + "\n")
        f.write(f"Confusion matrix (rows=true, cols=pred), labels order: {labels_order}:\n")
        f.write(str(cm) + "\n")
        
    joblib.dump(best_clf, MODELS_DIR / "priority_model.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "priority_vectorizer.joblib")
    with open(MODELS_DIR / "priority_model_meta.json", "w") as f:
        json.dump({"model_name": best_name, "macro_f1": results[best_name],
                   "classes": labels_order}, f, indent=2)
        
    print(f"\nSaved model + vectorizer to {MODELS_DIR}")
    print(f"Saved evaluation to {EVAL_PATH}")
    
if __name__ == "__main__":
    main()
    