# Dedicated Gunicorn Entrypoint.

# Why: this repo has both a top-level app.py (used for `python app.py` local dev) and 
# an app/ package (Flask application code). Python's import resolution gives the app/ package 
# priority over app.py when both share same name on same path, so target str "app:app" is ambiguous 
# and actually resolves to app/__init__.py (which has no top-level `app` variable) rather than 
# app.py — Gunicorn fails with "Failed to find attribute 'app' in 'app'". This file gives
# Gunicorn an unambiguous target without renaming either existing file.

from app import create_app

app = create_app()