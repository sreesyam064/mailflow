# application entrypoint.
# Run with: dev     -> python app.py
#           prod    -> gunicorn --bind 127.0.0.1:5000 --workers 1 --timeout 30 wsgi:app
# (prod uses wsgi.py, not app.py as Gunicorn target — see wsgi.py for why: this file and 
# app/ package share a name, which makes "app:app" ambiguous for Python's import resolution.)

from app import create_app
from app.config import DEBUG, HOST, PORT

app = create_app()

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
    
    '''
    Lightweight AI/ML email triage system — 
    classifies intent & priority (TF-IDF + scikit-learn), analyzes sentiment, extracts entities, 
    and routes support emails to the right department. Flask API + Streamlit UI, single Docker image, 
    deployed on Hugging Face Spaces with GitHub Actions CI/CD.
    '''