# MailFlow AI — production image

# Runs Gunicorn (Flask API, internal-only on 127.0.0.1:5000) and Streamlit
# (public, foreground, on $PORT) as two processes inside a single container.
# This single-container design works unchanged across Render that runs one Docker 
# container per service — Streamlit is one public port; Flask is never exposed outside container.
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files / buffering stdout — cleaner container
# logs, no unnecessary sidk writes.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# System deps kept minimal on purpose (spec: lightweight, no GPU/CUDA)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY frontend/ ./frontend/
COPY models/ ./models/
COPY app.py .
COPY wsgi.py .
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Default port for local `docker run` without -e PORT=... . Render inject their own PORT at container runtime,
# which overrides this — docker/entrypoint.sh already reads it via ${PORT:-7860}, so no code change needed to support Render.
# Flask/Gunicorn always binds to 127.0.0.1:5000 internally — it is never exposed outside the container, on any platform..
ENV PORT=7860 \
    API_URL=http://127.0.0.1:5000

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health')" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
