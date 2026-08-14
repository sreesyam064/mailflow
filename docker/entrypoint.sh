#!/bin/sh
# Starts Gunicorn (Flask API) as a background process bound to 127.0.0.1:5000 (internal only,
# never exposed by container), then exec Streamlit in foreground on $PORT ((7860 for Hugging Face Spaces, 
# overrides elsewhere). Streamlit is container's public/foreground process,
# Gunicorn is started exactly once here, not by Streamlit or any user session.
set -e

gunicorn \
    --bind 127.0.0.1:5000 \
    --workers 1 \
    --timeout 30 \
    wsgi:app &

# Wait for Flask to actually accept connections before starting Streamlit, instead of a
# fixed sleep — fails fast and loudly if Gunicorn never comes up, rather than Streamlit
# silently against a backend that isn't there yet.
echo "Waiting for Flask (Gunicorn) to become ready on 127.0.0.1:5000..."
for i in $(seq 1 30); do
    if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=1)" 2>/dev/null; then
        echo "Flask is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Flask did not become ready in time." >&2
        exit 1
    fi
    sleep 1
done

exec streamlit run frontend/app.py \
    --server.port="${PORT:-7860}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
    