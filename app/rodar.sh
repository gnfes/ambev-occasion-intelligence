#!/usr/bin/env bash
# Sobe a API (porta 8000) e o app Streamlit (porta 8501). Ctrl+C encerra os dois.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run uvicorn occasion.api.app:app --port 8000 &
API=$!
trap 'kill $API 2>/dev/null' EXIT
export OCCASION_API_URL="${OCCASION_API_URL:-http://localhost:8000}"
uv run streamlit run app/streamlit_app.py
