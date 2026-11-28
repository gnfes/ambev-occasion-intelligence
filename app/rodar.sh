#!/usr/bin/env bash
# Sobe a API (porta 8000, ou PORTA_API) e o app Streamlit (porta 8501). Ctrl+C encerra os dois.
set -euo pipefail
cd "$(dirname "$0")/.."
PORTA="${PORTA_API:-8000}"
uv run uvicorn occasion.api.app:app --port "$PORTA" &
API=$!
trap 'kill $API 2>/dev/null' EXIT
export OCCASION_API_URL="http://localhost:$PORTA"
# Espera a API carregar o modelo antes de abrir o app (até 60 s).
for _ in $(seq 60); do
  curl -sf "$OCCASION_API_URL/health" >/dev/null && break
  sleep 1
done
if ! curl -sf "$OCCASION_API_URL/health" >/dev/null; then
  echo "A API não respondeu em $OCCASION_API_URL/health em 60 s; veja o erro acima." >&2
  exit 1
fi
uv run streamlit run app/streamlit_app.py
