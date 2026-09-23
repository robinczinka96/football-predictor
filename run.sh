#!/usr/bin/env bash
set -euo pipefail
# Football Predictor indítása
cd "$(dirname "$0")"
exec streamlit run ui/app.py \
  --server.address "${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}" \
  --server.port "${STREAMLIT_SERVER_PORT:-8501}" \
  --server.headless "${STREAMLIT_SERVER_HEADLESS:-true}"
