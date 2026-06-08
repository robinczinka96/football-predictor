#!/bin/bash
# Football Predictor indítása
cd "$(dirname "$0")"
export PATH="$HOME/Library/Python/3.9/bin:$PATH"
streamlit run ui/app.py --server.port 8501 --server.headless false
