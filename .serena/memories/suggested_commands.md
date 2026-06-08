# Suggested Commands

## Running the Application
```bash
# Start the Streamlit web UI (main entry point)
./run.sh

# Or directly with streamlit
streamlit run ui/app.py --server.port 8501 --server.headless false
```

## Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Python version: 3.9+
# Check: python --version
```

## Data and Model Management
```bash
# Training data is cached in a `processed_cache` file
# The project fetches data from football-data.org API

# Relevant API endpoints are in data/fetcher.py
# Model training happens in model/trainer.py
```

## System Commands
- Standard Unix: `find`, `grep`, `ls`, `cd`, `cat`
- Git management (if needed): `git status`, `git log`, etc.
- Python execution: `python script.py`

## Common Workflows
1. **Run the UI**: `./run.sh` → Opens Streamlit dashboard on port 8501
2. **Fetch data**: Logic in `data/fetcher.py` → `fetch_training_data()` and `get_upcoming_matches()`
3. **Train model**: Logic in `model/trainer.py`
4. **Make predictions**: `model/predictor.py` → `predict_all_matches()`
5. **Analyze news**: `news/news_fetcher.py` → `fetch_all_news()`
