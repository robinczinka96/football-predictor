# Code Style and Conventions

## Python Style
- **Type Hints**: Used on function parameters and returns (see `model/predictor.py`)
- **Naming**: Mix of snake_case and camelCase; some Hungarian variable names/comments
- **Docstrings**: Minimal; comments are inline where needed
- **Comments**: Mix of Hungarian and English

## Patterns Observed
- Configuration via dictionaries (e.g., `SUPPORTED_COMPETITIONS` dict in config.py)
- Feature engineering functions that return dictionaries (e.g., `get_team_features()`)
- Model loading/caching pattern with `load_model()`
- Blending probabilities from multiple models (XGBoost + Dixon-Coles + market odds)

## File Organization
- One main responsibility per module
- Utility functions grouped in feature files (e.g., `fetcher.py` handles all data loading)
- Streamlit app uses functions for UI components (e.g., `bet_label()`, `conf_color()`)

## Data Structures
- DataFrames for historical match data
- Dictionaries for predictions, team stats, and features
- JSON-compatible output formats
