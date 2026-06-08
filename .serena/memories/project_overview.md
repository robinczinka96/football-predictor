# Football Predictor Project

## Purpose
A machine learning-based football (soccer) prediction system that:
- Predicts match outcomes using XGBoost/LightGBM models trained on historical data
- Incorporates news analysis via Groq/Gemini APIs to adjust predictions
- Identifies value bets by comparing model probabilities against market odds
- Provides a web UI for browsing predictions and bet recommendations
- Tracks team performance using Elo ratings and streak analysis

## Technology Stack
- **Frontend**: Streamlit
- **Python Version**: 3.9+
- **ML/Data**: XGBoost, LightGBM, scikit-learn, Optuna, pandas, numpy, scipy
- **Web Scraping**: BeautifulSoup, Playwright
- **APIs**: 
  - football-data.org (match data)
  - newsapi, BBC Sport, Sky Sports (news)
  - Groq/Gemini (AI analysis)
  - API Football
- **Utilities**: requests, python-dotenv

## Key Components
1. **Data Pipeline** (`data/`): Fetch match data, build training datasets with team stats, Elo, H2H
2. **Models** (`model/`): XGBoost predictor, Dixon-Coles model, hyperparameter tuning
3. **News Analysis** (`news/`): Fetch relevant news, analyze impact on predictions
4. **Web Scraping** (`scraper/`): Vegas odds scraping
5. **UI** (`ui/app.py`): Streamlit dashboard showing predictions and bet opportunities

## Configuration
- API keys stored in `.env`
- Supported leagues: Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Championship, etc.
- Configuration constants in `config.py` (VALUE_THRESHOLD, MIN_CONFIDENCE)
