# Football Predictor 🎯

A machine learning-based football prediction system that predicts match outcomes, analyzes relevant news, and identifies value betting opportunities.

## Features

- **ML Predictions**: Uses XGBoost and LightGBM trained on historical match data
- **Elo Rating System**: Tracks team strength over time
- **Head-to-Head Analysis**: Historical matchup statistics
- **News Integration**: Analyzes news events for injury/transfer impact
- **Value Betting**: Identifies bets with positive expected value
- **Web UI**: Streamlit dashboard for viewing predictions and opportunities
- **Multiple Leagues**: Supports Premier League, La Liga, Bundesliga, Serie A, Ligue 1, and more

## Tech Stack

- **Frontend**: Streamlit
- **ML/Data**: XGBoost, LightGBM, scikit-learn, Optuna, pandas, numpy
- **Data Sources**: football-data.org, newsapi, Groq/Gemini AI
- **Web Scraping**: BeautifulSoup, Playwright
- **Python**: 3.9+

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/robinczinka96/football-predictor.git
   cd football-predictor
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env  # (if available, or create .env with required API keys)
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Required API Keys

Set these in your `.env` file:
- `FOOTBALL_DATA_API_KEY` - from football-data.org
- `NEWS_API_KEY` - from newsapi.org
- `GROQ_API_KEY` - from Groq
- `GEMINI_API_KEY` - from Google AI Studio
- `API_FOOTBALL_KEY` - from api-football.com (optional)
- Proxy settings if using SmartProxy

## Usage

### Run the Web UI
```bash
./run.sh
```
or
```bash
streamlit run ui/app.py --server.port 8501
```

This launches the Streamlit dashboard on `http://localhost:8501`

### Project Structure

```
football-predictor/
├── ui/                 # Streamlit web interface
├── model/              # ML models and predictions
│   ├── predictor.py   # Main prediction logic
│   ├── trainer.py     # Model training
│   ├── dixon_coles.py # Dixon-Coles model
│   └── combo_builder.py # Combination/parlay builder
├── data/               # Data fetching and processing
│   ├── fetcher.py     # Main data fetcher
│   └── api_football_fetcher.py
├── news/               # News analysis
│   ├── news_fetcher.py
│   └── gemini_analyzer.py
├── scraper/            # Web scraping utilities
└── config.py           # Configuration constants
```

## Model Architecture

1. **Feature Engineering**: Extracts team stats, Elo ratings, H2H records, rest days
2. **Model Training**: XGBoost/LightGBM trained on historical data
3. **Probability Blending**: Combines XGBoost, Dixon-Coles, and market odds
4. **Value Detection**: Compares model probability vs. implied odds

## Configuration

Key settings in `config.py`:
- `SUPPORTED_COMPETITIONS` - Leagues to analyze
- `VALUE_THRESHOLD` - Minimum EV for showing bets (default: 5%)
- `MIN_CONFIDENCE` - Minimum confidence for predictions (default: 40%)

## License

Private project - All rights reserved

## Contact

Created by Robin Czinka
