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

## Docker

The production image runs as an unprivileged user and includes a health check:

```bash
docker build -t football-predictor .
docker run --rm -p 8501:8501 --env-file .env football-predictor
```

For a production-like local run, create `.env.app` from `.env.example`, then run:

```bash
printf 'IMAGE=football-predictor\nIMAGE_TAG=latest\n' > .env.production
docker compose --env-file .env.production -f compose.production.yml up -d
```

The service binds to `127.0.0.1:8501` by default, ready to sit behind an HTTPS
reverse proxy. Set `APP_BIND_ADDRESS=0.0.0.0` only when the port must be exposed
directly. Cached datasets and trained models are kept in named Docker volumes.

## Automatic VPS deployment with GitHub Actions

`.github/workflows/deploy.yml` builds every pull request, and on pushes to `main`
publishes an immutable image to GitHub Container Registry before deploying it.
The deployment waits for the Streamlit health check and fails if the new container
does not become healthy.

### One-time VPS preparation

1. Install Docker Engine with the Compose v2 plugin.
2. Add the deployment user to the `docker` group (or otherwise grant it access to
   the Docker socket), and allow SSH public-key authentication.
3. Point your reverse proxy at `http://127.0.0.1:8501`. TLS should terminate at
   the reverse proxy.
4. If the repository/package is private, keep the workflow's `packages: write`
   permission enabled. The workflow logs the VPS in to GHCR for each release.

No repository checkout is needed on the VPS; release files are copied to
`~/football-predictor` automatically.

### GitHub `production` environment

Create an environment named `production`, add protection rules if desired, and
configure these **environment secrets**:

| Secret | Purpose |
| --- | --- |
| `VPS_HOST` | VPS hostname or IP address |
| `VPS_USER` | SSH deployment user |
| `VPS_SSH_PRIVATE_KEY` | Private SSH key (the multiline OpenSSH value) |
| `VPS_SSH_KNOWN_HOSTS` | Pinned host-key line from a separately verified `ssh-keyscan` |
| `VPS_PORT` | SSH port; optional, defaults to `22` |
| `FOOTBALL_DATA_API_KEY` | football-data.org key |
| `NEWS_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY` | News/AI provider keys |
| `API_FOOTBALL_KEY` | Optional API-Football key |
| `SMARTPROXY_HOST`, `SMARTPROXY_PORT`, `SMARTPROXY_USER`, `SMARTPROXY_PASS` | Optional proxy settings |

The API values are written only to the VPS-side `.env.app` file (mode `0600`),
not baked into the container image. Also configure optional environment variables
`APP_BIND_ADDRESS` (default `127.0.0.1`) and `APP_PORT` (default `8501`). Finally,
ensure repository Actions settings allow workflows read/write access so the image
can be pushed to GHCR, then push to `main` or start **Build and deploy** manually.

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
