import os
from dotenv import load_dotenv

load_dotenv()

FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
SMARTPROXY_HOST = os.getenv("SMARTPROXY_HOST")
SMARTPROXY_PORT = os.getenv("SMARTPROXY_PORT")
SMARTPROXY_USER = os.getenv("SMARTPROXY_USER")
SMARTPROXY_PASS = os.getenv("SMARTPROXY_PASS")
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

# Ingyenes tier ligák
SUPPORTED_COMPETITIONS = {
    # Top 5 liga + Primeira Liga + Championship + Eredivisie
    "PL":  "Premier League",
    "PD":  "La Liga",
    "BL1": "Bundesliga",
    "SA":  "Serie A",
    "FL1": "Ligue 1",
    "PPL": "Primeira Liga",
    "ELC": "Championship",
    "DED": "Eredivisie",
    # Brasileirão (football-data.org ingyenes)
    "BSA": "Brasileirão Serie A",
    # Kupák (sokszor 429, de megpróbáljuk)
    "CL":  "Champions League",
    "EL":  "Europa League",
    "EC":  "Európa-bajnokság",
    "WC":  "Világbajnokság",
}

# Value bet küszöb: ha a mi valószínűségünk ennyivel magasabb az implied odds-nál
VALUE_THRESHOLD = 0.05  # 5%

# Minimális confidence a megjelenítéshez
MIN_CONFIDENCE = 0.40  # 40%
