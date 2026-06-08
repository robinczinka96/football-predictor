"""
api-football.com integrálás - Vegas.hu lefedettség bővítése.
Ingyenes tier: 100 kérés/nap.
Caching: adatokat JSON fájlokban tároljuk, hogy minimalizáljuk az API hívásokat.
"""
import requests
import json
import os
import time
from datetime import datetime, timedelta

# Alap könyvtár a cache-hez
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "api_football")

# API-Football v3
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

# Liga ID-k Vegas.hu-n jellemzően megtalálható, de adatbázisunkból hiányzó ligákhoz
# Format: liga_id: ("Liga neve", "kompetíció kód a df-ben")
API_FOOTBALL_LEAGUES = {
    128:  ("Argentine Primera División", "ARG1"),
    253:  ("MLS",                         "MLS"),
    72:   ("Brazilian Série B",           "BSB"),
    197:  ("Greek Super League",          "GRE1"),
    106:  ("Polish Ekstraklasa",          "POL1"),
    345:  ("Czech Fortuna Liga",          "CZE1"),
    207:  ("Swiss Super League",          "SUI1"),
    333:  ("Ukrainian Premier League",    "UKR1"),
    98:   ("J-League",                    "JPN1"),
    292:  ("K-League 1",                  "KOR1"),
    169:  ("Chinese Super League",        "CHN1"),
    188:  ("Australian A-League",         "AUS1"),
    283:  ("Romanian Liga 1",             "ROM1"),
    113:  ("Norwegian Eliteserien",       "NOR1"),
    103:  ("Finnish Veikkausliiga",       "FIN1"),
    244:  ("Danish Superliga",            "DEN1"),
    308:  ("Saudi Pro League",            "KSA1"),
    203:  ("Turkish 1. Lig",              "TL2"),
    40:   ("Championship",               "ELC2"),   # fallback ha football-data.org limitál
    88:   ("Eredivisie",                  "DED2"),   # fallback
}

# Hány napon belül tekintsük a cache-t frissnek
CACHE_FRESHNESS_DAYS = 7


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(league_id: int, season: int) -> str:
    return os.path.join(CACHE_DIR, f"{league_id}_{season}.json")


def _is_cache_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return datetime.now() - mtime < timedelta(days=CACHE_FRESHNESS_DAYS)


def _load_cache(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(path: str, data: list):
    _ensure_cache_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _fetch_fixtures(league_id: int, season: int, api_key: str) -> list:
    """Letölti egy liga egy szezonjának befejezett meccseit."""
    url = f"{API_FOOTBALL_BASE_URL}/fixtures"
    params = {"league": league_id, "season": season, "status": "FT"}
    headers = {"x-apisports-key": api_key}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            print(f"    Rate limit elérve (429) - {league_id} {season}")
            return None  # None = rate limit, üres list = nincs adat
        if resp.status_code != 200:
            print(f"    Hiba {league_id} {season}: {resp.status_code}")
            return []
        data = resp.json()
        remaining = resp.headers.get("x-ratelimit-requests-remaining", "?")
        print(f"    Maradék API kérés: {remaining}")
        return data.get("response", [])
    except Exception as e:
        print(f"    Kivétel {league_id} {season}: {e}")
        return []


def _parse_fixture(fixture: dict, competition_code: str):
    try:
        home = fixture["teams"]["home"]["name"]
        away = fixture["teams"]["away"]["name"]
        hg = fixture["goals"]["home"]
        ag = fixture["goals"]["away"]
        if hg is None or ag is None:
            return None
        result = "H" if hg > ag else ("A" if hg < ag else "D")
        date_str = fixture["fixture"]["date"][:10]
        return {
            "date":        date_str,
            "competition": competition_code,
            "home_team":   home,
            "away_team":   away,
            "home_goals":  int(hg),
            "away_goals":  int(ag),
            "result":      result,
        }
    except (KeyError, TypeError):
        return None


def fetch_api_football_data(api_key: str, seasons: list = None, league_ids: list = None) -> list:
    """
    Letölti az API-Football adatokat a megadott ligákra és szezonokra.
    Cache-t használ hogy minimalizálja az API hívásokat (100 req/nap limit).

    Args:
        api_key: api-football.com API kulcs
        seasons: Szezonok listája (pl. [2022, 2023, 2024])
        league_ids: Melyik ligákat töltse le (None = összes API_FOOTBALL_LEAGUES)

    Returns:
        Nyers meccs lista (ugyanolyan formátum mint fetch_training_data raw_df-je)
    """
    if not api_key:
        print("  API-Football: nincs API kulcs beállítva, kihagyva.")
        return []

    if seasons is None:
        current_year = datetime.now().year
        seasons = [current_year - i for i in range(4, 0, -1)]

    if league_ids is None:
        league_ids = list(API_FOOTBALL_LEAGUES.keys())

    _ensure_cache_dir()
    all_matches = []
    api_calls_made = 0

    for league_id in league_ids:
        league_name, comp_code = API_FOOTBALL_LEAGUES[league_id]

        for season in seasons:
            cache_file = _cache_path(league_id, season)

            if _is_cache_fresh(cache_file):
                cached = _load_cache(cache_file)
                if cached:
                    print(f"  [cache] {league_name} {season}: {len(cached)} meccs")
                    all_matches.extend(cached)
                continue

            # Cache nincs / elavult → API hívás
            print(f"  API-Football: {league_name} {season}...")
            fixtures = _fetch_fixtures(league_id, season, api_key)
            api_calls_made += 1

            if fixtures is None:
                # Rate limit → megállunk
                print(f"  Rate limit elérve! {api_calls_made} hívás után megállunk.")
                return all_matches

            parsed = []
            for fx in fixtures:
                m = _parse_fixture(fx, comp_code)
                if m:
                    parsed.append(m)

            _save_cache(cache_file, parsed)
            print(f"    {len(parsed)} meccs mentve cache-be")
            all_matches.extend(parsed)

            # Tiszteletből kis szünet az API felé
            time.sleep(0.3)

    print(f"  API-Football összesen: {len(all_matches)} meccs ({api_calls_made} friss API hívás)")
    return all_matches


def get_remaining_requests(api_key: str) -> int:
    """Lekéri hány API kérés maradt a napi limitből."""
    if not api_key:
        return 0
    try:
        resp = requests.get(
            f"{API_FOOTBALL_BASE_URL}/status",
            headers={"x-apisports-key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json().get("response", {})
            return data.get("requests", {}).get("current", 0)
    except Exception:
        pass
    return -1
