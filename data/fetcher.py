"""
football-data.org API - historikus meccsadatok letöltése és feldolgozása.
v5: cache, párhuzamos letöltés, gyorsabb tanítás
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FOOTBALL_DATA_API_KEY, FOOTBALL_DATA_BASE_URL, SUPPORTED_COMPETITIONS
from data.csv_fetcher import fetch_csv_data
from data.api_football_fetcher import fetch_api_football_data

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
PROCESSED_CACHE = os.path.join(CACHE_DIR, "processed_df.parquet")
PROCESSED_HASH  = os.path.join(CACHE_DIR, "processed_hash.txt")

HEADERS = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}

ELO_START = 1500
ELO_K = 30


def get_matches(competition_code: str, season: int) -> list:
    url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{competition_code}/matches"
    params = {"season": season, "status": "FINISHED"}
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        print(f"  Hiba {competition_code} {season}: {resp.status_code}")
        return []
    return resp.json().get("matches", [])


def parse_match(m: dict):
    try:
        home = m["homeTeam"]["name"]
        away = m["awayTeam"]["name"]
        hg = m["score"]["fullTime"]["home"]
        ag = m["score"]["fullTime"]["away"]
        if hg is None or ag is None:
            return None
        result = "H" if hg > ag else ("A" if hg < ag else "D")
        return {
            "date": m["utcDate"][:10],
            "competition": m["competition"]["code"],
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "result": result,
        }
    except (KeyError, TypeError):
        return None


def _elo_goal_multiplier(goal_diff: int) -> float:
    """Gólkülönbség súlya az Elo frissítésénél (Diego Fórlan formula)."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return 1.75 + (gd - 3) * 0.25


def _exp_weighted_mean(arr) -> float:
    """Exponenciális súlyozású átlag - a legújabb meccsek nagyobb súlyt kapnak."""
    if len(arr) == 0:
        return 0.0
    weights = np.exp(np.linspace(0, 1, len(arr)))
    return float(np.average(arr, weights=weights))


def _get_h2h_stats(home_team: str, away_team: str, past_df: pd.DataFrame) -> dict:
    """H2H statisztikák az elmúlt 5 egymás elleni meccsből."""
    h2h = past_df[
        ((past_df["home_team"] == home_team) & (past_df["away_team"] == away_team)) |
        ((past_df["home_team"] == away_team) & (past_df["away_team"] == home_team))
    ].tail(5)

    if len(h2h) == 0:
        return {
            "h2h_home_wins": 0.33,
            "h2h_draws": 0.33,
            "h2h_away_wins": 0.33,
            "h2h_avg_goals": 2.5,
            "h2h_n": 0,
        }

    home_wins = draws = away_wins = 0
    total_goals = 0
    for _, row in h2h.iterrows():
        if row["home_team"] == home_team:
            if row["result"] == "H":
                home_wins += 1
            elif row["result"] == "D":
                draws += 1
            else:
                away_wins += 1
        else:
            # Fordított irány: vendégként nyert = home_team győzelem
            if row["result"] == "A":
                home_wins += 1
            elif row["result"] == "D":
                draws += 1
            else:
                away_wins += 1
        total_goals += row["home_goals"] + row["away_goals"]

    n = len(h2h)
    return {
        "h2h_home_wins": home_wins / n,
        "h2h_draws": draws / n,
        "h2h_away_wins": away_wins / n,
        "h2h_avg_goals": total_goals / n,
        "h2h_n": n,
    }


def _get_days_rest(team: str, past_df: pd.DataFrame, current_date: str) -> int:
    """Hány nap telt el a csapat utolsó meccse óta (max 21)."""
    past_home = past_df[past_df["home_team"] == team]["date"]
    past_away = past_df[past_df["away_team"] == team]["date"]
    all_dates = pd.concat([past_home, past_away]).sort_values()
    if len(all_dates) == 0:
        return 7  # default: heti pihentség
    last = pd.to_datetime(all_dates.iloc[-1])
    curr = pd.to_datetime(current_date)
    days = int((curr - last).days)
    return min(max(days, 0), 21)


def build_team_stats(df: pd.DataFrame) -> tuple:
    """
    Gördülő ablakos statisztikák - v5 (O(n) optimalizált):
    Az előző O(n²) verziót felváltja: futó csapat-história listák,
    nincs df.iloc[:idx] másolás → ~50-100x gyorsabb nagy adathalmazon.
    """
    from collections import defaultdict

    df = df.sort_values("date").reset_index(drop=True)
    WINDOW = 8
    OVERALL_WINDOW = 5

    elo = {}
    streaks = {}
    records = []

    # Futó história listák csapatonként — O(1) elérés, O(WINDOW) számítás
    # Minden bejegyzés: {"idx", "date", "home_goals", "away_goals", "result", "home_team", "away_team"}
    team_home_hist = defaultdict(list)   # mikor a csapat otthon játszott
    team_away_hist = defaultdict(list)   # mikor a csapat vendégként játszott
    h2h_hist       = defaultdict(list)   # (sorted pair) → egymás elleni meccsek

    def _roll(hist_list, gf_key, ga_key, win_result, window, default_gf, default_ga, default_form):
        recent = hist_list[-window:]
        n = len(recent)
        if n == 0:
            return default_gf, default_ga, default_form, 0
        weights = np.exp(np.linspace(0, 1, n))
        gf   = float(np.average([r[gf_key] for r in recent], weights=weights))
        ga   = float(np.average([r[ga_key] for r in recent], weights=weights))
        form = float(np.average([1.0 if r["result"] == win_result else 0.0
                                  for r in recent], weights=weights))
        return gf, ga, form, n

    def _overall(home_hist, away_hist, h_win, a_win, overall_window):
        combined = []
        for r in home_hist:
            combined.append({
                "idx": r["idx"], "gf": r["home_goals"], "ga": r["away_goals"],
                "won": 1.0 if r["result"] == h_win else 0.0,
                "pts": 3 if r["result"] == h_win else (1 if r["result"] == "D" else 0),
            })
        for r in away_hist:
            combined.append({
                "idx": r["idx"], "gf": r["away_goals"], "ga": r["home_goals"],
                "won": 1.0 if r["result"] == a_win else 0.0,
                "pts": 3 if r["result"] == a_win else (1 if r["result"] == "D" else 0),
            })
        combined.sort(key=lambda x: x["idx"])
        recent = combined[-overall_window:]
        if not recent:
            return 0.33, 1.2, 1.2, 1.0
        n = len(recent)
        w = np.exp(np.linspace(0, 1, n))
        return (
            float(np.average([r["won"] for r in recent], weights=w)),
            float(np.average([r["gf"]  for r in recent], weights=w)),
            float(np.average([r["ga"]  for r in recent], weights=w)),
            float(np.average([r["pts"] for r in recent], weights=w)),
        )

    def _h2h(h_team, a_team, pair_key):
        recent = h2h_hist[pair_key][-5:]
        if not recent:
            return {"h2h_home_wins": 0.33, "h2h_draws": 0.33, "h2h_away_wins": 0.33,
                    "h2h_avg_goals": 2.5, "h2h_n": 0}
        hw = dr = aw = tg = 0
        for m in recent:
            tg += m["home_goals"] + m["away_goals"]
            if m["home_team"] == h_team:
                if m["result"] == "H": hw += 1
                elif m["result"] == "D": dr += 1
                else: aw += 1
            else:
                if m["result"] == "A": hw += 1
                elif m["result"] == "D": dr += 1
                else: aw += 1
        n = len(recent)
        return {"h2h_home_wins": hw/n, "h2h_draws": dr/n, "h2h_away_wins": aw/n,
                "h2h_avg_goals": tg/n, "h2h_n": n}

    def _days_rest(home_hist, away_hist, current_date):
        all_dates = [r["date"] for r in home_hist] + [r["date"] for r in away_hist]
        if not all_dates:
            return 7
        last = max(all_dates)
        try:
            days = int((datetime.strptime(current_date, "%Y-%m-%d")
                        - datetime.strptime(last, "%Y-%m-%d")).days)
        except ValueError:
            return 7
        return min(max(days, 0), 21)

    def _get_streak(team):
        stype, slen = streaks[team]
        if stype == "W": return slen
        if stype == "L": return -slen
        return 0

    for idx, row in df.iterrows():
        h, a   = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]
        date_str = row["date"]

        elo.setdefault(h, ELO_START)
        elo.setdefault(a, ELO_START)
        streaks.setdefault(h, ("", 0))
        streaks.setdefault(a, ("", 0))

        pair_key = tuple(sorted([h, a]))

        # ── Hazai csapat statisztikái ─────────────────────────────────────
        h_home_gf, h_home_ga, h_home_form, h_home_n = _roll(
            team_home_hist[h], "home_goals", "away_goals", "H",
            WINDOW, 1.3, 1.3, 0.33)
        h_away_gf, h_away_ga, h_away_form, h_away_n = _roll(
            team_away_hist[h], "away_goals", "home_goals", "A",
            WINDOW, 1.1, 1.4, 0.25)
        h_ov_form, h_ov_gf, h_ov_ga, h_momentum = _overall(
            team_home_hist[h], team_away_hist[h], "H", "A", OVERALL_WINDOW)

        # ── Vendég csapat statisztikái ────────────────────────────────────
        a_home_gf, a_home_ga, a_home_form, a_home_n = _roll(
            team_home_hist[a], "home_goals", "away_goals", "H",
            WINDOW, 1.3, 1.3, 0.33)
        a_away_gf, a_away_ga, a_away_form, a_away_n = _roll(
            team_away_hist[a], "away_goals", "home_goals", "A",
            WINDOW, 1.1, 1.4, 0.25)
        a_ov_form, a_ov_gf, a_ov_ga, a_momentum = _overall(
            team_home_hist[a], team_away_hist[a], "H", "A", OVERALL_WINDOW)

        stats = {
            "home_home_gf": h_home_gf, "home_home_ga": h_home_ga,
            "home_home_form": h_home_form, "home_home_n": h_home_n,
            "home_away_gf": h_away_gf, "home_away_ga": h_away_ga,
            "home_away_form": h_away_form, "home_away_n": h_away_n,
            "home_overall_form": h_ov_form, "home_overall_gf": h_ov_gf,
            "home_overall_ga": h_ov_ga, "home_momentum": h_momentum,
            "away_home_gf": a_home_gf, "away_home_ga": a_home_ga,
            "away_home_form": a_home_form, "away_home_n": a_home_n,
            "away_away_gf": a_away_gf, "away_away_ga": a_away_ga,
            "away_away_form": a_away_form, "away_away_n": a_away_n,
            "away_overall_form": a_ov_form, "away_overall_gf": a_ov_gf,
            "away_overall_ga": a_ov_ga, "away_momentum": a_momentum,
            "home_elo": elo[h], "away_elo": elo[a], "elo_diff": elo[h] - elo[a],
            "home_streak": _get_streak(h), "away_streak": _get_streak(a),
        }
        stats.update(_h2h(h, a, pair_key))
        stats["home_days_rest"] = _days_rest(team_home_hist[h], team_away_hist[h], date_str)
        stats["away_days_rest"] = _days_rest(team_home_hist[a], team_away_hist[a], date_str)
        stats["result"]      = row["result"]
        stats["date"]        = date_str
        stats["home_team"]   = h
        stats["away_team"]   = a
        stats["competition"] = row["competition"]
        records.append(stats)

        # ── História frissítése (a jelenlegi meccs hozzáadása) ────────────
        match_rec = {
            "idx": idx, "date": date_str,
            "home_team": h, "away_team": a,
            "home_goals": hg, "away_goals": ag,
            "result": row["result"],
        }
        team_home_hist[h].append(match_rec)
        team_away_hist[a].append(match_rec)
        h2h_hist[pair_key].append(match_rec)

        # ── Elo frissítés ─────────────────────────────────────────────────
        exp_h     = 1 / (1 + 10 ** ((elo[a] - elo[h]) / 400))
        goal_diff = hg - ag
        mult      = _elo_goal_multiplier(goal_diff)
        s_h, s_a  = (1.0, 0.0) if row["result"] == "H" else \
                    (0.0, 1.0) if row["result"] == "A" else (0.5, 0.5)
        elo[h] += ELO_K * mult * (s_h - exp_h)
        elo[a] += ELO_K * mult * (s_a - (1 - exp_h))

        # ── Streak frissítés ──────────────────────────────────────────────
        for team, won, drew in [
            (h, row["result"] == "H", row["result"] == "D"),
            (a, row["result"] == "A", row["result"] == "D"),
        ]:
            stype, slen = streaks[team]
            if won:
                streaks[team] = ("W", slen + 1) if stype == "W" else ("W", 1)
            elif drew:
                streaks[team] = ("D", slen + 1) if stype == "D" else ("D", 1)
            else:
                streaks[team] = ("L", slen + 1) if stype == "L" else ("L", 1)

    return pd.DataFrame(records), elo, streaks


def _raw_df_hash(df: pd.DataFrame) -> str:
    """Egyedi hash a raw_df-hez — ha változott az adat, újraszámol."""
    key = df[["date", "home_team", "away_team", "home_goals", "away_goals"]].to_csv(index=False)
    return hashlib.md5(key.encode()).hexdigest()


def _load_processed_cache(raw_hash: str):
    """Ha a cache friss és a hash egyezik, visszaadja a processed_df-et."""
    if not os.path.exists(PROCESSED_CACHE) or not os.path.exists(PROCESSED_HASH):
        return None, None, None
    with open(PROCESSED_HASH) as f:
        cached_hash, elo_json, streak_json = f.read().split("\n", 2)
    if cached_hash != raw_hash:
        return None, None, None
    import json
    df = pd.read_parquet(PROCESSED_CACHE)
    elo_dict = json.loads(elo_json)
    streak_dict = {k: tuple(v) for k, v in json.loads(streak_json).items()}
    print("  [cache] Processed adat betöltve — build_team_stats kihagyva!")
    return df, elo_dict, streak_dict


def _save_processed_cache(raw_hash: str, processed_df: pd.DataFrame, elo_dict: dict, streak_dict: dict):
    import json
    os.makedirs(CACHE_DIR, exist_ok=True)
    processed_df.to_parquet(PROCESSED_CACHE, index=False)
    elo_json = json.dumps(elo_dict)
    streak_json = json.dumps({k: list(v) for k, v in streak_dict.items()})
    with open(PROCESSED_HASH, "w") as f:
        f.write(f"{raw_hash}\n{elo_json}\n{streak_json}")
    print("  Processed adat cache-elve.")


def fetch_training_data(seasons: list = None) -> tuple:
    """Visszatér: (raw_df, processed_df, elo_dict, streak_dict)"""
    if seasons is None:
        current_year = datetime.now().year
        seasons = [current_year - i for i in range(6, 0, -1)]

    all_matches = []

    # ── Párhuzamos football-data.org letöltés ────────────────────────────────
    tasks = [
        (code, season)
        for code in SUPPORTED_COMPETITIONS
        for season in seasons
    ]

    def _fetch_one(args):
        code, season = args
        matches = get_matches(code, season)
        parsed = [parse_match(m) for m in matches if parse_match(m)]
        if parsed:
            print(f"  {SUPPORTED_COMPETITIONS[code]} {season}: {len(parsed)} meccs")
        return parsed

    print(f"  football-data.org: {len(tasks)} liga/szezon párhuzamos letöltés...")
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tasks}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                all_matches.extend(result)

    # ── CSV ligák (football-data.co.uk) ──────────────────────────────────────
    print("  CSV ligák letöltése (football-data.co.uk)...")
    csv_matches = fetch_csv_data(seasons=seasons)
    all_matches.extend(csv_matches)
    print(f"  CSV: {len(csv_matches)} meccs")

    # ── API-Football ─────────────────────────────────────────────────────────
    try:
        from config import API_FOOTBALL_KEY
        if API_FOOTBALL_KEY:
            print("  API-Football ligák letöltése (cache-sel)...")
            api_fb_matches = fetch_api_football_data(
                api_key=API_FOOTBALL_KEY,
                seasons=seasons,
            )
            all_matches.extend(api_fb_matches)
            print(f"  API-Football: {len(api_fb_matches)} meccs")
        else:
            print("  API-Football: nincs kulcs (kihagyva)")
    except ImportError:
        pass

    if not all_matches:
        print("Nem sikerült adatot letölteni!")
        return pd.DataFrame(), pd.DataFrame(), {}, {}

    raw_df = pd.DataFrame(all_matches)
    raw_df = raw_df.drop_duplicates(
        subset=["date", "home_team", "away_team"]
    ).sort_values("date").reset_index(drop=True)
    print(f"  Összesen {len(raw_df)} meccs ({len(set(raw_df['competition']))} liga).")

    # ── Cache check ───────────────────────────────────────────────────────────
    raw_hash = _raw_df_hash(raw_df)
    cached = _load_processed_cache(raw_hash)
    if cached[0] is not None:
        return raw_df, cached[0], cached[1], cached[2]

    # ── Friss build_team_stats ────────────────────────────────────────────────
    print("  Feature engineering (build_team_stats)...")
    processed_df, elo_dict, streak_dict = build_team_stats(raw_df)
    _save_processed_cache(raw_hash, processed_df, elo_dict, streak_dict)
    return raw_df, processed_df, elo_dict, streak_dict


def get_upcoming_matches() -> list:
    upcoming = []
    today = datetime.now().strftime("%Y-%m-%d")
    next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    for code in SUPPORTED_COMPETITIONS:
        url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{code}/matches"
        params = {"status": "SCHEDULED", "dateFrom": today, "dateTo": next_week}
        resp = requests.get(url, headers=HEADERS, params=params)
        if resp.status_code == 200:
            for m in resp.json().get("matches", []):
                upcoming.append({
                    "competition": code,
                    "competition_name": SUPPORTED_COMPETITIONS[code],
                    "date": m["utcDate"][:10],
                    "home_team": m["homeTeam"]["name"],
                    "away_team": m["awayTeam"]["name"],
                })
    return upcoming
