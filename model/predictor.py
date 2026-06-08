"""
Előrejelzések és value bet kalkuláció - v4.
Újdonságok: H2H features, fáradtság, Dixon-Coles blend, odds blend.
"""
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VALUE_THRESHOLD, MIN_CONFIDENCE
from model.trainer import load_model, FEATURES

ELO_START = 1500
WINDOW = 8

# Vegas.hu → football-data.org névmapping
# (Vegas rövidített neveket használ, az API teljes neveket)
TEAM_NAME_MAP = {
    # Angol
    "Manchester City": "Manchester City FC",
    "Manchester United": "Manchester United FC",
    "Arsenal": "Arsenal FC",
    "Chelsea": "Chelsea FC",
    "Liverpool": "Liverpool FC",
    "Tottenham": "Tottenham Hotspur FC",
    "Tottenham Hotspur": "Tottenham Hotspur FC",
    "Newcastle": "Newcastle United FC",
    "Newcastle United": "Newcastle United FC",
    "Aston Villa": "Aston Villa FC",
    "Brighton": "Brighton & Hove Albion FC",
    "West Ham": "West Ham United FC",
    "West Ham United": "West Ham United FC",
    "Fulham": "Fulham FC",
    "Wolves": "Wolverhampton Wanderers FC",
    "Wolverhampton": "Wolverhampton Wanderers FC",
    "Brentford": "Brentford FC",
    "Everton": "Everton FC",
    "Crystal Palace": "Crystal Palace FC",
    "Nottingham Forest": "Nottingham Forest FC",
    "Bournemouth": "AFC Bournemouth",
    "Southampton": "Southampton FC",
    "Sheffield United": "Sheffield United FC",
    "Luton": "Luton Town FC",
    "Burnley": "Burnley FC",
    "Leicester": "Leicester City FC",
    "Leicester City": "Leicester City FC",
    "Ipswich": "Ipswich Town FC",
    "Ipswich Town": "Ipswich Town FC",
    # Német
    "Bayern München": "FC Bayern München",
    "Bayern Munich": "FC Bayern München",
    "Borussia Dortmund": "Borussia Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Leverkusen": "Bayer 04 Leverkusen",
    "Bayer Leverkusen": "Bayer 04 Leverkusen",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Wolfsburg": "VfL Wolfsburg",
    "Freiburg": "SC Freiburg",
    "Hoffenheim": "TSG 1899 Hoffenheim",
    "TSG Hoffenheim": "TSG 1899 Hoffenheim",
    "Stuttgart": "VfB Stuttgart",
    "Werder Bremen": "SV Werder Bremen",
    "Augsburg": "FC Augsburg",
    "Mainz": "1. FSV Mainz 05",
    "Union Berlin": "1. FC Union Berlin",
    "Köln": "1. FC Köln",
    "Darmstadt": "SV Darmstadt 98",
    "Heidenheim": "1. FC Heidenheim 1846",
    "1. FC Heidenheim 1846": "1. FC Heidenheim 1846",
    "Bochum": "VfL Bochum",
    "Gladbach": "Borussia Mönchengladbach",
    # Spanyol
    "Real Madrid": "Real Madrid CF",
    "Barcelona": "FC Barcelona",
    "Atletico Madrid": "Club Atlético de Madrid",
    "Atlético Madrid": "Club Atlético de Madrid",
    "Sevilla": "Sevilla FC",
    "Villarreal": "Villarreal CF",
    "Real Sociedad": "Real Sociedad de Fútbol",
    "Betis": "Real Betis Balompié",
    "Real Betis": "Real Betis Balompié",
    "Athletic Bilbao": "Athletic Club",
    "Athletic Club": "Athletic Club",
    "Valencia": "Valencia CF",
    "Osasuna": "CA Osasuna",
    "Celta Vigo": "RC Celta de Vigo",
    "Getafe": "Getafe CF",
    "Mallorca": "RCD Mallorca",
    "Almeria": "UD Almería",
    "Almería": "UD Almería",
    "Girona": "Girona FC",
    "Las Palmas": "UD Las Palmas",
    "Rayo Vallecano": "Rayo Vallecano de Madrid",
    "Alaves": "Deportivo Alavés",
    "Alavés": "Deportivo Alavés",
    "Espanol": "RCD Espanyol de Barcelona",
    "Espanyol": "RCD Espanyol de Barcelona",
    # Olasz
    "Inter Milan": "FC Internazionale Milano",
    "Inter": "FC Internazionale Milano",
    "AC Milan": "AC Milan",
    "Milan": "AC Milan",
    "Juventus": "Juventus FC",
    "Napoli": "SSC Napoli",
    "Roma": "AS Roma",
    "Lazio": "SS Lazio",
    "Atalanta": "Atalanta BC",
    "Fiorentina": "ACF Fiorentina",
    "Torino": "Torino FC",
    "Bologna": "Bologna FC 1909",
    "Udinese": "Udinese Calcio",
    "Monza": "AC Monza",
    "Lecce": "US Lecce",
    "Frosinone": "Frosinone Calcio",
    "Cagliari": "Cagliari Calcio",
    "Sassuolo": "US Sassuolo Calcio",
    "Empoli": "Empoli FC",
    "Hellas Verona": "Hellas Verona FC",
    "Genoa": "Genoa CFC",
    "Salernitana": "US Salernitana 1919",
    # Championship (angol 2. osztály)
    "Leeds": "Leeds United FC",
    "Leeds United": "Leeds United FC",
    "Sunderland": "Sunderland AFC",
    "Bristol City": "Bristol City FC",
    "Middlesbrough": "Middlesbrough FC",
    "Coventry": "Coventry City FC",
    "Coventry City": "Coventry City FC",
    "Norwich": "Norwich City FC",
    "Norwich City": "Norwich City FC",
    "Millwall": "Millwall FC",
    "Watford": "Watford FC",
    "Stoke": "Stoke City FC",
    "Stoke City": "Stoke City FC",
    "Hull": "Hull City AFC",
    "Hull City": "Hull City AFC",
    "Cardiff": "Cardiff City FC",
    "Cardiff City": "Cardiff City FC",
    "Derby": "Derby County FC",
    "Derby County": "Derby County FC",
    "Preston": "Preston North End FC",
    "Swansea": "Swansea City AFC",
    "QPR": "Queens Park Rangers FC",
    "Blackburn": "Blackburn Rovers FC",
    "West Brom": "West Bromwich Albion FC",
    "WBA": "West Bromwich Albion FC",
    "Plymouth": "Plymouth Argyle FC",
    "Oxford": "Oxford United FC",
    "Oxford United": "Oxford United FC",
    "Portsmouth": "Portsmouth FC",
    "Luton Town": "Luton Town FC",
    # Eredivisie (holland)
    "Ajax": "AFC Ajax",
    "PSV": "PSV",
    "Feyenoord": "Feyenoord Rotterdam",
    "AZ": "AZ Alkmaar",
    "Twente": "FC Twente",
    "Utrecht": "FC Utrecht",
    "Groningen": "FC Groningen",
    "Heerenveen": "SC Heerenveen",
    "Vitesse": "SBV Vitesse",
    "NEC Nijmegen": "NEC",
    "Sparta Rotterdam": "Sparta Rotterdam",
    "RKC Waalwijk": "RKC Waalwijk",
    "Go Ahead Eagles": "Go Ahead Eagles",
    "Almere": "Almere City FC",
    # Francia
    "PSG": "Paris Saint-Germain FC",
    "Paris Saint-Germain": "Paris Saint-Germain FC",
    "Marseille": "Olympique de Marseille",
    "Lyon": "Olympique Lyonnais",
    "Monaco": "AS Monaco FC",
    "Lens": "RC Lens",
    "Lille": "LOSC Lille",
    "Nice": "OGC Nice",
    "Rennes": "Stade Rennais FC",
    "Strasbourg": "RC Strasbourg Alsace",
    "Montpellier": "Montpellier HSC",
    "Nantes": "FC Nantes",
    "Toulouse": "Toulouse FC",
    "Reims": "Stade de Reims",
    "Le Havre": "Le Havre AC",
    "Metz": "FC Metz",
    "Lorient": "FC Lorient",
    "Clermont": "Clermont Foot 63",
}


def is_womens_team(name: str) -> bool:
    """Megvizsgálja, hogy a csapatnév női csapatra utal-e (Vegas.hu: N = Nők)."""
    return name.strip().endswith("(N)")


def normalize_team_name(name: str) -> str:
    """Vegas.hu csapatnevet konvertál football-data.org formátumra."""
    name = name.strip()
    return TEAM_NAME_MAP.get(name, name)

# Dixon-Coles és odds blend súlyok
DC_WEIGHT = 0.30      # 30% Dixon-Coles, 70% ML ensemble
MARKET_WEIGHT = 0.20  # 20% piaci odds, 80% modell


def _exp_weighted_mean(arr) -> float:
    """Exponenciális súlyozású átlag."""
    if len(arr) == 0:
        return 0.0
    weights = np.exp(np.linspace(0, 1, len(arr)))
    return float(np.average(arr, weights=weights))


def get_team_features(team: str, is_home_team: bool, raw_df: pd.DataFrame,
                      elo_dict: dict, streak_dict: dict = None) -> dict:
    prefix = "home" if is_home_team else "away"

    home_g = raw_df[raw_df["home_team"] == team].tail(WINDOW)
    away_g = raw_df[raw_df["away_team"] == team].tail(WINDOW)

    # Hazai teljesítmény (exp súlyozással)
    if len(home_g) > 0:
        h_gf = _exp_weighted_mean(home_g["home_goals"].values)
        h_ga = _exp_weighted_mean(home_g["away_goals"].values)
        h_form = _exp_weighted_mean(
            (home_g["result"] == "H").astype(float).values
        )
        h_n = len(home_g)
    else:
        h_gf, h_ga, h_form, h_n = 1.3, 1.3, 0.33, 0

    # Vendég teljesítmény (exp súlyozással)
    if len(away_g) > 0:
        a_gf = _exp_weighted_mean(away_g["away_goals"].values)
        a_ga = _exp_weighted_mean(away_g["home_goals"].values)
        a_form = _exp_weighted_mean(
            (away_g["result"] == "A").astype(float).values
        )
        a_n = len(away_g)
    else:
        a_gf, a_ga, a_form, a_n = 1.1, 1.4, 0.25, 0

    # Összesített forma + momentum (exp súlyozással)
    all_h = home_g[["date", "home_goals", "away_goals", "result"]].rename(
        columns={"home_goals": "gf", "away_goals": "ga"})
    all_h = all_h.assign(
        won=(home_g["result"].values == "H"),
        pts=home_g["result"].map({"H": 3, "D": 1, "A": 0}).values
    )
    all_a = away_g[["date", "home_goals", "away_goals", "result"]].rename(
        columns={"away_goals": "gf", "home_goals": "ga"})
    all_a = all_a.assign(
        won=(away_g["result"].values == "A"),
        pts=away_g["result"].map({"A": 3, "D": 1, "H": 0}).values
    )
    all_g = pd.concat([all_h, all_a]).sort_values("date").tail(5)

    if len(all_g) > 0:
        overall_form = _exp_weighted_mean(all_g["won"].astype(float).values)
        overall_gf = _exp_weighted_mean(all_g["gf"].values)
        overall_ga = _exp_weighted_mean(all_g["ga"].values)
        momentum = _exp_weighted_mean(all_g["pts"].values)
    else:
        overall_form, overall_gf, overall_ga, momentum = 0.33, 1.2, 1.2, 1.0

    elo = elo_dict.get(team, ELO_START)

    streak = 0
    if streak_dict and team in streak_dict:
        stype, slen = streak_dict[team]
        streak = slen if stype == "W" else (-slen if stype == "L" else 0)

    return {
        f"{prefix}_home_gf": h_gf, f"{prefix}_home_ga": h_ga,
        f"{prefix}_home_form": h_form, f"{prefix}_home_n": h_n,
        f"{prefix}_away_gf": a_gf, f"{prefix}_away_ga": a_ga,
        f"{prefix}_away_form": a_form, f"{prefix}_away_n": a_n,
        f"{prefix}_overall_form": overall_form,
        f"{prefix}_overall_gf": overall_gf, f"{prefix}_overall_ga": overall_ga,
        f"{prefix}_momentum": momentum,
        f"{prefix}_elo": elo,
        f"{prefix}_streak": streak,
    }


def get_h2h_features(home_team: str, away_team: str, raw_df: pd.DataFrame) -> dict:
    """H2H statisztikák az elmúlt 5 egymás elleni meccsből."""
    h2h = raw_df[
        ((raw_df["home_team"] == home_team) & (raw_df["away_team"] == away_team)) |
        ((raw_df["home_team"] == away_team) & (raw_df["away_team"] == home_team))
    ].tail(5)

    if len(h2h) == 0:
        return {
            "h2h_home_wins": 0.33, "h2h_draws": 0.33, "h2h_away_wins": 0.33,
            "h2h_avg_goals": 2.5, "h2h_n": 0,
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


def get_days_rest(team: str, raw_df: pd.DataFrame, match_date: str) -> int:
    """Napok száma a csapat utolsó meccse óta (0-21)."""
    past_home = raw_df[raw_df["home_team"] == team]["date"]
    past_away = raw_df[raw_df["away_team"] == team]["date"]
    all_dates = pd.concat([past_home, past_away]).sort_values()
    if len(all_dates) == 0:
        return 7
    last = pd.to_datetime(all_dates.iloc[-1])
    curr = pd.to_datetime(match_date) if match_date else pd.Timestamp.now()
    days = int((curr - last).days)
    return min(max(days, 0), 21)


def _blend_with_dc(p_H: float, p_D: float, p_A: float,
                   home_team: str, away_team: str) -> tuple:
    """Dixon-Coles modellel való keverés."""
    try:
        from model.dixon_coles import load_dc_model
        dc_model = load_dc_model()
        if dc_model is None:
            return p_H, p_D, p_A

        dc = dc_model.predict_proba(home_team, away_team)
        w = DC_WEIGHT
        new_H = (1 - w) * p_H + w * dc[0]
        new_D = (1 - w) * p_D + w * dc[1]
        new_A = (1 - w) * p_A + w * dc[2]
        return new_H, new_D, new_A
    except Exception:
        return p_H, p_D, p_A


def _blend_with_market(p_H: float, p_D: float, p_A: float,
                       odds_H: float, odds_D: float, odds_A: float) -> tuple:
    """Piaci oddsokkal való keverés (vig nélkül)."""
    try:
        if not (odds_H and odds_D and odds_A and
                odds_H > 1 and odds_D > 1 and odds_A > 1):
            return p_H, p_D, p_A

        raw_H = 1 / odds_H
        raw_D = 1 / odds_D
        raw_A = 1 / odds_A
        total = raw_H + raw_D + raw_A  # vig eltávolítása
        imp_H = raw_H / total
        imp_D = raw_D / total
        imp_A = raw_A / total

        w = MARKET_WEIGHT
        new_H = (1 - w) * p_H + w * imp_H
        new_D = (1 - w) * p_D + w * imp_D
        new_A = (1 - w) * p_A + w * imp_A
        return new_H, new_D, new_A
    except Exception:
        return p_H, p_D, p_A


def _has_enough_data(team: str, raw_df: pd.DataFrame, min_matches: int = 3) -> bool:
    """Ellenőrzi, hogy van-e elegendő historikus adatunk a csapatról."""
    home_count = (raw_df["home_team"] == team).sum()
    away_count = (raw_df["away_team"] == team).sum()
    return (home_count + away_count) >= min_matches


def predict_match(home_team: str, away_team: str, raw_df: pd.DataFrame,
                  elo_dict: dict, streak_dict: dict = None,
                  match_date: str = None,
                  odds_H: float = None, odds_D: float = None,
                  odds_A: float = None):
    model, le = load_model()
    if model is None:
        return None

    # Mindkét csapatról kell elegendő adat, különben a predikció félrevezető
    # (pl. kiesett PL csapat Championship meccsein torzított statisztikák)
    home_known = _has_enough_data(home_team, raw_df)
    away_known = _has_enough_data(away_team, raw_df)
    if not home_known or not away_known:
        return None  # Ha bármelyik csapat ismeretlen → kihagyjuk

    home_feats = get_team_features(home_team, True, raw_df, elo_dict, streak_dict)
    away_feats = get_team_features(away_team, False, raw_df, elo_dict, streak_dict)
    h2h_feats = get_h2h_features(home_team, away_team, raw_df)

    date_str = match_date or datetime.now().strftime("%Y-%m-%d")
    home_rest = get_days_rest(home_team, raw_df, date_str)
    away_rest = get_days_rest(away_team, raw_df, date_str)

    features = {**home_feats, **away_feats}
    features["elo_diff"] = features["home_elo"] - features["away_elo"]
    features.update(h2h_feats)
    features["home_days_rest"] = home_rest
    features["away_days_rest"] = away_rest

    X = np.array([[features[f] for f in FEATURES]])
    probs = model.predict_proba(X)[0]
    classes = le.classes_
    prob_dict = {cls: prob for cls, prob in zip(classes, probs)}

    p_H = prob_dict.get("H", 0.0)
    p_D = prob_dict.get("D", 0.0)
    p_A = prob_dict.get("A", 0.0)

    # Dixon-Coles blend
    p_H, p_D, p_A = _blend_with_dc(p_H, p_D, p_A, home_team, away_team)

    # Piaci odds blend (ha elérhetők)
    if odds_H and odds_D and odds_A:
        p_H, p_D, p_A = _blend_with_market(p_H, p_D, p_A, odds_H, odds_D, odds_A)

    probs_final = {"H": p_H, "D": p_D, "A": p_A}
    return {
        "home_team": home_team,
        "away_team": away_team,
        "prob_H": p_H,
        "prob_D": p_D,
        "prob_A": p_A,
        "predicted": max(probs_final, key=probs_final.get),
        "confidence": max(p_H, p_D, p_A),
    }


def calculate_value(prediction: dict, odds_H: float, odds_D: float,
                    odds_A: float) -> dict:
    implied_H = 1 / odds_H if odds_H and odds_H > 1 else None
    implied_D = 1 / odds_D if odds_D and odds_D > 1 else None
    implied_A = 1 / odds_A if odds_A and odds_A > 1 else None

    value_H = prediction["prob_H"] - implied_H if implied_H else None
    value_D = prediction["prob_D"] - implied_D if implied_D else None
    value_A = prediction["prob_A"] - implied_A if implied_A else None

    best_value = max(
        [(v, k) for v, k in
         [(value_H, "H"), (value_D, "D"), (value_A, "A")] if v is not None],
        key=lambda x: x[0],
        default=(None, None),
    )

    return {
        **prediction,
        "odds_H": odds_H, "odds_D": odds_D, "odds_A": odds_A,
        "value_H": value_H, "value_D": value_D, "value_A": value_A,
        "best_value": best_value[0],
        "best_bet": best_value[1],
        "is_value_bet": best_value[0] is not None and best_value[0] >= VALUE_THRESHOLD,
    }


def predict_all_matches(matches: list, raw_df: pd.DataFrame,
                        elo_dict: dict = None, streak_dict: dict = None,
                        min_confidence: float = 0.40) -> pd.DataFrame:
    if elo_dict is None:
        elo_dict = {}
    if streak_dict is None:
        streak_dict = {}

    results = []
    for m in matches:
        odds_H = m.get("odds_H")
        odds_D = m.get("odds_D")
        odds_A = m.get("odds_A")
        match_date = m.get("date")

        # Női meccsek kiszűrése (Vegas.hu: "(N)" = Nők)
        if is_womens_team(m["home_team"]) or is_womens_team(m["away_team"]):
            continue

        # Névnormalizálás: Vegas rövidített nevek → football-data.org nevek
        home_team = normalize_team_name(m["home_team"])
        away_team = normalize_team_name(m["away_team"])

        pred = predict_match(
            home_team, away_team,
            raw_df, elo_dict, streak_dict,
            match_date=match_date,
            odds_H=odds_H, odds_D=odds_D, odds_A=odds_A,
        )
        if pred is None:
            continue

        if odds_H and odds_D and odds_A:
            pred = calculate_value(pred, odds_H, odds_D, odds_A)

        pred["date"] = match_date or ""
        pred["competition"] = m.get("competition_name", m.get("competition", ""))
        # Megőrizzük az eredeti Vegas nevet a megjelenítéshez
        pred["home_team"] = m["home_team"]
        pred["away_team"] = m["away_team"]
        results.append(pred)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df[df["confidence"] >= min_confidence].copy()
    return df.sort_values("confidence", ascending=False)
