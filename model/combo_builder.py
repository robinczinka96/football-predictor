"""
Kombinált szelvény optimalizáló.
Megkeresi az optimális meccs-kombinációkat EV + Kelly-kritérium alapján.
"""
import numpy as np
import pandas as pd
from itertools import combinations


# ── Kockázati kategóriák ────────────────────────────────────────────────────
RISK_TIERS = {
    "safe":   {"label": "🟢 Biztos",   "min_prob": 0.40, "max_prob": 1.00, "size_range": (2, 3)},
    "medium": {"label": "🟡 Közepes",  "min_prob": 0.22, "max_prob": 0.40, "size_range": (3, 4)},
    "bold":   {"label": "🔴 Merész",   "min_prob": 0.10, "max_prob": 0.22, "size_range": (4, 5)},
}


def _get_bet_odds(row: pd.Series) -> float:
    """A tippelt kimenetelhez tartozó odds."""
    outcome = row.get("best_outcome") or row.get("predicted", "H")
    mapping = {"H": "odds_H", "D": "odds_D", "A": "odds_A"}
    key = mapping.get(outcome, "odds_H")
    val = row.get(key)
    return float(val) if val and val > 1 else 1.0


def _get_bet_prob(row: pd.Series) -> float:
    """A tippelt kimenetel valószínűsége."""
    outcome = row.get("best_outcome") or row.get("predicted", "H")
    mapping = {"H": "prob_H", "D": "prob_D", "A": "prob_A"}
    key = mapping.get(outcome, "prob_H")
    val = row.get(key)
    return float(val) if val else float(row.get("confidence", 0.5))


def _kelly_fraction(prob: float, decimal_odds: float) -> float:
    """
    Kelly-kritérium: a tőke mekkora hányadát érdemes feltennni.
    1/4 Kelly-t használunk a biztonság kedvéért.
    """
    b = decimal_odds - 1.0
    if b <= 0 or prob <= 0:
        return 0.0
    q = 1.0 - prob
    f = (prob * b - q) / b
    f = max(0.0, f)
    return round(f * 0.25, 4)  # 1/4 Kelly


def _score_combo(combined_prob: float, combined_odds: float, ev: float) -> float:
    """
    Kombináció értékelése: EV × kombinált_valószínűség^0.4
    Bünteti az alacsony valószínűségű kombinációkat.
    """
    if combined_prob <= 0 or combined_odds <= 1:
        return -1
    return ev * (combined_prob ** 0.4)


def build_combos(
    predictions_df: pd.DataFrame,
    max_combo_size: int = 5,
    top_n_input: int = 15,
    min_individual_conf: float = 0.52,
    top_per_tier: int = 3,
) -> dict:
    """
    Optimális fogadási kombinációkat generál.

    Visszatér: dict {tier_name: [combo_dict, ...]}
    Minden combo_dict tartalmazza:
      - matches: list of rows
      - combined_prob: float
      - combined_odds: float
      - ev: float
      - kelly: float (1/4 Kelly, tőkearány)
      - score: float
    """
    df = predictions_df.copy()

    # ── Jelöltek szűrése ────────────────────────────────────────────────────
    # Csak ahol van odds és elég magas az egyedi konfidencia
    has_odds = (
        df.get("odds_H", pd.Series(dtype=float)).notna() &
        df.get("odds_D", pd.Series(dtype=float)).notna() &
        df.get("odds_A", pd.Series(dtype=float)).notna()
    ) if "odds_H" in df.columns else pd.Series([True] * len(df))

    candidates = df[
        has_odds &
        (df["confidence"] >= min_individual_conf)
    ].head(top_n_input)

    if len(candidates) < 2:
        return {}

    # Előre kiszámítjuk a tipp oddsot és valószínűséget
    rows = []
    for _, row in candidates.iterrows():
        bet_odds = _get_bet_odds(row)
        bet_prob = _get_bet_prob(row)
        ev_single = bet_prob * bet_odds
        if bet_odds > 1.01:  # irreálisan alacsony oddsok kizárása
            rows.append({
                "row": row,
                "bet_odds": bet_odds,
                "bet_prob": bet_prob,
                "ev_single": ev_single,
            })

    if len(rows) < 2:
        return {}

    # ── Kombináció generálás ─────────────────────────────────────────────────
    all_combos = []
    for size in range(2, min(max_combo_size + 1, len(rows) + 1)):
        for combo in combinations(rows, size):
            combined_prob = float(np.prod([c["bet_prob"] for c in combo]))
            combined_odds = float(np.prod([c["bet_odds"] for c in combo]))
            ev = combined_prob * combined_odds
            kelly = _kelly_fraction(combined_prob, combined_odds)
            score = _score_combo(combined_prob, combined_odds, ev)

            all_combos.append({
                "size": size,
                "matches": [c["row"] for c in combo],
                "combined_prob": combined_prob,
                "combined_odds": combined_odds,
                "ev": ev,
                "kelly": kelly,
                "score": score,
            })

    # ── Szortírozás és kategorizálás ─────────────────────────────────────────
    all_combos.sort(key=lambda x: x["score"], reverse=True)

    result = {}
    for tier_key, tier_cfg in RISK_TIERS.items():
        min_p = tier_cfg["min_prob"]
        max_p = tier_cfg["max_prob"]
        min_sz, max_sz = tier_cfg["size_range"]

        tier_combos = [
            c for c in all_combos
            if min_p <= c["combined_prob"] < max_p
            and min_sz <= c["size"] <= max_sz
        ]

        # Deduplikáció: ne szerepeljen ugyanaz a meccs kétszer az első N-ben
        seen_matches = set()
        deduped = []
        for c in tier_combos:
            match_set = frozenset(
                f"{r.get('home_team')} vs {r.get('away_team')}"
                for r in c["matches"]
            )
            if match_set not in seen_matches:
                seen_matches.add(match_set)
                deduped.append(c)
            if len(deduped) >= top_per_tier:
                break

        if deduped:
            result[tier_key] = {"label": tier_cfg["label"], "combos": deduped}

    return result


def format_combo_summary(combo: dict) -> str:
    """Szelvény rövid szöveges összefoglalója."""
    lines = []
    for row in combo["matches"]:
        outcome = row.get("best_outcome") or row.get("predicted", "?")
        label = {"H": f"🏠 {row.get('home_team','?')}",
                 "D": "🤝 Döntetlen",
                 "A": f"✈️ {row.get('away_team','?')}"}.get(outcome, outcome)
        odds = _get_bet_odds(row)
        prob = _get_bet_prob(row)
        lines.append(f"• {row.get('home_team','?')} vs {row.get('away_team','?')} → **{label}** ({prob:.0%}, @{odds:.2f})")
    return "\n".join(lines)
