"""
Vegas.hu meccsek és oddsok lekérése az Altenar API-n keresztül.
Közvetlen API hívás - nem kell Playwright vagy proxy!
"""
import requests

ALTENAR_BASE = "https://hu-sb2frontend-altenar2.biahosted.com/api/widget"
PARAMS = "culture=hu-HU&timezoneOffset=-120&integration=vegas.hu"
FOOTBALL_SPORT_ID = 66


def _clean_name(name: str) -> str:
    """Whitespace, tabulátor és felesleges szóköz eltávolítása."""
    return " ".join(name.split())


def scrape_vegas(count: int = 200) -> list:
    """
    Lekéri a vegas.hu összes közelgő focimeccsét oddsokkal együtt.
    Közvetlen Altenar API hívás - gyors és megbízható.
    """
    url = f"{ALTENAR_BASE}/GetUpcoming?{PARAMS}&sportId={FOOTBALL_SPORT_ID}&count={count}"

    try:
        print(f"  Vegas.hu (Altenar API) lekérdezése...")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  API hiba: {e}")
        return []

    # Segédtáblák felépítése
    competitors = {c["id"]: c["name"] for c in data.get("competitors", [])}
    markets = {m["id"]: m for m in data.get("markets", [])}
    odds_map = {o["id"]: o.get("price") for o in data.get("odds", [])}

    matches = []
    for event in data.get("events", []):
        try:
            comp_ids = event.get("competitorIds", [])
            if len(comp_ids) < 2:
                continue

            home_team = _clean_name(competitors.get(comp_ids[0], "?"))
            away_team = _clean_name(competitors.get(comp_ids[1], "?"))
            date = event.get("startDate", "")[:10]

            # 1X2 piac keresése
            odds_H = odds_D = odds_A = None
            for mid in event.get("marketIds", []):
                m = markets.get(mid, {})
                if m.get("typeId") == 1 and m.get("name") == "1X2":
                    odd_ids = m.get("oddIds", [])
                    if len(odd_ids) >= 3:
                        odds_H = odds_map.get(odd_ids[0])
                        odds_D = odds_map.get(odd_ids[1])
                        odds_A = odds_map.get(odd_ids[2])
                    break

            if odds_H and odds_D and odds_A:
                matches.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "date": date,
                    "competition_name": "vegas.hu",
                    "odds_H": odds_H,
                    "odds_D": odds_D,
                    "odds_A": odds_A,
                    "source": "vegas.hu",
                })
        except Exception:
            continue

    print(f"  Vegas.hu: {len(matches)} meccs oddsokkal")
    return matches


if __name__ == "__main__":
    matches = scrape_vegas()
    for m in matches[:10]:
        print(f"{m['date']} | {m['home_team']} vs {m['away_team']} | {m['odds_H']} / {m['odds_D']} / {m['odds_A']}")
