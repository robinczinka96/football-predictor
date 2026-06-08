"""
football-data.co.uk ingyenes CSV adatok letöltése.
Regisztráció és API kulcs nélkül elérhető.
Lefedi: 2. Bundesliga, Belgian Pro League, Turkish Süper Lig,
        Scottish Premiership, Serie B, Ligue 2, La Liga Segunda
"""
import requests
import pandas as pd
import io
from datetime import datetime

# Elérhető ligák football-data.co.uk-n
# Formátum: "kod": ("Liganév", "verseny_kód_a_df-ben")
CSV_LEAGUES = {
    "D2":  ("2. Bundesliga",       "BL2"),
    "B1":  ("Belgian Pro League",  "BEL"),
    "T1":  ("Turkish Süper Lig",   "TSL"),
    "SC0": ("Scottish Premiership","SPL"),
    "I2":  ("Serie B",             "SB"),
    "F2":  ("Ligue 2",             "FL2"),
    "SP2": ("La Liga Segunda",     "SD"),
}

# Szezon URL formátum: 2024-25 → "2425"
def _season_code(year: int) -> str:
    return f"{str(year)[2:]}{str(year+1)[2:]}"


def _download_csv(league_code: str, season_year: int) -> pd.DataFrame:
    """Letölt egy CSV fájlt és visszaadja DataFrame-ként."""
    sc = _season_code(season_year)
    url = f"https://www.football-data.co.uk/mmz4281/{sc}/{league_code}.csv"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text), on_bad_lines="skip")
        return df
    except Exception:
        return pd.DataFrame()


def _parse_csv_matches(df: pd.DataFrame, competition_code: str, competition_name: str) -> list:
    """CSV DataFrame-et a saját formátumunkra konvertál."""
    needed = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Date"]
    if not all(c in df.columns for c in needed):
        return []

    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]).copy()
    df = df[df["FTR"].isin(["H", "D", "A"])]

    matches = []
    for _, row in df.iterrows():
        try:
            # Dátum parse: DD/MM/YY vagy DD/MM/YYYY
            raw_date = str(row["Date"]).strip()
            for fmt in ("%d/%m/%y", "%d/%m/%Y"):
                try:
                    date = datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            else:
                continue

            matches.append({
                "date":        date,
                "competition": competition_code,
                "home_team":   str(row["HomeTeam"]).strip(),
                "away_team":   str(row["AwayTeam"]).strip(),
                "home_goals":  int(row["FTHG"]),
                "away_goals":  int(row["FTAG"]),
                "result":      str(row["FTR"]).strip(),
            })
        except Exception:
            continue

    return matches


def fetch_csv_data(seasons: list = None) -> list:
    """
    Letölti az összes CSV liga adatait a megadott szezonokra.
    Visszatér: nyers meccs lista (ugyanolyan formátum mint fetch_training_data raw_df-je)
    """
    if seasons is None:
        current_year = datetime.now().year
        seasons = [current_year - i for i in range(4, 0, -1)]

    all_matches = []
    for league_code, (league_name, comp_code) in CSV_LEAGUES.items():
        for season in seasons:
            print(f"  CSV letöltés: {league_name} {season}/{season+1}...")
            df = _download_csv(league_code, season)
            if df.empty:
                print(f"    Nem elérhető: {league_code} {season}")
                continue
            matches = _parse_csv_matches(df, comp_code, league_name)
            print(f"    {len(matches)} meccs")
            all_matches.extend(matches)

    return all_matches
