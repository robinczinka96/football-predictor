"""
Hírek lekérése 3 forrásból:
1. NewsAPI - általános sporthírek
2. Transfermarkt - sérüléslista
3. BBC Sport - megbízható angol hírek
"""
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NEWS_API_KEY

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# ── 1. NewsAPI ─────────────────────────────────────────────────────────────────

def fetch_newsapi(team_name: str, days: int = 7) :
    """NewsAPI-ból keres híreket a csapatról."""
    if not NEWS_API_KEY:
        return []
    try:
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": f'"{team_name}" (injury OR injured OR suspension OR suspended OR sérült OR eltiltott)',
                "from": from_date,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": 5,
                "apiKey": NEWS_API_KEY,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        articles = resp.json().get("articles", [])
        return [
            {
                "source": "NewsAPI",
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "url": a.get("url", ""),
                "published": a.get("publishedAt", "")[:10],
            }
            for a in articles if a.get("title")
        ]
    except Exception:
        return []


# ── 2. Transfermarkt ───────────────────────────────────────────────────────────

def _search_transfermarkt(team_name: str):
    """Megkeresi a csapat Transfermarkt URL-jét."""
    try:
        resp = requests.get(
            "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche",
            params={"query": team_name, "Verein_page": 0},
            headers={**HEADERS, "Accept-Language": "en-US,en;q=0.9"},
            timeout=10,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        # Első csapat találat linkje
        club_link = soup.select_one("table.items td.hauptlink a")
        if club_link and club_link.get("href"):
            return "https://www.transfermarkt.com" + club_link["href"]
        return None
    except Exception:
        return None


def fetch_transfermarkt(team_name: str) :
    """Transfermarkt sérülés/eltiltás lista lekérése."""
    try:
        team_url = _search_transfermarkt(team_name)
        if not team_url:
            return []

        # Sérülés oldal: /verletzungen/verein/[id]
        club_id = team_url.split("/")[-1]
        club_slug = team_url.split("/")[3]
        injury_url = f"https://www.transfermarkt.com/{club_slug}/verletzungen/verein/{club_id}"

        resp = requests.get(
            injury_url,
            headers={**HEADERS, "Accept-Language": "en-US,en;q=0.9"},
            timeout=10,
        )
        soup = BeautifulSoup(resp.text, "html.parser")

        injuries = []
        rows = soup.select("table.items tbody tr")
        for row in rows[:8]:
            cols = row.select("td")
            if len(cols) < 4:
                continue
            player = row.select_one("td.hauptlink a")
            if not player:
                continue
            player_name = player.text.strip()
            injury_type = cols[3].text.strip() if len(cols) > 3 else ""
            since = cols[4].text.strip() if len(cols) > 4 else ""
            until = cols[5].text.strip() if len(cols) > 5 else ""

            if player_name:
                injuries.append({
                    "source": "Transfermarkt",
                    "title": f"{player_name} - {injury_type}",
                    "description": f"Sérült: {since}, Várható visszatérés: {until}",
                    "url": injury_url,
                    "published": datetime.now().strftime("%Y-%m-%d"),
                })

        return injuries
    except Exception:
        return []


# ── 3. BBC Sport ───────────────────────────────────────────────────────────────

def fetch_bbc_sport(team_name: str) :
    """BBC Sport hírek keresése a csapatról."""
    try:
        # BBC Sport keresés
        resp = requests.get(
            f"https://www.bbc.com/sport/football",
            headers=HEADERS,
            timeout=10,
        )
        soup = BeautifulSoup(resp.text, "html.parser")

        articles = []
        team_lower = team_name.lower()

        for article in soup.select("article, [data-testid='card']")[:30]:
            title_el = article.select_one("h3, h2, [data-testid='card-headline']")
            if not title_el:
                continue
            title = title_el.text.strip()
            if not any(word.lower() in title.lower() for word in team_name.split()):
                continue

            link_el = article.select_one("a[href]")
            url = ""
            if link_el:
                href = link_el.get("href", "")
                url = f"https://www.bbc.com{href}" if href.startswith("/") else href

            desc_el = article.select_one("p")
            desc = desc_el.text.strip() if desc_el else ""

            articles.append({
                "source": "BBC Sport",
                "title": title,
                "description": desc,
                "url": url,
                "published": datetime.now().strftime("%Y-%m-%d"),
            })

        return articles[:5]
    except Exception:
        return []


# ── Sky Sports ─────────────────────────────────────────────────────────────────

def fetch_sky_sports(team_name: str) :
    """Sky Sports hírek keresése."""
    try:
        resp = requests.get(
            f"https://www.skysports.com/football/news",
            headers=HEADERS,
            timeout=10,
        )
        soup = BeautifulSoup(resp.text, "html.parser")

        articles = []
        for item in soup.select(".news-list__item, .sdc-site-tile")[:30]:
            title_el = item.select_one("h3, h4, .sdc-site-tile__headline-text")
            if not title_el:
                continue
            title = title_el.text.strip()
            if not any(word.lower() in title.lower() for word in team_name.split()):
                continue

            link_el = item.select_one("a[href]")
            url = link_el["href"] if link_el else ""

            articles.append({
                "source": "Sky Sports",
                "title": title,
                "description": "",
                "url": url,
                "published": datetime.now().strftime("%Y-%m-%d"),
            })

        return articles[:3]
    except Exception:
        return []


# ── Aggregátor ─────────────────────────────────────────────────────────────────

def fetch_all_news(team_name: str) :
    """Mind a 3 forrásból összegyűjti a híreket."""
    news = []
    news.extend(fetch_transfermarkt(team_name))
    news.extend(fetch_newsapi(team_name))
    news.extend(fetch_bbc_sport(team_name))
    news.extend(fetch_sky_sports(team_name))
    return news
