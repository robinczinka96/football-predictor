"""
Groq (Llama 3.3 70B) hírelemzés - meccs-hatás becslése sérülés/eltiltás hírekből.
"""
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY

from groq import Groq


def analyze_match_news(
    home_team: str,
    away_team: str,
    home_news: list,
    away_news: list,
    prob_H: float,
    prob_D: float,
    prob_A: float,
) -> dict:
    """
    Groq Llama elemzi a híreket és becsüli a valószínűség-módosítást.
    """
    if not GROQ_API_KEY:
        return _empty_result("Groq API kulcs hiányzik")

    def format_news(news_list: list) -> str:
        if not news_list:
            return "Nincs releváns hír."
        lines = []
        for n in news_list[:6]:
            lines.append(f"[{n['source']}] {n['title']}")
            if n.get("description"):
                lines.append(f"  → {n['description'][:150]}")
        return "\n".join(lines)

    prompt = f"""You are a professional sports betting analyst. Analyze the following match news and estimate the probability adjustment.

MATCH: {home_team} (home) vs {away_team} (away)

CURRENT PROBABILITIES (ML model):
- Home win: {prob_H:.1%}
- Draw: {prob_D:.1%}
- Away win: {prob_A:.1%}

{home_team} NEWS:
{format_news(home_news)}

{away_team} NEWS:
{format_news(away_news)}

Instructions:
1. Identify key injuries or suspensions (goalkeeper, captain, top scorer, key midfielder)
2. Estimate probability adjustment between -0.15 and +0.15
3. Write a SHORT summary in Hungarian

Respond ONLY with valid JSON, no extra text:
{{
  "home_adjustment": 0.00,
  "away_adjustment": 0.00,
  "summary": "Magyar összefoglaló...",
  "key_news": ["hír1", "hír2"],
  "severity": "none",
  "confidence": "medium"
}}

Rules:
- severity: "high" (key player missing), "medium" (important player missing), "low" (minor impact), "none" (no relevant news)
- If no relevant injury/suspension news found, set both adjustments to 0.00
- Keep summary under 100 words in Hungarian"""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content.strip()
        result = json.loads(text)

        result["home_adjustment"] = max(-0.15, min(0.15, float(result.get("home_adjustment", 0))))
        result["away_adjustment"] = max(-0.15, min(0.15, float(result.get("away_adjustment", 0))))
        result.setdefault("summary", "Nincs elegendő hír az elemzéshez.")
        result.setdefault("key_news", [])
        result.setdefault("severity", "none")
        result.setdefault("confidence", "medium")

        return result

    except Exception as e:
        return _empty_result(f"Elemzési hiba: {e}")


def _empty_result(reason: str = "") -> dict:
    return {
        "home_adjustment": 0.0,
        "away_adjustment": 0.0,
        "summary": reason or "Nem sikerült elemezni.",
        "key_news": [],
        "severity": "none",
        "confidence": "low",
    }
