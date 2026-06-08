"""
Football Predictor - Streamlit Dashboard
Fókusz: Legjobb fogadási tippek (legmagasabb konfidencia)
"""
import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import fetch_training_data, get_upcoming_matches
from model.trainer import train, load_model, MODEL_PATH
from model.predictor import predict_all_matches
from model.combo_builder import build_combos, format_combo_summary, _get_bet_odds, _get_bet_prob
from scraper.vegas_scraper import scrape_vegas
from news.news_fetcher import fetch_all_news
from news.gemini_analyzer import analyze_match_news

# ── Oldal konfig ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
)

# ── Session state init ─────────────────────────────────────────────────────────
for key in ["raw_df", "historical_df", "predictions", "combos"]:
    if key not in st.session_state:
        st.session_state[key] = None
for key in ["elo_dict", "streak_dict"]:
    if key not in st.session_state:
        st.session_state[key] = {}
if "news_analysis" not in st.session_state:
    st.session_state.news_analysis = {}

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("⚽ Football Predictor")
st.caption("Legjobb fogadási tippek | football-data.org + vegas.hu")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Beállítások")

    model_exists = os.path.exists(MODEL_PATH)
    if model_exists:
        st.success("✅ Modell betöltve")
    else:
        st.warning("⚠️ Nincs betanított modell")

    st.divider()
    st.subheader("1. Adatok & Modell")

    if st.button("📥 Adatok letöltése & Modell tanítása", use_container_width=True):
        with st.spinner("Historikus adatok letöltése (párhuzamos, ~2-3 perc)..."):
            raw_df, processed_df, elo_dict, streak_dict = fetch_training_data()
            if processed_df.empty:
                st.error("Nem sikerült adatot letölteni!")
            else:
                st.session_state.raw_df = raw_df
                st.session_state.historical_df = processed_df
                st.session_state.elo_dict = elo_dict
                st.session_state.streak_dict = streak_dict
                st.success(f"{len(raw_df)} meccs letöltve")

        if st.session_state.historical_df is not None:
            with st.spinner("Modell tanítása (Optuna 8 trial + Ensemble + Dixon-Coles, ~3-5 perc)..."):
                try:
                    from model.dixon_coles import DixonColesModel, save_dc_model
                    train(st.session_state.historical_df)
                    dc = DixonColesModel()
                    dc.fit(st.session_state.raw_df)
                    save_dc_model(dc)
                    st.success("Modell sikeresen betanítva!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Tanítási hiba: {e}")

    st.divider()
    st.subheader("2. Előrejelzések")

    source = st.radio(
        "Meccs forrás:",
        ["🌐 Vegas.hu (élő oddsok)", "📅 Következő 7 nap (API)"],
        index=0,
    )

    min_conf = st.slider(
        "Min. konfidencia",
        min_value=35, max_value=75, value=50, step=5,
        help="Csak ennyi % feletti tippek jelennek meg"
    ) / 100.0

    top_n = st.slider(
        "Megjelenített top tippek",
        min_value=5, max_value=50, value=20, step=5,
        help="Hány legjobb tipp jelenjen meg kiemelve"
    )

    if st.button("🔮 Előrejelzések futtatása", use_container_width=True, disabled=not model_exists):
        if st.session_state.raw_df is None:
            with st.spinner("Historikus adatok betöltése..."):
                raw_df, processed_df, elo_dict, streak_dict = fetch_training_data(seasons=[2023, 2024])
                st.session_state.raw_df = raw_df
                st.session_state.historical_df = processed_df
                st.session_state.elo_dict = elo_dict
                st.session_state.streak_dict = streak_dict

        if "Vegas" in source:
            with st.spinner("Vegas.hu scraping..."):
                matches = scrape_vegas()
            if not matches:
                st.warning("Nem sikerült meccseket letölteni. Próbáld az API forrást!")
                matches = []
        else:
            with st.spinner("Közelgő meccsek letöltése..."):
                matches = get_upcoming_matches()

        if matches and st.session_state.raw_df is not None:
            with st.spinner("Előrejelzések számítása..."):
                preds = predict_all_matches(
                    matches,
                    st.session_state.raw_df,
                    st.session_state.elo_dict,
                    st.session_state.streak_dict,
                    min_confidence=min_conf,
                )
                st.session_state.predictions = preds
                st.success(f"{len(preds)} tipp kész!")
        elif not matches:
            st.warning("Nem találtunk meccseket.")

    st.divider()
    st.subheader("3. Szelvényajánló")
    st.caption("Optimális kombináció EV + Kelly alapján")

    combo_min_conf = st.slider(
        "Min. egyedi konfidencia (szelvény)",
        min_value=50, max_value=75, value=55, step=5,
        help="Csak ennyi % feletti meccsekből épít szelvényt"
    ) / 100.0

    combo_max_size = st.slider(
        "Max. meccs/szelvény", min_value=2, max_value=6, value=5,
        help="Maximálisan ennyi meccs kerülhet egy szelvényre"
    )

    if st.button("🎯 Szelvények generálása", use_container_width=True,
                 disabled=st.session_state.predictions is None
                 or st.session_state.predictions.empty):
        with st.spinner("Kombinációk optimalizálása..."):
            combos = build_combos(
                st.session_state.predictions,
                max_combo_size=combo_max_size,
                min_individual_conf=combo_min_conf,
            )
            st.session_state.combos = combos
        if combos:
            total = sum(len(v["combos"]) for v in combos.values())
            st.success(f"{total} szelvény generálva!")
        else:
            st.warning("Nincs elég magas konfidenciájú meccs. Csökkentsd a min. konfidenciát!")

    st.divider()
    st.subheader("4. Hírelemzés")
    st.caption("Top tippek elemzése Groq Llama 3.3-mal")

    news_count = st.slider("Hány meccs elemzése?", 1, 10, 5)

    if st.button("📰 Hírelemzés futtatása", use_container_width=True,
                 disabled=st.session_state.predictions is None
                 or st.session_state.predictions.empty):
        preds = st.session_state.predictions
        top_matches = preds.head(news_count)

        analysis = {}
        progress = st.progress(0)
        for i, (_, row) in enumerate(top_matches.iterrows()):
            h, a = row["home_team"], row["away_team"]
            st.write(f"  Elemzés: {h} vs {a}...")
            home_news = fetch_all_news(h)
            away_news = fetch_all_news(a)
            result = analyze_match_news(
                h, a, home_news, away_news,
                row.get("prob_H", 0.33),
                row.get("prob_D", 0.33),
                row.get("prob_A", 0.33),
            )
            result["home_news"] = home_news
            result["away_news"] = away_news
            analysis[f"{h} vs {a}"] = result
            progress.progress((i + 1) / len(top_matches))

        st.session_state.news_analysis = analysis
        st.success(f"{len(analysis)} meccs elemezve!")

    st.divider()
    st.caption("v4.0 | XGBoost + LightGBM + Dixon-Coles + Groq")


# ── Fő tartalom ────────────────────────────────────────────────────────────────
if st.session_state.predictions is not None and not st.session_state.predictions.empty:
    df = st.session_state.predictions.copy()

    # ── Tipp oszlop: a legjobb fogadás (legmagasabb valószínűségű kimenetel) ──
    def best_bet_row(row):
        return max(
            [("H", row["prob_H"]), ("D", row["prob_D"]), ("A", row["prob_A"])],
            key=lambda x: x[1]
        )

    df["best_outcome"] = df.apply(lambda r: best_bet_row(r)[0], axis=1)
    df["best_prob"] = df.apply(lambda r: best_bet_row(r)[1], axis=1)

    def bet_label(row):
        o = row["best_outcome"]
        if o == "H":
            return f"🏠 {row['home_team']}"
        elif o == "A":
            return f"✈️ {row['away_team']}"
        return "🤝 Döntetlen"

    def conf_color(conf):
        if conf >= 0.65:
            return "🟢"
        elif conf >= 0.55:
            return "🟡"
        return "🔴"

    def bet_odds(row):
        o = row["best_outcome"]
        if o == "H" and "odds_H" in row:
            return row.get("odds_H")
        elif o == "D" and "odds_D" in row:
            return row.get("odds_D")
        elif o == "A" and "odds_A" in row:
            return row.get("odds_A")
        return None

    # ── Metrikák ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Összes tipp", len(df))
    with col2:
        st.metric("🟢 65%+ konfidencia", int((df["confidence"] >= 0.65).sum()))
    with col3:
        st.metric("🟡 55-65% konfidencia", int(((df["confidence"] >= 0.55) & (df["confidence"] < 0.65)).sum()))
    with col4:
        st.metric("Átl. konfidencia", f"{df['confidence'].mean():.1%}")

    st.divider()

    # ── Szelvényajánló ──────────────────────────────────────────────────────
    combos = st.session_state.get("combos")
    if combos:
        st.subheader("🎯 Szelvényajánló")
        st.caption("EV és Kelly-kritérium alapján optimalizált kombinációk")

        tier_cols = st.columns(len(combos))
        for col, (tier_key, tier_data) in zip(tier_cols, combos.items()):
            with col:
                st.markdown(f"### {tier_data['label']}")

        for tier_key, tier_data in combos.items():
            st.markdown(f"#### {tier_data['label']}")
            for i, combo in enumerate(tier_data["combos"], 1):
                prob = combo["combined_prob"]
                odds = combo["combined_odds"]
                ev = combo["ev"]
                kelly = combo["kelly"]

                with st.container():
                    h1, h2, h3, h4 = st.columns([1, 1, 1, 1])
                    with h1:
                        st.metric("Szorzó", f"{odds:.2f}x")
                    with h2:
                        st.metric("Valószínűség", f"{prob:.0%}")
                    with h3:
                        ev_str = f"{ev:.2f}" + (" ✅" if ev >= 1.0 else " ⚠️")
                        st.metric("EV", ev_str)
                    with h4:
                        st.metric("Kelly tét", f"{kelly:.1%}",
                                  help="Tőkéd ekkora részét tedd fel (1/4 Kelly)")

                    st.markdown("**Meccsek:**")
                    for row in combo["matches"]:
                        outcome = row.get("best_outcome") or row.get("predicted", "?")
                        label = {
                            "H": f"🏠 **{row.get('home_team','?')}** győz",
                            "D": "🤝 **Döntetlen**",
                            "A": f"✈️ **{row.get('away_team','?')}** győz",
                        }.get(outcome, outcome)
                        bet_odds = _get_bet_odds(row)
                        bet_prob = _get_bet_prob(row)
                        st.markdown(
                            f"&nbsp;&nbsp;• {row.get('home_team','?')} vs {row.get('away_team','?')} "
                            f"→ {label} &nbsp; `{bet_prob:.0%}` @ `{bet_odds:.2f}`"
                        )

                    # Tét javaslat konkrét összegekben
                    st.caption(
                        f"💡 Pl. 10.000 Ft tét esetén: "
                        f"nyerés = **{10000 * odds:,.0f} Ft** | "
                        f"várható érték = **{10000 * ev:,.0f} Ft**"
                    )
                    st.divider()

        st.divider()

    # ── Top N legjobb tipp ──
    top_df = df.head(top_n)
    news_analysis = st.session_state.news_analysis or {}

    st.subheader(f"🏆 Top {top_n} legjobb tipp")
    st.caption("Konfidencia szerint rendezve — ezekre érdemes fogadni")

    for _, row in top_df.iterrows():
        match_key = f"{row['home_team']} vs {row['away_team']}"
        analysis = news_analysis.get(match_key)
        conf = row["confidence"]
        icon = conf_color(conf)
        label = bet_label(row)
        odds_val = bet_odds(row)

        with st.container():
            c1, c2, c3, c4 = st.columns([3, 2, 1, 2])
            with c1:
                st.markdown(f"**{row['home_team']} vs {row['away_team']}**")
                st.caption(f"{row.get('competition', '')} · {row.get('date', '')}")
            with c2:
                st.markdown(f"**Tipp:** {label}")
                st.caption(
                    f"H: {row['prob_H']:.0%} · D: {row['prob_D']:.0%} · A: {row['prob_A']:.0%}"
                )
            with c3:
                # Konfidencia + módosítás ha van news
                adj = 0.0
                if analysis:
                    adj = analysis.get("home_adjustment", 0) if row.get("best_outcome") == "H" \
                        else (analysis.get("away_adjustment", 0) if row.get("best_outcome") == "A" else 0)
                if adj != 0:
                    st.markdown(f"{icon} **{conf:.0%} → {conf + adj:.0%}**")
                    st.caption("(hír módosítás)")
                else:
                    st.markdown(f"{icon} **{conf:.0%}**")
            with c4:
                if odds_val and odds_val > 1:
                    st.markdown(f"**Odds: {odds_val:.2f}**")
                    # Várható érték = conf * odds
                    ev = conf * odds_val
                    ev_label = f"EV: {ev:.2f}" + (" ✅" if ev > 1.0 else " ⚠️")
                    st.caption(ev_label)
                else:
                    st.markdown("Odds: –")

            # Hírelemzés panel
            if analysis:
                sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢", "none": "⚪"}.get(
                    analysis.get("severity", "none"), "⚪")
                with st.expander(f"{sev_icon} Hírelemzés (Groq Llama)"):
                    st.write(analysis.get("summary", ""))
                    key_news = analysis.get("key_news", [])
                    if key_news:
                        st.markdown("**Kulcshírek:**")
                        for n in key_news:
                            st.markdown(f"- {n}")
                    adj_h = analysis.get("home_adjustment", 0)
                    adj_a = analysis.get("away_adjustment", 0)
                    if adj_h != 0 or adj_a != 0:
                        st.markdown(
                            f"**Valószínűség módosítás:** Hazai {adj_h:+.1%} | Vendég {adj_a:+.1%}"
                        )
                    all_news = analysis.get("home_news", []) + analysis.get("away_news", [])
                    if all_news:
                        st.markdown("**Forrás hírek:**")
                        for n in all_news[:6]:
                            st.markdown(f"- [{n['source']}] {n['title']}")

            st.divider()

    # ── Összes tipp táblázat ──
    st.subheader("📊 Összes tipp (konfidencia szerint)")

    show_df = df[[
        "date", "competition", "home_team", "away_team",
        "prob_H", "prob_D", "prob_A", "best_outcome", "confidence",
    ]].copy()

    if "odds_H" in df.columns:
        show_df["tipp_odds"] = df.apply(bet_odds, axis=1)

    for col in ["prob_H", "prob_D", "prob_A", "confidence"]:
        show_df[col] = show_df[col].apply(lambda x: f"{x:.0%}")

    if "tipp_odds" in show_df.columns:
        show_df["tipp_odds"] = show_df["tipp_odds"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) and x else "–"
        )

    show_df = show_df.rename(columns={
        "date": "Dátum", "competition": "Liga",
        "home_team": "Hazai", "away_team": "Vendég",
        "prob_H": "H%", "prob_D": "D%", "prob_A": "A%",
        "best_outcome": "Tipp", "confidence": "Konfidencia",
        "tipp_odds": "Odds",
    })

    st.dataframe(show_df, use_container_width=True, hide_index=True)

else:
    st.info("""
    ### Hogyan használd?

    1. **Sidebar → "Adatok letöltése & Modell tanítása"**
       - Letölti az elmúlt 6 év meccseit (~8-10 perc)
       - Betanítja az XGBoost + LightGBM + Dixon-Coles modellt

    2. **Sidebar → "Előrejelzések futtatása"**
       - Vegas.hu: élő oddsokat + meccseket tölt le
       - API: a következő 7 nap meccseit mutatja

    3. **Nézd meg a Top Tippeket**
       - Konfidencia szerint rendezve
       - 🟢 65%+ = erős tipp | 🟡 55-65% = közepes | 🔴 <55% = gyengébb

    4. **EV (Expected Value)** = konfidencia × odds
       - Ha EV > 1.0 ✅ → hosszú távon nyereséges lehet
       - Ha EV < 1.0 ⚠️ → az odds nem kompenzál elég

    ---
    💡 **Tipp:** Először tanítsd be a modellt, utána futtasd az előrejelzéseket!
    """)
