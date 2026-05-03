"""
NBA Betting Picks — Streamlit Web App
Run with:  streamlit run app.py
Then open: http://localhost:8501
"""

import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ET = ZoneInfo("America/New_York")

st.set_page_config(
    page_title="NBA Betting Picks",
    page_icon="🏀",
    layout="wide",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.pick-card { background: #1a1a2e; border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem; }
.game-title { font-size: 1.4rem; font-weight: 700; color: #e0e0e0; }
.game-time  { font-size: 0.9rem; color: #888; margin-bottom: 0.8rem; }
.edge-strong { color: #00e676; font-weight: 700; }
.edge-lean   { color: #ffb300; font-weight: 600; }
.no-edge     { color: #666; }
.reason-bullet { color: #ccc; font-size: 0.95rem; margin: 0.15rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
now = datetime.now(ET)
col_title, col_btn = st.columns([4, 1])
with col_title:
    st.markdown(f"# 🏀 NBA Betting Picks")
    st.markdown(f"**{now.strftime('%A, %B')} {now.day}, {now.year}** &nbsp;·&nbsp; "
                f"Last refreshed: {now.strftime('%I:%M %p ET').lstrip('0')}")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Odds", use_container_width=True):
        for fname in ["game_odds.pkl", "player_props.pkl", "todays_schedule.pkl",
                      "current_games_2025_26.pkl", "current_player_logs_2025_26.pkl"]:
            p = Path("cache") / fname
            if p.exists():
                p.unlink()
        st.cache_data.clear()
        st.rerun()

st.divider()


# ── Load picks (cached 30 min so the page doesn't hammer the API on every visit) ──
@st.cache_data(ttl=1800, show_spinner=False)
def load_picks(upcoming: bool):
    from models.predict import get_picks
    return get_picks(upcoming=upcoming)


view = st.radio("Show games:", ["Tonight only", "All upcoming"], horizontal=True, label_visibility="collapsed")
upcoming = (view == "All upcoming")

with st.spinner("Fetching odds and computing picks..."):
    try:
        picks = load_picks(upcoming)
    except Exception as e:
        st.error(f"Error loading picks: {e}\n\nMake sure you've run `python main.py train` first.")
        st.stop()

if not picks:
    st.warning("No games found. Models may not be trained yet — run `python main.py train`.")
    st.stop()

edge_picks = [p for p in picks if p["has_any_edge"]]
no_edge_picks = [p for p in picks if not p["has_any_edge"]]

# ── Summary banner ──────────────────────────────────────────────────────────────
if edge_picks:
    st.success(f"**{len(edge_picks)} game(s) with edge** found out of {len(picks)} today.")
else:
    st.info(f"**No edges found** in {len(picks)} game(s) today. The model doesn't see value against current lines.")

st.markdown("---")


# ── Helper to render a bet badge ───────────────────────────────────────────────
def bet_badge(label: str, is_strong: bool) -> str:
    if is_strong:
        return f"<span class='edge-strong'>★ STRONG LEAN &nbsp;{label}</span>"
    return f"<span class='edge-lean'>LEAN &nbsp;{label}</span>"


def no_edge_badge(edge_str: str) -> str:
    return f"<span class='no-edge'>No edge &nbsp;({edge_str})</span>"


# ── Edge games ──────────────────────────────────────────────────────────────────
for pick in edge_picks:
    home, away = pick["home_team"], pick["away_team"]
    sp = pick["spread"]
    ml = pick["moneyline"]

    with st.container():
        st.markdown(
            f"<div class='game-title'>{pick['away_name']} &nbsp;@&nbsp; {pick['home_name']}</div>"
            f"<div class='game-time'>{pick['time']}</div>",
            unsafe_allow_html=True,
        )

        col_sp, col_ml = st.columns(2)

        # Spread column
        with col_sp:
            st.markdown("**SPREAD**")
            if sp["line"] is not None:
                st.markdown(f"Line: `{home} {sp['line']:+.1f}` &nbsp;·&nbsp; Model: `{home} {sp['predicted_margin']:+.1f} pts`")
                if sp["has_edge"]:
                    st.markdown(bet_badge(sp["pick_label"], sp["is_strong"]) +
                                f" &nbsp;<small>(edge {sp['edge']:+.1f} pts)</small>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(no_edge_badge(f"{sp['edge']:+.1f} pts"), unsafe_allow_html=True)
            else:
                st.markdown("Line not available")

        # Moneyline column
        with col_ml:
            st.markdown("**MONEYLINE**")
            if ml["implied_home"] is not None:
                home_odds_str = f"+{int(ml['home_odds'])}" if ml['home_odds'] > 0 else str(int(ml['home_odds']))
                away_odds_str = f"+{int(ml['away_odds'])}" if ml['away_odds'] > 0 else str(int(ml['away_odds']))
                st.markdown(
                    f"`{home}` {home_odds_str} ({ml['implied_home']:.0%}) &nbsp;·&nbsp; "
                    f"`{away}` {away_odds_str} ({ml['implied_away']:.0%})"
                )
                st.markdown(f"Model: `{home}` {ml['home_prob']:.0%} &nbsp;·&nbsp; `{away}` {ml['away_prob']:.0%}")
                if ml["has_edge"]:
                    st.markdown(bet_badge(ml["pick_name"], ml["is_strong"]) +
                                f" &nbsp;<small>(edge {ml['best_edge']:+.1%})</small>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(no_edge_badge(f"{ml['best_edge']:+.1%}"), unsafe_allow_html=True)
            else:
                st.markdown(f"Model: `{home}` {ml['home_prob']:.0%} &nbsp;·&nbsp; `{away}` {ml['away_prob']:.0%}")
                st.markdown("Odds not available")

        # Reasons
        if pick["reasons"]:
            st.markdown("**Why the model sees edge:**")
            for r in pick["reasons"]:
                st.markdown(f"<div class='reason-bullet'>• {r}</div>", unsafe_allow_html=True)

        # Player props
        prop_rows = pick["props"]
        if prop_rows:
            with st.expander(f"Player Props ({len([p for p in prop_rows if p['has_edge']])} edges, {len(prop_rows)} total lines)"):
                df = pd.DataFrame(prop_rows)[["player", "team", "stat", "line", "model", "edge", "direction", "has_edge", "is_strong"]]
                df = df.sort_values("edge", key=abs, ascending=False)

                def _style_edge(val):
                    if isinstance(val, bool):
                        return ""
                    return ""

                def _pick_str(row):
                    if row["is_strong"]:
                        return f"★ {row['direction']}"
                    elif row["has_edge"]:
                        return row["direction"]
                    return "—"

                df["pick"] = df.apply(_pick_str, axis=1)
                display_df = df[["player", "team", "stat", "line", "model", "edge", "pick"]].copy()
                display_df.columns = ["Player", "Team", "Stat", "Line", "Model", "Edge", "Pick"]

                st.dataframe(
                    display_df.style.apply(
                        lambda row: [
                            "color: #00e676; font-weight: bold" if row["Pick"].startswith("★")
                            else "color: #ffb300" if row["Pick"] in ("OVER", "UNDER")
                            else "color: #666"
                            for _ in row
                        ], axis=1
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        elif not pick["props"]:
            st.caption("Player prop lines not available for this game")

        st.divider()


# ── No-edge games (collapsed) ───────────────────────────────────────────────────
if no_edge_picks:
    with st.expander(f"Games with no edge ({len(no_edge_picks)})"):
        for pick in no_edge_picks:
            sp = pick["spread"]
            ml = pick["moneyline"]
            sp_str = f"Model: {pick['home_team']} {sp['predicted_margin']:+.1f} pts vs line {sp['line']:+.1f}" if sp["line"] else "No spread available"
            ml_str = f"Model: {pick['home_team']} {ml['home_prob']:.0%} vs implied {ml['implied_home']:.0%}" if ml["implied_home"] else ""
            st.markdown(
                f"**{pick['away_name']} @ {pick['home_name']}** &nbsp;·&nbsp; {pick['time']}  \n"
                f"{sp_str}  \n{ml_str}"
            )
            st.markdown("---")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.caption("★ = Strong edge  ·  Edges are model predictions vs. book lines  ·  Bet responsibly")
