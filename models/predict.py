"""Loads trained models and generates picks for tonight's NBA games."""

import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.nba_stats import fetch_todays_game_schedule, fetch_current_season_games, fetch_current_season_player_logs
from data.odds_api import fetch_game_odds, fetch_player_prop_odds, american_to_prob
from data.weather import fetch_todays_weather
from features.team_features import build_prediction_row, get_feature_cols
from features.player_features import build_player_prediction_row, get_player_feature_cols

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

STRONG_MONEYLINE_EDGE = 0.08
STRONG_SPREAD_EDGE = 4.0
STRONG_PROPS_EDGE = 2.5


def _load(name: str):
    path = config.MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def _fmt_odds(american):
    if american is None:
        return "N/A"
    return f"+{int(american)}" if american > 0 else str(int(american))


def _fmt_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone(ET)
        return dt.strftime("%I:%M %p ET").lstrip("0")
    except Exception:
        return iso_str


def generate_reasons(home: str, away: str, feat_row: dict,
                     pred_margin: float, spread_line, home_win_prob: float) -> list:
    """Generate human-readable bullet points explaining why the model sees an edge."""
    reasons = []
    home_name = config.TEAM_ABBR_TO_NAME.get(home, home)
    away_name = config.TEAM_ABBR_TO_NAME.get(away, away)

    # Win/loss streaks
    home_streak = int(feat_row.get("home_streak", 0))
    away_streak = int(feat_row.get("away_streak", 0))
    if home_streak >= 3:
        reasons.append(f"{home_name} is on a {home_streak}-game win streak")
    elif home_streak <= -3:
        reasons.append(f"{home_name} has lost {abs(home_streak)} straight")
    if away_streak >= 3:
        reasons.append(f"{away_name} is on a {away_streak}-game win streak")
    elif away_streak <= -3:
        reasons.append(f"{away_name} has lost {abs(away_streak)} straight")

    # Back-to-back / rest disadvantage
    if feat_row.get("away_is_back_to_back") == 1:
        reasons.append(f"{away_name} is on a back-to-back with zero days rest")
    elif feat_row.get("home_is_back_to_back") == 1:
        reasons.append(f"{home_name} is on a back-to-back with zero days rest")

    # Rest day advantage
    home_rest = float(feat_row.get("home_rest_days", 2))
    away_rest = float(feat_row.get("away_rest_days", 2))
    diff = home_rest - away_rest
    if diff >= 2:
        reasons.append(f"{home_name} has {int(diff)} more rest days than {away_name}")
    elif diff <= -2:
        reasons.append(f"{away_name} has {int(-diff)} more rest days than {home_name}")

    # Offense vs defense matchup
    home_pts = feat_row.get("home_pts_avg_10", 0) or 0
    away_pts = feat_row.get("away_pts_avg_10", 0) or 0
    home_def = feat_row.get("home_pts_allowed_avg_10", 0) or 0
    away_def = feat_row.get("away_pts_allowed_avg_10", 0) or 0

    if home_pts > 0 and away_def > 0 and home_pts > away_def + 5:
        reasons.append(
            f"{home_name} scoring {home_pts:.1f} ppg vs {away_name}'s defense allowing {away_def:.1f} ppg (last 10)"
        )
    if away_pts > 0 and home_def > 0 and home_def > away_pts + 5:
        reasons.append(
            f"{home_name}'s defense holding opponents to {home_def:.1f} ppg; {away_name} scores only {away_pts:.1f} (last 10)"
        )
    if away_pts > 0 and home_def > 0 and away_pts > home_def + 5:
        reasons.append(
            f"{away_name} scoring {away_pts:.1f} ppg vs {home_name}'s defense allowing {home_def:.1f} ppg (last 10)"
        )

    # Recent win rate
    home_wr = feat_row.get("home_wl_num_avg_10", 0.5) or 0.5
    away_wr = feat_row.get("away_wl_num_avg_10", 0.5) or 0.5
    if home_wr >= 0.7:
        w = round(home_wr * 10)
        reasons.append(f"{home_name} is {w}-{10-w} in their last 10 games")
    elif home_wr <= 0.3:
        w = round(home_wr * 10)
        reasons.append(f"{home_name} is just {w}-{10-w} in their last 10 games")
    if away_wr >= 0.7:
        w = round(away_wr * 10)
        reasons.append(f"{away_name} is {w}-{10-w} in their last 10 games")
    elif away_wr <= 0.3:
        w = round(away_wr * 10)
        reasons.append(f"{away_name} is just {w}-{10-w} in their last 10 games")

    # Head-to-head history
    h2h_pct = feat_row.get("h2h_home_win_pct", 0.5) or 0.5
    if h2h_pct >= 0.7:
        wins = round(h2h_pct * 10)
        reasons.append(f"{home_name} has won {wins} of their last 10 head-to-head matchups vs {away_name}")
    elif h2h_pct <= 0.3:
        wins = round((1 - h2h_pct) * 10)
        reasons.append(f"{away_name} has won {wins} of their last 10 head-to-head matchups vs {home_name}")

    # Model vs line explanation (always include this as final reason)
    if spread_line is not None:
        edge = pred_margin - (-spread_line)
        if pred_margin > 0:
            reasons.append(
                f"Model projects {home_name} winning by {abs(pred_margin):.1f} pts "
                f"— {'covers' if edge > 0 else 'does not cover'} the {spread_line:+.1f} spread"
            )
        else:
            reasons.append(
                f"Model projects {away_name} winning by {abs(pred_margin):.1f} pts "
                f"— {'covers' if edge < 0 else 'does not cover'} the {-spread_line:+.1f} spread"
            )

    return reasons[:6]


def get_picks(upcoming: bool = False) -> list:
    """
    Compute picks for tonight's (or upcoming) NBA games.
    Returns a list of game dicts with moneyline, spread, props, and reasons.
    Only games where the model finds at least one edge are flagged has_any_edge=True.
    """
    ml_model = _load("moneyline_model")
    spread_model = _load("spread_model")
    team_feat_cols = _load("team_feature_cols")
    player_feat_cols = _load("player_feature_cols")

    if ml_model is None or spread_model is None:
        return []

    prop_models = {s: _load(f"props_{s}_model") for s in ["pts", "reb", "ast", "fg3m"]}

    current_games = fetch_current_season_games()
    current_player_logs = fetch_current_season_player_logs()
    weather = fetch_todays_weather()
    game_odds = fetch_game_odds()
    prop_odds_list = fetch_player_prop_odds()

    # Filter to today only (or upcoming)
    today_et = datetime.now(ET).date()
    def _keep(iso_str: str) -> bool:
        try:
            gd = datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone(ET).date()
            return gd >= today_et if upcoming else gd == today_et
        except Exception:
            return True

    game_odds = [g for g in game_odds if _keep(g.get("commence_time", ""))]
    prop_odds_list = [p for p in prop_odds_list
                      if any(g["event_id"] == p.get("event_id") for g in game_odds)]

    # Build prop lookup {(player, stat): {over: {line, odds}, under: {line, odds}}}
    prop_lookup = {}
    for p in prop_odds_list:
        key = (p["player_name"], p["stat"])
        if key not in prop_lookup:
            prop_lookup[key] = {}
        prop_lookup[key][p["direction"].lower()] = {"line": p["line"], "odds": p["odds"]}

    schedule = fetch_todays_game_schedule()
    if not game_odds and schedule:
        game_odds = [
            {"event_id": g["game_id"], "commence_time": g.get("game_status", ""),
             "home_team": g["home_team"], "away_team": g["away_team"],
             "home_team_full": config.TEAM_ABBR_TO_NAME.get(g["home_team"], g["home_team"]),
             "away_team_full": config.TEAM_ABBR_TO_NAME.get(g["away_team"], g["away_team"]),
             "ml_home": None, "ml_away": None, "spread_home": None, "spread_home_odds": None}
            for g in schedule
        ]

    results = []

    for game in game_odds:
        home = game["home_team"]
        away = game["away_team"]
        home_name = config.TEAM_ABBR_TO_NAME.get(home, home)
        away_name = config.TEAM_ABBR_TO_NAME.get(away, away)
        time_str = _fmt_time(game.get("commence_time", ""))

        # Build team features
        feat_row = build_prediction_row(home, away, current_games, weather)
        feat_df = pd.DataFrame([feat_row])
        X = feat_df.reindex(columns=team_feat_cols, fill_value=0.0)

        # Moneyline
        home_win_prob = float(ml_model.predict_proba(X)[0][1])
        away_win_prob = 1.0 - home_win_prob
        ml_home = game.get("ml_home")
        ml_away = game.get("ml_away")
        implied_home = american_to_prob(ml_home) if ml_home else None
        implied_away = american_to_prob(ml_away) if ml_away else None

        edge_home = (home_win_prob - implied_home) if implied_home else 0.0
        edge_away = (away_win_prob - implied_away) if implied_away else 0.0
        ml_best_edge = max(edge_home, edge_away)
        ml_pick_team = home if edge_home >= edge_away else away
        ml_pick_name = home_name if ml_pick_team == home else away_name
        ml_has_edge = ml_best_edge >= config.MIN_EDGE_MONEYLINE and implied_home is not None

        moneyline = {
            "home_prob": home_win_prob,
            "away_prob": away_win_prob,
            "home_odds": ml_home,
            "away_odds": ml_away,
            "implied_home": implied_home,
            "implied_away": implied_away,
            "edge_home": edge_home,
            "edge_away": edge_away,
            "best_edge": ml_best_edge,
            "pick_team": ml_pick_team,
            "pick_name": ml_pick_name,
            "has_edge": ml_has_edge,
            "is_strong": ml_best_edge >= STRONG_MONEYLINE_EDGE,
        }

        # Spread
        pred_margin = float(spread_model.predict(X)[0])
        spread_line = game.get("spread_home")
        if spread_line is not None:
            sp_edge = pred_margin - (-spread_line)
            sp_cover_home = pred_margin > -spread_line
            sp_pick_team = home if sp_cover_home else away
            sp_pick_label = f"{home} {spread_line:+.1f}" if sp_cover_home else f"{away} {-spread_line:+.1f}"
            sp_has_edge = abs(sp_edge) >= config.MIN_EDGE_SPREAD
        else:
            sp_edge = 0.0
            sp_pick_team = home
            sp_pick_label = "N/A"
            sp_has_edge = False

        spread = {
            "line": spread_line,
            "predicted_margin": pred_margin,
            "edge": sp_edge,
            "pick_team": sp_pick_team,
            "pick_label": sp_pick_label,
            "has_edge": sp_has_edge,
            "is_strong": abs(sp_edge) >= STRONG_SPREAD_EDGE,
        }

        # Reasons
        reasons = generate_reasons(home, away, feat_row, pred_margin, spread_line, home_win_prob)

        # Player props
        game_props_raw = [p for p in prop_odds_list
                          if p["home_team"] == home or p["away_team"] == home]
        game_players = sorted({p["player_name"] for p in game_props_raw})
        props = []

        for player_name in game_players:
            plogs = current_player_logs[current_player_logs["PLAYER_NAME"] == player_name]
            if plogs.empty:
                continue
            player_team = plogs["TEAM_ABBREVIATION"].iloc[-1]
            is_home_player = (player_team == home)

            feat_row_p = build_player_prediction_row(
                player_name, is_home_player,
                away if is_home_player else home,
                current_player_logs, current_games
            )
            if not feat_row_p:
                continue

            feat_df_p = pd.DataFrame([feat_row_p])
            aligned_p = [c for c in player_feat_cols if c in feat_df_p.columns]
            if not aligned_p:
                continue
            X_p = feat_df_p[aligned_p].reindex(columns=player_feat_cols, fill_value=0.0)

            stat_display = {"pts": "PTS", "reb": "REB", "ast": "AST", "fg3m": "3PM"}
            for stat_key, stat_label in stat_display.items():
                model = prop_models.get(stat_key)
                if model is None:
                    continue
                pkey = (player_name, stat_key)
                if pkey not in prop_lookup:
                    continue
                over_info = prop_lookup[pkey].get("over", {})
                line = over_info.get("line")
                if line is None:
                    continue

                pred_val = float(model.predict(X_p)[0])
                edge = pred_val - line
                direction = "OVER" if edge > 0 else "UNDER"
                props.append({
                    "player": player_name,
                    "team": player_team,
                    "stat": stat_label,
                    "line": line,
                    "model": round(pred_val, 1),
                    "edge": round(edge, 1),
                    "direction": direction,
                    "has_edge": abs(edge) >= config.MIN_EDGE_PROPS,
                    "is_strong": abs(edge) >= STRONG_PROPS_EDGE,
                    "over_odds": over_info.get("odds"),
                    "under_odds": prop_lookup[pkey].get("under", {}).get("odds"),
                })

        props.sort(key=lambda x: abs(x["edge"]), reverse=True)
        has_any_edge = ml_has_edge or sp_has_edge or any(p["has_edge"] for p in props)

        results.append({
            "home_team": home,
            "away_team": away,
            "home_name": home_name,
            "away_name": away_name,
            "time": time_str,
            "moneyline": moneyline,
            "spread": spread,
            "reasons": reasons,
            "props": props,
            "has_any_edge": has_any_edge,
        })

    return results


def run_predictions(upcoming: bool = False) -> None:
    """Print picks to terminal. Only shows bets where the model finds an edge."""
    print("\n" + "=" * 62)
    now = datetime.now(ET)
    label = "UPCOMING PICKS" if upcoming else "TONIGHT'S PICKS"
    print(f"  NBA BETTING — {label}  —  {now.strftime('%A, %B')} {now.day}, {now.year}")
    print("=" * 62)

    picks = get_picks(upcoming=upcoming)

    if not picks:
        print("\n  Models not found. Run:  python main.py train\n")
        return

    edge_picks = [p for p in picks if p["has_any_edge"]]

    if not edge_picks:
        print(f"\n  {len(picks)} game(s) today — no edges found against current lines.")
        print("  The model doesn't see value worth betting today.\n")
        return

    print(f"\n  {len(edge_picks)} game(s) with edge (out of {len(picks)} today)\n")

    for pick in edge_picks:
        home, away = pick["home_team"], pick["away_team"]
        print(f"{'─' * 62}")
        print(f"  {pick['away_name']} @ {pick['home_name']}  |  {pick['time']}")
        print(f"{'─' * 62}")

        sp = pick["spread"]
        ml = pick["moneyline"]

        # Spread
        print(f"\n  SPREAD")
        print(f"    Model: {home} by {sp['predicted_margin']:+.1f} pts  |  Line: {home} {sp['line']:+.1f}" if sp["line"] else f"    Model: {home} by {sp['predicted_margin']:+.1f} pts  |  Line: N/A")
        if sp["has_edge"]:
            tag = "★ STRONG LEAN" if sp["is_strong"] else "LEAN"
            print(f"    Pick:  {tag} {sp['pick_label']}  (edge {sp['edge']:+.1f} pts)")
        else:
            print(f"    Pick:  No edge  ({sp['edge']:+.1f} pts)")

        # Moneyline
        print(f"\n  MONEYLINE")
        if ml["implied_home"] is not None:
            print(f"    Model: {home} {ml['home_prob']:.0%}  |  {away} {ml['away_prob']:.0%}")
            print(f"    Odds:  {home} {_fmt_odds(ml['home_odds'])} ({ml['implied_home']:.0%})  |  {away} {_fmt_odds(ml['away_odds'])} ({ml['implied_away']:.0%})")
        if ml["has_edge"]:
            tag = "★ STRONG LEAN" if ml["is_strong"] else "LEAN"
            print(f"    Pick:  {tag} {ml['pick_name']}  (edge {ml['best_edge']:+.1%})")
        else:
            print(f"    Pick:  No edge  ({ml['best_edge']:+.1%})")

        # Reasons
        if pick["reasons"]:
            print(f"\n  WHY:")
            for r in pick["reasons"]:
                print(f"    • {r}")

        # Props — only show edge picks
        edge_props = [p for p in pick["props"] if p["has_edge"]]
        if edge_props:
            print(f"\n  PLAYER PROPS")
            print(f"    {'Player':<22} {'Stat':<5} {'Line':>5}  {'Model':>5}  {'Edge':>5}  Pick")
            print("    " + "─" * 54)
            for p in edge_props:
                tag = "★ " if p["is_strong"] else "  "
                print(f"    {p['player']:<22} {p['stat']:<5} {p['line']:>5.1f}  "
                      f"{p['model']:>5.1f}  {p['edge']:>+5.1f}  {tag}{p['direction']}")

    print(f"\n{'=' * 62}")
    print("  ★ = Strong edge  |  Bet responsibly.\n")
