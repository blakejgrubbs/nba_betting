"""
NBA Betting Model
─────────────────
SETUP (first time only):
  1. pip install -r requirements.txt
  2. Copy .env.example to .env and add your free Odds API key
     → Sign up at https://the-odds-api.com/  (500 free requests/month)
  3. python main.py train      ← pulls 3 seasons of NBA data, trains models (~10-20 min)
  4. python main.py predict    ← shows tonight's picks

DAILY USE:
  python main.py predict              ← tonight's edge picks
  python main.py predict --upcoming   ← all upcoming games with odds
  python main.py debug-props          ← diagnose why player props aren't showing
  streamlit run app.py                ← open the web dashboard
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "predict", "debug-props"):
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]

    if command == "train":
        from models.train import train_all_models
        train_all_models()

    elif command == "predict":
        from models.predict import run_predictions
        upcoming = "--upcoming" in sys.argv
        run_predictions(upcoming=upcoming)

    elif command == "debug-props":
        _debug_props()


def _debug_props():
    """Diagnose the player props pipeline step by step."""
    import pandas as pd
    from data.nba_stats import fetch_current_season_games, fetch_current_season_player_logs
    from data.odds_api import fetch_game_odds, fetch_player_prop_odds
    from features.player_features import find_player_in_logs, build_player_prediction_row, get_player_feature_cols
    import joblib, config
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")
    today_et = datetime.now(ET).date()

    print("\n=== PLAYER PROPS DIAGNOSTIC ===\n")

    # 1. Fetch odds
    print("Step 1: Fetching game odds...")
    game_odds = fetch_game_odds()
    today_games = [g for g in game_odds if _date_et(g.get("commence_time","")) == today_et]
    print(f"  Found {len(today_games)} game(s) today\n")

    print("Step 2: Fetching player prop lines...")
    props = fetch_player_prop_odds()
    today_props = [p for p in props if any(g["event_id"] == p.get("event_id") for g in today_games)]
    players_with_lines = sorted({p["player_name"] for p in today_props})
    print(f"  Found {len(today_props)} prop lines for {len(players_with_lines)} players")
    if players_with_lines:
        print(f"  Players: {', '.join(players_with_lines[:10])}")
        if len(players_with_lines) > 10:
            print(f"           ...and {len(players_with_lines)-10} more")
    else:
        print("  ⚠ NO PROP LINES FOUND — check your Odds API key in .env")
        return
    print()

    # 2. Check name matching
    print("Step 3: Checking name matching against current season logs...")
    current_player_logs = fetch_current_season_player_logs()
    current_games = fetch_current_season_games()
    matched, unmatched = [], []
    for name in players_with_lines:
        logs = find_player_in_logs(name, current_player_logs)
        if logs.empty:
            unmatched.append(name)
        else:
            matched.append(name)

    print(f"  Matched:   {len(matched)} players")
    print(f"  Unmatched: {len(unmatched)} players")
    if unmatched:
        print(f"  ⚠ Could not find in nba_api: {', '.join(unmatched[:10])}")
    print()

    # 3. Check model loading
    print("Step 4: Checking prop models...")
    model_path = config.MODELS_DIR
    feat_cols = None
    for stat in ["pts", "reb", "ast", "fg3m"]:
        p = model_path / f"props_{stat}_model.pkl"
        print(f"  props_{stat}_model: {'✓ found' if p.exists() else '✗ MISSING — run: python main.py train'}")
    feat_p = model_path / "player_feature_cols.pkl"
    if feat_p.exists():
        feat_cols = joblib.load(feat_p)
        print(f"  player_feature_cols: ✓ found ({len(feat_cols)} features)")
    else:
        print("  player_feature_cols: ✗ MISSING — run: python main.py train")
        return
    print()

    # 4. Show sample predictions for first matched player
    if matched and feat_cols:
        print("Step 5: Sample predictions for first matched player...")
        sample_player = matched[0]
        sample_prop = next((p for p in today_props if p["player_name"] == sample_player), None)
        if sample_prop:
            home = sample_prop["home_team"]
            away = sample_prop["away_team"]
            logs = find_player_in_logs(sample_player, current_player_logs)
            player_team = logs["TEAM_ABBREVIATION"].iloc[-1]
            is_home = (player_team == home)
            opp = away if is_home else home

            feat_row = build_player_prediction_row(
                sample_player, is_home, opp, current_player_logs, current_games
            )
            print(f"  Player: {sample_player}  Team: {player_team}  Opp: {opp}")
            print(f"  Features built: {len(feat_row)} values")
            print(f"  Recent stats:")
            for stat in ["pts", "reb", "ast", "fg3m"]:
                val5  = feat_row.get(f"{stat}_avg_5", "N/A")
                val10 = feat_row.get(f"{stat}_avg_10", "N/A")
                print(f"    {stat.upper():<5}  last-5: {val5:.1f}  last-10: {val10:.1f}" if isinstance(val5, float) else f"    {stat.upper()}: N/A")

            model_pts = joblib.load(model_path / "props_pts_model.pkl")
            feat_df = pd.DataFrame([feat_row])
            X = feat_df.reindex(columns=feat_cols, fill_value=0.0)
            pred = float(model_pts.predict(X)[0])

            pts_props = [p for p in today_props if p["player_name"] == sample_player and p["stat"] == "points"]
            if pts_props:
                line = pts_props[0]["line"]
                print(f"\n  PTS line: {line}  →  Model predicts: {pred:.1f}  →  Edge: {pred-line:+.1f}")
                if abs(pred - line) >= 1.5:
                    print(f"  ✓ Edge found!")
                else:
                    print(f"  No edge (threshold is ±1.5)")
    print("\n=== END DIAGNOSTIC ===\n")


def _date_et(iso_str):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York")).date()
    except Exception:
        return None


if __name__ == "__main__":
    main()
