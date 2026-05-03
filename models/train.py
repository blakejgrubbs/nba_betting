"""
Run once to train all models and save them to saved_models/.
Usage: python main.py train
Training takes 5-20 minutes on first run (fetching NBA API data).
Subsequent runs use cached data and complete in ~2 minutes.
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRegressor
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.nba_stats import fetch_training_games, fetch_training_player_logs
from data.weather import fetch_all_training_weather
from features.team_features import (
    build_game_dataframe, add_rolling_stats, add_h2h_features,
    add_weather_features, build_feature_matrix,
)
from features.player_features import build_player_feature_matrix

logger = logging.getLogger(__name__)

XGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
}


def _save(name: str, obj) -> None:
    path = config.MODELS_DIR / f"{name}.pkl"
    joblib.dump(obj, path)
    logger.info(f"  Saved → {path.name}")


def train_all_models() -> None:
    print("\n=== NBA BETTING MODEL TRAINING ===\n")

    # ── 1. Fetch data ──────────────────────────────────────────────────────────
    print("Step 1/5  Fetching game logs (cached after first run)...")
    raw_games = fetch_training_games()
    print(f"         {len(raw_games) // 2:,} games loaded")

    print("Step 2/5  Fetching player logs...")
    raw_player_logs = fetch_training_player_logs()
    print(f"         {len(raw_player_logs):,} player-game records loaded")

    print("Step 3/5  Fetching historical weather (30 cities)...")
    weather_data = fetch_all_training_weather()
    print("         Weather data cached")

    # ── 2. Build team feature matrix ───────────────────────────────────────────
    print("Step 4/5  Engineering features...")
    games_df = build_game_dataframe(raw_games)
    print(f"         {len(games_df):,} game rows constructed")

    games_df = add_rolling_stats(games_df, raw_games)
    print("         Rolling stats added")

    print("         Computing head-to-head features (this takes ~1 min)...")
    games_df = add_h2h_features(games_df)

    games_df = add_weather_features(games_df, weather_data)
    print("         Weather features added")

    X_team, y_win, y_margin, team_feat_cols = build_feature_matrix(games_df)
    print(f"         Team feature matrix: {X_team.shape[0]} samples × {X_team.shape[1]} features")

    # Drop rows where targets are NaN (games not yet played shouldn't appear but guard anyway)
    valid = y_win.notna() & y_margin.notna()
    X_team = X_team[valid].reset_index(drop=True)
    y_win = y_win[valid].reset_index(drop=True)
    y_margin = y_margin[valid].reset_index(drop=True)

    # ── 3. Train moneyline model ───────────────────────────────────────────────
    print("\nStep 5/5  Training models...")
    print("  [1/6] Moneyline model (win/loss classifier)...")
    base_clf = XGBClassifier(eval_metric="logloss", **XGB_PARAMS)
    ml_model = CalibratedClassifierCV(base_clf, cv=5, method="isotonic")
    ml_model.fit(X_team, y_win)
    cv_scores = cross_val_score(base_clf, X_team, y_win, cv=5, scoring="roc_auc")
    print(f"         ROC-AUC (5-fold CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    _save("moneyline_model", ml_model)
    _save("team_feature_cols", team_feat_cols)

    # ── 4. Train spread model ──────────────────────────────────────────────────
    print("  [2/6] Spread model (point margin regressor)...")
    spread_model = XGBRegressor(**XGB_PARAMS)
    spread_model.fit(X_team, y_margin)
    cv_mae = -cross_val_score(spread_model, X_team, y_margin, cv=5, scoring="neg_mean_absolute_error")
    print(f"         MAE (5-fold CV): {cv_mae.mean():.2f} pts ± {cv_mae.std():.2f}")
    _save("spread_model", spread_model)

    # ── 5. Build player feature matrix ────────────────────────────────────────
    print("  [3-6/6] Player props models...")
    X_player, targets, player_feat_cols = build_player_feature_matrix(raw_player_logs, raw_games)
    print(f"         Player feature matrix: {X_player.shape[0]} samples × {X_player.shape[1]} features")
    _save("player_feature_cols", player_feat_cols)

    stat_labels = {"PTS": "Points", "REB": "Rebounds", "AST": "Assists", "FG3M": "3-Pointers"}
    for col, label in stat_labels.items():
        if col not in targets:
            continue
        y = targets[col].fillna(0)
        valid_mask = y > 0
        Xv, yv = X_player[valid_mask], y[valid_mask]
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(Xv, yv)
        cv_mae = -cross_val_score(model, Xv, yv, cv=5, scoring="neg_mean_absolute_error")
        print(f"         {label}: MAE = {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")
        _save(f"props_{col.lower()}_model", model)

    print("\n✓ All models trained and saved to saved_models/")
    print("  Run:  python main.py predict\n")
