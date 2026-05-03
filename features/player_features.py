import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PLAYER_STAT_COLS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "MIN"]
PROP_TARGETS = ["PTS", "REB", "AST", "FG3M"]
WINDOWS = [5, 10, 20]


def _compute_player_rolling(player_logs: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling averages per player with shift(1) to prevent lookahead."""
    df = player_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    df["WL_NUM"] = (df["WL"] == "W").astype(int)
    df["is_home"] = df["MATCHUP"].str.contains(r"vs\.", na=False).astype(int)

    for col in PLAYER_STAT_COLS:
        if col not in df.columns:
            continue
        for w in WINDOWS:
            df[f"{col.lower()}_avg_{w}"] = df.groupby("PLAYER_ID")[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )

    # Season average up to (but not including) the current game
    for col in PLAYER_STAT_COLS:
        if col not in df.columns:
            continue
        df[f"{col.lower()}_avg_season"] = df.groupby("PLAYER_ID")[col].transform(
            lambda x: x.shift(1).expanding().mean()
        )

    # Rest days
    df["prev_game"] = df.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    df["rest_days"] = (df["GAME_DATE"] - df["prev_game"]).dt.days.fillna(7).clip(upper=10)

    # Hot/cold trend: (5-game avg - 20-game avg) / 20-game avg
    for col in PROP_TARGETS:
        c = col.lower()
        if f"{c}_avg_5" in df.columns and f"{c}_avg_20" in df.columns:
            denom = df[f"{c}_avg_20"].replace(0, np.nan)
            df[f"{c}_trend"] = (df[f"{c}_avg_5"] - df[f"{c}_avg_20"]) / denom

    return df


def build_player_feature_matrix(player_logs: pd.DataFrame, team_games: pd.DataFrame):
    """
    Build feature matrix for training player props models.
    Returns (X, targets_dict, feature_cols).
    targets_dict = {"PTS": Series, "REB": Series, "AST": Series, "FG3M": Series}
    """
    df = _compute_player_rolling(player_logs)

    # Add opponent defensive stats: avg pts_allowed by opponent team
    team_def = (
        team_games.copy()
        .assign(PTS_ALLOWED=lambda x: x["PTS"] - x["PLUS_MINUS"])
    )
    team_def_avg = (
        team_def.groupby(["TEAM_ABBREVIATION", "GAME_DATE"])["PTS_ALLOWED"]
        .mean()
        .reset_index()
        .sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    )
    team_def_avg["opp_pts_allowed_avg_10"] = team_def_avg.groupby("TEAM_ABBREVIATION")["PTS_ALLOWED"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    # Extract opponent abbreviation from MATCHUP ("PLAYER @ OPP" or "PLAYER vs. OPP")
    df["opp_abbr"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s+([A-Z]+)")[0]

    # Merge opponent defensive rating
    opp_lookup = team_def_avg[["TEAM_ABBREVIATION", "GAME_DATE", "opp_pts_allowed_avg_10"]].rename(
        columns={"TEAM_ABBREVIATION": "opp_abbr"}
    )
    df = df.merge(opp_lookup, on=["opp_abbr", "GAME_DATE"], how="left")

    feat_cols = get_player_feature_cols(df)
    X = df[feat_cols].fillna(df[feat_cols].mean(numeric_only=True))
    targets = {col: df[col] for col in PROP_TARGETS if col in df.columns}

    return X, targets, feat_cols


def get_player_feature_cols(df: pd.DataFrame) -> list:
    """Return available player feature columns."""
    candidates = (
        [f"{c.lower()}_avg_{w}" for c in PLAYER_STAT_COLS for w in WINDOWS]
        + [f"{c.lower()}_avg_season" for c in PLAYER_STAT_COLS]
        + [f"{c.lower()}_trend" for c in PROP_TARGETS]
        + ["rest_days", "is_home", "opp_pts_allowed_avg_10"]
    )
    return [c for c in candidates if c in df.columns]


def build_player_prediction_row(player_name: str, is_home: bool,
                                 opp_team: str, player_logs: pd.DataFrame,
                                 team_games: pd.DataFrame) -> dict:
    """Build feature row for a single player's upcoming game."""
    logs = player_logs[player_logs["PLAYER_NAME"] == player_name].copy()
    if logs.empty:
        return {}

    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs = logs.sort_values("GAME_DATE").reset_index(drop=True)

    row = {"is_home": int(is_home)}

    for col in PLAYER_STAT_COLS:
        if col not in logs.columns:
            continue
        vals = logs[col]
        for w in WINDOWS:
            row[f"{col.lower()}_avg_{w}"] = float(vals.tail(w).mean()) if len(vals) > 0 else 0.0
        row[f"{col.lower()}_avg_season"] = float(vals.mean()) if len(vals) > 0 else 0.0

    # Hot/cold trend
    for col in PROP_TARGETS:
        c = col.lower()
        avg5 = row.get(f"{c}_avg_5", 0.0)
        avg20 = row.get(f"{c}_avg_20", 1.0)
        row[f"{c}_trend"] = (avg5 - avg20) / avg20 if avg20 != 0 else 0.0

    # Rest days
    if len(logs) > 0:
        from datetime import date
        last = logs["GAME_DATE"].iloc[-1]
        row["rest_days"] = min(10, (pd.Timestamp(date.today()) - last).days)
    else:
        row["rest_days"] = 3

    # Opponent defensive quality
    opp_games = team_games[team_games["TEAM_ABBREVIATION"] == opp_team].copy()
    if not opp_games.empty:
        opp_games["PTS_ALLOWED"] = opp_games["PTS"] - opp_games["PLUS_MINUS"]
        row["opp_pts_allowed_avg_10"] = float(opp_games["PTS_ALLOWED"].tail(10).mean())
    else:
        row["opp_pts_allowed_avg_10"] = 110.0  # league average fallback

    return row
