import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.weather import lookup_game_weather

STAT_COLS = ["PTS", "FG_PCT", "FG3_PCT", "REB", "AST", "TOV", "PLUS_MINUS", "WL_NUM"]
WINDOWS = [5, 10, 20]


def build_game_dataframe(raw_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw game logs (2 rows per game) into 1 row per game
    with home_* and away_* columns.
    """
    df = raw_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WL_NUM"] = (df["WL"] == "W").astype(int)
    df["is_home"] = df["MATCHUP"].str.contains(r"vs\.", na=False)
    df["PTS_ALLOWED"] = df["PTS"] - df["PLUS_MINUS"]

    keep = ["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "SEASON_ID",
            "PTS", "PTS_ALLOWED", "FG_PCT", "FG3_PCT", "REB", "AST",
            "TOV", "PLUS_MINUS", "WL_NUM", "is_home"]

    home = df[df["is_home"]][keep].rename(
        columns={c: f"home_{c}" for c in keep if c != "GAME_ID"}
    )
    away = df[~df["is_home"]][keep].rename(
        columns={c: f"away_{c}" for c in keep if c != "GAME_ID"}
    )

    games = home.merge(away, on="GAME_ID")
    games["home_margin"] = games["home_PTS"] - games["away_PTS"]
    games["home_win"] = (games["home_margin"] > 0).astype(int)
    games["game_date"] = games["home_GAME_DATE"]
    games["day_of_week"] = games["game_date"].dt.dayofweek
    games["is_weekend"] = games["day_of_week"].isin([4, 5, 6]).astype(int)

    return games.sort_values("game_date").reset_index(drop=True)


def _compute_team_rolling(raw_logs: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team rolling stats with shift(1) to prevent lookahead."""
    df = raw_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WL_NUM"] = (df["WL"] == "W").astype(int)
    df["PTS_ALLOWED"] = df["PTS"] - df["PLUS_MINUS"]
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

    all_stats = STAT_COLS + ["PTS_ALLOWED"]
    for col in all_stats:
        if col not in df.columns:
            continue
        for w in WINDOWS:
            df[f"{col.lower()}_avg_{w}"] = df.groupby("TEAM_ABBREVIATION")[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )

    # Rest days (capped at 10 to avoid season-start outliers)
    df["prev_game"] = df.groupby("TEAM_ABBREVIATION")["GAME_DATE"].shift(1)
    df["rest_days"] = (df["GAME_DATE"] - df["prev_game"]).dt.days.fillna(7).clip(upper=10)
    df["is_back_to_back"] = (df["rest_days"] == 1).astype(int)

    # Win/loss streak: positive = consecutive wins, negative = consecutive losses
    def _streak(series: pd.Series) -> pd.Series:
        out, cur = [], 0
        for v in series.shift(1):
            if pd.isna(v):
                cur = 0
            elif v == 1:
                cur = cur + 1 if cur > 0 else 1
            else:
                cur = cur - 1 if cur < 0 else -1
            out.append(cur)
        return pd.Series(out, index=series.index)

    df["streak"] = df.groupby("TEAM_ABBREVIATION")["WL_NUM"].transform(_streak)

    return df


def add_rolling_stats(games_df: pd.DataFrame, raw_logs: pd.DataFrame) -> pd.DataFrame:
    """Merge per-team rolling stats into the games DataFrame."""
    team_df = _compute_team_rolling(raw_logs)

    feature_cols = (
        [f"{c.lower()}_avg_{w}" for c in (STAT_COLS + ["PTS_ALLOWED"]) for w in WINDOWS
         if f"{c.lower()}_avg_{w}" in team_df.columns]
        + ["rest_days", "is_back_to_back", "streak"]
    )

    result = games_df.copy()
    for side in ("home", "away"):
        side_features = team_df[["GAME_ID", "TEAM_ABBREVIATION"] + feature_cols].rename(
            columns={"TEAM_ABBREVIATION": f"{side}_TEAM_ABBREVIATION",
                     **{c: f"{side}_{c}" for c in feature_cols}}
        )
        result = result.merge(side_features, on=["GAME_ID", f"{side}_TEAM_ABBREVIATION"], how="left")

    return result


def add_h2h_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """Add head-to-head win rate and average margin (home team perspective, last 10 meetings)."""
    df = games_df.sort_values("game_date").reset_index(drop=True)
    h2h_win_pcts, h2h_margins = [], []

    for idx in range(len(df)):
        row = df.iloc[idx]
        home, away = row["home_TEAM_ABBREVIATION"], row["away_TEAM_ABBREVIATION"]
        past = df.iloc[:idx]

        mask = (
            ((past["home_TEAM_ABBREVIATION"] == home) & (past["away_TEAM_ABBREVIATION"] == away)) |
            ((past["home_TEAM_ABBREVIATION"] == away) & (past["away_TEAM_ABBREVIATION"] == home))
        )
        h2h = past[mask].tail(10)

        if len(h2h) == 0:
            h2h_win_pcts.append(0.5)
            h2h_margins.append(0.0)
        else:
            wins = (
                ((h2h["home_TEAM_ABBREVIATION"] == home) & (h2h["home_win"] == 1)) |
                ((h2h["away_TEAM_ABBREVIATION"] == home) & (h2h["home_win"] == 0))
            ).sum()
            margins = np.where(
                h2h["home_TEAM_ABBREVIATION"].values == home,
                h2h["home_margin"].values,
                -h2h["home_margin"].values,
            )
            h2h_win_pcts.append(wins / len(h2h))
            h2h_margins.append(float(np.mean(margins)))

    df["h2h_home_win_pct"] = h2h_win_pcts
    df["h2h_home_margin_avg"] = h2h_margins
    return df


def add_weather_features(games_df: pd.DataFrame, weather_data: dict) -> pd.DataFrame:
    df = games_df.copy()
    temps, precips = [], []
    for _, row in df.iterrows():
        team = row.get("home_TEAM_ABBREVIATION", "")
        date_str = str(row["game_date"])[:10]
        t, p = lookup_game_weather(team, date_str, weather_data)
        temps.append(t)
        precips.append(p)
    df["weather_temp"] = temps
    df["weather_precip"] = precips
    return df


def get_feature_cols() -> list:
    """Return the ordered list of feature column names used for model training/prediction."""
    base = [f"{c.lower()}_avg_{w}" for c in (STAT_COLS + ["PTS_ALLOWED"]) for w in WINDOWS]
    extra = ["rest_days", "is_back_to_back", "streak"]
    team_feats = [f"{side}_{f}" for side in ("home", "away") for f in base + extra]
    game_feats = [
        "h2h_home_win_pct", "h2h_home_margin_avg",
        "day_of_week", "is_weekend",
        "weather_temp", "weather_precip",
    ]
    return team_feats + game_feats


def build_feature_matrix(games_df: pd.DataFrame):
    """Return (X DataFrame, y_win Series, y_margin Series)."""
    feat_cols = [c for c in get_feature_cols() if c in games_df.columns]
    X = games_df[feat_cols].copy()
    X = X.fillna(X.mean(numeric_only=True))
    y_win = games_df["home_win"]
    y_margin = games_df["home_margin"]
    return X, y_win, y_margin, feat_cols


def build_prediction_row(home_team: str, away_team: str,
                          current_games: pd.DataFrame, weather: dict,
                          game_date_str: str = None) -> dict:
    """
    Build a single feature row for a tonight's matchup.
    current_games: raw game logs for the current season.
    """
    from datetime import date as _date

    if game_date_str is None:
        game_date_str = _date.today().isoformat()

    def team_stats(abbr: str) -> dict:
        logs = current_games[current_games["TEAM_ABBREVIATION"] == abbr].copy()
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
        logs = logs.sort_values("GAME_DATE")
        logs["WL_NUM"] = (logs["WL"] == "W").astype(int)
        logs["PTS_ALLOWED"] = logs["PTS"] - logs["PLUS_MINUS"]

        row = {}
        for col in STAT_COLS + ["PTS_ALLOWED"]:
            if col not in logs.columns:
                continue
            for w in WINDOWS:
                vals = logs[col].tail(w)
                row[f"{col.lower()}_avg_{w}"] = float(vals.mean()) if len(vals) > 0 else 0.0

        if len(logs) >= 2:
            last_date = logs["GAME_DATE"].iloc[-1]
            prev_date = logs["GAME_DATE"].iloc[-2]
            row["rest_days"] = min(10, (pd.Timestamp(game_date_str) - last_date).days)
            row["is_back_to_back"] = int(row["rest_days"] == 1)
        else:
            row["rest_days"] = 3
            row["is_back_to_back"] = 0

        # Streak from most recent games
        recent = logs["WL_NUM"].tail(10).tolist()
        streak = 0
        for w in reversed(recent):
            if w == 1:
                streak = streak + 1 if streak >= 0 else 1
            else:
                streak = streak - 1 if streak <= 0 else -1
        row["streak"] = streak
        return row

    home_stats = team_stats(home_team)
    away_stats = team_stats(away_team)

    result = {}
    for k, v in home_stats.items():
        result[f"home_{k}"] = v
    for k, v in away_stats.items():
        result[f"away_{k}"] = v

    # H2H: look through current season game logs for past meetings
    # Use simplified 0.5 / 0.0 defaults since we're predicting from current season only
    result["h2h_home_win_pct"] = 0.5
    result["h2h_home_margin_avg"] = 0.0

    game_dt = pd.Timestamp(game_date_str)
    result["day_of_week"] = game_dt.dayofweek
    result["is_weekend"] = int(game_dt.dayofweek in [4, 5, 6])

    t, p = lookup_game_weather(home_team, game_date_str, weather)
    result["weather_temp"] = t
    result["weather_precip"] = p

    return result
