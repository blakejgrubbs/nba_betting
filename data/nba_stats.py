import time
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# Build team ID ↔ abbreviation mapping from static nba_api data
try:
    from nba_api.stats.static import teams as _nba_teams
    _teams_list = _nba_teams.get_teams()
    TEAM_ID_TO_ABBR = {t["id"]: t["abbreviation"] for t in _teams_list}
    TEAM_ABBR_TO_ID = {t["abbreviation"]: t["id"] for t in _teams_list}
except Exception:
    TEAM_ID_TO_ABBR = {}
    TEAM_ABBR_TO_ID = {}


def _cache_path(name: str) -> Path:
    return config.CACHE_DIR / f"{name}.pkl"


def _load_cache(name: str, max_age_hours: float = 24.0):
    path = _cache_path(name)
    if not path.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    if age > timedelta(hours=max_age_hours):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_cache(name: str, data) -> None:
    with open(_cache_path(name), "wb") as f:
        pickle.dump(data, f)


def fetch_season_games(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Fetch all team-game records for a season. Each game appears twice (one row per team)."""
    cache_key = f"games_{season.replace('-','_')}_{season_type.replace(' ','_')}"
    cached = _load_cache(cache_key, max_age_hours=168)
    if cached is not None:
        return cached

    from nba_api.stats.endpoints import leaguegamefinder
    logger.info(f"Fetching {season} {season_type} games from NBA API...")
    time.sleep(0.6)
    df = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable=season_type,
    ).get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    _save_cache(cache_key, df)
    logger.info(f"  → {len(df) // 2} games cached for {season}")
    return df


def fetch_training_games() -> pd.DataFrame:
    dfs = [fetch_season_games(s) for s in config.TRAINING_SEASONS]
    return pd.concat(dfs, ignore_index=True)


def fetch_season_player_logs(season: str) -> pd.DataFrame:
    """Fetch all player game logs for a season (one row per player per game)."""
    cache_key = f"player_logs_{season.replace('-','_')}"
    cached = _load_cache(cache_key, max_age_hours=168)
    if cached is not None:
        return cached

    from nba_api.stats.endpoints import playergamelogs
    logger.info(f"Fetching {season} player game logs...")
    time.sleep(0.6)
    df = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable="Regular Season",
    ).get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    _save_cache(cache_key, df)
    logger.info(f"  → {len(df)} player-game records cached for {season}")
    return df


def fetch_training_player_logs() -> pd.DataFrame:
    dfs = [fetch_season_player_logs(s) for s in config.TRAINING_SEASONS]
    return pd.concat(dfs, ignore_index=True)


def fetch_current_season_games() -> pd.DataFrame:
    season = config.get_current_season()
    cache_key = f"current_games_{season.replace('-','_')}"
    cached = _load_cache(cache_key, max_age_hours=4)
    if cached is not None:
        return cached
    df = fetch_season_games(season)
    # Also try to grab playoff games for current season
    try:
        playoff_df = fetch_season_games(season, season_type="Playoffs")
        df = pd.concat([df, playoff_df], ignore_index=True)
    except Exception:
        pass
    _save_cache(cache_key, df)
    return df


def fetch_current_season_player_logs() -> pd.DataFrame:
    season = config.get_current_season()
    cache_key = f"current_player_logs_{season.replace('-','_')}"
    cached = _load_cache(cache_key, max_age_hours=4)
    if cached is not None:
        return cached
    df = fetch_season_player_logs(season)
    _save_cache(cache_key, df)
    return df


def fetch_todays_game_schedule() -> list:
    """Return list of today's scheduled games with home/away team abbreviations."""
    cache_key = "todays_schedule"
    cached = _load_cache(cache_key, max_age_hours=1)
    if cached is not None:
        return cached

    from nba_api.stats.endpoints import scoreboardv2
    logger.info("Fetching today's game schedule...")
    time.sleep(0.6)
    board = scoreboardv2.ScoreboardV2()
    header = board.get_data_frames()[0]

    games = []
    for _, row in header.iterrows():
        home_abbr = TEAM_ID_TO_ABBR.get(row["HOME_TEAM_ID"], "")
        away_abbr = TEAM_ID_TO_ABBR.get(row["VISITOR_TEAM_ID"], "")
        if home_abbr and away_abbr:
            games.append({
                "game_id": row["GAME_ID"],
                "game_date": str(row["GAME_DATE_EST"])[:10],
                "game_status": row.get("GAME_STATUS_TEXT", ""),
                "home_team": home_abbr,
                "away_team": away_abbr,
            })

    _save_cache(cache_key, games)
    return games
