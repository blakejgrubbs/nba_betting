import pickle
import logging
from datetime import datetime, timedelta, date
from pathlib import Path

import requests
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _cache_path(name: str) -> Path:
    return config.CACHE_DIR / f"{name}.pkl"


def _load_cache(name: str, max_age_hours: float = 168.0):
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


def fetch_city_weather_range(team_abbr: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily max temperature and precipitation for a team's city over a date range."""
    cache_key = f"weather_{team_abbr}_{start_date}_{end_date}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    if team_abbr not in config.TEAM_CITIES:
        return pd.DataFrame(columns=["date", "temp_max", "precip"])

    lat, lon = config.TEAM_CITIES[team_abbr]
    today = date.today().isoformat()
    url = ARCHIVE_URL if end_date < today else FORECAST_URL

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,precipitation_sum",
        "timezone": "America/New_York",
        "start_date": start_date,
        "end_date": end_date,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        df = pd.DataFrame({
            "date": pd.to_datetime(daily.get("time", [])).date,
            "temp_max": daily.get("temperature_2m_max", []),
            "precip": daily.get("precipitation_sum", []),
        })
        _save_cache(cache_key, df)
        return df
    except Exception as e:
        logger.warning(f"Weather fetch failed for {team_abbr}: {e}")
        return pd.DataFrame(columns=["date", "temp_max", "precip"])


def fetch_all_training_weather() -> dict:
    """Fetch historical weather for all 30 cities covering training seasons (one call each)."""
    start = "2022-10-01"
    end = "2025-06-30"
    result = {}
    for abbr in config.TEAM_CITIES:
        result[abbr] = fetch_city_weather_range(abbr, start, end)
    return result


def fetch_todays_weather() -> dict:
    """Return {team_abbr: {temp_max, precip}} for today."""
    today = date.today().isoformat()
    result = {}
    for abbr in config.TEAM_CITIES:
        df = fetch_city_weather_range(abbr, today, today)
        if not df.empty:
            row = df.iloc[0]
            result[abbr] = {"temp_max": float(row.get("temp_max") or 70.0),
                            "precip": float(row.get("precip") or 0.0)}
        else:
            result[abbr] = {"temp_max": 70.0, "precip": 0.0}
    return result


def lookup_game_weather(team_abbr: str, game_date_str: str, weather_data: dict) -> tuple:
    """Look up (temp_max, precip) for a game. Returns defaults if not found."""
    city_df = weather_data.get(team_abbr, pd.DataFrame())
    if isinstance(city_df, pd.DataFrame) and not city_df.empty:
        target = pd.to_datetime(game_date_str).date()
        match = city_df[city_df["date"] == target]
        if not match.empty:
            return float(match.iloc[0]["temp_max"] or 70.0), float(match.iloc[0]["precip"] or 0.0)
    return 70.0, 0.0
