import pickle
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"


def _cache_path(name: str) -> Path:
    return config.CACHE_DIR / f"{name}.pkl"


def _load_cache(name: str, max_age_hours: float = 1.0):
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


def american_to_prob(odds: float) -> float:
    """Convert American odds to implied win probability (no vig)."""
    if odds is None:
        return 0.5
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _best_line(bookmakers: list, market_key: str, team_name: str):
    """Find the best price for a team across all bookmakers for a given market."""
    best_price = None
    best_point = None
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market["key"] != market_key:
                continue
            for outcome in market.get("outcomes", []):
                if outcome["name"] == team_name:
                    price = outcome.get("price")
                    point = outcome.get("point")
                    if best_price is None or price > best_price:
                        best_price = price
                        best_point = point
    return best_price, best_point


def fetch_game_odds() -> list:
    """Fetch moneyline + spread for all upcoming NBA games."""
    cached = _load_cache("game_odds")
    if cached is not None:
        return cached

    if not config.ODDS_API_KEY:
        logger.warning("No ODDS_API_KEY set — skipping odds fetch. Copy .env.example to .env and add your key.")
        return []

    params = {
        "apiKey": config.ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(f"{BASE_URL}/sports/{SPORT}/odds", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch game odds: {e}")
        return []

    games = []
    for g in data:
        home_full = g["home_team"]
        away_full = g["away_team"]
        home = config.TEAM_NAME_TO_ABBR.get(home_full, home_full)
        away = config.TEAM_NAME_TO_ABBR.get(away_full, away_full)

        ml_home, _ = _best_line(g.get("bookmakers", []), "h2h", home_full)
        ml_away, _ = _best_line(g.get("bookmakers", []), "h2h", away_full)
        spread_home_odds, spread_home = _best_line(g.get("bookmakers", []), "spreads", home_full)
        spread_away_odds, spread_away = _best_line(g.get("bookmakers", []), "spreads", away_full)

        games.append({
            "event_id": g["id"],
            "commence_time": g["commence_time"],
            "home_team": home,
            "away_team": away,
            "home_team_full": home_full,
            "away_team_full": away_full,
            "ml_home": ml_home,
            "ml_away": ml_away,
            "spread_home": spread_home,
            "spread_home_odds": spread_home_odds,
            "spread_away": spread_away,
            "spread_away_odds": spread_away_odds,
        })

    _save_cache("game_odds", games)
    logger.info(f"Fetched odds for {len(games)} NBA games")
    return games


def fetch_player_prop_odds() -> list:
    """Fetch player prop over/unders for tonight's games."""
    cached = _load_cache("player_props")
    if cached is not None:
        return cached

    if not config.ODDS_API_KEY:
        return []

    game_odds = fetch_game_odds()
    if not game_odds:
        return []

    prop_markets = "player_points,player_rebounds,player_assists,player_threes"
    all_props = []

    for game in game_odds:
        event_id = game["event_id"]
        url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
        params = {
            "apiKey": config.ODDS_API_KEY,
            "regions": "us",
            "markets": prop_markets,
            "oddsFormat": "american",
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            event_data = resp.json()
        except Exception as e:
            logger.warning(f"Props fetch failed for {event_id}: {e}")
            continue

        seen = {}  # (player, stat, direction) → best odds entry
        for bm in event_data.get("bookmakers", []):
            for market in bm.get("markets", []):
                stat = market["key"].replace("player_", "")
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description", "")
                    direction = outcome.get("name", "")  # "Over" or "Under"
                    line = outcome.get("point")
                    odds = outcome.get("price")
                    if not player or line is None:
                        continue
                    key = (player, stat, direction)
                    if key not in seen or (odds and seen[key]["odds"] < odds):
                        seen[key] = {
                            "event_id": event_id,
                            "home_team": game["home_team"],
                            "away_team": game["away_team"],
                            "player_name": player,
                            "stat": stat,
                            "direction": direction,
                            "line": line,
                            "odds": odds,
                        }
        all_props.extend(seen.values())
        time.sleep(0.3)

    _save_cache("player_props", all_props)
    logger.info(f"Fetched {len(all_props)} player prop lines")
    return all_props
