import os
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "saved_models"

CACHE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

TRAINING_SEASONS = ["2022-23", "2023-24", "2024-25"]


def get_current_season() -> str:
    today = date.today()
    if today.month >= 10:
        return f"{today.year}-{str(today.year + 1)[2:]}"
    return f"{today.year - 1}-{str(today.year)[2:]}"


MIN_EDGE_MONEYLINE = 0.04   # 4% implied probability edge to flag a pick
MIN_EDGE_SPREAD = 2.0       # 2-point margin edge
MIN_EDGE_PROPS = 1.5        # 1.5-unit stat edge

TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}
TEAM_ABBR_TO_NAME = {v: k for k, v in TEAM_NAME_TO_ABBR.items()}

# (latitude, longitude) for each team's home arena city
TEAM_CITIES = {
    "ATL": (33.749, -84.388), "BOS": (42.360, -71.059), "BKN": (40.678, -73.944),
    "CHA": (35.227, -80.843), "CHI": (41.878, -87.630), "CLE": (41.499, -81.695),
    "DAL": (32.779, -96.808), "DEN": (39.739, -104.984), "DET": (42.331, -83.046),
    "GSW": (37.774, -122.419), "HOU": (29.760, -95.370), "IND": (39.768, -86.158),
    "LAC": (34.052, -118.244), "LAL": (34.052, -118.244), "MEM": (35.149, -90.048),
    "MIA": (25.775, -80.209), "MIL": (43.044, -87.907), "MIN": (44.977, -93.265),
    "NOP": (29.951, -90.072), "NYK": (40.712, -74.006), "OKC": (35.467, -97.516),
    "ORL": (28.538, -81.379), "PHI": (39.953, -75.165), "PHX": (33.448, -112.074),
    "POR": (45.523, -122.676), "SAC": (38.582, -121.494), "SAS": (29.425, -98.494),
    "TOR": (43.651, -79.347), "UTA": (40.761, -111.891), "WAS": (38.907, -77.037),
}
