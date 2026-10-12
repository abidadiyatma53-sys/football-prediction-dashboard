"""
Configuration module for Football Data Pipeline.
Centralizes constants, league mappings, request headers, and ETL settings.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_FILE = BASE_DIR / "football_clean_data.csv"

# League mappings across different data sources
# Key: standardized league name
# Value: source-specific codes
LEAGUE_MAPPING = {
    "EPL": {
        "understat": "EPL",
        "football_data": "E0",
        "name": "English Premier League"
    },
    "La_Liga": {
        "understat": "La_liga",
        "football_data": "SP1",
        "name": "Spanish La Liga"
    },
    "Bundesliga": {
        "understat": "Bundesliga",
        "football_data": "D1",
        "name": "German Bundesliga"
    },
    "Serie_A": {
        "understat": "Serie_A",
        "football_data": "I1",
        "name": "Italian Serie A"
    },
    "Ligue_1": {
        "understat": "Ligue_1",
        "football_data": "F1",
        "name": "French Ligue 1"
    },
    "RFPL": {
        "understat": "RFPL",
        "football_data": "RUS",
        "name": "Russian Premier League"
    }
}

# Network and Rate Limiting Settings
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
}

UNDERSTAT_AJAX_HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

# Retry & Rate Limiting Constants
REQUEST_TIMEOUT = 12          # seconds
MAX_RETRIES = 3              # retry attempts on failure
BACKOFF_FACTOR = 1.5         # exponential backoff multiplier
REQUEST_DELAY = 0.4          # polite delay between requests in seconds
MAX_CONCURRENT_WORKERS = 4   # thread pool workers for match details

# Target Unified Data Schema
TARGET_COLUMNS = [
    "Date",
    "League",
    "Season",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "xG_home",
    "xG_away",
    "ResultSource",
    "xGSource",
    "StatsSource"
]
