"""
European Football Player Statistics Scraper Module.
Ingests individual player data across top European leagues (EPL, La Liga, Serie A, Bundesliga, Ligue 1)
including appearances, minutes, goals, assists, xG, xA, key passes, xGChain, and xGBuildup.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd

import config
from .base_scraper import BaseScraper
from normalizer import normalize_team_name

logger = logging.getLogger(__name__)

DEFAULT_PLAYERS_OUTPUT = Path("data/players_database.csv")


class PlayerScraper(BaseScraper):
    """
    Ingests and normalizes player-level statistics across top 5 European leagues.
    """

    BASE_URL = "https://understat.com"

    LEAGUE_SLUGS = {
        "EPL": "EPL",
        "La_Liga": "La_liga",
        "Bundesliga": "Bundesliga",
        "Serie_A": "Serie_A",
        "Ligue_1": "Ligue_1"
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fetch_league_players(self, league: str = "EPL", season: str = "2026") -> pd.DataFrame:
        """
        Fetches all player data for a given league and season.

        Args:
            league: Canonical league name ('EPL', 'La_Liga', 'Bundesliga', 'Serie_A', 'Ligue_1').
            season: Season year ('2026' for 2026/2027 ongoing, '2025', '2024', '2023').

        Returns:
            DataFrame containing normalized player statistics.
        """
        slug = self.LEAGUE_SLUGS.get(league, league)
        endpoint = f"{self.BASE_URL}/getLeagueData/{slug}/{season}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{self.BASE_URL}/league/{slug}/{season}"
        }

        logger.info(f"Fetching player statistics for {league} ({season})...")
        resp = self.get(endpoint, headers=headers)
        if not resp or resp.status_code != 200:
            logger.warning(f"Failed to fetch player data for {league} {season} (status: {resp.status_code if resp else 'None'})")
            return pd.DataFrame()

        try:
            data = resp.json()
            raw_players = data.get("players", [])
            if not raw_players:
                logger.warning(f"No players found for {league} {season}")
                return pd.DataFrame()

            cleaned_players = []
            for p in raw_players:
                minutes = int(p.get("time", 0))
                goals = int(p.get("goals", 0))
                assists = int(p.get("assists", 0))
                xg = float(p.get("xG", 0.0))
                xa = float(p.get("xA", 0.0))
                shots = int(p.get("shots", 0))
                key_passes = int(p.get("key_passes", 0))

                # Safe per 90 metrics
                p90_factor = (90.0 / minutes) if minutes >= 45 else 0.0

                cleaned_players.append({
                    "player_id": str(p.get("id")),
                    "player_name": p.get("player_name"),
                    "team": normalize_team_name(p.get("team_title", "")),
                    "league": league,
                    "season": season,
                    "position": p.get("position", "Unknown"),
                    "matches": int(p.get("games", 0)),
                    "minutes": minutes,
                    "goals": goals,
                    "npg": int(p.get("npg", 0)),
                    "assists": assists,
                    "shots": shots,
                    "key_passes": key_passes,
                    "xG": round(xg, 2),
                    "npxG": round(float(p.get("npxG", 0.0)), 2),
                    "xA": round(xa, 2),
                    "xGChain": round(float(p.get("xGChain", 0.0)), 2),
                    "xGBuildup": round(float(p.get("xGBuildup", 0.0)), 2),
                    "xG90": round(xg * p90_factor, 2),
                    "xA90": round(xa * p90_factor, 2),
                    "goals90": round(goals * p90_factor, 2),
                    "assists90": round(assists * p90_factor, 2),
                    "yellow_cards": int(p.get("yellow_cards", 0)),
                    "red_cards": int(p.get("red_cards", 0))
                })

            df_players = pd.DataFrame(cleaned_players)
            logger.info(f"Ingested {len(df_players)} players for {league} {season}.")
            return df_players

        except Exception as e:
            logger.error(f"Error parsing player data for {league} {season}: {e}")
            return pd.DataFrame()

    def scrape_all_european_leagues(
        self,
        leagues: Optional[List[str]] = None,
        season: str = "2026",
        save_path: Optional[Path] = DEFAULT_PLAYERS_OUTPUT
    ) -> pd.DataFrame:
        """
        Scrapes players across all top European flights and saves to CSV.
        """
        target_leagues = leagues or list(self.LEAGUE_SLUGS.keys())
        all_frames = []

        for lg in target_leagues:
            df_lg = self.fetch_league_players(league=lg, season=season)
            if not df_lg.empty:
                all_frames.append(df_lg)

        if not all_frames:
            logger.warning("No player data was ingested.")
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(save_path, index=False)
            logger.info(f"Saved {len(combined)} players to {save_path}")

        return combined
