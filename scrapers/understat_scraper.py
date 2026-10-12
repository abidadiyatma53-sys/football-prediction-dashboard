"""
Understat Scraper Module.
Extracts match data, xG (Expected Goals), xGA, and shot metrics from Understat.com.
"""

import re
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import pandas as pd

import config
from .base_scraper import BaseScraper
from normalizer import normalize_team_name

logger = logging.getLogger(__name__)


class UnderstatScraper(BaseScraper):
    """
    Scraper for Understat.com covering European top flights:
    - Ingests fixtures, results, xG, xGA via internal JSON endpoints.
    - Scrapes match pages to retrieve shot volumes (HS, AS) and shots on target (HST, AST).
    """

    BASE_URL = "https://understat.com"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fetch_league_data(self, league: str, season: str) -> Optional[Dict[str, Any]]:
        """
        Fetches the primary league dataset from Understat.

        Args:
            league: League identifier (e.g. 'EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1').
            season: Season year (e.g. '2023', '2022').

        Returns:
            Dictionary containing 'dates', 'teams', 'players' or None.
        """
        endpoint = f"{self.BASE_URL}/getLeagueData/{league}/{season}"
        headers = config.UNDERSTAT_AJAX_HEADERS.copy()
        headers["Referer"] = f"{self.BASE_URL}/league/{league}/{season}"

        logger.info(f"Fetching Understat league data for {league} {season}...")
        return self.get_json(endpoint, headers=headers)

    def fetch_match_details(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Parses an individual match page to extract detailed shot metrics (HS, AS, HST, AST).

        Args:
            match_id: Understat match identifier string.

        Returns:
            Dict containing match info or None if unavailable.
        """
        url = f"{self.BASE_URL}/match/{match_id}"
        resp = self.get(url)
        if not resp or not resp.text:
            return None

        # Extract embedded match_info JSON
        match = re.search(r"JSON\.parse\(\x27([^\x27]+)\x27\)", resp.text)
        if not match:
            return None

        try:
            raw_payload = match.group(1).encode("utf-8").decode("unicode_escape")
            info = json.loads(raw_payload)
            return {
                "HS": int(info["h_shot"]) if info.get("h_shot") is not None else None,
                "AS": int(info["a_shot"]) if info.get("a_shot") is not None else None,
                "HST": int(info["h_shotOnTarget"]) if info.get("h_shotOnTarget") is not None else None,
                "AST": int(info["a_shotOnTarget"]) if info.get("a_shotOnTarget") is not None else None,
                "xG_home": float(info["h_xg"]) if info.get("h_xg") is not None else None,
                "xG_away": float(info["a_xg"]) if info.get("a_xg") is not None else None,
                "h_deep": int(info.get("h_deep", 0)),
                "a_deep": int(info.get("a_deep", 0)),
                "h_ppda": float(info.get("h_ppda", 0.0)),
                "a_ppda": float(info.get("a_ppda", 0.0)),
            }
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            logger.warning(f"Error parsing match details for match {match_id}: {e}")
            return None

    def scrape_season(
        self,
        league: str = "EPL",
        season: str = "2023",
        fetch_shots: bool = True,
        max_matches: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Scrapes and aggregates historical match data for a given league and season.

        Args:
            league: League name (e.g. 'EPL', 'La_Liga', 'Bundesliga').
            season: Season year string (e.g. '2023').
            fetch_shots: Whether to scrape individual match pages for HS, AS, HST, AST.
            max_matches: Optional limit on the number of matches (useful for quick testing).

        Returns:
            DataFrame containing standardized historical match statistics.
        """
        understat_league = config.LEAGUE_MAPPING.get(league, {}).get("understat", league)
        league_data = self.fetch_league_data(understat_league, season)

        if not league_data or "dates" not in league_data:
            logger.error(f"No match dates data found for {league} {season}")
            return pd.DataFrame()

        raw_matches = league_data.get("dates", [])
        # Filter completed matches only
        finished_matches = [
            m for m in raw_matches
            if (
                m.get("isResult") is True
                and m.get("goals", {}).get("h") is not None
                and m.get("goals", {}).get("a") is not None
            )
        ]

        if max_matches:
            finished_matches = finished_matches[:max_matches]

        logger.info(f"Found {len(finished_matches)} completed matches for {league} {season}")

        # Base match records
        records = []
        for m in finished_matches:
            try:
                fthg = int(m.get("goals", {}).get("h", 0))
                ftag = int(m.get("goals", {}).get("a", 0))
                ftr = "H" if fthg > ftag else ("A" if ftag > fthg else "D")

                records.append({
                    "match_id": str(m.get("id")),
                    "Date": m.get("datetime", "")[:10],
                    "League": league,
                    "Season": season,
                    "HomeTeam": normalize_team_name(m.get("h", {}).get("title")),
                    "AwayTeam": normalize_team_name(m.get("a", {}).get("title")),
                    "FTHG": fthg,
                    "FTAG": ftag,
                    "FTR": ftr,
                    "HS": None,
                    "AS": None,
                    "HST": None,
                    "AST": None,
                    "xG_home": (
                        round(float(m["xG"]["h"]), 4)
                        if m.get("xG", {}).get("h") is not None else None
                    ),
                    "xG_away": (
                        round(float(m["xG"]["a"]), 4)
                        if m.get("xG", {}).get("a") is not None else None
                    ),
                    "_Source": "Understat",
                })
            except Exception as parse_err:
                logger.warning(f"Error parsing date match: {parse_err}")

        # Enrich with shot statistics (HS, AS, HST, AST) if requested
        if fetch_shots and records:
            logger.info(f"Enriching {len(records)} matches with shots data from match pages...")
            match_stats_map = {}

            with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_WORKERS) as executor:
                future_to_id = {
                    executor.submit(self.fetch_match_details, rec["match_id"]): rec["match_id"]
                    for rec in records
                }
                for future in as_completed(future_to_id):
                    m_id = future_to_id[future]
                    try:
                        stats = future.result()
                        if stats:
                            match_stats_map[m_id] = stats
                    except Exception as exc:
                        logger.warning(f"Failed fetching stats for match {m_id}: {exc}")

            # Merge enriched stats back into records
            for rec in records:
                m_id = rec["match_id"]
                if m_id in match_stats_map:
                    s = match_stats_map[m_id]
                    rec["HS"] = s["HS"]
                    rec["AS"] = s["AS"]
                    rec["HST"] = s["HST"]
                    rec["AST"] = s["AST"]
                    if s.get("xG_home"):
                        rec["xG_home"] = round(s["xG_home"], 4)
                    if s.get("xG_away"):
                        rec["xG_away"] = round(s["xG_away"], 4)

        df = pd.DataFrame(records)
        if "match_id" in df.columns:
            df = df.drop(columns=["match_id"])

        return df

    def fetch_upcoming_fixtures(
        self,
        league: str = "EPL",
        season: str = "2026"
    ) -> List[Dict[str, Any]]:
        """
        Fetches upcoming unplayed fixtures for a given European league from Understat.

        Args:
            league: League code (e.g. 'EPL', 'La_Liga', 'Bundesliga', 'Serie_A', 'Ligue_1', 'RFPL').
            season: Season year string (e.g. '2026').

        Returns:
            List of standardized fixture dictionaries.
        """
        understat_league = config.LEAGUE_MAPPING.get(league, {}).get("understat", league)
        league_display = config.LEAGUE_MAPPING.get(league, {}).get("name", league)
        league_data = self.fetch_league_data(understat_league, season)
        if not league_data or "dates" not in league_data:
            return []

        raw_matches = league_data.get("dates", [])
        upcoming = [m for m in raw_matches if not m.get("isResult")]

        fixtures = []
        for m in upcoming:
            h_raw = m.get("h", {}).get("title", "Unknown")
            a_raw = m.get("a", {}).get("title", "Unknown")
            dt_str = m.get("datetime", "")
            fixtures.append({
                "match_id": str(m.get("id")),
                "league": league,
                "league_name": league_display,
                "season": season,
                "datetime": dt_str,
                "date": dt_str[:10] if dt_str else "",
                "time": dt_str[11:16] if len(dt_str) >= 16 else "",
                "home_team": normalize_team_name(h_raw),
                "away_team": normalize_team_name(a_raw),
                "home_raw": h_raw,
                "away_raw": a_raw,
            })
        return fixtures

    def fetch_all_upcoming_fixtures(
        self,
        leagues: Optional[List[str]] = None,
        season: str = "2026"
    ) -> List[Dict[str, Any]]:
        """
        Fetches upcoming fixtures across all European leagues and sorts chronologically.
        """
        target_leagues = leagues or list(config.LEAGUE_MAPPING.keys())
        all_fixtures = []
        for lg in target_leagues:
            try:
                fixs = self.fetch_upcoming_fixtures(lg, season=season)
                all_fixtures.extend(fixs)
            except Exception as e:
                logger.warning(f"Error fetching upcoming fixtures for {lg}: {e}")
        all_fixtures.sort(key=lambda x: x.get("datetime", ""))
        return all_fixtures
