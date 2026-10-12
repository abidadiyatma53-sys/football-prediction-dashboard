"""
Football-Data.co.uk Ingestion Module.
Downloads and parses structured historical match CSVs covering top European leagues.
Handles network timeouts, local caching, and mirror fallbacks.
"""

import io
import logging
from pathlib import Path
from typing import Optional, List
import pandas as pd

import config
from .base_scraper import BaseScraper
from normalizer import normalize_team_name

logger = logging.getLogger(__name__)


class FootballDataScraper(BaseScraper):
    """
    Ingestor for Football-Data.co.uk datasets:
    - Provides official match results and basic in-game statistics (goals, shots, shots on target).
    - Caches raw CSVs locally to avoid unnecessary network bandwidth and handle offline scenarios.
    """

    BASE_URL = "https://www.football-data.co.uk/mmz4281"

    def __init__(self, raw_cache_dir: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self.raw_cache_dir = raw_cache_dir or config.RAW_DATA_DIR
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_season_code(season_year: str) -> str:
        """
        Converts 4-digit year into Football-Data 4-digit season code.
        Example: '2023' -> '2324', '2022' -> '2223'.
        """
        year_int = int(season_year)
        y1 = str(year_int)[-2:]
        y2 = str(year_int + 1)[-2:]
        return f"{y1}{y2}"

    def download_csv(self, div_code: str, season_code: str) -> Optional[str]:
        """
        Downloads CSV text from Football-Data.co.uk or loads from local cache.

        Args:
            div_code: Division code (e.g. 'E0', 'SP1', 'D1', 'I1', 'F1').
            season_code: Season representation (e.g. '2324').

        Returns:
            CSV content as string, or None if download failed.
        """
        cache_path = self.raw_cache_dir / f"{div_code}_{season_code}.csv"

        # Check local cache or any matching local file in data/raw/
        if cache_path.exists():
            logger.info(f"Loading cached file: {cache_path}")
            try:
                return cache_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"Could not read cached file {cache_path}: {e}")

        # Check for alternative naming conventions in raw directory (e.g., E0.csv, 2324.csv)
        alt_matches = list(self.raw_cache_dir.glob(f"*{div_code}*{season_code}*.csv"))
        if alt_matches:
            logger.info(f"Found local CSV matching {div_code}: {alt_matches[0]}")
            try:
                return alt_matches[0].read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"Could not read local file {alt_matches[0]}: {e}")

        # Attempt direct download
        url = f"{self.BASE_URL}/{season_code}/{div_code}.csv"
        logger.info(f"Downloading from {url}...")
        resp = self.get(url, timeout=5)

        if resp and resp.status_code == 200:
            content = resp.text
            # Save to cache
            try:
                cache_path.write_text(content, encoding="utf-8", errors="replace")
                logger.info(f"Saved raw CSV to cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Could not write cache file {cache_path}: {e}")
            return content

        logger.warning(f"Direct download failed for {url}. Checking if local file is provided.")
        return None

    def parse_csv_data(
        self,
        csv_text: str,
        league_name: str,
        season_year: str
    ) -> pd.DataFrame:
        """
        Parses raw Football-Data CSV string into a standardized DataFrame.

        Args:
            csv_text: Raw CSV string.
            league_name: Standardized league name (e.g. 'EPL').
            season_year: Season string (e.g. '2023').

        Returns:
            Standardized DataFrame with normalized team names.
        """
        try:
            df_raw = pd.read_csv(io.StringIO(csv_text), on_bad_lines="skip")
        except Exception as e:
            logger.error(f"Error parsing CSV content: {e}")
            return pd.DataFrame()

        # Check essential columns
        required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        missing = [c for c in required if c not in df_raw.columns]
        if missing:
            logger.error(f"CSV missing mandatory columns: {missing}")
            return pd.DataFrame()

        # Reject unfinished/malformed scores instead of converting them to 0-0.
        df = df_raw.copy()
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna(subset=required).copy()

        # Standardize team names
        df["HomeTeam"] = df["HomeTeam"].astype(str).apply(normalize_team_name)
        df["AwayTeam"] = df["AwayTeam"].astype(str).apply(normalize_team_name)

        # Standardize shots columns (HS, AS, HST, AST)
        for col in ["HS", "AS", "HST", "AST"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            else:
                df[col] = pd.Series(pd.NA, index=df.index, dtype="Int64")

        # Goals & Results
        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)

        if "FTR" not in df.columns:
            df["FTR"] = df.apply(
                lambda r: "H" if r["FTHG"] > r["FTAG"] else ("A" if r["FTAG"] > r["FTHG"] else "D"),
                axis=1
            )

        # Standardize Date format to YYYY-MM-DD
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        df["League"] = league_name
        df["Season"] = season_year

        # Placeholder for xG (Football-Data doesn't natively provide xG)
        df["xG_home"] = None
        df["xG_away"] = None
        df["_Source"] = "Football-Data.co.uk"

        # Select standard columns
        cols = [c for c in config.TARGET_COLUMNS if c in df.columns]
        cols.append("_Source")
        return df[cols]

    def scrape_season(
        self,
        league: str = "EPL",
        season: str = "2023"
    ) -> pd.DataFrame:
        """
        High-level wrapper to ingest a league season from Football-Data.co.uk.

        Args:
            league: League name (e.g. 'EPL', 'La_Liga').
            season: 4-digit season year (e.g. '2023').

        Returns:
            Standardized DataFrame of match results.
        """
        league_conf = config.LEAGUE_MAPPING.get(league, {})
        div_code = league_conf.get("football_data", "E0")
        season_code = self.get_season_code(season)

        csv_text = self.download_csv(div_code, season_code)
        if not csv_text:
            logger.warning(
                f"Could not retrieve Football-Data CSV for {league} ({div_code}) season {season_code}."
            )
            return pd.DataFrame()

        return self.parse_csv_data(csv_text, league, season)
