"""
Football Data Pipeline Orchestrator.
Coordinates scraping from Understat and Football-Data.co.uk, executes cleaning,
and outputs a unified modeling-ready dataset to CSV.
"""

import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

import config
from scrapers.understat_scraper import UnderstatScraper
from scrapers.footballdata_scraper import FootballDataScraper
from cleaner import DataCleaner

logger = logging.getLogger(__name__)


class FootballDataPipeline:
    """
    End-to-end ETL Orchestrator:
    1. Extract: Scrapes/ingests historical football data from Understat and/or Football-Data.co.uk.
    2. Transform: Validates, deduplicates, imputes, normalizes team names, and casts datatypes.
    3. Load: Saves high-fidelity cleaned data to 'football_clean_data.csv'.
    """

    def __init__(
        self,
        output_path: Optional[Path] = None,
        enable_shots: bool = True,
        max_matches_per_league: Optional[int] = None
    ):
        self.output_path = output_path or config.OUTPUT_FILE
        self.enable_shots = enable_shots
        self.max_matches_per_league = max_matches_per_league
        self.cleaner = DataCleaner()

    def run(
        self,
        leagues: Optional[List[str]] = None,
        seasons: Optional[List[str]] = None,
        sources: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Executes the ETL pipeline across requested leagues and seasons.

        Args:
            leagues: List of league codes (e.g. ['EPL', 'La_Liga', 'Bundesliga']).
            seasons: List of 4-digit seasons (e.g. ['2023', '2022']).
            sources: Data sources to ingest from (['understat', 'football_data']).

        Returns:
            The final cleaned DataFrame.
        """
        target_leagues = leagues or ["EPL"]
        target_seasons = seasons or ["2023"]
        target_sources = sources or ["understat", "football_data"]

        logger.info("=" * 60)
        logger.info("STARTING FOOTBALL DATA INGESTION PIPELINE")
        logger.info(f"Target Leagues: {target_leagues}")
        logger.info(f"Target Seasons: {target_seasons}")
        logger.info(f"Sources: {target_sources}")
        logger.info("=" * 60)

        raw_frames = []

        # 1. Ingest from Understat (Provides xG, xGA, and Shots)
        if "understat" in target_sources:
            logger.info("--- Starting Understat Extraction ---")
            with UnderstatScraper() as scraper:
                for l in target_leagues:
                    for s in target_seasons:
                        try:
                            df_u = scraper.scrape_season(
                                league=l,
                                season=s,
                                fetch_shots=self.enable_shots,
                                max_matches=self.max_matches_per_league
                            )
                            if not df_u.empty:
                                logger.info(f"Retrieved {len(df_u)} matches from Understat for {l} {s}.")
                                raw_frames.append(df_u)
                        except Exception as e:
                            logger.error(f"Failed scraping Understat for {l} {s}: {e}")

        # 2. Ingest from Football-Data.co.uk (Provides Goals, Shots, Betting/Ref metrics)
        if "football_data" in target_sources:
            logger.info("--- Starting Football-Data.co.uk Ingestion ---")
            with FootballDataScraper() as scraper:
                for l in target_leagues:
                    for s in target_seasons:
                        try:
                            df_fd = scraper.scrape_season(league=l, season=s)
                            if not df_fd.empty:
                                logger.info(f"Retrieved {len(df_fd)} matches from Football-Data for {l} {s}.")
                                raw_frames.append(df_fd)
                        except Exception as e:
                            logger.error(f"Failed ingesting Football-Data for {l} {s}: {e}")

        # Combine raw extractions
        if not raw_frames:
            logger.error("No data extracted from any source. Aborting pipeline.")
            return pd.DataFrame()

        combined_raw = pd.concat(raw_frames, ignore_index=True)
        logger.info(f"Total raw records gathered across all sources: {len(combined_raw)}")

        # 3. Transform & Clean
        clean_df = self.cleaner.clean_dataset(combined_raw)

        # 4. Load & Persist
        self.save_output(clean_df)

        self._print_pipeline_summary(clean_df)
        return clean_df

    def save_output(self, df: pd.DataFrame):
        """Saves clean data to output CSV."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False, encoding="utf-8")
        logger.info(f"SUCCESS: Clean dataset saved to {self.output_path} ({len(df)} rows)")

    def _print_pipeline_summary(self, df: pd.DataFrame):
        """Displays data health report."""
        if df.empty:
            logger.warning("Cleaned DataFrame is empty.")
            return

        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total Clean Records : {len(df):,}")
        print(f"Output File Location: {self.output_path}")
        print(f"Date Coverage Range : {df['Date'].min()} to {df['Date'].max()}")
        print(f"Leagues Ingested    : {df['League'].unique().tolist()}")
        print(f"Unique Teams Count  : {df['HomeTeam'].nunique()}")
        print("\nMissing Values Count:")
        print(df.isnull().sum())
        print("\nSample Cleaned Data (First 5 Rows):")
        print(df.head())
        print("=" * 60 + "\n")
