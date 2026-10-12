"""
Main Execution Script for Football Data Pipeline.
Provides command-line arguments to trigger extraction, transformation, and CSV output.

Example Usage:
    # Full Premier League 2023-2024 with xG and Shots:
    python main.py --leagues EPL --seasons 2023

    # Quick test (first 10 matches):
    python main.py --leagues EPL --seasons 2023 --limit 10

    # Multi-league extraction:
    python main.py --leagues EPL La_Liga Bundesliga --seasons 2023
"""

import sys
import argparse
import logging
from pathlib import Path

import config
from pipeline import FootballDataPipeline


def setup_logging(verbose: bool = False):
    """Configures structured console logging with timestamp formatting."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_args():
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Football Data Ingestion Pipeline for Score Prediction."
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=["EPL"],
        choices=list(config.LEAGUE_MAPPING.keys()),
        help="List of leagues to ingest (default: EPL)."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2023"],
        help="List of 4-digit seasons to extract (default: 2023)."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["understat", "football_data"],
        choices=["understat", "football_data"],
        help="Data sources to ingest from (default: understat football_data)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.OUTPUT_FILE,
        help="Path for clean output CSV (default: football_clean_data.csv)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of matches per league (useful for quick verification)."
    )
    parser.add_argument(
        "--no-shots",
        action="store_true",
        help="Skip scraping individual match detail pages for shot statistics (faster)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    pipeline = FootballDataPipeline(
        output_path=args.output,
        enable_shots=not args.no_shots,
        max_matches_per_league=args.limit
    )

    clean_df = pipeline.run(
        leagues=args.leagues,
        seasons=args.seasons,
        sources=args.sources
    )

    if clean_df.empty:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
