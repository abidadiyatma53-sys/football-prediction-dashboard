"""
Full European Football Data Ingestion & Upcoming Fixture Predictor.
Scrapes historical data and upcoming fixtures across all top European leagues
(Premier League, La Liga, Bundesliga, Serie A, Ligue 1, and RFPL),
standardizes data, trains prediction models, and precomputes scoreline forecasts.
"""

import json
import hashlib
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

import config
from scrapers.understat_scraper import UnderstatScraper
from cleaner import DataCleaner
from value_bet import ValueBettingEngine

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("EuropeanDataPipeline")

LEAGUES = ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1", "RFPL"]
SEASONS = ["2023", "2024", "2025", "2026"]
OUTPUT_FIXTURES_PATH = config.DATA_DIR / "upcoming_european_fixtures.json"


def ingest_all_european_leagues() -> pd.DataFrame:
    """Scrapes completed match records across all European leagues."""
    logger.info("Ingesting match records across European top flights...")
    scraper = UnderstatScraper()
    all_frames = []
    missing_feeds = []

    for lg in LEAGUES:
        for s in SEASONS:
            try:
                df = scraper.scrape_season(league=lg, season=s, fetch_shots=False)
                if not df.empty:
                    logger.info(f"Retrieved {len(df)} matches for {lg} {s}")
                    all_frames.append(df)
                else:
                    missing_feeds.append(f"{lg} {s}")
            except Exception as e:
                logger.error(f"Failed scraping {lg} {s}: {e}")
                missing_feeds.append(f"{lg} {s}")

    if missing_feeds:
        raise RuntimeError(
            "Refusing to replace the existing dataset because these Understat feeds "
            f"were empty or unavailable: {', '.join(missing_feeds)}"
        )

    combined = pd.concat(all_frames, ignore_index=True)
    cleaner = DataCleaner()
    cleaned = cleaner.clean_dataset(combined)
    cleaned.to_csv(config.OUTPUT_FILE, index=False)
    logger.info(f"Saved {len(cleaned)} cleaned European match records to {config.OUTPUT_FILE}")
    return cleaned


def ingest_and_predict_upcoming_fixtures(df_clean: pd.DataFrame) -> List[Dict[str, Any]]:
    """Fetches upcoming fixtures for all leagues and generates model predictions."""
    logger.info("Fetching upcoming fixtures across all European leagues...")
    scraper = UnderstatScraper()
    raw_fixtures = scraper.fetch_all_upcoming_fixtures(leagues=LEAGUES, season="2026")
    logger.info(f"Found {len(raw_fixtures)} total upcoming European fixtures.")

    from predictor import MatchPredictorService
    service = MatchPredictorService(data_path=str(config.OUTPUT_FILE))
    data_sha256 = hashlib.sha256(config.OUTPUT_FILE.read_bytes()).hexdigest()
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    model_data_through = str(service.df_raw["Date"].max())

    predicted_fixtures = []
    prediction_rows = []
    value_engine = ValueBettingEngine()
    for fix in raw_fixtures:
        h = fix["home_team"]
        a = fix["away_team"]
        try:
            pred = service.predict_match(
                h, a, method="ml", display=False, match_date=fix.get("date")
            )
            probs = pred["probabilities_3way"]
            top_scores = pred.get("top_exact_scores", [])

            fix_record = {
                **fix,
                "prediction_generated_at_utc": generated_at,
                "model_data_through": model_data_through,
                "training_data_sha256": data_sha256,
                "prediction_method": pred["method"],
                "predicted_score": pred["most_likely_score"],
                "score_prob_pct": pred["most_likely_probability"],
                "lambda_home": pred["lambda_home"],
                "lambda_away": pred["lambda_away"],
                "home_win_prob": probs["home_win"],
                "draw_prob": probs["draw"],
                "away_win_prob": probs["away_win"],
                "home_elo": pred.get("home_elo", 1500.0),
                "away_elo": pred.get("away_elo", 1500.0),
                "diff_elo": pred.get("diff_elo", 0.0),
                "top_scorelines": top_scores[:3] if top_scores else []
            }
            predicted_fixtures.append(fix_record)
            prediction_rows.append({
                "Generated_At_UTC": generated_at,
                "Model_Data_Through": model_data_through,
                "Training_Data_SHA256": data_sha256,
                "Fixture_ID": fix.get("match_id"),
                "Date": fix.get("date"),
                "League": fix.get("league"),
                "HomeTeam": pred["home_team"],
                "AwayTeam": pred["away_team"],
                "Method": pred["method"],
                "Expected_Goals_H": pred["lambda_home"],
                "Expected_Goals_A": pred["lambda_away"],
                "Most_Likely_Score": pred["most_likely_score"],
                "Score_Prob_Pct": pred["most_likely_probability"],
                "Prob_Home_Win_Pct": probs["home_win"],
                "Prob_Draw_Pct": probs["draw"],
                "Prob_Away_Win_Pct": probs["away_win"],
                "Fair_Odds_Home": value_engine.calculate_fair_odds(probs["home_win"] / 100.0),
                "Fair_Odds_Draw": value_engine.calculate_fair_odds(probs["draw"] / 100.0),
                "Fair_Odds_Away": value_engine.calculate_fair_odds(probs["away_win"] / 100.0),
                "Market_Odds_Home": None,
                "Market_Odds_Draw": None,
                "Market_Odds_Away": None,
                "Best_Value_Bet": "N/A",
                "Edge_Pct": "N/A"
            })
        except Exception as e:
            logger.warning(f"Error predicting fixture {h} vs {a}: {e}")
            predicted_fixtures.append({**fix, "prediction_error": str(e)})

    failed_predictions = [
        fix for fix in predicted_fixtures if "prediction_error" in fix
    ]
    if failed_predictions:
        raise RuntimeError(
            f"Refusing to replace forecast files: {len(failed_predictions)} "
            f"of {len(predicted_fixtures)} fixture predictions failed."
        )
    if not prediction_rows:
        raise RuntimeError("No upcoming fixtures were available to predict.")

    # Save to JSON
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FIXTURES_PATH, "w", encoding="utf-8") as f:
        json.dump(predicted_fixtures, f, indent=2, ensure_ascii=False)

    predictions_csv_path = config.DATA_DIR / "latest_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_csv_path, index=False)
    logger.info(f"Saved {len(predicted_fixtures)} predicted upcoming fixtures to {OUTPUT_FIXTURES_PATH}")
    logger.info(f"Saved {len(prediction_rows)} auditable predictions to {predictions_csv_path}")
    return predicted_fixtures


def main():
    t0 = time.time()
    logger.info("=== STARTING EUROPEAN DATA PIPELINE & FORECASTER ===")
    df_clean = ingest_all_european_leagues()
    fixtures = ingest_and_predict_upcoming_fixtures(df_clean)
    elapsed = time.time() - t0
    logger.info(f"=== PIPELINE COMPLETED IN {elapsed:.1f} SECONDS ===")
    logger.info(f"Clubs covered: {df_clean['HomeTeam'].nunique()} | Upcoming fixtures: {len(fixtures)}")


if __name__ == "__main__":
    main()
