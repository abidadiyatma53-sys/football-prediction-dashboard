"""
Automated Matchday Inference Script.
Batch predicts upcoming football matches for a scheduled matchday,
detects Positive EV opportunities against market odds,
and exports structured forecast outputs for tracking and deployment.

Usage:
    python run_matchday_forecast.py
    python run_matchday_forecast.py --fixtures my_fixtures.json --method poisson
"""

import sys
import json
import hashlib
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from normalizer import normalize_team_name
from predictor import MatchPredictorService
from value_bet import ValueBettingEngine

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MatchdayInference")

def run_batch_inference(
    fixtures: List[Dict[str, Any]],
    method: str = "ml",
    output_path: Path = Path("data/latest_predictions.csv"),
    data_path: str = "football_clean_data.csv"
) -> pd.DataFrame:
    """
    Executes automated batch predictions across all fixtures.
    """
    logger.info(f"Loading prediction service with dataset '{data_path}'...")
    service = MatchPredictorService(data_path=data_path)
    data_sha256 = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    model_data_through = str(service.df_raw["Date"].max())
    engine = ValueBettingEngine(default_kelly_fraction=0.25)

    records = []
    logger.info(f"Computing forecasts for {len(fixtures)} upcoming fixtures (Method: {method})...")

    for fix in fixtures:
        home = fix["home"]
        away = fix["away"]
        try:
            pred = service.predict_match(
                home,
                away,
                method=method,
                display=False,
                match_date=fix.get("date") or None
            )
            probs = pred["probabilities_3way"]

            # Evaluate market odds if provided
            has_odds = "odds_home" in fix and "odds_draw" in fix and "odds_away" in fix
            best_value_info = "N/A"
            edge_summary = "N/A"
            fair_h = engine.calculate_fair_odds(probs["home_win"] / 100.0)
            fair_d = engine.calculate_fair_odds(probs["draw"] / 100.0)
            fair_a = engine.calculate_fair_odds(probs["away_win"] / 100.0)

            if has_odds:
                market_dict = {
                    "home_win": fix["odds_home"],
                    "draw": fix["odds_draw"],
                    "away_win": fix["odds_away"]
                }
                analysis = engine.analyze_market_odds(probs, market_dict)
                value_bets = [item for item in analysis if item["is_value_bet"]]
                if value_bets:
                    top_v = value_bets[0]
                    best_value_info = f"{top_v['label']} (EV +{top_v['ev_pct']}%, Rec: {top_v['recommended_stake_pct']}%)"
                    edge_summary = f"+{top_v['edge_pct']}%"

            records.append({
                "Generated_At_UTC": generated_at,
                "Model_Data_Through": model_data_through,
                "Training_Data_SHA256": data_sha256,
                "Fixture_ID": fix.get("match_id", fix.get("id")),
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
                "Fair_Odds_Home": fair_h,
                "Fair_Odds_Draw": fair_d,
                "Fair_Odds_Away": fair_a,
                "Market_Odds_Home": fix.get("odds_home", None),
                "Market_Odds_Draw": fix.get("odds_draw", None),
                "Market_Odds_Away": fix.get("odds_away", None),
                "Best_Value_Bet": best_value_info,
                "Edge_Pct": edge_summary
            })
        except Exception as e:
            logger.warning(f"Skipping match {home} vs {away}: {e}")

    df_out = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    logger.info(f"Successfully generated {len(df_out)} predictions. Saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(" AUTOMATED MATCHDAY FORECAST & VALUE BETTING SUMMARY")
    print("=" * 90)
    format_str = "{:<28} | {:<7} | {:<7} | {:<7} | {:<10} | {:<22}"
    print(format_str.format("Fixture", "H Win %", "Draw %", "A Win %", "Top Score", "Value Bet Recommendation"))
    print("-" * 90)

    for _, r in df_out.iterrows():
        fixture_name = f"{r['HomeTeam']} vs {r['AwayTeam']}"
        print(format_str.format(
            fixture_name[:28],
            f"{r['Prob_Home_Win_Pct']}%",
            f"{r['Prob_Draw_Pct']}%",
            f"{r['Prob_Away_Win_Pct']}%",
            f"{r['Most_Likely_Score']}",
            f"{r['Best_Value_Bet'][:22]}"
        ))
    print("=" * 90 + "\n")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Automated Matchday Inference Script.")
    parser.add_argument(
        "--fixtures",
        type=str,
        default=None,
        help="Path to JSON file containing list of fixture dicts."
    )
    parser.add_argument(
        "--league",
        type=str,
        default=None,
        help="Filter fixtures by league (e.g. EPL, La_Liga, Bundesliga, Serie_A, Ligue_1, RFPL)."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ml",
        choices=["poisson", "ml"],
        help="Prediction model: 'ml' or 'poisson' (default: ml)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/latest_predictions.csv"),
        help="Output CSV path (default: data/latest_predictions.csv)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of fixtures to forecast (default: all)."
    )

    args = parser.parse_args()

    if args.fixtures:
        with open(args.fixtures, "r") as f:
            fixtures = json.load(f)
    else:
        euro_path = Path("data/upcoming_european_fixtures.json")
        if euro_path.exists():
            with open(euro_path, "r", encoding="utf-8") as f:
                raw_euro = json.load(f)
                fixtures = [
                    {
                        "home": fx["home_team"],
                        "away": fx["away_team"],
                        "match_id": fx.get("match_id"),
                        "league": fx.get("league", "EPL"),
                        "date": fx.get("date", "")
                    }
                    for fx in raw_euro
                ]
        else:
            raise FileNotFoundError(
                "No real fixture source found. Provide --fixtures or run "
                "update_all_european_data.py to fetch current fixtures."
            )

    if args.league:
        target = args.league.strip().lower().replace("_", "")
        fixtures = [fx for fx in fixtures if fx.get("league", "").lower().replace("_", "") == target]
        logger.info(f"Filtered to {len(fixtures)} fixtures for league: {args.league}")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer.")
    selected_fixtures = fixtures[:args.limit] if args.limit else fixtures
    run_batch_inference(selected_fixtures, method=args.method, output_path=args.output)


if __name__ == "__main__":
    main()
