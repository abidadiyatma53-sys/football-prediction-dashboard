"""
CLI Execution Script for Football Match Score Prediction.
Allows predicting any matchup using either Poisson or XGBoost ML approaches.

Usage Examples:
    python run_prediction.py --home "Arsenal" --away "Chelsea"
    python run_prediction.py --home "Man City" --away "Liverpool" --method ml
    python run_prediction.py --home "Aston Villa" --away "Tottenham" --compare
"""

import sys
import argparse
from predictor import MatchPredictorService


def main():
    parser = argparse.ArgumentParser(
        description="Football Match Score & Outcome Predictor (Poisson & XGBoost ML)."
    )
    parser.add_argument(
        "--home",
        type=str,
        default="Arsenal",
        help="Name of Home Team (default: 'Arsenal')."
    )
    parser.add_argument(
        "--away",
        type=str,
        default="Chelsea",
        help="Name of Away Team (default: 'Chelsea')."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ml",
        choices=["poisson", "ml", "both"],
        help="Modeling method to use: 'ml' (XGBoost), 'poisson', or 'both' (default: ml)."
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run and display both Poisson and Machine Learning models for comparison."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="football_clean_data.csv",
        help="Path to clean match data CSV (default: 'football_clean_data.csv')."
    )

    args = parser.parse_args()

    service = MatchPredictorService(data_path=args.data)

    if args.compare or args.method == "both":
        print("\n" + "#" * 65)
        print(" METHOD 1: POISSON DISTRIBUTION GOAL EXPECTANCY")
        print("#" * 65)
        service.predict_match(args.home, args.away, method="poisson", display=True)

        print("\n" + "#" * 65)
        print(" METHOD 2: MACHINE LEARNING (XGBOOST MULTI-OUTPUT)")
        print("#" * 65)
        service.predict_match(args.home, args.away, method="ml", display=True)
    else:
        service.predict_match(args.home, args.away, method=args.method, display=True)


if __name__ == "__main__":
    main()
