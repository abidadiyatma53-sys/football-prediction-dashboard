"""
Prediction Settlement & P&L Reporting CLI.
Reconciles pre-match forecasts and value bets against actual match scores,
updating the SQLite database ledger and printing a financial P&L summary.

Usage:
    # Auto-settle all pending predictions using clean dataset:
    python settle_predictions.py --auto

    # Settle an individual match manually:
    python settle_predictions.py --home "Arsenal" --away "Chelsea" --home-goals 2 --away-goals 1

    # View current P&L ledger statement:
    python settle_predictions.py --report
"""

import argparse
import logging
from tracking_service import PredictionTrackingService

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")


def print_pnl_report(summary: dict):
    print("\n" + "=" * 65)
    print(" REAL-WORLD PREDICTION LEDGER & P&L STATEMENT")
    print("=" * 65)
    print(f" Total Predictions Logged : {summary['total_predictions']}")
    print(f" Settled Matches          : {summary['settled_count']}")
    print(f" Pending Matches          : {summary['pending_count']}")
    print("-" * 65)
    print(f" Total Value Bets Placed  : {summary['total_bets']}")
    print(f" Won Bets                 : {summary['won_bets']}")
    print(f" Lost Bets                : {summary['lost_bets']}")
    print(f" Win Rate (%)             : {summary['win_rate_pct']}%")
    print("-" * 65)
    print(f" Total Staked (Units)     : {summary['total_staked_units']} u")
    pnl_sign = "+" if summary['total_pnl_units'] > 0 else ""
    print(f" Net Realized P&L         : {pnl_sign}{summary['total_pnl_units']} u")
    roi_sign = "+" if summary['roi_pct'] > 0 else ""
    print(f" Return on Investment     : {roi_sign}{summary['roi_pct']}%")
    if summary.get('brier_score') is not None:
        print(f" Realized Brier Score     : {summary['brier_score']:.4f}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Football Prediction Settlement & P&L Engine.")
    parser.add_argument("--auto", action="store_true", help="Auto-settle pending matches against clean data CSV.")
    parser.add_argument("--report", action="store_true", help="Display current P&L ledger summary.")
    parser.add_argument("--home", type=str, help="Home team name for manual settlement.")
    parser.add_argument("--away", type=str, help="Away team name for manual settlement.")
    parser.add_argument("--home-goals", type=int, help="Actual goals scored by home team.")
    parser.add_argument("--away-goals", type=int, help="Actual goals scored by away team.")
    parser.add_argument("--data", type=str, default="football_clean_data.csv", help="Path to clean match data.")

    args = parser.parse_args()
    service = PredictionTrackingService(data_path=args.data)

    if args.auto:
        print("Running automated settlement across all pending matches...")
        res = service.auto_settle_from_dataset()
        print(f"Settled {res['settled_count']} of {res['total_pending_scanned']} pending matches.")
        print_pnl_report(res["ledger_summary"])
        return

    if args.home and args.away and args.home_goals is not None and args.away_goals is not None:
        pending = service.db.get_pending_predictions()
        matched = [p for p in pending if p["home_team"].lower() == args.home.lower() and p["away_team"].lower() == args.away.lower()]
        if not matched:
            print(f"No pending prediction found for {args.home} vs {args.away}.")
            return
        for p in matched:
            service.db.settle_prediction(p["id"], args.home_goals, args.away_goals)
            print(f"Successfully settled prediction ID #{p['id']} for {args.home} {args.home_goals} - {args.away_goals} {args.away}.")
        print_pnl_report(service.get_summary())
        return

    # Default action: print report
    print_pnl_report(service.get_summary())


if __name__ == "__main__":
    main()
