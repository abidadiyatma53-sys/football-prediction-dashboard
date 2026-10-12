"""
Live Odds Synchronization & Value Bet Scanner CLI.
Pulls real-time upcoming matchday fixtures and bookmaker odds,
detects Positive EV opportunities using Dixon-Coles Poisson,
and optionally commits +EV bets directly to the SQLite tracking ledger.

Usage:
    # Scan live odds and display terminal report:
    python sync_live_odds.py

    # Scan and automatically commit all +EV bets to the database:
    python sync_live_odds.py --commit-to-db
"""

import sys
import logging
import argparse
from live_odds import LiveOddsSyncClient, OddsSourceUnavailable
from tracking_service import PredictionTrackingService

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("LiveOddsSync")


def main():
    parser = argparse.ArgumentParser(description="Live Odds & Value Bet Scanner.")
    parser.add_argument("--commit-to-db", action="store_true", help="Automatically commit +EV bets into SQLite ledger.")
    parser.add_argument("--min-ev", type=float, default=0.0, help="Minimum EV percentage threshold for committing bets (default: 0.0%).")
    parser.add_argument("--api-key", type=str, default=None, help="The Odds API Key (optional).")
    parser.add_argument("--method", type=str, default="ml", choices=["poisson", "ml"], help="Model engine.")
    parser.add_argument("--force", action="store_true", help="Force commit even if match is already pending.")

    args = parser.parse_args()

    client = LiveOddsSyncClient(api_key=args.api_key)
    logger.info("Fetching upcoming fixtures and live bookmaker odds...")
    try:
        fixtures = client.fetch_live_fixtures_and_odds()
    except OddsSourceUnavailable as error:
        logger.error("%s", error)
        return 2

    logger.info(f"Analyzing {len(fixtures)} upcoming fixtures for +EV value opportunities...")
    scan_results = client.scan_for_value_bets(fixtures, method=args.method)

    print("\n" + "=" * 95)
    print(" LIVE MATCHDAY ODDS & POSITIVE EV (+EV) SCANNER REPORT")
    print("=" * 95)
    header = "{:<28} | {:<12} | {:<10} | {:<7} | {:<9} | {:<14}"
    print(header.format("Fixture", "Bookmaker", "Top Pick", "Odds", "EV (%)", "Rec. Stake (Kelly)"))
    print("-" * 95)

    tracking_svc = PredictionTrackingService() if args.commit_to_db else None
    committed_count = 0
    skipped_count = 0

    for item in scan_results:
        bb = item["best_bet"]
        fixture_lbl = item["fixture"][:28]
        bm_lbl = item["bookmaker"][:12]

        if bb:
            pick_lbl = bb["label"][:10]
            odds_lbl = f"{bb['market_odds']:.2f}"
            ev_lbl = f"+{bb['ev_pct']:.1f}%"
            stake_lbl = f"{bb['recommended_stake_pct']}% bankroll"

            if args.commit_to_db and tracking_svc and bb["ev_pct"] >= args.min_ev:
                log_res = tracking_svc.predict_and_log(
                    home_team=item["home_team"],
                    away_team=item["away_team"],
                    market_odds=item["market_odds"],
                    method=args.method,
                    stake_units=1.0,
                    match_date=item.get("date", "")[:10],
                    skip_if_pending=not args.force
                )
                if log_res.get("skipped"):
                    skipped_count += 1
                else:
                    committed_count += 1
        else:
            pick_lbl = "No Value"
            odds_lbl = "-"
            ev_lbl = "-"
            stake_lbl = "0.0%"

        print(header.format(fixture_lbl, bm_lbl, pick_lbl, odds_lbl, ev_lbl, stake_lbl))

    print("=" * 95)

    if args.commit_to_db:
        print(f"SUCCESS: Committed {committed_count} +EV predictions (Skipped existing pending: {skipped_count}) into data/predictions_ledger.db\n")
    else:
        print("Tip: Run with --commit-to-db to automatically record all +EV bets to the SQLite audit ledger.\n")


if __name__ == "__main__":
    raise SystemExit(main())
