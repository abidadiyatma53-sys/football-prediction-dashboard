"""
Prediction Tracking & Automated Settlement Service.
Handles logging pre-kickoff predictions, calculating +EV bet recommendations,
and reconciling finished match results from historical/live datasets to track real-world P&L.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

from normalizer import normalize_team_name
from predictor import MatchPredictorService
from value_bet import ValueBettingEngine
from tracker_db import PredictionDatabase

logger = logging.getLogger(__name__)


class PredictionTrackingService:
    """
    Service coordinating the entire prediction lifecycle:
    1. Pre-Match: Predict match outcome, identify +EV bets, log to SQLite ledger.
    2. Post-Match: Settle pending predictions against actual match scores, update P&L and ROI.
    """

    def __init__(self, db_path: Optional[Path] = None, data_path: str = "football_clean_data.csv"):
        self.db = PredictionDatabase(db_path=db_path) if db_path else PredictionDatabase()
        self.data_path = data_path
        self._predictor: Optional[MatchPredictorService] = None
        self.value_engine = ValueBettingEngine(default_kelly_fraction=0.25)

    @property
    def predictor(self) -> MatchPredictorService:
        if self._predictor is None:
            self._predictor = MatchPredictorService(data_path=self.data_path)
        return self._predictor

    def predict_and_log(
        self,
        home_team: str,
        away_team: str,
        market_odds: Optional[Dict[str, float]] = None,
        method: str = "ml",
        stake_units: float = 1.0,
        match_date: Optional[str] = None,
        skip_if_pending: bool = False
    ) -> Dict[str, Any]:
        """
        Executes prediction, analyzes value bets against market odds,
        and saves an immutable record into the SQLite ledger.
        """
        h_norm = normalize_team_name(home_team)
        a_norm = normalize_team_name(away_team)
        date_str = match_date or pd.Timestamp.now().strftime("%Y-%m-%d")

        if skip_if_pending and self.db.has_pending_prediction(h_norm, a_norm, date_str):
            logger.info(f"Skipping duplicate prediction for {h_norm} vs {a_norm} ({date_str}) - already pending.")
            return {
                "prediction_id": None,
                "skipped": True,
                "match": f"{h_norm} vs {a_norm}",
                "reason": "Already pending in ledger"
            }

        pred = self.predictor.predict_match(
            h_norm, a_norm, method=method, display=False, match_date=date_str
        )
        probs = pred["probabilities_3way"]

        bet_selection = None
        bet_odds = None
        bet_ev = None
        recommended_stake = stake_units

        # If market odds supplied, identify if there is a +EV value bet
        if market_odds:
            analysis = self.value_engine.analyze_market_odds(probs, market_odds)
            value_bets = [item for item in analysis if item["is_value_bet"]]
            if value_bets:
                top_bet = value_bets[0]
                bet_selection = top_bet["outcome"]
                bet_odds = top_bet["market_odds"]
                bet_ev = top_bet["ev_pct"]
                # Convert Kelly percentage to unit multiplier (e.g. 5% = 1.5 units, 2.5% = 1.0 unit)
                recommended_stake = round(max(0.5, (top_bet["recommended_stake_pct"] / 2.5) * stake_units), 2)

        match_timestamp = pd.Timestamp(date_str)
        record = {
            "match_date": date_str,
            "league": pred.get("home_league") or pred.get("away_league") or "Unknown",
            "season": str(match_timestamp.year if match_timestamp.month >= 7 else match_timestamp.year - 1),
            "home_team": h_norm,
            "away_team": a_norm,
            "model_method": pred["method"],
            "prob_home": probs["home_win"],
            "prob_draw": probs["draw"],
            "prob_away": probs["away_win"],
            "lambda_home": pred["lambda_home"],
            "lambda_away": pred["lambda_away"],
            "predicted_score": pred["most_likely_score"],
            "predicted_score_prob": pred["most_likely_probability"],
            "market_odds_home": market_odds.get("home_win") if market_odds else None,
            "market_odds_draw": market_odds.get("draw") if market_odds else None,
            "market_odds_away": market_odds.get("away_win") if market_odds else None,
            "bet_selection": bet_selection,
            "bet_odds": bet_odds,
            "bet_ev_pct": bet_ev,
            "bet_stake_units": recommended_stake
        }

        row_id = self.db.log_prediction(record)
        logger.info(f"Logged prediction #{row_id} for {h_norm} vs {a_norm} (Bet: {bet_selection or 'None'})")

        clean_pred = dict(pred)
        if "score_matrix" in clean_pred and hasattr(clean_pred["score_matrix"], "tolist"):
            clean_pred["score_matrix"] = clean_pred["score_matrix"].tolist()

        return {
            "prediction_id": row_id,
            "match": f"{h_norm} vs {a_norm}",
            "prediction": clean_pred,
            "bet_details": {
                "selection": bet_selection,
                "odds": bet_odds,
                "ev_pct": bet_ev,
                "stake_units": recommended_stake
            }
        }

    def auto_settle_from_dataset(self, csv_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Scans all PENDING predictions in the SQLite ledger and reconciles them
        against completed match results found in the clean dataset.
        """
        source_csv = csv_path or self.data_path
        if not Path(source_csv).exists():
            return {"status": "error", "message": f"Dataset {source_csv} not found."}

        df_matches = pd.read_csv(source_csv)
        if not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(df_matches.columns):
            raise ValueError(f"Dataset {source_csv} is missing columns required for settlement.")
        df_matches["Date"] = pd.to_datetime(df_matches["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        pending = self.db.get_pending_predictions()

        settled_count = 0
        total_pending = len(pending)

        for p in pending:
            h = p["home_team"]
            a = p["away_team"]
            match_date = p.get("match_date")
            if not match_date:
                logger.warning("Skipping prediction %s: ledger record has no match date.", p["id"])
                continue

            # Match on date as well as teams; the same fixture can recur each season.
            match_row = df_matches[
                (df_matches["Date"] == match_date)
                & (df_matches["HomeTeam"] == h)
                & (df_matches["AwayTeam"] == a)
            ]

            if len(match_row) == 1:
                row = match_row.iloc[0]
                if self.db.settle_prediction(
                    prediction_id=p["id"],
                    actual_home_goals=int(row["FTHG"]),
                    actual_away_goals=int(row["FTAG"])
                ):
                    settled_count += 1
            elif len(match_row) > 1:
                logger.warning(
                    "Skipping prediction %s: multiple dataset rows match %s vs %s on %s.",
                    p["id"], h, a, match_date
                )

        summary = self.db.get_pnl_summary()
        return {
            "status": "success",
            "total_pending_scanned": total_pending,
            "settled_count": settled_count,
            "remaining_pending": total_pending - settled_count,
            "ledger_summary": summary
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns aggregate financial metrics."""
        return self.db.get_pnl_summary()

    def get_history_dataframe(self, limit: int = 100) -> pd.DataFrame:
        """Returns recent predictions history."""
        return self.db.get_all_predictions(limit=limit)
