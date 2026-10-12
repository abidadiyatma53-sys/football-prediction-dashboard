"""
Match Prediction Service & CLI Interface.
Provides the predict_match(home_team, away_team) function to generate match forecasts
using either Poisson Distribution or Machine Learning (XGBoost).
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd

from normalizer import normalize_team_name
from features import FootballFeatureEngineer
from models.poisson_model import PoissonMatchPredictor
from models.ml_model import MLMatchPredictor

logger = logging.getLogger(__name__)


class MatchPredictorService:
    """
    Unified Prediction Engine:
    - Ingests cleaned data, computes engineered features.
    - Fits both Poisson and Machine Learning models.
    - Exposes a clean predict_match interface.
    """

    def __init__(self, data_path: str = "football_clean_data.csv"):
        self.data_path = data_path
        self.df_raw = pd.read_csv(data_path)
        required_columns = ["Date", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        missing_columns = [column for column in required_columns if column not in self.df_raw]
        if missing_columns:
            raise ValueError(
                f"Prediction dataset is missing required columns: {', '.join(missing_columns)}"
            )

        self.df_raw["FTHG"] = pd.to_numeric(self.df_raw["FTHG"], errors="coerce")
        self.df_raw["FTAG"] = pd.to_numeric(self.df_raw["FTAG"], errors="coerce")
        valid_matches = self.df_raw[required_columns].notna().all(axis=1)
        invalid_count = int((~valid_matches).sum())
        if invalid_count:
            logger.warning(
                "Ignoring %s dataset rows with missing or invalid fixture results.",
                invalid_count
            )
            self.df_raw = self.df_raw.loc[valid_matches].copy()
        if self.df_raw.empty:
            raise ValueError("Prediction dataset has no matches with valid final scores.")
        self.df_raw["FTHG"] = self.df_raw["FTHG"].astype(int)
        self.df_raw["FTAG"] = self.df_raw["FTAG"].astype(int)
        self.fe = FootballFeatureEngineer()

        logger.info("Computing features for prediction service...")
        self.df_features = self.fe.transform(self.df_raw)

        # Initialize models
        self.poisson_model = PoissonMatchPredictor()
        self.poisson_model.fit(self.df_raw)

        self.ml_model = MLMatchPredictor()
        logger.info("Training XGBoost regression models...")
        self.ml_model.fit(self.df_features)

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        method: str = "ml",
        display: bool = True,
        match_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predicts the outcome of an upcoming football match.

        Args:
            home_team: Name of the home team (raw or canonical).
            away_team: Name of the away team (raw or canonical).
            method: 'poisson' (Approach A) or 'ml' (Approach B: XGBoost).
            display: Whether to print formatted terminal output.

        Returns:
            Dict containing match prediction statistics.
        """
        h_norm = normalize_team_name(home_team)
        a_norm = normalize_team_name(away_team)

        if h_norm == a_norm:
            raise ValueError(f"Home team and Away team cannot be identical: {h_norm}")

        # Retrieve latest statistical profiles with home/away venue context
        h_feats = self.fe.get_latest_team_features(
            self.df_features, h_norm, is_home=True, match_date=match_date
        )
        a_feats = self.fe.get_latest_team_features(
            self.df_features, a_norm, is_home=False, match_date=match_date
        )

        # Execute chosen prediction method
        method_lower = method.lower()
        if method_lower in ["ml", "xgboost"]:
            res = self.ml_model.predict(h_feats, a_feats)
        elif method_lower in ["poisson", "dixon-coles"]:
            res = self.poisson_model.predict(h_feats, a_feats)
        else:
            raise ValueError("Prediction method must be 'poisson' or 'ml'.")

        res["home_team"] = h_norm
        res["away_team"] = a_norm
        res["home_league"] = h_feats.get("league")
        res["away_league"] = a_feats.get("league")
        res["match_date"] = match_date
        res["home_attack_strength"] = round(h_feats.get("attack_strength_10", 1.0), 2)
        res["away_attack_strength"] = round(a_feats.get("attack_strength_10", 1.0), 2)
        res["home_defense_weakness"] = round(h_feats.get("defense_weakness_10", 1.0), 2)
        res["away_defense_weakness"] = round(a_feats.get("defense_weakness_10", 1.0), 2)
        res["home_form_pts_5"] = h_feats.get("form_pts_5", 0)
        res["away_form_pts_5"] = a_feats.get("form_pts_5", 0)
        res["home_elo"] = round(h_feats.get("elo", 1500.0), 1)
        res["away_elo"] = round(a_feats.get("elo", 1500.0), 1)
        res["diff_elo"] = round((h_feats.get("elo", 1500.0) + 65.0) - a_feats.get("elo", 1500.0), 1)

        # Contextual Fatigue & Venue Form Attributes
        res["home_rest_days"] = int(round(h_feats.get("rest_days", 7.0)))
        res["away_rest_days"] = int(round(a_feats.get("rest_days", 7.0)))
        res["diff_rest_days"] = res["home_rest_days"] - res["away_rest_days"]
        res["home_venue_pts_5"] = int(round(h_feats.get("venue_form_pts_5", 7.0)))
        res["away_venue_pts_5"] = int(round(a_feats.get("venue_form_pts_5", 5.0)))

        if display:
            self._display_prediction(res)

        return res

    def _display_prediction(self, res: Dict[str, Any]):
        """Prints a professional match analytics card."""
        h = res["home_team"]
        a = res["away_team"]
        probs = res["probabilities_3way"]
        top_scores = res["top_exact_scores"]

        print("\n" + "=" * 65)
        print(f" MATCH PREDICTION: {h.upper()} vs {a.upper()}")
        print(f" Model Engine: {res['method']}")
        print("=" * 65)
        print(f" Expected Goals (xG) : {h} [{res['lambda_home']:.2f}] - [{res['lambda_away']:.2f}] {a}")
        print(f" World Football Elo  : {h} [{res.get('home_elo', 1500)}] vs [{res.get('away_elo', 1500)}] {a} (Diff: {res.get('diff_elo', 0):+})")
        print(f" Rest & Preparation  : {h} [{res.get('home_rest_days', 7)}d rest] vs [{res.get('away_rest_days', 7)}d rest] {a} (Edge: {res.get('diff_rest_days', 0):+}d)")
        print(f" Venue Fortress Form : {h} Home [{res.get('home_venue_pts_5', 7)} pts] vs {a} Away [{res.get('away_venue_pts_5', 5)} pts]")
        print(f" Attack Strength     : {h} [{res['home_attack_strength']:.2f}] vs {a} [{res['away_attack_strength']:.2f}]")
        print(f" Defense Weakness    : {h} [{res['home_defense_weakness']:.2f}] vs {a} [{res['away_defense_weakness']:.2f}]")
        print(f" Overall Form (5 Pts): {h} [{int(res['home_form_pts_5'])} pts] vs {a} [{int(res['away_form_pts_5'])} pts]")
        print("-" * 65)
        print(f" MOST LIKELY SCORE   : {res['most_likely_score']}  ({res['most_likely_probability']}%)")
        print("-" * 65)
        print(" 3-WAY MATCH OUTCOMES (1X2):")
        print(f"   * Home Win ({h}) : {probs['home_win']}%")
        print(f"   * Draw            : {probs['draw']}%")
        print(f"   * Away Win ({a}) : {probs['away_win']}%")
        print("-" * 65)
        print(" TOP 3 EXACT SCORE PROBABILITIES:")
        for idx, s in enumerate(top_scores, 1):
            bar = "#" * int(s['percentage'] / 2)
            print(f"   {idx}. Score {s['score']:<7} : {s['percentage']:>5.1f}%  | {bar}")
        print("=" * 65 + "\n")


# Global singleton instance for quick script calls
_service_instance: Optional[MatchPredictorService] = None


def predict_match(
    home_team: str,
    away_team: str,
    method: str = "ml",
    match_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Stand-alone user function to quickly predict any match.

    Example:
        predict_match('Arsenal', 'Chelsea', method='poisson')
        predict_match('Manchester City', 'Liverpool', method='ml')
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = MatchPredictorService()
    return _service_instance.predict_match(
        home_team, away_team, method=method, display=True, match_date=match_date
    )
