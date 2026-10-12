"""
Machine Learning Match Predictor (Approach B).
Employs XGBoost Multi-Output Regression to predict Home and Away goals separately
from engineered statistical features, deriving exact score matrices and 1X2 probabilities.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class MLMatchPredictor:
    """
    Supervised Machine Learning Regressor for Football Goal Forecasting:
    - Multi-Output Regression predicting both Home Goals (FTHG) and Away Goals (FTAG).
    - Uses XGBoost with Poisson objective count model for non-negative goal outcomes.
    - Evaluates performance using chronological Out-Of-Time (OOT) validation.
    - Converts predicted expected goals into discrete scoreline probability distributions.
    """

    FEATURE_COLUMNS = [
        "home_attack_strength_10",
        "home_defense_weakness_10",
        "home_form_pts_5",
        "home_form_gd_5",
        "home_rolling_scored_5",
        "home_rolling_conceded_5",
        "home_rolling_xg_for_5",
        "home_rolling_xg_against_5",
        "away_attack_strength_10",
        "away_defense_weakness_10",
        "away_form_pts_5",
        "away_form_gd_5",
        "away_rolling_scored_5",
        "away_rolling_conceded_5",
        "away_rolling_xg_for_5",
        "away_rolling_xg_against_5",
        "diff_form_pts_5",
        "diff_form_gd_5",
        "diff_rolling_xg_5",
        "home_attack_vs_away_defense",
        "away_attack_vs_home_defense",
        "home_elo",
        "away_elo",
        "diff_elo",
        # Contextual Fatigue & Schedule Congestion
        "home_rest_days",
        "away_rest_days",
        "diff_rest_days",
        "home_is_congested",
        "away_is_congested",
        # Venue-Specific Home Fortress & Away Travel Form
        "home_venue_form_pts_5",
        "home_venue_form_gd_5",
        "away_venue_form_pts_5",
        "away_venue_form_gd_5",
        "diff_venue_form_pts_5",
        "diff_venue_form_gd_5"
    ]

    def __init__(self, max_goals: int = 5):
        self.max_goals = max_goals
        self.model: Optional[MultiOutputRegressor] = None
        self.metrics: Dict[str, float] = {}
        self.is_fitted = False

    def fit(self, df_features: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        """
        Trains the Multi-Output XGBoost model using a chronological train/test split.
        """
        # Ensure all feature columns exist
        available_feats = [c for c in self.FEATURE_COLUMNS if c in df_features.columns]
        if len(available_feats) < 5:
            raise ValueError(f"Insufficient feature columns in DataFrame: {available_feats}")

        self.feature_cols = available_feats

        # Sort chronologically
        df_sorted = df_features.sort_values(by="Date", kind="stable").reset_index(drop=True)

        X = df_sorted[self.feature_cols].values
        y = df_sorted[["FTHG", "FTAG"]].values

        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be greater than 0 and less than 1.")
        unique_dates = df_sorted["Date"].drop_duplicates().reset_index(drop=True)
        split_date_idx = int(len(unique_dates) * (1.0 - test_size))
        split_date = unique_dates.iloc[split_date_idx]
        train_mask = df_sorted["Date"].to_numpy() < split_date
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
        if not len(X_train) or not len(X_test):
            raise ValueError("Not enough distinct match dates for the requested validation split.")

        logger.info(f"Training XGBoost on {len(X_train)} samples, testing on {len(X_test)} samples...")

        # Initialize base estimator with count:poisson objective
        base_xgb = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="count:poisson",
            random_state=42
        )

        validation_model = MultiOutputRegressor(base_xgb)
        validation_model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = validation_model.predict(X_test)
        mae_home = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
        mae_away = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
        rmse_home = root_mean_squared_error(y_test[:, 0], y_pred[:, 0])
        rmse_away = root_mean_squared_error(y_test[:, 1], y_pred[:, 1])

        self.metrics = {
            "test_samples": len(X_test),
            "mae_home_goals": round(mae_home, 3),
            "mae_away_goals": round(mae_away, 3),
            "rmse_home_goals": round(rmse_home, 3),
            "rmse_away_goals": round(rmse_away, 3),
        }

        logger.info(f"XGBoost evaluation: MAE Home={mae_home:.3f}, MAE Away={mae_away:.3f}")
        self.model = MultiOutputRegressor(base_xgb)
        self.model.fit(X, y)
        self.is_fitted = True
        return self.metrics

    def predict_goals(self, feature_row: np.ndarray) -> Tuple[float, float]:
        """
        Predicts continuous goal expectations (Home Goals, Away Goals).
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict.")

        pred = self.model.predict(feature_row.reshape(1, -1))[0]
        pred_home = max(0.15, float(pred[0]))
        pred_away = max(0.15, float(pred[1]))
        return round(pred_home, 3), round(pred_away, 3)

    def generate_score_matrix(self, lambda_h: float, lambda_a: float) -> np.ndarray:
        """
        Constructs a (max_goals+1) x (max_goals+1) probability matrix.
        """
        max_goals = max(
            self.max_goals,
            int(poisson.ppf(1.0 - 1e-12, max(lambda_h, lambda_a)))
        )
        goals_range = np.arange(max_goals + 1)
        home_probs = poisson.pmf(goals_range, lambda_h)
        away_probs = poisson.pmf(goals_range, lambda_a)

        matrix = np.outer(home_probs, away_probs)
        return matrix

    def predict(
        self,
        home_feats: Dict[str, float],
        away_feats: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Combines team features, predicts goals using XGBoost, and generates full
        outcome distribution.
        """
        # Construct single feature row matching training columns
        row_dict = {}
        for k, v in home_feats.items():
            row_dict[f"home_{k}"] = v
        for k, v in away_feats.items():
            row_dict[f"away_{k}"] = v

        row_dict["diff_form_pts_5"] = row_dict.get("home_form_pts_5", 0) - row_dict.get("away_form_pts_5", 0)
        row_dict["diff_form_gd_5"] = row_dict.get("home_form_gd_5", 0) - row_dict.get("away_form_gd_5", 0)
        row_dict["diff_rolling_xg_5"] = row_dict.get("home_rolling_xg_for_5", 0) - row_dict.get("away_rolling_xg_for_5", 0)
        row_dict["home_attack_vs_away_defense"] = (
            row_dict.get("home_attack_strength_10", 1.0) * row_dict.get("away_defense_weakness_10", 1.0)
        )
        row_dict["away_attack_vs_home_defense"] = (
            row_dict.get("away_attack_strength_10", 1.0) * row_dict.get("home_defense_weakness_10", 1.0)
        )

        h_elo = home_feats.get("elo", 1500.0)
        a_elo = away_feats.get("elo", 1500.0)
        row_dict["home_elo"] = h_elo
        row_dict["away_elo"] = a_elo
        row_dict["diff_elo"] = (h_elo + 65.0) - a_elo

        # Contextual Fatigue & Schedule Congestion
        h_rest = home_feats.get("rest_days", 7.0)
        a_rest = away_feats.get("rest_days", 7.0)
        row_dict["home_rest_days"] = h_rest
        row_dict["away_rest_days"] = a_rest
        row_dict["diff_rest_days"] = h_rest - a_rest
        row_dict["home_is_congested"] = float(h_rest <= 3.0)
        row_dict["away_is_congested"] = float(a_rest <= 3.0)

        # Venue-Specific Form
        h_v_pts = home_feats.get("venue_form_pts_5", 7.0)
        h_v_gd = home_feats.get("venue_form_gd_5", 0.0)
        a_v_pts = away_feats.get("venue_form_pts_5", 5.0)
        a_v_gd = away_feats.get("venue_form_gd_5", 0.0)
        row_dict["home_venue_form_pts_5"] = h_v_pts
        row_dict["home_venue_form_gd_5"] = h_v_gd
        row_dict["away_venue_form_pts_5"] = a_v_pts
        row_dict["away_venue_form_gd_5"] = a_v_gd
        row_dict["diff_venue_form_pts_5"] = h_v_pts - a_v_pts
        row_dict["diff_venue_form_gd_5"] = h_v_gd - a_v_gd

        feature_vector = np.array([row_dict.get(c, 1.0) for c in self.feature_cols])

        pred_h, pred_a = self.predict_goals(feature_vector)
        matrix = self.generate_score_matrix(pred_h, pred_a)

        p_home_win = float(np.sum(np.tril(matrix, -1)))
        p_draw = float(np.sum(np.diag(matrix)))
        p_away_win = float(np.sum(np.triu(matrix, 1)))

        score_list = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                score_list.append({
                    "score": f"{i} - {j}",
                    "home_goals": i,
                    "away_goals": j,
                    "probability": float(matrix[i, j]),
                    "percentage": round(float(matrix[i, j]) * 100, 2)
                })

        score_list.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "method": "Machine Learning (XGBoost Multi-Output)",
            "lambda_home": pred_h,
            "lambda_away": pred_a,
            "probabilities_3way": {
                "home_win": round(p_home_win * 100, 2),
                "draw": round(p_draw * 100, 2),
                "away_win": round(p_away_win * 100, 2)
            },
            "most_likely_score": score_list[0]["score"],
            "most_likely_probability": score_list[0]["percentage"],
            "top_exact_scores": score_list[:3],
            "score_matrix": matrix
        }
