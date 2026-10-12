"""
Dixon-Coles & Poisson Distribution Match Predictor (Approach A).
Calculates expected goal rates (Lambda) and generates exact score probability matrices
along with 3-way match outcomes (Home Win, Draw, Away Win), with Dixon-Coles bivariate
adjustment to correct for low-score dependency (0-0, 1-1, 1-0, 0-1 correlation).
"""

import math
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


class PoissonMatchPredictor:
    """
    Parametric Football Prediction Model using Dixon-Coles Adjusted Poisson Distribution:
    - Computes lambda (expected goals) for home and away sides based on
      Attack Strength, Defense Weakness, and League Baseline Averages.
    - Applies Dixon-Coles (1997) bivariate correction factor tau(x, y) to capture
      structural correlations in low-scoring football outcomes (0-0, 1-1, 1-0, 0-1).
    - Generates full exact scoreline probability matrices (e.g. 0-0 up to 5-5).
    - Derives 3-way outcomes (Home Win, Draw, Away Win) and Top exact scores.
    """

    def __init__(
        self,
        max_goals: int = 5,
        xg_weight: float = 0.4,
        use_dixon_coles: bool = True,
        rho: Optional[float] = None
    ):
        """
        Args:
            max_goals: Maximum goal count considered in matrix (default: 5).
            xg_weight: Blending weight for rolling xG (0.0 to 1.0).
            use_dixon_coles: Whether to apply Dixon-Coles low-score adjustment.
            rho: Pre-set Dixon-Coles correlation parameter (if None, estimated via MLE in fit()).
        """
        self.max_goals = max_goals
        self.xg_weight = xg_weight
        self.use_dixon_coles = use_dixon_coles
        self.rho = rho if rho is not None else -0.11
        self.league_home_avg = 1.55
        self.league_away_avg = 1.25
        self.league_goal_averages: Dict[str, Tuple[float, float]] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Calibrates league baseline goal rates and Dixon-Coles rho parameter
        via Maximum Likelihood Estimation (MLE) from historical data.
        """
        if not df.empty:
            self.league_home_avg = float(df["FTHG"].mean()) if "FTHG" in df.columns else 1.55
            self.league_away_avg = float(df["FTAG"].mean()) if "FTAG" in df.columns else 1.25
            if "League" in df.columns:
                league_averages = df.groupby("League")[["FTHG", "FTAG"]].mean()
                self.league_goal_averages = {
                    str(league): (float(row["FTHG"]), float(row["FTAG"]))
                    for league, row in league_averages.iterrows()
                }

            # Optimize Dixon-Coles rho parameter via MLE if enabled and not hardcoded
            if self.use_dixon_coles and "FTHG" in df.columns and "FTAG" in df.columns:
                try:
                    self.rho = self._estimate_mle_rho(df)
                    logger.info(f"Dixon-Coles parameter calibrated via MLE: rho = {self.rho:.4f}")
                except Exception as e:
                    logger.warning(f"MLE estimation for Dixon-Coles rho failed ({e}), using default rho = {self.rho:.4f}")

            logger.info(
                f"Poisson model calibrated: League Home Avg={self.league_home_avg:.2f}, "
                f"Away Avg={self.league_away_avg:.2f}, Dixon-Coles rho={self.rho:.4f}"
            )
        self.is_fitted = True

    def _estimate_mle_rho(self, df: pd.DataFrame) -> float:
        """
        Estimates optimal rho parameter by minimizing negative log-likelihood
        over observed historical scorelines.
        """
        valid_df = df.dropna(subset=["FTHG", "FTAG"])
        if len(valid_df) < 20:
            return -0.11

        def neg_log_likelihood(candidate_rho: float) -> float:
            nll = 0.0
            for _, row in valid_df.iterrows():
                x = int(row["FTHG"])
                y = int(row["FTAG"])
                league = str(row.get("League", ""))
                lh, la = self.league_goal_averages.get(
                    league, (self.league_home_avg, self.league_away_avg)
                )
                tau = self.tau_adjustment(x, y, lh, la, candidate_rho)
                p_x = poisson.pmf(x, lh)
                p_y = poisson.pmf(y, la)
                p = max(1e-9, tau * p_x * p_y)
                nll -= math.log(p)
            return nll

        opt_res = minimize_scalar(
            neg_log_likelihood,
            bounds=(-0.25, 0.15),
            method="bounded"
        )
        return float(opt_res.x)

    def tau_adjustment(
        self,
        x: int,
        y: int,
        lambda_h: float,
        lambda_a: float,
        rho: Optional[float] = None
    ) -> float:
        """
        Dixon & Coles (1997) correlation factor tau(x, y):
        - Inflates 0-0 and 1-1 probabilities when rho < 0.
        - Deflates 1-0 and 0-1 probabilities accordingly.
        - Preserves marginal goal distributions for higher scores.
        """
        if not self.use_dixon_coles:
            return 1.0

        r = self.rho if rho is None else rho
        if r == 0.0:
            return 1.0

        if x == 0 and y == 0:
            return max(0.001, 1.0 - (lambda_h * lambda_a * r))
        elif x == 0 and y == 1:
            return max(0.001, 1.0 + (lambda_h * r))
        elif x == 1 and y == 0:
            return max(0.001, 1.0 + (lambda_a * r))
        elif x == 1 and y == 1:
            return max(0.001, 1.0 - r)
        return 1.0

    def calculate_lambdas(
        self,
        home_feats: Dict[str, float],
        away_feats: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Calculates expected goal rates (lambda_home, lambda_away):
        Base Lambda = Attack Strength * Opponent Defense Weakness * League Average.
        Blends with rolling xG for enhanced predictive precision.
        """
        # Feature-based expected goals
        h_att = home_feats.get("attack_strength_10", 1.0)
        a_def = away_feats.get("defense_weakness_10", 1.0)
        a_att = away_feats.get("attack_strength_10", 1.0)
        h_def = home_feats.get("defense_weakness_10", 1.0)

        home_league_avg = self.league_goal_averages.get(
            str(home_feats.get("league", "")),
            (self.league_home_avg, self.league_away_avg)
        )[0]
        away_league_avg = self.league_goal_averages.get(
            str(away_feats.get("league", "")),
            (self.league_home_avg, self.league_away_avg)
        )[1]
        lambda_h_stats = h_att * a_def * home_league_avg
        lambda_a_stats = a_att * h_def * away_league_avg

        # Blend with recent rolling xG if available
        h_xg = home_feats.get("rolling_xg_for_5", lambda_h_stats)
        a_xg = away_feats.get("rolling_xg_for_5", lambda_a_stats)

        lambda_h = (1.0 - self.xg_weight) * lambda_h_stats + self.xg_weight * h_xg
        lambda_a = (1.0 - self.xg_weight) * lambda_a_stats + self.xg_weight * a_xg

        # Floor at 0.15 to avoid zero probabilities
        lambda_h = max(0.15, float(lambda_h))
        lambda_a = max(0.15, float(lambda_a))

        return round(lambda_h, 3), round(lambda_a, 3)

    def generate_score_matrix(self, lambda_h: float, lambda_a: float) -> np.ndarray:
        """
        Constructs a (max_goals+1) x (max_goals+1) probability matrix
        with Dixon-Coles bivariate adjustment applied:
        cell [i, j] represents P(Home Goals = i AND Away Goals = j).
        """
        max_goals = max(
            self.max_goals,
            int(poisson.ppf(1.0 - 1e-12, max(lambda_h, lambda_a)))
        )
        goals_range = np.arange(max_goals + 1)
        home_probs = poisson.pmf(goals_range, lambda_h)
        away_probs = poisson.pmf(goals_range, lambda_a)

        # Base independence outer product
        score_matrix = np.outer(home_probs, away_probs)

        # Apply Dixon-Coles low-score adjustment factor
        if self.use_dixon_coles and self.rho != 0.0:
            for i in range(min(2, self.max_goals + 1)):
                for j in range(min(2, self.max_goals + 1)):
                    score_matrix[i, j] *= self.tau_adjustment(i, j, lambda_h, lambda_a)

        return score_matrix

    def predict(
        self,
        home_feats: Dict[str, float],
        away_feats: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generates full predictive output for a fixture:
        1. Expected goals (Lambda Home & Away).
        2. Score matrix with Dixon-Coles adjustment.
        3. 3-Way match outcome probabilities (1X2).
        4. Most likely score and Top-N exact scores.
        """
        lambda_h, lambda_a = self.calculate_lambdas(home_feats, away_feats)
        matrix = self.generate_score_matrix(lambda_h, lambda_a)

        # 1X2 Probabilities
        # Home Win: elements below the main diagonal (row > col)
        p_home_win = float(np.sum(np.tril(matrix, -1)))
        # Draw: diagonal elements (row == col)
        p_draw = float(np.sum(np.diag(matrix)))
        # Away Win: elements above the main diagonal (row < col)
        p_away_win = float(np.sum(np.triu(matrix, 1)))

        # Flatten and sort exact scores by probability
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

        engine_name = "Dixon-Coles Poisson Distribution" if self.use_dixon_coles else "Poisson Distribution"

        return {
            "method": engine_name,
            "lambda_home": lambda_h,
            "lambda_away": lambda_a,
            "rho": round(self.rho, 4) if self.use_dixon_coles else 0.0,
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
