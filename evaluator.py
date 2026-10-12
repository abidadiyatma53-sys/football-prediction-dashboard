"""
Model Evaluation and Backtesting Engine for Football Analytics.
Performs Out-of-Time (OOT) time-series evaluation, benchmarking Poisson vs XGBoost vs Baseline,
computes quantitative probabilistic metrics (Log-Loss, Brier Score) and regression metrics (MAE, RMSE),
and generates publication-quality diagnostic visualizations.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, log_loss

from features import FootballFeatureEngineer
from models.poisson_model import PoissonMatchPredictor
from models.ml_model import MLMatchPredictor

logger = logging.getLogger(__name__)


def brier_score_multiclass(y_true_onehot: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes multi-class Brier Score:
    BS = (1/N) * sum_i sum_k (p_{i,k} - y_{i,k})^2
    Lower is better. Measures both calibration and sharpness of probability forecasts.
    """
    return float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))


class FootballBacktestEngine:
    """
    Quantitative Backtesting & Evaluation Harness:
    1. Splits dataset chronologically (Time-Series Out-Of-Time Split).
    2. Trains models strictly on past matches.
    3. Evaluates predictions on unseen future matches.
    4. Computes comprehensive regression, probabilistic, and classification metrics.
    5. Benchmarks against a naive historical baseline model.
    6. Produces confusion matrices and error diagnostic visualizations.
    """

    def __init__(
        self,
        data_path: str = "football_clean_data.csv",
        train_ratio: float = 0.8,
        charts_output_path: str = "evaluation_charts.png"
    ):
        self.data_path = data_path
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be greater than 0 and less than 1.")
        self.train_ratio = train_ratio
        self.charts_output_path = Path(charts_output_path)

        self.df_raw = pd.read_csv(data_path)
        self.fe = FootballFeatureEngineer()
        self.df_features = self.fe.transform(self.df_raw)

        # Split data chronologically
        self.train_df, self.test_df = self._split_time_series()

    def _split_time_series(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Performs strict chronological time-series splitting."""
        df_sorted = self.df_features.sort_values(by="Date", kind="stable").reset_index(drop=True)
        unique_dates = df_sorted["Date"].drop_duplicates().reset_index(drop=True)
        split_date_idx = int(len(unique_dates) * self.train_ratio)
        split_date = unique_dates.iloc[split_date_idx]

        train_df = df_sorted[df_sorted["Date"] < split_date].copy().reset_index(drop=True)
        test_df = df_sorted[df_sorted["Date"] >= split_date].copy().reset_index(drop=True)

        logger.info(
            f"Chronological split: Train = {len(train_df)} matches "
            f"({train_df['Date'].min()} to {train_df['Date'].max()}), "
            f"Test = {len(test_df)} matches ({test_df['Date'].min()} to {test_df['Date'].max()})"
        )
        return train_df, test_df

    def run_backtest(self) -> Dict[str, Any]:
        """
        Executes backtesting for:
        1. Baseline Model (Historical Mode & Base Rates).
        2. Poisson Goal Expectancy Model.
        3. XGBoost Multi-Output Machine Learning Model.
        """
        logger.info("Initializing and training models on training fold...")

        # 1. Baseline Model
        # Predicts mode score and historical outcome frequencies from training set
        mode_score = f"{int(self.train_df['FTHG'].mode()[0])} - {int(self.train_df['FTAG'].mode()[0])}"
        train_ftr_counts = self.train_df["FTR"].value_counts(normalize=True)
        baseline_prob_h = float(train_ftr_counts.get("H", 0.45))
        baseline_prob_d = float(train_ftr_counts.get("D", 0.25))
        baseline_prob_a = float(train_ftr_counts.get("A", 0.30))
        baseline_mean_h = float(self.train_df["FTHG"].mean())
        baseline_mean_a = float(self.train_df["FTAG"].mean())

        # 2. Poisson Model
        poisson_model = PoissonMatchPredictor()
        poisson_model.fit(self.train_df)

        # 3. XGBoost Model
        ml_model = MLMatchPredictor()
        # Train on training fold
        ml_model.fit(self.train_df, test_size=0.1)

        # Test set containers
        test_size = len(self.test_df)
        y_true_h = self.test_df["FTHG"].values
        y_true_a = self.test_df["FTAG"].values
        y_true_ftr = self.test_df["FTR"].values
        y_true_scores = [f"{int(h)} - {int(a)}" for h, a in zip(y_true_h, y_true_a)]

        # Map FTR to categorical index: 0 -> H, 1 -> D, 2 -> A
        ftr_map = {"H": 0, "D": 1, "A": 2}
        ftr_labels = ["H", "D", "A"]
        y_true_idx = np.array([ftr_map[r] for r in y_true_ftr])
        y_true_onehot = np.zeros((test_size, 3))
        for i, idx in enumerate(y_true_idx):
            y_true_onehot[i, idx] = 1.0

        # Predictions storage
        preds = {
            "baseline": {"h_goals": [], "a_goals": [], "probs": [], "pred_ftr": [], "pred_score": []},
            "poisson": {"h_goals": [], "a_goals": [], "probs": [], "pred_ftr": [], "pred_score": []},
            "xgboost": {"h_goals": [], "a_goals": [], "probs": [], "pred_ftr": [], "pred_score": []},
        }

        # Baseline predictions
        for _ in range(test_size):
            preds["baseline"]["h_goals"].append(baseline_mean_h)
            preds["baseline"]["a_goals"].append(baseline_mean_a)
            preds["baseline"]["probs"].append([baseline_prob_h, baseline_prob_d, baseline_prob_a])
            preds["baseline"]["pred_ftr"].append("H" if baseline_prob_h >= max(baseline_prob_d, baseline_prob_a) else "D")
            preds["baseline"]["pred_score"].append(mode_score)

        # Each test row already contains features built strictly from matches
        # before its kickoff date. Use those row-specific values so test
        # fixtures advance chronologically instead of reusing the train cutoff.
        for idx, row in self.test_df.iterrows():
            h_feats = {
                column[len("home_"):]: float(row[column])
                for column in row.index
                if column.startswith("home_") and pd.notna(row[column])
            }
            a_feats = {
                column[len("away_"):]: float(row[column])
                for column in row.index
                if column.startswith("away_") and pd.notna(row[column])
            }
            h_feats["league"] = str(row["League"])
            a_feats["league"] = str(row["League"])

            # Poisson
            p_res = poisson_model.predict(h_feats, a_feats)
            p_probs = [
                p_res["probabilities_3way"]["home_win"] / 100.0,
                p_res["probabilities_3way"]["draw"] / 100.0,
                p_res["probabilities_3way"]["away_win"] / 100.0,
            ]
            preds["poisson"]["h_goals"].append(p_res["lambda_home"])
            preds["poisson"]["a_goals"].append(p_res["lambda_away"])
            preds["poisson"]["probs"].append(p_probs)
            preds["poisson"]["pred_ftr"].append(ftr_labels[np.argmax(p_probs)])
            preds["poisson"]["pred_score"].append(p_res["most_likely_score"])

            # XGBoost
            m_res = ml_model.predict(h_feats, a_feats)
            m_probs = [
                m_res["probabilities_3way"]["home_win"] / 100.0,
                m_res["probabilities_3way"]["draw"] / 100.0,
                m_res["probabilities_3way"]["away_win"] / 100.0,
            ]
            preds["xgboost"]["h_goals"].append(m_res["lambda_home"])
            preds["xgboost"]["a_goals"].append(m_res["lambda_away"])
            preds["xgboost"]["probs"].append(m_probs)
            preds["xgboost"]["pred_ftr"].append(ftr_labels[np.argmax(m_probs)])
            preds["xgboost"]["pred_score"].append(m_res["most_likely_score"])

        # Calculate metrics for each model
        metrics = {}
        for m_name in ["baseline", "poisson", "xgboost"]:
            pred_h = np.array(preds[m_name]["h_goals"])
            pred_a = np.array(preds[m_name]["a_goals"])
            prob_mat = np.array(preds[m_name]["probs"])
            pred_ftr_arr = np.array(preds[m_name]["pred_ftr"])
            pred_score_arr = preds[m_name]["pred_score"]

            mae_h = mean_absolute_error(y_true_h, pred_h)
            mae_a = mean_absolute_error(y_true_a, pred_a)
            rmse_h = root_mean_squared_error(y_true_h, pred_h)
            rmse_a = root_mean_squared_error(y_true_a, pred_a)

            # Probabilistic scoring
            # Clip probabilities to avoid log(0)
            clipped_probs = np.clip(prob_mat, 1e-6, 1.0 - 1e-6)
            clipped_probs = clipped_probs / np.sum(clipped_probs, axis=1, keepdims=True)

            loss = log_loss(y_true_idx, clipped_probs, labels=[0, 1, 2])
            brier = brier_score_multiclass(y_true_onehot, clipped_probs)

            # Accuracy
            acc_1x2 = np.mean(pred_ftr_arr == y_true_ftr) * 100.0
            acc_exact = np.mean([p == t for p, t in zip(pred_score_arr, y_true_scores)]) * 100.0

            metrics[m_name] = {
                "mae_home": round(mae_h, 3),
                "mae_away": round(mae_a, 3),
                "mae_overall": round((mae_h + mae_a) / 2.0, 3),
                "rmse_home": round(rmse_h, 3),
                "rmse_away": round(rmse_a, 3),
                "log_loss": round(loss, 4),
                "brier_score": round(brier, 4),
                "accuracy_1x2": round(acc_1x2, 2),
                "accuracy_exact": round(acc_exact, 2),
                "pred_ftr": pred_ftr_arr,
                "pred_h": pred_h,
                "pred_a": pred_a,
                "probs": prob_mat
            }

        # Generate diagnostic charts
        self._generate_visualizations(y_true_h, y_true_a, y_true_ftr, metrics)

        return {
            "test_size": test_size,
            "train_size": len(self.train_df),
            "date_range": f"{self.test_df['Date'].min()} to {self.test_df['Date'].max()}",
            "metrics": metrics
        }

    def _generate_visualizations(
        self,
        y_true_h: np.ndarray,
        y_true_a: np.ndarray,
        y_true_ftr: np.ndarray,
        metrics: Dict[str, Any]
    ):
        """Generates a 4-panel diagnostic dashboard for model benchmarking."""
        sns.set_theme(style="whitegrid", font="sans-serif")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        labels = ["H", "D", "A"]

        # Panel 1: Confusion Matrix - Poisson
        from sklearn.metrics import confusion_matrix
        cm_poisson = confusion_matrix(y_true_ftr, metrics["poisson"]["pred_ftr"], labels=labels)
        sns.heatmap(
            cm_poisson,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[0, 0],
            cbar=False
        )
        axes[0, 0].set_title(
            f"Poisson Model: 1X2 Confusion Matrix\n(Accuracy: {metrics['poisson']['accuracy_1x2']}%)",
            fontsize=12, fontweight="bold"
        )
        axes[0, 0].set_xlabel("Predicted Outcome")
        axes[0, 0].set_ylabel("Actual Outcome")

        # Panel 2: Confusion Matrix - XGBoost
        cm_xgb = confusion_matrix(y_true_ftr, metrics["xgboost"]["pred_ftr"], labels=labels)
        sns.heatmap(
            cm_xgb,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[0, 1],
            cbar=False
        )
        axes[0, 1].set_title(
            f"XGBoost Model: 1X2 Confusion Matrix\n(Accuracy: {metrics['xgboost']['accuracy_1x2']}%)",
            fontsize=12, fontweight="bold"
        )
        axes[0, 1].set_xlabel("Predicted Outcome")
        axes[0, 1].set_ylabel("Actual Outcome")

        # Panel 3: Actual vs Predicted Goals (Scatter with Jitter)
        jitter = np.random.normal(0, 0.08, size=len(y_true_h))
        axes[1, 0].scatter(
            y_true_h + jitter,
            metrics["xgboost"]["pred_h"],
            color="#2b5c8f",
            alpha=0.6,
            label="XGBoost Home Goals"
        )
        axes[1, 0].scatter(
            y_true_a + jitter,
            metrics["xgboost"]["pred_a"],
            color="#d95f02",
            alpha=0.6,
            label="XGBoost Away Goals"
        )
        # Reference diagonal
        max_val = max(int(np.max(y_true_h)), int(np.max(y_true_a)), 5)
        axes[1, 0].plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect Line")
        axes[1, 0].set_title("Actual vs Predicted Goals (XGBoost)", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Actual Goals (with visual jitter)")
        axes[1, 0].set_ylabel(r"Predicted Goals ($\hat{y}$)")
        axes[1, 0].set_xlim(-0.5, max_val + 0.5)
        axes[1, 0].set_ylim(0, max_val + 0.5)
        axes[1, 0].legend()

        # Panel 4: Metric Comparison Bar Chart
        model_names = ["Baseline", "Dixon-Coles", "XGBoost"]
        acc_1x2_vals = [metrics["baseline"]["accuracy_1x2"], metrics["poisson"]["accuracy_1x2"], metrics["xgboost"]["accuracy_1x2"]]
        brier_vals = [metrics["baseline"]["brier_score"] * 100, metrics["poisson"]["brier_score"] * 100, metrics["xgboost"]["brier_score"] * 100]
        logloss_vals = [metrics["baseline"]["log_loss"] * 50, metrics["poisson"]["log_loss"] * 50, metrics["xgboost"]["log_loss"] * 50]

        x = np.arange(len(model_names))
        width = 0.25

        axes[1, 1].bar(x - width, acc_1x2_vals, width, label="1X2 Accuracy (%)", color="#386cb0")
        axes[1, 1].bar(x, brier_vals, width, label="Brier Score (x100, Lower=Better)", color="#f0027f")
        axes[1, 1].bar(x + width, logloss_vals, width, label="Log-Loss (x50, Lower=Better)", color="#7fc97f")

        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names, fontweight="bold")
        axes[1, 1].set_title("Benchmarking: Calibration & Accuracy Metrics", fontsize=12, fontweight="bold")
        axes[1, 1].set_ylabel("Score / Percentage")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.charts_output_path, dpi=200)
        plt.close()
        logger.info(f"Visualizations saved to {self.charts_output_path}")

    def print_evaluation_report(self, results: Dict[str, Any]):
        """Prints formatted quant performance summary report."""
        m = results["metrics"]
        test_sz = results["test_size"]
        train_sz = results["train_size"]
        date_rng = results["date_range"]

        print("\n" + "=" * 80)
        print(" FOOTBALL MODEL BACKTESTING & EVALUATION REPORT")
        print("=" * 80)
        print(f" Train Split  : {train_sz} matches (Chronological {int(self.train_ratio*100)}%)")
        print(f" Test Split   : {test_sz} matches (Out-of-Time Last {int((1-self.train_ratio)*100)}%)")
        print(f" Test Period  : {date_rng}")
        print(f" Visual Chart : {self.charts_output_path.resolve()}")
        print("-" * 80)

        # Tabular Summary
        headers = [
            "Model Name",
            "1X2 Acc (%)",
            "Exact Acc (%)",
            "Log-Loss",
            "Brier Score",
            "Goal MAE (H/A)",
            "Goal RMSE (H/A)"
        ]

        print(f"{headers[0]:<12} | {headers[1]:<11} | {headers[2]:<13} | {headers[3]:<9} | {headers[4]:<11} | {headers[5]:<14} | {headers[6]:<14}")
        print("-" * 80)

        for name, key in [("Baseline", "baseline"), ("Dixon-Coles", "poisson"), ("XGBoost", "xgboost")]:
            row = m[key]
            mae_str = f"{row['mae_home']:.2f} / {row['mae_away']:.2f}"
            rmse_str = f"{row['rmse_home']:.2f} / {row['rmse_away']:.2f}"
            print(
                f"{name:<12} | "
                f"{row['accuracy_1x2']:>10.2f}% | "
                f"{row['accuracy_exact']:>12.2f}% | "
                f"{row['log_loss']:>9.4f} | "
                f"{row['brier_score']:>11.4f} | "
                f"{mae_str:>14} | "
                f"{rmse_str:>14}"
            )

        print("=" * 80 + "\n")
