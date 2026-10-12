"""
CLI Execution Script for Model Backtesting & Evaluation.
Executes Out-of-Time backtesting, generates performance metrics, and plots evaluation charts.

Example:
    python run_backtest.py
    python run_backtest.py --train-ratio 0.75 --output-chart my_charts.png
"""

import argparse
import json
from pathlib import Path
from evaluator import FootballBacktestEngine


def main():
    parser = argparse.ArgumentParser(
        description="Backtesting & Model Evaluation Engine for Football Match Outcome Prediction."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="football_clean_data.csv",
        help="Path to clean match CSV dataset (default: football_clean_data.csv)."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Chronological train split ratio (default: 0.8 for 80% train, 20% test)."
    )
    parser.add_argument(
        "--output-chart",
        type=str,
        default="evaluation_charts.png",
        help="Path to save evaluation visualization figure (default: evaluation_charts.png)."
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("evaluation_metrics.json"),
        help="Path to save machine-readable evaluation metrics (default: evaluation_metrics.json)."
    )

    args = parser.parse_args()

    engine = FootballBacktestEngine(
        data_path=args.data,
        train_ratio=args.train_ratio,
        charts_output_path=args.output_chart
    )

    results = engine.run_backtest()
    engine.print_evaluation_report(results)
    metric_keys = (
        "mae_home", "mae_away", "rmse_home", "rmse_away", "log_loss",
        "brier_score", "accuracy_1x2", "accuracy_exact"
    )
    summary = {
        key: {metric: float(value[metric]) for metric in metric_keys}
        for key, value in results["metrics"].items()
    }
    payload = {
        "data_path": str(Path(args.data)),
        "chart_path": str(Path(args.output_chart)),
        "train_ratio": args.train_ratio,
        "train_size": results["train_size"],
        "test_size": results["test_size"],
        "date_range": results["date_range"],
        "metrics": summary,
    }
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f" Evaluation Metrics: {args.output_metrics.resolve()}")


if __name__ == "__main__":
    main()
