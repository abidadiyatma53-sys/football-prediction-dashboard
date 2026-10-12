import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluator import FootballBacktestEngine
from features import FootballFeatureEngineer
from models.ml_model import MLMatchPredictor
from models.poisson_model import PoissonMatchPredictor
from tracking_service import PredictionTrackingService


def sample_matches():
    return pd.DataFrame([
        {
            "Date": "2025-01-01", "League": "EPL", "HomeTeam": "Alpha",
            "AwayTeam": "Beta", "FTHG": 2, "FTAG": 1, "FTR": "H",
            "HS": 10, "AS": 7, "HST": 4, "AST": 3,
            "xG_home": 1.8, "xG_away": 0.9,
        },
        {
            "Date": "2025-01-08", "League": "EPL", "HomeTeam": "Gamma",
            "AwayTeam": "Alpha", "FTHG": 2, "FTAG": 0, "FTR": "H",
            "HS": 12, "AS": 8, "HST": 5, "AST": 2,
            "xG_home": 1.9, "xG_away": 0.7,
        },
        {
            "Date": "2025-01-15", "League": "EPL", "HomeTeam": "Alpha",
            "AwayTeam": "Gamma", "FTHG": 0, "FTAG": 2, "FTR": "A",
            "HS": 6, "AS": 11, "HST": 1, "AST": 5,
            "xG_home": 0.5, "xG_away": 1.7,
        },
        {
            "Date": "2025-01-22", "League": "EPL", "HomeTeam": "Beta",
            "AwayTeam": "Gamma", "FTHG": 1, "FTAG": 1, "FTR": "D",
            "HS": 8, "AS": 8, "HST": 3, "AST": 3,
            "xG_home": 1.0, "xG_away": 1.1,
        },
    ])


class FeatureTimingTests(unittest.TestCase):
    def test_future_results_do_not_change_prior_features(self):
        earlier = sample_matches()
        later_match = pd.DataFrame([{
            **sample_matches().iloc[0].to_dict(),
            "Date": "2025-02-01", "HomeTeam": "Alpha", "AwayTeam": "Beta",
            "FTHG": 9, "FTAG": 0, "FTR": "H",
        }])
        engineer = FootballFeatureEngineer()
        prior_features = engineer.transform(earlier)
        extended_features = engineer.transform(pd.concat([earlier, later_match], ignore_index=True))

        feature_columns = [
            column for column in prior_features
            if column.startswith(("home_", "away_", "diff_"))
        ]
        pd.testing.assert_frame_equal(
            prior_features[feature_columns],
            extended_features.iloc[:len(earlier)][feature_columns],
            check_dtype=False,
        )

    def test_live_profile_includes_latest_completed_result(self):
        engineer = FootballFeatureEngineer()
        features = engineer.transform(sample_matches())
        profile = engineer.get_latest_team_features(features, "Alpha", is_home=True)
        self.assertAlmostEqual(profile["rolling_scored_5"], 2.0 / 3.0)
        self.assertEqual(profile["form_pts_5"], 3.0)

    def test_prediction_profile_excludes_matches_on_or_after_fixture_date(self):
        engineer = FootballFeatureEngineer()
        features = engineer.transform(sample_matches())
        profile = engineer.get_latest_team_features(
            features, "Alpha", is_home=True, match_date="2025-01-10"
        )
        self.assertEqual(profile["form_pts_5"], 3.0)
        self.assertEqual(profile["rest_days"], 2.0)
        self.assertEqual(profile["league"], "EPL")
        self.assertEqual(profile["league_home_avg"], 2.0)
        self.assertEqual(profile["league_away_avg"], 0.5)

    def test_same_day_results_do_not_change_other_fixture_features(self):
        fixtures = pd.DataFrame([
            {
                "Date": "2025-01-01", "League": "EPL", "HomeTeam": "Alpha",
                "AwayTeam": "Beta", "FTHG": 2, "FTAG": 1, "FTR": "H",
                "HS": 10, "AS": 7, "HST": 4, "AST": 3,
                "xG_home": 1.8, "xG_away": 0.9,
            },
            {
                "Date": "2025-01-01", "League": "EPL", "HomeTeam": "Gamma",
                "AwayTeam": "Delta", "FTHG": 1, "FTAG": 0, "FTR": "H",
                "HS": 9, "AS": 5, "HST": 3, "AST": 2,
                "xG_home": 1.2, "xG_away": 0.4,
            },
        ])
        changed_first_result = fixtures.copy()
        changed_first_result.loc[0, ["FTHG", "FTAG", "FTR"]] = [9, 0, "H"]
        engineer = FootballFeatureEngineer()
        original = engineer.transform(fixtures)
        changed = engineer.transform(changed_first_result)
        same_day_features = [
            "home_rolling_scored_5", "home_rolling_conceded_5",
            "away_rolling_scored_5", "away_rolling_conceded_5",
        ]
        pd.testing.assert_series_equal(
            original.loc[1, same_day_features],
            changed.loc[1, same_day_features],
            check_dtype=False,
        )

    def test_train_test_split_keeps_each_date_in_one_fold(self):
        engine = FootballBacktestEngine.__new__(FootballBacktestEngine)
        engine.train_ratio = 0.5
        engine.df_features = pd.DataFrame({
            "Date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03"],
            "value": [1, 2, 3, 4],
        })
        train, test = engine._split_time_series()
        self.assertTrue(train["Date"].max() < test["Date"].min())


class SettlementMatchingTests(unittest.TestCase):
    def test_auto_settlement_matches_exact_fixture_date(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            matches_path = root / "matches.csv"
            pd.DataFrame([
                {
                    "Date": "2023-02-01", "HomeTeam": "Arsenal",
                    "AwayTeam": "Chelsea", "FTHG": 4, "FTAG": 0,
                },
                {
                    "Date": "2026-10-07", "HomeTeam": "Arsenal",
                    "AwayTeam": "Chelsea", "FTHG": 2, "FTAG": 1,
                },
            ]).to_csv(matches_path, index=False)
            service = PredictionTrackingService(
                db_path=root / "ledger.db",
                data_path=str(matches_path),
            )
            record = {
                "match_date": "2026-10-08", "league": "EPL", "season": "2026",
                "home_team": "Arsenal", "away_team": "Chelsea", "model_method": "test",
                "prob_home": 50, "prob_draw": 25, "prob_away": 25,
                "lambda_home": 1.5, "lambda_away": 1.0,
                "predicted_score": "1 - 0", "predicted_score_prob": 10,
                "bet_selection": None, "bet_stake_units": 1,
            }
            future_id = service.db.log_prediction(record)
            no_match_result = service.auto_settle_from_dataset()
            self.assertEqual(no_match_result["settled_count"], 0)
            self.assertEqual(len(service.db.get_pending_predictions()), 1)

            completed_id = service.db.log_prediction({
                **record, "match_date": "2026-10-07",
            })
            settled_result = service.auto_settle_from_dataset()
            self.assertEqual(settled_result["settled_count"], 1)
            settled = service.db.get_all_predictions(limit=2).set_index("id").loc[completed_id]
            self.assertEqual(settled["actual_home_goals"], 2)
            self.assertEqual(settled["actual_away_goals"], 1)
            self.assertFalse(service.db.settle_prediction(completed_id, 0, 0))
            self.assertIn(future_id, service.db.get_pending_predictions()[0].values())


class PoissonCalibrationTests(unittest.TestCase):
    def test_goal_baselines_are_calibrated_per_league(self):
        matches = pd.DataFrame([
            {"League": "EPL", "FTHG": 2, "FTAG": 1},
            {"League": "EPL", "FTHG": 4, "FTAG": 3},
            {"League": "La_Liga", "FTHG": 0, "FTAG": 0},
            {"League": "La_Liga", "FTHG": 2, "FTAG": 1},
        ])
        model = PoissonMatchPredictor(xg_weight=0.0)
        model.fit(matches)
        home_goals, away_goals = model.calculate_lambdas(
            {"league": "EPL", "attack_strength_10": 1.0},
            {"league": "EPL", "defense_weakness_10": 1.0, "attack_strength_10": 1.0},
        )
        other_home_goals, other_away_goals = model.calculate_lambdas(
            {"league": "La_Liga", "attack_strength_10": 1.0},
            {"league": "La_Liga", "defense_weakness_10": 1.0, "attack_strength_10": 1.0},
        )
        self.assertEqual((home_goals, away_goals), (3.0, 2.0))
        self.assertEqual((other_home_goals, other_away_goals), (1.0, 0.5))

    def test_score_distribution_includes_high_scores_without_renormalizing_tail(self):
        model = PoissonMatchPredictor(max_goals=5, rho=0.0, use_dixon_coles=False)
        matrix = model.generate_score_matrix(8.0, 7.0)
        self.assertGreater(matrix.shape[0], 6)
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=10)

    def test_ml_score_distribution_includes_high_scores_without_renormalizing_tail(self):
        model = MLMatchPredictor(max_goals=5)
        matrix = model.generate_score_matrix(8.0, 7.0)
        self.assertGreater(matrix.shape[0], 6)
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=10)

    def test_rho_estimation_does_not_use_post_match_xg(self):
        matches = pd.DataFrame([
            {
                "League": "EPL",
                "FTHG": index % 3,
                "FTAG": (index + 1) % 3,
                "xG_home": 0.2 + (index % 5),
                "xG_away": 0.4 + (index % 4),
            }
            for index in range(30)
        ])
        altered_xg = matches.copy()
        altered_xg[["xG_home", "xG_away"]] *= 4
        original_model = PoissonMatchPredictor()
        original_model.fit(matches)
        altered_model = PoissonMatchPredictor()
        altered_model.fit(altered_xg)
        self.assertAlmostEqual(original_model.rho, altered_model.rho, places=10)


if __name__ == "__main__":
    unittest.main()
