import unittest

from monte_carlo import simulate_match


class MonteCarloTests(unittest.TestCase):
    def test_simulation_is_reproducible_and_returns_probability_distribution(self):
        scenarios = [
            {
                "name": "base",
                "weight": 70,
                "home_multiplier": 1.0,
                "away_multiplier": 1.0,
            },
            {
                "name": "home lineup change",
                "weight": 30,
                "home_multiplier": 0.8,
                "away_multiplier": 1.0,
            },
        ]
        first = simulate_match(1.6, 1.2, scenarios, simulations=20_000, seed=7)
        second = simulate_match(1.6, 1.2, scenarios, simulations=20_000, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(first["simulations"], 20_000)
        self.assertEqual(sum(item["percentage"] for item in first["outcomes"].values()), 100)
        self.assertEqual(sum(item["weight_pct"] for item in first["scenario_results"]), 100)
        self.assertTrue(first["top_scorelines"])

    def test_lineup_scenario_changes_expected_simulated_goals(self):
        result = simulate_match(
            1.5,
            1.0,
            [
                {"name": "baseline", "weight": 1, "home_multiplier": 1, "away_multiplier": 1},
                {"name": "weaker home attack", "weight": 1, "home_multiplier": 0.5, "away_multiplier": 1},
            ],
            simulations=40_000,
            parameter_cv=0,
            seed=42,
        )
        scenarios = {row["scenario"]: row for row in result["scenario_results"]}

        self.assertAlmostEqual(scenarios["baseline"]["mean_home_goals"], 1.5, delta=0.04)
        self.assertAlmostEqual(scenarios["weaker home attack"]["mean_home_goals"], 0.75, delta=0.04)

    def test_scenario_weights_are_normalized_and_zero_weight_is_omitted(self):
        result = simulate_match(
            1.4,
            1.1,
            [
                {"name": "active", "weight": 2, "home_multiplier": 1, "away_multiplier": 1},
                {"name": "inactive", "weight": 0, "home_multiplier": 1, "away_multiplier": 1},
            ],
            simulations=1_000,
            parameter_cv=0,
            seed=1,
        )
        self.assertEqual(len(result["scenario_results"]), 1)
        self.assertEqual(result["scenario_results"][0]["weight_pct"], 100)

    def test_dixon_coles_rho_is_reported_and_applied(self):
        without_adjustment = simulate_match(
            1.3,
            1.1,
            [{"weight": 100}],
            simulations=50_000,
            parameter_cv=0,
            rho=0,
            seed=99,
        )
        with_adjustment = simulate_match(
            1.3,
            1.1,
            [{"weight": 100}],
            simulations=50_000,
            parameter_cv=0,
            rho=-0.2,
            seed=99,
        )
        self.assertEqual(with_adjustment["dixon_coles_rho"], -0.2)
        self.assertNotEqual(
            without_adjustment["outcomes"]["draw"]["percentage"],
            with_adjustment["outcomes"]["draw"]["percentage"],
        )

    def test_invalid_simulation_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "At least one scenario"):
            simulate_match(1.2, 1.0, [], simulations=1_000)
        with self.assertRaisesRegex(ValueError, "positive probability"):
            simulate_match(1.2, 1.0, [{"weight": 0}], simulations=1_000)
        with self.assertRaisesRegex(ValueError, "from 1,000 to 100,000"):
            simulate_match(1.2, 1.0, [{"weight": 100}], simulations=999)
        with self.assertRaisesRegex(ValueError, "between 0 and 0.5"):
            simulate_match(
                1.2, 1.0, [{"weight": 100}], simulations=1_000, parameter_cv=0.6
            )


if __name__ == "__main__":
    unittest.main()
