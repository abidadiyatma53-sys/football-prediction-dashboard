"""Scenario-based Monte Carlo simulation for football score predictions."""

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


def simulate_match(
    lambda_home: float,
    lambda_away: float,
    scenarios: Sequence[Mapping[str, Any]],
    simulations: int = 20_000,
    parameter_cv: float = 0.10,
    rho: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Simulate score outcomes across weighted scenarios and uncertain goal rates.

    Scenario weights express user-supplied probabilities. Goal-rate multipliers
    express user-supplied lineup or tactical adjustments; they are not estimated
    player effects. ``parameter_cv`` is a stress-test assumption, not a calibrated
    confidence interval.
    """
    if not np.isfinite(lambda_home) or not np.isfinite(lambda_away):
        raise ValueError("Expected goals must be finite numbers.")
    if lambda_home <= 0 or lambda_away <= 0:
        raise ValueError("Expected goals must be greater than zero.")
    if not isinstance(simulations, int) or not 1_000 <= simulations <= 100_000:
        raise ValueError("Simulation count must be an integer from 1,000 to 100,000.")
    if not np.isfinite(parameter_cv) or not 0 <= parameter_cv <= 0.5:
        raise ValueError("Parameter uncertainty (CV) must be between 0 and 0.5.")
    if rho is not None and (not np.isfinite(rho) or not -0.25 <= rho <= 0.15):
        raise ValueError("Dixon-Coles rho must be between -0.25 and 0.15.")
    if not scenarios:
        raise ValueError("At least one scenario is required.")

    weights = []
    home_multipliers = []
    away_multipliers = []
    scenario_names = []
    for index, scenario in enumerate(scenarios, start=1):
        weight = float(scenario.get("weight", 0))
        home_multiplier = float(scenario.get("home_multiplier", 1))
        away_multiplier = float(scenario.get("away_multiplier", 1))
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"Scenario {index} weight must be finite and non-negative.")
        if not np.isfinite(home_multiplier) or not 0.1 <= home_multiplier <= 3:
            raise ValueError(f"Scenario {index} home multiplier must be between 0.1 and 3.")
        if not np.isfinite(away_multiplier) or not 0.1 <= away_multiplier <= 3:
            raise ValueError(f"Scenario {index} away multiplier must be between 0.1 and 3.")
        weights.append(weight)
        home_multipliers.append(home_multiplier)
        away_multipliers.append(away_multiplier)
        scenario_names.append(str(scenario.get("name") or f"Scenario {index}"))

    weights_array = np.asarray(weights, dtype=float)
    if weights_array.sum() <= 0:
        raise ValueError("At least one scenario must have a positive probability weight.")
    weights_array /= weights_array.sum()

    rng = np.random.default_rng(seed)
    chosen_scenarios = rng.choice(len(scenarios), size=simulations, p=weights_array)
    lambda_h = lambda_home * np.asarray(home_multipliers)[chosen_scenarios]
    lambda_a = lambda_away * np.asarray(away_multipliers)[chosen_scenarios]

    if parameter_cv:
        sigma_squared = np.log1p(parameter_cv**2)
        sigma = np.sqrt(sigma_squared)
        lambda_h = rng.lognormal(np.log(lambda_h) - sigma_squared / 2, sigma)
        lambda_a = rng.lognormal(np.log(lambda_a) - sigma_squared / 2, sigma)

    home_goals, away_goals = _draw_scores(
        rng, lambda_h, lambda_a, rho if rho is not None else 0.0
    )

    outcome_labels = np.where(
        home_goals > away_goals,
        "home_win",
        np.where(home_goals == away_goals, "draw", "away_win"),
    )
    outcome_counts = {
        outcome: int(np.count_nonzero(outcome_labels == outcome))
        for outcome in ("home_win", "draw", "away_win")
    }
    score_keys = home_goals.astype(str) + " - " + away_goals.astype(str)
    unique_scores, score_counts = np.unique(score_keys, return_counts=True)
    order = np.argsort(score_counts)[::-1]
    top_scores = [
        {
            "score": str(unique_scores[position]),
            "count": int(score_counts[position]),
            "percentage": round(float(score_counts[position] / simulations * 100), 2),
        }
        for position in order[:10]
    ]

    scenario_results = []
    for index, name in enumerate(scenario_names):
        selected = chosen_scenarios == index
        count = int(np.count_nonzero(selected))
        if not count:
            continue
        scenario_results.append({
            "scenario": name,
            "weight_pct": round(float(weights_array[index] * 100), 2),
            "simulations": count,
            "home_win_pct": round(float(np.mean(home_goals[selected] > away_goals[selected]) * 100), 2),
            "draw_pct": round(float(np.mean(home_goals[selected] == away_goals[selected]) * 100), 2),
            "away_win_pct": round(float(np.mean(home_goals[selected] < away_goals[selected]) * 100), 2),
            "mean_home_goals": round(float(np.mean(home_goals[selected])), 3),
            "mean_away_goals": round(float(np.mean(away_goals[selected])), 3),
        })

    return {
        "simulations": simulations,
        "parameter_cv_pct": round(parameter_cv * 100, 1),
        "dixon_coles_rho": rho,
        "scenario_probabilities_normalized": True,
        "outcomes": {
            outcome: {
                "count": count,
                "percentage": round(count / simulations * 100, 2),
                "ci95_pct": _wilson_interval(count, simulations),
            }
            for outcome, count in outcome_counts.items()
        },
        "mean_home_goals": round(float(np.mean(home_goals)), 3),
        "mean_away_goals": round(float(np.mean(away_goals)), 3),
        "home_goals_90th_percentile": int(np.quantile(home_goals, 0.9)),
        "away_goals_90th_percentile": int(np.quantile(away_goals, 0.9)),
        "top_scorelines": top_scores,
        "scenario_results": scenario_results,
    }


def _draw_scores(
    rng: np.random.Generator,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw independent Poisson scores, rejection-corrected for Dixon-Coles."""
    count = len(lambda_home)
    home_goals = rng.poisson(lambda_home)
    away_goals = rng.poisson(lambda_away)
    if rho == 0:
        return home_goals, away_goals

    correction = _tau_adjustments(home_goals, away_goals, lambda_home, lambda_away, rho)
    max_correction = _maximum_tau(lambda_home, lambda_away, rho)
    accepted = rng.random(count) < correction / max_correction
    while not np.all(accepted):
        pending = np.flatnonzero(~accepted)
        candidate_home = rng.poisson(lambda_home[pending])
        candidate_away = rng.poisson(lambda_away[pending])
        correction = _tau_adjustments(
            candidate_home,
            candidate_away,
            lambda_home[pending],
            lambda_away[pending],
            rho,
        )
        accepted_now = rng.random(len(pending)) < correction / max_correction[pending]
        home_goals[pending[accepted_now]] = candidate_home[accepted_now]
        away_goals[pending[accepted_now]] = candidate_away[accepted_now]
        accepted[pending[accepted_now]] = True
    return home_goals, away_goals


def _tau_adjustments(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: float,
) -> np.ndarray:
    tau = np.ones(len(home_goals), dtype=float)
    zero_zero = (home_goals == 0) & (away_goals == 0)
    zero_one = (home_goals == 0) & (away_goals == 1)
    one_zero = (home_goals == 1) & (away_goals == 0)
    one_one = (home_goals == 1) & (away_goals == 1)
    tau[zero_zero] = np.maximum(
        0.001, 1 - lambda_home[zero_zero] * lambda_away[zero_zero] * rho
    )
    tau[zero_one] = np.maximum(0.001, 1 + lambda_home[zero_one] * rho)
    tau[one_zero] = np.maximum(0.001, 1 + lambda_away[one_zero] * rho)
    tau[one_one] = max(0.001, 1 - rho)
    return tau


def _maximum_tau(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: float,
) -> np.ndarray:
    return np.maximum.reduce([
        np.ones(len(lambda_home), dtype=float),
        np.maximum(0.001, 1 - lambda_home * lambda_away * rho),
        np.maximum(0.001, 1 + lambda_home * rho),
        np.maximum(0.001, 1 + lambda_away * rho),
        np.full(len(lambda_home), max(0.001, 1 - rho)),
    ])


def _wilson_interval(successes: int, trials: int) -> List[float]:
    """95% Wilson score interval, expressed as percentages."""
    if trials <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(
        proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)
    ) / denominator
    return [round(float(max(0, center - margin) * 100), 2),
            round(float(min(1, center + margin) * 100), 2)]
