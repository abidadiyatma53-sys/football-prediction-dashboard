"""
Value Betting & Odds Comparison Engine.
Identifies Positive Expected Value (+EV) opportunities by comparing
model-derived win probabilities against bookmaker/market odds,
and calculates optimal bankroll allocation via Fractional Kelly Criterion.
"""

from typing import Dict, Any, List, Optional
import math


class ValueBettingEngine:
    """
    Mathematical Engine for finding Value Bets and sizing stakes.
    
    Formulas:
    - Fair Odds: 1.0 / P(outcome)
    - Expected Value (EV): (P(outcome) * Market_Odds) - 1.0
    - Full Kelly Fraction: (Odds * P - 1) / (Odds - 1)
    - Fractional Kelly: Kelly Fraction * Fraction (default 0.25 for Quarter Kelly)
    """

    def __init__(self, default_kelly_fraction: float = 0.25, min_ev_threshold: float = 0.03):
        """
        Args:
            default_kelly_fraction: Multiplier for Kelly stake sizing (0.25 = quarter Kelly).
            min_ev_threshold: Minimum positive EV (e.g., 0.03 = +3%) to flag as high-confidence value.
        """
        self.kelly_multiplier = default_kelly_fraction
        self.min_ev_threshold = min_ev_threshold

    @staticmethod
    def calculate_fair_odds(probability: float) -> float:
        """Calculates fair decimal odds from true probability."""
        if probability <= 0.0:
            return 999.0
        return round(1.0 / probability, 2)

    @staticmethod
    def calculate_expected_value(probability: float, market_odds: float) -> float:
        """
        Calculates decimal Expected Value.
        Example: p = 0.60, market_odds = 2.10 -> EV = (0.60 * 2.10) - 1 = +0.26 (+26%)
        """
        if market_odds <= 1.0 or probability <= 0.0:
            return -1.0
        return (probability * market_odds) - 1.0

    def calculate_kelly_stake(self, probability: float, market_odds: float) -> float:
        """
        Calculates recommended bankroll allocation percentage using Fractional Kelly.
        Returns percentage (0.0 to 100.0).
        """
        if market_odds <= 1.0 or probability <= 0.0:
            return 0.0

        b = market_odds - 1.0
        p = probability
        q = 1.0 - p

        full_kelly = (b * p - q) / b
        if full_kelly <= 0.0:
            return 0.0

        fractional_kelly = full_kelly * self.kelly_multiplier
        # Cap max recommended stake at 5% of bankroll for strict bankroll risk preservation
        capped_stake = min(fractional_kelly, 0.05)
        return round(capped_stake * 100.0, 2)

    def analyze_market_odds(
        self,
        model_probabilities: Dict[str, float],
        market_odds: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Compares model probabilities against available market odds.

        Args:
            model_probabilities: Dict e.g. {'home_win': 0.52, 'draw': 0.28, 'away_win': 0.20}
                                (in decimal form 0.0 to 1.0 or percentage 0 to 100).
            market_odds: Dict e.g. {'home_win': 2.15, 'draw': 3.40, 'away_win': 3.60}

        Returns:
            List of analyzed betting options sorted by EV descending.
        """
        results = []

        for outcome, odds in market_odds.items():
            if outcome not in model_probabilities:
                continue

            raw_p = model_probabilities[outcome]
            # Convert percentage to decimal if needed
            p = raw_p / 100.0 if raw_p > 1.0 else raw_p
            p = max(0.0001, min(0.9999, p))

            fair_odds = self.calculate_fair_odds(p)
            ev = self.calculate_expected_value(p, odds)
            ev_pct = round(ev * 100.0, 2)
            edge_pct = round((p - (1.0 / odds)) * 100.0, 2)
            kelly_pct = self.calculate_kelly_stake(p, odds)
            is_value = ev >= self.min_ev_threshold

            results.append({
                "outcome": outcome,
                "label": outcome.replace("_", " ").title(),
                "model_prob_pct": round(p * 100.0, 1),
                "market_implied_prob_pct": round((1.0 / odds) * 100.0, 1),
                "fair_odds": fair_odds,
                "market_odds": round(odds, 2),
                "edge_pct": edge_pct,
                "ev_pct": ev_pct,
                "is_value_bet": is_value,
                "recommended_stake_pct": kelly_pct if is_value else 0.0,
                "verdict": "POSITIVE VALUE (+EV)" if is_value else ("FAIR" if ev >= 0 else "NEGATIVE VALUE (-EV)")
            })

        # Sort by highest Expected Value first
        results.sort(key=lambda x: x["ev_pct"], reverse=True)
        return results
