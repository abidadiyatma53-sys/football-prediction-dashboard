"""
World Football Dynamic Elo Rating Engine.
Calculates leak-free historical team ratings, opponent-adjusted strength,
home advantage adjustments, and match win expectancies based on FIFA & FiveThirtyEight algorithms.
"""

import math
import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from normalizer import normalize_team_name

logger = logging.getLogger(__name__)


class FootballEloEngine:
    """
    World Football Elo Rating System:
    - Base Rating: 1500
    - Home Field Advantage: +65 Elo points
    - K-Factor: 20
    - Goal Difference Multiplier: Scales with margin of victory
    - Zero Leakage: Match features strictly utilize pre-kickoff Elo ratings.
    """

    def __init__(
        self,
        base_elo: float = 1500.0,
        k_factor: float = 20.0,
        home_advantage: float = 65.0
    ):
        self.base_elo = base_elo
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.current_ratings: Dict[str, float] = {}
        self.rating_history: Dict[str, List[Dict[str, Any]]] = {}

    def get_goal_difference_multiplier(self, goal_diff: int) -> float:
        """
        Scales rating impact based on margin of victory (FiveThirtyEight & World Football Elo formula).
        """
        abs_diff = abs(goal_diff)
        if abs_diff <= 1:
            return 1.0
        elif abs_diff == 2:
            return 1.5
        elif abs_diff == 3:
            return 1.75
        else:
            return 1.75 + (abs_diff - 3) / 8.0

    def calculate_win_expectancy(self, rating_diff: float) -> float:
        """
        Calculates expected score outcome We between 0.0 and 1.0:
        We = 1 / (10^(-rating_diff / 400) + 1)
        """
        return 1.0 / (1.0 + math.pow(10.0, -rating_diff / 400.0))

    def calculate_1x2_probabilities(self, home_elo: float, away_elo: float) -> Dict[str, float]:
        """
        Derives discrete 1X2 probabilities (Home, Draw, Away) from Elo ratings.
        Draw probability peaks around 27% when ratings are equal.
        """
        effective_diff = (home_elo + self.home_advantage) - away_elo
        we_home = self.calculate_win_expectancy(effective_diff)

        # Baseline draw probability formula calibrated for European league football
        # Equal teams draw ~26-28% of the time, heavily mismatched teams draw ~15-18%
        diff_magnitude = abs(effective_diff)
        p_draw = max(0.12, 0.27 - (diff_magnitude / 2500.0))

        remaining = 1.0 - p_draw
        p_home = round(we_home * remaining, 4)
        p_away = round((1.0 - we_home) * remaining, 4)
        p_draw = round(1.0 - (p_home + p_away), 4)

        return {
            "home_win": round(p_home * 100.0, 2),
            "draw": round(p_draw * 100.0, 2),
            "away_win": round(p_away * 100.0, 2)
        }

    def compute_season_elo(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Processes matches chronologically and attaches pre-match Elo ratings.
        Zero data leakage: features at row T strictly record ratings BEFORE row T.
        """
        df_sorted = df.sort_values(by="Date", kind="stable").reset_index(drop=True)
        
        # Initialize ratings
        teams = set(df_sorted["HomeTeam"].unique()).union(set(df_sorted["AwayTeam"].unique()))
        ratings = {t: self.base_elo for t in teams}
        self.rating_history = {t: [{"date": "Initial", "elo": self.base_elo}] for t in teams}

        home_elos = [self.base_elo] * len(df_sorted)
        away_elos = [self.base_elo] * len(df_sorted)
        elo_diffs = [0.0] * len(df_sorted)
        elo_home_win_probs = [0.0] * len(df_sorted)

        # Treat all fixtures on the same calendar date as simultaneous because
        # kickoff times are not consistently available in the historical data.
        for match_date, date_matches in df_sorted.groupby("Date", sort=False):
            updates = []
            for idx, row in date_matches.iterrows():
                h = row["HomeTeam"]
                a = row["AwayTeam"]

                h_elo_before = ratings.get(h, self.base_elo)
                a_elo_before = ratings.get(a, self.base_elo)
                home_elos[idx] = round(h_elo_before, 1)
                away_elos[idx] = round(a_elo_before, 1)

                eff_diff = (h_elo_before + self.home_advantage) - a_elo_before
                elo_diffs[idx] = round(eff_diff, 1)
                elo_home_win_probs[idx] = self.calculate_1x2_probabilities(
                    h_elo_before, a_elo_before
                )["home_win"]

                if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
                    continue

                gd = int(row["FTHG"]) - int(row["FTAG"])
                w_home = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
                we_home = self.calculate_win_expectancy(eff_diff)
                g_mult = self.get_goal_difference_multiplier(gd)
                delta_h = self.k_factor * g_mult * (w_home - we_home)
                delta_a = -delta_h
                updates.append((h, a, delta_h, delta_a))

            for h, a, delta_h, delta_a in updates:
                ratings[h] += delta_h
                ratings[a] += delta_a
                self.rating_history[h].append({"date": match_date, "elo": round(ratings[h], 1)})
                self.rating_history[a].append({"date": match_date, "elo": round(ratings[a], 1)})

        df_result = df_sorted.copy()
        df_result["home_elo"] = home_elos
        df_result["away_elo"] = away_elos
        df_result["diff_elo"] = elo_diffs
        df_result["elo_home_win_prob"] = elo_home_win_probs

        self.current_ratings = ratings
        return df_result, ratings

    def get_rankings_table(self) -> pd.DataFrame:
        """Returns sorted Elo power rankings table across all clubs."""
        records = []
        for team, elo in self.current_ratings.items():
            records.append({
                "Team": team,
                "Elo_Rating": round(elo, 1),
                "Rating_Change": round(elo - self.base_elo, 1)
            })

        df_rankings = pd.DataFrame(records)
        if not df_rankings.empty:
            df_rankings = df_rankings.sort_values(by="Elo_Rating", ascending=False).reset_index(drop=True)
            df_rankings["Rank"] = df_rankings.index + 1
            df_rankings = df_rankings[["Rank", "Team", "Elo_Rating", "Rating_Change"]]
        return df_rankings

    def get_team_current_elo(self, team: str) -> float:
        """Returns current Elo for a specific club."""
        norm = normalize_team_name(team)
        return round(self.current_ratings.get(norm, self.base_elo), 1)
