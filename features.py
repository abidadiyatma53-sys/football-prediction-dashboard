"""
Feature Engineering Module for Football Match Prediction.
Extracts team attack strength, defense weakness, recent form, and rolling xG metrics
without data leakage (strictly chronological using shifted rolling windows).
"""

import logging
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

from normalizer import normalize_team_name
from elo import FootballEloEngine

logger = logging.getLogger(__name__)


class FootballFeatureEngineer:
    """
    Engineers predictive features from historical match data:
    1. Attack Strength & Defense Weakness (rolling 5 & 10 matches).
    2. Form (points accumulated & goal difference in last 5 matches).
    3. Expected Goals (rolling average xG and xGA).
    4. Home/Away split specific advantages.
    5. Dynamic World Football Elo Ratings (leak-free opponent-adjusted strength).
    6. Zero Data Leakage: strictly applies shift(1) so a match on date T
       only uses data from prior matches.
    """

    def __init__(self, rolling_window_short: int = 5, rolling_window_long: int = 10):
        self.w_short = rolling_window_short
        self.w_long = rolling_window_long
        self.elo_engine = FootballEloEngine()

    def prepare_team_match_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms match-level data into a long-form team-match log
        (two rows per match: one for home team, one for away team).
        """
        # Ensure chronological ordering
        df_sorted = df.sort_values(by="Date", kind="stable").reset_index(drop=True)

        home_records = []
        away_records = []

        for idx, row in df_sorted.iterrows():
            fthg = int(row["FTHG"])
            ftag = int(row["FTAG"])
            xg_h = float(row.get("xG_home", 0.0))
            xg_a = float(row.get("xG_away", 0.0))
            hs = pd.to_numeric(pd.Series([row.get("HS")]), errors="coerce").iloc[0]
            as_ = pd.to_numeric(pd.Series([row.get("AS")]), errors="coerce").iloc[0]
            hst = pd.to_numeric(pd.Series([row.get("HST")]), errors="coerce").iloc[0]
            ast = pd.to_numeric(pd.Series([row.get("AST")]), errors="coerce").iloc[0]

            # Home team view
            home_pts = 3 if fthg > ftag else (1 if fthg == ftag else 0)
            home_records.append({
                "match_idx": idx,
                "Date": row["Date"],
                "League": row.get("League", "Unknown"),
                "Team": row["HomeTeam"],
                "Opponent": row["AwayTeam"],
                "is_home": 1,
                "goals_scored": fthg,
                "goals_conceded": ftag,
                "xg_for": xg_h,
                "xg_against": xg_a,
                "shots_for": hs,
                "shots_against": as_,
                "sot_for": hst,
                "sot_against": ast,
                "points": home_pts
            })

            # Away team view
            away_pts = 3 if ftag > fthg else (1 if ftag == fthg else 0)
            away_records.append({
                "match_idx": idx,
                "Date": row["Date"],
                "League": row.get("League", "Unknown"),
                "Team": row["AwayTeam"],
                "Opponent": row["HomeTeam"],
                "is_home": 0,
                "goals_scored": ftag,
                "goals_conceded": fthg,
                "xg_for": xg_a,
                "xg_against": xg_h,
                "shots_for": as_,
                "shots_against": hs,
                "sot_for": ast,
                "sot_against": hst,
                "points": away_pts
            })

        team_logs = pd.concat([pd.DataFrame(home_records), pd.DataFrame(away_records)], ignore_index=True)
        team_logs = team_logs.sort_values(by=["Date", "match_idx"]).reset_index(drop=True)
        return team_logs

    def build_team_rolling_features(self, team_logs: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rolling statistics per team with shift(1) to avoid lookahead bias.
        """
        team_groups = team_logs.groupby("Team", group_keys=False)

        # Estimate league goal rates from prior fixtures only.
        league_date_goals = (
            team_logs.groupby(["League", "Date", "match_idx"], sort=False)["goals_scored"]
            .mean()
            .rename("match_goal_avg")
            .reset_index()
            .groupby(["League", "Date"], as_index=False, sort=False)["match_goal_avg"]
            .mean()
            .sort_values("Date", kind="stable")
        )
        league_date_goals["prior_league_goal_avg"] = (
            league_date_goals.groupby("League", sort=False)["match_goal_avg"]
            .transform(lambda values: values.shift(1).expanding(min_periods=1).mean())
            .fillna(1.35)
        )
        team_logs = team_logs.merge(
            league_date_goals[["League", "Date", "prior_league_goal_avg"]],
            on=["League", "Date"],
            how="left",
            validate="many_to_one"
        )
        team_logs = team_logs.sort_values(
            ["Date", "match_idx"], kind="stable"
        ).reset_index(drop=True)
        team_groups = team_logs.groupby("Team", group_keys=False)
        league_overall_avg = team_logs["prior_league_goal_avg"].clip(lower=0.1)

        # 1. Form features (Last 5 matches)
        team_logs["form_pts_5"] = team_groups["points"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).sum()
        ).fillna(1.0)

        team_logs["goal_diff"] = team_logs["goals_scored"] - team_logs["goals_conceded"]
        team_logs["form_gd_5"] = team_groups["goal_diff"].transform(
            lambda values: values.shift(1).rolling(self.w_short, min_periods=1).sum()
        ).fillna(0.0)

        # 2. Rolling Goals (Attack & Defense)
        team_logs["rolling_scored_5"] = team_groups["goals_scored"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).mean()
        ).fillna(league_overall_avg)

        team_logs["rolling_conceded_5"] = team_groups["goals_conceded"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).mean()
        ).fillna(league_overall_avg)

        team_logs["rolling_scored_10"] = team_groups["goals_scored"].transform(
            lambda x: x.shift(1).rolling(self.w_long, min_periods=1).mean()
        ).fillna(league_overall_avg)

        team_logs["rolling_conceded_10"] = team_groups["goals_conceded"].transform(
            lambda x: x.shift(1).rolling(self.w_long, min_periods=1).mean()
        ).fillna(league_overall_avg)

        # Attack Strength & Defense Weakness ratios
        # Strength > 1.0 = above average attacking; Weakness > 1.0 = concedes more than average
        team_logs["attack_strength_10"] = (team_logs["rolling_scored_10"] / league_overall_avg).clip(0.3, 3.0)
        team_logs["defense_weakness_10"] = (team_logs["rolling_conceded_10"] / league_overall_avg).clip(0.3, 3.0)

        # 3. Rolling Expected Goals (xG & xGA)
        team_logs["rolling_xg_for_5"] = team_groups["xg_for"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).mean()
        ).fillna(league_overall_avg)

        team_logs["rolling_xg_against_5"] = team_groups["xg_against"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).mean()
        ).fillna(league_overall_avg)

        team_logs["rolling_xg_diff_5"] = team_logs["rolling_xg_for_5"] - team_logs["rolling_xg_against_5"]

        # 4. Rolling Shots on Target
        team_logs["rolling_sot_for_5"] = team_groups["sot_for"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).mean()
        ).fillna(4.0)

        team_logs["rolling_sot_against_5"] = team_groups["sot_against"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).mean()
        ).fillna(4.0)

        # 5. Rest Days & Schedule Congestion (Zero-leakage: calendar days since previous match)
        team_logs["Date_dt"] = pd.to_datetime(team_logs["Date"])
        team_logs["rest_days"] = team_groups["Date_dt"].diff().dt.days.fillna(7.0).clip(lower=2.0, upper=14.0)
        team_logs["is_congested"] = (team_logs["rest_days"] <= 3.0).astype(float)

        return team_logs

    def merge_features_to_matches(self, df_matches: pd.DataFrame, team_logs: pd.DataFrame) -> pd.DataFrame:
        """
        Merges team-level rolling features back into the match-level dataset,
        including venue-specific home advantage and away travel fatigue.
        """
        home_feats = team_logs[team_logs["is_home"] == 1].copy()
        away_feats = team_logs[team_logs["is_home"] == 0].copy()

        # Compute Venue-Specific Rolling Form with strict shift(1)
        home_feats["goal_diff"] = home_feats["goals_scored"] - home_feats["goals_conceded"]
        h_groups = home_feats.groupby("Team", group_keys=False)
        home_feats["venue_form_pts_5"] = h_groups["points"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).sum()
        ).fillna(7.0)
        home_feats["venue_form_gd_5"] = h_groups["goal_diff"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).sum()
        ).fillna(0.0)
        home_feats["venue_scored_5"] = h_groups["goals_scored"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).mean()
        ).fillna(1.5)
        home_feats["venue_conceded_5"] = h_groups["goals_conceded"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).mean()
        ).fillna(1.2)

        away_feats["goal_diff"] = away_feats["goals_scored"] - away_feats["goals_conceded"]
        a_groups = away_feats.groupby("Team", group_keys=False)
        away_feats["venue_form_pts_5"] = a_groups["points"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).sum()
        ).fillna(5.0)
        away_feats["venue_form_gd_5"] = a_groups["goal_diff"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).sum()
        ).fillna(0.0)
        away_feats["venue_scored_5"] = a_groups["goals_scored"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).mean()
        ).fillna(1.2)
        away_feats["venue_conceded_5"] = a_groups["goals_conceded"].transform(
            lambda x: x.shift(1).rolling(self.w_short, min_periods=1).mean()
        ).fillna(1.5)

        feature_cols = [
            "form_pts_5", "form_gd_5", "rolling_scored_5", "rolling_conceded_5",
            "attack_strength_10", "defense_weakness_10",
            "rolling_xg_for_5", "rolling_xg_against_5", "rolling_xg_diff_5",
            "rolling_sot_for_5", "rolling_sot_against_5",
            "rest_days", "is_congested",
            "venue_form_pts_5", "venue_form_gd_5", "venue_scored_5", "venue_conceded_5"
        ]

        # Rename for home and away
        home_cols_map = {col: f"home_{col}" for col in feature_cols}
        away_cols_map = {col: f"away_{col}" for col in feature_cols}

        home_feats_clean = home_feats[["match_idx"] + feature_cols].rename(columns=home_cols_map)
        away_feats_clean = away_feats[["match_idx"] + feature_cols].rename(columns=away_cols_map)

        df_enriched = df_matches.reset_index(names="match_idx")
        df_enriched = df_enriched.merge(home_feats_clean, on="match_idx", how="left")
        df_enriched = df_enriched.merge(away_feats_clean, on="match_idx", how="left")

        # Interaction Features
        df_enriched["diff_form_pts_5"] = df_enriched["home_form_pts_5"] - df_enriched["away_form_pts_5"]
        df_enriched["diff_form_gd_5"] = df_enriched["home_form_gd_5"] - df_enriched["away_form_gd_5"]
        df_enriched["diff_rolling_xg_5"] = df_enriched["home_rolling_xg_for_5"] - df_enriched["away_rolling_xg_for_5"]
        df_enriched["home_attack_vs_away_defense"] = (
            df_enriched["home_attack_strength_10"] * df_enriched["away_defense_weakness_10"]
        )
        df_enriched["away_attack_vs_home_defense"] = (
            df_enriched["away_attack_strength_10"] * df_enriched["home_defense_weakness_10"]
        )

        # Contextual Rest & Venue Interaction Features
        df_enriched["diff_rest_days"] = df_enriched["home_rest_days"] - df_enriched["away_rest_days"]
        df_enriched["diff_venue_form_pts_5"] = df_enriched["home_venue_form_pts_5"] - df_enriched["away_venue_form_pts_5"]
        df_enriched["diff_venue_form_gd_5"] = df_enriched["home_venue_form_gd_5"] - df_enriched["away_venue_form_gd_5"]

        df_enriched = df_enriched.drop(columns=["match_idx"])
        return df_enriched

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes end-to-end feature engineering on the match dataset.
        """
        df = df.sort_values(by="Date", kind="stable").reset_index(drop=True)
        logger.info("Extracting team match logs...")
        team_logs = self.prepare_team_match_logs(df)

        logger.info("Computing rolling attack, defense, form, and xG features...")
        team_logs = self.build_team_rolling_features(team_logs)

        logger.info("Merging features back to match format...")
        df_features = self.merge_features_to_matches(df, team_logs)

        logger.info("Computing dynamic World Football Elo ratings...")
        df_elo, _ = self.elo_engine.compute_season_elo(df)
        df_features["home_elo"] = df_elo["home_elo"].values
        df_features["away_elo"] = df_elo["away_elo"].values
        df_features["diff_elo"] = df_elo["diff_elo"].values

        return df_features

    def get_latest_team_features(
        self,
        df_features: pd.DataFrame,
        team_name: str,
        is_home: Optional[bool] = None,
        match_date: Optional[str] = None
    ) -> dict:
        """
        Retrieves the most recent statistical state for a given team
        to use during live inference/prediction, factoring in home/away venue context.
        """
        norm_name = normalize_team_name(team_name)
        target_date = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        if match_date:
            target_date = pd.to_datetime(match_date, errors="raise")
            if target_date.tzinfo is not None:
                target_date = target_date.tz_convert("UTC").tz_localize(None)
            target_date = target_date.normalize()

        team_matches = df_features[
            (df_features["HomeTeam"] == norm_name) | (df_features["AwayTeam"] == norm_name)
        ].copy()
        team_matches["_feature_date"] = pd.to_datetime(team_matches["Date"], errors="raise")
        team_matches = team_matches[
            team_matches["_feature_date"] < target_date
        ].sort_values(by="_feature_date")

        if team_matches.empty:
            logger.warning(f"No match history found for team: {norm_name}. Using baseline defaults.")
            return {
                "form_pts_5": 7.0,
                "form_gd_5": 0.0,
                "rolling_scored_5": 1.4,
                "rolling_conceded_5": 1.4,
                "attack_strength_10": 1.0,
                "defense_weakness_10": 1.0,
                "rolling_xg_for_5": 1.35,
                "rolling_xg_against_5": 1.35,
                "rolling_xg_diff_5": 0.0,
                "rolling_sot_for_5": 4.5,
                "rolling_sot_against_5": 4.5,
                "rest_days": 7.0,
                "is_congested": 0.0,
                "venue_form_pts_5": 7.0 if (is_home is None or is_home) else 5.0,
                "venue_form_gd_5": 0.0,
                "venue_scored_5": 1.5 if (is_home is None or is_home) else 1.2,
                "venue_conceded_5": 1.2 if (is_home is None or is_home) else 1.5,
                "elo": 1500.0,
            }

        team_log_rows = []
        for _, match in team_matches.iterrows():
            was_home = match["HomeTeam"] == norm_name
            goals_for = int(match["FTHG"] if was_home else match["FTAG"])
            goals_against = int(match["FTAG"] if was_home else match["FTHG"])
            team_log_rows.append({
                "Date": match["Date"],
                "is_home": was_home,
                "goals_scored": goals_for,
                "goals_conceded": goals_against,
                "points": 3 if goals_for > goals_against else (1 if goals_for == goals_against else 0),
                "xg_for": float(match.get("xG_home" if was_home else "xG_away", 0.0) or 0.0),
                "xg_against": float(match.get("xG_away" if was_home else "xG_home", 0.0) or 0.0),
                "sot_for": float(match.get("HST" if was_home else "AST", 0.0) or 0.0),
                "sot_against": float(match.get("AST" if was_home else "HST", 0.0) or 0.0),
            })

        history = pd.DataFrame(team_log_rows).sort_values("Date")
        recent = history.tail(self.w_short)
        recent_long = history.tail(self.w_long)
        latest_match = team_matches.iloc[-1]
        league = latest_match.get("League")
        league_matches = df_features[
            (df_features["League"] == league)
            & (pd.to_datetime(df_features["Date"], errors="raise") < target_date)
        ]
        league_home_avg = (
            float(league_matches["FTHG"].mean()) if not league_matches.empty else 1.55
        )
        league_away_avg = (
            float(league_matches["FTAG"].mean()) if not league_matches.empty else 1.25
        )
        league_goal_avg = max((league_home_avg + league_away_avg) / 2.0, 0.1)
        last_match_date = latest_match["_feature_date"]
        rest_days = int((target_date - last_match_date).days)
        rest_days = max(2, min(14, rest_days))

        result = {
            "league": str(league),
            "league_home_avg": league_home_avg,
            "league_away_avg": league_away_avg,
            "form_pts_5": float(recent["points"].sum()),
            "form_gd_5": float((recent["goals_scored"] - recent["goals_conceded"]).sum()),
            "rolling_scored_5": float(recent["goals_scored"].mean()),
            "rolling_conceded_5": float(recent["goals_conceded"].mean()),
            "attack_strength_10": float(np.clip(recent_long["goals_scored"].mean() / league_goal_avg, 0.3, 3.0)),
            "defense_weakness_10": float(np.clip(recent_long["goals_conceded"].mean() / league_goal_avg, 0.3, 3.0)),
            "rolling_xg_for_5": float(recent["xg_for"].mean()),
            "rolling_xg_against_5": float(recent["xg_against"].mean()),
            "rolling_xg_diff_5": float((recent["xg_for"] - recent["xg_against"]).mean()),
            "rolling_sot_for_5": float(recent["sot_for"].mean()),
            "rolling_sot_against_5": float(recent["sot_against"].mean()),
            "rest_days": float(rest_days),
            "is_congested": float(rest_days <= 3),
        }

        venue_is_home = (
            bool(team_matches.iloc[-1]["HomeTeam"] == norm_name)
            if is_home is None else is_home
        )
        venue_matches = history[history["is_home"] == venue_is_home].tail(self.w_short)
        result.update({
            "venue_form_pts_5": float(venue_matches["points"].sum()) if not venue_matches.empty else (7.0 if is_home is not False else 5.0),
            "venue_form_gd_5": float((venue_matches["goals_scored"] - venue_matches["goals_conceded"]).sum()),
            "venue_scored_5": float(venue_matches["goals_scored"].mean()) if not venue_matches.empty else (1.5 if is_home is not False else 1.2),
            "venue_conceded_5": float(venue_matches["goals_conceded"].mean()) if not venue_matches.empty else (1.2 if is_home is not False else 1.5),
        })
        latest_elo_column = "home_elo" if bool(latest_match["HomeTeam"] == norm_name) else "away_elo"
        latest_dataset_date = pd.to_datetime(df_features["Date"], errors="raise").max()
        if target_date > latest_dataset_date:
            result["elo"] = self.elo_engine.get_team_current_elo(norm_name)
        else:
            result["elo"] = float(latest_match.get(latest_elo_column, 1500.0))
        return result
