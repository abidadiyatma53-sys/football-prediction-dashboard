"""
Data Cleaning and Preparation Module.
Performs data validation, missing value imputation, deduplication,
schema enforcement, and sanity checks on football match data.
"""

import logging
from typing import List, Optional
import pandas as pd
import numpy as np

import config
from normalizer import normalize_team_name

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Standardizes raw ingested football data into high-quality, modeling-ready datasets:
    - Removes duplicates and malformed records.
    - Imputes or resolves missing values.
    - Normalizes team names across sources.
    - Enforces strict data types.
    - Executes domain-specific sanity checks (e.g. shots on target <= total shots).
    """

    def __init__(self, target_columns: Optional[List[str]] = None):
        self.target_columns = target_columns or config.TARGET_COLUMNS

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the full cleaning pipeline on the input DataFrame.

        Args:
            df: Raw combined DataFrame.

        Returns:
            Clean, validated, and normalized DataFrame.
        """
        if df.empty:
            logger.warning("Cleaning pipeline received an empty DataFrame.")
            return pd.DataFrame(columns=self.target_columns)

        logger.info(f"Starting data cleaning on {len(df)} initial records...")
        cleaned = df.copy()

        # Step 1: Ensure required columns exist
        cleaned = self._ensure_required_columns(cleaned)

        # Step 2: Drop invalid rows (missing primary keys)
        cleaned = self._filter_invalid_rows(cleaned)

        # Step 3: Normalize team names
        cleaned = self._normalize_teams(cleaned)

        # Step 4: Sanity check: HomeTeam cannot equal AwayTeam
        cleaned = cleaned[cleaned["HomeTeam"] != cleaned["AwayTeam"]].copy()

        # Step 5: Enforce Date format
        cleaned = self._format_dates(cleaned)

        # Step 6: Cast and validate numeric columns
        cleaned = self._cast_and_validate_numerics(cleaned)

        # Step 7: Calculate or validate Match Result (FTR)
        cleaned = self._compute_ftr(cleaned)

        # Step 8: Domain validation checks (shots logic)
        cleaned = self._apply_domain_sanity_rules(cleaned)

        # Step 9: Deduplication (keep the most complete record)
        cleaned = self._deduplicate_matches(cleaned)
        cleaned = self._compute_ftr(cleaned)

        # Step 10: Sort chronologically and reorder columns
        cleaned = cleaned.sort_values(by=["Date", "League", "HomeTeam"]).reset_index(drop=True)
        final_cols = [c for c in self.target_columns if c in cleaned.columns]
        cleaned = cleaned[final_cols]

        logger.info(f"Cleaning complete. Output shape: {cleaned.shape}")
        return cleaned

    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures the unified schema is present without fabricating statistics."""
        if "_Source" not in df.columns:
            df["_Source"] = "Unknown"
        for col in self.target_columns:
            if col not in df.columns:
                if col in ["HS", "AS", "HST", "AST", "FTHG", "FTAG", "xG_home", "xG_away"]:
                    df[col] = np.nan
                elif col in ["ResultSource", "xGSource", "StatsSource"]:
                    df[col] = None
                else:
                    df[col] = "Unknown"
        source = df["_Source"].fillna("Unknown").astype(str)
        if "ResultSource" in df.columns:
            df["ResultSource"] = df["ResultSource"].fillna(
                source.where(df["FTHG"].notna() & df["FTAG"].notna())
            )
        if "xGSource" in df.columns:
            df["xGSource"] = df["xGSource"].fillna(
                source.where(df["xG_home"].notna() & df["xG_away"].notna())
            )
        if "StatsSource" in df.columns:
            has_stats = df[["HS", "AS", "HST", "AST"]].notna().any(axis=1)
            df["StatsSource"] = df["StatsSource"].fillna(source.where(has_stats))
        return df

    def _filter_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops rows with missing primary identification fields."""
        crucial_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        before_count = len(df)
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna(subset=crucial_cols).copy()
        dropped = before_count - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with missing essential keys {crucial_cols}.")
        return df

    def _normalize_teams(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies canonical team name normalization."""
        df["HomeTeam"] = df["HomeTeam"].astype(str).apply(normalize_team_name)
        df["AwayTeam"] = df["AwayTeam"].astype(str).apply(normalize_team_name)
        return df

    def _format_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts Date column to uniform YYYY-MM-DD string."""
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).copy()
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        return df

    def _cast_and_validate_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts and bounds numeric statistics."""
        # Final scores are required; shot counts remain nullable when not collected.
        for col in ["FTHG", "FTAG"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(int).clip(lower=0)
        for col in ["HS", "AS", "HST", "AST"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)

        # Float statistics (xG)
        float_cols = ["xG_home", "xG_away"]
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            df[col] = df[col].clip(lower=0.0).round(4)

        return df

    def _compute_ftr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes or verifies Full Time Result ('H', 'D', 'A')."""
        df["FTR"] = np.where(
            df["FTHG"] > df["FTAG"], "H",
            np.where(df["FTAG"] > df["FTHG"], "A", "D")
        )
        return df

    def _apply_domain_sanity_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforces domain-specific logic:
        1. Shots on target (HST/AST) cannot exceed total shots (HS/AS).
        2. Shots on target cannot be less than goals scored (excluding own goals in rare cases, but
           generally HST >= FTHG).
        """
        # If HST > HS and HS > 0, set HST = HS
        invalid_hst = (df["HST"] > df["HS"]) & (df["HS"] > 0)
        if invalid_hst.any():
            logger.debug(f"Correcting {invalid_hst.sum()} rows where HST > HS.")
            df.loc[invalid_hst, "HST"] = df.loc[invalid_hst, "HS"]

        invalid_ast = (df["AST"] > df["AS"]) & (df["AS"] > 0)
        if invalid_ast.any():
            logger.debug(f"Correcting {invalid_ast.sum()} rows where AST > AS.")
            df.loc[invalid_ast, "AST"] = df.loc[invalid_ast, "AS"]

        return df

    def _deduplicate_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolves duplicates field-by-field, preserving each field's source provenance.
        """
        before_count = len(df)
        resolved = []
        source_preferences = {
            "result": "Football-Data.co.uk",
            "xg": "Understat",
            "stats": "Football-Data.co.uk"
        }
        for _, group in df.groupby(
            ["Date", "League", "HomeTeam", "AwayTeam"], sort=False, dropna=False
        ):
            record = group.iloc[0].copy()

            def choose(preferred_source: str, valid_rows: pd.Series) -> pd.Series:
                candidates = group[valid_rows]
                preferred = candidates[
                    candidates["_Source"].astype(str).str.contains(preferred_source, case=False, na=False)
                ]
                return (preferred if not preferred.empty else candidates).iloc[0]

            result_row = choose(
                source_preferences["result"],
                group["FTHG"].notna() & group["FTAG"].notna()
            )
            xg_row = choose(
                source_preferences["xg"],
                group["xG_home"].notna() & group["xG_away"].notna()
            ) if (group["xG_home"].notna() & group["xG_away"].notna()).any() else None
            stats_valid = group[["HS", "AS", "HST", "AST"]].notna().any(axis=1)
            stats_row = choose(source_preferences["stats"], stats_valid) if stats_valid.any() else None

            for column in ["FTHG", "FTAG", "FTR"]:
                record[column] = result_row[column]
            record["ResultSource"] = result_row["ResultSource"]
            if xg_row is not None:
                for column in ["xG_home", "xG_away"]:
                    record[column] = xg_row[column]
                record["xGSource"] = xg_row["xGSource"]
            else:
                record[["xG_home", "xG_away"]] = np.nan
                record["xGSource"] = None
            if stats_row is not None:
                for column in ["HS", "AS", "HST", "AST"]:
                    record[column] = stats_row[column]
                record["StatsSource"] = stats_row["StatsSource"]
            else:
                record[["HS", "AS", "HST", "AST"]] = np.nan
                record["StatsSource"] = None
            resolved.append(record)

        df = pd.DataFrame(resolved, columns=df.columns)
        dropped = before_count - len(df)
        if dropped > 0:
            logger.info(f"Removed {dropped} duplicate match records.")

        return df
