"""
SQLite Prediction Ledger & Audit Trail Database.
Manages persistent storage for pre-kickoff predictions, market odds taken,
settlement against actual results, and cumulative P&L tracking.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from contextlib import contextmanager
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/predictions_ledger.db")


class PredictionDatabase:
    """
    Thread-safe SQLite database manager for football prediction tracking & financial ledger.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        """Yields a dict-row connection and closes it after the transaction."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        """Initializes schema if not already present."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                match_date TEXT,
                league TEXT DEFAULT 'EPL',
                season TEXT DEFAULT '2023',
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                model_method TEXT NOT NULL,
                prob_home REAL NOT NULL,
                prob_draw REAL NOT NULL,
                prob_away REAL NOT NULL,
                lambda_home REAL,
                lambda_away REAL,
                predicted_score TEXT,
                predicted_score_prob REAL,
                market_odds_home REAL,
                market_odds_draw REAL,
                market_odds_away REAL,
                bet_selection TEXT,        -- 'home_win', 'draw', 'away_win', or NULL
                bet_odds REAL,
                bet_ev_pct REAL,
                bet_stake_units REAL DEFAULT 1.0,
                status TEXT DEFAULT 'PENDING', -- 'PENDING', 'SETTLED', 'VOID'
                settled_at TIMESTAMP,
                actual_home_goals INTEGER,
                actual_away_goals INTEGER,
                actual_result TEXT,        -- 'H', 'D', 'A'
                is_won INTEGER,            -- 1 (won), 0 (lost), NULL (pending)
                pnl_units REAL DEFAULT 0.0,
                closing_odds REAL,
                clv_pct REAL
            );
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_match_teams ON predictions(home_team, away_team, status);
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON predictions(status);
            """)
            conn.commit()

    def log_prediction(self, pred_data: Dict[str, Any]) -> int:
        """
        Logs a pre-match forecast into the ledger.

        Args:
            pred_data: Dictionary containing match details, model probabilities, and betting odds.

        Returns:
            The inserted prediction row ID.
        """
        query = """
        INSERT INTO predictions (
            match_date, league, season, home_team, away_team, model_method,
            prob_home, prob_draw, prob_away, lambda_home, lambda_away,
            predicted_score, predicted_score_prob,
            market_odds_home, market_odds_draw, market_odds_away,
            bet_selection, bet_odds, bet_ev_pct, bet_stake_units, status
        ) VALUES (
            :match_date, :league, :season, :home_team, :away_team, :model_method,
            :prob_home, :prob_draw, :prob_away, :lambda_home, :lambda_away,
            :predicted_score, :predicted_score_prob,
            :market_odds_home, :market_odds_draw, :market_odds_away,
            :bet_selection, :bet_odds, :bet_ev_pct, :bet_stake_units, 'PENDING'
        );
        """
        defaults = {
            "match_date": datetime.now().strftime("%Y-%m-%d"),
            "league": "EPL",
            "season": "2023",
            "bet_selection": None,
            "bet_odds": None,
            "bet_ev_pct": None,
            "bet_stake_units": 1.0,
            "market_odds_home": None,
            "market_odds_draw": None,
            "market_odds_away": None,
        }
        params = {**defaults, **pred_data}

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid

    def settle_prediction(
        self,
        prediction_id: int,
        actual_home_goals: int,
        actual_away_goals: int,
        closing_odds: Optional[float] = None
    ) -> bool:
        """
        Settles a specific prediction record against actual match scores.
        """
        if actual_home_goals < 0 or actual_away_goals < 0:
            raise ValueError("Actual goals cannot be negative.")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM predictions WHERE id = ? AND status = 'PENDING'",
                (prediction_id,)
            )
            row = cursor.fetchone()
            if not row:
                logger.warning(
                    "Prediction ID %s was not found or is already settled.",
                    prediction_id
                )
                return False

            if actual_home_goals > actual_away_goals:
                actual_res = "H"
                winning_sel = "home_win"
            elif actual_home_goals == actual_away_goals:
                actual_res = "D"
                winning_sel = "draw"
            else:
                actual_res = "A"
                winning_sel = "away_win"

            bet_sel = row["bet_selection"]
            odds = row["bet_odds"] or 0.0
            stake = row["bet_stake_units"] or 1.0

            is_won = 0
            pnl = 0.0

            if bet_sel:
                if bet_sel == winning_sel:
                    is_won = 1
                    pnl = round(stake * (odds - 1.0), 3)
                else:
                    is_won = 0
                    pnl = round(-stake, 3)

            # Closing Line Value (CLV) Calculation if closing odds available
            clv_pct = None
            if closing_odds and odds > 0:
                clv_pct = round(((odds / closing_odds) - 1.0) * 100.0, 2)

            cursor.execute("""
            UPDATE predictions SET
                status = 'SETTLED',
                settled_at = CURRENT_TIMESTAMP,
                actual_home_goals = ?,
                actual_away_goals = ?,
                actual_result = ?,
                is_won = ?,
                pnl_units = ?,
                closing_odds = ?,
                clv_pct = ?
            WHERE id = ?;
            """, (
                actual_home_goals,
                actual_away_goals,
                actual_res,
                is_won,
                pnl,
                closing_odds,
                clv_pct,
                prediction_id
            ))
            conn.commit()
            return True

    def get_all_predictions(self, limit: int = 100) -> pd.DataFrame:
        """Retrieves history of predictions as pandas DataFrame."""
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,)
            )
            return df

    def get_pending_predictions(self) -> List[Dict[str, Any]]:
        """Returns all pending predictions waiting for actual results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE status = 'PENDING' ORDER BY id ASC")
            return [dict(r) for r in cursor.fetchall()]

    def has_pending_prediction(self, home_team: str, away_team: str, match_date: Optional[str] = None) -> bool:
        """Checks if a pending prediction already exists for this matchup to prevent duplicates."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if match_date:
                cursor.execute(
                    "SELECT id FROM predictions WHERE home_team = ? AND away_team = ? AND match_date = ? AND status = 'PENDING'",
                    (home_team, away_team, match_date)
                )
            else:
                cursor.execute(
                    "SELECT id FROM predictions WHERE home_team = ? AND away_team = ? AND status = 'PENDING'",
                    (home_team, away_team)
                )
            return cursor.fetchone() is not None

    def get_pnl_summary(self) -> Dict[str, Any]:
        """
        Calculates aggregate P&L, Win Rate, ROI %, and Brier Score on settled bets.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN status = 'SETTLED' THEN 1 ELSE 0 END) as settled_count,
                SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) as pending_count,
                SUM(CASE WHEN bet_selection IS NOT NULL AND status = 'SETTLED' THEN 1 ELSE 0 END) as total_bets,
                SUM(CASE WHEN is_won = 1 THEN 1 ELSE 0 END) as won_bets,
                SUM(CASE WHEN is_won = 0 AND bet_selection IS NOT NULL AND status = 'SETTLED' THEN 1 ELSE 0 END) as lost_bets,
                COALESCE(SUM(bet_stake_units), 0) as total_staked,
                COALESCE(SUM(pnl_units), 0) as total_pnl
            FROM predictions;
            """)
            summary = dict(cursor.fetchone())

            total_bets = summary["total_bets"] or 0
            won_bets = summary["won_bets"] or 0
            staked = summary["total_staked"] or 0.0
            pnl = summary["total_pnl"] or 0.0

            win_rate = round((won_bets / total_bets) * 100.0, 2) if total_bets > 0 else 0.0
            roi_pct = round((pnl / staked) * 100.0, 2) if staked > 0 else 0.0

            # Calculate Brier score on settled predictions
            cursor.execute("""
            SELECT prob_home, prob_draw, prob_away, actual_result
            FROM predictions
            WHERE status = 'SETTLED' AND actual_result IS NOT NULL;
            """)
            rows = cursor.fetchall()
            brier_sum = 0.0
            for r in rows:
                p_h = r["prob_home"] / 100.0 if r["prob_home"] > 1.0 else r["prob_home"]
                p_d = r["prob_draw"] / 100.0 if r["prob_draw"] > 1.0 else r["prob_draw"]
                p_a = r["prob_away"] / 100.0 if r["prob_away"] > 1.0 else r["prob_away"]
                y_h = 1.0 if r["actual_result"] == "H" else 0.0
                y_d = 1.0 if r["actual_result"] == "D" else 0.0
                y_a = 1.0 if r["actual_result"] == "A" else 0.0
                brier_sum += ((p_h - y_h)**2 + (p_d - y_d)**2 + (p_a - y_a)**2)

            brier_score = round(brier_sum / len(rows), 4) if rows else None

            return {
                "total_predictions": summary["total_predictions"],
                "settled_count": summary["settled_count"],
                "pending_count": summary["pending_count"],
                "total_bets": total_bets,
                "won_bets": won_bets,
                "lost_bets": summary["lost_bets"] or 0,
                "win_rate_pct": win_rate,
                "total_staked_units": round(staked, 2),
                "total_pnl_units": round(pnl, 2),
                "roi_pct": roi_pct,
                "brier_score": brier_score
            }
