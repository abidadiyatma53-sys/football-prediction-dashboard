"""
Live Odds & Upcoming Fixtures Synchronization Module.
Fetches real-time bookmaker odds exclusively via The Odds API and normalizes team names,
and scans upcoming matchdays for Positive EV (+EV) value betting opportunities.
"""

import os
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import requests

import config
from normalizer import normalize_team_name
from predictor import MatchPredictorService
from value_bet import ValueBettingEngine

logger = logging.getLogger(__name__)


class OddsSourceUnavailable(RuntimeError):
    """Raised when a fresh bookmaker feed cannot be obtained."""


class LiveOddsSyncClient:
    """
    Client for ingesting live bookmaker odds and upcoming fixtures:
    - Uses only fresh bookmaker prices returned by The Odds API.
    - Never substitutes model-derived, hard-coded, or stale prices for market odds.
    - Normalizes all team names to canonical forms.
    - Scans entire fixture slates for +EV bets with Kelly stake sizing.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        self.value_engine = ValueBettingEngine(default_kelly_fraction=0.25)
        self._predictor: Optional[MatchPredictorService] = None

    @property
    def predictor(self) -> MatchPredictorService:
        if self._predictor is None:
            self._predictor = MatchPredictorService(data_path=str(config.OUTPUT_FILE))
        return self._predictor

    def fetch_live_fixtures_and_odds(
        self,
        sport_key: str = "soccer_epl",
        regions: str = "eu,uk"
    ) -> List[Dict[str, Any]]:
        """
        Retrieves upcoming fixtures with real bookmaker odds from The Odds API.
        Raises OddsSourceUnavailable instead of returning simulated or stale odds.
        """
        if not self.api_key:
            raise OddsSourceUnavailable(
                "The Odds API key is not configured; live bookmaker odds are unavailable."
            )

        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso"
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as error:
            logger.error("The Odds API request failed: %s", error)
            raise OddsSourceUnavailable("The Odds API request failed.") from error
        except ValueError as error:
            logger.error("The Odds API returned invalid JSON: %s", error)
            raise OddsSourceUnavailable("The Odds API returned invalid JSON.") from error

        if not isinstance(data, list):
            raise OddsSourceUnavailable("The Odds API returned an unexpected response.")
        return self._parse_odds_api_response(data)

    def _parse_odds_api_response(self, api_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parses raw JSON from The Odds API into canonical format."""
        results = []
        now_utc = datetime.now(timezone.utc)
        for item in api_data:
            home = normalize_team_name(item.get("home_team", ""))
            away = normalize_team_name(item.get("away_team", ""))
            commence_time = item.get("commence_time", "")

            try:
                kickoff = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            except (TypeError, ValueError):
                logger.warning("Skipping odds record with invalid kickoff time: %r", commence_time)
                continue
            if kickoff <= now_utc:
                continue

            best_prices: Dict[str, tuple[float, str]] = {}

            for bm in item.get("bookmakers", []):
                bookmaker_name = bm.get("title", "Unknown bookmaker")
                for market in bm.get("markets", []):
                    if market.get("key") == "h2h":
                        for outcome in market.get("outcomes", []):
                            name = outcome.get("name")
                            try:
                                price = float(outcome.get("price", 0.0))
                            except (TypeError, ValueError):
                                continue
                            if price <= 1.0:
                                continue
                            if name == item.get("home_team"):
                                market_key = "home_win"
                            elif name == item.get("away_team"):
                                market_key = "away_win"
                            elif isinstance(name, str) and name.lower() == "draw":
                                market_key = "draw"
                            else:
                                continue
                            if market_key not in best_prices or price > best_prices[market_key][0]:
                                best_prices[market_key] = (price, bookmaker_name)

            required_outcomes = {"home_win", "draw", "away_win"}
            if home and away and required_outcomes.issubset(best_prices):
                bookmakers = {
                    outcome: best_prices[outcome][1]
                    for outcome in ("home_win", "draw", "away_win")
                }
                results.append({
                    "fixture_id": item.get("id"),
                    "home": home,
                    "away": away,
                    "date": kickoff.isoformat(),
                    "odds_home": round(best_prices["home_win"][0], 2),
                    "odds_draw": round(best_prices["draw"][0], 2),
                    "odds_away": round(best_prices["away_win"][0], 2),
                    "bookmaker": "Best available odds",
                    "bookmakers": bookmakers,
                    "source": "The Odds API",
                    "fetched_at_utc": now_utc.isoformat()
                })

        return results

    def scan_for_value_bets(
        self,
        fixtures: Optional[List[Dict[str, Any]]] = None,
        method: str = "ml"
    ) -> List[Dict[str, Any]]:
        """
        Batch analyzes upcoming fixtures against live bookmaker odds,
        running Dixon-Coles Poisson and returning actionable +EV value bets.
        """
        target_fixtures = (
            self.fetch_live_fixtures_and_odds() if fixtures is None else fixtures
        )
        findings = []

        for fix in target_fixtures:
            h = normalize_team_name(fix["home"])
            a = normalize_team_name(fix["away"])

            try:
                fixture_date = fix.get("date")
                match_date = (
                    datetime.fromisoformat(fixture_date.replace("Z", "+00:00")).date().isoformat()
                    if isinstance(fixture_date, str) and fixture_date != "Upcoming"
                    else None
                )
                pred = self.predictor.predict_match(
                    h,
                    a,
                    method=method,
                    display=False,
                    match_date=match_date
                )
                probs = pred["probabilities_3way"]

                market_dict = {
                    "home_win": fix["odds_home"],
                    "draw": fix["odds_draw"],
                    "away_win": fix["odds_away"]
                }

                analysis = self.value_engine.analyze_market_odds(probs, market_dict)
                value_bets = [item for item in analysis if item["is_value_bet"]]

                findings.append({
                    "fixture": f"{h} vs {a}",
                    "home_team": h,
                    "away_team": a,
                    "date": fix.get("date", "Upcoming"),
                    "bookmaker": fix.get("bookmaker", "Unknown bookmaker"),
                    "bookmakers": fix.get("bookmakers", {}),
                    "source": fix.get("source", "Unknown"),
                    "fetched_at_utc": fix.get("fetched_at_utc"),
                    "expected_goals": f"{pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}",
                    "most_likely_score": pred["most_likely_score"],
                    "score_prob_pct": pred["most_likely_probability"],
                    "probabilities": probs,
                    "market_odds": market_dict,
                    "has_value_bet": len(value_bets) > 0,
                    "value_bets_count": len(value_bets),
                    "best_bet": value_bets[0] if value_bets else None,
                    "all_analysis": analysis
                })
            except Exception as e:
                logger.warning(f"Error analyzing fixture {h} vs {a}: {e}")

        # Sort matches so fixtures with highest +EV bets appear first
        findings.sort(
            key=lambda x: x["best_bet"]["ev_pct"] if x["best_bet"] else -999.0,
            reverse=True
        )
        return findings
