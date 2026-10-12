"""Fixture and result verification through football-data.org."""

import os
import time
from datetime import date
from typing import Any, Dict, Optional

import requests

from normalizer import normalize_team_name


BASE_URL = "https://api.football-data.org/v4"
COMPETITION_CODES = {
    "EPL": "PL",
    "La_Liga": "PD",
    "Bundesliga": "BL1",
    "Serie_A": "SA",
    "Ligue_1": "FL1",
}


class FootballDataSourceUnavailable(RuntimeError):
    """Raised when football-data.org cannot return fixture data."""


class FootballDataClient:
    """Verify scheduled fixtures and expose final scores; no lineup data is claimed."""

    def __init__(
        self,
        api_token: Optional[str] = None,
        timeout: int = 12,
        cache_seconds: int = 300,
        max_cache_entries: int = 128,
    ):
        self.api_token = api_token or os.getenv("FOOTBALL_DATA_API_TOKEN", "")
        self.timeout = timeout
        self.cache_seconds = cache_seconds
        self.max_cache_entries = max_cache_entries
        self._cache: Dict[str, tuple[float, Dict[str, Any]]] = {}

    def verify_fixture(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        league: str,
    ) -> Dict[str, Any]:
        if not self.api_token:
            raise FootballDataSourceUnavailable(
                "FOOTBALL_DATA_API_TOKEN is not configured; fixture verification is unavailable."
            )
        if league not in COMPETITION_CODES:
            raise ValueError(
                f"football-data.org fixture verification is not configured for league: {league}"
            )
        try:
            parsed_date = date.fromisoformat(match_date)
        except (TypeError, ValueError) as error:
            raise ValueError("match_date must use YYYY-MM-DD format.") from error

        home = normalize_team_name(home_team)
        away = normalize_team_name(away_team)
        cache_key = f"{league}:{parsed_date.isoformat()}:{home}:{away}"
        cached = self._cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < self.cache_seconds:
            return cached[1]

        matches = self._request(
            "/matches",
            {
                "dateFrom": parsed_date.isoformat(),
                "dateTo": parsed_date.isoformat(),
                "competitions": COMPETITION_CODES[league],
            },
        )
        candidates = []
        for match in matches:
            match_home = normalize_team_name(match.get("homeTeam", {}).get("name", ""))
            match_away = normalize_team_name(match.get("awayTeam", {}).get("name", ""))
            if match_home == home and match_away == away:
                candidates.append(match)

        if len(candidates) > 1:
            raise FootballDataSourceUnavailable(
                "Multiple fixtures matched teams and date; verification was stopped."
            )
        if not candidates:
            result = {
                "status": "fixture_not_found",
                "source": "football-data.org",
                "match_date": parsed_date.isoformat(),
                "home_team": home,
                "away_team": away,
                "league": league,
                "lineups_available": False,
                "message": (
                    "No matching fixture was found in this competition/date. "
                    "football-data.org free/basic data does not provide lineups."
                ),
            }
        else:
            match = candidates[0]
            score = match.get("score", {})
            full_time = score.get("fullTime") or {}
            result = {
                "status": match.get("status", "UNKNOWN").lower(),
                "source": "football-data.org",
                "fixture_id": match.get("id"),
                "match_date": parsed_date.isoformat(),
                "kickoff": match.get("utcDate"),
                "home_team": home,
                "away_team": away,
                "league": league,
                "lineups_available": False,
                "score": {
                    "home": full_time.get("home"),
                    "away": full_time.get("away"),
                },
                "message": (
                    "Fixture/status/score verified through football-data.org. "
                    "The API does not provide lineup data; enter lineups manually."
                ),
            }
        self._store_cache(cache_key, result)
        return result

    def _request(self, endpoint: str, params: Dict[str, str]) -> list:
        try:
            response = requests.get(
                f"{BASE_URL}{endpoint}",
                headers={"X-Auth-Token": self.api_token},
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            raise FootballDataSourceUnavailable(
                "football-data.org request failed; verify the token, access tier, and quota."
            ) from error
        except ValueError as error:
            raise FootballDataSourceUnavailable(
                "football-data.org returned invalid JSON."
            ) from error

        if not isinstance(payload, dict):
            raise FootballDataSourceUnavailable(
                "football-data.org returned an unexpected response."
            )
        if "matches" not in payload or not isinstance(payload["matches"], list):
            raise FootballDataSourceUnavailable(
                "football-data.org response did not include a matches list."
            )
        return payload["matches"]

    def _store_cache(self, key: str, value: Dict[str, Any]) -> None:
        now = time.monotonic()
        self._cache = {
            cached_key: entry
            for cached_key, entry in self._cache.items()
            if now - entry[0] < self.cache_seconds
        }
        if key not in self._cache and len(self._cache) >= self.max_cache_entries:
            oldest_key = min(self._cache, key=lambda item: self._cache[item][0])
            del self._cache[oldest_key]
        self._cache[key] = (now, value)
