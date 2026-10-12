import unittest
from unittest.mock import Mock, patch

import requests

from football_data_service import (
    FootballDataClient,
    FootballDataSourceUnavailable,
)


def api_response(data):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = data
    return response


class FootballDataClientTests(unittest.TestCase):
    def test_token_is_required(self):
        client = FootballDataClient(api_token="")
        with self.assertRaisesRegex(
            FootballDataSourceUnavailable, "FOOTBALL_DATA_API_TOKEN"
        ):
            client.verify_fixture("Arsenal", "Chelsea", "2026-10-10", "EPL")

    @patch("football_data_service.requests.get")
    def test_fixture_and_final_score_are_returned(self, mock_get):
        mock_get.return_value = api_response(
            {
                "matches": [
                    {
                        "id": 1234,
                        "utcDate": "2026-10-10T14:00:00Z",
                        "status": "FINISHED",
                        "homeTeam": {"name": "Arsenal"},
                        "awayTeam": {"name": "Chelsea"},
                        "score": {"fullTime": {"home": 2, "away": 1}},
                    }
                ]
            }
        )
        client = FootballDataClient(api_token="test-token")

        result = client.verify_fixture("Arsenal", "Chelsea", "2026-10-10", "EPL")

        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["score"], {"home": 2, "away": 1})
        self.assertFalse(result["lineups_available"])
        self.assertEqual(result["source"], "football-data.org")
        self.assertEqual(mock_get.call_args.kwargs["headers"], {"X-Auth-Token": "test-token"})
        self.assertEqual(
            mock_get.call_args.kwargs["params"],
            {
                "dateFrom": "2026-10-10",
                "dateTo": "2026-10-10",
                "competitions": "PL",
            },
        )

    @patch("football_data_service.requests.get")
    def test_fixture_lookup_is_cached(self, mock_get):
        mock_get.return_value = api_response({"matches": []})
        client = FootballDataClient(api_token="test-token")

        client.verify_fixture("Arsenal", "Chelsea", "2026-10-10", "EPL")
        client.verify_fixture("Arsenal", "Chelsea", "2026-10-10", "EPL")

        mock_get.assert_called_once()

    @patch("football_data_service.requests.get")
    def test_missing_fixture_is_reported_without_fabricating_data(self, mock_get):
        mock_get.return_value = api_response({"matches": []})

        result = FootballDataClient(api_token="test-token").verify_fixture(
            "Arsenal", "Chelsea", "2026-10-10", "EPL"
        )

        self.assertEqual(result["status"], "fixture_not_found")
        self.assertIsNone(result.get("score"))

    @patch(
        "football_data_service.requests.get",
        side_effect=requests.Timeout("timeout"),
    )
    def test_network_failure_is_reported(self, _mock_get):
        client = FootballDataClient(api_token="test-token")
        with self.assertRaisesRegex(FootballDataSourceUnavailable, "request failed"):
            client.verify_fixture("Arsenal", "Chelsea", "2026-10-10", "EPL")

    def test_invalid_date_and_unsupported_league_are_rejected(self):
        client = FootballDataClient(api_token="test-token")
        with self.assertRaises(ValueError):
            client.verify_fixture("Arsenal", "Chelsea", "not-a-date", "EPL")
        with self.assertRaisesRegex(ValueError, "not configured"):
            client.verify_fixture("Arsenal", "Chelsea", "2026-10-10", "RFPL")


class FootballDataApiTests(unittest.TestCase):
    def test_missing_token_returns_explicit_service_unavailable(self):
        from fastapi.testclient import TestClient
        from api import app

        with patch("api.football_data_client", FootballDataClient(api_token="")):
            response = TestClient(app).get(
                "/fixtures/verify",
                params={
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "match_date": "2026-10-10",
                    "league": "EPL",
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("FOOTBALL_DATA_API_TOKEN", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
