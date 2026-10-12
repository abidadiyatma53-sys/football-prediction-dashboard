import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

from cleaner import DataCleaner
from live_odds import LiveOddsSyncClient, OddsSourceUnavailable
from predictor import MatchPredictorService
from scrapers.footballdata_scraper import FootballDataScraper


class DataSourceIntegrityTests(unittest.TestCase):
    def test_unavailable_shot_statistics_remain_missing(self):
        cleaned = DataCleaner().clean_dataset(pd.DataFrame([{
            "Date": "2025-01-01",
            "League": "EPL",
            "Season": "2024",
            "HomeTeam": "Alpha",
            "AwayTeam": "Beta",
            "FTHG": 1,
            "FTAG": 0,
            "xG_home": 1.2,
            "xG_away": 0.4,
            "_Source": "Understat",
        }]))
        self.assertTrue(cleaned.loc[0, ["HS", "AS", "HST", "AST"]].isna().all())
        self.assertEqual(cleaned.loc[0, "ResultSource"], "Understat")
        self.assertEqual(cleaned.loc[0, "xGSource"], "Understat")
        self.assertTrue(pd.isna(cleaned.loc[0, "StatsSource"]))

    def test_predictor_discards_rows_without_valid_final_scores(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory) / "matches.csv"
            pd.DataFrame([
                {
                    "Date": "2025-01-01", "League": "EPL",
                    "HomeTeam": "Alpha", "AwayTeam": "Beta",
                    "FTHG": 1, "FTAG": 0,
                },
                {
                    "Date": "2025-01-08", "League": "EPL",
                    "HomeTeam": "Gamma", "AwayTeam": "Alpha",
                    "FTHG": float("nan"), "FTAG": 2,
                },
            ]).to_csv(data_path, index=False)
            with patch("predictor.FootballFeatureEngineer") as feature_engineer, \
                    patch("predictor.PoissonMatchPredictor"), \
                    patch("predictor.MLMatchPredictor"):
                feature_engineer.return_value.transform.return_value = pd.DataFrame({
                    "Date": ["2025-01-01"]
                })
                service = MatchPredictorService(data_path=str(data_path))
            self.assertEqual(len(service.df_raw), 1)
            self.assertEqual(service.df_raw.iloc[0]["FTHG"], 1)

    def test_duplicate_fixture_uses_field_specific_source_precedence(self):
        records = [
            {
                "Date": "2025-01-01", "League": "EPL", "Season": "2024",
                "HomeTeam": "Alpha", "AwayTeam": "Beta", "FTHG": 1, "FTAG": 0,
                "xG_home": 1.4, "xG_away": 0.3,
                "HS": None, "AS": None, "HST": None, "AST": None,
                "_Source": "Understat",
            },
            {
                "Date": "2025-01-01", "League": "EPL", "Season": "2024",
                "HomeTeam": "Alpha", "AwayTeam": "Beta", "FTHG": 2, "FTAG": 0,
                "xG_home": None, "xG_away": None,
                "HS": 11, "AS": 7, "HST": 4, "AST": 2,
                "_Source": "Football-Data.co.uk",
            },
        ]
        cleaned = DataCleaner().clean_dataset(pd.DataFrame(records))
        self.assertEqual(len(cleaned), 1)
        match = cleaned.iloc[0]
        self.assertEqual((match["FTHG"], match["FTAG"]), (2, 0))
        self.assertEqual(match["ResultSource"], "Football-Data.co.uk")
        self.assertEqual((match["xG_home"], match["xG_away"]), (1.4, 0.3))
        self.assertEqual(match["xGSource"], "Understat")
        self.assertEqual((match["HS"], match["AST"]), (11, 2))
        self.assertEqual(match["StatsSource"], "Football-Data.co.uk")

    def test_live_odds_requires_configured_key(self):
        client = LiveOddsSyncClient(api_key="")
        with self.assertRaises(OddsSourceUnavailable):
            client.fetch_live_fixtures_and_odds()

    def test_http_failure_is_not_replaced_with_fallback_odds(self):
        client = LiveOddsSyncClient(api_key="configured-test-key")
        with patch("live_odds.requests.get", side_effect=requests.Timeout("timeout")):
            with self.assertRaises(OddsSourceUnavailable):
                client.fetch_live_fixtures_and_odds()

    def test_expired_odds_fixture_is_ignored_and_empty_scan_stays_empty(self):
        client = LiveOddsSyncClient(api_key="")
        old_fixture = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "commence_time": "2020-01-01T12:00:00Z",
            "bookmakers": [],
        }
        self.assertEqual(client._parse_odds_api_response([old_fixture]), [])
        self.assertEqual(client.scan_for_value_bets(fixtures=[]), [])

    def test_odds_keep_the_bookmaker_for_each_best_price(self):
        client = LiveOddsSyncClient(api_key="configured-test-key")
        fixture = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "commence_time": "2099-01-01T12:00:00Z",
            "bookmakers": [
                {
                    "title": "Book A",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.2},
                            {"name": "Draw", "price": 3.1},
                            {"name": "Chelsea", "price": 4.0},
                        ],
                    }],
                },
                {
                    "title": "Book B",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.0},
                            {"name": "Draw", "price": 3.4},
                            {"name": "Chelsea", "price": 3.8},
                        ],
                    }],
                },
            ],
        }
        parsed = client._parse_odds_api_response([fixture])
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["bookmakers"], {
            "home_win": "Book A",
            "draw": "Book B",
            "away_win": "Book A",
        })
        self.assertEqual(parsed[0]["source"], "The Odds API")
        self.assertEqual(parsed[0]["fixture_id"], fixture.get("id"))
        self.assertTrue(parsed[0]["fetched_at_utc"])

    def test_football_data_cache_must_match_requested_season(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = Path(temporary_directory)
            (cache / "mirror_E0_2324.csv").write_text("old-season", encoding="utf-8")
            (cache / "mirror_E0_2425.csv").write_text("requested-season", encoding="utf-8")
            scraper = FootballDataScraper(raw_cache_dir=cache)
            self.assertEqual(scraper.download_csv("E0", "2425"), "requested-season")

    def test_football_data_parser_keeps_missing_shots_missing(self):
        csv_data = (
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,HS,AS,HST,AST\n"
            "01/01/25,Alpha,Beta,0,0,,,,\n"
            "02/01/25,Alpha,Gamma,,,,,,\n"
        )
        parsed = FootballDataScraper().parse_csv_data(csv_data, "EPL", "2024")
        self.assertEqual(len(parsed), 1)
        self.assertEqual((parsed.iloc[0]["FTHG"], parsed.iloc[0]["FTAG"]), (0, 0))
        self.assertTrue(parsed.loc[0, ["HS", "AS", "HST", "AST"]].isna().all())
        self.assertEqual(parsed.iloc[0]["_Source"], "Football-Data.co.uk")


if __name__ == "__main__":
    unittest.main()
