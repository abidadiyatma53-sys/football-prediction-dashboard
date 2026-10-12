"""
FastAPI REST API Service for Football Score Prediction & Value Betting.
Provides RESTful endpoints for integration with web apps, mobile clients, and Telegram bots.
"""

import hashlib
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

import config
from normalizer import normalize_team_name, CANONICAL_TEAMS
from predictor import MatchPredictorService
from value_bet import ValueBettingEngine
from live_odds import LiveOddsSyncClient, OddsSourceUnavailable
from football_data_service import FootballDataClient, FootballDataSourceUnavailable

FIXTURES_PATH = config.DATA_DIR / "upcoming_european_fixtures.json"
PLAYERS_DB_PATH = config.DATA_DIR / "players_database.csv"
MAX_FIXTURE_SNAPSHOT_AGE_HOURS = 24

app = FastAPI(
    title="Football Match Prediction & Analytics API",
    description="Probabilistic match forecasting (Poisson & XGBoost ML) with Positive EV value betting detection.",
    version="1.0.0"
)

# Enable CORS for external frontend or web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services initialized lazily or on startup
predictor_service: Optional[MatchPredictorService] = None
value_engine = ValueBettingEngine()


def get_predictor() -> MatchPredictorService:
    global predictor_service
    if predictor_service is None:
        predictor_service = MatchPredictorService(data_path=str(config.OUTPUT_FILE))
    return predictor_service


class PredictionRequest(BaseModel):
    home_team: str = Field(..., example="Arsenal", description="Name of the Home team")
    away_team: str = Field(..., example="Chelsea", description="Name of the Away team")
    method: str = Field("ml", example="ml", description="'ml' or 'poisson'")
    match_date: Optional[str] = Field(
        None, example="2026-10-10", description="Fixture date used for pre-match team form."
    )


class ValueBetRequest(BaseModel):
    home_team: str = Field(..., example="Arsenal")
    away_team: str = Field(..., example="Chelsea")
    method: str = Field("ml", example="ml")
    match_date: Optional[str] = Field(
        None, example="2026-10-10", description="Fixture date used for pre-match team form."
    )
    market_odds: Dict[str, float] = Field(
        ...,
        description=(
            "Current bookmaker decimal odds for match outcomes. Get live odds from "
            "/odds/live; manually submitted prices cannot be independently verified."
        )
    )
    kelly_fraction: Optional[float] = Field(0.25, description="Kelly stake multiplier (0.25 = quarter)")


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": "Football Prediction REST API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": [
            "GET  /teams",
            "GET  /predict",
            "POST /predict",
            "POST /value-bet",
            "GET  /fixtures/upcoming",
            "GET  /fixtures/verify",
            "GET  /odds/live",
            "GET  /fixtures/sample (mock data)"
        ]
    }


@app.get("/teams", tags=["Metadata"])
def get_available_teams(league: Optional[str] = Query(None, description="Optional league code e.g. EPL, La_Liga, Bundesliga, Serie_A, Ligue_1, RFPL")):
    """Returns list of normalized available football clubs across Europe."""
    service = get_predictor()
    if league:
        l_norm = league.strip().lower()
        sub_df = service.df_raw[service.df_raw["League"].str.lower() == l_norm]
        if sub_df.empty:
            sub_df = service.df_raw[service.df_raw["League"].str.lower().str.replace("_", "") == l_norm.replace("_", "")]
        teams = sorted(list(set(sub_df["HomeTeam"]).union(set(sub_df["AwayTeam"])))) if not sub_df.empty else []
    else:
        teams = sorted(list(service.df_raw["HomeTeam"].unique()))
    return {"count": len(teams), "league": league or "all", "teams": teams}


@app.get("/fixtures/upcoming", tags=["Fixtures & Odds"])
def get_upcoming_fixtures(
    league: Optional[str] = Query(None, description="Filter by league (e.g. EPL, La_Liga, Bundesliga, Serie_A, Ligue_1, RFPL)"),
    date: Optional[str] = Query(None, description="Filter by date YYYY-MM-DD"),
    limit: int = Query(50, ge=1, le=500, description="Max matches to return")
):
    """Returns fresh predictions for fixtures from a recent Understat snapshot."""
    import json
    if not FIXTURES_PATH.exists():
        raise HTTPException(status_code=404, detail="Upcoming fixtures data not found. Run pipeline first.")

    snapshot_updated_at = pd.Timestamp(FIXTURES_PATH.stat().st_mtime, unit="s", tz="UTC")
    snapshot_age_hours = (
        pd.Timestamp.now(tz="UTC") - snapshot_updated_at
    ).total_seconds() / 3600
    if snapshot_age_hours > MAX_FIXTURE_SNAPSHOT_AGE_HOURS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Fixture snapshot is {snapshot_age_hours:.1f} hours old. "
                "Refresh it with update_all_european_data.py before using predictions."
            )
        )

    with FIXTURES_PATH.open("r", encoding="utf-8") as f:
        fixtures = json.load(f)
    today = pd.Timestamp.now(tz="UTC").date().isoformat()
    fixtures = [
        fixture for fixture in fixtures
        if fixture.get("date", "") >= today
    ]

    if league:
        l_norm = league.strip().lower().replace(" ", "_")
        fixtures = [
            f for f in fixtures
            if f.get("league", "").lower() == l_norm
            or f.get("league_name", "").lower() == l_norm
            or l_norm in f.get("league", "").lower()
        ]

    if date:
        d_clean = date.strip()
        fixtures = [f for f in fixtures if f.get("date") == d_clean]

    service = get_predictor()
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    data_sha256 = hashlib.sha256(config.OUTPUT_FILE.read_bytes()).hexdigest()
    fresh_fixtures = []
    for fixture in fixtures[:limit]:
        try:
            prediction = service.predict_match(
                fixture["home_team"],
                fixture["away_team"],
                method="ml",
                display=False,
                match_date=fixture.get("date")
            )
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Could not generate a fresh prediction for "
                    f"{fixture.get('home_team')} vs {fixture.get('away_team')}: {error}"
                )
            ) from error

        probs = prediction["probabilities_3way"]
        fresh_fixtures.append({
            **fixture,
            "predicted_score": prediction["most_likely_score"],
            "score_prob_pct": prediction["most_likely_probability"],
            "lambda_home": prediction["lambda_home"],
            "lambda_away": prediction["lambda_away"],
            "home_win_prob": probs["home_win"],
            "draw_prob": probs["draw"],
            "away_win_prob": probs["away_win"],
            "top_scorelines": prediction["top_exact_scores"],
            "prediction_method": prediction["method"],
            "prediction_generated_at_utc": generated_at,
            "model_data_through": str(service.df_raw["Date"].max()),
            "training_data_sha256": data_sha256
        })

    return {
        "status": "success",
        "source": "Understat local fixture snapshot; predictions regenerated per request",
        "snapshot_updated_at": snapshot_updated_at.isoformat(),
        "snapshot_age_hours": round(snapshot_age_hours, 2),
        "prediction_generated_at_utc": generated_at,
        "training_data_sha256": data_sha256,
        "model_data_through": str(service.df_raw["Date"].max()),
        "total_available": len(fixtures),
        "count": len(fresh_fixtures),
        "fixtures": fresh_fixtures
    }


@app.get("/fixtures/verify", tags=["Fixtures & Odds"])
def verify_fixture(
    home_team: str = Query(..., description="Home team name"),
    away_team: str = Query(..., description="Away team name"),
    match_date: str = Query(..., description="Fixture date in YYYY-MM-DD format"),
    league: str = Query(..., description="Supported codes: EPL, La_Liga, Bundesliga, Serie_A, Ligue_1"),
):
    """Verifies fixture/status/score; football-data.org does not provide lineups."""
    try:
        return football_data_client.verify_fixture(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            league=league,
        )
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except FootballDataSourceUnavailable as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


@app.get("/fixtures/leagues", tags=["Metadata"])
def get_leagues_summary():
    """Returns list of covered European leagues and statistics."""
    import config
    service = get_predictor()
    leagues_data = []
    for lg_code, lg_info in config.LEAGUE_MAPPING.items():
        matches = service.df_raw[service.df_raw["League"].str.lower() == lg_code.lower()]
        teams = set(matches["HomeTeam"]).union(set(matches["AwayTeam"]))
        leagues_data.append({
            "code": lg_code,
            "name": lg_info["name"],
            "matches_count": len(matches),
            "teams_count": len(teams),
            "teams": sorted(list(teams))
        })
    return {"status": "success", "leagues": leagues_data}



@app.get("/elo/rankings", tags=["Metadata"])
def get_elo_rankings():
    """Returns World Football Elo power rankings for all clubs."""
    service = get_predictor()
    rankings_df = service.fe.elo_engine.get_rankings_table()
    return {"status": "success", "count": len(rankings_df), "rankings": rankings_df.to_dict(orient="records")}


@app.get("/predict", tags=["Prediction"])
def predict_get(
    home_team: str = Query(..., description="Home team name"),
    away_team: str = Query(..., description="Away team name"),
    method: str = Query("ml", description="'ml' or 'poisson'"),
    match_date: Optional[str] = Query(None, description="Fixture date YYYY-MM-DD")
):
    """Generates match outcome prediction via GET query parameters."""
    return execute_prediction(home_team, away_team, method, match_date)


@app.post("/predict", tags=["Prediction"])
def predict_post(req: PredictionRequest):
    """Generates match outcome prediction via POST JSON body."""
    return execute_prediction(req.home_team, req.away_team, req.method, req.match_date)


def execute_prediction(
    home_team: str,
    away_team: str,
    method: str,
    match_date: Optional[str] = None
) -> Dict[str, Any]:
    try:
        service = get_predictor()
        res = service.predict_match(
            home_team, away_team, method=method, display=False, match_date=match_date
        )
        # Ensure score_matrix is JSON serializable
        clean_res = dict(res)
        clean_res["model_data_through"] = str(service.df_raw["Date"].max())
        clean_res["prediction_generated_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        clean_res["training_data_sha256"] = hashlib.sha256(
            config.OUTPUT_FILE.read_bytes()
        ).hexdigest()
        if "score_matrix" in clean_res and hasattr(clean_res["score_matrix"], "tolist"):
            clean_res["score_matrix"] = clean_res["score_matrix"].tolist()
        return {"status": "success", "data": clean_res}
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}")


@app.post("/value-bet", tags=["Value Betting"])
def check_value_bets(req: ValueBetRequest):
    """
    Evaluates market odds against model probabilities to identify +EV betting opportunities.
    """
    service = get_predictor()
    try:
        pred = service.predict_match(
            req.home_team,
            req.away_team,
            method=req.method,
            display=False,
            match_date=req.match_date
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    engine = ValueBettingEngine(default_kelly_fraction=req.kelly_fraction or 0.25)
    model_probs = pred["probabilities_3way"]

    analysis = engine.analyze_market_odds(model_probs, req.market_odds)

    return {
        "status": "success",
        "home_team": pred["home_team"],
        "away_team": pred["away_team"],
        "model_method": pred["method"],
        "disclaimer": (
            "Probabilities and expected value are estimates, not guarantees of score, "
            "betting success, or profit. Submitted odds are not independently verified."
        ),
        "market_analysis": analysis,
        "summary": {
            "has_value_bets": any(item["is_value_bet"] for item in analysis),
            "best_bet": analysis[0] if analysis else None
        }
    }


odds_client = LiveOddsSyncClient()
football_data_client = FootballDataClient()


@app.get("/fixtures/sample", tags=["Fixtures & Odds"])
def get_sample_fixtures():
    """
    Provides upcoming mock matchday fixtures for testing batch predictions.
    """
    fixtures = [
        {"id": 1, "home": "Arsenal", "away": "Chelsea"},
        {"id": 2, "home": "Manchester City", "away": "Liverpool"},
        {"id": 3, "home": "Tottenham", "away": "Aston Villa"},
        {"id": 4, "home": "Newcastle United", "away": "Manchester United"},
        {"id": 5, "home": "Brighton", "away": "West Ham"},
    ]
    return {
        "source": "hard-coded sample",
        "is_mock": True,
        "use_for_betting": False,
        "fixtures": fixtures
    }


@app.get("/odds/live", tags=["Fixtures & Odds"])
def get_live_odds_endpoint(method: str = "ml"):
    """
    Fetches upcoming fixtures with live bookmaker odds (Pinnacle, Bet365, etc.)
    and automatically highlights Positive EV (+EV) value bets.
    """
    try:
        scan_results = odds_client.scan_for_value_bets(method=method)
    except OddsSourceUnavailable as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error)
        ) from error
    return {
        "status": "success",
        "count": len(scan_results),
        "fixtures": scan_results
    }


class SyncOddsRequest(BaseModel):
    method: str = Field("ml", example="ml")
    min_ev: float = Field(0.0, example=5.0)
    force: bool = Field(False, example=False)


@app.post("/odds/sync-and-commit", tags=["Fixtures & Odds"])
def sync_and_commit_odds_endpoint(req: SyncOddsRequest):
    """
    Fetches live upcoming fixtures and bookmaker odds, scans for +EV bets,
    and commits qualified bets directly to the SQLite tracking ledger.
    Ideal for external scheduler/cron automations.
    """
    try:
        fixtures = odds_client.fetch_live_fixtures_and_odds()
    except OddsSourceUnavailable as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error)
        ) from error
    scan_results = odds_client.scan_for_value_bets(fixtures, method=req.method)
    committed = []
    skipped = []

    for item in scan_results:
        bb = item.get("best_bet")
        if bb and bb.get("ev_pct", 0.0) >= req.min_ev:
            res = tracking_service.predict_and_log(
                home_team=item["home_team"],
                away_team=item["away_team"],
                market_odds=item["market_odds"],
                method=req.method,
                stake_units=1.0,
                match_date=item.get("date", "")[:10],
                skip_if_pending=not req.force
            )
            if res.get("skipped"):
                skipped.append(item["fixture"])
            else:
                committed.append({"fixture": item["fixture"], "prediction_id": res.get("prediction_id"), "ev_pct": bb.get("ev_pct")})

    return {
        "status": "success",
        "scanned_matches": len(scan_results),
        "committed_bets_count": len(committed),
        "skipped_bets_count": len(skipped),
        "committed_bets": committed,
        "skipped_fixtures": skipped
    }


# ==============================================================================
# PREDICTION TRACKING & SETTLEMENT LEDGER ENDPOINTS
# ==============================================================================

from tracking_service import PredictionTrackingService
tracking_service = PredictionTrackingService()


class LogPredictionRequest(BaseModel):
    home_team: str = Field(..., example="Arsenal")
    away_team: str = Field(..., example="Chelsea")
    market_odds: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Current bookmaker decimal odds; obtain them from /odds/live. "
            "Submitted prices are not independently verified."
        )
    )
    method: str = Field("ml", example="ml")
    stake_units: float = Field(1.0, example=1.0)
    match_date: Optional[str] = Field(None, example="2026-10-10")


class SettlePredictionRequest(BaseModel):
    prediction_id: int = Field(..., example=1)
    actual_home_goals: int = Field(..., example=2)
    actual_away_goals: int = Field(..., example=1)
    closing_odds: Optional[float] = Field(None, example=2.05)


@app.post("/tracking/log", tags=["Audit & Settlement Ledger"])
def log_prediction_endpoint(req: LogPredictionRequest):
    """Logs a pre-kickoff prediction into the SQLite tracking ledger."""
    try:
        res = tracking_service.predict_and_log(
            home_team=req.home_team,
            away_team=req.away_team,
            market_odds=req.market_odds,
            method=req.method,
            stake_units=req.stake_units,
            match_date=req.match_date
        )
        return {"status": "success", "data": res}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/tracking/settle", tags=["Audit & Settlement Ledger"])
def settle_prediction_endpoint(req: SettlePredictionRequest):
    """Settles a logged prediction against actual match score."""
    success = tracking_service.db.settle_prediction(
        prediction_id=req.prediction_id,
        actual_home_goals=req.actual_home_goals,
        actual_away_goals=req.actual_away_goals,
        closing_odds=req.closing_odds
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Prediction ID {req.prediction_id} not found.")
    return {"status": "success", "message": f"Prediction #{req.prediction_id} settled successfully."}


@app.post("/tracking/auto-settle", tags=["Audit & Settlement Ledger"])
def auto_settle_endpoint():
    """Auto-settles all pending predictions using recent matches from the clean dataset."""
    res = tracking_service.auto_settle_from_dataset()
    return res


@app.get("/tracking/summary", tags=["Audit & Settlement Ledger"])
def get_pnl_summary_endpoint():
    """Returns aggregate real-world P&L statement, Win Rate %, and ROI %."""
    return {"status": "success", "data": tracking_service.get_summary()}


@app.get("/tracking/history", tags=["Audit & Settlement Ledger"])
def get_prediction_history_endpoint(limit: int = 50):
    """Retrieves list of recently logged and settled predictions."""
    import math
    df = tracking_service.get_history_dataframe(limit=limit)
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                r[k] = None
    return {"status": "success", "count": len(records), "records": records}


# ==============================================================================
# EUROPEAN PLAYER SCOUTING & STATS ENDPOINTS
# ==============================================================================

@app.get("/players", tags=["Player Scouting"])
def get_players_endpoint(
    league: Optional[str] = None,
    team: Optional[str] = None,
    position: Optional[str] = None,
    min_minutes: int = 0,
    sort_by: str = "goals",
    limit: int = 50
):
    """
    Returns player-level performance metrics across top European leagues (EPL, La Liga, Serie A, etc.).
    Supports sorting by goals, xG, assists, xA, key_passes, xGChain.
    """
    if not PLAYERS_DB_PATH.exists():
        from scrapers.player_scraper import PlayerScraper
        scraper = PlayerScraper()
        scraper.scrape_all_european_leagues(
            season="2026",
            save_path=PLAYERS_DB_PATH
        )

    if not PLAYERS_DB_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Player statistics are unavailable; the Understat player feed could not be downloaded."
        )
    df_p = pd.read_csv(PLAYERS_DB_PATH)
    if league:
        df_p = df_p[df_p["league"].str.lower() == league.lower()]
    if team:
        df_p = df_p[df_p["team"].str.lower().str.contains(team.lower())]
    if position and position.lower() != "all":
        df_p = df_p[df_p["position"].str.lower().str.contains(position.lower())]
    if min_minutes > 0:
        df_p = df_p[df_p["minutes"] >= min_minutes]

    sort_col = sort_by if sort_by in df_p.columns else "goals"
    df_p = df_p.sort_values(by=sort_col, ascending=False).head(limit)
    return {
        "status": "success",
        "count": len(df_p),
        "players": df_p.to_dict(orient="records")
    }


@app.get("/players/leaderboard", tags=["Player Scouting"])
def get_players_leaderboard_endpoint():
    """
    Returns European Top Scorers (Goals & xG) and Top Playmakers (Assists & xA).
    """
    if not PLAYERS_DB_PATH.exists():
        from scrapers.player_scraper import PlayerScraper
        scraper = PlayerScraper()
        scraper.scrape_all_european_leagues(
            season="2026",
            save_path=PLAYERS_DB_PATH
        )

    if not PLAYERS_DB_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Player statistics are unavailable; the Understat player feed could not be downloaded."
        )
    df_p = pd.read_csv(PLAYERS_DB_PATH)
    top_scorers = df_p.sort_values(by="goals", ascending=False).head(10).to_dict(orient="records")
    top_playmakers = df_p.sort_values(by="assists", ascending=False).head(10).to_dict(orient="records")
    top_xg = df_p.sort_values(by="xG", ascending=False).head(10).to_dict(orient="records")

    return {
        "status": "success",
        "top_scorers": top_scorers,
        "top_playmakers": top_playmakers,
        "top_xg_threats": top_xg
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
