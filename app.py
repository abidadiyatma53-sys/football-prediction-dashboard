"""
Streamlit Web Dashboard for Football Score Prediction & Value Betting.
Provides interactive match forecasting, market odds comparison (+EV),
scoreline probability heatmaps, and batch fixture simulation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from pathlib import Path

import config
from normalizer import normalize_team_name, CANONICAL_TEAMS
from predictor import MatchPredictorService
from value_bet import ValueBettingEngine
from tracking_service import PredictionTrackingService
from live_odds import LiveOddsSyncClient, OddsSourceUnavailable
from football_data_service import FootballDataClient, FootballDataSourceUnavailable
from monte_carlo import simulate_match
from scrapers.understat_scraper import UnderstatScraper

# Page configuration
st.set_page_config(
    page_title="Football Match Prediction & Value Betting Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek Dark Theme CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #0b0f19 !important;
        color: #f1f5f9 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }

    /* KPI Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        margin-bottom: 0.5rem;
    }
    .metric-card .title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }
    .metric-card .value {
        font-size: 1.85rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #f8fafc;
    }
    .metric-card .sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Matchup Banner */
    .matchup-hero {
        background: radial-gradient(circle at 50% 0%, rgba(59, 130, 246, 0.15), transparent 70%),
                    linear-gradient(180deg, #131d35 0%, #0b0f19 100%);
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .hero-team {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .hero-vs {
        font-size: 1.1rem;
        font-weight: 700;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        display: inline-block;
    }
    .score-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 800;
        color: #10b981;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.2rem 1.25rem;
        border-radius: 12px;
        display: inline-block;
    }
    
    /* Value Bet Badges */
    .badge-value {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-positive {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-negative {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Tabs styling */
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_secret_or_environment(name: str) -> str:
    """Read an optional credential from environment variables or Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, ""))
    except FileNotFoundError:
        return ""


@st.cache_resource
def load_prediction_service():
    """Initializes and caches the heavy prediction service."""
    return MatchPredictorService(data_path=str(config.OUTPUT_FILE))


try:
    service = load_prediction_service()
    available_teams = sorted(list(service.df_raw["HomeTeam"].unique()))
except Exception as e:
    st.error(f"Error initializing predictor service: {e}")
    st.stop()

value_engine = ValueBettingEngine()
tracking_service = PredictionTrackingService()
odds_sync_client = LiveOddsSyncClient(
    api_key=get_secret_or_environment("ODDS_API_KEY") or None
)
football_data_client = FootballDataClient(
    api_token=get_secret_or_environment("FOOTBALL_DATA_API_TOKEN") or None
)

# Sidebar: Match Selection & Global Settings
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/53/53283.png", width=42)
    st.title("Football Predictive Engine")
    st.caption("European Top Flights: EPL, La Liga, Serie A, Bundesliga, Ligue 1, RFPL")
    st.divider()

    st.subheader("🌍 Liga & Klub Eropa")
    
    league_choices = {
        "ALL": "🌍 Semua Liga Eropa (149 Klub)",
        "EPL": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League (Inggris)",
        "La_liga": "🇪🇸 La Liga (Spanyol)",
        "Bundesliga": "🇩🇪 Bundesliga (Jerman)",
        "Serie_A": "🇮🇹 Serie A (Italia)",
        "Ligue_1": "🇫🇷 Ligue 1 (Prancis)",
        "RFPL": "🇷🇺 Russian Premier League"
    }
    
    selected_lg = st.selectbox(
        "Filter Berdasarkan Liga",
        options=list(league_choices.keys()),
        format_func=lambda x: league_choices[x],
        index=0,
        key="sidebar_league_filter"
    )

    if selected_lg == "ALL":
        filtered_teams = available_teams
    else:
        sub_df = service.df_raw[service.df_raw["League"].str.lower() == selected_lg.lower()]
        teams_in_lg = set(sub_df["HomeTeam"]).union(set(sub_df["AwayTeam"]))
        filtered_teams = sorted([t for t in available_teams if t in teams_in_lg])
        if not filtered_teams:
            filtered_teams = available_teams

    # Check session state for direct selection from Tab 3 (Jadwal)
    preset_home = st.session_state.get("selected_fixture_home")
    preset_away = st.session_state.get("selected_fixture_away")

    default_home = preset_home if preset_home and preset_home in filtered_teams else (
        "Arsenal" if "Arsenal" in filtered_teams else (
            "Barcelona" if "Barcelona" in filtered_teams else (
                "Real Madrid" if "Real Madrid" in filtered_teams else (
                    "Bayern Munich" if "Bayern Munich" in filtered_teams else filtered_teams[0]
                )
            )
        )
    )

    home_team = st.selectbox(
        "Klub Tuan Rumah (Home)",
        options=filtered_teams,
        index=filtered_teams.index(default_home) if default_home in filtered_teams else 0,
        key="home_select"
    )

    remaining_teams = [t for t in filtered_teams if t != home_team]
    if not remaining_teams:
        remaining_teams = [t for t in available_teams if t != home_team]

    default_away = preset_away if preset_away and preset_away in remaining_teams else (
        "Chelsea" if "Chelsea" in remaining_teams else (
            "Atletico Madrid" if "Atletico Madrid" in remaining_teams else (
                "Getafe" if "Getafe" in remaining_teams else (
                    "Borussia Dortmund" if "Borussia Dortmund" in remaining_teams else remaining_teams[0]
                )
            )
        )
    )

    away_team = st.selectbox(
        "Klub Tamu (Away)",
        options=remaining_teams,
        index=remaining_teams.index(default_away) if default_away in remaining_teams else 0,
        key="away_select"
    )

    # Quick Pick from Upcoming European Schedule
    euro_fixtures_path = config.DATA_DIR / "upcoming_european_fixtures.json"
    if euro_fixtures_path.exists():
        try:
            with open(euro_fixtures_path, "r", encoding="utf-8") as f:
                upcoming_list = json.load(f)
            today = pd.Timestamp.now(tz="UTC").date().isoformat()
            upcoming_list = [
                fixture for fixture in upcoming_list
                if fixture.get("date", "") >= today
            ]
            
            # Format top 35 upcoming fixtures for dropdown
            fix_options = ["-- Pilih dari Jadwal Pertandingan --"] + [
                f"{fx.get('date')} [{fx.get('league', '')}] {fx.get('home_team')} vs {fx.get('away_team')}"
                for fx in upcoming_list[:40]
            ]
            
            quick_pick = st.selectbox(
                "⚡ Quick Pick dari Jadwal:",
                options=fix_options,
                index=0,
                help="Pilih pertandingan terjadwal untuk memuat klub secara instan"
            )
            
            if quick_pick != "-- Pilih dari Jadwal Pertandingan --":
                chosen_idx = fix_options.index(quick_pick) - 1
                chosen_fix = upcoming_list[chosen_idx]
                if st.button("Terapkan Laga Ini", width="stretch"):
                    st.session_state["selected_fixture_home"] = chosen_fix["home_team"]
                    st.session_state["selected_fixture_away"] = chosen_fix["away_team"]
                    st.session_state["selected_fixture_context"] = {
                        "home_team": chosen_fix["home_team"],
                        "away_team": chosen_fix["away_team"],
                        "date": chosen_fix.get("date"),
                        "league": chosen_fix.get("league"),
                    }
                    st.rerun()
        except Exception as fixture_error:
            st.warning(f"Jadwal pilihan tidak dapat dimuat: {fixture_error}")

    st.divider()
    st.subheader("Model Configuration")
    model_method = st.radio(
        "Prediction Algorithm",
        options=["Dixon-Coles Poisson", "XGBoost Machine Learning", "Compare Both"],
        index=0,
        help="Dixon-Coles Poisson: Parametric goal expectancy with bivariate low-score (0-0, 1-1) correlation adjustment factor rho. XGBoost: Supervised gradient boosting regression."
    )

    st.divider()
    st.markdown("### Production Tools")
    st.markdown("• **FastAPI API:** perlu di-deploy/dijalankan secara terpisah")
    st.markdown("• **SQLite Ledger:** penyimpanan lokal/instance hosting")
    st.caption("Engineered Features: Rolling xG, Attack/Defense Index, Form Pts")


# Main App Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 Prediksi Skor Pertandingan",
    "💰 Value Betting & Odds (+EV)",
    "📅 Jadwal & Prediksi Liga Eropa",
    "📈 Real P&L & Audit Ledger",
    "📊 Backtest & Performance",
    "🌟 European Player Scouting (2026/2027)",
    "🚀 API & Telegram Integration"
])


# ==============================================================================
# TAB 1: MATCH PREDICTION & PROBABILITIES
# ==============================================================================
with tab1:
    method_key = "poisson" if "Poisson" in model_method else ("ml" if "XGBoost" in model_method else "both")

    # Run Prediction
    pred_poisson = service.predict_match(home_team, away_team, method="poisson", display=False)
    pred_ml = service.predict_match(home_team, away_team, method="ml", display=False)
    active_pred = pred_poisson if method_key == "poisson" else pred_ml

    # Matchup Banner
    h_elo_val = active_pred.get('home_elo', 1500.0)
    a_elo_val = active_pred.get('away_elo', 1500.0)
    diff_elo_val = active_pred.get('diff_elo', 0.0)
    
    # Identify league
    h_lg_matches = service.df_raw[service.df_raw["HomeTeam"] == home_team]["League"].values
    lg_label = config.LEAGUE_MAPPING.get(h_lg_matches[0], {}).get("name", h_lg_matches[0]) if len(h_lg_matches) > 0 else "European League"

    top3_scores = active_pred.get("top_exact_scores", [])[:3]
    top3_badges_html = " ".join([
        f"<span style='background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 6px; padding: 0.2rem 0.5rem; margin: 0 0.25rem; font-size: 0.78rem; font-weight: 600; color: #f8fafc;'>{sc['score']} <span style='color:#38bdf8;'>({sc['percentage']}%)</span></span>"
        for sc in top3_scores
    ])

    st.markdown(f"""
    <div class="matchup-hero">
        <div style="text-align: center; margin-bottom: 0.75rem;">
            <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 20px; padding: 0.25rem 0.9rem; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em; color: #38bdf8; text-transform: uppercase;">
                ⚽ {lg_label}
            </span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; text-align: center;">
            <div style="flex: 1;">
                <div class="hero-team">{home_team}</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">TUAN RUMAH (HOME)</div>
                <div style="color: #38bdf8; font-family: monospace; font-size: 0.95rem; font-weight: 700; margin-top: 0.2rem;">Elo: {h_elo_val:.1f}</div>
            </div>
            <div style="padding: 0 1.5rem;">
                <span class="hero-vs">VS</span>
                <div style="margin-top: 0.5rem;">
                    <div style="color: #94a3b8; font-size: 0.75rem; letter-spacing: 0.05em; margin-bottom: 0.2rem;">PREDIKSI SKOR UTAMA</div>
                    <div class="score-badge">{active_pred['most_likely_score']}</div>
                    <div style="color: #38bdf8; font-size: 0.8rem; font-weight: 700; margin-top: 0.25rem;">PROBABILITAS: {active_pred['most_likely_probability']}%</div>
                    <div style="color: #a78bfa; font-size: 0.75rem; margin-top: 0.2rem; font-weight: 600;">Net Elo Edge: {diff_elo_val:+.1f}</div>
                    <div style="margin-top: 0.6rem; color: #94a3b8; font-size: 0.75rem;">
                        Top Alternatif: {top3_badges_html}
                    </div>
                </div>
            </div>
            <div style="flex: 1;">
                <div class="hero-team">{away_team}</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">TIM TAMU (AWAY)</div>
                <div style="color: #34d399; font-family: monospace; font-size: 0.95rem; font-weight: 700; margin-top: 0.2rem;">Elo: {a_elo_val:.1f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 KPI Stat Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Expected Goals (xG)</div>
            <div class="value" style="color: #38bdf8;">{active_pred['lambda_home']:.2f} - {active_pred['lambda_away']:.2f}</div>
            <div class="sub">{home_team} vs {away_team}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Attack Strength Ratio</div>
            <div class="value" style="color: #a78bfa;">{active_pred['home_attack_strength']:.2f} vs {active_pred['away_attack_strength']:.2f}</div>
            <div class="sub">League Benchmark = 1.00</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Defense Weakness Ratio</div>
            <div class="value" style="color: #f43f5e;">{active_pred['home_defense_weakness']:.2f} vs {active_pred['away_defense_weakness']:.2f}</div>
            <div class="sub">Lower is Stronger Defense</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Recent Form (Last 5)</div>
            <div class="value" style="color: #34d399;">{int(active_pred['home_form_pts_5'])} pts vs {int(active_pred['away_form_pts_5'])} pts</div>
            <div class="sub">Points from last 5 matches</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Contextual Rest & Venue Badge Bar
    h_rest = active_pred.get("home_rest_days", 7)
    a_rest = active_pred.get("away_rest_days", 7)
    diff_rest = active_pred.get("diff_rest_days", 0)
    h_vpts = active_pred.get("home_venue_pts_5", 7)
    a_vpts = active_pred.get("away_venue_pts_5", 5)

    c_rest1, c_rest2 = st.columns(2)
    with c_rest1:
        rest_edge_color = "#34d399" if diff_rest > 0 else ("#f87171" if diff_rest < 0 else "#94a3b8")
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid #334155; border-radius: 8px; padding: 0.6rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <b>⏳ Rest Days & Squad Fatigue:</b> {home_team} <span style="color: #38bdf8;">({h_rest}d)</span> vs {away_team} <span style="color: #34d399;">({a_rest}d)</span>
            </div>
            <div style="color: {rest_edge_color}; font-weight: 700;">
                Rest Advantage: {diff_rest:+} Days
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c_rest2:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid #334155; border-radius: 8px; padding: 0.6rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <b>🏟️ Venue-Specific Split Form:</b> {home_team} Home <span style="color: #38bdf8;">({h_vpts} pts)</span> vs {away_team} Away <span style="color: #34d399;">({a_vpts} pts)</span>
            </div>
            <div style="color: #a78bfa; font-weight: 700;">
                Home Ground Factor
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 1X2 Match Outcomes Probabilities
    col_prob, col_scores = st.columns([1.2, 1.0])

    with col_prob:
        st.subheader("📊 3-Way Match Outcome Probabilities (1X2)")
        probs = active_pred["probabilities_3way"]

        # Tripartite Stacked Probability Chart
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Bar(
            y=["Outcome"], x=[probs["home_win"]], name=f"Home: {home_team}",
            orientation='h', marker=dict(color="#3b82f6"),
            text=f"{probs['home_win']}%", textposition="auto"
        ))
        fig_prob.add_trace(go.Bar(
            y=["Outcome"], x=[probs["draw"]], name="Draw",
            orientation='h', marker=dict(color="#64748b"),
            text=f"{probs['draw']}%", textposition="auto"
        ))
        fig_prob.add_trace(go.Bar(
            y=["Outcome"], x=[probs["away_win"]], name=f"Away: {away_team}",
            orientation='h', marker=dict(color="#10b981"),
            text=f"{probs['away_win']}%", textposition="auto"
        ))
        fig_prob.update_layout(
            barmode='stack', height=130, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
            xaxis=dict(showgrid=False, range=[0, 100], showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        st.plotly_chart(fig_prob, width="stretch")

        if method_key == "both":
            st.caption("Comparison: Poisson vs XGBoost ML Probabilities")
            comp_df = pd.DataFrame([
                {"Model": "Poisson Distribution", "Home Win": f"{pred_poisson['probabilities_3way']['home_win']}%", "Draw": f"{pred_poisson['probabilities_3way']['draw']}%", "Away Win": f"{pred_poisson['probabilities_3way']['away_win']}%", "xG Home": pred_poisson['lambda_home'], "xG Away": pred_poisson['lambda_away']},
                {"Model": "XGBoost Multi-Output", "Home Win": f"{pred_ml['probabilities_3way']['home_win']}%", "Draw": f"{pred_ml['probabilities_3way']['draw']}%", "Away Win": f"{pred_ml['probabilities_3way']['away_win']}%", "xG Home": pred_ml['lambda_home'], "xG Away": pred_ml['lambda_away']}
            ])
            st.dataframe(comp_df, width="stretch", hide_index=True)

    with col_scores:
        st.subheader("🎯 Top Exact Scorelines")
        top_scores = active_pred["top_exact_scores"]
        scores_df = pd.DataFrame(top_scores)
        
        fig_scores = px.bar(
            scores_df,
            x="percentage",
            y="score",
            orientation="h",
            text="percentage",
            color="percentage",
            color_continuous_scale="Blues",
            labels={"score": "Scoreline", "percentage": "Probability (%)"}
        )
        fig_scores.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed")
        )
        fig_scores.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_scores, width="stretch")

    # Scoreline Heatmap Matrix (Poisson 6x6)
    if "score_matrix" in active_pred:
        st.subheader(f"🔥 Exact Score Probability Matrix ({home_team} vs {away_team})")
        matrix = np.asarray(active_pred["score_matrix"], dtype=float) * 100.0
        home_goals = [str(i) for i in range(matrix.shape[0])]
        away_goals = [str(i) for i in range(matrix.shape[1])]
        
        fig_matrix = px.imshow(
            matrix,
            labels=dict(x=f"{away_team} Goals", y=f"{home_team} Goals", color="Prob (%)"),
            x=away_goals,
            y=home_goals,
            text_auto=".1f",
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        fig_matrix.update_layout(
            height=340,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_matrix, width="stretch")

    with st.expander("🎲 Simulasi Monte Carlo — skenario & ketidakpastian", expanded=False):
        st.caption(
            "Jalankan simulasi skor berulang dengan ketidakpastian xG dan skenario lineup. "
            "Dampak lineup tidak ditebak otomatis: bobot skenario dan pengali xG harus "
            "ditentukan pengguna. Contoh pengali 0,90 berarti estimasi gol tim turun 10%; "
            "dampak lineup bertahan dapat dimodelkan dengan mengubah xG lawan. Angka ini "
            "adalah asumsi, bukan rating pemain tervalidasi."
        )
        with st.form(f"monte_carlo_form_{home_team}_{away_team}"):
            st.markdown("**Asumsi simulasi**")
            mc_col1, mc_col2, mc_col3 = st.columns(3)
            with mc_col1:
                mc_simulations = st.number_input(
                    "Jumlah simulasi",
                    min_value=1_000,
                    max_value=100_000,
                    value=20_000,
                    step=5_000,
                    key=f"mc_count_{home_team}_{away_team}",
                )
            with mc_col2:
                mc_parameter_cv = st.number_input(
                    "Ketidakpastian xG (CV %)",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0,
                    help="Asumsi stress-test yang diterapkan sebagai variasi lognormal di sekitar xG model; bukan confidence interval yang dipelajari dari data.",
                    key=f"mc_cv_{home_team}_{away_team}",
                )
            with mc_col3:
                mc_seed = st.number_input(
                    "Seed (agar hasil dapat diulang)",
                    min_value=0,
                    max_value=2_147_483_647,
                    value=42,
                    step=1,
                    key=f"mc_seed_{home_team}_{away_team}",
                )

            st.markdown(
                "**Kemungkinan lineup / kondisi tim** — probabilitas skenario dinormalisasi "
                "oleh simulasi bila totalnya bukan 100%."
            )
            scenario_headers = st.columns([2, 1, 1, 1])
            for column, label in zip(
                scenario_headers,
                ["Nama skenario", "Bobot (%)", "Pengali xG kandang", "Pengali xG tandang"],
            ):
                column.caption(label)

            mc_scenarios = []
            scenario_defaults = [
                ("Skenario dasar", 100, 1.0, 1.0),
                ("Kemungkinan lineup 2", 0, 1.0, 1.0),
                ("Kemungkinan lineup 3", 0, 1.0, 1.0),
            ]
            for index, (default_name, default_weight, default_home, default_away) in enumerate(
                scenario_defaults, start=1
            ):
                scenario_cols = st.columns([2, 1, 1, 1])
                scenario = {"name": "", "weight": 0, "home_multiplier": 1.0, "away_multiplier": 1.0}
                with scenario_cols[0]:
                    scenario["name"] = st.text_input(
                        f"Nama skenario {index}",
                        value=default_name,
                        label_visibility="collapsed",
                        key=f"mc_name_{index}_{home_team}_{away_team}",
                    )
                with scenario_cols[1]:
                    scenario["weight"] = st.number_input(
                        f"Bobot skenario {index}",
                        min_value=0,
                        max_value=100,
                        value=default_weight,
                        step=5,
                        label_visibility="collapsed",
                        key=f"mc_weight_{index}_{home_team}_{away_team}",
                    )
                with scenario_cols[2]:
                    scenario["home_multiplier"] = st.number_input(
                        f"Pengali xG kandang skenario {index}",
                        min_value=0.1,
                        max_value=3.0,
                        value=default_home,
                        step=0.05,
                        label_visibility="collapsed",
                        key=f"mc_home_mult_{index}_{home_team}_{away_team}",
                    )
                with scenario_cols[3]:
                    scenario["away_multiplier"] = st.number_input(
                        f"Pengali xG tandang skenario {index}",
                        min_value=0.1,
                        max_value=3.0,
                        value=default_away,
                        step=0.05,
                        label_visibility="collapsed",
                        key=f"mc_away_mult_{index}_{home_team}_{away_team}",
                    )
                mc_scenarios.append(scenario)

            run_monte_carlo = st.form_submit_button(
                "Jalankan simulasi",
                type="primary",
                width="stretch",
            )

        if run_monte_carlo:
            try:
                mc_result = simulate_match(
                    lambda_home=float(active_pred["lambda_home"]),
                    lambda_away=float(active_pred["lambda_away"]),
                    scenarios=mc_scenarios,
                    simulations=int(mc_simulations),
                    parameter_cv=float(mc_parameter_cv) / 100,
                    rho=(
                        float(pred_poisson["rho"])
                        if method_key == "poisson"
                        else None
                    ),
                    seed=int(mc_seed),
                )
                st.session_state["monte_carlo_result"] = {
                    "key": f"{home_team}:{away_team}:{method_key}",
                    "data": mc_result,
                }
            except ValueError as error:
                st.error(f"Parameter simulasi tidak valid: {error}")

        saved_mc = st.session_state.get("monte_carlo_result", {})
        if saved_mc.get("key") == f"{home_team}:{away_team}:{method_key}":
            mc_result = saved_mc["data"]
            st.caption(
                f"{mc_result['simulations']:,} simulasi · model {active_pred['method']} · "
                f"CV xG {mc_result['parameter_cv_pct']:.1f}%"
                + (
                    f" · Dixon–Coles ρ={mc_result['dixon_coles_rho']:.4f}"
                    if mc_result["dixon_coles_rho"] is not None
                    else " · gol kandang/tandang independen (Poisson)"
                )
            )
            mc_kpi1, mc_kpi2, mc_kpi3 = st.columns(3)
            for container, outcome, title in [
                (mc_kpi1, "home_win", f"Peluang menang {home_team}"),
                (mc_kpi2, "draw", "Peluang seri"),
                (mc_kpi3, "away_win", f"Peluang menang {away_team}"),
            ]:
                result = mc_result["outcomes"][outcome]
                container.metric(title, f"{result['percentage']:.2f}%")
                container.caption(
                    f"95% CI sampling: {result['ci95_pct'][0]:.2f}%–"
                    f"{result['ci95_pct'][1]:.2f}%"
                )

            mc_scores_col, mc_scenarios_col = st.columns(2)
            with mc_scores_col:
                st.markdown("**Skor yang paling sering muncul di simulasi**")
                st.dataframe(
                    pd.DataFrame(mc_result["top_scorelines"]),
                    hide_index=True,
                    width="stretch",
                )
            with mc_scenarios_col:
                st.markdown("**Hasil per skenario**")
                st.dataframe(
                    pd.DataFrame(mc_result["scenario_results"]),
                    hide_index=True,
                    width="stretch",
                )
            st.caption(
                f"Rata-rata gol simulasi: {mc_result['mean_home_goals']:.2f}–"
                f"{mc_result['mean_away_goals']:.2f}. P90 gol: "
                f"{mc_result['home_goals_90th_percentile']}–"
                f"{mc_result['away_goals_90th_percentile']}. Interval di atas hanya "
                "mengukur variasi sampling Monte Carlo, bukan jaminan atau rentang pasti hasil laga."
            )

    with st.expander("👥 Lineup Manual & Verifikasi Fixture", expanded=False):
        st.caption(
            "Football-data.org dapat memverifikasi fixture, status, dan skor akhir, tetapi "
            "tidak menyediakan lineup. Masukkan susunan pemain secara manual; data ini hanya "
            "ditampilkan sebagai konteks dan tidak mengubah prediksi skor."
        )
        fixture_context = st.session_state.get("selected_fixture_context", {})
        context_matches_selection = (
            fixture_context.get("home_team") == home_team
            and fixture_context.get("away_team") == away_team
        )
        default_date = (
            fixture_context.get("date")
            if context_matches_selection and fixture_context.get("date")
            else pd.Timestamp.now().date().isoformat()
        )
        team_rows = service.df_raw[
            service.df_raw["HomeTeam"].isin([home_team, away_team])
            | service.df_raw["AwayTeam"].isin([home_team, away_team])
        ]
        home_leagues = set(team_rows[
            team_rows["HomeTeam"].eq(home_team) | team_rows["AwayTeam"].eq(home_team)
        ]["League"])
        away_leagues = set(team_rows[
            team_rows["HomeTeam"].eq(away_team) | team_rows["AwayTeam"].eq(away_team)
        ]["League"])
        shared_leagues = sorted(home_leagues & away_leagues)
        lineup_leagues = ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]
        if shared_leagues:
            lineup_leagues = [league for league in lineup_leagues if league in shared_leagues]
        context_league = (
            fixture_context.get("league")
            if context_matches_selection else None
        )
        if context_league not in lineup_leagues:
            context_league = lineup_leagues[0] if lineup_leagues else "EPL"

        lineup_col1, lineup_col2, lineup_col3 = st.columns([1, 1, 1])
        with lineup_col1:
            lineup_date = st.date_input(
                "Tanggal pertandingan",
                value=pd.Timestamp(default_date).date(),
                key=f"lineup_date_{home_team}_{away_team}_{default_date}",
            )
        with lineup_col2:
            lineup_league = st.selectbox(
                "Liga",
                lineup_leagues or ["EPL"],
                index=(lineup_leagues.index(context_league) if context_league in lineup_leagues else 0),
                format_func=lambda code: config.LEAGUE_MAPPING.get(code, {}).get("name", code),
                key=f"lineup_league_{home_team}_{away_team}_{default_date}",
            )
        with lineup_col3:
            st.write("")
            verify_fixture = st.button(
                "Verifikasi fixture",
                key=f"verify_fixture_{home_team}_{away_team}_{default_date}",
                width="stretch",
            )

        fixture_key = (
            f"{lineup_league}:{lineup_date.isoformat()}:{home_team}:{away_team}"
        )
        if verify_fixture:
            try:
                st.session_state["football_data_fixture_result"] = {
                    "key": fixture_key,
                    "data": football_data_client.verify_fixture(
                        home_team=home_team,
                        away_team=away_team,
                        match_date=lineup_date.isoformat(),
                        league=lineup_league,
                    ),
                }
            except ValueError as error:
                st.error(str(error))
            except FootballDataSourceUnavailable as error:
                st.error(f"Verifikasi Football-data.org gagal: {error}")

        saved_fixture = st.session_state.get("football_data_fixture_result", {})
        if saved_fixture.get("key") == fixture_key:
            fixture_data = saved_fixture["data"]
            if fixture_data["status"] == "fixture_not_found":
                st.warning(fixture_data["message"])
            else:
                st.success(fixture_data["message"])
                st.caption(
                    f"Status: {fixture_data['status']} · Kick-off: "
                    f"{fixture_data.get('kickoff') or 'belum tersedia'} · "
                    f"Sumber: {fixture_data['source']}"
                )
                score = fixture_data.get("score", {})
                if score.get("home") is not None and score.get("away") is not None:
                    st.metric("Skor akhir dari sumber data", f"{score['home']} - {score['away']}")

        manual_home_col, manual_away_col = st.columns(2)
        with manual_home_col:
            manual_home_lineup = st.text_area(
                f"Lineup manual — {home_team}",
                placeholder="Contoh: Kiper; Bek 1, Bek 2; Gelandang 1; Penyerang 1",
                key=f"manual_lineup_home_{home_team}_{away_team}",
            )
        with manual_away_col:
            manual_away_lineup = st.text_area(
                f"Lineup manual — {away_team}",
                placeholder="Contoh: Kiper; Bek 1, Bek 2; Gelandang 1; Penyerang 1",
                key=f"manual_lineup_away_{home_team}_{away_team}",
            )
        if manual_home_lineup.strip() or manual_away_lineup.strip():
            st.info("Lineup manual tersimpan di sesi dashboard ini; prediksi skor tidak diubah.")

    with st.expander("🏆 World Football Elo Power Rankings (League Table)"):
        st.markdown("""
        Elo ratings adjust dynamically after every match based on opponent rating and margin of victory.
        Base rating is 1500, with a +65 home field advantage factor.
        """)
        rankings_df = service.fe.elo_engine.get_rankings_table()
        if not rankings_df.empty:
            c_tbl, c_chart = st.columns([1.2, 1.8])
            with c_tbl:
                st.dataframe(rankings_df, width="stretch", hide_index=True)
            with c_chart:
                fig_elo = px.bar(
                    rankings_df.sort_values(by="Elo_Rating", ascending=True),
                    x="Elo_Rating",
                    y="Team",
                    orientation="h",
                    color="Elo_Rating",
                    color_continuous_scale="Tealgrn",
                    text="Elo_Rating",
                    title="Premier League Elo Strength Distribution"
                )
                fig_elo.update_layout(
                    height=450,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#f8fafc"),
                    coloraxis_showscale=False
                )
                fig_elo.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                st.plotly_chart(fig_elo, width="stretch")


# ==============================================================================
# TAB 2: VALUE BETTING & ODDS COMPARISON (+EV & KELLY)
# ==============================================================================
with tab2:
    st.subheader("💰 Positive Expected Value (+EV) Betting Finder")
    st.markdown("""
    Bandingkan probabilitas model dengan harga bookmaker aktual dari The Odds API.
    Odds contoh/simulasi tidak digunakan. Jika feed tidak tersedia atau fixture tidak cocok,
    analisis EV dinonaktifkan.
    """)

    odds_leagues = {
        "EPL": ("Premier League", "soccer_epl"),
        "La_Liga": ("La Liga", "soccer_spain_la_liga"),
        "Bundesliga": ("Bundesliga", "soccer_germany_bundesliga"),
        "Serie_A": ("Serie A", "soccer_italy_serie_a"),
        "Ligue_1": ("Ligue 1", "soccer_france_ligue_one"),
        "RFPL": ("Russian Premier League", "soccer_russia_premier_league"),
    }
    team_leagues = service.df_raw[
        service.df_raw["HomeTeam"].isin([home_team, away_team])
        | service.df_raw["AwayTeam"].isin([home_team, away_team])
    ].groupby("League").size().sort_values(ascending=False)
    likely_league = str(team_leagues.index[0]) if not team_leagues.empty else "EPL"
    league_options = list(odds_leagues)
    league_lookup = {code.lower().replace("_", ""): code for code in league_options}
    league_index = league_options.index(
        league_lookup.get(likely_league.lower().replace("_", ""), "EPL")
    )
    odds_league = st.selectbox(
        "Liga untuk odds live",
        options=league_options,
        index=league_index,
        format_func=lambda code: odds_leagues[code][0],
        key="tab2_odds_league"
    )
    odds_selection_key = f"{odds_league}:{home_team}:{away_team}"
    if st.session_state.get("live_odds_selection_key") != odds_selection_key:
        st.session_state.pop("live_market_fixture", None)
        st.session_state["live_odds_selection_key"] = odds_selection_key

    if st.button("Ambil odds aktual dari The Odds API", type="primary"):
        try:
            live_fixtures = odds_sync_client.fetch_live_fixtures_and_odds(
                sport_key=odds_leagues[odds_league][1]
            )
        except OddsSourceUnavailable as error:
            st.error(
                f"Odds live tidak tersedia: {error} "
                "Periksa ODDS_API_KEY. Tidak ada odds simulasi yang digunakan."
            )
        else:
            matching_fixtures = [
                fixture for fixture in live_fixtures
                if normalize_team_name(fixture["home"]) == normalize_team_name(home_team)
                and normalize_team_name(fixture["away"]) == normalize_team_name(away_team)
            ]
            if len(matching_fixtures) == 1:
                st.session_state["live_market_fixture"] = matching_fixtures[0]
                st.success("Harga live untuk fixture yang cocok berhasil diambil.")
            elif len(matching_fixtures) > 1:
                st.session_state.pop("live_market_fixture", None)
                st.error("Lebih dari satu fixture cocok; analisis dibatalkan demi keamanan.")
            else:
                st.session_state.pop("live_market_fixture", None)
                st.warning(
                    "Tidak ada harga live untuk pasangan klub ini. "
                    "Pastikan liga dan fixture sudah benar; analisis EV tidak dijalankan."
                )

    market_fixture = st.session_state.get("live_market_fixture")
    if market_fixture:
        fetched_at = pd.to_datetime(market_fixture["fetched_at_utc"], utc=True)
        odds_age_minutes = (
            pd.Timestamp.now(tz="UTC") - fetched_at
        ).total_seconds() / 60
        if odds_age_minutes > 10:
            st.warning("Harga odds sudah lebih dari 10 menit. Ambil ulang sebelum menganalisis.")
            st.session_state.pop("live_market_fixture", None)
            market_fixture = None
        else:
            bookmaker_names = ", ".join(sorted(set(market_fixture["bookmakers"].values())))
            st.caption(
                f"Sumber: {market_fixture['source']} · Bookmaker: {bookmaker_names} · "
                f"Kick-off: {market_fixture['date']} · Diambil: "
                f"{fetched_at.strftime('%Y-%m-%d %H:%M:%S UTC')} · "
                f"Umur data: {odds_age_minutes:.1f} menit"
            )
            st.dataframe(
                pd.DataFrame([{
                    "Hasil": label,
                    "Odds aktual": market_fixture[key],
                    "Bookmaker": market_fixture["bookmakers"][key]
                } for label, key in [
                    (f"Menang {home_team}", "home_win"),
                    ("Seri", "draw"),
                    (f"Menang {away_team}", "away_win")
                ]]),
                hide_index=True,
                width="stretch"
            )

    with st.container():
        kelly_frac = st.slider(
            "Kelly Multiplier (Bankroll Risk)",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="0.25 = Quarter Kelly. Ukuran stake bukan jaminan hasil."
        )

    analysis = []
    market_odds_dict = None
    prediction_for_odds = None
    if market_fixture:
        market_odds_dict = {
            "home_win": market_fixture["odds_home"],
            "draw": market_fixture["odds_draw"],
            "away_win": market_fixture["odds_away"]
        }
        fixture_date = pd.Timestamp(market_fixture["date"]).date().isoformat()
        analysis_method = "ml" if method_key == "both" else method_key
        prediction_for_odds = service.predict_match(
            home_team,
            away_team,
            method=analysis_method,
            display=False,
            match_date=fixture_date
        )
        engine_custom = ValueBettingEngine(default_kelly_fraction=kelly_frac)
        analysis = engine_custom.analyze_market_odds(
            prediction_for_odds["probabilities_3way"], market_odds_dict
        )

    if analysis:
        st.markdown("#### Analisis probabilitas vs. odds aktual")
        st.caption(
            f"Probabilitas model untuk laga {market_fixture['date'][:10]}; "
            "hasil pertandingan tetap tidak pasti dan tidak ada jaminan profit."
        )
    else:
        st.info(
            "Ambil odds aktual untuk laga yang dipilih. Nilai fair odds sintetis "
            "tidak dipakai sebagai odds pasar."
        )

    val_cols = st.columns(3)
    for idx, item in enumerate(analysis):
        with val_cols[idx]:
            is_val = item["is_value_bet"]
            badge_class = "badge-positive" if is_val else "badge-negative"
            ev_color = "#34d399" if item["ev_pct"] > 0 else "#f87171"

            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {'#10b981' if is_val else '#ef4444'};">
                <div class="title">{item['label']}</div>
                <div style="margin: 0.5rem 0;">
                    <span class="badge-value {badge_class}">{item['verdict']}</span>
                </div>
                <div class="value" style="color: {ev_color};">
                    {'+' if item['ev_pct'] > 0 else ''}{item['ev_pct']}%
                </div>
                <div class="sub">Expected Value (EV)</div>
                <hr style="border-color: #334155; margin: 0.75rem 0;">
                <div style="font-size: 0.85rem; text-align: left; line-height: 1.6;">
                    <div>• <b>Model Probability:</b> {item['model_prob_pct']}%</div>
                    <div>• <b>Fair Odds (1/P):</b> {item['fair_odds']}</div>
                    <div>• <b>Market Odds:</b> {item['market_odds']}</div>
                    <div>• <b>Market Implied Probability:</b> {item['market_implied_prob_pct']}%</div>
                    <div>• <b>Edge:</b> {'+' if item['edge_pct'] > 0 else ''}{item['edge_pct']}%</div>
                    <div style="margin-top: 0.5rem; color: #38bdf8; font-weight: 700;">
                        • Suggested Kelly Stake: {item['recommended_stake_pct']}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if market_fixture and prediction_for_odds and market_odds_dict:
        st.warning(
            "EV adalah estimasi berdasarkan model dan harga pasar yang tercantum. "
            "Ini bukan jaminan skor benar, menang taruhan, atau profit."
        )
        c_save_btn, c_save_status = st.columns([1.5, 2.5])
        with c_save_btn:
            if st.button("📌 Log Prediction & Actual Odds", type="secondary", width="stretch"):
                log_res = tracking_service.predict_and_log(
                    home_team=home_team,
                    away_team=away_team,
                    market_odds=market_odds_dict,
                    method=prediction_for_odds["method"],
                    stake_units=1.0,
                    match_date=pd.Timestamp(market_fixture["date"]).date().isoformat()
                )
                st.success(
                    f"Saved prediction #{log_res['prediction_id']} with "
                    f"{market_fixture['source']} odds."
                )

    # Explanation Formula Card
    with st.expander("ℹ️ How Expected Value (+EV) and Kelly Criterion Work"):
        st.markdown(r"""
        - **Fair Odds**: Probabilitas model dibalik, yaitu $$1/P(\text{Outcome})$$.
        - **Expected Value (EV)**: $$EV = (P_{\text{model}} \times \text{Odds}_{\text{aktual}}) - 1$$.
        - **Fractional Kelly**: Sizing matematis, bukan perlindungan dari kerugian atau jaminan hasil.
        - Harga odds diambil dari bookmaker melalui The Odds API; perbarui jika sudah lebih dari 10 menit.
        """)


# ==============================================================================
# TAB 3: JADWAL & PREDIKSI SKOR SEMUA LIGA EROPA
# ==============================================================================
with tab3:
    st.subheader("📅 Jadwal Pertandingan & Prediksi Skor Semua Liga Eropa")
    st.caption("Pantau jadwal pertandingan lengkap dari 6 liga top Eropa (Premier League, La Liga, Serie A, Bundesliga, Ligue 1, RFPL) dengan prediksi skor eksak Poisson & Machine Learning.")

    # Load European Fixtures
    euro_fixtures_file = config.DATA_DIR / "upcoming_european_fixtures.json"
    all_euro_fixtures = []
    if euro_fixtures_file.exists():
        try:
            with open(euro_fixtures_file, "r", encoding="utf-8") as f:
                all_euro_fixtures = json.load(f)
            today = pd.Timestamp.now(tz="UTC").date().isoformat()
            all_euro_fixtures = [
                fixture for fixture in all_euro_fixtures
                if fixture.get("date", "") >= today
            ]
        except Exception as e:
            st.warning(f"Gagal memuat jadwal Eropa: {e}")

    # Top Toolbar
    tb_c1, tb_c2, tb_c3, tb_c4 = st.columns([1.8, 1.4, 1.4, 1.4])
    
    with tb_c1:
        league_filter_options = {
            "ALL": "🌍 Semua Liga Eropa",
            "La_Liga": "🇪🇸 La Liga (Spanyol)",
            "EPL": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League (Inggris)",
            "Bundesliga": "🇩🇪 Bundesliga (Jerman)",
            "Serie_A": "🇮🇹 Serie A (Italia)",
            "Ligue_1": "🇫🇷 Ligue 1 (Prancis)",
            "RFPL": "🇷🇺 Russian Premier League"
        }
        tab3_selected_lg = st.selectbox(
            "Pilih Liga",
            options=list(league_filter_options.keys()),
            format_func=lambda x: league_filter_options[x],
            index=0,
            key="tab3_league_filter"
        )

    # Filter available dates
    available_dates = sorted(list(set(f.get("date", "") for f in all_euro_fixtures if f.get("date"))))
    with tb_c2:
        tab3_date_filter = st.selectbox(
            "Filter Tanggal",
            options=["Semua Tanggal"] + available_dates,
            index=0,
            key="tab3_date_filter"
        )

    with tb_c3:
        tab3_search = st.text_input("🔍 Cari Klub", placeholder="Contoh: Alaves, Madrid, Arsenal...", key="tab3_search")

    with tb_c4:
        st.write("")
        st.write("")
        if st.button("🔄 Sinkron Jadwal Understat", width="stretch", type="secondary"):
            with st.spinner("Mengunduh jadwal terkini & menghitung prediksi skor..."):
                try:
                    from update_all_european_data import ingest_and_predict_upcoming_fixtures
                    updated_fix = ingest_and_predict_upcoming_fixtures(service.df_raw)
                    all_euro_fixtures = updated_fix
                    st.success(f"Berhasil memperbarui {len(updated_fix)} jadwal laga Eropa!")
                    st.rerun()
                except Exception as sync_err:
                    st.error(f"Gagal memperbarui jadwal: {sync_err}")

    # Apply Filters
    filtered_fixtures = all_euro_fixtures
    if tab3_selected_lg != "ALL":
        target_code = tab3_selected_lg.lower().replace("_", "")
        filtered_fixtures = [
            f for f in filtered_fixtures
            if f.get("league", "").lower().replace("_", "") == target_code
        ]

    if tab3_date_filter != "Semua Tanggal":
        filtered_fixtures = [f for f in filtered_fixtures if f.get("date") == tab3_date_filter]

    if tab3_search.strip():
        q = tab3_search.strip().lower()
        filtered_fixtures = [
            f for f in filtered_fixtures
            if q in f.get("home_team", "").lower()
            or q in f.get("away_team", "").lower()
            or q in f.get("home_raw", "").lower()
            or q in f.get("away_raw", "").lower()
        ]

    # Metrics Summary
    kpi_f1, kpi_f2, kpi_f3, kpi_f4 = st.columns(4)
    with kpi_f1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Total Laga Terjadwal</div>
            <div class="value" style="color: #38bdf8;">{len(all_euro_fixtures)}</div>
            <div class="sub">Seluruh Eropa 2026/2027</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_f2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Laga Sesuai Filter</div>
            <div class="value" style="color: #34d399;">{len(filtered_fixtures)}</div>
            <div class="sub">{league_filter_options[tab3_selected_lg]}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_f3:
        next_kickoff = filtered_fixtures[0].get("datetime", "-") if filtered_fixtures else "-"
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Kickoff Terdekat</div>
            <div class="value" style="font-size: 1.15rem; color: #a78bfa;">{next_kickoff[:16]}</div>
            <div class="sub">Waktu Pertandingan</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_f4:
        predictable_fixtures = [
            item for item in filtered_fixtures
            if not item.get("prediction_error")
            and item.get("home_win_prob") is not None
            and item.get("away_win_prob") is not None
        ]
        fav_match = max(
            predictable_fixtures,
            key=lambda item: max(item["home_win_prob"], item["away_win_prob"])
        ) if predictable_fixtures else None
        fav_info = f"{fav_match['home_team']} ({fav_match['home_win_prob']}%)" if fav_match else "-"
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Peluang Menang Tertinggi</div>
            <div class="value" style="font-size: 1.1rem; color: #f59e0b;">{fav_info}</div>
            <div class="sub">Berdasarkan Model Poisson</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Table of Scheduled Matches & Predicted Scores
    st.markdown("#### 📋 Tabel Jadwal Pertandingan & Prediksi Skor")

    table_data = []
    for item in filtered_fixtures:
        top_alt = ", ".join([f"{ts['score']} ({ts['percentage']}%)" for ts in item.get("top_scorelines", [])[:3]]) if item.get("top_scorelines") else "-"
        prediction_failed = bool(item.get("prediction_error"))
        table_data.append({
            "Kickoff": f"{item.get('date', '')} {item.get('time', '')}",
            "Liga": item.get("league_name", item.get("league", "")),
            "Tuan Rumah (Home)": item.get("home_team", ""),
            "Tamu (Away)": item.get("away_team", ""),
            "Status Prediksi": f"Gagal: {item['prediction_error']}" if prediction_failed else "Tersedia",
            "🎯 Prediksi Skor": "-" if prediction_failed else f"{item.get('predicted_score', '-')} ({item.get('score_prob_pct', 0)}%)",
            "Peluang (Home / Seri / Away)": "-" if prediction_failed else f"{item.get('home_win_prob', 0)}% / {item.get('draw_prob', 0)}% / {item.get('away_win_prob', 0)}%",
            "xG (Home - Away)": "-" if prediction_failed else f"{item.get('lambda_home', 0):.2f} - {item.get('lambda_away', 0):.2f}",
            "Alternatif Skor": top_alt,
            "Net Elo": "-" if prediction_failed else f"{item.get('diff_elo', 0):+.1f}"
        })

    if table_data:
        df_display = pd.DataFrame(table_data)
        st.dataframe(df_display, width="stretch", hide_index=True)

        c_down1, c_down2 = st.columns([1.5, 3])
        with c_down1:
            csv_data = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Jadwal & Prediksi Skor (CSV)",
                data=csv_data,
                file_name=f"jadwal_prediksi_{tab3_selected_lg.lower()}.csv",
                mime="text/csv",
                width="stretch"
            )

        # Match Quick Inspector & One-Click Deep Dive to Tab 1
        st.markdown("#### 🔍 Pilih Pertandingan dari Jadwal untuk Analisis Mendalam di Tab 1")
        fix_options_cards = [
            f"{fx.get('date')} [{fx.get('league', '')}] {fx.get('home_team')} vs {fx.get('away_team')} -> Prediksi: {fx.get('predicted_score')} ({fx.get('score_prob_pct')}%)"
            for fx in filtered_fixtures[:60]
        ]
        
        sel_col1, sel_col2 = st.columns([3, 1])
        with sel_col1:
            selected_fixture_str = st.selectbox(
                "Pilih Laga Terjadwal:",
                options=fix_options_cards,
                key="tab3_inspector_select"
            )
        with sel_col2:
            st.write("")
            st.write("")
            if st.button("🚀 Buka Prediksi di Tab 1", type="primary", width="stretch"):
                chosen_fx_idx = fix_options_cards.index(selected_fixture_str)
                target_fx = filtered_fixtures[chosen_fx_idx]
                st.session_state["selected_fixture_home"] = target_fx["home_team"]
                st.session_state["selected_fixture_away"] = target_fx["away_team"]
                st.session_state["selected_fixture_context"] = {
                    "home_team": target_fx["home_team"],
                    "away_team": target_fx["away_team"],
                    "date": target_fx.get("date"),
                    "league": target_fx.get("league"),
                }
                st.success(f"Memuat {target_fx['home_team']} vs {target_fx['away_team']} ke Tab 1...")
                st.rerun()

        # Detailed breakdown of first 10 matches in expanders
        st.markdown("##### 📌 Rincian Prediksi Skor per Laga")
        for item in filtered_fixtures[:15]:
            fixture_title = f"⚽ {item['home_team']} vs {item['away_team']} | Kickoff: {item.get('date')} {item.get('time')} | Prediksi Skor: {item.get('predicted_score')} ({item.get('score_prob_pct')}%)"
            with st.expander(fixture_title):
                exp_c1, exp_c2, exp_c3 = st.columns([1.3, 1.7, 1.0])
                with exp_c1:
                    st.markdown(f"**Liga:** {item.get('league_name', item.get('league'))}")
                    st.markdown(f"**Prediksi Skor Utama:** `{item.get('predicted_score')}` ({item.get('score_prob_pct')}%)")
                    st.markdown(f"**Estimasi Gol (xG):** {item.get('lambda_home', 0):.2f} - {item.get('lambda_away', 0):.2f}")
                    st.markdown(f"**Net Elo Advantage:** {item.get('diff_elo', 0):+.1f}")
                with exp_c2:
                    st.markdown("**Peluang Hasil Pertandingan (3-Way):**")
                    p_h = item.get("home_win_prob", 33)
                    p_d = item.get("draw_prob", 33)
                    p_a = item.get("away_win_prob", 33)
                    st.progress(int(p_h), text=f"Menang {item['home_team']}: {p_h}%")
                    st.progress(int(p_d), text=f"Seri (Draw): {p_d}%")
                    st.progress(int(p_a), text=f"Menang {item['away_team']}: {p_a}%")
                with exp_c3:
                    if st.button("Analisis di Tab 1", key=f"btn_deep_{item.get('match_id', item['home_team'])}"):
                        st.session_state["selected_fixture_home"] = item["home_team"]
                        st.session_state["selected_fixture_away"] = item["away_team"]
                        st.session_state["selected_fixture_context"] = {
                            "home_team": item["home_team"],
                            "away_team": item["away_team"],
                            "date": item.get("date"),
                            "league": item.get("league"),
                        }
                        st.rerun()

    else:
        st.info("Tidak ada pertandingan yang cocok dengan filter atau pencarian Anda.")


# ==============================================================================
# TAB 4: REAL P&L & AUDIT LEDGER (SQLITE)
# ==============================================================================
with tab4:
    st.subheader("📈 Real-World Prediction Audit Trail & P&L Ledger")
    st.markdown("""
    Every prediction can be locked in the persistent SQLite database prior to kickoff.
    Once matches finish, the automated settlement engine reconciles actual scores against model picks to compute real P&L and ROI.
    """)

    # Top Action Buttons
    c_act1, c_act2 = st.columns([1.5, 3])
    with c_act1:
        if st.button("⚡ Auto-Settle Pending Matches Against Dataset", type="primary", width="stretch"):
            settle_res = tracking_service.auto_settle_from_dataset()
            st.success(f"Settled {settle_res['settled_count']} pending matches against actual scores!")
            st.rerun()

    # Load Summary Stats
    summary = tracking_service.get_summary()

    p1, p2, p3, p4, p5, p6 = st.columns(6)
    with p1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Total Predictions</div>
            <div class="value">{summary['total_predictions']}</div>
            <div class="sub">Settled: {summary['settled_count']} | Pending: {summary['pending_count']}</div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Total Bets Placed</div>
            <div class="value">{summary['total_bets']}</div>
            <div class="sub">Won: {summary['won_bets']} | Lost: {summary['lost_bets']}</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Win Rate (%)</div>
            <div class="value" style="color: #38bdf8;">{summary['win_rate_pct']}%</div>
            <div class="sub">On Settled Value Bets</div>
        </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Total Staked</div>
            <div class="value">{summary['total_staked_units']} u</div>
            <div class="sub">Bankroll Units</div>
        </div>
        """, unsafe_allow_html=True)
    with p5:
        pnl_color = "#34d399" if summary['total_pnl_units'] >= 0 else "#f87171"
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Net Realized P&L</div>
            <div class="value" style="color: {pnl_color};">{'+' if summary['total_pnl_units'] > 0 else ''}{summary['total_pnl_units']} u</div>
            <div class="sub">Net Profit / Loss</div>
        </div>
        """, unsafe_allow_html=True)
    with p6:
        roi_color = "#34d399" if summary['roi_pct'] >= 0 else "#f87171"
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">ROI (%)</div>
            <div class="value" style="color: {roi_color};">{'+' if summary['roi_pct'] > 0 else ''}{summary['roi_pct']}%</div>
            <div class="sub">Return on Investment</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # History Table & Cumulative P&L Plot
    df_history = tracking_service.get_history_dataframe(limit=100)

    if not df_history.empty and summary["settled_count"] > 0:
        st.subheader("📊 Cumulative Bankroll Growth Curve (P&L Units)")
        df_settled = df_history[df_history["status"] == "SETTLED"].copy()
        df_settled = df_settled.sort_values(by="id", ascending=True)
        df_settled["cum_pnl"] = df_settled["pnl_units"].cumsum()

        fig_pnl = px.line(
            df_settled,
            x="id",
            y="cum_pnl",
            markers=True,
            labels={"id": "Bet Index", "cum_pnl": "Cumulative P&L (Units)"},
            title="Realized P&L Trajectory"
        )
        fig_pnl.update_traces(line_color="#10b981", marker=dict(size=8, color="#34d399"))
        fig_pnl.update_layout(
            height=260,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_pnl, width="stretch")

    st.subheader("📜 Detailed Prediction Audit Trail")
    if not df_history.empty:
        display_cols = [
            "id", "created_at", "home_team", "away_team", "predicted_score",
            "bet_selection", "bet_odds", "bet_ev_pct", "status",
            "actual_home_goals", "actual_away_goals", "actual_result",
            "is_won", "pnl_units"
        ]
        available_cols = [c for c in display_cols if c in df_history.columns]
        st.dataframe(df_history[available_cols], width="stretch", hide_index=True)
    else:
        st.info("No predictions logged yet. Use the 'Commit Prediction to SQLite' button in the Value Betting tab to record forecasts.")


# ==============================================================================
# TAB 5: BACKTEST & QUANT EVALUATION
# ==============================================================================
with tab5:
    st.subheader("📊 Model Evaluation & Out-of-Time (OOT) Backtest")
    metrics_path = config.BASE_DIR / "evaluation_metrics.json"
    if not metrics_path.exists():
        st.info("Belum ada hasil evaluasi terbaru. Jalankan `python run_backtest.py` untuk membuat metrik dan grafik.")
    else:
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            evaluation = json.load(metrics_file)
        data_path = Path(evaluation["data_path"])
        if not data_path.is_absolute():
            data_path = config.BASE_DIR / data_path
        if data_path.exists() and data_path.stat().st_mtime > metrics_path.stat().st_mtime:
            st.warning("Dataset berubah setelah backtest terakhir. Jalankan `python run_backtest.py` untuk memperbarui hasil.")
        else:
            chart_path = Path(evaluation.get("chart_path", "evaluation_charts.png"))
            if not chart_path.is_absolute():
                chart_path = config.BASE_DIR / chart_path
            xgb = evaluation["metrics"]["xgboost"]
            poisson = evaluation["metrics"]["poisson"]
            st.caption(
                f"Data uji: {evaluation['test_size']:,} pertandingan | "
                f"Periode: {evaluation['date_range']} | "
                f"Rasio latih: {evaluation['train_ratio']:.0%}"
            )
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("XGBoost 1X2 Accuracy", f"{xgb['accuracy_1x2']:.2f}%")
            b2.metric("XGBoost Exact Score", f"{xgb['accuracy_exact']:.2f}%")
            b3.metric("XGBoost Brier (lebih rendah lebih baik)", f"{xgb['brier_score']:.4f}")
            b4.metric("XGBoost Log-loss (lebih rendah lebih baik)", f"{xgb['log_loss']:.4f}")
            st.caption(
                f"Poisson: akurasi 1X2 {poisson['accuracy_1x2']:.2f}%, "
                f"akurasi skor tepat {poisson['accuracy_exact']:.2f}%."
            )
            if chart_path.exists():
                st.image(
                    str(chart_path),
                    caption=f"Backtest {evaluation['date_range']} ({evaluation['test_size']:,} laga)",
                    width="stretch"
                )


# ==============================================================================
# TAB 6: EUROPEAN PLAYER SCOUTING & RADAR (2026/2027)
# ==============================================================================
with tab6:
    st.subheader("🌟 European Player Scouting & Radar (Season 2026/2027)")
    st.caption("Live expected metrics (xG, xA, Key Passes, xGChain, 90s stats) across all 5 Top European Leagues.")

    players_file = config.DATA_DIR / "players_database.csv"
    if not players_file.exists():
        st.warning("⚠️ Player database not found (`data/players_database.csv`).")
        if st.button("📥 Scrape All European Players Now"):
            with st.spinner("Scraping Top 5 European Leagues (EPL, La Liga, Serie A, Bundesliga, Ligue 1)..."):
                from scrapers.player_scraper import PlayerScraper
                scraper = PlayerScraper()
                df_all = scraper.scrape_all_european_leagues(
                    season="2026",
                    save_path=players_file
                )
                st.success(f"Successfully scraped and stored {len(df_all)} players!")
                st.rerun()
    else:
        df_players = pd.read_csv(players_file)
        
        # Top KPI Metrics Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Scouted Players", f"{len(df_players):,}", delta="5 European Leagues")
        with kpi2:
            top_scorer = df_players.sort_values(by="goals", ascending=False).iloc[0]
            st.metric("Top European Scorer", f"{top_scorer['player_name']}", f"{top_scorer['goals']} Goals ({top_scorer['team']})")
        with kpi3:
            top_xg = df_players.sort_values(by="xG", ascending=False).iloc[0]
            st.metric("Highest xG Threat", f"{top_xg['player_name']}", f"{top_xg['xG']:.2f} xG ({top_xg['league']})")
        with kpi4:
            top_assist = df_players.sort_values(by="assists", ascending=False).iloc[0]
            st.metric("Top Playmaker", f"{top_assist['player_name']}", f"{top_assist['assists']} Assists ({top_assist['key_passes']} KP)")

        st.markdown("---")

        # Filters
        f1, f2, f3, f4 = st.columns([1.5, 1.2, 1.5, 2])
        with f1:
            all_leagues = ["All"] + sorted(list(df_players["league"].dropna().unique()))
            selected_league = st.selectbox("League", options=all_leagues, index=0)
        with f2:
            all_positions = ["All", "F", "M", "D", "GK"]
            selected_pos = st.selectbox("Position", options=all_positions, index=0)
        with f3:
            min_min = st.slider("Min Minutes Played", min_value=0, max_value=int(df_players["minutes"].max()), value=90, step=30)
        with f4:
            search_query = st.text_input("🔍 Search Player / Team", placeholder="e.g. Haaland, Mbappe, Real Madrid, Arsenal...")

        # Apply Filters
        filtered_df = df_players.copy()
        if selected_league != "All":
            filtered_df = filtered_df[filtered_df["league"] == selected_league]
        if selected_pos != "All":
            filtered_df = filtered_df[filtered_df["position"].astype(str).str.contains(selected_pos, na=False)]
        filtered_df = filtered_df[filtered_df["minutes"] >= min_min]
        if search_query:
            q = search_query.strip().lower()
            filtered_df = filtered_df[
                filtered_df["player_name"].str.lower().str.contains(q, na=False) |
                filtered_df["team"].str.lower().str.contains(q, na=False)
            ]

        # Leaderboard Tabs inside Tab 6
        l_tab1, l_tab2 = st.tabs(["📊 Interactive Radar & Scatter", "📋 Full Player Data Table"])

        with l_tab1:
            st.markdown("#### 🎯 Attacking Efficiency: Expected Goals per 90 (xG90) vs Expected Assists per 90 (xA90)")
            if not filtered_df.empty:
                scatter_df = filtered_df[filtered_df["minutes"] >= 90].copy()
                if scatter_df.empty:
                    scatter_df = filtered_df.copy()

                fig_scatter = px.scatter(
                    scatter_df,
                    x="xA90",
                    y="xG90",
                    color="league",
                    size="minutes",
                    hover_name="player_name",
                    hover_data={
                        "team": True,
                        "goals": True,
                        "assists": True,
                        "xG": ":.2f",
                        "xA": ":.2f",
                        "xGChain": ":.2f",
                        "minutes": True,
                        "xA90": False,
                        "xG90": False
                    },
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    title="Expected Output per 90 Minutes (Bubble Size = Minutes Played)"
                )
                fig_scatter.update_layout(
                    paper_bgcolor="#0b0f19",
                    plot_bgcolor="#0f172a",
                    font_color="#f1f5f9",
                    height=520,
                    xaxis=dict(gridcolor="#1e293b", title="Expected Assists per 90 (xA90)"),
                    yaxis=dict(gridcolor="#1e293b", title="Expected Goals per 90 (xG90)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_scatter, width="stretch")

            # Top Scorers vs Top Creators Side-by-Side
            col_sc, col_cr = st.columns(2)
            with col_sc:
                st.markdown("##### ⚽ Top Scorers & xG")
                top_sc_df = filtered_df.sort_values(by="goals", ascending=False).head(10)[
                    ["player_name", "team", "league", "matches", "goals", "xG", "shots"]
                ]
                st.dataframe(top_sc_df, width="stretch", hide_index=True)

            with col_cr:
                st.markdown("##### 🪄 Top Creators & xA")
                top_cr_df = filtered_df.sort_values(by="assists", ascending=False).head(10)[
                    ["player_name", "team", "league", "matches", "assists", "xA", "key_passes"]
                ]
                st.dataframe(top_cr_df, width="stretch", hide_index=True)

        with l_tab2:
            st.markdown(f"#### 📋 Filtered Players Database ({len(filtered_df):,} matching)")
            display_cols = [
                "player_name", "team", "league", "position", "matches", "minutes",
                "goals", "assists", "xG", "xA", "key_passes", "shots", "xGChain", "xG90", "xA90"
            ]
            valid_cols = [c for c in display_cols if c in filtered_df.columns]
            st.dataframe(filtered_df[valid_cols].sort_values(by="goals", ascending=False), width="stretch", hide_index=True)

            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Players as CSV",
                data=csv_data,
                file_name=f"european_players_2026_{selected_league}.csv",
                mime="text/csv"
            )


# ==============================================================================
# TAB 7: REST API & TELEGRAM BOT INTEGRATION
# ==============================================================================
with tab7:
    st.subheader("🚀 FastAPI REST API & Bot Automation")
    st.markdown("""
    You can trigger live predictions or integrate with Telegram Bots and mobile apps via our REST API.
    """)
    st.info(
        "Dashboard saja tidak menjalankan FastAPI. Contoh endpoint di bawah hanya berfungsi "
        "jika layanan `api.py` juga dijalankan atau di-host secara terpisah."
    )

    st.markdown("#### 1. Live Endpoints")
    st.code("""
# Fetch real bookmaker odds and the model's value analysis.
# Requires ODDS_API_KEY; HTTP 503 means no live feed is available.
curl -X GET "http://127.0.0.1:8000/odds/live?method=ml"

# GET Single Match Prediction
curl -X GET "http://127.0.0.1:8000/predict?home_team=Arsenal&away_team=Chelsea&method=poisson"

# Verify fixture status and final score (requires FOOTBALL_DATA_API_TOKEN).
curl -X GET "http://127.0.0.1:8000/fixtures/verify?home_team=Arsenal&away_team=Chelsea&match_date=2026-10-10&league=EPL"

# GET European Players Database (Season 2026/2027)
curl -X GET "http://127.0.0.1:8000/players?league=EPL&sort_by=goals&limit=10"

# GET European Leaderboard (Top Scorers & Playmakers)
curl -X GET "http://127.0.0.1:8000/players/leaderboard"

# POST /value-bet accepts decimal prices copied from /odds/live.
# Include match_date and current prices; caller-supplied prices are not verified.
    """, language="bash")

    st.caption(
        "Odds bersumber dari The Odds API dan membutuhkan ODDS_API_KEY. "
        "Probabilitas model adalah estimasi, bukan jaminan skor, kemenangan taruhan, atau profit."
    )
    st.markdown("#### 2. Python Telegram Bot Webhook Example")
    st.code("""
import requests

def send_telegram_forecast(bot_token, chat_id, home, away):
    api_url = f"http://127.0.0.1:8000/predict?home_team={home}&away_team={away}"
    res = requests.get(api_url).json()
    data = res["data"]
    
    msg = (
        f"⚽ *MATCH FORECAST: {data['home_team']} vs {data['away_team']}*\\n"
        f"🎯 Expected Score: `{data['most_likely_score']}` ({data['most_likely_probability']}%)\\n"
        f"📊 Win Probs: H {data['probabilities_3way']['home_win']}% | "
        f"D {data['probabilities_3way']['draw']}% | "
        f"A {data['probabilities_3way']['away_win']}%"
    )
    requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
    )
    """, language="python")
