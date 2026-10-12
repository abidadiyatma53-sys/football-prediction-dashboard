"""
Team Name Normalizer Module.
Standardizes club names across disparate football data sources (Understat, Football-Data.co.uk, FBref)
to ensure consistent aggregation, merging, and modeling.
"""

import re
import unicodedata
from typing import Optional


# Canonical team mappings: Maps known variations/abbreviations to standard canonical names
CANONICAL_TEAMS = {
    # Premier League
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "manchester united": "Manchester United",
    "manchester utd": "Manchester United",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur",
    "spurs": "Tottenham Hotspur",
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    "wolverhampton wanderers": "Wolverhampton Wanderers",
    "newcastle": "Newcastle United",
    "newcastle utd": "Newcastle United",
    "newcastle united": "Newcastle United",
    "west ham": "West Ham United",
    "west ham utd": "West Ham United",
    "west ham united": "West Ham United",
    "brighton": "Brighton and Hove Albion",
    "brighton and hove albion": "Brighton and Hove Albion",
    "brighton & hove albion": "Brighton and Hove Albion",
    "nottingham forest": "Nottingham Forest",
    "nott'm forest": "Nottingham Forest",
    "nottm forest": "Nottingham Forest",
    "nottingham": "Nottingham Forest",
    "sheffield united": "Sheffield United",
    "sheffield utd": "Sheffield United",
    "sheffield weds": "Sheffield Wednesday",
    "leeds": "Leeds United",
    "leeds united": "Leeds United",
    "leicester": "Leicester City",
    "leicester city": "Leicester City",
    "norwich": "Norwich City",
    "norwich city": "Norwich City",
    "luton": "Luton Town",
    "luton town": "Luton Town",
    "bournemouth": "AFC Bournemouth",
    "afc bournemouth": "AFC Bournemouth",
    "crystal palace": "Crystal Palace",
    "aston villa": "Aston Villa",
    "arsenal": "Arsenal",
    "chelsea": "Chelsea",
    "liverpool": "Liverpool",
    "everton": "Everton",
    "fulham": "Fulham",
    "brentford": "Brentford",
    "burnley": "Burnley",
    "watford": "Watford",
    "southampton": "Southampton",
    "ipswich": "Ipswich Town",
    "ipswich town": "Ipswich Town",

    # La Liga
    "atletico madrid": "Atletico Madrid",
    "atletico de madrid": "Atletico Madrid",
    "athletic club": "Athletic Bilbao",
    "athletic bilbao": "Athletic Bilbao",
    "barcelona": "Barcelona",
    "real madrid": "Real Madrid",
    "sevilla": "Sevilla",
    "valencia": "Valencia",
    "villarreal": "Villarreal",
    "real betis": "Real Betis",
    "betis": "Real Betis",
    "real sociedad": "Real Sociedad",
    "celta vigo": "Celta Vigo",
    "celta de vigo": "Celta Vigo",
    "rayo vallecano": "Rayo Vallecano",
    "osasuna": "Osasuna",
    "getafe": "Getafe",
    "girona": "Girona",
    "mallorca": "Mallorca",
    "real mallorca": "Mallorca",
    "alaves": "Deportivo Alaves",
    "deportivo alaves": "Deportivo Alaves",
    "cadiz": "Cadiz",
    "granada": "Granada",
    "las palmas": "Las Palmas",
    "almeria": "Almeria",

    # Serie A
    "inter": "Inter Milan",
    "internazionale": "Inter Milan",
    "inter milan": "Inter Milan",
    "milan": "AC Milan",
    "ac milan": "AC Milan",
    "juventus": "Juventus",
    "roma": "AS Roma",
    "as roma": "AS Roma",
    "lazio": "Lazio",
    "ss lazio": "Lazio",
    "napoli": "Napoli",
    "atalanta": "Atalanta",
    "fiorentina": "Fiorentina",
    "bologna": "Bologna",
    "torino": "Torino",
    "monza": "Monza",
    "genoa": "Genoa",
    "udinese": "Udinese",
    "cagliari": "Cagliari",
    "hellas verona": "Hellas Verona",
    "verona": "Hellas Verona",
    "empoli": "Empoli",
    "lecce": "Lecce",
    "frosinone": "Frosinone",
    "salernitana": "Salernitana",
    "sassuolo": "Sassuolo",

    # Bundesliga
    "bayern munich": "Bayern Munich",
    "bayern munchen": "Bayern Munich",
    "fc bayern munchen": "Bayern Munich",
    "borussia dortmund": "Borussia Dortmund",
    "bvb dortmund": "Borussia Dortmund",
    "dortmund": "Borussia Dortmund",
    "bayer leverkusen": "Bayer Leverkusen",
    "leverkusen": "Bayer Leverkusen",
    "rb leipzig": "RB Leipzig",
    "rasenballsport leipzig": "RB Leipzig",
    "stuttgart": "VfB Stuttgart",
    "vfb stuttgart": "VfB Stuttgart",
    "eintracht frankfurt": "Eintracht Frankfurt",
    "frankfurt": "Eintracht Frankfurt",
    "borussia monchengladbach": "Borussia Monchengladbach",
    "m gladbach": "Borussia Monchengladbach",
    "monchengladbach": "Borussia Monchengladbach",
    "hoffenheim": "TSG Hoffenheim",
    "tsg hoffenheim": "TSG Hoffenheim",
    "freiburg": "SC Freiburg",
    "sc freiburg": "SC Freiburg",
    "augsburg": "FC Augsburg",
    "wolfsburg": "VfL Wolfsburg",
    "mainz 05": "Mainz 05",
    "mainz": "Mainz 05",
    "union berlin": "Union Berlin",
    "werder bremen": "Werder Bremen",
    "bochum": "VfL Bochum",
    "koln": "FC Koln",
    "fc cologne": "FC Koln",
    "cologne": "FC Koln",
    "darmstadt": "SV Darmstadt 98",
    "heidenheim": "FC Heidenheim",

    # Ligue 1
    "paris saint germain": "Paris Saint-Germain",
    "paris saint-germain": "Paris Saint-Germain",
    "paris sg": "Paris Saint-Germain",
    "psg": "Paris Saint-Germain",
    "marseille": "Olympique Marseille",
    "olympique de marseille": "Olympique Marseille",
    "lyon": "Olympique Lyon",
    "olympique lyonnais": "Olympique Lyon",
    "monaco": "AS Monaco",
    "as monaco": "AS Monaco",
    "lille": "Lille",
    "nice": "OGC Nice",
    "lens": "RC Lens",
    "rennes": "Stade Rennais",
    "reims": "Stade de Reims",
    "toulouse": "Toulouse",
    "montpellier": "Montpellier",
    "strasbourg": "Strasbourg",
    "nantes": "FC Nantes",
    "le havre": "Le Havre",
    "metz": "FC Metz",
    "lorient": "FC Lorient",
    "clermont foot": "Clermont",
    "brest": "Stade Brestois"
}


def strip_accents(text: str) -> str:
    """Removes diacritics and accents (e.g., München -> Munchen, Atlético -> Atletico)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def clean_team_name(raw_name: Optional[str]) -> str:
    """
    Cleans raw team name strings by:
    1. Handling None/empty strings.
    2. Stripping accents/diacritics.
    3. Normalizing whitespace, punctuation, and common suffixes (FC, CF, etc.).
    """
    if not raw_name or not isinstance(raw_name, str):
        return ""

    # Convert to lowercase and strip accents
    cleaned = strip_accents(raw_name.strip().lower())

    # Replace common symbols
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"['\"\.]", "", cleaned)
    cleaned = re.sub(r"[-_/]", " ", cleaned)

    # Remove generic club designations at start/end
    cleaned = re.sub(r"\b(fc|cf|afc|sc|sv|vfl|vfb|tsg|rc|ogc|as|ss)\b", "", cleaned)

    # Collapse multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def normalize_team_name(raw_name: Optional[str]) -> str:
    """
    Normalizes a team name into its standard canonical representation.
    Falls back to a clean Title Case string if not in known dictionary.

    Args:
        raw_name: The raw string representation of a team name.

    Returns:
        The canonical, standardized team name.
    """
    if not raw_name or not isinstance(raw_name, str):
        return "Unknown Team"

    # Step 1: Direct lookup in canonical dictionary (lowercased)
    lower_raw = raw_name.strip().lower()
    if lower_raw in CANONICAL_TEAMS:
        return CANONICAL_TEAMS[lower_raw]

    # Step 2: Cleaned lookup without prefixes/accents
    cleaned = clean_team_name(raw_name)
    if cleaned in CANONICAL_TEAMS:
        return CANONICAL_TEAMS[cleaned]

    # Step 3: Partial substring matching in canonical dictionary
    for alias, canonical in CANONICAL_TEAMS.items():
        if cleaned == alias or (len(alias) > 4 and alias in cleaned):
            return canonical

    # Step 4: Fallback to Title Cased clean string
    return raw_name.strip().title()
