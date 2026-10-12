"""
Scrapers package initialization.
"""

from .base_scraper import BaseScraper
from .understat_scraper import UnderstatScraper
from .footballdata_scraper import FootballDataScraper

__all__ = ["BaseScraper", "UnderstatScraper", "FootballDataScraper"]
