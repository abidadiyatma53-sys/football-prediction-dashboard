"""
Base Scraper Module.
Provides robust HTTP session management, rate limiting, and exponential backoff retry logic.
"""

import time
import logging
from typing import Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config

logger = logging.getLogger(__name__)


class BaseScraper:
    """
    Base HTTP client implementing reliable scraping practices:
    - Session reuse for HTTP Keep-Alive.
    - Automatic exponential backoff retries for transient errors & rate limits (429, 5xx).
    - Polite request delay between requests to avoid blocking.
    """

    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = config.REQUEST_TIMEOUT,
        max_retries: int = config.MAX_RETRIES,
        backoff_factor: float = config.BACKOFF_FACTOR,
        request_delay: float = config.REQUEST_DELAY
    ):
        self.timeout = timeout
        self.request_delay = request_delay
        self.session = requests.Session()

        # Merge default headers
        final_headers = config.DEFAULT_HEADERS.copy()
        if headers:
            final_headers.update(headers)
        self.session.headers.update(final_headers)

        # Configure retry strategy: retry on HTTP status codes (429, 5xx)
        # Set connect=0 to fail-fast on unreachable/blocked domains instead of waiting minutes
        retry_strategy = Retry(
            total=max_retries,
            connect=0,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _wait_rate_limit(self):
        """Enforces a polite delay between successive requests."""
        if self.request_delay > 0:
            time.sleep(self.request_delay)

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Optional[requests.Response]:
        """
        Executes an HTTP GET request safely with error logging.

        Args:
            url: Target URL to fetch.
            params: Optional query parameters.
            headers: Optional request-specific headers.
            timeout: Optional custom timeout in seconds.

        Returns:
            Response object if successful, or None on critical failure.
        """
        self._wait_rate_limit()
        t = timeout or self.timeout

        try:
            logger.debug(f"Fetching URL: {url}")
            response = self.session.get(url, params=params, headers=headers, timeout=t)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as err_h:
            logger.error(f"HTTP error fetching {url}: {err_h}")
        except requests.exceptions.ConnectionError as err_c:
            logger.error(f"Connection error fetching {url}: {err_c}")
        except requests.exceptions.Timeout as err_t:
            logger.error(f"Timeout error fetching {url}: {err_t}")
        except requests.exceptions.RequestException as err:
            logger.error(f"Unexpected error fetching {url}: {err}")

        return None

    def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Optional[Any]:
        """
        Fetches an endpoint and parses JSON content safely.
        """
        response = self.get(url, params=params, headers=headers, timeout=timeout)
        if response is not None:
            try:
                return response.json()
            except ValueError as json_err:
                logger.error(f"Failed to parse JSON from {url}: {json_err}")
        return None

    def close(self):
        """Closes the underlying requests session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
