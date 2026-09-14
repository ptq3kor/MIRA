"""
Shared HTTP retry/backoff helper for API calls.
"""

import time
import requests


def post_with_retry(url: str, json: dict, headers: dict | None = None,
                     timeout: int = 60, max_retries: int = 3) -> dict:
    """
    POST request with exponential backoff retry.
    
    Args:
        url: Target URL.
        json: JSON payload.
        headers: Optional headers.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
    
    Returns:
        Parsed JSON response.
    
    Raises:
        RuntimeError: If all retries exhausted.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=json, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise RuntimeError(f"Request to {url} failed after {max_retries} attempts") from last_exc