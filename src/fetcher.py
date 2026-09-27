"""
ETH price data fetcher using CoinGecko public API (no API key required).
Includes in-memory caching to avoid rate limits (429).
"""

import time
import requests
import pandas as pd
from datetime import datetime

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Simple in-memory cache: {key: (timestamp, data)}
_cache: dict = {}
CACHE_TTL = 300  # seconds (5 minutes)


def _cached(key: str, fetch_fn):
    now = time.time()
    if key in _cache:
        ts, data = _cache[key]
        if now - ts < CACHE_TTL:
            return data
    data = fetch_fn()
    _cache[key] = (now, data)
    return data


def _get_with_retry(url: str, params: dict, retries: int = 3) -> requests.Response:
    """GET with exponential backoff on 429."""
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code == 429:
            wait = 2 ** attempt * 5  # 5s, 10s, 20s
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def fetch_ohlc(days: int = 90) -> pd.DataFrame:
    """
    Fetch ETH OHLC candlestick data from CoinGecko (cached 5 min).
    """
    def _fetch():
        url = f"{COINGECKO_BASE}/coins/ethereum/ohlc"
        params = {"vs_currency": "usd", "days": days}
        try:
            resp = _get_with_retry(url, params)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch OHLC data: {e}")
        raw = resp.json()
        if not raw:
            raise ValueError("Empty OHLC response from CoinGecko.")
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    return _cached(f"ohlc_{days}", _fetch)


def fetch_market_data() -> dict:
    """
    Fetch current ETH market snapshot (cached 5 min).
    """
    def _fetch():
        url = f"{COINGECKO_BASE}/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": "ethereum",
            "order": "market_cap_desc",
            "per_page": 1,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "1h,24h,7d",
        }
        try:
            resp = _get_with_retry(url, params)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch market data: {e}")
        data = resp.json()
        if not data:
            raise ValueError("Empty market data response.")
        coin = data[0]
        return {
            "price":       coin.get("current_price", 0),
            "market_cap":  coin.get("market_cap", 0),
            "volume_24h":  coin.get("total_volume", 0),
            "change_1h":   coin.get("price_change_percentage_1h_in_currency", 0),
            "change_24h":  coin.get("price_change_percentage_24h_in_currency", 0),
            "change_7d":   coin.get("price_change_percentage_7d_in_currency", 0),
            "high_24h":    coin.get("high_24h", 0),
            "low_24h":     coin.get("low_24h", 0),
            "last_updated":coin.get("last_updated", ""),
        }

    return _cached("market", _fetch)
