"""
ETH price data fetcher.

Sources:
  - Binance public API  → intraday candles (1m, 5m, 15m, 1h, 4h) — no key needed
  - CoinGecko public API → daily candles (swing mode) + market snapshot

Cache: 5 min TTL for all requests to avoid rate limits.
"""

import time
import requests
import pandas as pd

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL = 300  # 5 minutes


def _cached(key: str, fetch_fn):
    now = time.time()
    if key in _cache:
        ts, data = _cache[key]
        if now - ts < CACHE_TTL:
            return data
    data = fetch_fn()
    _cache[key] = (now, data)
    return data


def _get(url: str, params: dict = None, retries: int = 3) -> requests.Response:
    """GET with exponential backoff on 429 / 5xx."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params or {}, timeout=20)
            if resp.status_code == 429:
                time.sleep(2 ** attempt * 5)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError as e:
            if attempt == retries - 1:
                raise ConnectionError(f"Connection failed: {e}")
            time.sleep(2 ** attempt)
    resp.raise_for_status()
    return resp


# ── Binance intraday candles ──────────────────────────────────────────────────
BINANCE_BASE = "https://api.binance.com/api/v3"

# Maps our internal interval names to Binance kline interval codes
BINANCE_INTERVALS = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}

# How many candles to fetch per interval for day-trading context
INTERVAL_LIMITS = {
    "1m":  200,   # ~3.3 hours
    "5m":  200,   # ~16.7 hours
    "15m": 200,   # ~50 hours (2 days)
    "30m": 200,   # ~100 hours (4 days)
    "1h":  200,   # ~8 days
    "4h":  200,   # ~33 days
    "1d":  200,   # ~200 days
}


def fetch_binance_ohlc(interval: str = "15m", limit: int = None) -> pd.DataFrame:
    """
    Fetch ETH/USDT OHLC candles from Binance.

    Args:
        interval: Candle size — '1m','5m','15m','30m','1h','4h','1d'
        limit:    Number of candles (default from INTERVAL_LIMITS)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if interval not in BINANCE_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}. Use one of {list(BINANCE_INTERVALS)}")

    n = limit or INTERVAL_LIMITS.get(interval, 200)
    cache_key = f"binance_{interval}_{n}"

    def _fetch():
        url = f"{BINANCE_BASE}/klines"
        params = {
            "symbol":   "ETHUSDT",
            "interval": BINANCE_INTERVALS[interval],
            "limit":    n,
        }
        try:
            resp = _get(url, params)
        except Exception as e:
            raise ConnectionError(f"Binance OHLC fetch failed: {e}")

        raw = resp.json()
        if not raw:
            raise ValueError("Empty response from Binance.")

        # Binance kline format:
        # [open_time, open, high, low, close, volume, close_time, ...]
        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    return _cached(cache_key, _fetch)


# ── CoinGecko daily (swing mode fallback) ─────────────────────────────────────
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def fetch_ohlc(days: int = 90) -> pd.DataFrame:
    """Fetch daily OHLC from CoinGecko (swing trading / long-term view)."""
    def _fetch():
        url = f"{COINGECKO_BASE}/coins/ethereum/ohlc"
        params = {"vs_currency": "usd", "days": days}
        try:
            resp = _get(url, params)
        except Exception as e:
            raise ConnectionError(f"CoinGecko OHLC fetch failed: {e}")
        raw = resp.json()
        if not raw:
            raise ValueError("Empty OHLC response from CoinGecko.")
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["volume"] = 0.0
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    return _cached(f"cg_ohlc_{days}", _fetch)


# ── Market snapshot (CoinGecko) ────────────────────────────────────────────────
def fetch_market_data() -> dict:
    """Fetch current ETH price, changes, volume from CoinGecko (cached 5 min)."""
    def _fetch():
        url = f"{COINGECKO_BASE}/coins/markets"
        params = {
            "vs_currency": "usd", "ids": "ethereum",
            "order": "market_cap_desc", "per_page": 1, "page": 1,
            "sparkline": False, "price_change_percentage": "1h,24h,7d",
        }
        try:
            resp = _get(url, params)
        except Exception as e:
            raise ConnectionError(f"Market data fetch failed: {e}")
        data = resp.json()
        if not data:
            raise ValueError("Empty market data response.")
        c = data[0]
        return {
            "price":        c.get("current_price", 0),
            "market_cap":   c.get("market_cap", 0),
            "volume_24h":   c.get("total_volume", 0),
            "change_1h":    c.get("price_change_percentage_1h_in_currency", 0),
            "change_24h":   c.get("price_change_percentage_24h_in_currency", 0),
            "change_7d":    c.get("price_change_percentage_7d_in_currency", 0),
            "high_24h":     c.get("high_24h", 0),
            "low_24h":      c.get("low_24h", 0),
            "last_updated": c.get("last_updated", ""),
        }

    return _cached("market", _fetch)
