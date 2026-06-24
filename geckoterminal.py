"""
On-chain OHLCV candles via GeckoTerminal (CoinGecko's DEX data API).

DexScreener resolves a symbol to a (chain, pair_address), but its *own* chart
endpoint (io.dexscreener.com) sits behind Cloudflare and returns 403 to
server-side requests. GeckoTerminal exposes the same on-chain candles through a
free, key-less, documented REST API keyed by network + pool address — which is
exactly what DexScreener hands us.

Docs: https://www.geckoterminal.com/dex-api
"""

import logging

import requests
import pandas as pd

logger = logging.getLogger(__name__)

_BASE = "https://api.geckoterminal.com/api/v2"
_TIMEOUT = 12
_HEADERS = {"Accept": "application/json"}
_MAX_LIMIT = 1000  # GeckoTerminal hard cap per request

# DexScreener chainId -> GeckoTerminal network id. Most match; these differ.
# Anything not listed passes through unchanged.
_NETWORK_MAP = {
    "ethereum": "eth",
    "binance-smart-chain": "bsc",
    "bsc": "bsc",
    "polygon": "polygon_pos",
    "avalanche": "avax",
    "optimism": "optimism",
    "arbitrum": "arbitrum",
    "base": "base",
    "solana": "solana",
}

# GeckoTerminal natively supports: minute {1,5,15}, hour {1,4,12}, day {1}.
# Target timeframe -> (unit, aggregate, pandas resample rule). Non-native
# targets are built by resampling a finer native resolution.
_TF_MAP = {
    "1m":  ("minute", 1,  None),
    "5m":  ("minute", 5,  None),
    "15m": ("minute", 15, None),
    "30m": ("minute", 15, "30min"),
    "1h":  ("hour",   1,  None),
    "2h":  ("hour",   1,  "2h"),
    "4h":  ("hour",   4,  None),
    "6h":  ("hour",   1,  "6h"),
    "8h":  ("hour",   4,  "8h"),
    "12h": ("hour",   12, None),
    "1d":  ("day",    1,  None),
    "3d":  ("day",    1,  "3D"),
    "1w":  ("day",    1,  "1W"),
}

_UNIT_MINUTES = {"minute": 1, "hour": 60, "day": 1440}
_TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
    "4h": 240, "6h": 360, "8h": 480, "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080,
}


def network_id(chain: str) -> str:
    c = (chain or "").lower()
    return _NETWORK_MAP.get(c, c)


def fetch_ohlcv(pool_address: str, chain: str, timeframe: str, limit: int = 200):
    """Return DataFrame[timestamp, open, high, low, close, volume] or None."""
    spec = _TF_MAP.get(timeframe)
    if not spec or not pool_address:
        return None
    unit, aggregate, resample_rule = spec
    net = network_id(chain)

    # Pull enough base candles to cover `limit` target candles post-resample.
    if resample_rule:
        base_minutes = _UNIT_MINUTES[unit] * aggregate
        factor = max(1, round(_TF_MINUTES.get(timeframe, base_minutes) / base_minutes))
    else:
        factor = 1
    base_limit = min(_MAX_LIMIT, max(limit * factor, 50))

    url = f"{_BASE}/networks/{net}/pools/{pool_address}/ohlcv/{unit}"
    params = {"aggregate": aggregate, "limit": base_limit, "currency": "usd"}
    try:
        r = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code != 200:
            logger.warning("GeckoTerminal %s %s -> HTTP %s", net, timeframe, r.status_code)
            return None
        rows = ((r.json().get("data") or {}).get("attributes") or {}).get("ohlcv_list") or []
    except Exception as e:
        logger.warning("GeckoTerminal fetch failed (%s %s): %s", net, timeframe, e)
        return None

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="s")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    if resample_rule:
        df = (
            df.set_index("timestamp")
            .resample(resample_rule, label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )

    if df is None or len(df) == 0:
        return None
    return df.tail(limit).reset_index(drop=True)
