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
import time

import requests
import pandas as pd

logger = logging.getLogger(__name__)

_RATE_LIMIT_RETRIES = 3
_RATE_LIMIT_BACKOFF = 3.0  # seconds; GeckoTerminal's free tier is ~30 req/min

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


def fetch_ohlcv(pool_address: str, chain: str, timeframe: str, limit: int = 200,
                 before_timestamp: int = None):
    """
    Return DataFrame[timestamp, open, high, low, close, volume] or None.

    before_timestamp (unix seconds) pages backward from that point instead of
    from now — used to fetch a window anchored to a specific past moment
    (e.g. the candles around when an alert fired), rather than only "recent".
    """
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
    if before_timestamp:
        params["before_timestamp"] = int(before_timestamp)
    r = None
    try:
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            r = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
            if r.status_code != 429:
                break
            if attempt < _RATE_LIMIT_RETRIES:
                time.sleep(_RATE_LIMIT_BACKOFF * (attempt + 1))
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


def fetch_trades(pool_address: str, chain: str, min_volume_usd: float = 0.0) -> list:
    """
    Recent trades for a pool (GeckoTerminal returns up to the last ~300 trades
    from the past 24h). Each trade includes the wallet address that sent the
    tx — the raw material for smart-money tracking.

    Returns a list of normalized dicts, newest first:
      {trade_key, tx_hash, wallet, side ('buy'|'sell'), amount_usd,
       price_usd, block_ts (ISO, naive UTC)}
    """
    if not pool_address:
        return []
    net = network_id(chain)
    url = f"{_BASE}/networks/{net}/pools/{pool_address}/trades"
    params = {}
    if min_volume_usd:
        params["trade_volume_in_usd_greater_than"] = min_volume_usd
    try:
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            r = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
            if r.status_code != 429:
                break
            if attempt < _RATE_LIMIT_RETRIES:
                time.sleep(_RATE_LIMIT_BACKOFF * (attempt + 1))
        if r.status_code != 200:
            logger.warning("GeckoTerminal trades %s/%s -> HTTP %s",
                           net, pool_address, r.status_code)
            return []
        rows = (r.json().get("data")) or []
    except Exception as e:
        logger.warning("GeckoTerminal trades fetch failed (%s/%s): %s",
                       net, pool_address, e)
        return []

    out = []
    for row in rows:
        a = row.get("attributes") or {}
        wallet = (a.get("tx_from_address") or "").strip()
        side = a.get("kind")
        if not wallet or side not in ("buy", "sell"):
            continue
        try:
            usd = float(a.get("volume_in_usd") or 0.0)
        except (TypeError, ValueError):
            usd = 0.0
        # Base-token USD price at trade time: for a buy the base token is the
        # "to" side, for a sell it's the "from" side.
        raw_price = a.get("price_to_in_usd") if side == "buy" else a.get("price_from_in_usd")
        try:
            price_usd = float(raw_price) if raw_price is not None else None
        except (TypeError, ValueError):
            price_usd = None
        block_ts = (a.get("block_timestamp") or "").replace("Z", "").replace(" ", "T")
        out.append({
            "trade_key": row.get("id") or f"{a.get('tx_hash')}_{block_ts}",
            "tx_hash":   a.get("tx_hash"),
            "wallet":    wallet.lower() if wallet.startswith("0x") else wallet,
            "side":      side,
            "amount_usd": usd,
            "price_usd": price_usd,
            "block_ts":  block_ts,
        })
    return out
