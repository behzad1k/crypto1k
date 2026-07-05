"""
Backfills forward price outcomes for past alerts, so the evaluation report
(crypto1k/core/evaluation.py) can measure whether an alert was actually worth
sending — instead of guessing from the alert-time metrics alone.

Approach: alerts already store a DexScreener url ("https://dexscreener.com/
{chain}/{pair_address}"), and GeckoTerminal serves free OHLCV candles for any
DexScreener pair keyed by (network, pool_address). For each alert we fetch one
window of 5m candles anchored to the alert's own timestamp — 1h before through
24h after — and read off the closes we need. No continuous background
recording required; this works retroactively on alerts already in the DB.

Public surface:
  backfill_outcomes(limit=200) -> {'processed': int, 'completed': int, 'skipped': int}
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import pandas as pd

from crypto1k.config.scanner_config import BTC_IMPACT
from crypto1k.core import db
from crypto1k.data import dexscreener, geckoterminal

logger = logging.getLogger(__name__)

_WINDOW_BEFORE = timedelta(hours=1)
_WINDOW_AFTER = timedelta(hours=24)
_PER_REQUEST_PAUSE = 2.2  # GeckoTerminal free tier is ~30 req/min

# Per-process cache for the case-recovery relookup (see _resolve_pool), so a
# backfill run only hits DexScreener once per distinct pool, not once per alert.
_recovered_case_cache = {}


def _parse_pool(url: str):
    """'https://dexscreener.com/base/0xabc...' -> ('base', '0xabc...')."""
    if not url:
        return None, None
    try:
        parts = urlparse(url).path.strip('/').split('/')
        if len(parts) < 2:
            return None, None
        return parts[0], parts[1]
    except Exception:
        return None, None


def _resolve_pool(alert: dict):
    """
    (chain, correctly-cased pool address) for an alert.

    Newer alerts store the real-cased pair_address directly (see
    scanner.evaluate()). Older alerts only have the DexScreener url, whose
    slug lowercases the address — harmless for EVM chains but silently wrong
    for case-sensitive base58 addresses (Solana), which 404 on GeckoTerminal.
    For those we recover the real case with one case-insensitive DexScreener
    lookup, cached per pool for the rest of the run.
    """
    chain, url_slug = _parse_pool(alert.get('url'))
    if alert.get('pair_address'):
        return chain, alert['pair_address']
    if not url_slug:
        return None, None

    cache_key = (chain, url_slug.lower())
    if cache_key in _recovered_case_cache:
        return chain, _recovered_case_cache[cache_key]

    pair = dexscreener.get_pair(chain, url_slug)
    real_address = (pair or {}).get('pairAddress')
    _recovered_case_cache[cache_key] = real_address  # cache the miss too (None)
    return chain, real_address


def _closest_close(df: pd.DataFrame, target: datetime, tolerance_minutes: float = 45):
    """Close of the candle nearest `target`, or None if none is within tolerance."""
    if df is None or df.empty:
        return None
    diffs = (df['timestamp'] - pd.Timestamp(target)).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > pd.Timedelta(minutes=tolerance_minutes):
        return None
    return float(df.loc[idx, 'close'])


def _window_extreme(df: pd.DataFrame, start: datetime, end: datetime, col: str, how: str):
    mask = (df['timestamp'] >= pd.Timestamp(start)) & (df['timestamp'] <= pd.Timestamp(end))
    sub = df.loc[mask, col]
    if sub.empty:
        return None
    return float(sub.max() if how == 'max' else sub.min())


_btc_candles = {'df': None, 'fetched_at': None}


def _btc_window_change(start: datetime, end: datetime):
    """
    BTC's % move over [start, end], from 1h candles of the reference WBTC pool
    (same one btc_market quotes). One fetch covers ~41 days of alerts and is
    reused for the whole backfill run, so this adds a single request.
    """
    if not BTC_IMPACT.get('enabled'):
        return None
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cached = _btc_candles['df']
    if cached is None or (now - _btc_candles['fetched_at']) > timedelta(hours=1):
        chain, pool = BTC_IMPACT['pair']
        cached = geckoterminal.fetch_ohlcv(pool, chain, '1h', limit=1000)
        if cached is None or cached.empty:
            return None
        _btc_candles.update(df=cached, fetched_at=now)
        time.sleep(_PER_REQUEST_PAUSE)

    btc_start = _closest_close(cached, start, tolerance_minutes=90)
    btc_end = _closest_close(cached, end, tolerance_minutes=90)
    if not btc_start or not btc_end:
        return None
    return round((btc_end - btc_start) / btc_start * 100, 2)


def _window_extreme_at(df: pd.DataFrame, start: datetime, end: datetime, col: str, how: str):
    """(extreme value, candle timestamp ISO string) — or (None, None)."""
    mask = (df['timestamp'] >= pd.Timestamp(start)) & (df['timestamp'] <= pd.Timestamp(end))
    sub = df.loc[mask, col]
    if sub.empty:
        return None, None
    idx = sub.idxmax() if how == 'max' else sub.idxmin()
    return float(sub.loc[idx]), df.loc[idx, 'timestamp'].isoformat()


def compute_outcome_for_alert(alert: dict) -> dict:
    """
    Fetch candles for one alert and derive its outcome fields.
    Returns a dict ready for db.upsert_alert_outcome(), or None if the pair's
    candles aren't available (delisted pool, unsupported chain, etc).
    """
    chain, pool = _resolve_pool(alert)
    if not pool:
        return None

    # Naive UTC throughout — GeckoTerminal candle timestamps come back as
    # tz-naive (pd.to_datetime(..., unit='s')), so comparisons must match.
    alert_time = datetime.fromisoformat(alert['created_at'].replace(' ', 'T'))
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    window_end = min(alert_time + _WINDOW_AFTER, now)
    fetch_until = alert_time + _WINDOW_AFTER
    span_minutes = (fetch_until - (alert_time - _WINDOW_BEFORE)).total_seconds() / 60
    candle_limit = min(1000, int(span_minutes / 5) + 10)

    df = geckoterminal.fetch_ohlcv(
        pool, chain, '5m', limit=candle_limit,
        before_timestamp=int(fetch_until.timestamp()),
    )
    if df is None or df.empty:
        return None

    price_at_alert = _closest_close(df, alert_time)
    if price_at_alert is None:
        return None

    data_complete = now >= alert_time + _WINDOW_AFTER

    # Micro-cap pairs commonly trade heavily right at alert time (that's why
    # they alerted) and go quiet afterward, so candles get sparser the further
    # out we look — a tight tolerance works near the alert but misses real gaps
    # by the 4h/24h mark. Widen tolerance for the later horizons accordingly.
    high_24h, high_24h_at = _window_extreme_at(df, alert_time, window_end, 'high', 'max')

    fields = {
        'price_at_alert':  price_at_alert,
        'price_1h_before': _closest_close(df, alert_time - timedelta(hours=1), tolerance_minutes=60),
        'price_15m':       _closest_close(df, alert_time + timedelta(minutes=15)),
        'price_1h':        _closest_close(df, alert_time + timedelta(hours=1), tolerance_minutes=60),
        'price_4h':        _closest_close(df, alert_time + timedelta(hours=4), tolerance_minutes=120),
        'price_24h':       _closest_close(df, alert_time + timedelta(hours=24), tolerance_minutes=240) if data_complete else None,
        'high_1h':         _window_extreme(df, alert_time, alert_time + timedelta(hours=1), 'high', 'max'),
        'low_1h':          _window_extreme(df, alert_time, alert_time + timedelta(hours=1), 'low', 'min'),
        'high_24h':        high_24h,
        'high_24h_at':     high_24h_at,
        'btc_change_window': _btc_window_change(alert_time, window_end),
        'low_24h':         _window_extreme(df, alert_time, window_end, 'low', 'min'),
        'candles_used':    len(df),
        'data_complete':   int(data_complete),
    }
    return fields


def backfill_outcomes(limit: int = 200) -> dict:
    """Compute outcomes for alerts that don't have a complete one yet."""
    alerts = db.get_alerts_needing_outcome(limit=limit)
    processed = completed = skipped = 0

    for alert in alerts:
        processed += 1
        try:
            fields = compute_outcome_for_alert(alert)
        except Exception as e:
            logger.warning(f"Outcome backfill failed for alert {alert['id']} "
                            f"({alert['symbol']}): {e}")
            fields = None

        if fields is None:
            skipped += 1
        else:
            db.upsert_alert_outcome(alert['id'], fields)
            if fields['data_complete']:
                completed += 1
        time.sleep(_PER_REQUEST_PAUSE)

    logger.info(f"Outcome backfill: {processed} processed, "
                f"{completed} complete, {skipped} skipped (no candle data)")
    return {'processed': processed, 'completed': completed, 'skipped': skipped}
