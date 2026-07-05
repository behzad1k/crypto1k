"""
BTC market regime: price, recent moves, and a 0–10 regime score.

BTC's movement drags nearly every coin with it (instantly through non-stable
quote currencies, within minutes-to-hours through trader behavior), so each
alert gets stamped with the BTC picture at fire time. The regime score is kept
SEPARATE from the coin's own validity score on purpose: the validity score
answers "how real is this coin's spike?", the regime score answers "is the
market letting anything pump right now?" — blending them would just move the
alert threshold market-wide while making individual scores unreadable.

Score model (weights in scanner_config.BTC_IMPACT):
    5.0 (neutral) + m5% * w_m5 + h1% * w_h1 + h24% * w_h24, clamped to 0–10.
So flat BTC ≈ 5, BTC -2% on the hour ≈ 3 (risk-off), BTC +2% ≈ 7 (risk-on).
Short windows are weighted hardest because a fresh BTC dump kills alt pumps
faster than a slow daily drift.

Data source: DexScreener quote for a deep WBTC/USDC pool (same API and rate
budget the scanner already uses), cached for BTC_IMPACT['cache_seconds'] so a
whole scan cycle costs one extra request at most.

Public surface:
  get_btc_context() -> dict | None
      {'price', 'change_m5', 'change_h1', 'change_h24',
       'regime_score', 'regime', 'fetched_at'}
"""

import logging
import time
from datetime import datetime, timezone

from crypto1k.config.scanner_config import BTC_IMPACT
from crypto1k.data import dexscreener

logger = logging.getLogger(__name__)

_cache = {'at': 0.0, 'ctx': None}


def get_btc_context():
    """Current BTC regime, cached. None when disabled or the fetch fails."""
    if not BTC_IMPACT.get('enabled'):
        return None
    now = time.monotonic()
    if _cache['ctx'] is not None and now - _cache['at'] < BTC_IMPACT.get('cache_seconds', 60):
        return _cache['ctx']

    chain, pool = BTC_IMPACT['pair']
    pair = dexscreener.get_pair(chain, pool)
    if not pair:
        logger.warning('BTC regime: pair fetch failed — proceeding without BTC context')
        return _cache['ctx']  # serve stale over nothing

    try:
        changes = pair.get('priceChange') or {}
        ctx = {
            'price':      float(pair.get('priceUsd')),
            'change_m5':  float(changes.get('m5') or 0),
            'change_h1':  float(changes.get('h1') or 0),
            'change_h24': float(changes.get('h24') or 0),
            'fetched_at': datetime.now(timezone.utc).isoformat(),
        }
    except (TypeError, ValueError) as e:
        logger.warning(f'BTC regime: bad pair payload: {e}')
        return _cache['ctx']

    ctx['regime_score'] = _score(ctx)
    ctx['regime'] = _label(ctx['regime_score'])
    _cache.update(at=now, ctx=ctx)
    return ctx


def _score(ctx: dict) -> float:
    w = BTC_IMPACT['score_weights']
    raw = (5.0
           + ctx['change_m5'] * w['m5']
           + ctx['change_h1'] * w['h1']
           + ctx['change_h24'] * w['h24'])
    return round(max(0.0, min(10.0, raw)), 1)


def _label(score: float) -> str:
    if score >= BTC_IMPACT['risk_on_score']:
        return 'risk-on'
    if score <= BTC_IMPACT['risk_off_score']:
        return 'risk-off'
    return 'neutral'
