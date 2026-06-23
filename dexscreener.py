"""
DexScreener API client.

Free, no API key. Docs: https://docs.dexscreener.com/api/reference

The public API returns *multi-window aggregates* per trading pair
(5m / 1h / 6h / 24h volume, price change, buy/sell counts, liquidity) rather
than raw OHLCV candles — which is exactly what a volume scanner needs.

Public surface:
  search_pairs(query)              -> list[raw pair]
  pairs_for_token(address, chain)  -> list[raw pair]
  resolve_symbol(symbol)           -> normalized pair (best match) or None
  normalize(raw_pair)              -> flat dict the scanner consumes
"""

import logging
import requests

from scanner_config import DEXSCREENER

logger = logging.getLogger(__name__)

_BASE = DEXSCREENER['base_url'].rstrip('/')
_TIMEOUT = DEXSCREENER['timeout']


# ── HTTP ────────────────────────────────────────────────────────────────────

def _get(path: str, params: dict = None):
    url = f'{_BASE}{path}'
    try:
        r = requests.get(url, params=params, timeout=_TIMEOUT,
                         headers={'Accept': 'application/json'})
        if r.status_code == 200:
            return r.json()
        logger.warning(f'DexScreener {path} -> HTTP {r.status_code}')
    except Exception as e:
        logger.warning(f'DexScreener {path} failed: {e}')
    return None


def search_pairs(query: str) -> list:
    """Full-text search across all DEX pairs (symbol, name, or address)."""
    data = _get('/latest/dex/search', {'q': query})
    return (data or {}).get('pairs') or []


def pairs_for_token(token_address: str, chain: str = None) -> list:
    """All pairs for a specific token contract address."""
    if chain:
        data = _get(f'/latest/dex/tokens/{token_address}')
    else:
        data = _get(f'/latest/dex/tokens/{token_address}')
    return (data or {}).get('pairs') or []


# ── Symbol resolution ───────────────────────────────────────────────────────

def _looks_like_address(s: str) -> bool:
    s = s.strip()
    # EVM (0x + 40 hex) or Solana-ish base58 (>= 32 chars, no separators)
    if s.startswith('0x') and len(s) == 42:
        return True
    if len(s) >= 32 and s.isalnum() and '-' not in s and '/' not in s:
        return True
    return False


def _pair_score(p: dict) -> float:
    """
    Rank candidate pairs to find the *canonical* one for a coin.

    A real, tradeable pair has BOTH meaningful liquidity AND real 24h volume.
    Scam/decoy pairs fake one but not the other (e.g. $9B fake liquidity, $0
    volume), so we rank by min(liquidity, volume) — which demands both — and
    apply a bonus for trusted quote tokens (USDC/WETH/SOL/…).
    """
    liq   = ((p.get('liquidity') or {}).get('usd')) or 0.0
    vol24 = ((p.get('volume') or {}).get('h24')) or 0.0
    quote = ((p.get('quoteToken') or {}).get('symbol') or '').upper()
    bonus = 1.0
    prefs = [q.upper() for q in DEXSCREENER['preferred_quotes']]
    if quote in prefs:
        # earlier in the preference list = bigger bonus
        bonus = 1.5 - (prefs.index(quote) * 0.03)
    return min(float(liq), float(vol24)) * bonus


def resolve_symbol(symbol: str):
    """
    Turn a user-typed symbol/address into the single best normalized pair.

    Strategy:
      - If it looks like a contract address, look up its pairs directly.
      - Otherwise, search and keep pairs whose BASE token symbol matches.
      - Filter by allowed chains (if configured), then pick deepest liquidity
        among preferred quote tokens.
    Returns a normalized dict (see normalize()) or None.
    """
    symbol = (symbol or '').strip()
    if not symbol:
        return None

    # Allow "CHAIN:ADDRESS" or plain address
    raw = symbol.split(':', 1)[-1] if ':' in symbol else symbol

    if _looks_like_address(raw):
        candidates = pairs_for_token(raw)
    else:
        # Match on the base token symbol, case-insensitive (strip common suffixes)
        base = symbol.upper().replace('-USDT', '').replace('/USDT', '') \
                             .replace('-USD', '').replace('USDT', '').strip()
        base = base or symbol.upper()
        results = search_pairs(base)
        candidates = [
            p for p in results
            if ((p.get('baseToken') or {}).get('symbol') or '').upper() == base
        ]
        # Fall back to anything returned if no exact base-symbol match
        if not candidates:
            candidates = results

    allowed = [c.lower() for c in DEXSCREENER['allowed_chains']]
    if allowed:
        candidates = [p for p in candidates if (p.get('chainId') or '').lower() in allowed]

    if not candidates:
        return None

    best = max(candidates, key=_pair_score)
    return normalize(best)


# ── Normalization ───────────────────────────────────────────────────────────

def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def normalize(p: dict) -> dict:
    """Flatten a raw DexScreener pair into the shape the scanner expects."""
    vol   = p.get('volume')     or {}
    chg   = p.get('priceChange') or {}
    txns  = p.get('txns')       or {}
    liq   = p.get('liquidity')  or {}
    base  = p.get('baseToken')  or {}
    quote = p.get('quoteToken') or {}

    def txn(window):
        t = txns.get(window) or {}
        return {'buys': int(t.get('buys') or 0), 'sells': int(t.get('sells') or 0)}

    return {
        'symbol':        (base.get('symbol') or '').upper(),
        'name':          base.get('name'),
        'chain':         p.get('chainId'),
        'dex':           p.get('dexId'),
        'pair_address':  p.get('pairAddress'),
        'token_address': base.get('address'),
        'quote_symbol':  (quote.get('symbol') or '').upper(),
        'url':           p.get('url'),
        'price_usd':     _num(p.get('priceUsd')),
        'liquidity_usd': _num(liq.get('usd')) or 0.0,
        'fdv':           _num(p.get('fdv')),
        'market_cap':    _num(p.get('marketCap')),
        'pair_created_at': p.get('pairCreatedAt'),   # ms epoch
        'volume': {
            'm5':  _num(vol.get('m5'))  or 0.0,
            'h1':  _num(vol.get('h1'))  or 0.0,
            'h6':  _num(vol.get('h6'))  or 0.0,
            'h24': _num(vol.get('h24')) or 0.0,
        },
        'price_change': {
            'm5':  _num(chg.get('m5'))  or 0.0,
            'h1':  _num(chg.get('h1'))  or 0.0,
            'h6':  _num(chg.get('h6'))  or 0.0,
            'h24': _num(chg.get('h24')) or 0.0,
        },
        'txns': {
            'm5':  txn('m5'),
            'h1':  txn('h1'),
            'h6':  txn('h6'),
            'h24': txn('h24'),
        },
    }
