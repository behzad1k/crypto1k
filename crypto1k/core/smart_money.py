"""
Smart money engine.

Two jobs, one data source (GeckoTerminal per-pool trades — free, keyless, and
every trade carries the wallet address that sent it):

  1. PER-COIN ENRICHMENT — during a scan, fetch whale-sized trades for the
     coin's pool, compute net whale flow over a lookback window, and check
     whether any *tracked smart wallet* bought. Produces extra signals, a
     score bonus, and (optionally) its own alert path.

  2. WALLET INTELLIGENCE — every whale trade is recorded; each BUY is scored
     1h/24h later from OHLCV candles (same pipeline the alert outcomes use).
     Wallets whose buys consistently precede pumps auto-qualify as smart
     money; the user can also track wallets manually. The /smart-money page
     shows the movements feed and the leaderboard.

Coverage: wallets are observed on the pools the scanner watches (watchlist +
alerted coins) — not globally across all chains. That's the honest limit of
staying on free keyless APIs.

Public surface:
  begin_cycle()                — reset the per-scan fetch budget
  enrich(result)               — add smart-money signals/score to a scan result
  maybe_backfill()             — throttled outcome backfill (monitor loop)
  backfill_trade_outcomes()    — score pending whale buys now
  feed() / leaderboard() / overview() / wallet_detail(address)
  add_wallet() / remove_wallet()
"""

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from crypto1k.core import db
from crypto1k.data import geckoterminal
from crypto1k.config.scanner_config import SMART_MONEY, SCORING

logger = logging.getLogger(__name__)


def _now() -> datetime:
    """Naive UTC now — matches GeckoTerminal timestamps and sqlite datetime()."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_ts(iso: str):
    try:
        return datetime.fromisoformat(iso.replace(' ', 'T'))
    except Exception:
        return None


def _short(addr: str) -> str:
    return f"{addr[:6]}…{addr[-4:]}" if addr and len(addr) > 12 else (addr or '?')


# ── Tracked wallet set (manual + auto-qualified), cached briefly ─────────────

_tracked_cache = {'at': 0.0, 'wallets': {}}
_TRACKED_TTL = 120  # seconds


def tracked_wallets() -> dict:
    now = time.monotonic()
    if now - _tracked_cache['at'] > _TRACKED_TTL:
        try:
            _tracked_cache['wallets'] = db.get_tracked_wallets(SMART_MONEY['qualify'])
            _tracked_cache['at'] = now
        except Exception as e:
            logger.warning(f'tracked_wallets refresh failed: {e}')
    return _tracked_cache['wallets']


# ── Per-coin enrichment ──────────────────────────────────────────────────────

# Per-pool metrics cache + a fetch budget per scan cycle, so enrichment can
# never blow GeckoTerminal's ~30 req/min free tier no matter the watchlist size.
_pool_cache = {}          # pool_address -> (monotonic_ts, metrics dict)
_fetches_this_cycle = 0


def begin_cycle():
    """Call at the start of each scan cycle to reset the fetch budget."""
    global _fetches_this_cycle
    _fetches_this_cycle = 0


def _pool_metrics(chain: str, pool: str, symbol: str):
    """Whale-flow metrics for one pool, cached for cache_ttl_seconds."""
    global _fetches_this_cycle
    cached = _pool_cache.get(pool)
    if cached and time.monotonic() - cached[0] < SMART_MONEY['cache_ttl_seconds']:
        return cached[1]
    if _fetches_this_cycle >= SMART_MONEY['max_fetch_per_cycle']:
        return cached[1] if cached else None  # stale beats nothing

    _fetches_this_cycle += 1
    trades = geckoterminal.fetch_trades(pool, chain, SMART_MONEY['min_trade_usd'])
    time.sleep(SMART_MONEY['per_request_pause'])
    if not trades:
        _pool_cache[pool] = (time.monotonic(), None)
        return None

    try:
        db.record_wallet_trades(trades, chain, pool, symbol)
    except Exception as e:
        logger.warning(f'record_wallet_trades failed for {symbol}: {e}')

    cutoff = _now() - timedelta(minutes=SMART_MONEY['lookback_minutes'])
    smart = tracked_wallets()

    buy_usd = sell_usd = 0.0
    buyers, sellers = set(), set()
    biggest_buy = biggest_sell = 0.0
    smart_hits = []
    for t in trades:
        ts = _parse_ts(t['block_ts'])
        if ts is None or ts < cutoff:
            continue
        usd = t['amount_usd'] or 0.0
        if t['side'] == 'buy':
            buy_usd += usd
            buyers.add(t['wallet'])
            biggest_buy = max(biggest_buy, usd)
            if t['wallet'] in smart:
                info = smart[t['wallet']]
                smart_hits.append({
                    'wallet':  t['wallet'],
                    'label':   info.get('label'),
                    'source':  info.get('source'),
                    'usd':     usd,
                    'minutes_ago': round((_now() - ts).total_seconds() / 60),
                })
        else:
            sell_usd += usd
            sellers.add(t['wallet'])
            biggest_sell = max(biggest_sell, usd)

    metrics = {
        'buy_usd':          round(buy_usd),
        'sell_usd':         round(sell_usd),
        'net_flow_usd':     round(buy_usd - sell_usd),
        'buyers':           len(buyers),
        'sellers':          len(sellers),
        'biggest_buy_usd':  round(biggest_buy),
        'biggest_sell_usd': round(biggest_sell),
        'smart_buys':       len(smart_hits),
        'smart_buy_usd':    round(sum(h['usd'] for h in smart_hits)),
        'smart_wallets':    sorted(smart_hits, key=lambda h: -h['usd'])[:5],
        'lookback_minutes': SMART_MONEY['lookback_minutes'],
        'min_trade_usd':    SMART_MONEY['min_trade_usd'],
    }
    _pool_cache[pool] = (time.monotonic(), metrics)
    return metrics


def _sig(name, direction, detail, value):
    return {'name': name, 'category': 'smart_money', 'direction': direction,
            'detail': detail, 'value': value}


def enrich(result: dict) -> dict:
    """
    Add whale/smart-wallet signals and a score bonus to one scan result.
    Mutates and returns `result`. The caller (scanner) rebuilds the summary
    and label afterward — this module deliberately doesn't import scanner.
    """
    if not SMART_MONEY.get('enabled') or result.get('error'):
        return result
    if not result.get('passes_filters'):
        return result
    if result.get('score', 0) < SMART_MONEY['min_score_to_fetch']:
        return result
    pool, chain = result.get('pair_address'), result.get('chain')
    if not pool or not chain:
        return result

    try:
        m = _pool_metrics(chain, pool, result.get('symbol'))
    except Exception as e:
        logger.warning(f"smart-money enrich failed for {result.get('symbol')}: {e}")
        return result
    if not m:
        return result

    result['smart_money'] = m
    signals = result['signals']
    bonus = 0.0
    lb = m['lookback_minutes']

    # Tracked smart wallet bought — the headline signal.
    if m['smart_buys']:
        top = m['smart_wallets'][0]
        who = top.get('label') or _short(top['wallet'])
        detail = (f"🧠 {m['smart_buys']} tracked smart wallet(s) bought "
                  f"${m['smart_buy_usd']:,.0f} in the last {lb}m — "
                  f"latest: {who} (${top['usd']:,.0f}, {top['minutes_ago']}m ago)")
        signals.append(_sig('smart_wallet_buy', 'bullish', detail, m['smart_buy_usd']))
        bonus += SMART_MONEY['pts_smart_wallet_buy']

    # Whale net flow over the lookback window.
    net = m['net_flow_usd']
    if abs(net) >= SMART_MONEY['net_flow_notable_usd']:
        direction = 'bullish' if net > 0 else 'bearish'
        strength = ('strong' if abs(net) >= SMART_MONEY['net_flow_strong_usd']
                    else 'notable')
        signals.append(_sig('whale_net_flow', direction,
            f"whale net flow {net:+,.0f} USD over {lb}m "
            f"({m['buyers']} buyers / {m['sellers']} sellers "
            f"≥ ${m['min_trade_usd']:,.0f}) — {strength}", net))
        if net > 0:
            bonus += (SMART_MONEY['pts_net_flow_strong'] if strength == 'strong'
                      else SMART_MONEY['pts_net_flow_notable'])

    # A single outsized print.
    if m['biggest_buy_usd'] >= SMART_MONEY['big_trade_usd']:
        signals.append(_sig('whale_print', 'bullish',
            f"single whale buy of ${m['biggest_buy_usd']:,.0f} in the last {lb}m",
            m['biggest_buy_usd']))
        bonus += SMART_MONEY['pts_big_trade']
    elif m['biggest_sell_usd'] >= SMART_MONEY['big_trade_usd']:
        signals.append(_sig('whale_print', 'bearish',
            f"single whale sell of ${m['biggest_sell_usd']:,.0f} in the last {lb}m",
            m['biggest_sell_usd']))

    # Score bonus (never lets a wash/thin-exit suspect climb past its cap).
    bonus = min(bonus, SCORING['max_smart_money'])
    if bonus:
        capped = result.get('wash_warning') or result.get('thin_exit_warning')
        new_score = min(result['score'] + bonus, 10.0)
        if capped:
            new_score = min(new_score, SCORING['label_good'])
        result['score'] = round(new_score, 1)
        result['smart_money']['score_bonus'] = round(bonus, 2)

    # Direction: recount with the new signals included.
    bull = sum(1 for s in signals if s['direction'] == 'bullish')
    bear = sum(1 for s in signals if s['direction'] == 'bearish')
    result['direction'] = ('bullish' if bull > bear
                           else 'bearish' if bear > bull else 'neutral')

    # Third alert path: a tracked winner just bought this coin.
    if (SMART_MONEY.get('alert_on_smart_wallet_buy')
            and m['smart_buys']
            and not result.get('wash_warning')
            and not result.get('thin_exit_warning')
            and result['direction'] != 'bearish'
            and not result.get('is_alert')):
        result['is_alert'] = True
        result['alert_path'] = 'smart'

    return result


# ── Outcome backfill (scores whale buys 1h/24h later) ───────────────────────

def _closest_close(df: pd.DataFrame, target: datetime, tolerance_minutes: float):
    if df is None or df.empty:
        return None
    diffs = (df['timestamp'] - pd.Timestamp(target)).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > pd.Timedelta(minutes=tolerance_minutes):
        return None
    return float(df.loc[idx, 'close'])


def backfill_trade_outcomes(max_pools: int = None) -> dict:
    """
    Score pending whale buys. Trades are grouped by pool so ONE candle fetch
    scores every pending buy on that pool — the request budget is per pool,
    not per trade.
    """
    max_pools = max_pools or SMART_MONEY['backfill_max_pools']
    pending = db.get_trades_needing_outcome(SMART_MONEY['min_trade_age_minutes'])
    if not pending:
        return {'pools': 0, 'scored': 0, 'gave_up': 0}

    by_pool = {}
    for t in pending:
        by_pool.setdefault((t['chain'], t['pool_address']), []).append(t)

    now = _now()
    give_up = timedelta(hours=SMART_MONEY['give_up_after_hours'])
    scored = gave_up = pools_done = 0

    for (chain, pool), trades in list(by_pool.items())[:max_pools]:
        pools_done += 1
        df = geckoterminal.fetch_ohlcv(pool, chain, '5m', limit=1000)
        time.sleep(SMART_MONEY['per_request_pause'])

        for t in trades:
            ts = _parse_ts(t['block_ts'])
            if ts is None:
                continue
            age = now - ts

            if df is None or df.empty:
                # Pool has no candles (delisted/dead). Retry until give-up.
                if age >= give_up:
                    db.upsert_wallet_trade_outcome(t['id'], {'data_complete': 1})
                    gave_up += 1
                continue

            price_at = t.get('price_usd') or _closest_close(df, ts, 30)
            price_1h = _closest_close(df, ts + timedelta(hours=1), 30)
            price_24h = (_closest_close(df, ts + timedelta(hours=24), 240)
                         if age >= timedelta(hours=24) else None)

            fields = {
                'price_at_trade': price_at,
                'price_1h':       price_1h,
                'price_24h':      price_24h,
                'ret_1h_pct':  (round((price_1h - price_at) / price_at * 100, 2)
                                if price_at and price_1h else None),
                'ret_24h_pct': (round((price_24h - price_at) / price_at * 100, 2)
                                if price_at and price_24h else None),
                'data_complete': int(
                    (age >= timedelta(hours=24) and price_24h is not None)
                    or age >= give_up
                ),
            }
            db.upsert_wallet_trade_outcome(t['id'], fields)
            if fields['ret_1h_pct'] is not None:
                scored += 1

    logger.info(f'Smart-money backfill: {pools_done} pool(s), '
                f'{scored} buy(s) scored, {gave_up} given up')
    return {'pools': pools_done, 'scored': scored, 'gave_up': gave_up}


def maybe_backfill():
    """Throttled backfill for the monitor loop — cheap no-op between runs."""
    if not SMART_MONEY.get('enabled'):
        return
    last = db.get_state('smart_money_last_backfill')
    if last:
        try:
            elapsed = (_now() - datetime.fromisoformat(last)).total_seconds() / 60
            if elapsed < SMART_MONEY['backfill_every_minutes']:
                return
        except Exception:
            pass
    db.set_state('smart_money_last_backfill', _now().isoformat())
    try:
        backfill_trade_outcomes()
    except Exception as e:
        logger.warning(f'Smart-money backfill failed: {e}')


# ── Web-facing wrappers ──────────────────────────────────────────────────────

def feed(limit: int = None) -> list:
    return db.smart_money_feed(SMART_MONEY['qualify'],
                               limit or SMART_MONEY['feed_limit'])


def leaderboard(min_buys: int = 2, limit: int = 200) -> list:
    return db.wallet_leaderboard(SMART_MONEY['qualify'], min_buys, limit)


def overview() -> dict:
    d = db.smart_money_overview(SMART_MONEY['qualify'])
    d['qualify'] = SMART_MONEY['qualify']
    d['min_trade_usd'] = SMART_MONEY['min_trade_usd']
    return d


def wallet_detail(address: str) -> dict:
    trades = db.get_wallet_trades(address)
    q = SMART_MONEY['qualify']
    buys = [t for t in trades if t['side'] == 'buy' and t['ret_1h_pct'] is not None]
    wins = [t for t in buys
            if t['ret_1h_pct'] >= q['win_ret_1h_pct']
            or (t['ret_24h_pct'] is not None and t['ret_24h_pct'] >= q['win_ret_24h_pct'])]
    return {
        'address':     address,
        'trades':      trades,
        'scored_buys': len(buys),
        'wins':        len(wins),
        'win_rate':    round(len(wins) / len(buys), 3) if buys else None,
        'tracked':     address in tracked_wallets(),
    }


def add_wallet(address: str, chain: str = '', label: str = None):
    db.add_smart_wallet(address, chain, label, source='manual')
    _tracked_cache['at'] = 0.0  # force refresh


def remove_wallet(address: str, chain: str = ''):
    db.remove_smart_wallet(address, chain)
    _tracked_cache['at'] = 0.0
