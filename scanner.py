"""
Shitcoin volume scanner.

Evaluates a DexScreener pair against a volume / momentum / buy-pressure model
derived from the pair's multi-window aggregates, produces a 0–10 validity score
(same spirit as the setup score in app.py), decides whether it warrants an
alert, and writes a human-readable description for the email.

All thresholds come from scanner_config.py — nothing is hard-coded here.

Public surface:
  evaluate(pair)            -> result dict (signals, score, alert flag, summary)
  scan_symbols(symbols)     -> [result, ...] sorted best-first
  scan_and_alert(symbols)   -> scans, sends emails for new alerts (respects cooldown)
  start_monitor() / stop_monitor() / monitor_status()
"""

import time
import logging
import threading
from datetime import datetime, timezone

import dexscreener
import emailer
import telegram_notify
import db
from scanner_config import FILTERS, THRESHOLDS, SCORING, ALERTING, MONITOR

logger = logging.getLogger(__name__)


# ── Small helpers ───────────────────────────────────────────────────────────

def _pace(window_vol: float, vol_24h: float, slices: float) -> float:
    """
    Volume "pace ratio": how hot a recent window is vs. its 24h average.
      slices = how many of these windows fit in 24h (24 for 1h, 288 for 5m).
    Returns 0 when there's no baseline.
    """
    avg = (vol_24h or 0.0) / slices
    if avg <= 0:
        return 0.0
    return round((window_vol or 0.0) / avg, 2)


def _age_hours(pair_created_at):
    if not pair_created_at:
        return None
    try:
        created = datetime.fromtimestamp(pair_created_at / 1000, tz=timezone.utc)
        return (datetime.now(timezone.utc) - created).total_seconds() / 3600
    except Exception:
        return None


def _fmt_price(p) -> str:
    """Format a USD price readably across the huge range of token prices."""
    if p is None:
        return "n/a"
    if p >= 1:
        return f"${p:,.2f}"
    if p >= 0.01:
        return f"${p:.4f}"
    return f"${p:.8f}".rstrip('0').rstrip('.')  # tiny prices: trim trailing zeros


def _label(score: float) -> str:
    if score >= SCORING['label_strong']:
        return 'STRONG'
    if score >= SCORING['label_good']:
        return 'GOOD'
    if score >= SCORING['label_fair']:
        return 'FAIR'
    return 'WEAK'


# ── Core evaluation ─────────────────────────────────────────────────────────

def evaluate(pair: dict) -> dict:
    """
    Score one normalized DexScreener pair. Returns a result dict even when the
    coin fails filters (passes_filters=False) so the UI can show why.
    """
    vol  = pair['volume']
    chg  = pair['price_change']
    txns = pair['txns']
    liq  = pair['liquidity_usd']

    # ── Derived metrics ──────────────────────────────────────────────────────
    vol_pace_1h = _pace(vol['h1'], vol['h24'], 24)
    vol_pace_5m = _pace(vol['m5'], vol['h24'], 288)
    txns_1h     = txns['h1']['buys'] + txns['h1']['sells']
    buy_ratio   = round(txns['h1']['buys'] / max(txns['h1']['sells'], 1), 2)
    turnover    = round((vol['h24'] or 0.0) / liq, 1) if liq > 0 else None
    age_h       = _age_hours(pair['pair_created_at'])

    metrics = {
        'vol_pace_1h':  vol_pace_1h,
        'vol_pace_5m':  vol_pace_5m,
        'txns_1h':      txns_1h,
        'buy_ratio_1h': buy_ratio,
        'turnover':     turnover,
        'age_hours':    round(age_h, 1) if age_h is not None else None,
    }

    # ── Filters (gates) ──────────────────────────────────────────────────────
    filter_fails = []
    if liq < FILTERS['min_liquidity_usd']:
        filter_fails.append(f"liquidity ${liq:,.0f} < ${FILTERS['min_liquidity_usd']:,.0f}")
    if vol['h24'] < FILTERS['min_volume_24h_usd']:
        filter_fails.append(f"24h vol ${vol['h24']:,.0f} < ${FILTERS['min_volume_24h_usd']:,.0f}")
    if txns_1h < FILTERS['min_txns_1h']:
        filter_fails.append(f"{txns_1h} txns/1h < {FILTERS['min_txns_1h']}")
    if age_h is not None and age_h < FILTERS['min_pair_age_hours']:
        filter_fails.append(f"pair age {age_h:.1f}h < {FILTERS['min_pair_age_hours']}h")
    # Data-sanity: reject corrupted price feeds (e.g. "+501,497% in 1h" from a
    # broken thin pool) so they never score or alert.
    max_chg = FILTERS.get('max_price_change_1h_pct', 5_000.0)
    if abs(chg['h1']) > max_chg:
        filter_fails.append(f"1h change {chg['h1']:+,.0f}% looks corrupted (> {max_chg:,.0f}%)")
    passes_filters = not filter_fails

    wash_warning = (turnover is not None and turnover >= FILTERS['wash_turnover_ratio'])

    # ── Signals + scoring ────────────────────────────────────────────────────
    signals = []
    score = 0.0

    # Price moves drive *direction*. Volume is directionless — a surge confirms
    # whichever way price is moving, so volume signals inherit the price
    # direction of their window (1h for the surge, 5m for the acceleration).
    # Without this, a heavy sell-off reads as "bullish" purely on volume.
    pc1, pc5m, pc6 = chg['h1'], chg['m5'], chg['h6']

    def _dir(x):
        return 'bullish' if x > 0 else 'bearish' if x < 0 else 'neutral'

    mom_dir = _dir(pc1)
    vol_dir_1h = _dir(pc1)
    vol_dir_5m = _dir(pc5m)

    # (1) Volume surge — headline factor (direction follows price)
    vp = max(vol_pace_1h, vol_pace_5m)
    if vol_pace_1h >= THRESHOLDS['vol_pace_1h_strong']:
        pts = SCORING['max_volume_surge']
        signals.append(_sig('volume_surge', 'volume', vol_dir_1h,
            f"1h volume {vol_pace_1h}× its 24h average — strong surge on a {vol_dir_1h} move", vol_pace_1h))
    elif vol_pace_1h >= THRESHOLDS['vol_pace_1h_notable']:
        pts = SCORING['max_volume_surge'] * 0.6
        signals.append(_sig('volume_surge', 'volume', vol_dir_1h,
            f"1h volume {vol_pace_1h}× its 24h average — notable surge on a {vol_dir_1h} move", vol_pace_1h))
    else:
        pts = SCORING['max_volume_surge'] * 0.15 * min(vol_pace_1h, 2.0)
    # 5-minute acceleration bonus (within the headline budget)
    if vol_pace_5m >= THRESHOLDS['vol_pace_5m_strong']:
        signals.append(_sig('volume_acceleration', 'volume', vol_dir_5m,
            f"5m volume {vol_pace_5m}× average — accelerating right now ({vol_dir_5m})", vol_pace_5m))
        pts = SCORING['max_volume_surge']
    elif vol_pace_5m >= THRESHOLDS['vol_pace_5m_notable']:
        signals.append(_sig('volume_acceleration', 'volume', vol_dir_5m,
            f"5m volume {vol_pace_5m}× average — picking up ({vol_dir_5m})", vol_pace_5m))
    score += min(pts, SCORING['max_volume_surge'])

    # (2) Momentum — price move + multi-window alignment
    aligned = (pc5m > 0 and pc1 > 0 and pc6 > 0) or (pc5m < 0 and pc1 < 0 and pc6 < 0)
    mpts = 0.0
    if abs(pc1) >= THRESHOLDS['price_move_1h_strong']:
        mpts += SCORING['max_momentum'] * 0.6
        signals.append(_sig('price_momentum', 'momentum', mom_dir,
            f"price {pc1:+.1f}% in 1h — strong move", pc1))
    elif abs(pc1) >= THRESHOLDS['price_move_1h_notable']:
        mpts += SCORING['max_momentum'] * 0.35
        signals.append(_sig('price_momentum', 'momentum', mom_dir,
            f"price {pc1:+.1f}% in 1h", pc1))
    if aligned:
        mpts += SCORING['max_momentum'] * 0.4
        signals.append(_sig('momentum_alignment', 'trend', mom_dir,
            f"5m/1h/6h all {mom_dir} — aligned momentum", None))
    score += min(mpts, SCORING['max_momentum'])

    # (3) Buy pressure
    bpts = 0.0
    if buy_ratio >= THRESHOLDS['buy_ratio_strong']:
        bpts = SCORING['max_buy_pressure']
        signals.append(_sig('buy_pressure', 'volume', 'bullish',
            f"{txns['h1']['buys']} buys vs {txns['h1']['sells']} sells ({buy_ratio}:1) — strong buying",
            buy_ratio))
    elif buy_ratio >= THRESHOLDS['buy_ratio_notable']:
        bpts = SCORING['max_buy_pressure'] * 0.5
        signals.append(_sig('buy_pressure', 'volume', 'bullish',
            f"{txns['h1']['buys']} buys vs {txns['h1']['sells']} sells ({buy_ratio}:1) — buyers leading",
            buy_ratio))
    elif buy_ratio > 0 and buy_ratio <= (1 / THRESHOLDS['buy_ratio_notable']):
        signals.append(_sig('sell_pressure', 'volume', 'bearish',
            f"{txns['h1']['sells']} sells vs {txns['h1']['buys']} buys — sellers leading", buy_ratio))
    score += bpts

    # (4) Activity / liquidity health
    apts = 0.0
    if liq >= FILTERS['min_liquidity_usd'] * 3 and txns_1h >= FILTERS['min_txns_1h'] * 3:
        apts = SCORING['max_activity']
    elif passes_filters:
        apts = SCORING['max_activity'] * 0.5
    score += apts

    # Wash-trade caution caps the score
    if wash_warning:
        score = min(score, SCORING['label_good'])

    score = round(min(score, 10.0), 1)

    # ── Net direction ────────────────────────────────────────────────────────
    bull = sum(1 for s in signals if s['direction'] == 'bullish')
    bear = sum(1 for s in signals if s['direction'] == 'bearish')
    direction = 'bullish' if bull > bear else 'bearish' if bear > bull else 'neutral'

    # ── Alert decision (lenient by config) ───────────────────────────────────
    is_alert = (
        passes_filters
        and score >= ALERTING['min_validity_score']
        and vol_pace_1h >= ALERTING['min_vol_pace_1h']
        and (not ALERTING['require_bullish'] or direction == 'bullish')
    )

    result = {
        'symbol':         pair['symbol'],
        'name':           pair['name'],
        'chain':          pair['chain'],
        'dex':            pair['dex'],
        'quote_symbol':   pair['quote_symbol'],
        'url':            pair['url'],
        'price_usd':      pair['price_usd'],
        'liquidity_usd':  liq,
        'market_cap':     pair['market_cap'],
        'fdv':            pair['fdv'],
        'volume':         pair['volume'],
        'price_change':   pair['price_change'],
        'txns':           pair['txns'],
        'metrics':        metrics,
        'signals':        signals,
        'score':          score,
        'label':          _label(score),
        'direction':      direction,
        'passes_filters': passes_filters,
        'filter_fails':   filter_fails,
        'wash_warning':   wash_warning,
        'is_alert':       is_alert,
        'scanned_at':     datetime.now(timezone.utc).isoformat(),
    }
    result['summary'] = build_summary(result)
    return result


def _sig(name, category, direction, detail, value):
    return {'name': name, 'category': category, 'direction': direction,
            'detail': detail, 'value': value}


# ── Human-readable description (used in the email + UI) ──────────────────────

def build_summary(r: dict) -> str:
    """A plain-English paragraph describing what the scanner sees."""
    m = r['metrics']
    sym = r['symbol']
    parts = []

    signal_label = {
        'bullish': '🟢 BUY SIGNAL',
        'bearish': '🔴 SELL SIGNAL',
    }.get(r['direction'], '⚪ NEUTRAL — no clear direction')
    parts.append(
        f"{signal_label} — {sym} @ {_fmt_price(r.get('price_usd'))}: "
        f"setup quality {r['label']} ({r['score']}/10), leaning {r['direction']}."
    )

    if m['vol_pace_1h'] >= THRESHOLDS['vol_pace_1h_notable']:
        parts.append(
            f"Volume in the last hour (${r['volume']['h1']:,.0f}) is running "
            f"{m['vol_pace_1h']}× its 24h average pace"
            + (f", and the last 5 minutes are {m['vol_pace_5m']}× — actively accelerating."
               if m['vol_pace_5m'] >= THRESHOLDS['vol_pace_5m_notable'] else ".")
        )
    else:
        parts.append(f"1h volume is ${r['volume']['h1']:,.0f} ({m['vol_pace_1h']}× average pace).")

    pc = r['price_change']
    parts.append(
        f"Price is {pc['h1']:+.1f}% over 1h ({pc['m5']:+.1f}% in the last 5m, {pc['h6']:+.1f}% over 6h)."
    )

    t1 = r['txns']['h1']
    if m['buy_ratio_1h'] >= THRESHOLDS['buy_ratio_notable']:
        parts.append(f"Buyers dominate: {t1['buys']} buys vs {t1['sells']} sells ({m['buy_ratio_1h']}:1).")
    elif m['buy_ratio_1h'] > 0 and m['buy_ratio_1h'] <= (1 / THRESHOLDS['buy_ratio_notable']):
        parts.append(f"Sellers dominate: {t1['sells']} sells vs {t1['buys']} buys.")
    else:
        parts.append(f"Order flow balanced: {t1['buys']} buys / {t1['sells']} sells.")

    parts.append(f"Liquidity ${r['liquidity_usd']:,.0f}"
                 + (f", market cap ${r['market_cap']:,.0f}" if r['market_cap'] else "")
                 + f" on {r['dex']} / {r['chain']}.")

    if r['wash_warning']:
        parts.append("⚠️ Very high volume-to-liquidity ratio — possible wash trading, treat with caution.")

    return ' '.join(parts)


# ── Scanning many symbols ───────────────────────────────────────────────────

def scan_symbols(symbols: list) -> list:
    """Resolve + evaluate each symbol. Returns results sorted best-first."""
    results = []
    for sym in symbols:
        try:
            pair = dexscreener.resolve_symbol(sym)
            if pair is None:
                results.append({'symbol': sym.upper(), 'error': 'not found on DexScreener',
                                'score': 0, 'is_alert': False, 'passes_filters': False})
            else:
                results.append(evaluate(pair))
        except Exception as e:
            logger.warning(f'Scan failed for {sym}: {e}')
            results.append({'symbol': str(sym).upper(), 'error': str(e),
                            'score': 0, 'is_alert': False, 'passes_filters': False})
        time.sleep(MONITOR['per_request_pause'])
    results.sort(key=lambda r: r.get('score', 0), reverse=True)
    # Persist the most recent scan (manual or automated) to shared state so the
    # live view is consistent across all workers. The monitor loop overwrites
    # this with a richer summary (alert/email counts) right after.
    try:
        db.set_monitor_last_run(
            datetime.now(timezone.utc).isoformat(),
            {'scanned': len(results)},
            results,
        )
    except Exception as e:
        logger.warning(f'Could not persist last scan: {e}')
    return results


def scan_and_alert(symbols: list = None) -> dict:
    """
    Scan the watchlist (or the given symbols), then email any coin that is an
    alert AND is not still in its cooldown window. Records every alert in the DB.
    Returns a summary dict.
    """
    if symbols is None:
        symbols = db.get_watchlist()
    if not symbols:
        return {'scanned': 0, 'alerts': 0, 'emailed': 0, 'results': []}

    results = scan_symbols(symbols)
    emailed = 0
    telegrammed = 0
    fired = []

    for r in results:
        if not r.get('is_alert'):
            continue
        if db.in_cooldown(r['symbol'], ALERTING['cooldown_minutes']):
            logger.info(f"{r['symbol']} alert suppressed (cooldown)")
            continue
        db.record_alert(r)
        try:
            if emailer.send_alert(r):
                emailed += 1
        except Exception as e:
            logger.warning(f"Email failed for {r['symbol']}: {e}")
        try:
            if telegram_notify.send_alert(r):
                telegrammed += 1
        except Exception as e:
            logger.warning(f"Telegram failed for {r['symbol']}: {e}")
        fired.append(r['symbol'])

    return {
        'scanned': len(results),
        'alerts':  len(fired),
        'emailed': emailed,
        'telegrammed': telegrammed,
        'fired':   fired,
        'results': results,
    }


# ── Background monitor ──────────────────────────────────────────────────────

# The monitor's on/off flag and last run live in the DB (db.scanner_state) so
# every gunicorn worker sees the same state. A single always-on *supervisor*
# thread runs in just one worker (the boot file-lock holder in app.py); each
# cycle it checks the shared flag and scans only when enabled. start/stop just
# flip the shared flag, so they work no matter which worker handles the request.
_supervisor_thread = None
_supervisor_lock = threading.Lock()


def _monitor_loop():
    logger.info('Scanner monitor supervisor started')
    last_scan_at = 0.0  # monotonic; 0 = scan immediately once enabled
    while True:
        try:
            if db.is_monitor_enabled():
                now = time.monotonic()
                if now - last_scan_at >= MONITOR['interval_seconds']:
                    last_scan_at = now
                    summary = scan_and_alert()
                    db.set_monitor_last_run(
                        datetime.now(timezone.utc).isoformat(),
                        {k: summary[k] for k in
                         ('scanned', 'alerts', 'emailed', 'telegrammed', 'fired')
                         if k in summary},
                        summary.get('results', []),
                    )
                    if summary['alerts']:
                        logger.info(f"Monitor: {summary['alerts']} alert(s), "
                                    f"{summary['emailed']} email(s)")
            else:
                last_scan_at = 0.0  # re-enable should scan right away
        except Exception as e:
            logger.warning(f'Monitor cycle error: {e}')
        # Poll the shared flag frequently so start/stop feel responsive.
        time.sleep(3)


def start_supervisor() -> bool:
    """Start the single always-on supervisor thread (boot, in the lock owner)."""
    global _supervisor_thread
    with _supervisor_lock:
        if _supervisor_thread and _supervisor_thread.is_alive():
            return False
        _supervisor_thread = threading.Thread(target=_monitor_loop, daemon=True)
        _supervisor_thread.start()
        return True


def start_monitor() -> bool:
    """Turn the monitor on (shared flag). Returns False if already running."""
    if db.is_monitor_enabled():
        return False
    db.set_monitor_enabled(True)
    return True


def stop_monitor() -> bool:
    """Turn the monitor off (shared flag). Returns False if already stopped."""
    if not db.is_monitor_enabled():
        return False
    db.set_monitor_enabled(False)
    return True


def monitor_status() -> dict:
    lr = db.get_monitor_last_run()
    return {
        'running':          db.is_monitor_enabled(),
        'interval_seconds': MONITOR['interval_seconds'],
        'last_run':         lr['at'],
        'last_summary':     lr['summary'],
        'watchlist_count':  len(db.get_watchlist()),
        'email_ready':      emailer.is_configured(),
        'recipients':       emailer.recipient_count(),
    }


def last_scan() -> dict:
    """Most recent scan results (from the monitor) for the live view."""
    lr = db.get_monitor_last_run()
    return {
        'running':          db.is_monitor_enabled(),
        'interval_seconds': MONITOR['interval_seconds'],
        'last_run':         lr['at'],
        'last_summary':     lr['summary'],
        'results':          lr['results'],
    }
