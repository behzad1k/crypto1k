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

from crypto1k.core import db, digest, smart_money
from crypto1k.data import btc_market, dexscreener
from crypto1k.notify import emailer, telegram_notify
from crypto1k.config.scanner_config import (
    FILTERS, THRESHOLDS, SCORING, ALERTING, MONITOR, BTC_IMPACT,
    EXTENSION, EXIT_PLAN,
)

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

    # Market-structure ratios (mcap falls back to FDV — micro-caps often only
    # report one of the two on DexScreener).
    mcap = pair['market_cap'] or pair['fdv']
    vol_mcap_24h = round((vol['h24'] or 0.0) / mcap, 2) if mcap else None
    liq_mcap_pct = round(liq / mcap * 100, 2) if mcap else None

    metrics = {
        'vol_pace_1h':  vol_pace_1h,
        'vol_pace_5m':  vol_pace_5m,
        'txns_1h':      txns_1h,
        'buy_ratio_1h': buy_ratio,
        'turnover':     turnover,
        'vol_mcap_24h': vol_mcap_24h,
        'liq_mcap_pct': liq_mcap_pct,
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
    # Exit-door check: liquidity too shallow relative to the market cap means
    # nobody can actually realize the valuation — sells crater the price.
    thin_exit_warning = (liq_mcap_pct is not None
                         and liq_mcap_pct < FILTERS['min_liq_to_mcap_pct'])

    # ── Signals + scoring ────────────────────────────────────────────────────
    #
    # Weights were rebalanced on 2026-07-26 against 810 alerts with complete
    # 24h outcomes. The short version: volume pace is the only factor whose
    # relationship to forward return survived a temporal train/test split, and
    # price momentum was actively harmful — so volume now carries half the
    # score and momentum an eighth of it. See scanner_config.SCORING.
    #
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

    # (1) Volume surge — now graded up to an "extreme" tier. The 10×+ bucket
    # returned +11.9% / +10.9% at 1h across the two halves of the split (60%
    # and 71% win) and is where nearly a third of alerts touch +10% inside the
    # hour, so it earns the full budget rather than sharing a tier with 5×.
    if vol_pace_1h >= THRESHOLDS['vol_pace_1h_extreme']:
        pts = SCORING['max_volume_surge']
        signals.append(_sig('volume_surge', 'volume', vol_dir_1h,
            f"1h volume {vol_pace_1h}× its 24h average — extreme surge on a {vol_dir_1h} move", vol_pace_1h))
    elif vol_pace_1h >= THRESHOLDS['vol_pace_1h_strong']:
        pts = SCORING['max_volume_surge'] * 0.85
        signals.append(_sig('volume_surge', 'volume', vol_dir_1h,
            f"1h volume {vol_pace_1h}× its 24h average — strong surge on a {vol_dir_1h} move", vol_pace_1h))
    elif vol_pace_1h >= THRESHOLDS['vol_pace_1h_notable']:
        pts = SCORING['max_volume_surge'] * 0.5
        signals.append(_sig('volume_surge', 'volume', vol_dir_1h,
            f"1h volume {vol_pace_1h}× its 24h average — notable surge on a {vol_dir_1h} move", vol_pace_1h))
    else:
        pts = SCORING['max_volume_surge'] * 0.12 * min(vol_pace_1h, 2.0)
    # 5-minute acceleration (within the headline budget). Only the 12×+ band
    # separated in both halves; 6–12× was flat, so it no longer maxes the tier.
    if vol_pace_5m >= THRESHOLDS['vol_pace_5m_strong']:
        signals.append(_sig('volume_acceleration', 'volume', vol_dir_5m,
            f"5m volume {vol_pace_5m}× average — accelerating hard right now ({vol_dir_5m})", vol_pace_5m))
        pts = max(pts, SCORING['max_volume_surge'] * 0.8)
    elif vol_pace_5m >= THRESHOLDS['vol_pace_5m_notable']:
        signals.append(_sig('volume_acceleration', 'volume', vol_dir_5m,
            f"5m volume {vol_pace_5m}× average — picking up ({vol_dir_5m})", vol_pace_5m))
        pts = max(pts, SCORING['max_volume_surge'] * 0.45)
    score += min(pts, SCORING['max_volume_surge'])

    # (2) Momentum — deliberately small, and now *inverted* at the top end. A
    # move that is merely underway is mildly useful; one that has already gone
    # far is a warning. 24h mean return by 1h price change at alert time ran
    # -1.4% (0–3%), -6.3% (3–8%), -6.8% (8–15%), -16.4% (15–30%), -28.5% (30%+),
    # so the "strong move" tier now scores LESS than the notable one.
    aligned = (pc5m > 0 and pc1 > 0 and pc6 > 0) or (pc5m < 0 and pc1 < 0 and pc6 < 0)
    mpts = 0.0
    if abs(pc1) >= THRESHOLDS['price_move_1h_strong']:
        mpts += SCORING['max_momentum'] * 0.25
        signals.append(_sig('price_momentum', 'momentum', mom_dir,
            f"price {pc1:+.1f}% in 1h — already extended", pc1))
    elif abs(pc1) >= THRESHOLDS['price_move_1h_notable']:
        mpts += SCORING['max_momentum'] * 0.5
        signals.append(_sig('price_momentum', 'momentum', mom_dir,
            f"price {pc1:+.1f}% in 1h", pc1))
    if abs(pc5m) >= THRESHOLDS['price_move_5m_strong']:
        mpts += SCORING['max_momentum'] * 0.2
        signals.append(_sig('price_momentum_5m', 'momentum', vol_dir_5m,
            f"price {pc5m:+.1f}% in the last 5m — moving fast", pc5m))
    elif abs(pc5m) >= THRESHOLDS['price_move_5m_notable']:
        mpts += SCORING['max_momentum'] * 0.5
        signals.append(_sig('price_momentum_5m', 'momentum', vol_dir_5m,
            f"price {pc5m:+.1f}% in the last 5m", pc5m))
    if aligned:
        mpts += SCORING['max_momentum'] * 0.3
        signals.append(_sig('momentum_alignment', 'trend', mom_dir,
            f"5m/1h/6h all {mom_dir} — aligned momentum", None))
    score += min(mpts, SCORING['max_momentum'])

    # (2b) Over-extension penalty — softer than the hard gate below, for moves
    # that are stretched but not yet disqualifying.
    over_extended = []
    if EXTENSION.get('enabled'):
        if abs(pc1) >= EXTENSION['penalty_price_change_1h_pct']:
            score -= EXTENSION['penalty_points']
            over_extended.append(f"{pc1:+.1f}% in 1h")
        if abs(pc6) >= EXTENSION['penalty_price_change_6h_pct']:
            score -= EXTENSION['penalty_points_6h']
            over_extended.append(f"{pc6:+.1f}% in 6h")
        if over_extended:
            signals.append(_sig('over_extended', 'structure', 'bearish',
                f"move already stretched ({', '.join(over_extended)}) — most of "
                f"this run is behind us; late entries here averaged a double-digit "
                f"24h loss", pc1))

    # (3) Buy pressure — halved. The buy_ratio bands that looked best in the
    # first half of the data (1.5–2.5×) were the worst in the second half, so
    # this is treated as weak corroboration rather than evidence.
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

    # (4) Activity / liquidity depth — raised from a 1-point rounding term to a
    # real 2.5-point factor, and now graded on absolute depth rather than on a
    # multiple of the (now much higher) filter floor. Liquidity was the second
    # most robust predictor in the review: alerts under $150k liquidity
    # averaged -13.8% at 24h with a 19.8% win rate against -4.0% / 41.0% above.
    apts = 0.0
    if liq >= 1_000_000 and txns_1h >= 100:
        apts = SCORING['max_activity']
    elif liq >= 500_000 and txns_1h >= 60:
        apts = SCORING['max_activity'] * 0.8
    elif liq >= FILTERS['min_liquidity_usd'] and txns_1h >= 30:
        apts = SCORING['max_activity'] * 0.5
    elif passes_filters:
        apts = SCORING['max_activity'] * 0.25
    score += apts

    # (5) Market structure — informational signals (no extra points, but the
    # exit-risk one caps the score just like the wash-trade check).
    if vol_mcap_24h is not None and vol_mcap_24h >= THRESHOLDS['vol_mcap_strong']:
        signals.append(_sig('vol_vs_mcap', 'structure', mom_dir,
            f"24h volume is {vol_mcap_24h}× the market cap — the entire valuation "
            f"is trading hands, real participation", vol_mcap_24h))
    elif vol_mcap_24h is not None and vol_mcap_24h >= THRESHOLDS['vol_mcap_notable']:
        signals.append(_sig('vol_vs_mcap', 'structure', mom_dir,
            f"24h volume is {vol_mcap_24h}× the market cap — actively traded", vol_mcap_24h))
    if thin_exit_warning:
        signals.append(_sig('exit_liquidity_risk', 'structure', 'bearish',
            f"liquidity is only {liq_mcap_pct}% of market cap "
            f"(< {FILTERS['min_liq_to_mcap_pct']}%) — exit door is thin, sells "
            f"will crater the price", liq_mcap_pct))

    # Wash-trade / thin-exit caution caps the score
    if wash_warning or thin_exit_warning:
        score = min(score, SCORING['label_good'])

    score = round(max(0.0, min(score, 10.0)), 1)

    # ── Net direction ────────────────────────────────────────────────────────
    bull = sum(1 for s in signals if s['direction'] == 'bullish')
    bear = sum(1 for s in signals if s['direction'] == 'bearish')
    direction = 'bullish' if bull > bear else 'bearish' if bear > bull else 'neutral'

    # ── Over-extension hard gate ─────────────────────────────────────────────
    # Past these thresholds we are buying the tail of a move someone else is
    # already selling. The 15%+/1h bucket averaged -16.4% at 24h (24% win) and
    # the 50%+/6h bucket -38.2% (0 of 10 profitable). No score, and no alert
    # path, may override this.
    extension_block = None
    if EXTENSION.get('enabled'):
        if abs(pc1) >= EXTENSION['max_price_change_1h_pct']:
            extension_block = (f"1h move {pc1:+.1f}% ≥ "
                               f"{EXTENSION['max_price_change_1h_pct']}%")
        elif pc5m >= EXTENSION['max_price_change_5m_pct']:
            extension_block = (f"5m move {pc5m:+.1f}% ≥ "
                               f"{EXTENSION['max_price_change_5m_pct']}%")
        elif pc6 >= EXTENSION['max_price_change_6h_pct']:
            extension_block = (f"6h move {pc6:+.1f}% ≥ "
                               f"{EXTENSION['max_price_change_6h_pct']}%")

    # ── Alert decision ───────────────────────────────────────────────────────
    # Volume qualifies on a 1h surge OR a 5m acceleration — the latter catches
    # setups heating up right now while the full hour still reads average.
    vol_ok = (vol_pace_1h >= ALERTING['min_vol_pace_1h']
              or vol_pace_5m >= ALERTING.get('min_vol_pace_5m', float('inf')))
    standard_alert = (
        passes_filters
        and not extension_block
        and score >= ALERTING['min_validity_score']
        and vol_ok
        and (not ALERTING['require_bullish'] or direction == 'bullish')
    )

    # Fast path: an extreme, coherent 5m spike (volume + price + flow all in
    # the same direction) alerts without waiting for the score — the score is
    # mostly 1h windows, which lag a fresh move by 30–60 minutes. This was the
    # best-performing path in the outcome review, so it keeps its own bar; it
    # is only subject to the extension gate, which exists to stop it firing on
    # a move that has already run 15%+.
    buys_5m, sells_5m = txns['m5']['buys'], txns['m5']['sells']
    flow_5m = round((buys_5m if pc5m >= 0 else sells_5m)
                    / max((sells_5m if pc5m >= 0 else buys_5m), 1), 2)
    fp = ALERTING.get('fast_path', {})
    fast_alert = (
        bool(fp.get('enabled'))
        and passes_filters
        and not extension_block
        and not wash_warning
        and vol_pace_5m >= fp['min_vol_pace_5m']
        and abs(pc5m) >= fp['min_price_move_5m_pct']
        and flow_5m >= fp['min_flow_ratio_5m']
        and (not ALERTING['require_bullish'] or pc5m > 0)
    )
    if fast_alert:
        signals.append(_sig('fast_spike', 'volume', vol_dir_5m,
            f"⚡ 5m spike: volume {vol_pace_5m}× average, price {pc5m:+.1f}%, "
            f"{buys_5m} buys / {sells_5m} sells — caught early", vol_pace_5m))

    is_alert = standard_alert or fast_alert

    result = {
        'symbol':         pair['symbol'],
        'name':           pair['name'],
        'chain':          pair['chain'],
        'dex':            pair['dex'],
        'quote_symbol':   pair['quote_symbol'],
        'url':            pair['url'],
        # Correctly-cased pool/token addresses. DexScreener's own URL slug
        # lowercases the address, which corrupts case-sensitive base58
        # addresses (Solana) — store these separately so downstream lookups
        # (e.g. GeckoTerminal outcome backfill) don't have to guess the case.
        'pair_address':   pair.get('pair_address'),
        'token_address':  pair.get('token_address'),
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
        'thin_exit_warning': thin_exit_warning,
        'over_extended':  bool(over_extended),
        'extension_block': extension_block,
        'is_alert':       is_alert,
        'alert_path':     ('fast' if fast_alert and not standard_alert
                           else 'standard' if is_alert else None),
        'scanned_at':     datetime.now(timezone.utc).isoformat(),
    }
    result['exit_plan'] = _exit_plan(result)
    result['summary'] = build_summary(result)
    return result


def _exit_plan(r: dict) -> dict:
    """
    Concrete take-profit / stop / time-stop levels for a bullish alert.

    The outcome review's central finding is that these alerts are short trades:
    bullish alerts averaged +0.55% at 15m and +0.34% at 1h but -5.5% by 24h,
    while 59% of them touched +5% and 39% touched +10% somewhere inside the
    day. The edge is real and it is brief, so the exit belongs in the alert
    itself rather than in the reader's judgement an hour later.
    """
    # Only for coins we are actually calling — attaching a trade plan to a
    # coin that failed filters or never alerted reads as a recommendation.
    if not EXIT_PLAN.get('enabled') or not r.get('is_alert'):
        return None
    if r.get('direction') != 'bullish':
        return None
    price = r.get('price_usd')
    if not price:
        return None
    tp, sl = EXIT_PLAN['take_profit_pct'], EXIT_PLAN['stop_loss_pct']
    return {
        'take_profit_pct':   tp,
        'stop_loss_pct':     sl,
        'max_hold_hours':    EXIT_PLAN['max_hold_hours'],
        'take_profit_price': price * (1 + tp / 100),
        'stop_loss_price':   price * (1 - sl / 100),
        'observed':          EXIT_PLAN.get('observed', {}),
    }


def _sig(name, category, direction, detail, value):
    return {'name': name, 'category': category, 'direction': direction,
            'detail': detail, 'value': value}


# ── Human-readable description (used in the email + UI) ──────────────────────

def build_summary(r: dict) -> str:
    """
    A plain-English paragraph describing what the scanner sees.

    The emoji in here are deliberate and must stay. This one string is sent to
    three places — Telegram, plain-text email, and the web UI — and only the
    last can render SVG. Telegram's formatting supports a small HTML subset
    with no <svg> or <use>, so emitting Phosphor markup here would put visible
    tag soup in the channel. The browser swaps these for Phosphor icons at
    render time via iconifySummary() in static/icons.js; if you add or change
    an emoji below, add it to SUMMARY_ICONS there too.
    """
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

    if r.get('alert_path') == 'fast':
        parts.append("⚡ Early catch: extreme 5-minute spike — volume, price and "
                     "order flow all agree, alerted before the 1h stats caught up.")
    elif r.get('alert_path') == 'smart':
        parts.append("🧠 Smart-money alert: a tracked winning wallet just bought "
                     "this coin.")

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

    if r.get('thin_exit_warning'):
        parts.append(f"⚠️ Thin exit: liquidity is only {m.get('liq_mcap_pct')}% of "
                     f"market cap — large sells will move the price hard.")

    if r.get('over_extended'):
        parts.append("⚠️ Already extended: a large part of this move has happened. "
                     "Alerts this late historically gave back more than they gained.")

    sm = r.get('smart_money')
    if sm:
        if sm.get('smart_buys'):
            parts.append(f"🧠 Smart money: {sm['smart_buys']} tracked wallet buy(s) "
                         f"in the last {sm['lookback_minutes']}m (${sm['smart_buy_usd']:,.0f}) "
                         f"— confirmation only, not a reason to buy on its own.")
        if sm.get('net_flow_usd'):
            flow = sm['net_flow_usd']
            parts.append(f"🐳 Whale net flow {'+' if flow >= 0 else ''}${flow:,.0f} over "
                         f"{sm['lookback_minutes']}m ({sm['buyers']} buyers / {sm['sellers']} sellers "
                         f"≥ ${sm['min_trade_usd']:,.0f}).")

    xp = r.get('exit_plan')
    if xp:
        obs = xp.get('observed', {})
        parts.append(
            f"🎯 Plan: take profit +{xp['take_profit_pct']:.0f}% "
            f"({_fmt_price(xp['take_profit_price'])}), stop -{xp['stop_loss_pct']:.0f}% "
            f"({_fmt_price(xp['stop_loss_price'])}), and close by "
            f"{xp['max_hold_hours']}h regardless."
        )
        if obs:
            parts.append(
                f"On comparable past alerts {obs.get('hit_tp_within_24h_pct')}% reached "
                f"the target within 24h and {obs.get('hit_sl_within_24h_pct')}% hit the "
                f"stop ({obs.get('reached_5pct_within_24h_pct')}% got at least +5%), with "
                f"the typical peak about {obs.get('median_hours_to_peak')}h in. The edge "
                f"decays fast — this is a same-day trade, not a hold."
            )

    return ' '.join(parts)


# ── Post-enrichment revalidation ────────────────────────────────────────────

def _revalidate_after_enrichment(r: dict) -> dict:
    """
    Re-apply the alert gates that depend on values smart-money enrichment can
    change. Enrichment appends signals and recounts `direction`, so a coin that
    was bullish when `evaluate` decided to alert can come out of enrichment
    bearish or neutral — with the alert flag still set from before.

    That is exactly how 148 bearish and 64 neutral alerts reached the feed
    during the review period despite require_bullish being on the whole time,
    and none of them had an edge (bearish alerts averaged a -1.2% adverse move
    at 24h with a 42.5% hit rate, i.e. worse than a coin flip). The direction
    check has to run last, after every signal is in.
    """
    if not r.get('is_alert'):
        return r
    if ALERTING['require_bullish'] and r.get('direction') != 'bullish':
        logger.info(f"{r.get('symbol')} alert dropped: direction became "
                    f"{r.get('direction')} after smart-money enrichment")
        r['is_alert'] = False
        r['alert_path'] = None
        return r
    # A score bonus must not carry a coin over the bar if enrichment also
    # pushed it below the minimum (possible via the wash/thin-exit cap).
    if r.get('alert_path') == 'standard' and r['score'] < ALERTING['min_validity_score']:
        logger.info(f"{r.get('symbol')} alert dropped: score {r['score']} fell "
                    f"below {ALERTING['min_validity_score']} after enrichment")
        r['is_alert'] = False
        r['alert_path'] = None
    return r


# ── Scanning many symbols ───────────────────────────────────────────────────

def scan_symbols(symbols: list) -> list:
    """Resolve + evaluate each symbol. Returns results sorted best-first."""
    results = []
    smart_money.begin_cycle()  # reset the per-cycle trades-fetch budget
    for sym in symbols:
        try:
            pair = dexscreener.resolve_symbol(sym)
            if pair is None:
                results.append({'symbol': sym.upper(), 'error': 'not found on DexScreener',
                                'score': 0, 'is_alert': False, 'passes_filters': False})
            else:
                r = evaluate(pair)
                r = smart_money.enrich(r)
                if r.get('smart_money'):
                    # Enrichment may have bumped score / direction / alert.
                    r['label'] = _label(r['score'])
                    _revalidate_after_enrichment(r)
                    r['exit_plan'] = _exit_plan(r)
                    r['summary'] = build_summary(r)
                results.append(r)
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

    # One BTC regime snapshot per scan — stamped on every alert this cycle so
    # the digest can later split performance by market regime.
    btc = btc_market.get_btc_context()

    for r in results:
        if not r.get('is_alert'):
            continue
        if btc:
            r['btc'] = btc
            r['summary'] = (r.get('summary') or '') + (
                f" ₿ BTC regime {btc['regime_score']}/10 ({btc['regime']}): "
                f"{btc['change_h1']:+.1f}% 1h, {btc['change_h24']:+.1f}% 24h."
            )
            gate = BTC_IMPACT.get('gate', {})
            if (gate.get('enabled')
                    and r.get('direction') == 'bullish'
                    and btc['regime_score'] < gate['min_regime_score']):
                logger.info(f"{r['symbol']} alert suppressed by BTC gate "
                            f"(regime {btc['regime_score']} < {gate['min_regime_score']})")
                continue
        # Daily per-symbol ceiling. Unlike the cooldown below, no score jump
        # can override this: repeat alerting on one symbol was a top source of
        # bad alerts (DIH fired 60 times in a month, averaging -19.9% at 24h
        # with a 22% win rate; CASHCAT fired 131 times), and the score-jump
        # re-arm is precisely what let a pumping-then-dumping coin keep
        # re-qualifying all day.
        cap = ALERTING.get('max_alerts_per_symbol_24h')
        if cap:
            sent_24h = db.alert_count_within(r['symbol'], 1440)
            if sent_24h >= cap:
                logger.info(f"{r['symbol']} alert suppressed "
                            f"(daily cap: {sent_24h}/{cap} in last 24h)")
                continue

        # Cooldown, with a re-arm: a repeat alert gets through if its score
        # beats the best one already sent in the window by a clear margin.
        prev_best = db.best_alert_score_within(r['symbol'], ALERTING['cooldown_minutes'])
        if prev_best is not None:
            jump = ALERTING.get('cooldown_rearm_score_jump')
            if jump is None or r['score'] < prev_best + jump:
                logger.info(f"{r['symbol']} alert suppressed "
                            f"(cooldown; score {r['score']} vs best {prev_best})")
                continue
            logger.info(f"{r['symbol']} re-alerting inside cooldown: "
                        f"score jumped {prev_best} → {r['score']}")
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
            # Daily 24h-outcome digest — runs even while scanning is paused,
            # since past alerts still deserve their scorecard.
            digest.maybe_send_daily_digest()
            # Score pending whale buys (throttled internally) so wallet
            # win-rates keep accruing even between scans.
            smart_money.maybe_backfill()
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
