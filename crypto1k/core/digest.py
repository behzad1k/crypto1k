"""
Daily Telegram digest: how yesterday's buy alerts actually paid off.

Once per day (after DIGEST['hour_local']), sends one message covering every
buy alert whose 24h outcome window has completed and that hasn't been reported
yet. Keyed on outcome completion rather than calendar day, so an alert fired
at 23:50 is judged on a full 24h of data and downtime self-heals: the next
digest simply catches up on everything that completed in between.

Per alert: score, fired time, price at alert vs now, best price within 24h
(with when it hit and % move — i.e. would a take-profit have filled), and the
worst dip. The footer aggregates the day's best-moves so the take-profit
threshold can be calibrated on evidence: the *minimum* best-move across alerts
is the gain every single alert reached.

Public surface:
  maybe_send_daily_digest()  -> None  (cheap no-op when not due; called from
                                       the scanner monitor loop every tick)
"""

import logging
import threading
import time
from datetime import datetime

from crypto1k.config.scanner_config import BTC_IMPACT, DIGEST, EXIT_PLAN
from crypto1k.core import db, outcomes
from crypto1k.data import dexscreener
from crypto1k.notify import telegram_notify

logger = logging.getLogger(__name__)

_TELEGRAM_CHUNK = 3500  # keep well under Telegram's 4096-char message cap

_lock = threading.Lock()
_running = False
_next_attempt_after = 0.0  # monotonic; throttles retries after a failure


def maybe_send_daily_digest():
    """Kick off the daily digest in its own thread when due. Cheap otherwise."""
    global _running
    if not DIGEST.get('enabled'):
        return
    now = datetime.now()
    today = now.strftime('%Y-%m-%d')
    if now.hour < DIGEST.get('hour_local', 9):
        return
    if db.get_state('digest_last_sent_date') == today:
        return
    if time.monotonic() < _next_attempt_after:
        return
    with _lock:
        if _running:
            return
        _running = True
    # Own thread: the outcome backfill and per-coin price fetches are paced by
    # API rate limits and can take minutes — must not stall the scan loop.
    threading.Thread(target=_run, args=(today,), daemon=True).start()


def _run(today: str):
    global _running, _next_attempt_after
    try:
        if _build_and_send(today):
            db.set_state('digest_last_sent_date', today)
        else:
            _next_attempt_after = (time.monotonic()
                                   + DIGEST.get('retry_minutes', 10) * 60)
    except Exception as e:
        logger.warning(f'Daily digest failed: {e}')
        _next_attempt_after = (time.monotonic()
                               + DIGEST.get('retry_minutes', 10) * 60)
    finally:
        _running = False


def _build_and_send(today: str) -> bool:
    # Make sure recently-elapsed alerts have their outcomes computed before we
    # decide what's reportable.
    try:
        outcomes.backfill_outcomes(limit=150)
    except Exception as e:
        logger.warning(f'Digest: outcome backfill errored, continuing: {e}')

    rows = db.get_alerts_for_digest()
    buys = [r for r in rows if r['direction'] == 'bullish'
            and r['price_at_alert'] and r['high_24h']]

    if not buys:
        ok = telegram_notify.send_message(
            f"📊 Daily digest — {today}\n"
            f"No buy alerts completed their 24h window since the last digest."
        )
    else:
        ok = _send_chunked(_build_message(today, buys))

    if ok:
        # Mark everything (including non-buys and unpriceable rows) so the
        # backlog doesn't grow forever while sell alerts are disabled.
        db.mark_digest_sent([r['id'] for r in rows])
        logger.info(f'Daily digest sent: {len(buys)} buy alert(s), '
                    f'{len(rows)} outcome row(s) marked reported')
    return ok


def _build_message(today: str, buys: list) -> str:
    lines = [f"📊 Daily digest — {today}",
             f"{len(buys)} buy alert(s) with a completed 24h window\n"]

    best_moves = []
    for r in buys:
        entry = r['price_at_alert']
        best_pct = (r['high_24h'] - entry) / entry * 100
        best_moves.append(best_pct)
        now_price = _current_price(r)
        now_pct = ((now_price - entry) / entry * 100) if now_price else None
        dip_pct = ((r['low_24h'] - entry) / entry * 100) if r['low_24h'] else None

        fired = _parse_ts(r['created_at'])
        best_at = _parse_ts(r['high_24h_at']) if r['high_24h_at'] else None
        hours_to_best = ((best_at - fired).total_seconds() / 3600
                         if best_at and fired else None)

        lines.append(f"🟢 {r['symbol']} — score {r['score']:.1f} · fired {_fmt_ts(fired)}")
        lines.append(f"   alert {_fmt_price(entry)} → now "
                     + (f"{_fmt_price(now_price)} ({now_pct:+.1f}%)" if now_price else "n/a"))
        lines.append(f"   best {_fmt_price(r['high_24h'])} ({best_pct:+.1f}%)"
                     + (f" at {_fmt_ts(best_at)}" if best_at else "")
                     + (f" · {hours_to_best:.1f}h after alert" if hours_to_best is not None else ""))
        if dip_pct is not None:
            lines.append(f"   worst dip {_fmt_price(r['low_24h'])} ({dip_pct:+.1f}%)")
        btc_bits = []
        if r.get('btc_regime_score') is not None:
            btc_bits.append(f"regime {r['btc_regime_score']:.1f}/10 at fire")
        if r.get('btc_change_window') is not None:
            btc_bits.append(f"{r['btc_change_window']:+.1f}% over window")
        if btc_bits:
            lines.append(f"   ₿ BTC: {' · '.join(btc_bits)}")
        lines.append("")

    # Report against the exit plan the alerts actually shipped with, so the
    # scorecard grades the advice given rather than a separate yardstick.
    tp = (EXIT_PLAN.get('take_profit_pct') if EXIT_PLAN.get('enabled')
          else DIGEST.get('tp_target_pct', 5.0))
    hit_tp = sum(1 for p in best_moves if p >= tp)
    srt = sorted(best_moves)
    median = srt[len(srt) // 2]
    lines.append(f"— TP check: {hit_tp}/{len(buys)} reached +{tp:g}% within 24h")
    if EXIT_PLAN.get('enabled'):
        sl = EXIT_PLAN['stop_loss_pct']
        hit_sl = sum(1 for r in buys
                     if r.get('low_24h') and r.get('price_at_alert')
                     and (r['low_24h'] - r['price_at_alert']) / r['price_at_alert'] * 100 <= -sl)
        lines.append(f"— SL check: {hit_sl}/{len(buys)} dropped to -{sl:g}% within 24h")
    lines.append(f"— best-move range: min {min(srt):+.1f}% · median {median:+.1f}% "
                 f"· max {max(srt):+.1f}%")
    lines.append(f"  (min = the gain every alert reached — a TP at or below it "
                 f"would have filled on all {len(buys)})")

    regime_line = _btc_regime_breakdown(buys, best_moves, tp)
    if regime_line:
        lines.append(regime_line)
    return "\n".join(lines)


def _btc_regime_breakdown(buys: list, best_moves: list, tp: float):
    """
    TP hit-rate split by the BTC regime each alert fired in. This is the
    evidence that decides whether (and where) to enable the BTC entry gate —
    if the risk-off bucket's hit-rate collapses, that's your cutoff.
    """
    buckets = {}
    for r, move in zip(buys, best_moves):
        score = r.get('btc_regime_score')
        if score is None:
            continue
        if score >= BTC_IMPACT['risk_on_score']:
            label = 'BTC risk-on'
        elif score <= BTC_IMPACT['risk_off_score']:
            label = 'BTC risk-off'
        else:
            label = 'BTC neutral'
        buckets.setdefault(label, []).append(move)
    if not buckets:
        return None
    parts = [f"{label}: {sum(1 for m in moves if m >= tp)}/{len(moves)} hit +{tp:g}%"
             for label, moves in buckets.items()]
    return "— by regime at fire: " + " · ".join(parts)


def _send_chunked(text: str) -> bool:
    """Split on line boundaries to stay under Telegram's message size cap."""
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > _TELEGRAM_CHUNK:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return all(telegram_notify.send_message(c) for c in chunks)


def _current_price(alert: dict):
    chain, pool = outcomes._resolve_pool(alert)
    if not pool:
        return None
    try:
        pair = dexscreener.get_pair(chain, pool)
        time.sleep(0.4)  # polite pacing, same as the scanner
        return float((pair or {}).get('priceUsd'))
    except (TypeError, ValueError):
        return None
    except Exception as e:
        logger.warning(f"Digest: price fetch failed for {alert['symbol']}: {e}")
        return None


def _parse_ts(raw: str):
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace(' ', 'T'))
    except ValueError:
        return None


def _fmt_ts(dt) -> str:
    return dt.strftime('%b %d %H:%M UTC') if dt else 'n/a'


def _fmt_price(p) -> str:
    if p is None:
        return 'n/a'
    if p >= 1:
        return f"${p:,.4f}".rstrip('0').rstrip('.')
    return f"${p:.8f}".rstrip('0')
