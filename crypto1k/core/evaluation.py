"""
Turns backfilled alert outcomes (crypto1k/core/outcomes.py) into an answer to
the only question that matters: was each alert worth sending, and how late
did it fire? Breaks that down by score band, alert path, and symbol so the
numbers point at concrete config changes in scanner_config.py.

Public surface:
  generate_report(min_alerts_per_bucket=3) -> dict
  format_report(report) -> str   (human-readable, for CLI / API)
"""

import json
from statistics import mean, median

from crypto1k.core import db

HIT_THRESHOLD_PCT = 2.0  # a "win" = reached this % in the alert's favor

# A signal needs at least this many alerts before its numbers mean anything.
MIN_SIGNAL_SAMPLE = 10


def _signed(alert: dict, favorable_pct=None, adverse_pct=None):
    """+1 for bullish, -1 for bearish, 0 (excluded) for neutral."""
    d = alert.get('direction')
    return 1 if d == 'bullish' else -1 if d == 'bearish' else 0


def _pct(a, b):
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1) * 100


def _enrich(alert: dict) -> dict:
    """Add direction-adjusted return/lateness fields to one joined alert row."""
    sign = _signed(alert)
    base = alert.get('price_at_alert')
    row = dict(alert)
    row['sign'] = sign

    for h, col in (('15m', 'price_15m'), ('1h', 'price_1h'),
                   ('4h', 'price_4h'), ('24h', 'price_24h')):
        raw = _pct(alert.get(col), base)
        row[f'ret_{h}'] = raw * sign if (raw is not None and sign) else None

    if sign == 1:
        row['max_favorable_1h'] = _pct(alert.get('high_1h'), base)
        row['max_adverse_1h'] = _pct(base, alert.get('low_1h'))
        row['max_favorable_24h'] = _pct(alert.get('high_24h'), base)
        row['max_adverse_24h'] = _pct(base, alert.get('low_24h'))
    elif sign == -1:
        row['max_favorable_1h'] = _pct(base, alert.get('low_1h'))
        row['max_adverse_1h'] = _pct(alert.get('high_1h'), base)
        row['max_favorable_24h'] = _pct(base, alert.get('low_24h'))
        row['max_adverse_24h'] = _pct(alert.get('high_24h'), base)
    else:
        row['max_favorable_1h'] = row['max_adverse_1h'] = None
        row['max_favorable_24h'] = row['max_adverse_24h'] = None

    # Lateness: how much of the eventual 1h-after move had already happened
    # in the hour *before* the alert fired (same direction, same magnitude scale).
    already = _pct(base, alert.get('price_1h_before'))
    row['moved_before_pct'] = already * sign if (already is not None and sign) else None
    row['hit_1h'] = (row['max_favorable_1h'] is not None
                      and row['max_favorable_1h'] >= HIT_THRESHOLD_PCT)
    row['drawdown_1h'] = (row['max_adverse_1h'] is not None
                           and row['max_adverse_1h'] >= HIT_THRESHOLD_PCT)
    return row


def _bucket_stats(rows: list, min_n: int) -> dict:
    n = len(rows)
    if n < min_n:
        return {'n': n, 'insufficient': True}

    def avg_at(field):
        vals = [r[field] for r in rows if r.get(field) is not None]
        return (round(mean(vals), 2), len(vals)) if vals else (None, 0)

    ret_15m, n_15m = avg_at('ret_15m')
    ret_1h, n_1h = avg_at('ret_1h')
    ret_4h, n_4h = avg_at('ret_4h')
    ret_24h, n_24h = avg_at('ret_24h')

    hits = [r for r in rows if r['hit_1h'] is not None]
    dd = [r for r in rows if r['drawdown_1h'] is not None]
    moved_before = [r['moved_before_pct'] for r in rows if r['moved_before_pct'] is not None]
    fav_1h = [r['max_favorable_1h'] for r in rows if r['max_favorable_1h'] is not None]
    fav_24h = [r['max_favorable_24h'] for r in rows if r['max_favorable_24h'] is not None]

    return {
        'n': n,
        # Return curve across horizons — shows where the edge peaks and where
        # it decays, i.e. how long an alert stays worth acting on.
        'avg_ret_15m_pct':  ret_15m, 'n_15m': n_15m,
        'avg_ret_1h_pct':   ret_1h,  'n_1h':  n_1h,
        'avg_ret_4h_pct':   ret_4h,  'n_4h':  n_4h,
        'avg_ret_24h_pct':  ret_24h, 'n_24h': n_24h,
        'hit_rate_1h_pct':     round(100 * sum(r['hit_1h'] for r in hits) / len(hits), 1) if hits else None,
        'drawdown_rate_1h_pct': round(100 * sum(r['drawdown_1h'] for r in dd) / len(dd), 1) if dd else None,
        'avg_moved_before_pct': round(mean(moved_before), 2) if moved_before else None,
        'avg_max_favorable_1h_pct': round(mean(fav_1h), 2) if fav_1h else None,
        'avg_max_favorable_24h_pct': round(mean(fav_24h), 2) if fav_24h else None,
    }


def _signal_names(alert: dict) -> list:
    raw = alert.get('signal_names')
    if not raw:
        return []
    if isinstance(raw, list):
        return [s for s in raw if s]
    try:
        return [s for s in json.loads(raw) if s]
    except (TypeError, ValueError):
        return []


def _by_signal(rows: list, min_n: int = MIN_SIGNAL_SAMPLE) -> list:
    """
    Per-signal effectiveness: for each signal name, how alerts carrying it
    performed versus alerts that did not.

    This is the question the score weights are supposed to answer and could
    not be asked from inside the app before — the 2026-07-26 rebalance was
    done by hand against a database copy. `edge_*_pp` is the difference in
    average return (percentage points) between alerts with and without the
    signal, which is what actually justifies a weight going up or down.
    Sorted best-edge first.
    """
    all_names = sorted({n for r in rows for n in _signal_names(r)})
    out = []
    for name in all_names:
        with_sig = [r for r in rows if name in _signal_names(r)]
        without = [r for r in rows if name not in _signal_names(r)]
        if len(with_sig) < min_n:
            continue

        def avg(subset, field):
            vals = [r[field] for r in subset if r.get(field) is not None]
            return mean(vals) if vals else None

        def winrate(subset, field):
            vals = [r[field] for r in subset if r.get(field) is not None]
            return 100 * sum(1 for v in vals if v > 0) / len(vals) if vals else None

        def edge(field):
            a, b = avg(with_sig, field), avg(without, field)
            return round(a - b, 2) if (a is not None and b is not None) else None

        out.append({
            'signal':        name,
            'n':             len(with_sig),
            'avg_ret_1h_pct':  _r(avg(with_sig, 'ret_1h')),
            'avg_ret_24h_pct': _r(avg(with_sig, 'ret_24h')),
            'win_rate_1h_pct':  _r(winrate(with_sig, 'ret_1h'), 1),
            'win_rate_24h_pct': _r(winrate(with_sig, 'ret_24h'), 1),
            'edge_1h_pp':    edge('ret_1h'),
            'edge_24h_pp':   edge('ret_24h'),
        })
    out.sort(key=lambda d: (d['edge_1h_pp'] is None, -(d['edge_1h_pp'] or 0)))
    return out


def _r(v, nd=2):
    return round(v, nd) if v is not None else None


def _exit_calibration(rows: list) -> dict:
    """
    How a take-profit / stop / time-stop would actually have played out.

    Bullish alerts in the review period peaked a median 5.6h after firing and
    then gave the move back, so the practical question is not "what is the 24h
    return" but "would the target have filled before the stop". These are the
    numbers that set EXIT_PLAN in scanner_config.
    """
    fav24 = [r['max_favorable_24h'] for r in rows if r.get('max_favorable_24h') is not None]
    adv24 = [r['max_adverse_24h'] for r in rows if r.get('max_adverse_24h') is not None]
    if not fav24:
        return {'n': 0}
    out = {'n': len(fav24)}
    for tp in (5, 10, 20):
        out[f'reached_plus_{tp}pct_within_24h'] = _r(
            100 * sum(1 for v in fav24 if v >= tp) / len(fav24), 1)
    for sl in (10, 20):
        out[f'drew_down_{sl}pct_within_24h'] = _r(
            100 * sum(1 for v in adv24 if v >= sl) / len(adv24), 1) if adv24 else None
    out['median_max_favorable_24h_pct'] = _r(median(fav24))
    out['median_max_adverse_24h_pct'] = _r(median(adv24)) if adv24 else None
    # Alerts that ran 10%+ in our favor and still closed the day red — the
    # clearest argument for taking profit rather than holding.
    closed = [r for r in rows
              if r.get('max_favorable_24h') is not None and r.get('ret_24h') is not None]
    if closed:
        gave_back = [r for r in closed if r['max_favorable_24h'] >= 10 and r['ret_24h'] < 0]
        out['ran_10pct_then_closed_red_pct'] = _r(100 * len(gave_back) / len(closed), 1)
    return out


def _score_band(score):
    if score is None:
        return 'unknown'
    if score >= 9:
        return '9-10'
    if score >= 8:
        return '8-9'
    if score >= 7:
        return '7-8'
    return '6.1-7'


def generate_report(min_alerts_per_bucket: int = 3) -> dict:
    raw = db.get_alerts_with_outcomes(limit=5000)
    rows = [_enrich(a) for a in raw if a.get('price_at_alert') is not None]
    complete = [r for r in rows if r.get('ret_1h') is not None]  # has ≥1h of data
    n_complete_24h = sum(1 for r in rows if r.get('ret_24h') is not None)

    by_score = {}
    for band in ('6.1-7', '7-8', '8-9', '9-10'):
        by_score[band] = _bucket_stats(
            [r for r in complete if _score_band(r.get('score')) == band],
            min_alerts_per_bucket)

    by_path = {}
    for path in ('fast', 'standard', 'smart'):
        by_path[path] = _bucket_stats(
            [r for r in complete if r.get('alert_path') == path], min_alerts_per_bucket)

    # Liquidity is the strongest single conditioner found in the review, so the
    # report now breaks it out rather than leaving it to be rediscovered.
    by_liquidity = {}
    for name, lo, hi in (('<150k', 0, 150_000), ('150k-500k', 150_000, 500_000),
                         ('500k-1m', 500_000, 1_000_000), ('>1m', 1_000_000, float('inf'))):
        by_liquidity[name] = _bucket_stats(
            [r for r in complete
             if r.get('liquidity_usd') is not None and lo <= r['liquidity_usd'] < hi],
            min_alerts_per_bucket)

    wash = _bucket_stats([r for r in complete if r.get('wash_warning') == 1], min_alerts_per_bucket)
    clean = _bucket_stats([r for r in complete if r.get('wash_warning') != 1], min_alerts_per_bucket)

    by_symbol = {}
    symbols = {r['symbol'] for r in complete}
    for sym in symbols:
        stats = _bucket_stats([r for r in complete if r['symbol'] == sym], min_alerts_per_bucket)
        if not stats.get('insufficient'):
            by_symbol[sym] = stats
    ranked_symbols = sorted(by_symbol.items(), key=lambda kv: kv[1]['avg_ret_1h_pct'] or 0)

    overall = _bucket_stats(complete, 1)
    lateness = _bucket_stats(complete, 1)  # same rows; report reads moved_before/max_favorable

    by_signal = _by_signal(complete)
    suggestions = _suggest(by_score, by_path, wash, clean, overall,
                           by_signal, by_liquidity)

    bullish = [r for r in complete if r.get('direction') == 'bullish']

    return {
        'n_alerts_total': len(raw),
        'n_alerts_with_1h_data': len(complete),
        'n_alerts_with_24h_data': n_complete_24h,
        'overall': overall,
        'by_score_band': by_score,
        'by_alert_path': by_path,
        'by_liquidity': by_liquidity,
        'by_signal': by_signal,
        'exit_calibration': _exit_calibration(bullish),
        'wash_warning_alerts': wash,
        'clean_alerts': clean,
        'worst_symbols': ranked_symbols[:5],
        'best_symbols': ranked_symbols[-5:][::-1],
        'suggestions': suggestions,
    }


def _suggest(by_score, by_path, wash, clean, overall,
             by_signal=None, by_liquidity=None) -> list:
    out = []

    # Signals that cost the alert money but still earn score points. This is
    # the check that would have caught the pre-2026-07-26 weighting, where
    # price momentum carried 3 of 10 points while being the most reliably
    # negative signal family in the data.
    for s in (by_signal or []):
        if s['edge_1h_pp'] is not None and s['edge_1h_pp'] < -0.5:
            out.append(
                f"Signal '{s['signal']}' underperforms: alerts carrying it average "
                f"{s['avg_ret_1h_pct']:+.2f}% at 1h, {abs(s['edge_1h_pp']):.2f}pp worse "
                f"than alerts without it (n={s['n']}). If it earns score points in "
                f"SCORING/THRESHOLDS, that weight is working against you."
            )
    best = next((s for s in (by_signal or [])
                 if s['edge_1h_pp'] is not None and s['edge_1h_pp'] > 0.5), None)
    if best:
        out.append(
            f"Signal '{best['signal']}' is the strongest performer: "
            f"{best['avg_ret_1h_pct']:+.2f}% at 1h, {best['edge_1h_pp']:+.2f}pp better "
            f"than alerts without it (n={best['n']}). Worth more weight, or its own "
            f"threshold tier."
        )

    if by_liquidity:
        thin = by_liquidity.get('<150k', {})
        deep = by_liquidity.get('>1m', {})
        if (not thin.get('insufficient') and not deep.get('insufficient')
                and thin.get('avg_ret_24h_pct') is not None
                and deep.get('avg_ret_24h_pct') is not None
                and thin['avg_ret_24h_pct'] < deep['avg_ret_24h_pct'] - 3.0):
            out.append(
                f"Sub-$150k-liquidity alerts average {thin['avg_ret_24h_pct']:+.1f}% at 24h "
                f"(n={thin['n']}) vs {deep['avg_ret_24h_pct']:+.1f}% above $1m (n={deep['n']}). "
                f"FILTERS['min_liquidity_usd'] is the highest-leverage knob here."
            )

    bands = ['6.1-7', '7-8', '8-9', '9-10']
    valid_bands = [(b, by_score[b]) for b in bands if not by_score[b].get('insufficient')]
    if len(valid_bands) >= 2:
        worst_b, worst_s = min(valid_bands, key=lambda kv: kv[1]['avg_ret_1h_pct'] or 0)
        best_b, best_s = max(valid_bands, key=lambda kv: kv[1]['avg_ret_1h_pct'] or 0)
        if (worst_s['avg_ret_1h_pct'] or 0) < 0 and worst_b in ('6.1-7', '7-8'):
            out.append(
                f"Score band {worst_b} averages {worst_s['avg_ret_1h_pct']:+.1f}% at 1h "
                f"(n={worst_s['n']}) vs {best_b} at {best_s['avg_ret_1h_pct']:+.1f}% "
                f"(n={best_s['n']}). Consider raising min_validity_score above {worst_b.split('-')[1]}."
            )

    fast, standard = by_path.get('fast', {}), by_path.get('standard', {})
    if not fast.get('insufficient') and not standard.get('insufficient'):
        if (fast['avg_ret_1h_pct'] or 0) < (standard['avg_ret_1h_pct'] or 0) - 1.0:
            out.append(
                f"Fast path averages {fast['avg_ret_1h_pct']:+.1f}% vs standard path "
                f"{standard['avg_ret_1h_pct']:+.1f}% (n={fast['n']} vs {standard['n']}). "
                f"Consider tightening ALERTING['fast_path'] thresholds in scanner_config.py."
            )
        elif (fast['avg_ret_1h_pct'] or 0) >= (standard['avg_ret_1h_pct'] or 0):
            out.append(
                f"Fast path holds up: {fast['avg_ret_1h_pct']:+.1f}% avg vs standard's "
                f"{standard['avg_ret_1h_pct']:+.1f}% (n={fast['n']} vs {standard['n']})."
            )

    if not wash.get('insufficient') and (wash['avg_ret_1h_pct'] or 0) < 0:
        out.append(
            f"wash_warning alerts average {wash['avg_ret_1h_pct']:+.1f}% at 1h (n={wash['n']}) "
            f"— consider excluding them from alerts entirely instead of just capping score."
        )

    if not overall.get('insufficient') and overall.get('avg_moved_before_pct') is not None:
        out.append(
            f"On average {overall['avg_moved_before_pct']:+.1f}% of the eventual 1h move had "
            f"already happened in the hour *before* the alert fired, and "
            f"{overall.get('avg_max_favorable_1h_pct', 0):+.1f}% more was still available after. "
            f"This is the direct 'how late are we' number — track it over time as the "
            f"fast-path / scoring changes take effect."
        )

    if not overall.get('insufficient'):
        curve = [('15m', overall.get('avg_ret_15m_pct')), ('1h', overall.get('avg_ret_1h_pct')),
                 ('4h', overall.get('avg_ret_4h_pct')), ('24h', overall.get('avg_ret_24h_pct'))]
        peak = max((c for c in curve if c[1] is not None), key=lambda c: c[1], default=None)
        last = next((c for c in reversed(curve) if c[1] is not None), None)
        if peak and last:
            if last[1] < peak[1] * 0.5 or (peak[1] > 0 and last[1] < 0):
                out.append(
                    f"Edge peaks around {peak[0]} ({peak[1]:+.1f}%) and fades to {last[1]:+.1f}% "
                    f"by {last[0]} — treat an alert as stale (not worth acting on) somewhere "
                    f"around the {peak[0]}-to-{last[0]} mark rather than holding it all day."
                )
            else:
                out.append(
                    f"Return holds up from {peak[0]} ({peak[1]:+.1f}%) through {last[0]} "
                    f"({last[1]:+.1f}%) — the edge isn't clearly decaying within this window; "
                    f"a hard expiry inside 24h doesn't look justified by this data yet."
                )

    if not out:
        out.append("Not enough data yet in any bucket for a confident suggestion — "
                    "re-run after more alerts accumulate outcomes.")
    return out


def format_report(r: dict) -> str:
    lines = []
    lines.append("═" * 70)
    lines.append("ALERT OUTCOME REPORT")
    lines.append("═" * 70)
    lines.append(f"Alerts total: {r['n_alerts_total']}  |  "
                  f"with ≥1h outcome: {r['n_alerts_with_1h_data']}  |  "
                  f"with full 24h outcome: {r['n_alerts_with_24h_data']}")
    lines.append("")

    def _fmt(v):
        return f"{v:+.2f}" if v is not None else "n/a"

    def fmt_bucket(name, s):
        if s.get('insufficient'):
            return f"  {name:<12} n={s['n']:<4} (insufficient data)"
        return (f"  {name:<12} n={s['n']:<4} "
                f"avg_1h={s['avg_ret_1h_pct']:+6.2f}%  avg_24h={_fmt(s['avg_ret_24h_pct'])}  "
                f"hit_rate_1h={_fmt(s['hit_rate_1h_pct'])}%  "
                f"moved_before={_fmt(s['avg_moved_before_pct'])}%")

    def fmt_curve(name, s):
        if s.get('insufficient'):
            return f"  {name:<12} n={s['n']:<4} (insufficient data)"
        return (f"  {name:<12} "
                f"15m={_fmt(s['avg_ret_15m_pct']):>7}% (n={s['n_15m']:<3}) "
                f"1h={_fmt(s['avg_ret_1h_pct']):>7}% (n={s['n_1h']:<3}) "
                f"4h={_fmt(s['avg_ret_4h_pct']):>7}% (n={s['n_4h']:<3}) "
                f"24h={_fmt(s['avg_ret_24h_pct']):>7}% (n={s['n_24h']:<3})")

    lines.append("Return decay curve — avg directional return at each horizon after")
    lines.append("the alert fired. Where this peaks and turns is your practical")
    lines.append("'how long is the signal worth acting on' answer:")
    lines.append(fmt_curve('all', r['overall']))
    for band in ('6.1-7', '7-8', '8-9', '9-10'):
        lines.append(fmt_curve(f'score {band}', r['by_score_band'][band]))
    lines.append("")

    lines.append("Overall:")
    lines.append(fmt_bucket('all', r['overall']))
    lines.append("")

    lines.append("By score band:")
    for band in ('6.1-7', '7-8', '8-9', '9-10'):
        lines.append(fmt_bucket(band, r['by_score_band'][band]))
    lines.append("")

    lines.append("By alert path:")
    for path in ('fast', 'standard', 'smart'):
        if path in r['by_alert_path']:
            lines.append(fmt_bucket(path, r['by_alert_path'][path]))
    lines.append("")

    if r.get('by_liquidity'):
        lines.append("By liquidity (the strongest single conditioner):")
        for name in ('<150k', '150k-500k', '500k-1m', '>1m'):
            if name in r['by_liquidity']:
                lines.append(fmt_bucket(name, r['by_liquidity'][name]))
        lines.append("")

    if r.get('by_signal'):
        lines.append("Per-signal effectiveness — 'edge' is the difference in average")
        lines.append("return between alerts carrying the signal and alerts without it.")
        lines.append("Negative edge on a signal that earns score points is a bug in the")
        lines.append("weights, not a fact about the market:")
        for s in r['by_signal']:
            lines.append(
                f"  {s['signal']:<22} n={s['n']:<4} "
                f"1h={_fmt(s['avg_ret_1h_pct']):>7}% (win {_fmt(s['win_rate_1h_pct'])}%) "
                f"24h={_fmt(s['avg_ret_24h_pct']):>7}%  "
                f"edge_1h={_fmt(s['edge_1h_pp']):>7}pp edge_24h={_fmt(s['edge_24h_pp']):>7}pp")
        lines.append("")

    xc = r.get('exit_calibration') or {}
    if xc.get('n'):
        lines.append(f"Exit calibration (bullish alerts, n={xc['n']}) — would the")
        lines.append("target have filled before the stop?")
        for tp in (5, 10, 20):
            k = f'reached_plus_{tp}pct_within_24h'
            if xc.get(k) is not None:
                lines.append(f"  reached +{tp}% within 24h:  {xc[k]:>5}%")
        for sl in (10, 20):
            k = f'drew_down_{sl}pct_within_24h'
            if xc.get(k) is not None:
                lines.append(f"  drew down -{sl}% within 24h: {xc[k]:>5}%")
        if xc.get('ran_10pct_then_closed_red_pct') is not None:
            lines.append(f"  ran +10% then closed 24h red: "
                          f"{xc['ran_10pct_then_closed_red_pct']}%  "
                          f"← the case for taking profit")
        lines.append("")

    lines.append("Wash-warning vs clean:")
    lines.append(fmt_bucket('wash_warning', r['wash_warning_alerts']))
    lines.append(fmt_bucket('clean', r['clean_alerts']))
    lines.append("")

    lines.append("Worst symbols (avg 1h return, min sample size):")
    for sym, s in r['worst_symbols']:
        lines.append(fmt_bucket(sym, s))
    lines.append("")
    lines.append("Best symbols:")
    for sym, s in r['best_symbols']:
        lines.append(fmt_bucket(sym, s))
    lines.append("")

    lines.append("Suggestions:")
    for s in r['suggestions']:
        lines.append(f"  • {s}")
    lines.append("═" * 70)
    return '\n'.join(lines)
