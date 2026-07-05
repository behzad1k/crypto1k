"""
Turns backfilled alert outcomes (crypto1k/core/outcomes.py) into an answer to
the only question that matters: was each alert worth sending, and how late
did it fire? Breaks that down by score band, alert path, and symbol so the
numbers point at concrete config changes in scanner_config.py.

Public surface:
  generate_report(min_alerts_per_bucket=3) -> dict
  format_report(report) -> str   (human-readable, for CLI / API)
"""

from statistics import mean

from crypto1k.core import db

HIT_THRESHOLD_PCT = 2.0  # a "win" = reached this % in the alert's favor


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
    for path in ('fast', 'standard'):
        by_path[path] = _bucket_stats(
            [r for r in complete if r.get('alert_path') == path], min_alerts_per_bucket)

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

    suggestions = _suggest(by_score, by_path, wash, clean, overall)

    return {
        'n_alerts_total': len(raw),
        'n_alerts_with_1h_data': len(complete),
        'n_alerts_with_24h_data': n_complete_24h,
        'overall': overall,
        'by_score_band': by_score,
        'by_alert_path': by_path,
        'wash_warning_alerts': wash,
        'clean_alerts': clean,
        'worst_symbols': ranked_symbols[:5],
        'best_symbols': ranked_symbols[-5:][::-1],
        'suggestions': suggestions,
    }


def _suggest(by_score, by_path, wash, clean, overall) -> list:
    out = []

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
    for path in ('fast', 'standard'):
        lines.append(fmt_bucket(path, r['by_alert_path'][path]))
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
