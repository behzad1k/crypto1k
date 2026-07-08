"""
Database layer for the v2 symbol analysis tool.

Schema is intentionally minimal — fact-checking, validation optimizing,
and combination analysis will be added later when sample data exists.

Tables:
  signals       — every detected signal with family/role/grade metadata
  analysis_runs — one record per analysis request (for history)
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Project root (two levels above crypto1k/core/) — keeps the DB in the same
# place no matter what directory the app is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = str(PROJECT_ROOT / 'signals_v2.db')

logger = logging.getLogger(__name__)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT    NOT NULL,
                timeframe       TEXT    NOT NULL,
                horizon         TEXT    NOT NULL,   -- short / mid / long
                signal_name     TEXT    NOT NULL,
                family          TEXT    NOT NULL,
                category        TEXT    NOT NULL,
                role            TEXT    NOT NULL,   -- primary / confirmation / context
                grade           TEXT    NOT NULL,   -- A / B / C
                direction       TEXT    NOT NULL,   -- bullish / bearish / neutral
                indicator_value REAL,               -- raw indicator reading (RSI: 22.8, ADX: 36)
                price_level     REAL,               -- actual price the signal references (support at $68,400)
                candles_ago     INTEGER,            -- how fresh the signal was at save time
                price           REAL    NOT NULL,
                detected_at     TEXT    NOT NULL,
                created_at      TEXT    DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_signals_symbol_horizon
                ON signals(symbol, horizon, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_signals_symbol_tf
                ON signals(symbol, timeframe, created_at DESC);

            CREATE TABLE IF NOT EXISTS analysis_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT    NOT NULL,
                horizon         TEXT    NOT NULL,
                timeframes      TEXT    NOT NULL,  -- JSON array
                bias_direction  TEXT,              -- bullish / bearish / neutral
                bias_score      REAL,              -- -1.0 to 1.0
                total_signals   INTEGER DEFAULT 0,
                grade_a_signals INTEGER DEFAULT 0,
                grade_b_signals INTEGER DEFAULT 0,
                grade_c_signals INTEGER DEFAULT 0,
                price           REAL,
                created_at      TEXT    DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_runs_symbol
                ON analysis_runs(symbol, created_at DESC);

            -- ── Scanner module ──────────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS scanner_watchlist (
                symbol      TEXT PRIMARY KEY,        -- coin symbol or token address
                added_at    TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS scanner_alerts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol        TEXT    NOT NULL,
                score         REAL,
                label         TEXT,
                direction     TEXT,
                price_usd     REAL,
                vol_pace_1h   REAL,
                price_change_1h REAL,
                liquidity_usd REAL,
                chain         TEXT,
                url           TEXT,
                summary       TEXT,
                payload       TEXT,                  -- full result JSON
                btc_regime_score REAL,               -- BTC market regime 0-10 at fire time
                btc_price     REAL,
                btc_change_1h REAL,
                btc_change_24h REAL,
                created_at    TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_alerts_symbol
                ON scanner_alerts(symbol, created_at DESC);

            -- Shared scanner state (so the monitor's on/off + last run are
            -- consistent across all gunicorn workers, not per-process).
            CREATE TABLE IF NOT EXISTS scanner_state (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

            -- Forward price outcome for each alert, backfilled from on-chain
            -- OHLCV candles anchored to the alert's timestamp. Lets us score
            -- whether an alert was actually worth sending, and how late.
            CREATE TABLE IF NOT EXISTS alert_outcomes (
                alert_id        INTEGER PRIMARY KEY REFERENCES scanner_alerts(id),
                computed_at     TEXT NOT NULL,
                price_at_alert  REAL,
                price_1h_before REAL,   -- close ~1h before the alert (lateness)
                price_15m       REAL,   -- close ~15m after
                price_1h        REAL,
                price_4h        REAL,
                price_24h       REAL,
                high_1h         REAL,   -- max close within 1h after
                low_1h          REAL,   -- min close within 1h after
                high_24h        REAL,   -- max close within 24h after
                high_24h_at     TEXT,   -- candle timestamp of that 24h high
                low_24h         REAL,   -- min close within 24h after
                btc_change_window REAL, -- BTC's % move over the same 24h window
                candles_used    INTEGER,
                data_complete   INTEGER NOT NULL DEFAULT 0, -- 1 once 24h has elapsed and candles were found
                digest_sent     INTEGER NOT NULL DEFAULT 0  -- 1 once reported in a daily digest
            );

            -- ── Smart money module ──────────────────────────────────────────
            -- Wallets we follow. source='manual' (user-added) or 'auto'
            -- (promoted by win-rate). Deactivating keeps history but drops the
            -- wallet from signals/feed.
            CREATE TABLE IF NOT EXISTS smart_wallets (
                address     TEXT NOT NULL,   -- lowercased for EVM, as-is for Solana
                chain       TEXT NOT NULL DEFAULT '',
                label       TEXT,
                source      TEXT NOT NULL DEFAULT 'manual',
                is_active   INTEGER NOT NULL DEFAULT 1,
                added_at    TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (address, chain)
            );

            -- Every whale-sized trade seen on a watched pool.
            CREATE TABLE IF NOT EXISTS wallet_trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_key    TEXT NOT NULL UNIQUE,  -- GeckoTerminal trade id
                tx_hash      TEXT,
                wallet       TEXT NOT NULL,
                chain        TEXT NOT NULL,
                pool_address TEXT NOT NULL,
                symbol       TEXT,
                side         TEXT NOT NULL,         -- buy / sell
                amount_usd   REAL,
                price_usd    REAL,
                block_ts     TEXT NOT NULL,         -- ISO, naive UTC
                recorded_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_wtrades_wallet
                ON wallet_trades(wallet, block_ts DESC);
            CREATE INDEX IF NOT EXISTS idx_wtrades_pool
                ON wallet_trades(pool_address, block_ts DESC);
            CREATE INDEX IF NOT EXISTS idx_wtrades_ts
                ON wallet_trades(block_ts DESC);

            -- Forward return of each whale BUY, backfilled from OHLCV candles —
            -- this is what turns "a whale" into "a proven smart wallet".
            CREATE TABLE IF NOT EXISTS wallet_trade_outcomes (
                trade_id       INTEGER PRIMARY KEY REFERENCES wallet_trades(id),
                computed_at    TEXT NOT NULL,
                price_at_trade REAL,
                price_1h       REAL,
                price_24h      REAL,
                ret_1h_pct     REAL,
                ret_24h_pct    REAL,
                data_complete  INTEGER NOT NULL DEFAULT 0
            );
        ''')
    # Migrate: add columns that were added after initial schema
    with get_conn() as conn:
        existing = {r[1] for r in conn.execute("PRAGMA table_info(signals)").fetchall()}
        for col, definition in [
            ('price_level', 'REAL'),
            ('candles_ago', 'INTEGER'),
        ]:
            if col not in existing:
                conn.execute(f'ALTER TABLE signals ADD COLUMN {col} {definition}')
                logger.info(f'Migrated: added signals.{col}')

        existing = {r[1] for r in conn.execute("PRAGMA table_info(scanner_alerts)").fetchall()}
        for col, definition in [
            # Correctly-cased pool address (DexScreener's url slug lowercases
            # it, which corrupts case-sensitive base58 addresses on Solana).
            ('pair_address', 'TEXT'),
            # BTC market regime at fire time (see data/btc_market.py) — kept
            # separate from the coin's own validity score.
            ('btc_regime_score', 'REAL'),
            ('btc_price', 'REAL'),
            ('btc_change_1h', 'REAL'),
            ('btc_change_24h', 'REAL'),
        ]:
            if col not in existing:
                conn.execute(f'ALTER TABLE scanner_alerts ADD COLUMN {col} {definition}')
                logger.info(f'Migrated: added scanner_alerts.{col}')

        existing = {r[1] for r in conn.execute("PRAGMA table_info(alert_outcomes)").fetchall()}
        if 'high_24h_at' not in existing:
            conn.execute('ALTER TABLE alert_outcomes ADD COLUMN high_24h_at TEXT')
            logger.info('Migrated: added alert_outcomes.high_24h_at')
        if 'btc_change_window' not in existing:
            # BTC's own % move over this alert's 24h outcome window — separates
            # "the coin was fake" from "BTC dumped and took everything down".
            conn.execute('ALTER TABLE alert_outcomes ADD COLUMN btc_change_window REAL')
            logger.info('Migrated: added alert_outcomes.btc_change_window')
        if 'digest_sent' not in existing:
            conn.execute('ALTER TABLE alert_outcomes ADD COLUMN digest_sent INTEGER NOT NULL DEFAULT 0')
            # Outcomes that were already complete before the digest feature
            # existed are old news — the first digest starts from now.
            conn.execute('UPDATE alert_outcomes SET digest_sent = 1 WHERE data_complete = 1')
            logger.info('Migrated: added alert_outcomes.digest_sent')
    logger.info('Database initialised')


def save_analysis(symbol: str, horizon: str, timeframes: list, result: dict):
    """
    Persist signals and analysis run from the structured result dict.
    Expected result shape:
      {
        'price': float,
        'bias': {'direction': str, 'score': float},
        'families': { family_key: { 'active_signals': [...] } },
        'all_signals': [
          { 'signal_name', 'family', 'category', 'role', 'grade',
            'direction', 'timeframe', 'indicator_value'?, 'detected_at' }
        ]
      }
    """
    with get_conn() as conn:
        now = datetime.utcnow().isoformat()
        price = result.get('price', 0)
        bias = result.get('bias', {})
        all_signals = result.get('all_signals', [])

        grade_counts = {'A': 0, 'B': 0, 'C': 0}
        for s in all_signals:
            g = s.get('grade', 'C')
            grade_counts[g] = grade_counts.get(g, 0) + 1

        run_id = conn.execute('''
            INSERT INTO analysis_runs
              (symbol, horizon, timeframes, bias_direction, bias_score,
               total_signals, grade_a_signals, grade_b_signals, grade_c_signals, price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, horizon, json.dumps(timeframes),
            bias.get('direction'), bias.get('score'),
            len(all_signals),
            grade_counts['A'], grade_counts['B'], grade_counts['C'],
            price,
        )).lastrowid

        for s in all_signals:
            conn.execute('''
                INSERT INTO signals
                  (symbol, timeframe, horizon, signal_name, family, category,
                   role, grade, direction, indicator_value, price_level,
                   candles_ago, price, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                s.get('timeframe'),
                horizon,
                s.get('signal_name'),
                s.get('family'),
                s.get('category'),
                s.get('role'),
                s.get('grade'),
                s.get('direction'),
                s.get('indicator_value'),
                s.get('price_level'),
                s.get('candles_ago'),
                price,
                s.get('detected_at', now),
            ))

    return run_id


def get_recent_signals(symbol: str, horizon: Optional[str] = None, limit: int = 100) -> list:
    with get_conn() as conn:
        if horizon:
            rows = conn.execute('''
                SELECT * FROM signals
                WHERE symbol = ? AND horizon = ?
                ORDER BY created_at DESC LIMIT ?
            ''', (symbol, horizon, limit)).fetchall()
        else:
            rows = conn.execute('''
                SELECT * FROM signals
                WHERE symbol = ?
                ORDER BY created_at DESC LIMIT ?
            ''', (symbol, limit)).fetchall()
    return [dict(r) for r in rows]


# ── Scanner: watchlist ──────────────────────────────────────────────────────

def get_watchlist() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            'SELECT symbol FROM scanner_watchlist ORDER BY added_at ASC'
        ).fetchall()
    return [r['symbol'] for r in rows]


def add_to_watchlist(symbols: list) -> int:
    """Add one or more symbols. Returns how many were newly inserted."""
    added = 0
    with get_conn() as conn:
        for sym in symbols:
            sym = (sym or '').strip().upper()
            if not sym:
                continue
            cur = conn.execute(
                'INSERT OR IGNORE INTO scanner_watchlist (symbol) VALUES (?)', (sym,)
            )
            added += cur.rowcount
    return added


def remove_from_watchlist(symbol: str):
    with get_conn() as conn:
        conn.execute('DELETE FROM scanner_watchlist WHERE symbol = ?',
                     ((symbol or '').strip().upper(),))


# ── Scanner: alerts + cooldown ──────────────────────────────────────────────

def record_alert(result: dict):
    m = result.get('metrics', {})
    with get_conn() as conn:
        conn.execute('''
            INSERT INTO scanner_alerts
              (symbol, score, label, direction, price_usd, vol_pace_1h,
               price_change_1h, liquidity_usd, chain, url, pair_address, summary, payload,
               btc_regime_score, btc_price, btc_change_1h, btc_change_24h)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.get('symbol'),
            result.get('score'),
            result.get('label'),
            result.get('direction'),
            result.get('price_usd'),
            m.get('vol_pace_1h'),
            (result.get('price_change') or {}).get('h1'),
            result.get('liquidity_usd'),
            result.get('chain'),
            result.get('url'),
            result.get('pair_address'),
            result.get('summary'),
            json.dumps(result, default=str),
            (result.get('btc') or {}).get('regime_score'),
            (result.get('btc') or {}).get('price'),
            (result.get('btc') or {}).get('change_h1'),
            (result.get('btc') or {}).get('change_h24'),
        ))


def best_alert_score_within(symbol: str, cooldown_minutes: int):
    """
    Highest score this symbol alerted with inside the cooldown window, or None
    if it hasn't alerted. MAX (not latest) so a slowly climbing score can't
    chain-fire an alert every scan.
    """
    with get_conn() as conn:
        row = conn.execute('''
            SELECT MAX(score) AS best FROM scanner_alerts
            WHERE symbol = ?
              AND created_at >= datetime('now', ?)
        ''', (symbol, f'-{int(cooldown_minutes)} minutes')).fetchone()
    return row['best'] if row and row['best'] is not None else None


def get_recent_alerts(limit: int = 50) -> list:
    with get_conn() as conn:
        rows = conn.execute('''
            SELECT id, symbol, score, label, direction, price_usd, vol_pace_1h,
                   price_change_1h, liquidity_usd, chain, url, summary, created_at
            FROM scanner_alerts
            ORDER BY created_at DESC LIMIT ?
        ''', (limit,)).fetchall()
    return [dict(r) for r in rows]


# ── Alert outcomes (forward returns, backfilled from OHLCV) ─────────────────

def get_alerts_needing_outcome(limit: int = 200, min_age_hours: float = 0.3) -> list:
    """
    Alerts with no outcome row yet, or an incomplete one that's now old enough
    to retry (e.g. it was backfilled before 24h had elapsed). Oldest first, so
    a partial backfill run always makes forward progress.
    """
    with get_conn() as conn:
        rows = conn.execute('''
            SELECT a.id, a.symbol, a.chain, a.url, a.pair_address, a.direction,
                   a.score, a.label, a.price_usd, a.created_at
            FROM scanner_alerts a
            LEFT JOIN alert_outcomes o ON o.alert_id = a.id
            WHERE a.created_at <= datetime('now', ?)
              AND (o.alert_id IS NULL OR o.data_complete = 0
                   OR (o.data_complete = 1 AND o.price_24h IS NULL))
            ORDER BY a.created_at ASC
            LIMIT ?
        ''', (f'-{min_age_hours} hours', limit)).fetchall()
    return [dict(r) for r in rows]


def upsert_alert_outcome(alert_id: int, fields: dict):
    cols = ['alert_id', 'computed_at'] + list(fields.keys())
    vals = [alert_id, datetime.utcnow().isoformat()] + list(fields.values())
    placeholders = ', '.join('?' for _ in cols)
    updates = ', '.join(f'{c} = excluded.{c}' for c in cols if c != 'alert_id')
    with get_conn() as conn:
        conn.execute(f'''
            INSERT INTO alert_outcomes ({', '.join(cols)})
            VALUES ({placeholders})
            ON CONFLICT(alert_id) DO UPDATE SET {updates}
        ''', vals)


def get_alerts_for_digest() -> list:
    """
    Alerts whose 24h outcome window has completed and that haven't been
    reported in a daily digest yet. All directions — the digest itself decides
    what to show, but everything returned here gets marked sent afterward.
    """
    with get_conn() as conn:
        rows = conn.execute('''
            SELECT a.id, a.symbol, a.score, a.label, a.direction, a.chain,
                   a.url, a.pair_address, a.created_at, a.btc_regime_score,
                   o.price_at_alert, o.price_24h, o.high_24h, o.high_24h_at,
                   o.low_24h, o.btc_change_window
            FROM scanner_alerts a
            JOIN alert_outcomes o ON o.alert_id = a.id
            WHERE o.data_complete = 1 AND o.digest_sent = 0
            ORDER BY a.created_at ASC
        ''').fetchall()
    return [dict(r) for r in rows]


def mark_digest_sent(alert_ids: list):
    if not alert_ids:
        return
    with get_conn() as conn:
        conn.executemany(
            'UPDATE alert_outcomes SET digest_sent = 1 WHERE alert_id = ?',
            [(i,) for i in alert_ids],
        )


def get_alerts_with_outcomes(limit: int = 2000) -> list:
    """Every alert that has a (possibly partial) outcome, richest fields first."""
    with get_conn() as conn:
        rows = conn.execute('''
            SELECT
                a.id, a.symbol, a.score, a.label, a.direction, a.price_usd,
                a.vol_pace_1h, a.liquidity_usd, a.chain, a.created_at,
                json_extract(a.payload, '$.alert_path')            AS alert_path,
                json_extract(a.payload, '$.metrics.vol_pace_5m')   AS vol_pace_5m,
                json_extract(a.payload, '$.price_change.m5')       AS price_change_5m,
                json_extract(a.payload, '$.wash_warning')          AS wash_warning,
                o.price_at_alert, o.price_1h_before, o.price_15m, o.price_1h,
                o.price_4h, o.price_24h, o.high_1h, o.low_1h, o.high_24h,
                o.low_24h, o.data_complete
            FROM scanner_alerts a
            JOIN alert_outcomes o ON o.alert_id = a.id
            ORDER BY a.created_at DESC
            LIMIT ?
        ''', (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_recent_runs(symbol: str, horizon: Optional[str] = None, limit: int = 20) -> list:
    with get_conn() as conn:
        if horizon:
            rows = conn.execute('''
                SELECT * FROM analysis_runs
                WHERE symbol = ? AND horizon = ?
                ORDER BY created_at DESC LIMIT ?
            ''', (symbol, horizon, limit)).fetchall()
        else:
            rows = conn.execute('''
                SELECT * FROM analysis_runs
                WHERE symbol = ?
                ORDER BY created_at DESC LIMIT ?
            ''', (symbol, limit)).fetchall()
    return [dict(r) for r in rows]


# ── Smart money: wallets, whale trades, outcomes ─────────────────────────────

# A buy is a WIN when the price was up ≥ win_1h% one hour later OR ≥ win_24h%
# a day later. Used both for auto-qualification and the leaderboard.
_WIN_CASE = ('(o.ret_1h_pct >= :w1 OR COALESCE(o.ret_24h_pct, -999999) >= :w24)')

# Wallets we treat as smart money: manually tracked, plus any wallet whose
# scored buys clear the win-rate bar.
_TRACKED_CTE = f'''
    WITH qualified AS (
        SELECT t.wallet, t.chain,
               COUNT(*) AS scored,
               AVG(CASE WHEN {_WIN_CASE} THEN 1.0 ELSE 0.0 END) AS win_rate
        FROM wallet_trades t
        JOIN wallet_trade_outcomes o ON o.trade_id = t.id
        WHERE t.side = 'buy' AND o.ret_1h_pct IS NOT NULL
        GROUP BY t.wallet, t.chain
        HAVING scored >= :min_buys AND win_rate >= :min_wr
    ),
    tracked AS (
        SELECT address AS wallet, chain, label, source
        FROM smart_wallets WHERE is_active = 1
        UNION
        SELECT q.wallet, q.chain, NULL AS label, 'auto' AS source
        FROM qualified q
        WHERE NOT EXISTS (
            SELECT 1 FROM smart_wallets s
            WHERE s.address = q.wallet AND (s.chain = q.chain OR s.chain = '')
        )
    )
'''


def _qualify_params(qualify: dict) -> dict:
    return {
        'w1':       qualify['win_ret_1h_pct'],
        'w24':      qualify['win_ret_24h_pct'],
        'min_buys': qualify['min_scored_buys'],
        'min_wr':   qualify['min_win_rate'],
    }


def record_wallet_trades(trades: list, chain: str, pool_address: str,
                         symbol: str) -> int:
    """Insert whale trades (deduped by trade_key). Returns how many were new."""
    added = 0
    with get_conn() as conn:
        for t in trades:
            cur = conn.execute('''
                INSERT OR IGNORE INTO wallet_trades
                  (trade_key, tx_hash, wallet, chain, pool_address, symbol,
                   side, amount_usd, price_usd, block_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (t['trade_key'], t.get('tx_hash'), t['wallet'], chain,
                  pool_address, symbol, t['side'], t.get('amount_usd'),
                  t.get('price_usd'), t['block_ts']))
            added += cur.rowcount
    return added


def add_smart_wallet(address: str, chain: str = '', label: str = None,
                     source: str = 'manual'):
    address = (address or '').strip()
    if address.startswith('0x'):
        address = address.lower()
    with get_conn() as conn:
        conn.execute('''
            INSERT INTO smart_wallets (address, chain, label, source, is_active)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(address, chain) DO UPDATE SET
                label = COALESCE(excluded.label, label),
                is_active = 1
        ''', (address, (chain or '').lower(), label, source))


def remove_smart_wallet(address: str, chain: str = ''):
    with get_conn() as conn:
        conn.execute('DELETE FROM smart_wallets WHERE address = ? AND chain = ?',
                     ((address or '').strip(), (chain or '').lower()))


def get_tracked_wallets(qualify: dict) -> dict:
    """
    {wallet_address: {'chain', 'label', 'source', 'win_rate'?}} for every
    wallet currently considered smart money (manual + auto-qualified).
    """
    with get_conn() as conn:
        rows = conn.execute(f'''
            {_TRACKED_CTE}
            SELECT wallet, chain, label, source FROM tracked
        ''', _qualify_params(qualify)).fetchall()
    return {r['wallet']: {'chain': r['chain'], 'label': r['label'],
                          'source': r['source']} for r in rows}


def smart_money_feed(qualify: dict, limit: int = 100) -> list:
    """Recent movements (all trades) of tracked/qualified wallets, newest first."""
    params = {**_qualify_params(qualify), 'limit': limit}
    with get_conn() as conn:
        rows = conn.execute(f'''
            {_TRACKED_CTE}
            SELECT tr.id, tr.wallet, tr.chain, tr.pool_address, tr.symbol,
                   tr.side, tr.amount_usd, tr.price_usd, tr.block_ts,
                   tk.label, tk.source,
                   o.ret_1h_pct, o.ret_24h_pct
            FROM wallet_trades tr
            JOIN tracked tk
              ON tk.wallet = tr.wallet AND (tk.chain = tr.chain OR tk.chain = '')
            LEFT JOIN wallet_trade_outcomes o ON o.trade_id = tr.id
            ORDER BY tr.block_ts DESC
            LIMIT :limit
        ''', params).fetchall()
    return [dict(r) for r in rows]


def wallet_leaderboard(qualify: dict, min_buys: int = 2, limit: int = 200) -> list:
    """
    Every whale wallet seen, aggregated: buys, scored buys, wins, win rate,
    average forward returns, volume, last seen. Sorted best-first.
    """
    params = {**_qualify_params(qualify), 'min_display_buys': min_buys,
              'limit': limit}
    with get_conn() as conn:
        rows = conn.execute(f'''
            SELECT t.wallet, t.chain,
                   COUNT(*)                                        AS trades,
                   SUM(CASE WHEN t.side = 'buy' THEN 1 ELSE 0 END) AS buys,
                   SUM(t.amount_usd)                               AS total_usd,
                   COUNT(DISTINCT t.pool_address)                  AS pools,
                   MAX(t.block_ts)                                 AS last_seen,
                   SUM(CASE WHEN t.side = 'buy' AND o.ret_1h_pct IS NOT NULL
                            THEN 1 ELSE 0 END)                     AS scored_buys,
                   SUM(CASE WHEN t.side = 'buy' AND o.ret_1h_pct IS NOT NULL
                            AND {_WIN_CASE} THEN 1 ELSE 0 END)     AS wins,
                   AVG(CASE WHEN t.side = 'buy' THEN o.ret_1h_pct END)  AS avg_ret_1h,
                   AVG(CASE WHEN t.side = 'buy' THEN o.ret_24h_pct END) AS avg_ret_24h,
                   MAX(s.label)     AS label,
                   MAX(s.source)    AS manual_source,
                   MAX(s.is_active) AS manually_tracked
            FROM wallet_trades t
            LEFT JOIN wallet_trade_outcomes o ON o.trade_id = t.id
            LEFT JOIN smart_wallets s
              ON s.address = t.wallet AND (s.chain = t.chain OR s.chain = '')
            GROUP BY t.wallet, t.chain
            HAVING buys >= :min_display_buys OR manually_tracked = 1
            ORDER BY wins DESC, scored_buys DESC, total_usd DESC
            LIMIT :limit
        ''', params).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        scored = d['scored_buys'] or 0
        d['win_rate'] = round((d['wins'] or 0) / scored, 3) if scored else None
        d['is_smart'] = bool(d['manually_tracked']) or (
            scored >= qualify['min_scored_buys']
            and (d['win_rate'] or 0) >= qualify['min_win_rate']
        )
        out.append(d)
    return out


def get_wallet_trades(address: str, limit: int = 100) -> list:
    """One wallet's recorded trades with outcomes, newest first."""
    with get_conn() as conn:
        rows = conn.execute('''
            SELECT t.id, t.wallet, t.chain, t.pool_address, t.symbol, t.side,
                   t.amount_usd, t.price_usd, t.block_ts,
                   o.ret_1h_pct, o.ret_24h_pct, o.data_complete
            FROM wallet_trades t
            LEFT JOIN wallet_trade_outcomes o ON o.trade_id = t.id
            WHERE t.wallet = ?
            ORDER BY t.block_ts DESC LIMIT ?
        ''', ((address or '').strip(), limit)).fetchall()
    return [dict(r) for r in rows]


def get_trades_needing_outcome(min_age_minutes: int, limit: int = 400) -> list:
    """
    Whale BUYS old enough to have an observable 1h return but no complete
    outcome yet. Oldest first; the backfill groups them by pool so one candle
    fetch scores every pending trade on that pool.
    """
    with get_conn() as conn:
        rows = conn.execute('''
            SELECT t.id, t.wallet, t.chain, t.pool_address, t.symbol,
                   t.price_usd, t.block_ts
            FROM wallet_trades t
            LEFT JOIN wallet_trade_outcomes o ON o.trade_id = t.id
            WHERE t.side = 'buy'
              AND t.block_ts <= datetime('now', ?)
              AND (o.trade_id IS NULL OR o.data_complete = 0)
            ORDER BY t.block_ts ASC
            LIMIT ?
        ''', (f'-{int(min_age_minutes)} minutes', limit)).fetchall()
    return [dict(r) for r in rows]


def upsert_wallet_trade_outcome(trade_id: int, fields: dict):
    cols = ['trade_id', 'computed_at'] + list(fields.keys())
    vals = [trade_id, datetime.utcnow().isoformat()] + list(fields.values())
    placeholders = ', '.join('?' for _ in cols)
    updates = ', '.join(f'{c} = excluded.{c}' for c in cols if c != 'trade_id')
    with get_conn() as conn:
        conn.execute(f'''
            INSERT INTO wallet_trade_outcomes ({', '.join(cols)})
            VALUES ({placeholders})
            ON CONFLICT(trade_id) DO UPDATE SET {updates}
        ''', vals)


def smart_money_overview(qualify: dict) -> dict:
    with get_conn() as conn:
        trades = conn.execute('SELECT COUNT(*) AS n, COUNT(DISTINCT wallet) AS w '
                              'FROM wallet_trades').fetchone()
        scored = conn.execute('SELECT COUNT(*) AS n FROM wallet_trade_outcomes '
                              'WHERE ret_1h_pct IS NOT NULL').fetchone()
        manual = conn.execute('SELECT COUNT(*) AS n FROM smart_wallets '
                              'WHERE is_active = 1').fetchone()
        auto = conn.execute(f'''
            {_TRACKED_CTE}
            SELECT COUNT(*) AS n FROM tracked WHERE source = 'auto'
        ''', _qualify_params(qualify)).fetchone()
    return {
        'trades_recorded': trades['n'],
        'wallets_seen':    trades['w'],
        'buys_scored':     scored['n'],
        'manual_wallets':  manual['n'],
        'auto_qualified':  auto['n'],
    }


# ── Shared scanner state (cross-worker) ──────────────────────────────────────

def get_state(key: str, default=None):
    with get_conn() as conn:
        row = conn.execute(
            'SELECT value FROM scanner_state WHERE key = ?', (key,)
        ).fetchone()
    return row['value'] if row else default


def set_state(key: str, value):
    with get_conn() as conn:
        conn.execute(
            'INSERT INTO scanner_state (key, value) VALUES (?, ?) '
            'ON CONFLICT(key) DO UPDATE SET value = excluded.value',
            (key, str(value)),
        )


def is_monitor_enabled() -> bool:
    return get_state('monitor_enabled', '0') == '1'


def set_monitor_enabled(on: bool):
    set_state('monitor_enabled', '1' if on else '0')


def set_monitor_last_run(at: str, summary: dict, results: list):
    set_state('monitor_last_run', json.dumps(
        {'at': at, 'summary': summary, 'results': results}
    ))


def get_monitor_last_run() -> dict:
    raw = get_state('monitor_last_run')
    if not raw:
        return {'at': None, 'summary': None, 'results': []}
    try:
        d = json.loads(raw)
        return {'at': d.get('at'), 'summary': d.get('summary'),
                'results': d.get('results', [])}
    except Exception:
        return {'at': None, 'summary': None, 'results': []}
