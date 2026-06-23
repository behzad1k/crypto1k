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
from typing import Optional

DB_PATH = 'signals_v2.db'

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
                created_at    TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_alerts_symbol
                ON scanner_alerts(symbol, created_at DESC);
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
               price_change_1h, liquidity_usd, chain, url, summary, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            result.get('summary'),
            json.dumps(result, default=str),
        ))


def in_cooldown(symbol: str, cooldown_minutes: int) -> bool:
    """True if this symbol was alerted within the cooldown window."""
    with get_conn() as conn:
        row = conn.execute('''
            SELECT created_at FROM scanner_alerts
            WHERE symbol = ?
              AND created_at >= datetime('now', ?)
            ORDER BY created_at DESC LIMIT 1
        ''', (symbol, f'-{int(cooldown_minutes)} minutes')).fetchone()
    return row is not None


def get_recent_alerts(limit: int = 50) -> list:
    with get_conn() as conn:
        rows = conn.execute('''
            SELECT id, symbol, score, label, direction, price_usd, vol_pace_1h,
                   price_change_1h, liquidity_usd, chain, url, summary, created_at
            FROM scanner_alerts
            ORDER BY created_at DESC LIMIT ?
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
