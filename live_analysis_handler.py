"""
Database handler for live signal analysis
Stores real-time signal results with full details
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List
import logging


class LiveAnalysisDB:
  """Handle live analysis signal storage"""

  def __init__(self, db_path: str = 'crypto_signals.db'):
    self.db_path = db_path
    self.init_database()

  def init_database(self):
    """Create tables for live analysis"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Main signals table
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal_name TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                confidence INTEGER NOT NULL,
                strength TEXT,
                signal_value REAL,
                price REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, signal_name, timestamp)
            )
        ''')

    # Analysis runs table (track each analysis)
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframes TEXT NOT NULL,
                total_signals INTEGER,
                buy_signals INTEGER,
                sell_signals INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

    # Create indexes
    cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_live_symbol_tf 
            ON live_signals(symbol, timeframe, created_at DESC)
        ''')

    cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_analysis_symbol 
            ON analysis_runs(symbol, created_at DESC)
        ''')

    conn.commit()
    conn.close()
    logging.info("✅ Live analysis database initialized")

  def save_analysis_result(self, result: Dict):
    """Save complete analysis result"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    try:
      symbol = result['symbol']
      total_signals = 0
      total_buy = 0
      total_sell = 0

      # Save signals from each timeframe
      for tf, data in result['timeframes'].items():
        if 'error' in data:
          continue

        price = data['price']
        timestamp = data['timestamp']

        total_signals += data['signal_count']
        total_buy += data['buy_signals']
        total_sell += data['sell_signals']

        # Import signal confidence
        from scalp_signal_analyzer import ScalpSignalAnalyzer

        # Save each signal
        for signal_name, signal_data in data['signals'].items():
          signal_info = ScalpSignalAnalyzer.SIGNAL_CONFIDENCE.get(signal_name, {})
          confidence = signal_info.get('confidence', 0)

          cursor.execute('''
                        INSERT OR REPLACE INTO live_signals 
                        (symbol, timeframe, signal_name, signal_type, confidence, 
                         strength, signal_value, price, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
            symbol, tf, signal_name,
            signal_data.get('signal', 'UNKNOWN'),
            confidence,
            signal_data.get('strength'),
            signal_data.get('value'),
            price, timestamp
          ))

      # Save analysis run summary
      cursor.execute('''
                INSERT INTO analysis_runs 
                (symbol, timeframes, total_signals, buy_signals, sell_signals)
                VALUES (?, ?, ?, ?, ?)
            ''', (
        symbol,
        json.dumps(list(result['timeframes'].keys())),
        total_signals, total_buy, total_sell
      ))

      conn.commit()
      logging.info(f"✅ Saved analysis for {symbol}: {total_signals} signals")

    except Exception as e:
      logging.error(f"Failed to save analysis: {e}")
      conn.rollback()
    finally:
      conn.close()

  def get_latest_signals(self, symbol: str, timeframe: str = None, limit: int = 50) -> List[Dict]:
    """Get latest signals for a symbol"""
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = '''
            SELECT * FROM live_signals 
            WHERE symbol = ?
        '''
    params = [symbol]

    if timeframe:
      query += ' AND timeframe = ?'
      params.append(timeframe)

    query += ' ORDER BY created_at DESC LIMIT ?'
    params.append(limit)

    cursor.execute(query, params)
    signals = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return signals

  def get_analysis_summary(self, symbol: str, hours: int = 24) -> Dict:
    """Get analysis summary for past X hours"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            SELECT 
                COUNT(*) as total_analyses,
                SUM(total_signals) as total_signals,
                SUM(buy_signals) as buy_signals,
                SUM(sell_signals) as sell_signals,
                MAX(created_at) as last_analysis
            FROM analysis_runs
            WHERE symbol = ?
            AND created_at >= datetime('now', '-' || ? || ' hours')
        ''', (symbol, hours))

    row = cursor.fetchone()
    conn.close()

    if row:
      return {
        'total_analyses': row[0] or 0,
        'total_signals': row[1] or 0,
        'buy_signals': row[2] or 0,
        'sell_signals': row[3] or 0,
        'last_analysis': row[4]
      }

    return {
      'total_analyses': 0,
      'total_signals': 0,
      'buy_signals': 0,
      'sell_signals': 0,
      'last_analysis': None
    }

  def get_signal_history(self, symbol: str, signal_name: str, days: int = 7) -> List[Dict]:
    """Get historical occurrences of a specific signal"""
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
            SELECT * FROM live_signals
            WHERE symbol = ? AND signal_name = ?
            AND created_at >= datetime('now', '-' || ? || ' days')
            ORDER BY created_at DESC
        ''', (symbol, signal_name, days))

    history = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return history

  def cleanup_old_signals(self, days: int = 30):
    """Remove signals older than X days"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            DELETE FROM live_signals
            WHERE created_at < datetime('now', '-' || ? || ' days')
        ''', (days,))

    cursor.execute('''
            DELETE FROM analysis_runs
            WHERE created_at < datetime('now', '-' || ? || ' days')
        ''', (days,))

    conn.commit()
    deleted = cursor.rowcount
    conn.close()

    logging.info(f"🗑️ Cleaned up {deleted} old records")
    return deleted