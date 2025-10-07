"""
Signal Fact Checker
Validates historical signals and adjusts confidence ratings based on actual performance
"""

import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from scalp_signal_analyzer import ScalpSignalAnalyzer

logging.basicConfig(level=logging.INFO)


class SignalFactChecker:
  """Fact-check signals and adjust confidence ratings based on actual performance"""

  def __init__(self, db_path: str = 'crypto_signals.db'):
    self.db_path = db_path
    self.analyzer = ScalpSignalAnalyzer()
    self.timeframe_minutes = {
      '1m': 1, '3m': 3, '5m': 5, '15m': 15,
      '30m': 30, '1h': 60, '2h': 120, '4h': 240,
      '6h': 360, '8h': 480, '12h': 720, '1d': 1440,
      '3d': 4320, '1w': 10080
    }

  def fetch_historical_price(self, symbol: str, timestamp: datetime,
                             timeframe: str, candles_ahead: int = 5) -> Optional[float]:
    """Fetch price N candles after a given timestamp"""
    minutes = self.timeframe_minutes.get(timeframe, 60)
    target_time = timestamp + timedelta(minutes=minutes * candles_ahead)

    try:
      # Try KuCoin first
      symbol_pair = f"{symbol}-USDT"
      end_time = int((target_time + timedelta(hours=1)).timestamp())
      start_time = int(timestamp.timestamp())

      url = "https://api.kucoin.com/api/v1/market/candles"
      tf_map = {
        '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min',
        '30m': '30min', '1h': '1hour', '2h': '2hour', '4h': '4hour',
        '6h': '6hour', '8h': '8hour', '12h': '12hour', '1d': '1day',
        '1w': '1week'
      }

      params = {
        'symbol': symbol_pair,
        'type': tf_map.get(timeframe, '1hour'),
        'startAt': start_time,
        'endAt': end_time
      }

      response = requests.get(url, params=params, timeout=10)
      response.raise_for_status()
      data = response.json()

      if data.get('code') == '200000' and data.get('data'):
        candles = data['data']
        if len(candles) > candles_ahead:
          # KuCoin returns [timestamp, open, close, high, low, volume, turnover]
          return float(candles[candles_ahead][2])  # close price

    except Exception as e:
      logging.warning(f"KuCoin fetch failed: {e}")

    # Fallback to Binance
    try:
      symbol_pair = f"{symbol}USDT"
      url = "https://api.binance.com/api/v3/klines"

      params = {
        'symbol': symbol_pair,
        'interval': timeframe.replace('m', 'm').replace('h', 'h').replace('d', 'd').replace('w', 'w'),
        'startTime': int(timestamp.timestamp() * 1000),
        'endTime': int((target_time + timedelta(hours=1)).timestamp() * 1000),
        'limit': 20
      }

      response = requests.get(url, params=params, timeout=10)
      response.raise_for_status()
      candles = response.json()

      if len(candles) > candles_ahead:
        return float(candles[candles_ahead][4])  # close price

    except Exception as e:
      logging.error(f"Binance fetch failed: {e}")

    return None

  def fact_check_signal(self, signal_name: str, signal_type: str,
                        timeframe: str, detected_at: datetime,
                        price_at_detection: float, symbol: str,
                        candles_ahead: int = 5) -> Optional[Dict]:
    """Fact-check a single signal"""
    future_price = self.fetch_historical_price(
      symbol, detected_at, timeframe, candles_ahead
    )

    if future_price is None:
      return None

    price_change_pct = ((future_price - price_at_detection) / price_at_detection) * 100

    # Determine if prediction was correct
    if signal_type == 'BUY':
      predicted_correctly = price_change_pct > 0
      actual_move = 'UP' if price_change_pct > 0 else 'DOWN'
    elif signal_type == 'SELL':
      predicted_correctly = price_change_pct < 0
      actual_move = 'DOWN' if price_change_pct < 0 else 'UP'
    else:
      predicted_correctly = False
      actual_move = 'NEUTRAL'

    return {
      'signal_name': signal_name,
      'timeframe': timeframe,
      'detected_at': detected_at,
      'price_at_detection': price_at_detection,
      'actual_move': actual_move,
      'predicted_correctly': predicted_correctly,
      'price_change_pct': price_change_pct,
      'checked_at': datetime.now(),
      'candles_elapsed': candles_ahead
    }

  def save_fact_check(self, result: Dict):
    """Save fact-check result to database"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            INSERT INTO signal_fact_checks (
                signal_name, timeframe, detected_at, price_at_detection,
                actual_move, predicted_correctly, price_change_pct,
                checked_at, candles_elapsed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
      result['signal_name'], result['timeframe'],
      result['detected_at'], result['price_at_detection'],
      result['actual_move'], result['predicted_correctly'],
      result['price_change_pct'], result['checked_at'],
      result['candles_elapsed']
    ))

    conn.commit()
    conn.close()

  def fact_check_position_signals(self, position_id: int, symbol: str,
                                  candles_ahead: int = 5) -> Dict:
    """Fact-check all signals for a position"""
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get position signals
    cursor.execute('''
            SELECT * FROM position_signals
            WHERE position_id = ?
            ORDER BY detected_at DESC
        ''', (position_id,))

    signals = cursor.fetchall()
    conn.close()

    results = {
      'total_checked': 0,
      'correct_predictions': 0,
      'incorrect_predictions': 0,
      'accuracy': 0,
      'details': []
    }

    for signal in signals:
      result = self.fact_check_signal(
        signal['signal_name'],
        signal['signal_type'],
        signal['timeframe'],
        datetime.fromisoformat(signal['detected_at']),
        signal['price_at_detection'],
        symbol,
        candles_ahead
      )

      if result:
        self.save_fact_check(result)
        results['details'].append(result)
        results['total_checked'] += 1

        if result['predicted_correctly']:
          results['correct_predictions'] += 1
        else:
          results['incorrect_predictions'] += 1

    if results['total_checked'] > 0:
      results['accuracy'] = (results['correct_predictions'] / results['total_checked']) * 100

    logging.info(f"✅ Fact-checked {results['total_checked']} signals for position #{position_id}")
    logging.info(f"   Accuracy: {results['accuracy']:.1f}%")

    return results

  def calculate_signal_accuracy(self, signal_name: str, timeframe: str = None,
                                min_samples: int = 10) -> Optional[Dict]:
    """Calculate accuracy for a specific signal across all fact-checks"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    query = '''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN predicted_correctly = 1 THEN 1 ELSE 0 END) as correct,
                AVG(price_change_pct) as avg_price_change
            FROM signal_fact_checks
            WHERE signal_name = ?
        '''
    params = [signal_name]

    if timeframe:
      query += ' AND timeframe = ?'
      params.append(timeframe)

    cursor.execute(query, params)
    result = cursor.fetchone()
    conn.close()

    if not result or result[0] < min_samples:
      return None

    total, correct, avg_change = result
    accuracy = (correct / total) * 100

    return {
      'signal_name': signal_name,
      'timeframe': timeframe,
      'total_samples': total,
      'correct_predictions': correct,
      'accuracy': accuracy,
      'avg_price_change': avg_change or 0
    }

  def adjust_signal_confidence(self, signal_name: str, timeframe: str,
                               min_samples: int = 10) -> Optional[Dict]:
    """Adjust signal confidence based on fact-check accuracy"""
    accuracy_data = self.calculate_signal_accuracy(signal_name, timeframe, min_samples)

    if not accuracy_data:
      return None

    # Get original confidence
    original_conf = ScalpSignalAnalyzer.SIGNAL_CONFIDENCE.get(signal_name, {}).get('confidence', 70)

    # Calculate adjusted confidence
    # Weight: 70% original, 30% actual accuracy
    adjusted_conf = int(original_conf * 0.7 + accuracy_data['accuracy'] * 0.3)

    # Clamp between 0-100
    adjusted_conf = max(0, min(100, adjusted_conf))

    # Save adjustment
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            INSERT OR REPLACE INTO signal_confidence_adjustments (
                signal_name, timeframe, original_confidence, adjusted_confidence,
                accuracy_rate, sample_size, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
      signal_name, timeframe, original_conf, adjusted_conf,
      accuracy_data['accuracy'], accuracy_data['total_samples'],
      datetime.now()
    ))

    conn.commit()
    conn.close()

    logging.info(f"✅ Adjusted {signal_name} [{timeframe}]: {original_conf} → {adjusted_conf}")

    return {
      'signal_name': signal_name,
      'timeframe': timeframe,
      'original_confidence': original_conf,
      'adjusted_confidence': adjusted_conf,
      'accuracy_rate': accuracy_data['accuracy'],
      'sample_size': accuracy_data['total_samples'],
      'confidence_change': adjusted_conf - original_conf
    }

  def get_adjusted_confidence(self, signal_name: str, timeframe: str) -> int:
    """Get adjusted confidence for a signal, or original if no adjustment exists"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            SELECT adjusted_confidence FROM signal_confidence_adjustments
            WHERE signal_name = ? AND timeframe = ?
        ''', (signal_name, timeframe))

    result = cursor.fetchone()
    conn.close()

    if result:
      return result[0]

    # Return original confidence
    return ScalpSignalAnalyzer.SIGNAL_CONFIDENCE.get(signal_name, {}).get('confidence', 70)

  def bulk_adjust_all_signals(self, min_samples: int = 10) -> Dict:
    """Adjust confidence for all signals with sufficient samples"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Get all unique signal-timeframe combinations
    cursor.execute('''
            SELECT DISTINCT signal_name, timeframe
            FROM signal_fact_checks
        ''')

    combinations = cursor.fetchall()
    conn.close()

    results = {
      'total_processed': 0,
      'adjusted': 0,
      'skipped_insufficient_samples': 0,
      'adjustments': []
    }

    for signal_name, timeframe in combinations:
      results['total_processed'] += 1

      adjustment = self.adjust_signal_confidence(signal_name, timeframe, min_samples)

      if adjustment:
        results['adjusted'] += 1
        results['adjustments'].append(adjustment)
      else:
        results['skipped_insufficient_samples'] += 1

    logging.info(f"✅ Bulk adjustment complete: {results['adjusted']}/{results['total_processed']} signals adjusted")

    return results

  def get_all_adjustments(self) -> List[Dict]:
    """Get all signal confidence adjustments"""
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
            SELECT * FROM signal_confidence_adjustments
            ORDER BY last_updated DESC
        ''')

    adjustments = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return adjustments

  def cleanup_old_fact_checks(self, days: int = 90) -> int:
    """Remove old fact-check records"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            DELETE FROM signal_fact_checks
            WHERE checked_at < datetime('now', '-' || ? || ' days')
        ''', (days,))

    deleted = cursor.rowcount
    conn.commit()
    conn.close()

    logging.info(f"🗑️ Cleaned up {deleted} old fact-check records")
    return deleted
