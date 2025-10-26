"""
Signal Validation Window Optimizer
Finds optimal validation windows for each signal-timeframe combination
Separate from fact-checking - this determines HOW LONG to wait before validating
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from scalp_signal_analyzer import ScalpSignalAnalyzer
import requests
import pandas as pd
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(level=logging.INFO)


class SignalValidationOptimizer:
  """
  Finds optimal validation windows for each signal-timeframe combination
  by testing different windows and finding which gives best price movement
  """

  # Maximum validation windows per timeframe (conservative estimates)
  # These are the UPPER BOUNDS for testing
  MAX_VALIDATION_WINDOWS = {
    '1m': 15,  # Max 15 candles = 15 minutes
    '3m': 15,  # Max 15 candles = 45 minutes
    '5m': 15,  # Max 15 candles = 75 minutes
    '15m': 12,  # Max 12 candles = 3 hours
    '30m': 10,  # Max 10 candles = 5 hours
    '1h': 12,  # Max 12 candles = 12 hours
    '2h': 10,  # Max 10 candles = 20 hours
    '4h': 8,  # Max 8 candles = 32 hours
    '6h': 8,  # Max 8 candles = 48 hours
    '8h': 6,  # Max 6 candles = 48 hours
    '12h': 6,  # Max 6 candles = 72 hours
    '1d': 7,  # Max 7 candles = 7 days
    '3d': 5,  # Max 5 candles = 15 days
    '1w': 4,  # Max 4 candles = 4 weeks
  }
  valid_tfs = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']

  MIN_PROFIT_THRESHOLD_PCT = 0.1  # Same as fact-checker

  def __init__(self, db_path: str = 'crypto_signals.db'):
    self.db_path = db_path
    self.session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
      pool_connections=20,
      pool_maxsize=20,
      max_retries=3
    )
    self.session.mount('http://', adapter)
    self.session.mount('https://', adapter)

    self.timeframe_minutes = {
      '1m': 1, '3m': 3, '5m': 5, '15m': 15,
      '30m': 30, '1h': 60, '2h': 120, '4h': 240,
      '6h': 360, '8h': 480, '12h': 720, '1d': 1440,
      '3d': 4320, '1w': 10080
    }

    self.init_signals_table()

  def init_signals_table(self):
    """Create and populate signals table with all signal definitions"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Create signals table
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_name TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                category TEXT NOT NULL,
                initial_validation_window INTEGER NOT NULL,
                validation_window INTEGER NOT NULL,
                max_validation_window INTEGER NOT NULL,
                initial_signal_accuracy REAL NOT NULL,
                signal_accuracy REAL NOT NULL,
                sample_size INTEGER DEFAULT 0,
                last_optimized TIMESTAMP,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(signal_name, timeframe)
            )
        ''')

    cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_signals_lookup 
            ON signals(signal_name, timeframe)
        ''')

    conn.commit()

    # Check if table is already populated
    cursor.execute('SELECT COUNT(*) FROM signals')
    if cursor.fetchone()[0] > 0:
      logging.info("✅ Signals table already populated")
      conn.close()
      return

    # Populate with all signals
    logging.info("📊 Populating signals table...")

    signal_categories = self._categorize_all_signals()

    for signal_name, signal_info in ScalpSignalAnalyzer.SIGNAL_CONFIDENCE.items():
      confidence = signal_info['confidence']
      category = signal_categories.get(signal_name, 'Other')
      description = self._get_signal_description(signal_name)

      # Insert for each timeframe
      for timeframe in self.valid_tfs:
        initial_window = self._estimate_initial_validation_window(
          signal_name, timeframe, category
        )
        max_window = self.MAX_VALIDATION_WINDOWS.get(timeframe, 10)

        cursor.execute('''
                    INSERT OR IGNORE INTO signals (
                        signal_name, timeframe, category,
                        initial_validation_window, validation_window,
                        max_validation_window,
                        initial_signal_accuracy, signal_accuracy,
                        description
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
          signal_name, timeframe, category,
          initial_window, initial_window, max_window,
          confidence, confidence, description
        ))

    conn.commit()
    conn.close()

    logging.info("✅ Signals table populated with all signal-timeframe combinations")

  def _categorize_all_signals(self) -> Dict[str, str]:
    """Categorize all signals"""
    categories = {}

    for signal_name in ScalpSignalAnalyzer.SIGNAL_CONFIDENCE.keys():
      signal_lower = signal_name.lower()

      if any(x in signal_lower for x in
             ['engulfing', 'hammer', 'star', 'doji', 'marubozu', 'soldiers', 'crows', 'piercing', 'tweezer']):
        categories[signal_name] = 'Candlestick Patterns'
      elif any(x in signal_lower for x in ['ma_', 'ema_', 'ribbon', 'moving_average']):
        categories[signal_name] = 'Moving Averages'
      elif any(x in signal_lower for x in ['rsi', 'macd', 'stoch', 'cci', 'williams', 'momentum']):
        categories[signal_name] = 'Momentum Indicators'
      elif any(x in signal_lower for x in ['volume', 'obv', 'vwap', 'accumulation']):
        categories[signal_name] = 'Volume Analysis'
      elif any(x in signal_lower for x in ['bollinger', 'atr', 'keltner', 'volatility']):
        categories[signal_name] = 'Volatility Indicators'
      elif any(x in signal_lower for x in ['adx', 'supertrend', 'sar', 'ichimoku', 'parabolic']):
        categories[signal_name] = 'Trend Indicators'
      elif any(x in signal_lower for x in
               ['support', 'resistance', 'structure', 'pivot', 'fibonacci', 'break', 'bounce', 'higher', 'lower']):
        categories[signal_name] = 'Price Action & Structure'
      else:
        categories[signal_name] = 'Other'

    return categories

  def _get_signal_description(self, signal_name: str) -> str:
    """Get signal description"""
    descriptions = {
      'engulfing_bullish': 'Bullish engulfing pattern - strong bullish reversal when large bullish candle engulfs previous bearish candle. Best at support levels.',
      'engulfing_bearish': 'Bearish engulfing pattern - strong bearish reversal when large bearish candle engulfs previous bullish candle. Best at resistance levels.',
      'hammer': 'Hammer pattern - bullish reversal with long lower shadow, small body. Indicates rejection of lower prices.',
      'shooting_star': 'Shooting star - bearish reversal with long upper shadow. Indicates rejection of higher prices.',
      'doji_reversal': 'Doji candle - indecision pattern with near-equal open/close. Potential reversal when at extremes.',
      'morning_star': 'Morning star - strong bullish reversal pattern with 3 candles. Highly reliable at bottoms.',
      'evening_star': 'Evening star - strong bearish reversal pattern with 3 candles. Highly reliable at tops.',
      'three_white_soldiers': 'Three white soldiers - powerful bullish continuation with 3 consecutive large bullish candles.',
      'three_black_crows': 'Three black crows - powerful bearish continuation with 3 consecutive large bearish candles.',
      'ma_cross_golden': 'Golden cross - 50 MA crosses above 200 MA. Major bullish signal indicating long-term trend change.',
      'ma_cross_death': 'Death cross - 50 MA crosses below 200 MA. Major bearish signal indicating long-term trend change.',
      'rsi_oversold': 'RSI below 30 - oversold condition. Price may bounce but needs confirmation.',
      'rsi_overbought': 'RSI above 70 - overbought condition. Price may retrace but needs confirmation.',
      'rsi_divergence_bullish': 'Bullish RSI divergence - price makes lower low but RSI makes higher low. Very strong reversal signal.',
      'rsi_divergence_bearish': 'Bearish RSI divergence - price makes higher high but RSI makes lower high. Very strong reversal signal.',
      'macd_cross_bullish': 'MACD bullish crossover - MACD line crosses above signal line. Momentum shift to bullish.',
      'macd_cross_bearish': 'MACD bearish crossover - MACD line crosses below signal line. Momentum shift to bearish.',
      'macd_divergence_bullish': 'Bullish MACD divergence - highest confidence reversal signal. Hidden buying strength.',
      'macd_divergence_bearish': 'Bearish MACD divergence - highest confidence reversal signal. Hidden selling strength.',
      'volume_spike_bullish': 'Volume spike on bullish candle - strong buying interest. Confirms bullish moves.',
      'volume_spike_bearish': 'Volume spike on bearish candle - strong selling pressure. Confirms bearish moves.',
      'bollinger_squeeze': 'Bollinger Band squeeze - extreme low volatility. Major breakout imminent in either direction.',
      'supertrend_bullish': 'SuperTrend indicator bullish - confirmed uptrend with trailing stop. High confidence trend signal.',
      'supertrend_bearish': 'SuperTrend indicator bearish - confirmed downtrend with trailing stop. High confidence trend signal.',
      'break_of_structure_bullish': 'Bullish break of structure - price breaks above lower high in downtrend. Trend reversal confirmed.',
      'break_of_structure_bearish': 'Bearish break of structure - price breaks below higher low in uptrend. Trend reversal confirmed.',
      'support_bounce': 'Price bounces off support level - rejection of lower prices at key level.',
      'resistance_rejection': 'Price rejected at resistance level - rejection of higher prices at key level.',
      'support_break': 'Price breaks below support - bearish signal as key level fails.',
      'resistance_break': 'Price breaks above resistance - bullish signal as key level breaks.',
    }

    return descriptions.get(signal_name, f'Technical analysis signal: {signal_name}')

  def _estimate_initial_validation_window(self, signal_name: str,
                                          timeframe: str, category: str) -> int:
    """Estimate initial validation window based on signal characteristics"""

    # Base windows by timeframe
    base_windows = {
      '1m': 10, '3m': 10, '5m': 12, '15m': 8,
      '30m': 8, '1h': 8, '2h': 6, '4h': 6,
      '6h': 8, '8h': 6, '12h': 8, '1d': 5,
      '3d': 5, '1w': 4,
    }

    base = base_windows.get(timeframe, 6)

    # Adjust by category
    if category == 'Candlestick Patterns':
      # Candles patterns typically resolve quickly
      return max(3, base - 2)
    elif category in ['Momentum Indicators', 'Volume Analysis']:
      # Momentum signals need more time
      return base
    elif category == 'Trend Indicators':
      # Trend signals need longer validation
      return min(self.MAX_VALIDATION_WINDOWS.get(timeframe, 10), base + 2)
    elif category == 'Volatility Indicators':
      # Volatility signals are fast
      return max(3, base - 1)
    elif category in ['Moving Averages', 'Price Action & Structure']:
      return base

    return base

  def fetch_price_journey(self, symbol: str, timestamp: datetime,
                          timeframe: str, candles_ahead: int) -> Optional[List[Dict]]:
    """Fetch price data for validation window testing (uses Binance then KuCoin fallback)"""
    minutes = self.timeframe_minutes.get(timeframe, 60)
    target_time = timestamp + timedelta(minutes=minutes * (candles_ahead + 2))

    # Try Binance first (higher rate limit)
    data = self._fetch_binance_journey(symbol, timestamp, target_time, timeframe, candles_ahead)
    if data:
      return data

    # Fallback to KuCoin
    data = self._fetch_kucoin_journey(symbol, timestamp, target_time, timeframe, candles_ahead)
    return data

  def _fetch_binance_journey(self, symbol: str, start_time: datetime,
                             end_time: datetime, timeframe: str,
                             candles_ahead: int) -> Optional[List[Dict]]:
    """Fetch from Binance"""
    try:
      symbol_pair = f"{symbol}USDT"
      url = "https://api.binance.com/api/v3/klines"

      params = {
        'symbol': symbol_pair,
        'interval': timeframe,
        'startTime': int(start_time.timestamp() * 1000),
        'endTime': int(end_time.timestamp() * 1000),
        'limit': candles_ahead + 5
      }

      response = self.session.get(url, params=params, timeout=10)
      response.raise_for_status()
      raw_candles = response.json()

      if not raw_candles or len(raw_candles) < candles_ahead + 1:
        return None

      candles = []
      for candle in raw_candles:
        candles.append({
          'timestamp': int(candle[0] / 1000),
          'open': float(candle[1]),
          'high': float(candle[2]),
          'low': float(candle[3]),
          'close': float(candle[4]),
          'volume': float(candle[5])
        })

      return candles

    except Exception as e:
      logging.debug(f"Binance fetch failed for {symbol}: {e}")
      return None

  def _fetch_kucoin_journey(self, symbol: str, start_time: datetime,
                            end_time: datetime, timeframe: str,
                            candles_ahead: int) -> Optional[List[Dict]]:
    """Fetch from KuCoin"""
    try:
      symbol_pair = f"{symbol}-USDT"
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
        'startAt': int(start_time.timestamp()),
        'endAt': int(end_time.timestamp())
      }

      response = self.session.get(url, params=params, timeout=10)
      response.raise_for_status()
      data = response.json()

      if data.get('code') != '200000' or not data.get('data'):
        return None

      raw_candles = data['data']

      candles = []
      for candle in reversed(raw_candles):
        candles.append({
          'timestamp': int(candle[0]),
          'open': float(candle[1]),
          'close': float(candle[2]),
          'high': float(candle[3]),
          'low': float(candle[4]),
          'volume': float(candle[5])
        })

      if len(candles) < candles_ahead + 1:
        return None

      return candles

    except Exception as e:
      logging.debug(f"KuCoin fetch failed for {symbol}: {e}")
      return None

  def find_best_validation_window(self, signal_name: str, signal_type: str,
                                  timeframe: str, detected_at: datetime,
                                  entry_price: float, symbol: str) -> Optional[Dict]:
    """
    Test different validation windows and find the one with highest price movement
    Only considers SUCCESSFUL signals (those that reached profit threshold)
    """
    max_window = self.MAX_VALIDATION_WINDOWS.get(timeframe, 10)

    # Fetch enough candles for testing all windows
    candles = self.fetch_price_journey(symbol, detected_at, timeframe, max_window)

    if not candles or len(candles) < 3:
      return None

    best_window = None
    best_price_change = 0.0
    reached_profit = False

    # Test each window from 2 to max_window
    for window in range(2, min(max_window + 1, len(candles))):
      window_candles = candles[:window + 1]

      # Calculate max price movement in signal direction
      if signal_type == 'BUY':
        # Find highest price in window
        max_price = max(c['high'] for c in window_candles[1:])
        price_change = ((max_price - entry_price) / entry_price) * 100
      else:  # SELL
        # Find lowest price in window
        min_price = min(c['low'] for c in window_candles[1:])
        price_change = ((entry_price - min_price) / entry_price) * 100

      # Only consider if it reached profit threshold
      if price_change >= self.MIN_PROFIT_THRESHOLD_PCT:
        reached_profit = True
        if price_change > best_price_change:
          best_price_change = price_change
          best_window = window

    # If signal never reached profit threshold, skip it
    if not reached_profit:
      return None

    return {
      'best_window': best_window,
      'max_price_change': best_price_change,
      'tested_windows': max_window
    }

  def optimize_signal_timeframe(self, signal_name: str, timeframe: str,
                                limit: int = None, max_workers: int = 10) -> Dict:
    """
    Optimize validation window for specific signal-timeframe combination
    Processes signals from both live_signals and position_signals
    Skips already fact-checked signals
    """
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get signals from live_signals that haven't been fact-checked
    cursor.execute('''
            SELECT ls.*, 'live_signals' as source_table
            FROM live_signals ls
            LEFT JOIN signal_fact_checks sfc 
                ON ls.signal_name = sfc.signal_name 
                AND ls.timeframe = sfc.timeframe 
                AND ls.timestamp = sfc.detected_at
            WHERE ls.signal_name = ? 
                AND ls.timeframe = ?
                AND sfc.id IS NULL
            ORDER BY ls.timestamp DESC
        ''', (signal_name, timeframe))

    live_signals = [dict(row) for row in cursor.fetchall()]

    # Get signals from position_signals that haven't been fact-checked
    cursor.execute('''
            SELECT 
                ps.*,
                tp.symbol,
                'position_signals' as source_table
            FROM position_signals ps
            INNER JOIN trading_positions tp ON ps.position_id = tp.id
            LEFT JOIN signal_fact_checks sfc 
                ON ps.signal_name = sfc.signal_name 
                AND ps.timeframe = sfc.timeframe 
                AND ps.detected_at = sfc.detected_at
            WHERE ps.signal_name = ? 
                AND ps.timeframe = ?
                AND sfc.id IS NULL
            ORDER BY ps.detected_at DESC
        ''', (signal_name, timeframe))

    position_signals = [dict(row) for row in cursor.fetchall()]

    conn.close()

    # Combine all signals
    all_signals = live_signals + position_signals

    if limit:
      all_signals = all_signals[:limit]

    if not all_signals:
      logging.info(f"⚠️  No unchecked signals for {signal_name}[{timeframe}]")
      return {
        'signal_name': signal_name,
        'timeframe': timeframe,
        'tested_count': 0,
        'successful_signals': 0,
        'skipped_signals': 0,
        'best_window': None,
        'avg_price_change': 0
      }

    logging.info(f"🔍 Testing {len(all_signals)} signals for {signal_name}[{timeframe}]...")

    results = {
      'signal_name': signal_name,
      'timeframe': timeframe,
      'tested_count': 0,
      'successful_signals': 0,
      'skipped_signals': 0,
      'window_votes': defaultdict(int),
      'window_price_changes': defaultdict(list),
      'best_window': None,
      'avg_price_change': 0
    }

    # Process in parallel
    def process_signal(signal):
      # Normalize field names
      if signal['source_table'] == 'position_signals':
        timestamp = datetime.fromisoformat(signal['detected_at'])
        price = signal['price_at_detection']
      else:
        timestamp = datetime.fromisoformat(signal['timestamp'])
        price = signal['price']

      result = self.find_best_validation_window(
        signal['signal_name'],
        signal['signal_type'],
        signal['timeframe'],
        timestamp,
        price,
        signal['symbol']
      )

      return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
      futures = {executor.submit(process_signal, sig): sig for sig in all_signals}

      for future in as_completed(futures):
        results['tested_count'] += 1

        if results['tested_count'] % 100 == 0:
          logging.info(f"   Progress: {results['tested_count']}/{len(all_signals)}")

        result = future.result()

        if result:
          results['successful_signals'] += 1
          window = result['best_window']
          price_change = result['max_price_change']

          results['window_votes'][window] += 1
          results['window_price_changes'][window].append(price_change)
        else:
          results['skipped_signals'] += 1

    # Determine best window (most votes)
    if results['window_votes']:
      best_window = max(results['window_votes'].items(), key=lambda x: x[1])[0]
      results['best_window'] = best_window

      # Calculate average price change for best window
      if results['window_price_changes'][best_window]:
        results['avg_price_change'] = sum(results['window_price_changes'][best_window]) / len(
          results['window_price_changes'][best_window])

      # Save to database
      self._update_signal_validation_window(signal_name, timeframe, best_window)

      logging.info(f"✅ {signal_name}[{timeframe}]: Best window = {best_window} candles")
      logging.info(f"   Successful: {results['successful_signals']}/{results['tested_count']}")
      logging.info(f"   Avg price change: {results['avg_price_change']:.2f}%")
    else:
      logging.warning(f"⚠️  No successful signals for {signal_name}[{timeframe}]")

    return results

  def _update_signal_validation_window(self, signal_name: str, timeframe: str,
                                       validation_window: int):
    """Update validation window in signals table"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            UPDATE signals 
            SET validation_window = ?,
                last_optimized = ?,
                updated_at = ?
            WHERE signal_name = ? AND timeframe = ?
        ''', (validation_window, datetime.now(), datetime.now(), signal_name, timeframe))

    conn.commit()
    conn.close()

  def optimize_all_signals(self, limit_per_signal: int = None,
                           max_workers: int = 10) -> Dict:
    """
    Optimize validation windows for ALL signal-timeframe combinations
    """
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Get all signal-timeframe combinations that have data
    cursor.execute('''
            SELECT DISTINCT signal_name, timeframe
            FROM (
                SELECT signal_name, timeframe FROM live_signals
                UNION
                SELECT signal_name, timeframe FROM position_signals
            )
            ORDER BY signal_name, timeframe
        ''')

    combinations = cursor.fetchall()
    conn.close()

    total = len(combinations)
    logging.info(f"\n{'=' * 80}")
    logging.info(f"OPTIMIZING VALIDATION WINDOWS FOR ALL SIGNALS")
    logging.info(f"{'=' * 80}")
    logging.info(f"Total combinations to process: {total}\n")

    results = {
      'total_combinations': total,
      'processed': 0,
      'successful': 0,
      'no_data': 0,
      'details': []
    }

    start_time = time.time()

    for idx, (signal_name, timeframe) in enumerate(combinations, 1):
      logging.info(f"\n[{idx}/{total}] Processing {signal_name}[{timeframe}]...")

      result = self.optimize_signal_timeframe(
        signal_name, timeframe,
        limit=limit_per_signal,
        max_workers=max_workers
      )

      results['processed'] += 1
      results['details'].append(result)

      if result['best_window'] is not None:
        results['successful'] += 1
      else:
        results['no_data'] += 1

      # Progress update
      elapsed = time.time() - start_time
      avg_time = elapsed / results['processed']
      eta = avg_time * (total - results['processed'])

      logging.info(f"\n📊 Overall Progress: {results['processed']}/{total} ({results['processed'] / total * 100:.1f}%)")
      logging.info(f"   Successful: {results['successful']}, No data: {results['no_data']}")
      logging.info(f"   ETA: {eta / 60:.1f} minutes\n")

    total_time = time.time() - start_time

    logging.info(f"\n{'=' * 80}")
    logging.info(f"OPTIMIZATION COMPLETE")
    logging.info(f"{'=' * 80}")
    logging.info(f"Total time: {total_time / 60:.1f} minutes")
    logging.info(f"Processed: {results['processed']}/{total}")
    logging.info(f"Successful: {results['successful']}")
    logging.info(f"No data: {results['no_data']}")

    return results

  def get_signal_validation_window(self, signal_name: str, timeframe: str) -> int:
    """Get validation window for a signal-timeframe combination"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            SELECT validation_window FROM signals
            WHERE signal_name = ? AND timeframe = ?
        ''', (signal_name, timeframe))

    result = cursor.fetchone()
    conn.close()

    if result:
      return result[0]

    # Fallback to default if not found
    return 5

  def __del__(self):
    if hasattr(self, 'session'):
      self.session.close()


def run_validation_window_optimization(limit_per_signal: int = None):
  """
  Step 1: Optimize validation windows for all signals
  This finds the best validation window for each signal-timeframe combination
  """
  print("\n" + "=" * 80)
  print("STEP 1: VALIDATION WINDOW OPTIMIZATION")
  print("=" * 80)
  print("\nThis process will:")
  print("  1. Test different validation windows for each signal-timeframe")
  print("  2. Find the window that gives the highest price movement")
  print("  3. Update the signals table with optimal validation windows")
  print(
    f"\nProcessing {'all signals' if not limit_per_signal else f'up to {limit_per_signal} signals per combination'}...")
  print("=" * 80 + "\n")

  start_time = time.time()

  optimizer = SignalValidationOptimizer()

  # Run optimization
  results = optimizer.optimize_all_signals(
    limit_per_signal=limit_per_signal,
    max_workers=10
  )

  elapsed = time.time() - start_time

  print("\n" + "=" * 80)
  print("VALIDATION WINDOW OPTIMIZATION COMPLETE")
  print("=" * 80)
  print(f"Time taken: {elapsed / 60:.1f} minutes")
  print(f"Total combinations: {results['total_combinations']}")
  print(f"Successfully optimized: {results['successful']}")
  print(f"No data available: {results['no_data']}")
  print(f"\nValidation windows have been saved to the signals table")
  print("=" * 80 + "\n")

  return results


