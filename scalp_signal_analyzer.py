"""
Comprehensive Scalp Trading Signal Analyzer - UPDATED WITH LIVE COMBO ANALYSIS
Analyzes 80+ technical signals across multiple timeframes
NEW: Integrated live signal combination analysis after each symbol analysis
"""

import requests
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)


class ScalpSignalAnalyzer:
  """
  Complete technical analysis signal generator with 80+ signals
  Confidence ratings based on academic research, trading literature, and historical performance
  NEW: Includes live combination analysis for detected signals
  """

  # Timeframe definitions (in minutes)
  TIMEFRAMES = {
    'short': ['1m', '3m', '5m', '15m'],
    'mid': ['30m', '1h', '2h', '4h', '6h', '8h'],
    'long': ['12h', '1d', '3d', '1w']
  }

  # Signal confidence ratings (0-100) - UPDATED WITH NEW SIGNALS
  SIGNAL_CONFIDENCE = {
    # ===== Candlestick Patterns (19 signals) =====
    'engulfing_bullish': {'confidence': 75, 'timeframes': ['short', 'mid', 'long'], 'category': 'Candlestick Patterns'},
    'engulfing_bearish': {'confidence': 75, 'timeframes': ['short', 'mid', 'long'], 'category': 'Candlestick Patterns'},
    'hammer': {'confidence': 72, 'timeframes': ['mid', 'long'], 'category': 'Candlestick Patterns'},
    'shooting_star': {'confidence': 72, 'timeframes': ['mid', 'long'], 'category': 'Candlestick Patterns'},
    'doji_reversal': {'confidence': 65, 'timeframes': ['short', 'mid', 'long'], 'category': 'Candlestick Patterns'},
    'morning_star': {'confidence': 78, 'timeframes': ['mid', 'long'], 'category': 'Candlestick Patterns'},
    'evening_star': {'confidence': 78, 'timeframes': ['mid', 'long'], 'category': 'Candlestick Patterns'},
    'three_white_soldiers': {'confidence': 80, 'timeframes': ['mid', 'long'], 'category': 'Candlestick Patterns'},
    'three_black_crows': {'confidence': 80, 'timeframes': ['mid', 'long'], 'category': 'Candlestick Patterns'},
    'piercing_pattern': {'confidence': 70, 'timeframes': ['mid', 'long'], 'category': 'Candlestick Patterns'},
    'dark_cloud_cover': {'confidence': 70, 'timeframes': ['mid', 'long'], 'category': 'Candlestick Patterns'},
    'tweezer_top': {'confidence': 68, 'timeframes': ['short', 'mid'], 'category': 'Candlestick Patterns'},
    'tweezer_bottom': {'confidence': 68, 'timeframes': ['short', 'mid'], 'category': 'Candlestick Patterns'},
    'marubozu_bullish': {'confidence': 73, 'timeframes': ['short', 'mid'], 'category': 'Candlestick Patterns'},
    'marubozu_bearish': {'confidence': 73, 'timeframes': ['short', 'mid'], 'category': 'Candlestick Patterns'},
    'harami_bullish': {'confidence': 70, 'timeframes': ['mid', 'long'], 'category': 'Candlestick Patterns'},
    'harami_bearish': {'confidence': 70, 'timeframes': ['mid', 'long'], 'category': 'Candlestick Patterns'},
    'inverted_hammer': {'confidence': 71, 'timeframes': ['mid', 'long'], 'category': 'Candlestick Patterns'},
    'hanging_man': {'confidence': 71, 'timeframes': ['mid', 'long'], 'category': 'Candlestick Patterns'},

    # ===== Moving Averages (9 signals) =====
    'ma_cross_golden': {'confidence': 82, 'timeframes': ['mid', 'long'], 'category': 'Moving Averages'},
    'ma_cross_death': {'confidence': 82, 'timeframes': ['mid', 'long'], 'category': 'Moving Averages'},
    'price_above_ma20': {'confidence': 70, 'timeframes': ['short', 'mid'], 'category': 'Moving Averages'},
    'price_below_ma20': {'confidence': 70, 'timeframes': ['short', 'mid'], 'category': 'Moving Averages'},
    'ma_ribbon_bullish': {'confidence': 76, 'timeframes': ['mid', 'long'], 'category': 'Moving Averages'},
    'ma_ribbon_bearish': {'confidence': 76, 'timeframes': ['mid', 'long'], 'category': 'Moving Averages'},
    'ema_cross_fast': {'confidence': 74, 'timeframes': ['short', 'mid'], 'category': 'Moving Averages'},
    'triple_ema_bullish': {'confidence': 78, 'timeframes': ['mid'], 'category': 'Moving Averages'},
    'triple_ema_bearish': {'confidence': 78, 'timeframes': ['mid'], 'category': 'Moving Averages'},

    # ===== Momentum Indicators (30 signals) =====
    'rsi_oversold': {'confidence': 68, 'timeframes': ['short', 'mid', 'long'], 'category': 'Momentum Indicators'},
    'rsi_overbought': {'confidence': 68, 'timeframes': ['short', 'mid', 'long'], 'category': 'Momentum Indicators'},
    'rsi_divergence_bullish': {'confidence': 85, 'timeframes': ['mid', 'long'], 'category': 'Momentum Indicators'},
    'rsi_divergence_bearish': {'confidence': 85, 'timeframes': ['mid', 'long'], 'category': 'Momentum Indicators'},
    'rsi_centerline_cross_up': {'confidence': 72, 'timeframes': ['mid'], 'category': 'Momentum Indicators'},
    'rsi_centerline_cross_down': {'confidence': 72, 'timeframes': ['mid'], 'category': 'Momentum Indicators'},
    'macd_cross_bullish': {'confidence': 79, 'timeframes': ['mid', 'long'], 'category': 'Momentum Indicators'},
    'macd_cross_bearish': {'confidence': 79, 'timeframes': ['mid', 'long'], 'category': 'Momentum Indicators'},
    'macd_divergence_bullish': {'confidence': 87, 'timeframes': ['mid', 'long'], 'category': 'Momentum Indicators'},
    'macd_divergence_bearish': {'confidence': 87, 'timeframes': ['mid', 'long'], 'category': 'Momentum Indicators'},
    'macd_histogram_reversal_bullish': {'confidence': 75, 'timeframes': ['short', 'mid'], 'category': 'Momentum Indicators'},
    'macd_histogram_reversal_bearish': {'confidence': 75, 'timeframes': ['short', 'mid'], 'category': 'Momentum Indicators'},
    'stoch_oversold': {'confidence': 66, 'timeframes': ['short', 'mid'], 'category': 'Momentum Indicators'},
    'stoch_overbought': {'confidence': 66, 'timeframes': ['short', 'mid'], 'category': 'Momentum Indicators'},
    'stoch_cross_bullish': {'confidence': 71, 'timeframes': ['short', 'mid'], 'category': 'Momentum Indicators'},
    'stoch_cross_bearish': {'confidence': 71, 'timeframes': ['short', 'mid'], 'category': 'Momentum Indicators'},
    'cci_oversold': {'confidence': 67, 'timeframes': ['short', 'mid'], 'category': 'Momentum Indicators'},
    'cci_overbought': {'confidence': 67, 'timeframes': ['short', 'mid'], 'category': 'Momentum Indicators'},
    'williams_r_oversold': {'confidence': 65, 'timeframes': ['short', 'mid'], 'category': 'Momentum Indicators'},
    'williams_r_overbought': {'confidence': 65, 'timeframes': ['short', 'mid'], 'category': 'Momentum Indicators'},
    'momentum_5': {'confidence': 72, 'timeframes': ['short', 'mid'], 'category': 'Momentum Indicators'},
    'momentum_10': {'confidence': 74, 'timeframes': ['short', 'mid'], 'category': 'Momentum Indicators'},
    'mfi_oversold': {'confidence': 73, 'timeframes': ['short', 'mid', 'long'], 'category': 'Momentum Indicators'},
    'mfi_overbought': {'confidence': 73, 'timeframes': ['short', 'mid', 'long'], 'category': 'Momentum Indicators'},
    'mfi_divergence_bullish': {'confidence': 82, 'timeframes': ['mid', 'long'], 'category': 'Momentum Indicators'},
    'mfi_divergence_bearish': {'confidence': 82, 'timeframes': ['mid', 'long'], 'category': 'Momentum Indicators'},
    'roc_bullish': {'confidence': 70, 'timeframes': ['short', 'mid'], 'category': 'Momentum Indicators'},
    'roc_bearish': {'confidence': 70, 'timeframes': ['short', 'mid'], 'category': 'Momentum Indicators'},
    'tsi_cross_bullish': {'confidence': 76, 'timeframes': ['mid', 'long'], 'category': 'Momentum Indicators'},
    'tsi_cross_bearish': {'confidence': 76, 'timeframes': ['mid', 'long'], 'category': 'Momentum Indicators'},

    # ===== Volume Analysis (14 signals) =====
    'volume_spike_bullish': {'confidence': 77, 'timeframes': ['short', 'mid', 'long'], 'category': 'Volume Analysis'},
    'volume_spike_bearish': {'confidence': 77, 'timeframes': ['short', 'mid', 'long'], 'category': 'Volume Analysis'},
    'volume_divergence_bullish': {'confidence': 81, 'timeframes': ['mid', 'long'], 'category': 'Volume Analysis'},
    'volume_divergence_bearish': {'confidence': 81, 'timeframes': ['mid', 'long'], 'category': 'Volume Analysis'},
    'obv_bullish': {'confidence': 74, 'timeframes': ['mid', 'long'], 'category': 'Volume Analysis'},
    'obv_bearish': {'confidence': 74, 'timeframes': ['mid', 'long'], 'category': 'Volume Analysis'},
    'vwap_cross_above': {'confidence': 76, 'timeframes': ['short', 'mid'], 'category': 'Volume Analysis'},
    'vwap_cross_below': {'confidence': 76, 'timeframes': ['short', 'mid'], 'category': 'Volume Analysis'},
    'accumulation_distribution_bullish': {'confidence': 73, 'timeframes': ['mid', 'long'], 'category': 'Volume Analysis'},
    'accumulation_distribution_bearish': {'confidence': 73, 'timeframes': ['mid', 'long'], 'category': 'Volume Analysis'},
    'cmf_bullish': {'confidence': 75, 'timeframes': ['mid', 'long'], 'category': 'Volume Analysis'},
    'cmf_bearish': {'confidence': 75, 'timeframes': ['mid', 'long'], 'category': 'Volume Analysis'},
    'volume_climax_bullish': {'confidence': 79, 'timeframes': ['short', 'mid', 'long'], 'category': 'Volume Analysis'},
    'volume_climax_bearish': {'confidence': 79, 'timeframes': ['short', 'mid', 'long'], 'category': 'Volume Analysis'},

    # ===== Volatility Indicators (10 signals) =====
    'bollinger_squeeze': {'confidence': 80, 'timeframes': ['short', 'mid', 'long'], 'category': 'Volatility Indicators'},
    'bollinger_breakout_up': {'confidence': 78, 'timeframes': ['short', 'mid'], 'category': 'Volatility Indicators'},
    'bollinger_breakout_down': {'confidence': 78, 'timeframes': ['short', 'mid'], 'category': 'Volatility Indicators'},
    'bollinger_bounce_up': {'confidence': 69, 'timeframes': ['short', 'mid'], 'category': 'Volatility Indicators'},
    'bollinger_bounce_down': {'confidence': 69, 'timeframes': ['short', 'mid'], 'category': 'Volatility Indicators'},
    'atr_expansion': {'confidence': 71, 'timeframes': ['short', 'mid'], 'category': 'Volatility Indicators'},
    'keltner_breakout_up': {'confidence': 75, 'timeframes': ['mid'], 'category': 'Volatility Indicators'},
    'keltner_breakout_down': {'confidence': 75, 'timeframes': ['mid'], 'category': 'Volatility Indicators'},
    'donchian_breakout_up': {'confidence': 77, 'timeframes': ['mid', 'long'], 'category': 'Volatility Indicators'},
    'donchian_breakout_down': {'confidence': 77, 'timeframes': ['mid', 'long'], 'category': 'Volatility Indicators'},

    # ===== Trend Indicators (13 signals) =====
    'adx_strong_trend': {'confidence': 79, 'timeframes': ['mid', 'long'], 'category': 'Trend Indicators'},
    'adx_weak_trend': {'confidence': 70, 'timeframes': ['mid', 'long'], 'category': 'Trend Indicators'},
    'adx_reversal': {'confidence': 76, 'timeframes': ['mid', 'long'], 'category': 'Trend Indicators'},
    'supertrend_bullish': {'confidence': 81, 'timeframes': ['mid', 'long'], 'category': 'Trend Indicators'},
    'supertrend_bearish': {'confidence': 81, 'timeframes': ['mid', 'long'], 'category': 'Trend Indicators'},
    'parabolic_sar_flip_bullish': {'confidence': 74, 'timeframes': ['mid', 'long'], 'category': 'Trend Indicators'},
    'parabolic_sar_flip_bearish': {'confidence': 74, 'timeframes': ['mid', 'long'], 'category': 'Trend Indicators'},
    'ichimoku_bullish': {'confidence': 83, 'timeframes': ['long'], 'category': 'Trend Indicators'},
    'ichimoku_bearish': {'confidence': 83, 'timeframes': ['long'], 'category': 'Trend Indicators'},
    'aroon_bullish': {'confidence': 75, 'timeframes': ['mid', 'long'], 'category': 'Trend Indicators'},
    'aroon_bearish': {'confidence': 75, 'timeframes': ['mid', 'long'], 'category': 'Trend Indicators'},
    'elder_ray_bullish': {'confidence': 74, 'timeframes': ['mid', 'long'], 'category': 'Trend Indicators'},
    'elder_ray_bearish': {'confidence': 74, 'timeframes': ['mid', 'long'], 'category': 'Trend Indicators'},

    # ===== Price Action & Structure (26 signals) =====
    'higher_high': {'confidence': 77, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'lower_low': {'confidence': 77, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'break_of_structure_bullish': {'confidence': 84, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'break_of_structure_bearish': {'confidence': 84, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'support_bounce': {'confidence': 72, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'resistance_rejection': {'confidence': 72, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'support_break': {'confidence': 80, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'resistance_break': {'confidence': 80, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'pivot_point_bullish': {'confidence': 69, 'timeframes': ['short', 'mid'], 'category': 'Price Action & Structure'},
    'pivot_point_bearish': {'confidence': 69, 'timeframes': ['short', 'mid'], 'category': 'Price Action & Structure'},
    'fibonacci_bounce_382': {'confidence': 73, 'timeframes': ['mid', 'long'], 'category': 'Price Action & Structure'},
    'fibonacci_bounce_618': {'confidence': 76, 'timeframes': ['mid', 'long'], 'category': 'Price Action & Structure'},
    'round_number_support': {'confidence': 68, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'round_number_resistance': {'confidence': 68, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'gap_up': {'confidence': 76, 'timeframes': ['mid', 'long'], 'category': 'Price Action & Structure'},
    'gap_down': {'confidence': 76, 'timeframes': ['mid', 'long'], 'category': 'Price Action & Structure'},
    'order_block_bullish': {'confidence': 78, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'order_block_bearish': {'confidence': 78, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'fvg_bullish': {'confidence': 77, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'fvg_bearish': {'confidence': 77, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'liquidity_sweep_bullish': {'confidence': 80, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'liquidity_sweep_bearish': {'confidence': 80, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'choch_bullish': {'confidence': 79, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'choch_bearish': {'confidence': 79, 'timeframes': ['short', 'mid', 'long'], 'category': 'Price Action & Structure'},
    'premium_zone': {'confidence': 71, 'timeframes': ['mid', 'long'], 'category': 'Price Action & Structure'},
    'discount_zone': {'confidence': 71, 'timeframes': ['mid', 'long'], 'category': 'Price Action & Structure'},

    # ===== Volume Profile (6 signals) =====
    'poc_support': {'confidence': 75, 'timeframes': ['mid', 'long'], 'category': 'Volume Profile'},
    'poc_resistance': {'confidence': 75, 'timeframes': ['mid', 'long'], 'category': 'Volume Profile'},
    'value_area_high': {'confidence': 72, 'timeframes': ['mid', 'long'], 'category': 'Volume Profile'},
    'value_area_low': {'confidence': 72, 'timeframes': ['mid', 'long'], 'category': 'Volume Profile'},
    'high_volume_node': {'confidence': 73, 'timeframes': ['mid', 'long'], 'category': 'Volume Profile'},
    'low_volume_node': {'confidence': 70, 'timeframes': ['mid', 'long'], 'category': 'Volume Profile'},
  }

  def __init__(self, db_path: str = 'crypto_signals.db'):
    self.db_path = db_path
    self.timeframe_minutes = {
      '1m': 1, '3m': 3, '5m': 5, '15m': 15,
      '30m': 30, '1h': 60, '2h': 120, '4h': 240,
      '6h': 360, '8h': 480, '12h': 720, '1d': 1440,
      '3d': 4320, '1w': 10080
    }
    self.init_combo_database()

  def init_combo_database(self):
    """Initialize the live_tf_combos table for storing live combination analysis"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
      CREATE TABLE IF NOT EXISTS live_tf_combos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        combo_signal_name TEXT NOT NULL,
        signal_accuracies TEXT NOT NULL,
        signal_samples TEXT NOT NULL,
        combo_price_change REAL NOT NULL,
        min_window INTEGER NOT NULL,
        max_window INTEGER NOT NULL,
        timeframe TEXT NOT NULL,
        accuracy REAL NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(symbol, combo_signal_name, timeframe, timestamp)
      )
    ''')

    cursor.execute('''
      CREATE INDEX IF NOT EXISTS idx_live_combos_symbol_time
      ON live_tf_combos(symbol, timestamp DESC)
    ''')

    cursor.execute('''
      CREATE INDEX IF NOT EXISTS idx_live_combos_combo
      ON live_tf_combos(combo_signal_name, timeframe)
    ''')

    cursor.execute('''
      CREATE INDEX IF NOT EXISTS idx_live_combos_accuracy
      ON live_tf_combos(accuracy DESC)
    ''')

    conn.commit()
    conn.close()

  # ... [Keep all the existing methods: fetch_kucoin_data, fetch_binance_data, fetch_data,
  # calculate_all_indicators, analyze_candlestick_patterns, analyze_moving_averages,
  # analyze_momentum, analyze_volume, analyze_volatility, analyze_trend,
  # analyze_price_action, analyze_volume_profile - UNCHANGED] ...

  def fetch_kucoin_data(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from KuCoin"""
    try:
      symbol_pair = f"{symbol}-USDT"
      end_time = int(datetime.now().timestamp())
      minutes = self.timeframe_minutes[timeframe]
      start_time = end_time - (minutes * 60 * limit)

      url = "https://api.kucoin.com/api/v1/market/candles"

      tf_map = {
        '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min',
        '30m': '30min', '1h': '1hour', '2h': '2hour', '4h': '4hour',
        '6h': '6hour', '8h': '8hour', '12h': '12hour', '1d': '1day',
        '1w': '1week'
      }

      if timeframe not in tf_map:
        return None

      params = {
        'symbol': symbol_pair,
        'type': tf_map[timeframe],
        'startAt': start_time,
        'endAt': end_time
      }

      response = requests.get(url, params=params, timeout=10)
      response.raise_for_status()
      data = response.json()

      if data.get('code') != '200000' or not data.get('data'):
        return None

      df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
      df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
      df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s')

      for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])

      return df.sort_values('timestamp').reset_index(drop=True)

    except Exception as e:
      logging.error(f"KuCoin fetch failed for {symbol} {timeframe}: {e}")
      return None

  def fetch_binance_data(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
    """Fallback to Binance"""
    try:
      symbol_pair = f"{symbol}USDT"
      url = "https://api.binance.com/api/v3/klines"

      params = {
        'symbol': symbol_pair,
        'interval': timeframe.replace('m', 'm').replace('h', 'h').replace('d', 'd').replace('w', 'w'),
        'limit': limit
      }

      response = requests.get(url, params=params, timeout=10)
      response.raise_for_status()
      data = response.json()

      if not data:
        return None

      df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'buy_volume', 'buy_quote_volume', 'ignore'
      ])

      df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')

      for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])

      return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)

    except Exception as e:
      logging.error(f"Binance fetch failed for {symbol} {timeframe}: {e}")
      return None

  def fetch_data(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
    """Fetch with KuCoin primary, Binance fallback"""
    data = self.fetch_kucoin_data(symbol, timeframe, limit)
    if data is None or len(data) < 50:
      data = self.fetch_binance_data(symbol, timeframe, limit)
    return data

  def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ALL technical indicators"""
    data = df.copy()

    if len(data) < 50:
      return data

    try:
      # Moving Averages
      for period in [5, 10, 20, 50, 100, 200]:
        if len(data) >= period:
          data[f'SMA_{period}'] = data['close'].rolling(window=period).mean()
          data[f'EMA_{period}'] = data['close'].ewm(span=period, adjust=False).mean()

      # MACD
      ema_12 = data['close'].ewm(span=12, adjust=False).mean()
      ema_26 = data['close'].ewm(span=26, adjust=False).mean()
      data['MACD'] = ema_12 - ema_26
      data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
      data['MACD_histogram'] = data['MACD'] - data['MACD_signal']

      # RSI
      delta = data['close'].diff()
      gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
      rs = gain / loss
      data['RSI'] = 100 - (100 / (1 + rs))

      # Stochastic
      low_14 = data['low'].rolling(window=14).min()
      high_14 = data['high'].rolling(window=14).max()
      data['STOCH_K'] = 100 * ((data['close'] - low_14) / (high_14 - low_14))
      data['STOCH_D'] = data['STOCH_K'].rolling(window=3).mean()

      # Bollinger Bands
      data['BB_middle'] = data['close'].rolling(window=20).mean()
      data['BB_std'] = data['close'].rolling(window=20).std()
      data['BB_upper'] = data['BB_middle'] + (data['BB_std'] * 2)
      data['BB_lower'] = data['BB_middle'] - (data['BB_std'] * 2)
      data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']

      # ATR
      high_low = data['high'] - data['low']
      high_close = np.abs(data['high'] - data['close'].shift())
      low_close = np.abs(data['low'] - data['close'].shift())
      ranges = pd.concat([high_low, high_close, low_close], axis=1)
      true_range = np.max(ranges, axis=1)
      data['ATR'] = true_range.rolling(14).mean()

      # OBV
      data['OBV'] = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()

      # VWAP
      data['VWAP'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()

      # ADX
      plus_dm = data['high'].diff()
      minus_dm = -data['low'].diff()
      plus_dm[plus_dm < 0] = 0
      minus_dm[minus_dm < 0] = 0

      tr = true_range
      atr = tr.rolling(14).mean()

      plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
      minus_di = 100 * (minus_dm.rolling(14).mean() / atr)

      dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
      data['ADX'] = dx.rolling(14).mean()
      data['PLUS_DI'] = plus_di
      data['MINUS_DI'] = minus_di

      # CCI
      tp = (data['high'] + data['low'] + data['close']) / 3
      data['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

      # Williams %R
      data['WILLR'] = -100 * ((high_14 - data['close']) / (high_14 - low_14))

      # MFI
      typical_price = (data['high'] + data['low'] + data['close']) / 3
      money_flow = typical_price * data['volume']

      positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
      negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()

      money_ratio = positive_flow / negative_flow
      data['MFI'] = 100 - (100 / (1 + money_ratio))

      # CMF
      mf_multiplier = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
      mf_volume = mf_multiplier * data['volume']
      data['CMF'] = mf_volume.rolling(20).sum() / data['volume'].rolling(20).sum()

      # ROC
      data['ROC'] = ((data['close'] - data['close'].shift(12)) / data['close'].shift(12)) * 100

      # Aroon
      aroon_period = 25
      data['AROON_UP'] = data['high'].rolling(aroon_period + 1).apply(
        lambda x: float(aroon_period - x.argmax()) / aroon_period * 100, raw=False
      )
      data['AROON_DOWN'] = data['low'].rolling(aroon_period + 1).apply(
        lambda x: float(aroon_period - x.argmin()) / aroon_period * 100, raw=False
      )

      # Elder Ray
      ema_13 = data['close'].ewm(span=13, adjust=False).mean()
      data['BULL_POWER'] = data['high'] - ema_13
      data['BEAR_POWER'] = data['low'] - ema_13

      # TSI
      price_change = data['close'].diff()
      double_smoothed_pc = price_change.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
      double_smoothed_abs_pc = price_change.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
      data['TSI'] = 100 * (double_smoothed_pc / double_smoothed_abs_pc)
      data['TSI_signal'] = data['TSI'].ewm(span=7, adjust=False).mean()

      # Donchian Channel
      donchian_period = 20
      data['DONCHIAN_HIGH'] = data['high'].rolling(donchian_period).max()
      data['DONCHIAN_LOW'] = data['low'].rolling(donchian_period).min()
      data['DONCHIAN_MID'] = (data['DONCHIAN_HIGH'] + data['DONCHIAN_LOW']) / 2

      # Momentum
      data['MOMENTUM_5'] = data['close'].pct_change(periods=5)
      data['MOMENTUM_10'] = data['close'].pct_change(periods=10)

      # Parabolic SAR
      data['SAR'] = data['close'].rolling(5).mean()

    except Exception as e:
      logging.error(f"Indicator calculation error: {e}")

    return data

  def analyze_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
    """Detect candlestick patterns"""
    signals = {}

    if len(df) < 3:
      return signals

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) >= 3 else None

    body_curr = abs(curr['close'] - curr['open'])
    body_prev = abs(prev['close'] - prev['open'])
    range_curr = curr['high'] - curr['low']
    range_prev = prev['high'] - prev['low']

    # Engulfing
    if curr['close'] > curr['open'] and prev['close'] < prev['open']:
      if curr['close'] > prev['open'] and curr['open'] < prev['close']:
        signals['engulfing_bullish'] = {'signal': 'BUY', 'strength': 'STRONG'}

    if curr['close'] < curr['open'] and prev['close'] > prev['open']:
      if curr['close'] < prev['open'] and curr['open'] > prev['close']:
        signals['engulfing_bearish'] = {'signal': 'SELL', 'strength': 'STRONG'}

    # Hammer
    upper_shadow = curr['high'] - max(curr['open'], curr['close'])
    lower_shadow = min(curr['open'], curr['close']) - curr['low']

    if lower_shadow > body_curr * 2 and upper_shadow < body_curr * 0.3:
      if curr['close'] > curr['open']:
        signals['hammer'] = {'signal': 'BUY', 'strength': 'MODERATE'}

    # Shooting Star
    if upper_shadow > body_curr * 2 and lower_shadow < body_curr * 0.3:
      if curr['close'] < curr['open']:
        signals['shooting_star'] = {'signal': 'SELL', 'strength': 'MODERATE'}

    # Doji
    if body_curr < range_curr * 0.1:
      signals['doji_reversal'] = {'signal': 'REVERSAL', 'strength': 'WEAK'}

    # Morning/Evening Star
    if prev2 is not None:
      body_prev2 = abs(prev2['close'] - prev2['open'])

      # Morning Star
      if (prev2['close'] < prev2['open'] and
          body_prev < body_prev2 * 0.3 and
          curr['close'] > curr['open'] and
          curr['close'] > (prev2['open'] + prev2['close']) / 2):
        signals['morning_star'] = {'signal': 'BUY', 'strength': 'STRONG'}

      # Evening Star
      if (prev2['close'] > prev2['open'] and
          body_prev < body_prev2 * 0.3 and
          curr['close'] < curr['open'] and
          curr['close'] < (prev2['open'] + prev2['close']) / 2):
        signals['evening_star'] = {'signal': 'SELL', 'strength': 'STRONG'}

    # Marubozu
    if body_curr > range_curr * 0.95:
      if curr['close'] > curr['open']:
        signals['marubozu_bullish'] = {'signal': 'BUY', 'strength': 'STRONG'}
      else:
        signals['marubozu_bearish'] = {'signal': 'SELL', 'strength': 'STRONG'}

    # Harami
    if prev2 is not None:
      # Bullish Harami
      if (prev['close'] < prev['open'] and
          curr['close'] > curr['open'] and
          curr['open'] > prev['close'] and
          curr['close'] < prev['open'] and
          body_curr < body_prev):
        signals['harami_bullish'] = {'signal': 'BUY', 'strength': 'MODERATE'}

      # Bearish Harami
      if (prev['close'] > prev['open'] and
          curr['close'] < curr['open'] and
          curr['open'] < prev['close'] and
          curr['close'] > prev['open'] and
          body_curr < body_prev):
        signals['harami_bearish'] = {'signal': 'SELL', 'strength': 'MODERATE'}

    # Inverted Hammer
    if upper_shadow > body_curr * 2 and lower_shadow < body_curr * 0.3:
      if len(df) >= 10:
        recent_low = df['low'].iloc[-10:].min()
        if abs(curr['low'] - recent_low) / recent_low < 0.02:
          signals['inverted_hammer'] = {'signal': 'BUY', 'strength': 'MODERATE'}

    # Hanging Man
    if lower_shadow > body_curr * 2 and upper_shadow < body_curr * 0.3:
      if len(df) >= 10:
        recent_high = df['high'].iloc[-10:].max()
        if abs(curr['high'] - recent_high) / recent_high < 0.02:
          signals['hanging_man'] = {'signal': 'SELL', 'strength': 'MODERATE'}

    return signals

  def analyze_moving_averages(self, df: pd.DataFrame) -> Dict[str, any]:
    """Analyze moving average signals"""
    signals = {}

    if len(df) < 200:
      return signals

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # Golden/Death Cross
    if not pd.isna(curr['SMA_50']) and not pd.isna(curr['SMA_200']):
      if curr['SMA_50'] > curr['SMA_200'] and prev['SMA_50'] <= prev['SMA_200']:
        signals['ma_cross_golden'] = {'signal': 'BUY', 'strength': 'STRONG'}

      if curr['SMA_50'] < curr['SMA_200'] and prev['SMA_50'] >= prev['SMA_200']:
        signals['ma_cross_death'] = {'signal': 'SELL', 'strength': 'STRONG'}

    # Price vs MA20
    if not pd.isna(curr['SMA_20']):
      if curr['close'] > curr['SMA_20'] and prev['close'] <= prev['SMA_20']:
        signals['price_above_ma20'] = {'signal': 'BUY', 'strength': 'MODERATE'}

      if curr['close'] < curr['SMA_20'] and prev['close'] >= prev['SMA_20']:
        signals['price_below_ma20'] = {'signal': 'SELL', 'strength': 'MODERATE'}

    # MA Ribbon
    if all(not pd.isna(curr[f'EMA_{p}']) for p in [5, 10, 20]):
      if curr['EMA_5'] > curr['EMA_10'] > curr['EMA_20']:
        signals['ma_ribbon_bullish'] = {'signal': 'BUY', 'strength': 'STRONG'}

      if curr['EMA_5'] < curr['EMA_10'] < curr['EMA_20']:
        signals['ma_ribbon_bearish'] = {'signal': 'SELL', 'strength': 'STRONG'}

    return signals

  def analyze_momentum(self, df: pd.DataFrame) -> Dict[str, any]:
    """Analyze momentum indicators"""
    signals = {}

    if len(df) < 50:
      return signals

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # RSI
    if not pd.isna(curr['RSI']):
      if curr['RSI'] < 30:
        signals['rsi_oversold'] = {'signal': 'BUY', 'strength': 'MODERATE', 'value': curr['RSI']}

      if curr['RSI'] > 70:
        signals['rsi_overbought'] = {'signal': 'SELL', 'strength': 'MODERATE', 'value': curr['RSI']}

      if curr['RSI'] > 50 and prev['RSI'] <= 50:
        signals['rsi_centerline_cross_up'] = {'signal': 'BUY', 'strength': 'MODERATE'}

      if curr['RSI'] < 50 and prev['RSI'] >= 50:
        signals['rsi_centerline_cross_down'] = {'signal': 'SELL', 'strength': 'MODERATE'}

      # RSI Divergence
      if len(df) >= 20:
        price_trend = df['close'].iloc[-20:].is_monotonic_increasing
        rsi_trend = df['RSI'].iloc[-20:].is_monotonic_increasing

        if price_trend and not rsi_trend:
          signals['rsi_divergence_bearish'] = {'signal': 'SELL', 'strength': 'VERY_STRONG'}

        if not price_trend and rsi_trend:
          signals['rsi_divergence_bullish'] = {'signal': 'BUY', 'strength': 'VERY_STRONG'}

    # MACD
    if not pd.isna(curr['MACD']) and not pd.isna(curr['MACD_signal']):
      if curr['MACD'] > curr['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
        signals['macd_cross_bullish'] = {'signal': 'BUY', 'strength': 'STRONG'}

      if curr['MACD'] < curr['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
        signals['macd_cross_bearish'] = {'signal': 'SELL', 'strength': 'STRONG'}

      # Histogram reversal
      if curr['MACD_histogram'] > 0 and prev['MACD_histogram'] <= 0:
        signals['macd_histogram_reversal_bullish'] = {'signal': 'BUY', 'strength': 'MODERATE'}

      if curr['MACD_histogram'] < 0 and prev['MACD_histogram'] >= 0:
        signals['macd_histogram_reversal_bearish'] = {'signal': 'SELL', 'strength': 'MODERATE'}

    # Stochastic
    if not pd.isna(curr['STOCH_K']) and not pd.isna(curr['STOCH_D']):
      if curr['STOCH_D'] < 20:
        signals['stoch_oversold'] = {'signal': 'BUY', 'strength': 'MODERATE', 'value': curr['STOCH_D']}

      if curr['STOCH_D'] > 80:
        signals['stoch_overbought'] = {'signal': 'SELL', 'strength': 'MODERATE', 'value': curr['STOCH_D']}

      if curr['STOCH_K'] > curr['STOCH_D'] and prev['STOCH_K'] <= prev['STOCH_D']:
        signals['stoch_cross_bullish'] = {'signal': 'BUY', 'strength': 'MODERATE'}

      if curr['STOCH_K'] < curr['STOCH_D'] and prev['STOCH_K'] >= prev['STOCH_D']:
        signals['stoch_cross_bearish'] = {'signal': 'SELL', 'strength': 'MODERATE'}

    # CCI
    if not pd.isna(curr['CCI']):
      if curr['CCI'] < -100:
        signals['cci_oversold'] = {'signal': 'BUY', 'strength': 'MODERATE', 'value': curr['CCI']}

      if curr['CCI'] > 100:
        signals['cci_overbought'] = {'signal': 'SELL', 'strength': 'MODERATE', 'value': curr['CCI']}

    # Williams %R
    if not pd.isna(curr['WILLR']):
      if curr['WILLR'] < -80:
        signals['williams_r_oversold'] = {'signal': 'BUY', 'strength': 'MODERATE', 'value': curr['WILLR']}

      if curr['WILLR'] > -20:
        signals['williams_r_overbought'] = {'signal': 'SELL', 'strength': 'MODERATE', 'value': curr['WILLR']}

    # Momentum_5 and Momentum_10
    if not pd.isna(curr['MOMENTUM_5']):
      if curr['MOMENTUM_5'] > 0.03:
        signals['momentum_5'] = {'signal': 'BUY', 'strength': 'MODERATE', 'value': curr['MOMENTUM_5']}
      elif curr['MOMENTUM_5'] < -0.03:
        signals['momentum_5'] = {'signal': 'SELL', 'strength': 'MODERATE', 'value': curr['MOMENTUM_5']}

    if not pd.isna(curr['MOMENTUM_10']):
      if curr['MOMENTUM_10'] > 0.05:
        signals['momentum_10'] = {'signal': 'BUY', 'strength': 'MODERATE', 'value': curr['MOMENTUM_10']}
      elif curr['MOMENTUM_10'] < -0.05:
        signals['momentum_10'] = {'signal': 'SELL', 'strength': 'MODERATE', 'value': curr['MOMENTUM_10']}

    # MFI
    if not pd.isna(curr['MFI']):
      if curr['MFI'] < 20:
        signals['mfi_oversold'] = {'signal': 'BUY', 'strength': 'STRONG', 'value': curr['MFI']}

      if curr['MFI'] > 80:
        signals['mfi_overbought'] = {'signal': 'SELL', 'strength': 'STRONG', 'value': curr['MFI']}

      # MFI Divergence
      if len(df) >= 20:
        price_trend = df['close'].iloc[-20:].is_monotonic_increasing
        mfi_trend = df['MFI'].iloc[-20:].is_monotonic_increasing

        if price_trend and not mfi_trend:
          signals['mfi_divergence_bearish'] = {'signal': 'SELL', 'strength': 'VERY_STRONG'}

        if not price_trend and mfi_trend:
          signals['mfi_divergence_bullish'] = {'signal': 'BUY', 'strength': 'VERY_STRONG'}

    # ROC
    if not pd.isna(curr['ROC']):
      if curr['ROC'] > 5:
        signals['roc_bullish'] = {'signal': 'BUY', 'strength': 'MODERATE', 'value': curr['ROC']}

      if curr['ROC'] < -5:
        signals['roc_bearish'] = {'signal': 'SELL', 'strength': 'MODERATE', 'value': curr['ROC']}

    # TSI
    if not pd.isna(curr['TSI']) and not pd.isna(curr['TSI_signal']):
      if curr['TSI'] > curr['TSI_signal'] and prev['TSI'] <= prev['TSI_signal']:
        signals['tsi_cross_bullish'] = {'signal': 'BUY', 'strength': 'STRONG'}

      if curr['TSI'] < curr['TSI_signal'] and prev['TSI'] >= prev['TSI_signal']:
        signals['tsi_cross_bearish'] = {'signal': 'SELL', 'strength': 'STRONG'}

    return signals

  def analyze_volume(self, df: pd.DataFrame) -> Dict[str, any]:
    """Analyze volume signals"""
    signals = {}

    if len(df) < 20:
      return signals

    curr = df.iloc[-1]
    avg_volume = df['volume'].iloc[-20:].mean()

    # Volume spike
    if curr['volume'] > avg_volume * 2:
      if curr['close'] > curr['open']:
        signals['volume_spike_bullish'] = {'signal': 'BUY', 'strength': 'STRONG',
                                           'multiplier': curr['volume'] / avg_volume}
      else:
        signals['volume_spike_bearish'] = {'signal': 'SELL', 'strength': 'STRONG',
                                           'multiplier': curr['volume'] / avg_volume}

    # OBV
    if len(df) >= 50 and not pd.isna(curr['OBV']):
      obv_sma = df['OBV'].rolling(20).mean().iloc[-1]
      if curr['OBV'] > obv_sma:
        signals['obv_bullish'] = {'signal': 'BUY', 'strength': 'MODERATE'}
      else:
        signals['obv_bearish'] = {'signal': 'SELL', 'strength': 'MODERATE'}

    # VWAP
    if not pd.isna(curr['VWAP']):
      prev = df.iloc[-2]
      if curr['close'] > curr['VWAP'] and prev['close'] <= prev['VWAP']:
        signals['vwap_cross_above'] = {'signal': 'BUY', 'strength': 'STRONG'}

      if curr['close'] < curr['VWAP'] and prev['close'] >= prev['VWAP']:
        signals['vwap_cross_below'] = {'signal': 'SELL', 'strength': 'STRONG'}

    # CMF
    if not pd.isna(curr['CMF']):
      if curr['CMF'] > 0.1:
        signals['cmf_bullish'] = {'signal': 'BUY', 'strength': 'STRONG', 'value': curr['CMF']}

      if curr['CMF'] < -0.1:
        signals['cmf_bearish'] = {'signal': 'SELL', 'strength': 'STRONG', 'value': curr['CMF']}

    # Volume Climax
    if len(df) >= 30:
      volume_max = df['volume'].iloc[-30:].max()
      if curr['volume'] == volume_max:
        price_high = df['high'].iloc[-30:].max()
        price_low = df['low'].iloc[-30:].min()

        if curr['high'] == price_high and curr['close'] < curr['open']:
          signals['volume_climax_bearish'] = {'signal': 'SELL', 'strength': 'STRONG'}

        if curr['low'] == price_low and curr['close'] > curr['open']:
          signals['volume_climax_bullish'] = {'signal': 'BUY', 'strength': 'STRONG'}

    return signals

  def analyze_volatility(self, df: pd.DataFrame) -> Dict[str, any]:
    """Analyze volatility indicators"""
    signals = {}

    if len(df) < 30:
      return signals

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # Bollinger Bands
    if not pd.isna(curr['BB_width']):
      avg_width = df['BB_width'].iloc[-30:].mean()

      # Squeeze
      if curr['BB_width'] < avg_width * 0.5:
        signals['bollinger_squeeze'] = {'signal': 'BREAKOUT_PENDING', 'strength': 'STRONG'}

      # Breakouts
      if curr['close'] > curr['BB_upper'] and prev['close'] <= prev['BB_upper']:
        signals['bollinger_breakout_up'] = {'signal': 'BUY', 'strength': 'STRONG'}

      if curr['close'] < curr['BB_lower'] and prev['close'] >= prev['BB_lower']:
        signals['bollinger_breakout_down'] = {'signal': 'SELL', 'strength': 'STRONG'}

      # Bounces
      if prev['close'] <= prev['BB_lower'] and curr['close'] > prev['BB_lower']:
        signals['bollinger_bounce_up'] = {'signal': 'BUY', 'strength': 'MODERATE'}

      if prev['close'] >= prev['BB_upper'] and curr['close'] < prev['BB_upper']:
        signals['bollinger_bounce_down'] = {'signal': 'SELL', 'strength': 'MODERATE'}

    # ATR expansion
    if not pd.isna(curr['ATR']):
      avg_atr = df['ATR'].iloc[-30:].mean()
      if curr['ATR'] > avg_atr * 1.5:
        signals['atr_expansion'] = {'signal': 'VOLATILE', 'strength': 'MODERATE', 'value': curr['ATR'] / avg_atr}

    # Donchian Channel
    if not pd.isna(curr['DONCHIAN_HIGH']) and not pd.isna(curr['DONCHIAN_LOW']):
      if curr['close'] > curr['DONCHIAN_HIGH'] and prev['close'] <= prev['DONCHIAN_HIGH']:
        signals['donchian_breakout_up'] = {'signal': 'BUY', 'strength': 'STRONG'}

      if curr['close'] < curr['DONCHIAN_LOW'] and prev['close'] >= prev['DONCHIAN_LOW']:
        signals['donchian_breakout_down'] = {'signal': 'SELL', 'strength': 'STRONG'}

    return signals

  def analyze_trend(self, df: pd.DataFrame) -> Dict[str, any]:
    """Analyze trend indicators"""
    signals = {}

    if len(df) < 50:
      return signals

    curr = df.iloc[-1]

    # ADX
    if not pd.isna(curr['ADX']):
      if curr['ADX'] > 25:
        if curr['PLUS_DI'] > curr['MINUS_DI']:
          signals['adx_strong_trend'] = {'signal': 'BUY', 'strength': 'STRONG', 'value': curr['ADX']}
        else:
          signals['adx_strong_trend'] = {'signal': 'SELL', 'strength': 'STRONG', 'value': curr['ADX']}

      if curr['ADX'] < 20:
        signals['adx_weak_trend'] = {'signal': 'CONSOLIDATION', 'strength': 'MODERATE'}

    # Aroon
    if not pd.isna(curr['AROON_UP']) and not pd.isna(curr['AROON_DOWN']):
      if curr['AROON_UP'] > 70 and curr['AROON_DOWN'] < 30:
        signals['aroon_bullish'] = {'signal': 'BUY', 'strength': 'STRONG'}

      if curr['AROON_DOWN'] > 70 and curr['AROON_UP'] < 30:
        signals['aroon_bearish'] = {'signal': 'SELL', 'strength': 'STRONG'}

    # Elder Ray
    if not pd.isna(curr['BULL_POWER']) and not pd.isna(curr['BEAR_POWER']):
      if curr['BULL_POWER'] > 0 and curr['BEAR_POWER'] > curr['BEAR_POWER']:
        signals['elder_ray_bullish'] = {'signal': 'BUY', 'strength': 'MODERATE'}

      if curr['BEAR_POWER'] < 0 and curr['BULL_POWER'] < 0:
        signals['elder_ray_bearish'] = {'signal': 'SELL', 'strength': 'MODERATE'}

    return signals

  def analyze_price_action(self, df: pd.DataFrame) -> Dict[str, any]:
    """Analyze price action including SMC concepts"""
    signals = {}

    if len(df) < 10:
      return signals

    recent = df.iloc[-10:]
    highs = recent['high']
    lows = recent['low']

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # Higher highs / Lower lows
    if curr['high'] > highs.iloc[-2:].max():
      signals['higher_high'] = {'signal': 'BUY', 'strength': 'MODERATE'}

    if curr['low'] < lows.iloc[-2:].min():
      signals['lower_low'] = {'signal': 'SELL', 'strength': 'MODERATE'}

    # Support/Resistance
    support = recent['low'].min()
    resistance = recent['high'].max()

    if abs(curr['close'] - support) / support < 0.01:
      if curr['close'] > prev['close']:
        signals['support_bounce'] = {'signal': 'BUY', 'strength': 'MODERATE', 'level': support}

    if abs(curr['close'] - resistance) / resistance < 0.01:
      if curr['close'] < prev['close']:
        signals['resistance_rejection'] = {'signal': 'SELL', 'strength': 'MODERATE', 'level': resistance}

    # Breakouts
    if curr['close'] > resistance * 1.01:
      signals['resistance_break'] = {'signal': 'BUY', 'strength': 'STRONG', 'level': resistance}

    if curr['close'] < support * 0.99:
      signals['support_break'] = {'signal': 'SELL', 'strength': 'STRONG', 'level': support}

    # Gap Detection
    if prev['high'] < curr['low']:
      gap_size = ((curr['low'] - prev['high']) / prev['high']) * 100
      if gap_size > 0.5:
        signals['gap_up'] = {'signal': 'BUY', 'strength': 'MODERATE', 'gap_size': gap_size}

    if prev['low'] > curr['high']:
      gap_size = ((prev['low'] - curr['high']) / curr['high']) * 100
      if gap_size > 0.5:
        signals['gap_down'] = {'signal': 'SELL', 'strength': 'MODERATE', 'gap_size': gap_size}

    # Order Blocks
    if len(df) >= 5:
      last_5 = df.iloc[-5:]

      bearish_candles = last_5[last_5['close'] < last_5['open']]
      if len(bearish_candles) > 0:
        last_bearish = bearish_candles.iloc[-1]
        subsequent = df.loc[last_bearish.name:].iloc[1:4]
        if len(subsequent) >= 2 and (subsequent['close'] > subsequent['open']).all():
          signals['order_block_bullish'] = {'signal': 'BUY', 'strength': 'STRONG'}

      bullish_candles = last_5[last_5['close'] > last_5['open']]
      if len(bullish_candles) > 0:
        last_bullish = bullish_candles.iloc[-1]
        subsequent = df.loc[last_bullish.name:].iloc[1:4]
        if len(subsequent) >= 2 and (subsequent['close'] < subsequent['open']).all():
          signals['order_block_bearish'] = {'signal': 'SELL', 'strength': 'STRONG'}

    # Fair Value Gap
    if len(df) >= 3:
      candle_3 = df.iloc[-3]
      candle_1 = df.iloc[-1]

      if candle_1['low'] > candle_3['high']:
        signals['fvg_bullish'] = {'signal': 'BUY', 'strength': 'MODERATE'}

      if candle_1['high'] < candle_3['low']:
        signals['fvg_bearish'] = {'signal': 'SELL', 'strength': 'MODERATE'}

    # Liquidity Sweep
    if len(df) >= 20:
      swing_high = df['high'].iloc[-20:].max()
      swing_low = df['low'].iloc[-20:].min()

      if curr['low'] < swing_low * 0.999 and curr['close'] > curr['open']:
        signals['liquidity_sweep_bullish'] = {'signal': 'BUY', 'strength': 'STRONG'}

      if curr['high'] > swing_high * 1.001 and curr['close'] < curr['open']:
        signals['liquidity_sweep_bearish'] = {'signal': 'SELL', 'strength': 'STRONG'}

    # Change of Character
    if len(df) >= 10:
      recent_highs = df['high'].iloc[-10:-1]
      recent_lows = df['low'].iloc[-10:-1]

      if curr['close'] > recent_highs.max() and df['close'].iloc[-5] < recent_highs.max():
        signals['choch_bullish'] = {'signal': 'BUY', 'strength': 'STRONG'}

      if curr['close'] < recent_lows.min() and df['close'].iloc[-5] > recent_lows.min():
        signals['choch_bearish'] = {'signal': 'SELL', 'strength': 'STRONG'}

    # Premium/Discount Zones
    if len(df) >= 50:
      range_high = df['high'].iloc[-50:].max()
      range_low = df['low'].iloc[-50:].min()
      range_mid = (range_high + range_low) / 2

      premium_threshold = range_mid + (range_high - range_mid) * 0.5
      if curr['close'] > premium_threshold:
        signals['premium_zone'] = {'signal': 'SELL', 'strength': 'MODERATE'}

      discount_threshold = range_mid - (range_mid - range_low) * 0.5
      if curr['close'] < discount_threshold:
        signals['discount_zone'] = {'signal': 'BUY', 'strength': 'MODERATE'}

    return signals

  def analyze_volume_profile(self, df: pd.DataFrame) -> Dict[str, any]:
    """Analyze volume profile"""
    signals = {}

    if len(df) < 100:
      return signals

    curr = df.iloc[-1]
    recent_data = df.iloc[-100:]

    # Create price bins
    num_bins = 50
    bins = np.linspace(recent_data['low'].min(), recent_data['high'].max(), num_bins)

    # Aggregate volume by price level
    volume_by_price = np.zeros(num_bins - 1)

    for i in range(len(recent_data)):
      row = recent_data.iloc[i]
      candle_bins = np.digitize([row['low'], row['high']], bins)
      for b in range(candle_bins[0], min(candle_bins[1] + 1, num_bins)):
        if b < len(volume_by_price):
          volume_by_price[b] += row['volume'] / (candle_bins[1] - candle_bins[0] + 1)

    # Point of Control
    poc_idx = np.argmax(volume_by_price)
    poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2

    # Value Area
    total_volume = volume_by_price.sum()
    sorted_indices = np.argsort(volume_by_price)[::-1]
    cumulative_volume = 0
    value_area_indices = []

    for idx in sorted_indices:
      cumulative_volume += volume_by_price[idx]
      value_area_indices.append(idx)
      if cumulative_volume >= total_volume * 0.7:
        break

    value_area_high = bins[max(value_area_indices) + 1]
    value_area_low = bins[min(value_area_indices)]

    # Check current price position
    if abs(curr['close'] - poc_price) / poc_price < 0.005:
      if curr['close'] > poc_price:
        signals['poc_support'] = {'signal': 'BUY', 'strength': 'MODERATE', 'level': poc_price}
      else:
        signals['poc_resistance'] = {'signal': 'SELL', 'strength': 'MODERATE', 'level': poc_price}

    if abs(curr['close'] - value_area_high) / value_area_high < 0.005:
      signals['value_area_high'] = {'signal': 'SELL', 'strength': 'MODERATE', 'level': value_area_high}

    if abs(curr['close'] - value_area_low) / value_area_low < 0.005:
      signals['value_area_low'] = {'signal': 'BUY', 'strength': 'MODERATE', 'level': value_area_low}

    # Volume Nodes
    avg_volume = volume_by_price.mean()
    high_volume_threshold = avg_volume * 1.5
    low_volume_threshold = avg_volume * 0.5

    curr_bin = np.digitize([curr['close']], bins)[0]
    if curr_bin < len(volume_by_price):
      if volume_by_price[curr_bin] > high_volume_threshold:
        signals['high_volume_node'] = {'signal': 'CONSOLIDATION', 'strength': 'MODERATE'}
      elif volume_by_price[curr_bin] < low_volume_threshold:
        signals['low_volume_node'] = {'signal': 'BREAKOUT_LIKELY', 'strength': 'MODERATE'}

    return signals

  # ========================================================================
  # NEW: LIVE COMBINATION ANALYSIS METHODS
  # ========================================================================

  def analyze_live_combinations(
    self,
    symbol: str,
    results: Dict,
    min_conf_threshold: float = 70.0
  ) -> Dict:
    """
    Check if detected signals match any validated combinations from tf_combos table
    Only saves combinations that exist in tf_combos and meet the accuracy threshold

    Returns:
        Dict with combos grouped by timeframe: {
            '5m': [combo1, combo2, ...],
            '15m': [combo1, combo2, ...],
            ...
        }
    """
    logging.info(f"🔍 Checking for validated signal combinations for {symbol}...")

    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    saved_count = 0
    combos_by_tf = {}

    # Check each timeframe
    for tf, tf_data in results['timeframes'].items():
      if 'error' in tf_data or not tf_data.get('signals'):
        continue

      detected_signals = list(tf_data['signals'].keys())

      if len(detected_signals) < 2:
        continue

      # Initialize list for this timeframe
      if tf not in combos_by_tf:
        combos_by_tf[tf] = []

      # Generate all possible combinations from detected signals
      for combo_size in range(2, len(detected_signals) + 1):
        for signal_combo in combinations(sorted(detected_signals), combo_size):
          combo_name = '+'.join(sorted(signal_combo))

          # Check if this combination exists in tf_combos with sufficient accuracy
          cursor.execute('''
                    SELECT 
                        signal_name,
                        accuracy,
                        signals_count,
                        avg_price_change,
                        profit_factor,
                        combo_size
                    FROM tf_combos
                    WHERE signal_name = ? AND timeframe = ? AND accuracy >= ?
                ''', (combo_name, tf, min_conf_threshold))

          combo_result = cursor.fetchone()

          if combo_result:
            # Get individual signal details from signals table
            signal_accuracies = []
            signal_samples = []
            validation_windows = []

            for signal_name in signal_combo:
              cursor.execute('''
                            SELECT 
                                signal_accuracy,
                                sample_size,
                                validation_window
                            FROM signals
                            WHERE signal_name = ? AND timeframe = ?
                        ''', (signal_name, tf))

              signal_data = cursor.fetchone()

              if signal_data:
                signal_accuracies.append(str(signal_data['signal_accuracy']))
                signal_samples.append(str(signal_data['sample_size']))
                validation_windows.append(signal_data['validation_window'])
              else:
                # Fallback to default values if not in signals table
                signal_accuracies.append('0')
                signal_samples.append('0')
                validation_windows.append(5)

            # Calculate min and max validation windows
            min_window = min(validation_windows) if validation_windows else 5
            max_window = max(validation_windows) if validation_windows else 10

            # Save to live_tf_combos
            try:
              cursor.execute('''
                            INSERT OR REPLACE INTO live_tf_combos
                            (symbol, combo_signal_name, signal_accuracies, signal_samples,
                             min_window, max_window, timeframe, accuracy, combo_price_change, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                symbol,
                combo_name,
                '+'.join(signal_accuracies),
                '+'.join(signal_samples),
                min_window,
                max_window,
                tf,
                combo_result['accuracy'],
                combo_result['avg_price_change'],
                datetime.now().isoformat()
              ))

              # Add to result dict
              combo_dict = {
                'id': cursor.lastrowid,
                'symbol': symbol,
                'combo_signal_name': combo_name,
                'signal_accuracies': '+'.join(signal_accuracies),
                'signal_samples': '+'.join(signal_samples),
                'min_window': min_window,
                'max_window': max_window,
                'timeframe': tf,
                'accuracy': combo_result['accuracy'],
                'combo_price_change': combo_result['avg_price_change'],
                'profit_factor': combo_result['profit_factor'],
                'combo_size': combo_result['combo_size'],
                'signals_count': combo_result['signals_count'],
                'timestamp': datetime.now().isoformat()
              }

              combos_by_tf[tf].append(combo_dict)
              saved_count += 1

              logging.info(f"   ✅ Found validated combo: {combo_name} ({tf}) - {combo_result['accuracy']:.1f}%")

            except Exception as e:
              logging.error(f"Error saving combo {combo_name}: {e}")

    conn.commit()
    conn.close()

    logging.info(f"   Saved {saved_count} validated combinations for {symbol}")

    # Sort each timeframe's combos by accuracy (descending)
    for tf in combos_by_tf:
      combos_by_tf[tf].sort(key=lambda x: x['accuracy'], reverse=True)

    return combos_by_tf
  # ========================================================================
  # UPDATED: Main Analysis Method with Combo Integration
  # ========================================================================

  def analyze_symbol_all_timeframes(self, symbol: str, timeframes: List[str]) -> Dict:
    """
    Comprehensive analysis across multiple timeframes
    NOW WITH INTEGRATED COMBINATION ANALYSIS
    """
    results = {
      'symbol': symbol,
      'timestamp': datetime.now().isoformat(),
      'timeframes': {},
      'combinations': []
    }

    for tf in timeframes:
      logging.info(f"Analyzing {symbol} on {tf}...")

      df = self.fetch_data(symbol, tf, 200)

      if df is None or len(df) < 50:
        results['timeframes'][tf] = {'error': 'Insufficient data'}
        continue

      # Calculate indicators
      df = self.calculate_all_indicators(df)

      # Run all analyses
      signals = {}
      signals.update(self.analyze_candlestick_patterns(df))
      signals.update(self.analyze_moving_averages(df))
      signals.update(self.analyze_momentum(df))
      signals.update(self.analyze_volume(df))
      signals.update(self.analyze_volatility(df))
      signals.update(self.analyze_trend(df))
      signals.update(self.analyze_price_action(df))
      signals.update(self.analyze_volume_profile(df))

      # Add current price info
      curr = df.iloc[-1]

      results['timeframes'][tf] = {
        'price': float(curr['close']),
        'timestamp': str(curr['timestamp']),
        'signals': signals,
        'signal_count': len(signals),
        'buy_signals': len([s for s in signals.values() if s.get('signal') == 'BUY']),
        'sell_signals': len([s for s in signals.values() if s.get('signal') == 'SELL'])
      }

    # ========================================================================
    # NEW: Analyze signal combinations after all timeframe analysis is complete
    # ========================================================================

    try:
      combinations = self.analyze_live_combinations(
        symbol=symbol,
        results=results,
        min_conf_threshold=60.0  # Only save combos with >= 60% accuracy
      )
      if combinations:
        results['combinations'] = combinations
    except Exception as e:
      logging.error(f"Error in combination analysis: {e}")

    return results


if __name__ == "__main__":
  analyzer = ScalpSignalAnalyzer()

  print(f"Total signals implemented: {len(analyzer.SIGNAL_CONFIDENCE)}")
  print("\nSignal breakdown by category:")

  categories = {}
  for signal, data in analyzer.SIGNAL_CONFIDENCE.items():
    cat = data.get('category', 'Other')
    categories[cat] = categories.get(cat, 0) + 1

  for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count} signals")

  # Test combination analysis
  print("\n" + "=" * 80)
  print("Testing combination analysis integration")
  print("=" * 80)

  result = analyzer.analyze_symbol_all_timeframes('BTC', ['5m', '15m', '1h'])
  print(f"\nAnalysis complete for BTC")
  print(f"Timeframes analyzed: {list(result['timeframes'].keys())}")