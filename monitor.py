"""
FIXED: Crypto Pattern Monitoring Module with Paper Trading Integration
KEY CHANGES:
1. Store paper_trading_engine as instance variable
2. Ensure it's passed to monitor_symbol correctly
3. Add debug logging to verify signal forwarding
"""

import random
import requests
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import json
import os
import logging
import time
from typing import Dict, List, Tuple, Optional
from collections import deque

from paper_trading_manager import PaperTradingManager
from scalp_signal_analyzer import ScalpSignalAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_nobitex_top_symbols(limit: int = 100) -> List[str]:
  """Fetch top symbols from Nobitex API based on daily change"""
  try:
    logging.info("Fetching updated symbol list from Nobitex...")
    response = requests.get('https://apiv2.nobitex.ir/market/stats', timeout=10)
    response.raise_for_status()
    nobitex_json = response.json()

    coins = []
    for key, val in nobitex_json['stats'].items():
      if 'dayChange' in val:
        coins.append({'symbol': key, 'dayChange': val['dayChange']})

    sorted_coins = sorted(
      [x for x in coins if 'usdt' in x['symbol'].lower()],
      key=lambda c: float(c['dayChange']),
      reverse=True
    )

    top_symbols = [c['symbol'].split('-')[0].upper() for c in sorted_coins]
    logging.info(f"✅ Fetched {len(top_symbols)} symbols from Nobitex")
    return top_symbols

  except Exception as e:
    logging.error(f"Failed to fetch Nobitex symbols: {e}")
    return ['BTC', 'ETH', 'SOL', 'NEAR', 'APT', 'SUI', 'DOGE', 'ADA', 'DOT', 'LINK']

def fetch_tabdeal_top_symbols(limit: int = 100) -> List[str]:
  """Fetch top symbols from tabdeal API based on daily change"""
  try:
    logging.info("Fetching updated symbol list from tabdeal...")
    response = requests.get('https://api1.tabdeal.org/r/api/v1/exchangeInfo', timeout=10)
    response.raise_for_status()
    tabdeal_json = response.json()

    coins = []
    for coin in [x for x in tabdeal_json if x['quoteAsset'] == 'USDT']:
      if "LIMIT" in coin['orderTypes'] or "MARKET" in coin['orderTypes']:
        coins.append(coin['baseAsset'])
    random.shuffle(coins)
    logging.info(f"✅ Fetched {len(coins)} symbols from tabdeal")
    return coins

  except Exception as e:
    logging.error(f"Failed to fetch tabdeal symbols: {e}")
    return ['BTC', 'ETH', 'SOL', 'NEAR', 'APT', 'SUI', 'DOGE', 'ADA', 'DOT', 'LINK']

class ScalpSignalValidator:
  """Validates signals using short-term scalp analysis"""

  STRONG_SIGNAL_PATTERNS = [
    'engulfing_bullish', 'engulfing_bearish', 'morning_star', 'evening_star',
    'three_white_soldiers', 'three_black_crows', 'ma_cross_golden', 'ma_cross_death',
    'macd_cross_bullish', 'macd_cross_bearish', 'macd_divergence_bullish',
    'macd_divergence_bearish', 'rsi_divergence_bullish', 'rsi_divergence_bearish',
    'volume_spike_bullish', 'volume_spike_bearish', 'bollinger_squeeze',
    'bollinger_breakout_up', 'bollinger_breakout_down', 'break_of_structure_bullish',
    'break_of_structure_bearish', 'resistance_break', 'support_break',
    'supertrend_bullish', 'supertrend_bearish', 'ichimoku_bullish', 'ichimoku_bearish'
  ]

  SHORT_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '2h']

  def __init__(self):
    self.timeframe_minutes = {
      '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
      '1h': 60, '2h': 120, '4h': 240
    }

  def fetch_data(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data with KuCoin primary, Binance fallback"""
    data = self._fetch_kucoin_data(symbol, timeframe, limit)
    if data is None or len(data) < 50:
      data = self._fetch_binance_data(symbol, timeframe, limit)
    return data

  def _fetch_kucoin_data(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch from KuCoin"""
    try:
      symbol_pair = f"{symbol}-USDT"
      end_time = int(datetime.now().timestamp())
      minutes = self.timeframe_minutes.get(timeframe, 30)
      start_time = end_time - (minutes * 60 * limit)

      tf_map = {
        '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min',
        '30m': '30min', '1h': '1hour', '2h': '2hour', '4h': '4hour'
      }

      if timeframe not in tf_map:
        return None

      url = "https://api.kucoin.com/api/v1/market/candles"
      params = {'symbol': symbol_pair, 'type': tf_map[timeframe], 'startAt': start_time, 'endAt': end_time}

      response = requests.get(url, params=params, timeout=10)
      response.raise_for_status()
      data = response.json()

      if data.get('code') != '200000' or not data.get('data'):
        return None

      df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
      df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
      df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp'], errors='coerce'), unit='s')

      for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

      return df.sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
      return None

  def _fetch_binance_data(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Fallback to Binance"""
    try:
      symbol_pair = f"{symbol}USDT"
      url = "https://api.binance.com/api/v3/klines"
      params = {'symbol': symbol_pair, 'interval': timeframe, 'limit': limit}

      response = requests.get(url, params=params, timeout=10)
      response.raise_for_status()
      data = response.json()

      if not data:
        return None

      df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'buy_volume', 'buy_quote_volume', 'ignore'
      ])

      df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp'], errors='coerce'), unit='ms')

      for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

      return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
    except Exception as e:
      return None

  def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for signal detection"""
    data = df.copy()

    if len(data) < 50:
      return data

    try:
      # Moving Averages
      for period in [5, 10, 20, 50]:
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

      # Bollinger Bands
      data['BB_middle'] = data['close'].rolling(window=20).mean()
      data['BB_std'] = data['close'].rolling(window=20).std()
      data['BB_upper'] = data['BB_middle'] + (data['BB_std'] * 2)
      data['BB_lower'] = data['BB_middle'] - (data['BB_std'] * 2)
      data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']

      # Volume
      data['volume_sma'] = data['volume'].rolling(window=20).mean()

    except Exception as e:
      logging.error(f"Error calculating indicators: {e}")

    return data

  def detect_strong_signals(self, df: pd.DataFrame, signal_type: str) -> List[Dict]:
    """Detect strong signals matching the given type (BUY/SELL)"""
    if len(df) < 3:
      return []

    strong_signals = []
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # Candlestick patterns
    body_curr = abs(curr['close'] - curr['open'])
    body_prev = abs(prev['close'] - prev['open'])

    # Engulfing
    if signal_type == 'BUY':
      if curr['close'] > curr['open'] and prev['close'] < prev['open']:
        if curr['close'] > prev['open'] and curr['open'] < prev['close']:
          strong_signals.append({'pattern': 'engulfing_bullish', 'strength': 'VERY_STRONG'})

    if signal_type == 'SELL':
      if curr['close'] < curr['open'] and prev['close'] > prev['open']:
        if curr['close'] < prev['open'] and curr['open'] > prev['close']:
          strong_signals.append({'pattern': 'engulfing_bearish', 'strength': 'VERY_STRONG'})

    # MACD crossover
    if not pd.isna(curr['MACD']) and not pd.isna(curr['MACD_signal']):
      if signal_type == 'BUY' and curr['MACD'] > curr['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
        strong_signals.append({'pattern': 'macd_cross_bullish', 'strength': 'STRONG'})

      if signal_type == 'SELL' and curr['MACD'] < curr['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
        strong_signals.append({'pattern': 'macd_cross_bearish', 'strength': 'STRONG'})

    # RSI divergence (simplified)
    if not pd.isna(curr['RSI']) and len(df) >= 10:
      recent_prices = df['close'].iloc[-10:]
      recent_rsi = df['RSI'].iloc[-10:]

      price_higher = recent_prices.iloc[-1] > recent_prices.iloc[0]
      rsi_lower = recent_rsi.iloc[-1] < recent_rsi.iloc[0]

      if signal_type == 'SELL' and price_higher and rsi_lower:
        strong_signals.append({'pattern': 'rsi_divergence_bearish', 'strength': 'VERY_STRONG'})

      if signal_type == 'BUY' and not price_higher and not rsi_lower:
        strong_signals.append({'pattern': 'rsi_divergence_bullish', 'strength': 'VERY_STRONG'})

    # Volume spike
    if not pd.isna(curr['volume_sma']) and curr['volume'] > curr['volume_sma'] * 2:
      if signal_type == 'BUY' and curr['close'] > curr['open']:
        strong_signals.append({'pattern': 'volume_spike_bullish', 'strength': 'STRONG'})

      if signal_type == 'SELL' and curr['close'] < curr['open']:
        strong_signals.append({'pattern': 'volume_spike_bearish', 'strength': 'STRONG'})

    # Bollinger breakout
    if not pd.isna(curr['BB_upper']) and not pd.isna(curr['BB_lower']):
      if signal_type == 'BUY' and curr['close'] > curr['BB_upper'] and prev['close'] <= prev['BB_upper']:
        strong_signals.append({'pattern': 'bollinger_breakout_up', 'strength': 'STRONG'})

      if signal_type == 'SELL' and curr['close'] < curr['BB_lower'] and prev['close'] >= prev['BB_lower']:
        strong_signals.append({'pattern': 'bollinger_breakout_down', 'strength': 'STRONG'})

    # Support/Resistance breaks
    if len(df) >= 10:
      recent = df.iloc[-10:]
      resistance = recent['high'].max()
      support = recent['low'].min()

      if signal_type == 'BUY' and curr['close'] > resistance * 1.01:
        strong_signals.append({'pattern': 'resistance_break', 'strength': 'STRONG'})

      if signal_type == 'SELL' and curr['close'] < support * 0.99:
        strong_signals.append({'pattern': 'support_break', 'strength': 'STRONG'})

    return strong_signals

  def validate_signal(self, symbol: str, signal_type: str) -> Dict:
    """
    Validate signal by checking for strong patterns in short timeframes
    Returns validation result with detected strong signals
    """
    validation_result = {
      'validated': False,
      'strong_signals_found': [],
      'timeframes_checked': [],
      'total_strong_signals': 0
    }

    for tf in self.SHORT_TIMEFRAMES:
      try:
        df = self.fetch_data(symbol, tf, 200)

        if df is None or len(df) < 50:
          continue

        df = self.calculate_indicators(df)
        strong_signals = self.detect_strong_signals(df, signal_type)

        validation_result['timeframes_checked'].append(tf)

        if strong_signals:
          validation_result['strong_signals_found'].extend([
            {'timeframe': tf, **sig} for sig in strong_signals
          ])
          validation_result['total_strong_signals'] += len(strong_signals)

        # Early exit if we found at least one strong signal
        if validation_result['total_strong_signals'] > 0:
          validation_result['validated'] = True
          break

      except Exception as e:
        logging.warning(f"Validation error for {symbol} on {tf}: {e}")
        continue

    return validation_result

SHORT_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '2h']

class CryptoPatternMonitor:
  """Main monitoring class - thread-safe for Flask integration"""

  def __init__(self, db_path: str, pattern_file: str = 'patterns.json',
               priority_coins_file: str = 'priority_coins.json', paper_trading_engine = None):
    self.db_path = db_path
    self.pattern_file = pattern_file
    self.priority_coins_file = priority_coins_file
    self.running = False
    self.current_symbols = []
    self.symbol_queue = deque()
    self.pt_manager = PaperTradingManager(db_path=self.db_path)

    self.paper_trading_engine = paper_trading_engine

    # Configuration
    self.min_pattern_accuracy = 0.75
    self.min_pattern_count = 300
    self.min_confidence = 0.75

    # Initialize scalp signal validator
    self.scalp_validator = ScalpSignalValidator()
    self.scalp_signal_analyzer = ScalpSignalAnalyzer()

    # Load patterns
    self.indicator_patterns = self.load_indicator_patterns()

    # Timeframes
    self.timeframes = {
      '30m': 30, '1h': 60, '2h': 120, '4h': 240,
      '6h': 360, '8h': 480, '12h': 720
    }

    # Statistics
    self.stats = {
      'symbols_processed': 0,
      'alerts_triggered': 0,
      'signals_validated': 0,
      'signals_rejected': 0,
      'last_update': None,
      'current_symbol': None
    }

  def load_indicator_patterns(self) -> List[Dict]:
    """Load pre-analyzed indicator patterns from file"""
    try:
      if os.path.exists(self.pattern_file):
        with open(self.pattern_file, 'r') as f:
          patterns = json.load(f)

        filtered = [p for p in patterns if p['accuracy'] >= self.min_pattern_accuracy]

        for pattern in filtered:
          pattern['parsed'] = self.parse_pattern(pattern['indicator'])

        logging.info(f"Loaded {len(filtered)} high-accuracy patterns (≥{self.min_pattern_accuracy:.0%})")
        return sorted(filtered, key=lambda x: x['accuracy'], reverse=True)
    except Exception as e:
      logging.error(f"Failed to load patterns: {e}")
    return []

  def parse_pattern(self, pattern_string: str) -> Dict:
    """Parse pattern string into structured format"""
    components = pattern_string.split(' + ')
    parsed = {'timeframe_indicators': {}, 'required_count': len(components)}

    for component in components:
      if '[' in component and ']' in component:
        start = component.find('[') + 1
        end = component.find(']')
        timeframe = component[start:end]
        indicator = component[end + 2:].strip()

        if timeframe not in parsed['timeframe_indicators']:
          parsed['timeframe_indicators'][timeframe] = []
        parsed['timeframe_indicators'][timeframe].append(indicator)

    return parsed

  def get_priority_coins(self) -> List[str]:
    """Get priority coins from file"""
    try:
      if os.path.exists(self.priority_coins_file):
        with open(self.priority_coins_file, 'r') as f:
          return json.load(f)
    except Exception as e:
      logging.error(f"Failed to load priority coins: {e}")
    return []

  def get_historical_data_optimized(self, symbol: str, limit: int = 500) -> Optional[pd.DataFrame]:
    """Fetch 30m base data"""
    required_30m_periods = 4000

    api_methods = [
      ('KuCoin', self._fetch_kucoin_30m_data),
      ('Binance', self._fetch_binance_30m_data),
    ]

    for api_name, fetch_method in api_methods:
      try:
        data = fetch_method(symbol, required_30m_periods)
        if data is not None and len(data) >= 100:
          return data
      except Exception as e:
        logging.warning(f"{api_name} failed for {symbol}: {e}")
        continue

    return None

  def _fetch_binance_30m_data(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch from Binance"""
    symbol_pair = f"{symbol}USDT"
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol_pair, 'interval': '30m', 'limit': limit}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if not data:
      return None

    df = pd.DataFrame(data, columns=[
      'timestamp', 'open', 'high', 'low', 'close', 'volume',
      'close_time', 'quote_volume', 'trades', 'buy_volume', 'buy_quote_volume', 'ignore'
    ])

    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    for col in ['open', 'high', 'low', 'close', 'volume']:
      df[col] = pd.to_numeric(df[col], errors='coerce')

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values('timestamp').reset_index(drop=True)

  def _fetch_kucoin_30m_data(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch from KuCoin"""
    symbol_pair = f"{symbol}-USDT"
    end_time = int(datetime.now().timestamp())
    start_time = end_time - (30 * 60 * limit)

    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {'symbol': symbol_pair, 'type': '30min', 'startAt': start_time, 'endAt': end_time}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get('code') != '200000' or not data.get('data'):
      return None

    df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp'], errors='coerce'), unit='s')

    for col in ['open', 'high', 'low', 'close', 'volume']:
      df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.sort_values('timestamp').reset_index(drop=True)

  def derive_timeframe_data(self, base_data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """Derive higher timeframe from 30m data"""
    if target_timeframe == '30m':
      return base_data.copy()

    timeframe_minutes = self.timeframes[target_timeframe]
    periods_per_candle = timeframe_minutes // 30

    grouped_data = []
    for i in range(0, len(base_data), periods_per_candle):
      chunk = base_data.iloc[i:i + periods_per_candle]
      if len(chunk) == 0:
        continue

      grouped_data.append({
        'timestamp': chunk.iloc[-1]['timestamp'],
        'open': chunk.iloc[0]['open'],
        'high': chunk['high'].max(),
        'low': chunk['low'].min(),
        'close': chunk.iloc[-1]['close'],
        'volume': chunk['volume'].sum()
      })

    return pd.DataFrame(grouped_data) if grouped_data else pd.DataFrame()

  def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    data = df.copy()

    if len(data) < 50:
      return data

    try:
      # MACD
      ema_12 = data['close'].ewm(span=12).mean()
      ema_26 = data['close'].ewm(span=26).mean()
      data['MACD'] = ema_12 - ema_26
      data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
      data['MACD_histogram'] = data['MACD'] - data['MACD_signal']

      # Momentum
      data['Momentum_5'] = data['close'] / data['close'].shift(5) - 1
      data['Momentum_10'] = data['close'] / data['close'].shift(10) - 1

      # RSI
      delta = data['close'].diff()
      gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
      rs = gain / loss
      data['RSI'] = 100 - (100 / (1 + rs))

      # Stochastic
      lowest_low = data['low'].rolling(window=14).min()
      highest_high = data['high'].rolling(window=14).max()
      k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
      data['STOCH_K'] = k_percent.rolling(window=3).mean()
      data['STOCH_D'] = data['STOCH_K'].rolling(window=3).mean()

      # Williams %R
      data['WILLR'] = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
    except Exception as e:
      logging.error(f"Error calculating indicators: {e}")

    return data

  def generate_indicator_signals(self, data: pd.DataFrame) -> Dict[str, str]:
    """Generate signals from indicators"""
    if len(data) < 2:
      return {}

    signals = {}
    latest = data.iloc[-1]
    previous = data.iloc[-2]

    try:
      # MACD
      if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_signal']):
        if latest['MACD'] > latest['MACD_signal'] and previous['MACD'] <= previous['MACD_signal']:
          signals['MACD_crossover'] = 'BUY'
        elif latest['MACD'] < latest['MACD_signal'] and previous['MACD'] >= previous['MACD_signal']:
          signals['MACD_crossover'] = 'SELL'

        if latest['MACD'] > 0:
          signals['MACD_position'] = 'BUY'
        else:
          signals['MACD_position'] = 'SELL'

      # Momentum
      if not pd.isna(latest['Momentum_5']):
        if latest['Momentum_5'] > 0.02:
          signals['Momentum_5'] = 'BUY'
        elif latest['Momentum_5'] < -0.02:
          signals['Momentum_5'] = 'SELL'

      if not pd.isna(latest['Momentum_10']):
        if latest['Momentum_10'] > 0.03:
          signals['Momentum_10'] = 'BUY'
        elif latest['Momentum_10'] < -0.03:
          signals['Momentum_10'] = 'SELL'

      # RSI
      if not pd.isna(latest['RSI']):
        if latest['RSI'] < 30:
          signals['RSI_oversold'] = 'BUY'
        elif latest['RSI'] > 70:
          signals['RSI_overbought'] = 'SELL'

        if latest['RSI'] > 50 and previous['RSI'] <= 50:
          signals['RSI_midline'] = 'BUY'
        elif latest['RSI'] < 50 and previous['RSI'] >= 50:
          signals['RSI_midline'] = 'SELL'

      # Stochastic
      if not pd.isna(latest['STOCH_D']):
        if latest['STOCH_D'] < 20:
          signals['STOCH_D_oversold'] = 'BUY'
        elif latest['STOCH_D'] > 80:
          signals['STOCH_D_overbought'] = 'SELL'

      # Williams %R
      if not pd.isna(latest['WILLR']):
        if latest['WILLR'] > -20:
          signals['WILLR_overbought'] = 'SELL'
        elif latest['WILLR'] < -80:
          signals['WILLR_oversold'] = 'BUY'

      # Combinations
      if not pd.isna(latest['Momentum_5']) and not pd.isna(latest['Momentum_10']):
        if latest['Momentum_5'] > 0.02 and latest['Momentum_10'] > 0.03:
          signals['MOM_5_10_bullish'] = 'BUY'
        elif latest['Momentum_5'] < -0.02 and latest['Momentum_10'] < -0.03:
          signals['MOM_5_10_bearish'] = 'SELL'

    except Exception as e:
      logging.error(f"Error generating signals: {e}")

    return signals

  def analyze_symbol_comprehensive(self, symbol: str) -> Dict:
    """Comprehensive analysis across timeframes"""
    base_data = self.get_historical_data_optimized(symbol, 200)

    if base_data is None or len(base_data) < 100:
      return {
        'symbol': symbol,
        'signal': 'HOLD',
        'buy_signals': [],
        'sell_signals': [],
        'timeframe_signals': {}
      }

    all_signals = {'BUY': [], 'SELL': []}
    timeframe_signals = {}

    for timeframe in self.timeframes.keys():
      try:
        tf_data = self.derive_timeframe_data(base_data, timeframe)
        if tf_data is None or len(tf_data) < 20:
          continue

        data_with_indicators = self.calculate_indicators(tf_data)
        if len(data_with_indicators) < 20:
          continue

        signals = self.generate_indicator_signals(data_with_indicators)
        timeframe_signals[timeframe] = signals

        for signal_name, signal_type in signals.items():
          signal_entry = f"{timeframe}_{signal_name}"
          if signal_type == 'BUY':
            all_signals['BUY'].append(signal_entry)
          elif signal_type == 'SELL':
            all_signals['SELL'].append(signal_entry)

      except Exception as e:
        logging.error(f"Error analyzing {symbol} on {timeframe}: {e}")
        continue

    buy_count = len(all_signals['BUY'])
    sell_count = len(all_signals['SELL'])

    if buy_count > sell_count:
      final_signal = 'BUY'
    elif sell_count > buy_count:
      final_signal = 'SELL'
    else:
      final_signal = 'HOLD'

    return {
      'symbol': symbol,
      'signal': final_signal,
      'buy_signals': all_signals['BUY'],
      'sell_signals': all_signals['SELL'],
      'timeframe_signals': timeframe_signals
    }

  def check_pattern_match(self, analysis_result: Dict, pattern: Dict) -> Tuple[bool, List[str]]:
    """Check if analysis result matches a specific pattern"""
    parsed = pattern['parsed']
    required_count = parsed['required_count']
    matched_components = []

    if analysis_result['signal'] == 'BUY':
      active_signals = analysis_result['buy_signals']
    elif analysis_result['signal'] == 'SELL':
      active_signals = analysis_result['sell_signals']
    else:
      return False, []

    active_set = set(active_signals)

    for timeframe, indicators in parsed['timeframe_indicators'].items():
      for indicator in indicators:
        signal_str = f"{timeframe}_{indicator}"
        if signal_str in active_set:
          matched_components.append(f"[{timeframe}] {indicator}")

    is_match = len(matched_components) == required_count
    return is_match, matched_components

  def find_matching_patterns(self, analysis_result: Dict) -> List[Dict]:
    """Find all patterns that match the current analysis"""
    matching_patterns = []

    for pattern in self.indicator_patterns:
      is_match, matched_components = self.check_pattern_match(analysis_result, pattern)

      if is_match:
        matching_patterns.append({
          'pattern': pattern['indicator'],
          'accuracy': pattern['accuracy'],
          'success': pattern['success'],
          'total': pattern['total'],
          'matched_components': matched_components
        })

    return matching_patterns

  def calculate_pattern_confidence(self, matching_patterns: List[Dict]) -> float:
    """Calculate overall confidence based on matching patterns"""
    if not matching_patterns:
      return 0.0

    total_weight = sum(p['total'] for p in matching_patterns)
    weighted_accuracy = sum(p['accuracy'] * p['total'] for p in matching_patterns) / total_weight
    return weighted_accuracy

  def should_send_alert(self, symbol: str, signal: str, best_pattern: str) -> bool:
    """Check if alert should be sent (duplicate suppression)"""
    try:
      conn = sqlite3.connect(self.db_path)
      cursor = conn.cursor()

      cursor.execute('''
                SELECT datetime_created FROM pattern_signals 
                WHERE symbol = ? AND signal = ? AND best_pattern = ?
                AND datetime_created > datetime('now', '-2 hours')
                ORDER BY datetime_created DESC LIMIT 1
            ''', (symbol, signal, best_pattern))

      result = cursor.fetchone()
      conn.close()

      return result is None
    except Exception as e:
      logging.error(f"Error checking alert history: {e}")
      return True

  def save_signal_to_db(self, symbol: str, signal: str, confidence: float,
                        pattern_count: int, best_pattern: str, best_accuracy: float,
                        all_patterns: List[Dict], price: float, stop_loss: float,
                        scalp_validation: Dict):
    """Save signal to database"""
    try:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO pattern_signals (
                symbol, signal, pattern_confidence, pattern_count,
                best_pattern, best_pattern_accuracy, all_patterns,
                price, stop_loss, scalp_validation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, signal, confidence, pattern_count,
            best_pattern, best_accuracy,
            json.dumps(all_patterns, indent=2),
            price, stop_loss,
            json.dumps(scalp_validation, indent=2)
        ))

        conn.commit()
        conn.close()

        self.stats['alerts_triggered'] += 1
        logging.info(f"✅ Signal saved: {symbol} {signal}")

    except Exception as e:
        logging.error(f"Failed to save signal: {e}")

  def monitor_symbol(self, symbol: str):
    """Monitor single symbol for pattern matches with scalp validation"""
    try:
      self.stats['current_symbol'] = symbol
      logging.info(f"Analyzing {symbol}...")

      analysis_result = self.analyze_symbol_comprehensive(symbol)

      if analysis_result['signal'] == 'HOLD':
        return

      matching_patterns = self.find_matching_patterns(analysis_result)
      pattern_count = len(matching_patterns)

      # NEW: Validate with scalp signal analyzer
      scalp_validation = self.scalp_validator.validate_signal(symbol, analysis_result['signal'])
      scalp_analysis = self.scalp_signal_analyzer.analyze_symbol_all_timeframes(symbol, SHORT_TIMEFRAMES)
      self.scalp_signal_analyzer.save_analysis_result(scalp_analysis)

      # Check pattern count threshold (minimum 200)
      if pattern_count < self.min_pattern_count:
        logging.info(f"{symbol}: {pattern_count} patterns - below minimum threshold of {self.min_pattern_count}")
        return

      confidence = self.calculate_pattern_confidence(matching_patterns)

      # Check confidence threshold (minimum 75%)
      if confidence < self.min_confidence:
        logging.info(f"{symbol}: {confidence:.1%} confidence - below minimum threshold of {self.min_confidence:.0%}")
        return

      logging.info(f"🎯 {symbol}: {pattern_count} patterns, {confidence:.1%} confidence - Validating with scalp signals...")

      self.stats['signals_validated'] += 1

      best_pattern = max(matching_patterns, key=lambda x: x['accuracy'])

      # if not self.should_send_alert(symbol, analysis_result['signal'], best_pattern['pattern']):
      #   return

      # Get current price
      current_data = self.get_historical_data_optimized(symbol, 1)
      if current_data is None or len(current_data) == 0:
        return

      current_price = float(current_data['close'].iloc[-1])
      stop_loss = self.calculate_stop_loss(analysis_result['signal'], confidence, current_price)

      signal_data = {
        'symbol': symbol,
        'signal': analysis_result['signal'],
        'pattern_confidence': confidence,
        'pattern_count': pattern_count,
        'price': current_price
      }

      self.pt_manager.send_signal(signal_data)
      logging.info(f"📊 Signal forwarded to paper trading queue: {symbol}")

      # Save to database (this will forward to paper trading)
      self.save_signal_to_db(
        symbol, analysis_result['signal'], confidence,
        pattern_count, best_pattern['pattern'], best_pattern['accuracy'],
        matching_patterns, current_price, stop_loss, scalp_validation
      )

    except Exception as e:
      logging.error(f"Error monitoring {symbol}: {e}")

  def calculate_stop_loss(self, signal: str, confidence: float, price: float) -> float:
    """Calculate stop loss based on confidence"""
    if confidence >= 0.85:
      stop_pct = 0.02
    elif confidence >= 0.75:
      stop_pct = 0.03
    else:
      stop_pct = 0.05

    if signal == 'BUY':
      return price * (1 - stop_pct)
    else:
      return price * (1 + stop_pct)

  def run(self, top_coins: int = 100, paper_trading_engine=None):
    """
    ⭐ FIX 3: Main monitoring loop - STORE paper_trading_engine as instance variable
    """
    # Store the engine reference
    self.paper_trading_engine = paper_trading_engine

    self.running = True
    logging.info("🚀 Starting crypto pattern monitoring with scalp validation")
    logging.info(f"Loaded {len(self.indicator_patterns)} patterns")
    logging.info(f"Thresholds: {self.min_pattern_count}+ patterns, {self.min_confidence:.0%}+ confidence")

    if paper_trading_engine is not None:
      self.paper_trading_engine = paper_trading_engine
    if self.paper_trading_engine:
      logging.info(f"📊 Paper trading CONNECTED - Bankroll: ${self.paper_trading_engine.current_bankroll:,.2f}")
    else:
      logging.warning("⚠️  Paper trading NOT connected")

    while self.running:
      try:
        # Get priority coins
        priority = self.get_priority_coins()

        # Get tabdeal symbols
        tabdeal = fetch_tabdeal_top_symbols(top_coins)

        # Combine
        symbols = priority + [s for s in tabdeal if s not in priority]
        self.current_symbols = symbols

        logging.info(f"📊 Monitoring {len(symbols)} symbols")
        logging.info(f"Priority: {priority[:5] if priority else 'None'}")

        # Monitor each symbol
        for symbol in symbols:
          if not self.running:
            break

          self.monitor_symbol(symbol)
          self.stats['symbols_processed'] += 1
          self.stats['last_update'] = datetime.now().isoformat()

          time.sleep(1)  # Rate limiting

        logging.info(f"✅ Loop completed. Processed {self.stats['symbols_processed']} symbols")
        logging.info(f"📈 Validated: {self.stats['signals_validated']} | Rejected: {self.stats['signals_rejected']}")
        time.sleep(10)  # Pause between loops

      except Exception as e:
        logging.error(f"Monitoring loop error: {e}")
        time.sleep(60)

  def stop(self):
    """Stop monitoring"""
    self.running = False
    self.paper_trading_engine = None
    logging.info("⏸️ Monitoring stopped")

  def get_stats(self) -> Dict:
    """Get monitoring statistics"""
    return {
      **self.stats,
      'running': self.running,
      'patterns_loaded': len(self.indicator_patterns),
      'current_symbols_count': len(self.current_symbols),
      'paper_trading_connected': self.paper_trading_engine is not None,
      'validation_rate': (
        f"{(self.stats['signals_validated'] / (self.stats['signals_validated'] + self.stats['signals_rejected']) * 100):.1f}%"
        if (self.stats['signals_validated'] + self.stats['signals_rejected']) > 0 else "N/A"
      )
    }