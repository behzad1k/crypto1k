"""
Pattern-Based Crypto Trading Alert System
Triggers alerts when MORE THAN 2 patterns are detected (regardless of accuracy/uniqueness)
Dynamically updates symbols from Nobitex API after each full loop
"""

import requests
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import json
import os
import logging
import sqlite3
from collections import deque, defaultdict
import threading
from typing import Dict, List, Tuple, Optional
import time

# Configure logging
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

    # Filter for USDT pairs with positive change and sort
    sorted_coins = sorted(
      [x for x in coins if 'usdt' in x['symbol'].lower() and float(x['dayChange']) > 0],
      key=lambda c: float(c['dayChange']),
      reverse=True
    )

    # Extract symbol names and convert to uppercase
    top_symbols = [c['symbol'].split('-')[0].upper() for c in sorted_coins[:limit]]

    logging.info(f"✅ Fetched {len(top_symbols)} symbols from Nobitex (top {limit} by daily change)")
    logging.info(f"Top 10: {top_symbols[:10]}")

    return top_symbols

  except Exception as e:
    logging.error(f"Failed to fetch Nobitex symbols: {e}")
    # Return fallback symbols if API fails
    return ['BTC', 'ETH', 'SOL', 'NEAR', 'APT', 'SUI', 'DOGE', 'ADA', 'DOT', 'LINK']


class PatternBasedAlertSystem:
  def __init__(self, email_config, pattern_file='indicator_patterns.json'):
    self.email_config = email_config
    self.symbol_queue = deque()
    self.symbol_stats = {}
    self.running = False
    self.db_path = 'crypto_signals.db'
    self.current_symbols = []  # Track current symbol list

    # Minimum accuracy threshold for patterns (must be set before loading)
    self.min_pattern_accuracy = 0.70  # 70% minimum accuracy

    # Load high-accuracy indicator patterns
    self.indicator_patterns = self.load_indicator_patterns(pattern_file)

    # Timeframes in minutes
    self.timeframes = {
      '30m': 30, '1h': 60, '2h': 120, '4h': 240,
      '6h': 360, '8h': 480, '12h': 720,
      # '1d': 1440
    }

    self.init_database()

  def load_indicator_patterns(self, pattern_file: str) -> List[Dict]:
    """Load pre-analyzed indicator patterns from file"""
    try:
      if os.path.exists(pattern_file):
        with open(pattern_file, 'r') as f:
          patterns = json.load(f)

        # Filter by minimum accuracy
        filtered = [p for p in patterns if p['accuracy'] >= self.min_pattern_accuracy]

        # Parse patterns for faster matching
        for pattern in filtered:
          pattern['parsed'] = self.parse_pattern(pattern['indicator'])

        logging.info(f"Loaded {len(filtered)} high-accuracy patterns (min accuracy: {self.min_pattern_accuracy:.1%})")
        return sorted(filtered, key=lambda x: x['accuracy'], reverse=True)
      else:
        logging.warning(f"Pattern file {pattern_file} not found, creating sample...")
        self.create_sample_patterns(pattern_file)
        return []
    except Exception as e:
      logging.error(f"Failed to load patterns: {e}")
      return []

  def create_sample_patterns(self, pattern_file: str):
    """Create sample pattern file"""
    sample_patterns = [
      {
        'indicator': '[12h] Momentum_5 + [2h] MACD_position + [30m] WILLR_oversold',
        'accuracy': 0.8461538461538461,
        'success': 11,
        'total': 13
      },
      {
        'indicator': '[12h] Momentum_5 + [8h] MOM_5_10_bullish + [8h] Momentum_10',
        'accuracy': 0.7647058823529411,
        'success': 13,
        'total': 17
      }
    ]

    with open(pattern_file, 'w') as f:
      json.dump(sample_patterns, f, indent=2)

    logging.info(f"Created sample pattern file: {pattern_file}")

  def calculate_pattern_uniqueness_score(self, matching_patterns: List[Dict]) -> float:
    """Calculate how unique/specific the pattern combination is"""
    if not matching_patterns:
      return 0.0

    # Factors that increase uniqueness:
    # 1. More diverse timeframes
    # 2. More specific indicators (not just generic momentum)
    # 3. Higher accuracy patterns

    all_timeframes = set()
    specific_indicators = 0
    generic_indicators = 0

    for pattern in matching_patterns:
      # Get pattern string and parse it
      pattern_str = pattern.get('pattern', '')
      components = pattern_str.split(' + ')

      for component in components:
        if '[' in component and ']' in component:
          start = component.find('[') + 1
          end = component.find(']')
          timeframe = component[start:end]
          indicator = component[end + 2:].strip()

          all_timeframes.add(timeframe)

          # Count specific vs generic indicators
          if any(specific in indicator.lower() for specific in
                 ['oversold', 'overbought', 'crossover', 'rsi_', 'stoch_', 'willr_']):
            specific_indicators += 1
          elif any(generic in indicator.lower() for generic in
                   ['momentum_5', 'momentum_10', 'mom_5_10', 'macd_position']):
            generic_indicators += 1

    # Uniqueness score based on diversity
    timeframe_diversity = len(all_timeframes) / len(self.timeframes)

    # Specificity ratio
    total_indicators = specific_indicators + generic_indicators
    specificity = specific_indicators / total_indicators if total_indicators > 0 else 0

    uniqueness = (timeframe_diversity * 0.4) + (specificity * 0.6)
    return uniqueness

  def parse_pattern(self, pattern_string: str) -> Dict:
    """Parse pattern string into structured format"""
    # Example: "[12h] Momentum_5 + [2h] MACD_position + [30m] WILLR_oversold"
    components = pattern_string.split(' + ')

    parsed = {
      'timeframe_indicators': {},
      'required_count': len(components)
    }

    for component in components:
      # Extract timeframe and indicator
      if '[' in component and ']' in component:
        start = component.find('[') + 1
        end = component.find(']')
        timeframe = component[start:end]
        indicator = component[end + 2:].strip()

        if timeframe not in parsed['timeframe_indicators']:
          parsed['timeframe_indicators'][timeframe] = []
        parsed['timeframe_indicators'][timeframe].append(indicator)

    return parsed

  def check_pattern_match(self, analysis_result: Dict, pattern: Dict) -> Tuple[bool, List[str]]:
    """Check if analysis result matches a specific pattern"""
    parsed = pattern['parsed']
    required_count = parsed['required_count']
    matched_components = []

    # Get active signals for this analysis
    if analysis_result['signal'] == 'BUY':
      active_signals = analysis_result['buy_signals']
    elif analysis_result['signal'] == 'SELL':
      active_signals = analysis_result['sell_signals']
    else:
      return False, []

    # Create a set of active timeframe-indicator combinations
    active_set = set(active_signals)

    # Check each required component
    for timeframe, indicators in parsed['timeframe_indicators'].items():
      for indicator in indicators:
        # Build the signal string that would appear in active_signals
        signal_str = f"{timeframe}_{indicator}"

        if signal_str in active_set:
          matched_components.append(f"[{timeframe}] {indicator}")

    # Pattern matches if all components are present
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
          'matched_components': matched_components,
          'parsed': pattern.get('parsed', {})
        })

    return matching_patterns

  def calculate_pattern_confidence(self, matching_patterns: List[Dict]) -> Tuple[float, Dict]:
    """Calculate overall confidence based on matching patterns"""
    if not matching_patterns:
      return 0.0, {}

    best_pattern = max(matching_patterns, key=lambda x: x['accuracy'])

    total_weight = sum(p['total'] for p in matching_patterns)
    weighted_accuracy = sum(p['accuracy'] * p['total'] for p in matching_patterns) / total_weight

    uniqueness = self.calculate_pattern_uniqueness_score(matching_patterns)

    confidence_info = {
      'best_pattern': best_pattern,
      'pattern_count': len(matching_patterns),
      'weighted_accuracy': weighted_accuracy,
      'uniqueness_score': uniqueness,
      'all_patterns': matching_patterns
    }

    return weighted_accuracy, confidence_info

  def should_send_alert(self, symbol: str, signal: str, pattern_hash: str) -> bool:
    """Check if alert should be sent based on recent history"""
    try:
      conn = sqlite3.connect(self.db_path)
      cursor = conn.cursor()

      # Check if this exact pattern was alerted for this symbol recently (last 2 hours)
      cursor.execute('''
                SELECT datetime_created FROM pattern_signals 
                WHERE symbol = ? AND signal = ? AND best_pattern = ?
                AND datetime_created > datetime('now', '-2 hours')
                ORDER BY datetime_created DESC LIMIT 1
            ''', (symbol, signal, pattern_hash))

      result = cursor.fetchone()
      conn.close()

      if result:
        logging.info(f"Duplicate alert suppressed for {symbol} - same pattern within 2h")
        return False

      return True

    except Exception as e:
      logging.error(f"Error checking alert history: {e}")
      return True  # Send alert if check fails

  def init_database(self):
    """Initialize SQLite database for pattern-based signals"""
    try:
      conn = sqlite3.connect(self.db_path)
      cursor = conn.cursor()

      cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    pattern_confidence REAL NOT NULL,
                    pattern_count INTEGER NOT NULL,
                    best_pattern TEXT NOT NULL,
                    best_pattern_accuracy REAL NOT NULL,
                    all_patterns TEXT,
                    price REAL NOT NULL,
                    stop_loss REAL,
                    datetime_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

      cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_pattern_symbol_datetime 
                ON pattern_signals(symbol, datetime_created)
            ''')

      conn.commit()
      conn.close()
      logging.info("Pattern-based database initialized")
    except Exception as e:
      logging.error(f"Failed to initialize database: {e}")
      raise

  def get_historical_data_optimized(self, symbol: str, limit: int = 500) -> Optional[pd.DataFrame]:
    """Fetch 30m base data (reuse from original code)"""
    required_30m_periods = 4000

    api_methods = [
      ('KuCoin', self._fetch_kucoin_30m_data),
      ('Binance', self._fetch_binance_30m_data),
    ]

    for api_name, fetch_method in api_methods:
      try:
        data = fetch_method(symbol, required_30m_periods)
        if data is not None and len(data) >= 100:
          logging.info(f"Fetched {len(data)} 30m periods for {symbol} from {api_name}")
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
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

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
    """Calculate technical indicators (same as original)"""
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
    """Generate signals from indicators (same as original)"""
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

  def monitor_symbol(self, symbol: str):
    """Monitor symbol with pattern matching - TRIGGERS ON >100 PATTERNS AND >0.80 CONFIDENCE"""
    try:
      logging.info(f"Analyzing {symbol}...")

      # Perform analysis
      analysis_result = self.analyze_symbol_comprehensive(symbol)

      if analysis_result['signal'] == 'HOLD':
        logging.info(f"{symbol}: HOLD - no clear direction")
        return

      # Check for pattern matches
      matching_patterns = self.find_matching_patterns(analysis_result)

      # Calculate pattern-based confidence first
      pattern_confidence, confidence_info = self.calculate_pattern_confidence(matching_patterns)

      # *** NEW THRESHOLDS: >100 patterns AND >0.80 confidence ***
      pattern_count = len(matching_patterns)

      if (pattern_count <= 300) or (pattern_count <= 800 and pattern_confidence <= 0.79):
        logging.info(f"{symbol}: {analysis_result['signal']} with {pattern_count} pattern(s) - need >100 patterns (threshold not met) or low conf{pattern_confidence:.1%}")
        return

      # if pattern_confidence <= 0.80:
      #   logging.info(f"{symbol}: {analysis_result['signal']} with {pattern_count} patterns but confidence {pattern_confidence:.1%} - need >80% (threshold not met)")
      #   return

      logging.info(f"🎯 {symbol}: {pattern_count} patterns detected with {pattern_confidence:.1%} confidence - BOTH THRESHOLDS MET!")

      # Check for duplicate alerts
      best_pattern_hash = confidence_info['best_pattern']['pattern']
      if not self.should_send_alert(symbol, analysis_result['signal'], best_pattern_hash):
        return

      # Get current price
      current_data = self.get_historical_data_optimized(symbol, 1)
      if current_data is None or len(current_data) == 0:
        logging.error(f"Could not get current price for {symbol}")
        return

      current_price = float(current_data['close'].iloc[-1])

      # Calculate stop loss based on pattern confidence
      stop_loss = self.calculate_stop_loss(analysis_result['signal'], pattern_confidence, current_price)

      # Save to database
      self.save_pattern_signal_to_db(symbol, analysis_result['signal'], pattern_confidence,
                                     confidence_info, current_price, stop_loss)

      # Send email alert
      self.send_pattern_alert(symbol, analysis_result['signal'], pattern_confidence,
                              confidence_info, current_price, stop_loss)

      uniqueness = confidence_info['uniqueness_score']
      logging.info(f"*** PATTERN ALERT TRIGGERED: {symbol} {analysis_result['signal']} - "
                   f"{pattern_count} patterns (>100 ✓), "
                   f"confidence: {pattern_confidence:.1%} (>80% ✓), "
                   f"uniqueness: {uniqueness:.1%} ***")

    except Exception as e:
      logging.error(f"Error monitoring {symbol}: {e}")

  def calculate_stop_loss(self, signal: str, confidence: float, price: float) -> float:
    """Calculate stop loss based on pattern confidence"""
    # Higher confidence = tighter stop loss
    if confidence >= 0.85:
      stop_pct = 0.02  # 2%
    elif confidence >= 0.75:
      stop_pct = 0.03  # 3%
    else:
      stop_pct = 0.05  # 5%

    if signal == 'BUY':
      return price * (1 - stop_pct)
    else:
      return price * (1 + stop_pct)

  def save_pattern_signal_to_db(self, symbol: str, signal: str, confidence: float,
                                confidence_info: Dict, price: float, stop_loss: float):
    """Save pattern-based signal to database"""
    try:
      conn = sqlite3.connect(self.db_path)
      cursor = conn.cursor()

      cursor.execute('''
                INSERT INTO pattern_signals (
                    symbol, signal, pattern_confidence, pattern_count,
                    best_pattern, best_pattern_accuracy, all_patterns,
                    price, stop_loss
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
        symbol,
        signal,
        confidence,
        confidence_info['pattern_count'],
        confidence_info['best_pattern']['pattern'],
        confidence_info['best_pattern']['accuracy'],
        json.dumps(confidence_info['all_patterns'], indent=2),
        price,
        stop_loss
      ))

      conn.commit()
      conn.close()

      logging.info(f"Pattern signal saved: {symbol} {signal}")

    except Exception as e:
      logging.error(f"Failed to save pattern signal: {e}")

  def send_pattern_alert(self, symbol: str, signal: str, confidence: float,
                         confidence_info: Dict, price: float, stop_loss: float):
    """Send email alert for pattern match"""
    try:
      recipients = self.email_config.get('recipient_emails', [])
      if not recipients:
        recipients = [self.email_config.get('recipient_email')]

      for recipient in recipients:
        if not recipient:
          continue

        msg = MIMEMultipart()
        msg['From'] = self.email_config['sender_email']
        msg['To'] = recipient
        msg['Subject'] = f"🎯 MULTIPLE PATTERNS: {signal} {symbol} - {len(confidence_info['all_patterns'])} Patterns Detected"

        body = self.create_pattern_email_body(symbol, signal, confidence,
                                              confidence_info, price, stop_loss)
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
        server.starttls()
        server.login(self.email_config['sender_email'], self.email_config['sender_password'])
        server.sendmail(self.email_config['sender_email'], recipient, msg.as_string())
        server.quit()

        logging.info(f"Pattern alert sent to {recipient}: {symbol} {signal}")

      return True

    except Exception as e:
      logging.error(f"Failed to send pattern alert: {e}")
      return False

  def create_pattern_email_body(self, symbol: str, signal: str, confidence: float,
                                confidence_info: Dict, price: float, stop_loss: float) -> str:
    """Create detailed email body for pattern match"""
    best = confidence_info['best_pattern']
    uniqueness = confidence_info.get('uniqueness_score', 0)

    # Uniqueness rating
    if uniqueness >= 0.7:
      uniqueness_rating = "VERY SPECIFIC"
    elif uniqueness >= 0.5:
      uniqueness_rating = "MODERATELY SPECIFIC"
    elif uniqueness >= 0.3:
      uniqueness_rating = "SOMEWHAT GENERIC"
    else:
      uniqueness_rating = "VERY GENERIC"

    body = f"""
🎯 MULTIPLE PATTERNS DETECTED 🎯

SYMBOL: {symbol}
SIGNAL: {signal}
CURRENT PRICE: ${price:,.4f}

📊 PATTERN DETECTION:
• Total Patterns Matched: {confidence_info['pattern_count']} (THRESHOLD MET: >100 ✓)
• Overall Confidence: {confidence:.1%} (THRESHOLD MET: >80% ✓)
• Best Pattern Accuracy: {best['accuracy']:.1%} ({best['success']}/{best['total']} trades)
• Pattern Uniqueness: {uniqueness:.1%} ({uniqueness_rating})

🏆 BEST MATCHING PATTERN:
{best['pattern']}

Matched Components:
"""
    for component in best['matched_components'][:30]:
      body += f"  ✓ {component}\n"

    body += f"\n📋 ALL {confidence_info['pattern_count']} MATCHING PATTERNS:\n"
    for i, pattern in enumerate(confidence_info['all_patterns'], 1):
      body += f"\n{i}. Accuracy: {pattern['accuracy']:.1%} ({pattern['success']}/{pattern['total']})\n"
      body += f"   {pattern['pattern']}\n"

    body += f"""

💰 TRADING PARAMETERS:
• Entry Price: ${price:,.4f}
• Stop Loss: ${stop_loss:,.4f}
• Risk: {abs((stop_loss - price) / price * 100):.2f}%

⚠️ RISK MANAGEMENT:
• Multiple patterns detected ({confidence_info['pattern_count']} total - >100 threshold met ✓)
• High confidence level: {confidence:.1%} (>80% threshold met ✓)
• Pattern accuracy: {best['accuracy']:.1%} success rate
• Pattern specificity: {uniqueness:.1%} ({uniqueness_rating})
• Always use 1-3% position sizing
• Set stop loss at ${stop_loss:,.4f}
• Monitor price action closely

📊 PATTERN QUALITY ANALYSIS:
"""

    # Add quality notes based on uniqueness
    if uniqueness >= 0.7:
      body += "• ✅ EXCELLENT: Highly specific pattern with diverse indicators\n"
    elif uniqueness >= 0.5:
      body += "• ✅ GOOD: Moderately specific pattern with reasonable diversity\n"
    elif uniqueness >= 0.3:
      body += "• ⚠️ FAIR: Pattern is somewhat generic, use caution\n"
    else:
      body += "• ⚠️ LOW: Pattern is very generic, high risk of false signals\n"

    body += f"""

📅 Alert Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Pattern-Based Crypto Alert System
Alert Triggers: {confidence_info['pattern_count']} patterns (>100) AND {confidence:.1%} confidence (>80%)
Historical Accuracy: {confidence:.1%}
Pattern Specificity: {uniqueness:.1%}
        """

    return body

  def update_symbols_from_nobitex(self, limit: int = 100):
    """Update symbol list from Nobitex API"""
    try:
      new_symbols = fetch_nobitex_top_symbols(limit)

      if new_symbols and len(new_symbols) > 0:
        self.current_symbols = new_symbols
        self.symbol_queue = deque(new_symbols)
        logging.info(f"✅ Symbol list updated: {len(new_symbols)} symbols")
        logging.info(f"Top 10: {new_symbols[:10]}")
        return True
      else:
        logging.warning("Failed to fetch new symbols, keeping current list")
        return False

    except Exception as e:
      logging.error(f"Error updating symbols from Nobitex: {e}")
      return False

  def run_continuous_monitoring(self, initial_symbols: List[str] = None, update_interval: int = 100):
    """Run continuous monitoring with dynamic symbol updates"""
    # Initialize with provided symbols or fetch from Nobitex
    if initial_symbols:
      self.current_symbols = initial_symbols
      self.symbol_queue = deque(initial_symbols)
      logging.info(f"Starting with provided symbols: {len(initial_symbols)}")
    else:
      logging.info("No initial symbols provided, fetching from Nobitex...")
      if not self.update_symbols_from_nobitex(update_interval):
        logging.error("Failed to fetch initial symbols, exiting")
        return

    self.running = True
    symbols_processed = 0

    logging.info(f"Starting pattern-based monitoring for {len(self.current_symbols)} symbols")
    logging.info(f"Loaded {len(self.indicator_patterns)} high-accuracy patterns")
    logging.info(f"Minimum pattern accuracy: {self.min_pattern_accuracy:.1%}")
    logging.info(f"⚠️ ALERT THRESHOLDS: >100 PATTERNS AND >80% CONFIDENCE (BOTH REQUIRED)")
    logging.info(f"🔄 Symbol list will update from Nobitex after every full loop")

    print(f"\n{'='*80}")
    print(f"Pattern-Based Monitoring Started")
    print(f"{'='*80}")
    print(f"Initial Symbols: {len(self.current_symbols)}")
    print(f"Top 10: {self.current_symbols[:10]}")
    print(f"Patterns: {len(self.indicator_patterns)} high-accuracy combinations")
    print(f"Alert Thresholds: >100 patterns AND >80% confidence (BOTH REQUIRED)")
    print(f"Symbol Update: After every {len(self.current_symbols)} symbols analyzed")
    print(f"Press Ctrl+C to stop\n")

    try:
      while self.running:
        if not self.symbol_queue:
          logging.warning("Symbol queue empty, refilling...")
          self.symbol_queue = deque(self.current_symbols)

        symbol = self.symbol_queue.popleft()

        # Monitor the symbol
        self.monitor_symbol(symbol)
        symbols_processed += 1

        # Check if we've completed a full loop
        if symbols_processed % len(self.current_symbols) == 0:
          logging.info(f"\n{'='*80}")
          logging.info(f"🔄 FULL LOOP COMPLETED: {symbols_processed} symbols analyzed")
          logging.info(f"Fetching updated symbol list from Nobitex...")
          logging.info(f"{'='*80}\n")

          # Update symbols from Nobitex
          update_success = self.update_symbols_from_nobitex(update_interval)

          if update_success:
            logging.info(f"✅ Symbol list refreshed with {len(self.current_symbols)} coins")
          else:
            logging.warning("⚠️ Symbol update failed, continuing with current list")
        else:
          # Add symbol back to queue for next loop
          self.symbol_queue.append(symbol)

        time.sleep(1)  # Small delay between symbols

    except KeyboardInterrupt:
      logging.info("Monitoring stopped by user")
      self.running = False


def load_email_config() -> Optional[Dict]:
  """Load email configuration"""
  try:
    with open('email_config.json', 'r') as f:
      config = json.load(f)

    if 'recipient_emails' not in config and 'recipient_email' in config:
      config['recipient_emails'] = [config['recipient_email']]

    return config
  except Exception as e:
    logging.error(f"Failed to load email config: {e}")
    return None


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='Pattern-Based Crypto Alert System with Dynamic Nobitex Updates')
  parser.add_argument('--symbols', nargs='+', default=None,
                      help='Initial symbols to monitor (if not provided, will fetch from Nobitex)')
  parser.add_argument('--patterns', default='patterns.json',
                      help='Path to indicator patterns file')
  parser.add_argument('--min-accuracy', type=float, default=0.70,
                      help='Minimum pattern accuracy threshold (0-1)')
  parser.add_argument('--top-coins', type=int, default=100,
                      help='Number of top coins to fetch from Nobitex (default: 100)')
  parser.add_argument('--show-patterns', action='store_true',
                      help='Show loaded patterns and exit')
  parser.add_argument('--test-pattern', type=str,
                      help='Test a specific symbol for pattern matches')
  parser.add_argument('--test-nobitex', action='store_true',
                      help='Test Nobitex API and show top coins')

  args = parser.parse_args()

  # Test Nobitex API
  if args.test_nobitex:
    print("\n🧪 Testing Nobitex API...")
    print("="*80)
    symbols = fetch_nobitex_top_symbols(args.top_coins)
    if symbols:
      print(f"✅ Successfully fetched {len(symbols)} symbols")
      print(f"\nTop 20 symbols by daily change:")
      for i, sym in enumerate(symbols[:20], 1):
        print(f"  {i:2d}. {sym}")
    else:
      print("❌ Failed to fetch symbols from Nobitex")
    exit(0)

  email_config = load_email_config()
  if not email_config:
    print("❌ Email configuration required")
    print("Create email_config.json with your SMTP settings")
    exit(1)

  # Initialize system
  system = PatternBasedAlertSystem(email_config, args.patterns)
  system.min_pattern_accuracy = args.min_accuracy

  # Show patterns command
  if args.show_patterns:
    print(f"\n📊 LOADED PATTERNS (Minimum Accuracy: {args.min_accuracy:.1%})")
    print("=" * 100)

    if not system.indicator_patterns:
      print("No patterns loaded. Check your pattern file.")
      exit(0)

    for i, pattern in enumerate(system.indicator_patterns, 1):
      print(f"\n{i}. Accuracy: {pattern['accuracy']:.1%} "
            f"({pattern['success']}/{pattern['total']} trades)")
      print(f"   Pattern: {pattern['indicator']}")

      # Show parsed components
      parsed = pattern['parsed']
      print(f"   Components: {parsed['required_count']}")
      for tf, indicators in parsed['timeframe_indicators'].items():
        for ind in indicators:
          print(f"     • [{tf}] {ind}")

    print(f"\n✅ Total: {len(system.indicator_patterns)} high-accuracy patterns loaded")
    exit(0)

  # Test pattern command
  if args.test_pattern:
    symbol = args.test_pattern.upper()
    print(f"\n🧪 Testing pattern matching for {symbol}...")
    print("=" * 80)

    try:
      # Analyze symbol
      analysis = system.analyze_symbol_comprehensive(symbol)

      print(f"\nAnalysis Results:")
      print(f"Signal: {analysis['signal']}")
      print(f"Buy Signals: {len(analysis['buy_signals'])}")
      print(f"Sell Signals: {len(analysis['sell_signals'])}")

      # Check for patterns
      matching_patterns = system.find_matching_patterns(analysis)

      if not matching_patterns:
        print(f"\n❌ No high-accuracy patterns matched for {symbol}")
        print(f"\nActive signals that didn't match patterns:")
        active_signals = (analysis['buy_signals'] if analysis['signal'] == 'BUY'
                          else analysis['sell_signals'])
        for sig in active_signals[:10]:
          print(f"  • {sig}")
      else:
        confidence, info = system.calculate_pattern_confidence(matching_patterns)
        print(f"\n✅ {len(matching_patterns)} PATTERN(S) MATCHED!")

        # Check if threshold is met
        if len(matching_patterns) > 2:
          print(f"🎯 ALERT THRESHOLD MET: {len(matching_patterns)} > 2 patterns!")
        else:
          print(f"⚠️ ALERT THRESHOLD NOT MET: {len(matching_patterns)} <= 2 patterns")

        print(f"Overall Confidence: {confidence:.1%}")
        print(f"\nMatching Patterns:")

        for i, p in enumerate(matching_patterns, 1):
          print(f"\n{i}. {p['pattern']}")
          print(f"   Accuracy: {p['accuracy']:.1%} ({p['success']}/{p['total']})")
          print(f"   Matched Components:")
          for comp in p['matched_components']:
            print(f"     ✓ {comp}")

    except Exception as e:
      print(f"❌ Error testing {symbol}: {e}")
      import traceback
      traceback.print_exc()

    exit(0)

  # Start monitoring
  print("\n🚀 Starting Pattern-Based Crypto Alert System")
  print("=" * 80)
  print(f"Patterns Loaded: {len(system.indicator_patterns)}")
  print(f"Min Accuracy: {args.min_accuracy:.1%}")
  print(f"⚠️ ALERT THRESHOLDS (BOTH REQUIRED):")
  print(f"   • Pattern Count: >100 patterns")
  print(f"   • Confidence Level: >80%")
  print(f"🔄 DYNAMIC UPDATES: Symbol list refreshes from Nobitex after each full loop")
  print(f"Top Coins to Track: {args.top_coins}")

  if args.symbols:
    print(f"Initial Symbols (Manual): {args.symbols}")
  else:
    print(f"Initial Symbols: Will fetch top {args.top_coins} from Nobitex")

  print(f"Alert Recipients: {len(email_config['recipient_emails'])}")
  print("\nMonitoring for multiple pattern matches...")
  print("Press Ctrl+C to stop\n")

  try:
    system.run_continuous_monitoring(args.symbols, args.top_coins)
  except KeyboardInterrupt:
    print("\n\n✅ Monitoring stopped")
  except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()