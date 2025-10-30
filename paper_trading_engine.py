"""
Paper Trading Engine - MODIFIED with Combination Validation
Manages buying queue, position monitoring, and profit/loss tracking
NEW FEATURES:
- 24h price change filter (MAX_24H_CHANGE)
- Validates same-timeframe and cross-timeframe combinations
- MIN_PATTERNS_THRESHOLD: minimum number of validated combinations
- MIN_ACCURACY_THRESHOLD: minimum accuracy for combinations
"""

import sqlite3
import requests
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging
from scalp_signal_analyzer import ScalpSignalAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PaperTradingEngine:
  """
  Complete paper trading simulation system with combination validation
  - Monitors signals and adds qualified coins to buying queue
  - Validates 24h price change
  - Checks same-timeframe and cross-timeframe combinations
  - Waits for optimal entry (price drop)
  - Executes simulated trades
  - Monitors positions for exit conditions
  - Tracks performance and history
  """

  # Configuration
  MIN_BUYING_WINDOW_PCT = 1.0  # Wait for 1% price drop before buying
  MAX_BUYING_WINDOW_TIME = 600  # 10 minutes max wait time (seconds)
  MAX_PROFIT_THRESHOLD_PCT = 5.0  # Take profit at 5%
  STOP_LOSS_PCT = 2.0  # Stop loss at 2%
  EXCHANGE_FEE = 0.1  # 0.1% exchange fee
  WATCH_BOUGHT_COIN_SLEEP_TIME = 60  # Check positions every 60 seconds
  SPLIT_BANKROLL_TO = 5  # Split bankroll into 5 positions
  MAX_POSITIONS = 5  # Maximum concurrent positions (should match SPLIT_BANKROLL_TO)

  # NEW: Signal combination validation thresholds
  MAX_24H_CHANGE = 15.0  # Maximum 24h price change percentage (avoid FOMO coins)
  MIN_PATTERNS_THRESHOLD = 2  # Minimum number of validated combinations required
  MIN_ACCURACY_THRESHOLD = 60.0  # Minimum accuracy percentage for combinations

  MIN_CROSS_TF_ALIGNMENT = 3  # Minimum aligned timeframes
  MIN_STRONG_SIGNALS_PER_HOUR = 5  # Minimum strong signals/hour
  SIGNAL_COOLDOWN_HOURS = 6  # Hours between signals for same symbol
  MAX_RECENT_LOSSES = 2  # Max recent losses before blacklisting
  RECENT_LOSS_WINDOW_DAYS = 7  # Days to look back for losses
  MIN_SCALP_AGREEMENT_PCT = 66.67  # Minimum scalp agreement percentage

  def __init__(self, db_path: str = 'crypto_signals.db', initial_bankroll: float = 10000.0):
    self.db_path = db_path
    self.initial_bankroll = initial_bankroll
    self.current_bankroll = initial_bankroll

    self.running = False
    self.buying_queue = {}  # {symbol: queue_data}
    self.active_positions = {}  # {symbol: position_data}

    self.signal_analyzer = ScalpSignalAnalyzer(db_path=db_path)

    # Threading
    self.buying_monitor_thread = None
    self.position_monitor_thread = None

    self.init_database()
    self.load_state()

    # Calculate position size based on SPLIT_BANKROLL_TO
    position_size = initial_bankroll / self.SPLIT_BANKROLL_TO

    logging.info(f"✅ Paper Trading Engine initialized (with combination validation)")
    logging.info(f"   Initial Bankroll: ${initial_bankroll:,.2f}")
    logging.info(f"   Split into: {self.SPLIT_BANKROLL_TO} positions")
    logging.info(f"   Position Size: ${position_size:,.2f} each")
    logging.info(f"   Max Concurrent Positions: {self.MAX_POSITIONS}")
    logging.info(f"   NEW FILTERS:")
    logging.info(f"     Max 24h Change: {self.MAX_24H_CHANGE}%")
    logging.info(f"     Min Patterns: {self.MIN_PATTERNS_THRESHOLD}")
    logging.info(f"     Min Accuracy: {self.MIN_ACCURACY_THRESHOLD}%")

  def init_database(self):
    """Initialize database tables for paper trading"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Trading engine state
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                current_bankroll REAL NOT NULL,
                initial_bankroll REAL NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_profit_loss REAL DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

    # Buying queue
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS buying_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                detected_price REAL NOT NULL,
                target_price REAL NOT NULL,
                signal_confidence REAL NOT NULL,
                signal_patterns INTEGER NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                status TEXT DEFAULT 'WAITING',
                price_24h_change REAL,
                validated_patterns_count INTEGER,
                avg_pattern_accuracy REAL
            )
        ''')

    # Active positions
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                entry_price REAL NOT NULL,
                entry_fee REAL NOT NULL,
                position_size REAL NOT NULL,
                quantity REAL NOT NULL,
                target_profit_price REAL NOT NULL,
                stop_loss_price REAL NOT NULL,
                opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                signal_confidence REAL NOT NULL,
                signal_patterns INTEGER NOT NULL,
                status TEXT DEFAULT 'OPEN',
                price_24h_change REAL,
                validated_patterns_count INTEGER,
                avg_pattern_accuracy REAL
            )
        ''')

    # Position history (closed positions)
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                entry_fee REAL NOT NULL,
                exit_fee REAL NOT NULL,
                position_size REAL NOT NULL,
                quantity REAL NOT NULL,
                profit_loss REAL NOT NULL,
                profit_loss_pct REAL NOT NULL,
                opened_at TIMESTAMP NOT NULL,
                closed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration_seconds INTEGER,
                exit_reason TEXT NOT NULL,
                signal_confidence REAL NOT NULL,
                signal_patterns INTEGER NOT NULL,
                price_24h_change REAL,
                validated_patterns_count INTEGER,
                avg_pattern_accuracy REAL
            )
        ''')

    # Position signals (live monitoring results)
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_monitoring (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                profit_loss_pct REAL NOT NULL,
                signal_count INTEGER NOT NULL,
                buy_signals INTEGER NOT NULL,
                sell_signals INTEGER NOT NULL,
                strong_signals TEXT,
                checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (position_id) REFERENCES active_positions(id)
            )
        ''')

    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_buying_queue_symbol ON buying_queue(symbol)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_active_positions_symbol ON active_positions(symbol)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_position_history_symbol ON position_history(symbol, closed_at DESC)')
    cursor.execute(
      'CREATE INDEX IF NOT EXISTS idx_position_monitoring_position ON position_monitoring(position_id, checked_at DESC)')

    # Initialize state if not exists
    cursor.execute('SELECT COUNT(*) FROM trading_state')
    if cursor.fetchone()[0] == 0:
      cursor.execute('''
                INSERT INTO trading_state (id, current_bankroll, initial_bankroll)
                VALUES (1, ?, ?)
            ''', (self.initial_bankroll, self.initial_bankroll))

    conn.commit()
    conn.close()
    logging.info("✅ Database tables initialized")

  def load_state(self):
    """Load current trading state from database"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT current_bankroll FROM trading_state WHERE id = 1')
    result = cursor.fetchone()
    if result:
      self.current_bankroll = result[0]

    # Load active positions
    cursor.execute('SELECT * FROM active_positions WHERE status = "OPEN"')
    for row in cursor.fetchall():
      self.active_positions[row[1]] = {
        'id': row[0],
        'symbol': row[1],
        'entry_price': row[2],
        'entry_fee': row[3],
        'position_size': row[4],
        'quantity': row[5],
        'target_profit_price': row[6],
        'stop_loss_price': row[7],
        'opened_at': row[8],
        'signal_confidence': row[9],
        'signal_patterns': row[10],
        'status': row[11],
        'price_24h_change': row[12] if len(row) > 12 else None,
        'validated_patterns_count': row[13] if len(row) > 13 else None,
        'avg_pattern_accuracy': row[14] if len(row) > 14 else None
      }

    # Load buying queue
    cursor.execute('SELECT * FROM buying_queue WHERE status = "WAITING"')
    for row in cursor.fetchall():
      self.buying_queue[row[1]] = {
        'id': row[0],
        'symbol': row[1],
        'detected_price': row[2],
        'target_price': row[3],
        'signal_confidence': row[4],
        'signal_patterns': row[5],
        'added_at': row[6],
        'expires_at': row[7],
        'status': row[8],
        'price_24h_change': row[9] if len(row) > 9 else None,
        'validated_patterns_count': row[10] if len(row) > 10 else None,
        'avg_pattern_accuracy': row[11] if len(row) > 11 else None
      }

    conn.close()
    logging.info(f"✅ Loaded state: {len(self.active_positions)} positions, {len(self.buying_queue)} in queue")

  def save_state(self):
    """Save current trading state to database"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            UPDATE trading_state 
            SET current_bankroll = ?, last_updated = CURRENT_TIMESTAMP
            WHERE id = 1
        ''', (self.current_bankroll,))

    conn.commit()
    conn.close()

  def get_current_price(self, symbol: str) -> Optional[float]:
    """Fetch current price from KuCoin API"""
    try:
      url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={symbol}-USDT"
      response = requests.get(url, timeout=5)
      response.raise_for_status()
      data = response.json()

      if data.get('code') == '200000' and data.get('data'):
        return float(data['data']['price'])
    except Exception as e:
      logging.error(f"Failed to fetch price for {symbol}: {e}")

    return None

  def get_24h_price_change(self, symbol: str) -> Optional[float]:
    """
    NEW: Fetch 24h price change percentage from KuCoin API
    """
    try:
      url = f"https://api.kucoin.com/api/v1/market/stats?symbol={symbol}-USDT"
      response = requests.get(url, timeout=5)
      response.raise_for_status()
      data = response.json()

      if data.get('code') == '200000' and data.get('data'):
        change_rate = float(data['data'].get('changeRate', 0))
        return change_rate * 100  # Convert to percentage
    except Exception as e:
      logging.error(f"Failed to fetch 24h change for {symbol}: {e}")

    return None

  def get_validated_combinations(self, symbol: str) -> Tuple[int, float]:
    """
    NEW: Get validated same-timeframe and cross-timeframe combinations
    Returns: (pattern_count, avg_accuracy)
    """
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get same-timeframe combinations from live_tf_combos
    cursor.execute('''
        SELECT COUNT(*) as count, AVG(accuracy) as avg_acc
        FROM live_tf_combos
        WHERE symbol = ? AND accuracy >= ?
        AND timestamp >= datetime('now', '-2 hours')
    ''', (symbol, self.MIN_ACCURACY_THRESHOLD))

    same_tf_result = cursor.fetchone()
    same_tf_count = same_tf_result['count'] if same_tf_result else 0
    same_tf_accuracy = same_tf_result['avg_acc'] if same_tf_result and same_tf_result['avg_acc'] else 0

    # Get cross-timeframe combinations (if they exist)
    # Check if we have any cross-tf combos for this symbol
    cursor.execute('''
        SELECT COUNT(*) FROM sqlite_master 
        WHERE type='table' AND name='cross_tf_live_combos'
    ''')

    has_cross_tf_table = cursor.fetchone()[0] > 0

    cross_tf_count = 0
    cross_tf_accuracy = 0

    if has_cross_tf_table:
      cursor.execute('''
          SELECT COUNT(*) as count, AVG(accuracy) as avg_acc
          FROM cross_tf_live_combos
          WHERE symbol = ? AND accuracy >= ?
          AND timestamp >= datetime('now', '-2 hours')
      ''', (symbol, self.MIN_ACCURACY_THRESHOLD))

      cross_tf_result = cursor.fetchone()
      cross_tf_count = cross_tf_result['count'] if cross_tf_result else 0
      cross_tf_accuracy = cross_tf_result['avg_acc'] if cross_tf_result and cross_tf_result['avg_acc'] else 0

    conn.close()

    # Combine results
    total_count = same_tf_count + cross_tf_count

    if total_count > 0:
      # Weighted average of accuracies
      total_weight = same_tf_count + cross_tf_count
      weighted_accuracy = (
        (same_tf_accuracy * same_tf_count + cross_tf_accuracy * cross_tf_count) / total_weight
      )
    else:
      weighted_accuracy = 0

    logging.debug(f"   {symbol} combinations: Same-TF={same_tf_count}, Cross-TF={cross_tf_count}, Avg={weighted_accuracy:.1f}%")

    return total_count, weighted_accuracy

  """
  Enhanced Paper Trading Engine - Additional Buying Conditions
  Add these methods to your PaperTradingEngine class
  """

  def evaluate_signal_for_entry(self, signal: Dict) -> Tuple[bool, str]:
    """
    ENHANCED version with additional filters

    New filters added:
    1. Cross-timeframe alignment (3+ timeframes)
    2. Strong signal density (5+ strong signals/hour)
    3. Volume confirmation
    4. Divergence signal priority
    5. Recent loss prevention
    6. Market structure confirmation
    7. Signal cooldown (6 hours)
    8. Enhanced scalp agreement (66%+ across short TFs)
    """

    # ===== EXISTING FILTERS =====
    if signal.get('signal') != 'BUY':
      return False, "NOT_BUY_SIGNAL"

    if signal.get('pattern_confidence', 0) < 0.75:
      return False, "LOW_CONFIDENCE"

    if signal.get('pattern_count', 0) < 700:
      return False, "LOW_PATTERN_COUNT"

    symbol = signal['symbol']

    if symbol in self.buying_queue or symbol in self.active_positions:
      return False, "ALREADY_TRACKED"

    if len(self.active_positions) >= self.MAX_POSITIONS:
      return False, "MAX_POSITIONS_REACHED"

    position_size = self.initial_bankroll / self.SPLIT_BANKROLL_TO
    if self.current_bankroll < position_size:
      return False, "INSUFFICIENT_BANKROLL"

    # 24h price change check
    price_24h_change = self.get_24h_price_change(symbol)
    if price_24h_change is None:
      logging.warning(f"⚠️  Could not fetch 24h change for {symbol}, skipping")
      return False, "PRICE_DATA_UNAVAILABLE"

    if abs(price_24h_change) > self.MAX_24H_CHANGE:
      logging.info(f"❌ {symbol} rejected: 24h change {price_24h_change:+.2f}% exceeds {self.MAX_24H_CHANGE}%")
      return False, f"24H_CHANGE_TOO_HIGH ({price_24h_change:+.2f}%)"

    # Validated combinations check
    patterns_count, avg_accuracy = self.get_validated_combinations(symbol)

    if patterns_count < self.MIN_PATTERNS_THRESHOLD:
      logging.info(f"❌ {symbol} rejected: Only {patterns_count} patterns (need {self.MIN_PATTERNS_THRESHOLD})")
      return False, f"INSUFFICIENT_PATTERNS ({patterns_count})"

    if avg_accuracy < self.MIN_ACCURACY_THRESHOLD:
      logging.info(f"❌ {symbol} rejected: Avg accuracy {avg_accuracy:.1f}% (need {self.MIN_ACCURACY_THRESHOLD}%)")
      return False, f"LOW_ACCURACY ({avg_accuracy:.1f}%)"

    # ===== NEW FILTERS =====

    # 1. Cross-timeframe alignment
    is_aligned, num_tfs = self.check_cross_timeframe_alignment(symbol)
    if not is_aligned:
      logging.info(f"❌ {symbol} rejected: Only {num_tfs} aligned timeframes (need 3+)")
      return False, f"INSUFFICIENT_TF_ALIGNMENT ({num_tfs})"

    # 2. Strong signal density
    passed, reason = self.check_strong_signal_density(signal)
    if not passed:
      logging.info(f"❌ {symbol} rejected: {reason}")
      return False, reason

    # 3. Volume confirmation
    passed, reason = self.check_volume_confirmation(symbol)
    if not passed:
      logging.info(f"❌ {symbol} rejected: {reason}")
      return False, reason

    # 4. Recent loss prevention
    passed, reason = self.check_recent_loss_history(symbol)
    if not passed:
      logging.info(f"❌ {symbol} rejected: {reason}")
      return False, reason

    # 5. Market structure confirmation
    passed, reason = self.check_market_structure(symbol)
    if not passed:
      logging.info(f"❌ {symbol} rejected: {reason}")
      return False, reason

    # 6. Signal cooldown
    passed, reason = self.check_signal_cooldown(symbol)
    if not passed:
      logging.info(f"❌ {symbol} rejected: {reason}")
      return False, reason

    # 7. Enhanced scalp agreement
    passed, agreement_pct = self.check_scalp_signal_agreement(symbol)
    if not passed:
      logging.info(f"❌ {symbol} rejected: Low scalp agreement {agreement_pct:.1f}% (need 66.7%+)")
      return False, f"LOW_SCALP_AGREEMENT ({agreement_pct:.1f}%)"

    # 8. BONUS: Divergence signal priority (gives bonus points, not required)
    has_divergence, div_count = self.check_divergence_signals(symbol)
    if has_divergence:
      logging.info(f"✨ {symbol} BONUS: Has {div_count} divergence signal(s)")

    # ===== ALL CHECKS PASSED =====
    logging.info(f"✅ {symbol} passed ALL validation checks:")
    logging.info(f"   24h change: {price_24h_change:+.2f}%")
    logging.info(f"   Validated patterns: {patterns_count} (accuracy: {avg_accuracy:.1f}%)")
    logging.info(f"   Cross-TF alignment: {num_tfs} timeframes")
    logging.info(f"   Scalp agreement: {agreement_pct:.1f}%")
    if has_divergence:
      logging.info(f"   🎯 Divergence signals: {div_count}")

    # Store validation data
    signal['price_24h_change'] = price_24h_change
    signal['validated_patterns_count'] = patterns_count
    signal['avg_pattern_accuracy'] = avg_accuracy
    signal['cross_tf_alignment'] = num_tfs
    signal['scalp_agreement'] = agreement_pct
    signal['has_divergence'] = has_divergence

    return True, "PASSED"

  # ===== NEW HELPER METHODS =====

  def check_cross_timeframe_alignment(self, symbol: str) -> Tuple[bool, int]:
    """Check if BUY signals appear across multiple timeframes"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
          SELECT DISTINCT timeframe, COUNT(*) as buy_count
          FROM live_signals
          WHERE symbol = ?
            AND signal_type = 'BUY'
            AND timestamp >= datetime('now', '-2 hours')
          GROUP BY timeframe
          HAVING buy_count >= 2
      ''', (symbol,))

    aligned_tfs = cursor.fetchall()
    conn.close()

    # Require at least 3 different timeframes showing BUY signals
    return len(aligned_tfs) >= 3, len(aligned_tfs)

  def check_strong_signal_density(self, signal: Dict) -> Tuple[bool, str]:
    """Check concentration of high-confidence signals"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
          SELECT COUNT(*) as strong_count
          FROM live_signals
          WHERE symbol = ?
            AND signal_type = 'BUY'
            AND confidence >= 75
            AND timestamp >= datetime('now', '-1 hour')
      ''', (signal['symbol'],))

    strong_count = cursor.fetchone()[0]
    conn.close()

    # Require at least 5 strong BUY signals in last hour
    if strong_count < 5:
      return False, f"INSUFFICIENT_STRONG_SIGNALS ({strong_count})"

    return True, "PASSED"

  def check_volume_confirmation(self, symbol: str) -> Tuple[bool, str]:
    """Verify volume supports the move"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
          SELECT COUNT(*) as volume_signals
          FROM live_signals
          WHERE symbol = ?
            AND signal_name IN ('volume_spike_bullish', 'volume_climax_bullish', 
                                'obv_bullish', 'cmf_bullish')
            AND timestamp >= datetime('now', '-2 hours')
      ''', (symbol,))

    volume_signals = cursor.fetchone()[0]
    conn.close()

    if volume_signals < 1:
      return False, "NO_VOLUME_CONFIRMATION"

    return True, "PASSED"

  def check_divergence_signals(self, symbol: str) -> Tuple[bool, int]:
    """Check for high-confidence divergence signals"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
          SELECT COUNT(*) as div_count
          FROM live_signals
          WHERE symbol = ?
            AND signal_type = 'BUY'
            AND signal_name LIKE '%divergence%bullish%'
            AND timestamp >= datetime('now', '-4 hours')
      ''', (symbol,))

    div_count = cursor.fetchone()[0]
    conn.close()

    return div_count > 0, div_count

  def check_recent_loss_history(self, symbol: str) -> Tuple[bool, str]:
    """Avoid symbols that recently stopped us out"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
          SELECT COUNT(*) as recent_losses
          FROM position_history
          WHERE symbol = ?
            AND exit_reason IN ('STOP_LOSS', 'STRONG_SELL_SIGNALS')
            AND closed_at >= datetime('now', '-7 days')
      ''', (symbol,))

    recent_losses = cursor.fetchone()[0]
    conn.close()

    if recent_losses >= 2:
      return False, f"RECENT_LOSSES ({recent_losses})"

    return True, "PASSED"

  def check_market_structure(self, symbol: str) -> Tuple[bool, str]:
    """Check for bullish market structure signals"""
    structure_signals = [
      'break_of_structure_bullish',
      'choch_bullish',
      'higher_high',
      'support_bounce',
      'resistance_break'
    ]

    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    placeholders = ','.join('?' * len(structure_signals))
    cursor.execute(f'''
          SELECT COUNT(*) as structure_count
          FROM live_signals
          WHERE symbol = ?
            AND signal_name IN ({placeholders})
            AND timestamp >= datetime('now', '-3 hours')
      ''', [symbol] + structure_signals)

    structure_count = cursor.fetchone()[0]
    conn.close()

    if structure_count < 1:
      return False, "NO_STRUCTURE_CONFIRMATION"

    return True, "PASSED"

  def check_signal_cooldown(self, symbol: str) -> Tuple[bool, str]:
    """Prevent buying same symbol too frequently"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
          SELECT MAX(datetime_created) as last_signal
          FROM pattern_signals
          WHERE symbol = ? AND signal = 'BUY'
      ''', (symbol,))

    result = cursor.fetchone()
    conn.close()

    if result and result[0]:
      last_signal = result[0]
      last_time = datetime.fromisoformat(last_signal)
      hours_since = (datetime.now() - last_time).total_seconds() / 3600

      if hours_since < 6:  # 6 hour cooldown
        return False, f"SIGNAL_COOLDOWN ({hours_since:.1f}h)"

    return True, "PASSED"

  def check_scalp_signal_agreement(self, symbol: str) -> Tuple[bool, float]:
    """Enhanced scalp validation - require agreement across SHORT timeframes"""
    short_tfs = ['1m', '5m', '15m']

    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
          SELECT 
              timeframe,
              SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_count,
              SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_count
          FROM live_signals
          WHERE symbol = ?
            AND timeframe IN (?, ?, ?)
            AND timestamp >= datetime('now', '-30 minutes')
          GROUP BY timeframe
      ''', [symbol] + short_tfs)

    results = cursor.fetchall()
    conn.close()

    if not results:
      return False, 0.0

    buy_agreement = sum(1 for r in results if r[1] > r[2])
    agreement_pct = (buy_agreement / len(short_tfs)) * 100

    if agreement_pct < 66.67:  # At least 2 of 3 timeframes agree
      return False, agreement_pct

    return True, agreement_pct

  # ===== CONFIGURATION CONSTANTS (add to class) =====

  def add_to_buying_queue(self, signal: Dict):
    """Add qualified signal to buying queue"""
    symbol = signal['symbol']
    detected_price = signal['price']

    # Calculate target entry price (1% below detected price)
    target_price = detected_price * (1 - self.MIN_BUYING_WINDOW_PCT / 100)

    # Calculate expiry time
    expires_at = datetime.now() + timedelta(seconds=self.MAX_BUYING_WINDOW_TIME)

    queue_data = {
      'symbol': symbol,
      'detected_price': detected_price,
      'target_price': target_price,
      'signal_confidence': signal['pattern_confidence'],
      'signal_patterns': signal['pattern_count'],
      'added_at': datetime.now().isoformat(),
      'expires_at': expires_at.isoformat(),
      'status': 'WAITING',
      'price_24h_change': signal.get('price_24h_change'),
      'validated_patterns_count': signal.get('validated_patterns_count'),
      'avg_pattern_accuracy': signal.get('avg_pattern_accuracy')
    }

    # Save to database
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            INSERT INTO buying_queue 
            (symbol, detected_price, target_price, signal_confidence, signal_patterns, 
             expires_at, price_24h_change, validated_patterns_count, avg_pattern_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, detected_price, target_price, signal['pattern_confidence'],
              signal['pattern_count'], expires_at,
              queue_data['price_24h_change'],
              queue_data['validated_patterns_count'],
              queue_data['avg_pattern_accuracy']))

    queue_data['id'] = cursor.lastrowid
    conn.commit()
    conn.close()

    self.buying_queue[symbol] = queue_data

    logging.info(f"📋 Added to buying queue: {symbol}")
    logging.info(f"   Detected: ${detected_price:.6f} → Target: ${target_price:.6f} (-{self.MIN_BUYING_WINDOW_PCT}%)")
    logging.info(f"   24h change: {queue_data['price_24h_change']:+.2f}%")
    logging.info(f"   Patterns: {queue_data['validated_patterns_count']} (avg: {queue_data['avg_pattern_accuracy']:.1f}%)")
    logging.info(f"   Expires in {self.MAX_BUYING_WINDOW_TIME // 60} minutes")

  def monitor_buying_queue(self):
    """Monitor buying queue and execute entries when conditions met"""
    logging.info("🔍 Buying queue monitor started")

    while self.running:
      try:
        expired_symbols = []

        for symbol, queue_data in list(self.buying_queue.items()):
          # Check expiry
          expires_at = datetime.fromisoformat(queue_data['expires_at'])
          if datetime.now() > expires_at:
            logging.info(f"⏱️  {symbol} expired from queue (no entry opportunity)")
            expired_symbols.append(symbol)
            continue

          # Check current price
          current_price = self.get_current_price(symbol)
          if current_price is None:
            continue

          target_price = queue_data['target_price']

          # If price dropped to target or below, execute entry
          if current_price <= target_price:
            logging.info(f"✅ Entry opportunity: {symbol} @ ${current_price:.6f} (target ${target_price:.6f})")
            self.execute_entry(symbol, current_price, queue_data)
            expired_symbols.append(symbol)

        # Clean up expired/executed entries
        for symbol in expired_symbols:
          self.remove_from_queue(symbol, 'EXPIRED' if symbol in self.buying_queue else 'EXECUTED')

        time.sleep(5)  # Check every 5 seconds

      except Exception as e:
        logging.error(f"Error in buying queue monitor: {e}")
        time.sleep(10)

  def execute_entry(self, symbol: str, entry_price: float, queue_data: Dict):
    """Execute simulated buy order"""
    # Calculate position size using SPLIT_BANKROLL_TO
    position_size = self.initial_bankroll / self.SPLIT_BANKROLL_TO

    # Calculate fees
    entry_fee = position_size * (self.EXCHANGE_FEE / 100)

    # Calculate quantity (after fees)
    net_investment = position_size - entry_fee
    quantity = net_investment / entry_price

    # Calculate targets
    target_profit_price = entry_price * (1 + self.MAX_PROFIT_THRESHOLD_PCT / 100)
    stop_loss_price = entry_price * (1 - self.STOP_LOSS_PCT / 100)

    position_data = {
      'symbol': symbol,
      'entry_price': entry_price,
      'entry_fee': entry_fee,
      'position_size': position_size,
      'quantity': quantity,
      'target_profit_price': target_profit_price,
      'stop_loss_price': stop_loss_price,
      'opened_at': datetime.now().isoformat(),
      'signal_confidence': queue_data['signal_confidence'],
      'signal_patterns': queue_data['signal_patterns'],
      'status': 'OPEN',
      'price_24h_change': queue_data.get('price_24h_change'),
      'validated_patterns_count': queue_data.get('validated_patterns_count'),
      'avg_pattern_accuracy': queue_data.get('avg_pattern_accuracy')
    }

    # Save to database
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            INSERT INTO active_positions 
            (symbol, entry_price, entry_fee, position_size, quantity, 
             target_profit_price, stop_loss_price, signal_confidence, signal_patterns,
             price_24h_change, validated_patterns_count, avg_pattern_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, entry_price, entry_fee, position_size, quantity,
              target_profit_price, stop_loss_price, queue_data['signal_confidence'],
              queue_data['signal_patterns'],
              position_data['price_24h_change'],
              position_data['validated_patterns_count'],
              position_data['avg_pattern_accuracy']))

    position_data['id'] = cursor.lastrowid
    conn.commit()
    conn.close()

    # Update bankroll
    self.current_bankroll -= position_size
    self.save_state()

    self.active_positions[symbol] = position_data

    logging.info(f"🟢 POSITION OPENED: {symbol}")
    logging.info(f"   Entry Price: ${entry_price:.6f}")
    logging.info(f"   Position Size: ${position_size:.2f} ({100/self.SPLIT_BANKROLL_TO:.1f}% of initial bankroll)")
    logging.info(f"   Quantity: {quantity:.6f}")
    logging.info(f"   Entry Fee: ${entry_fee:.2f}")
    logging.info(f"   Target Profit: ${target_profit_price:.6f} (+{self.MAX_PROFIT_THRESHOLD_PCT}%)")
    logging.info(f"   Stop Loss: ${stop_loss_price:.6f} (-{self.STOP_LOSS_PCT}%)")
    logging.info(f"   24h change: {position_data['price_24h_change']:+.2f}%")
    logging.info(f"   Validated patterns: {position_data['validated_patterns_count']} (avg: {position_data['avg_pattern_accuracy']:.1f}%)")
    logging.info(f"   Remaining Bankroll: ${self.current_bankroll:.2f}")
    logging.info(f"   Active Positions: {len(self.active_positions)}/{self.MAX_POSITIONS}")

  def remove_from_queue(self, symbol: str, status: str):
    """Remove symbol from buying queue"""
    if symbol in self.buying_queue:
      conn = sqlite3.connect(self.db_path)
      cursor = conn.cursor()

      cursor.execute('''
                UPDATE buying_queue 
                SET status = ?
                WHERE symbol = ?
            ''', (status, symbol))

      conn.commit()
      conn.close()

      del self.buying_queue[symbol]

  def monitor_positions(self):
    """Monitor active positions and execute exits when conditions met"""
    logging.info("👁️  Position monitor started")

    while self.running:
      try:
        for symbol, position in list(self.active_positions.items()):
          # Get current price
          current_price = self.get_current_price(symbol)
          if current_price is None:
            continue

          # Calculate current P/L
          entry_price = position['entry_price']
          profit_loss_pct = ((current_price - entry_price) / entry_price) * 100

          # Run signal analysis
          signal_analysis = self.signal_analyzer.analyze_symbol_all_timeframes(
            symbol,
            ['1m', '5m', '15m']
          )

          # Count signals
          total_signals = 0
          buy_signals = 0
          sell_signals = 0
          strong_sell_signals = []

          for tf, tf_data in signal_analysis['timeframes'].items():
            if 'error' not in tf_data:
              total_signals += tf_data['signal_count']
              buy_signals += tf_data['buy_signals']
              sell_signals += tf_data['sell_signals']

              # Collect strong sell signals
              for sig_name, sig_data in tf_data['signals'].items():
                if sig_data.get('signal') == 'SELL' and sig_data.get('strength') in ['STRONG', 'VERY_STRONG']:
                  strong_sell_signals.append(f"{sig_name}[{tf}]")

          # Log monitoring result
          self.log_position_monitoring(
            position['id'], symbol, current_price, profit_loss_pct,
            total_signals, buy_signals, sell_signals, strong_sell_signals
          )

          # Check exit conditions
          exit_reason = None

          # 1. Take profit
          if current_price >= position['target_profit_price']:
            exit_reason = 'TAKE_PROFIT'
            logging.info(f"🎯 Take profit triggered for {symbol}")

          # 2. Stop loss
          elif current_price <= position['stop_loss_price']:
            exit_reason = 'STOP_LOSS'
            logging.info(f"🛑 Stop loss triggered for {symbol}")

          # 3. Strong sell signals
          elif len(strong_sell_signals) >= 3:
            exit_reason = 'STRONG_SELL_SIGNALS'
            logging.info(f"📉 Strong sell signals detected for {symbol}: {', '.join(strong_sell_signals[:5])}")

          # Execute exit if reason found
          if exit_reason:
            self.execute_exit(symbol, current_price, exit_reason, position)

        time.sleep(self.WATCH_BOUGHT_COIN_SLEEP_TIME)

      except Exception as e:
        logging.error(f"Error in position monitor: {e}")
        time.sleep(30)

  def log_position_monitoring(self, position_id: int, symbol: str, price: float,
                              profit_loss_pct: float, total_signals: int,
                              buy_signals: int, sell_signals: int, strong_signals: List[str]):
    """Log position monitoring check to database"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            INSERT INTO position_monitoring 
            (position_id, symbol, price, profit_loss_pct, signal_count, 
             buy_signals, sell_signals, strong_signals)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (position_id, symbol, price, profit_loss_pct, total_signals,
              buy_signals, sell_signals, ','.join(strong_signals) if strong_signals else None))

    conn.commit()
    conn.close()

  def execute_exit(self, symbol: str, exit_price: float, exit_reason: str, position: Dict):
    """Execute simulated sell order"""
    entry_price = position['entry_price']
    quantity = position['quantity']
    position_size = position['position_size']
    entry_fee = position['entry_fee']

    # Calculate exit value
    gross_exit_value = quantity * exit_price
    exit_fee = gross_exit_value * (self.EXCHANGE_FEE / 100)
    net_exit_value = gross_exit_value - exit_fee

    # Calculate P/L
    total_fees = entry_fee + exit_fee
    profit_loss = net_exit_value - (position_size - entry_fee)
    profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100

    # Calculate duration
    opened_at = datetime.fromisoformat(position['opened_at'])
    duration_seconds = int((datetime.now() - opened_at).total_seconds())

    # Save to history
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            INSERT INTO position_history 
            (symbol, entry_price, exit_price, entry_fee, exit_fee, position_size, quantity,
             profit_loss, profit_loss_pct, opened_at, duration_seconds, exit_reason,
             signal_confidence, signal_patterns, price_24h_change, validated_patterns_count, avg_pattern_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, entry_price, exit_price, entry_fee, exit_fee, position_size, quantity,
              profit_loss, profit_loss_pct, position['opened_at'], duration_seconds, exit_reason,
              position['signal_confidence'], position['signal_patterns'],
              position.get('price_24h_change'),
              position.get('validated_patterns_count'),
              position.get('avg_pattern_accuracy')))

    # Update stats
    cursor.execute('''
            UPDATE trading_state 
            SET total_trades = total_trades + 1,
                winning_trades = winning_trades + ?,
                losing_trades = losing_trades + ?,
                total_profit_loss = total_profit_loss + ?
            WHERE id = 1
        ''', (1 if profit_loss > 0 else 0, 1 if profit_loss < 0 else 0, profit_loss))

    # Remove from active positions
    cursor.execute('DELETE FROM active_positions WHERE symbol = ?', (symbol,))

    conn.commit()
    conn.close()

    # Update bankroll
    self.current_bankroll += net_exit_value
    self.save_state()

    # Remove from memory
    del self.active_positions[symbol]

    # Log
    emoji = "🟢" if profit_loss > 0 else "🔴"
    logging.info(f"{emoji} POSITION CLOSED: {symbol}")
    logging.info(f"   Reason: {exit_reason}")
    logging.info(f"   Entry: ${entry_price:.6f} → Exit: ${exit_price:.6f}")
    logging.info(f"   P/L: ${profit_loss:+.2f} ({profit_loss_pct:+.2f}%)")
    logging.info(f"   Duration: {duration_seconds // 60} minutes")
    logging.info(f"   Total Fees: ${total_fees:.2f}")
    logging.info(f"   New Bankroll: ${self.current_bankroll:.2f}")
    logging.info(f"   Active Positions: {len(self.active_positions)}/{self.MAX_POSITIONS}")

  def process_new_signal(self, signal: Dict):
    """Process a new signal from the monitoring system"""
    logging.info(f"paper is Running: {self.running}")
    # if not self.running:
    #   return

    is_valid, reason = self.evaluate_signal_for_entry(signal)

    if is_valid:
      self.add_to_buying_queue(signal)
    else:
      logging.debug(f"Signal rejected for {signal['symbol']}: {reason}")

  def start(self):
    """Start paper trading engine"""
    if self.running:
      logging.warning("⚠️  Engine already running")
      return

    self.running = True

    # Start monitoring threads
    self.buying_monitor_thread = threading.Thread(target=self.monitor_buying_queue, daemon=True)
    self.position_monitor_thread = threading.Thread(target=self.monitor_positions, daemon=True)

    self.buying_monitor_thread.start()
    self.position_monitor_thread.start()

    logging.info("🚀 Paper Trading Engine STARTED (with combination validation)")
    logging.info(f"   Monitoring {self.MAX_POSITIONS} position slots")
    logging.info(f"   Position size: ${self.initial_bankroll / self.SPLIT_BANKROLL_TO:.2f} each")
    logging.info(f"   Filters: 24h<{self.MAX_24H_CHANGE}%, patterns>={self.MIN_PATTERNS_THRESHOLD}, accuracy>={self.MIN_ACCURACY_THRESHOLD}%")

  def stop(self):
    """Stop paper trading engine"""
    if not self.running:
      return

    self.running = False

    if self.buying_monitor_thread:
      self.buying_monitor_thread.join(timeout=5)
    if self.position_monitor_thread:
      self.position_monitor_thread.join(timeout=5)

    logging.info("⏸️  Paper Trading Engine STOPPED")

  def reset(self, new_bankroll: float = None):
    """Reset trading engine (close all positions, reset bankroll)"""
    if self.running:
      logging.warning("⚠️  Stop the engine before resetting")
      return False

    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    try:
      # Close all active positions at current price
      for symbol, position in list(self.active_positions.items()):
        current_price = self.get_current_price(symbol)
        if current_price:
          self.execute_exit(symbol, current_price, 'RESET', position)

      # Clear buying queue
      cursor.execute('UPDATE buying_queue SET status = "CANCELLED" WHERE status = "WAITING"')

      # Reset state
      if new_bankroll is not None:
        self.initial_bankroll = new_bankroll

      cursor.execute('''
                UPDATE trading_state 
                SET current_bankroll = ?,
                    initial_bankroll = ?,
                    total_trades = 0,
                    winning_trades = 0,
                    losing_trades = 0,
                    total_profit_loss = 0
                WHERE id = 1
            ''', (self.initial_bankroll, self.initial_bankroll))

      conn.commit()

      # Clear memory
      self.buying_queue.clear()
      self.active_positions.clear()
      self.current_bankroll = self.initial_bankroll

      logging.info(f"✅ Paper trading reset complete")
      logging.info(f"   New bankroll: ${self.initial_bankroll:,.2f}")
      return True

    except Exception as e:
      logging.error(f"Error resetting: {e}")
      conn.rollback()
      return False
    finally:
      conn.close()

  def get_stats(self) -> Dict:
    """Get current trading statistics"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM trading_state WHERE id = 1')
    state = cursor.fetchone()

    conn.close()

    if not state:
      return {}

    total_trades = state[3]
    winning_trades = state[4]
    losing_trades = state[5]
    total_profit_loss = state[6]

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    roi = ((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll * 100)

    return {
      'running': self.running,
      'initial_bankroll': self.initial_bankroll,
      'current_bankroll': self.current_bankroll,
      'total_profit_loss': total_profit_loss,
      'roi': roi,
      'total_trades': total_trades,
      'winning_trades': winning_trades,
      'losing_trades': losing_trades,
      'win_rate': win_rate,
      'active_positions': len(self.active_positions),
      'buying_queue': len(self.buying_queue),
      'position_limit': self.MAX_POSITIONS,
      'position_size': self.initial_bankroll / self.SPLIT_BANKROLL_TO,
      'filters': {
        'max_24h_change': self.MAX_24H_CHANGE,
        'min_patterns': self.MIN_PATTERNS_THRESHOLD,
        'min_accuracy': self.MIN_ACCURACY_THRESHOLD
      }
    }