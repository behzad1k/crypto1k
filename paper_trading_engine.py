"""
Paper Trading Engine - Complete Trading Simulation System
Manages buying queue, position monitoring, and profit/loss tracking
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
  Complete paper trading simulation system
  - Monitors signals and adds qualified coins to buying queue
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
  POSITION_SIZE_PCT = 10.0  # Use 10% of bankroll per position
  MAX_POSITIONS = 5  # Maximum concurrent positions

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

    logging.info(f"✅ Paper Trading Engine initialized")
    logging.info(f"   Initial Bankroll: ${initial_bankroll:,.2f}")
    logging.info(
      f"   Position Size: {self.POSITION_SIZE_PCT}% (${initial_bankroll * self.POSITION_SIZE_PCT / 100:,.2f})")

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
                status TEXT DEFAULT 'WAITING'
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
                status TEXT DEFAULT 'OPEN'
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
                signal_patterns INTEGER NOT NULL
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
        'status': row[11]
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
        'status': row[8]
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

  def evaluate_signal_for_entry(self, signal: Dict) -> bool:
    """
    Evaluate if a signal qualifies for adding to buying queue

    Criteria:
    - BUY signal only
    - Minimum confidence threshold
    - Minimum pattern count
    - Not already in queue or positions
    - Available bankroll for new position
    """
    # Check if BUY signal
    if signal.get('signal') != 'BUY':
      return False

    # Check confidence and patterns
    if signal.get('pattern_confidence', 0) < 0.75:  # 75% minimum
      return False

    if signal.get('pattern_count', 0) < 100:  # 100 patterns minimum
      return False

    symbol = signal['symbol']

    # Check if already in queue or position
    if symbol in self.buying_queue or symbol in self.active_positions:
      return False

    # Check if we can open new positions
    if len(self.active_positions) >= self.MAX_POSITIONS:
      logging.info(f"⏸️  Max positions reached ({self.MAX_POSITIONS}), skipping {symbol}")
      return False

    # Check available bankroll
    position_size = self.current_bankroll * (self.POSITION_SIZE_PCT / 100)
    if position_size < 10:  # Minimum $10 position
      logging.info(f"⏸️  Insufficient bankroll for new position")
      return False

    return True

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
      'status': 'WAITING'
    }

    # Save to database
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            INSERT INTO buying_queue 
            (symbol, detected_price, target_price, signal_confidence, signal_patterns, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (symbol, detected_price, target_price, signal['pattern_confidence'],
              signal['pattern_count'], expires_at))

    queue_data['id'] = cursor.lastrowid
    conn.commit()
    conn.close()

    self.buying_queue[symbol] = queue_data

    logging.info(f"📋 Added to buying queue: {symbol}")
    logging.info(f"   Detected: ${detected_price:.6f} → Target: ${target_price:.6f} (-{self.MIN_BUYING_WINDOW_PCT}%)")
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
    # Calculate position size
    position_size = self.current_bankroll * (self.POSITION_SIZE_PCT / 100)

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
      'status': 'OPEN'
    }

    # Save to database
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('''
            INSERT INTO active_positions 
            (symbol, entry_price, entry_fee, position_size, quantity, 
             target_profit_price, stop_loss_price, signal_confidence, signal_patterns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, entry_price, entry_fee, position_size, quantity,
              target_profit_price, stop_loss_price, queue_data['signal_confidence'],
              queue_data['signal_patterns']))

    position_data['id'] = cursor.lastrowid
    conn.commit()
    conn.close()

    # Update bankroll
    self.current_bankroll -= position_size
    self.save_state()

    self.active_positions[symbol] = position_data

    logging.info(f"🟢 POSITION OPENED: {symbol}")
    logging.info(f"   Entry Price: ${entry_price:.6f}")
    logging.info(f"   Position Size: ${position_size:.2f}")
    logging.info(f"   Quantity: {quantity:.6f}")
    logging.info(f"   Entry Fee: ${entry_fee:.2f}")
    logging.info(f"   Target Profit: ${target_profit_price:.6f} (+{self.MAX_PROFIT_THRESHOLD_PCT}%)")
    logging.info(f"   Stop Loss: ${stop_loss_price:.6f} (-{self.STOP_LOSS_PCT}%)")
    logging.info(f"   Remaining Bankroll: ${self.current_bankroll:.2f}")

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
             signal_confidence, signal_patterns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, entry_price, exit_price, entry_fee, exit_fee, position_size, quantity,
              profit_loss, profit_loss_pct, position['opened_at'], duration_seconds, exit_reason,
              position['signal_confidence'], position['signal_patterns']))

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

  def process_new_signal(self, signal: Dict):
    """Process a new signal from the monitoring system"""
    if not self.running:
      return

    if self.evaluate_signal_for_entry(signal):
      self.add_to_buying_queue(signal)

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

    logging.info("🚀 Paper Trading Engine STARTED")

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

  def get_stats(self) -> Dict:
    """Get current trading statistics"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM trading_state WHERE id = 1')
    state = cursor.fetchone()

    conn.close()

    if not state:
      return {}

    total_trades = state[2]
    winning_trades = state[3]
    losing_trades = state[4]
    total_profit_loss = state[5]

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
      'position_limit': self.MAX_POSITIONS
    }