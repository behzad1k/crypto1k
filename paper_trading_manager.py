"""
Paper Trading Manager - Process-Safe Communication
Uses file-based signaling and database as source of truth
"""

import sqlite3
import json
import time
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)


class PaperTradingManager:
  """
  Manages paper trading state across processes using database + file signals
  """

  SIGNAL_FILE = '/tmp/paper_trading_signals.json'
  STATE_FILE = '/tmp/paper_trading_state.json'

  def __init__(self, db_path='crypto_signals.db'):
    self.db_path = db_path
    Path(self.SIGNAL_FILE).touch(exist_ok=True)
    Path(self.STATE_FILE).touch(exist_ok=True)

  def send_signal(self, signal_data: dict):
    """
    Send signal to paper trading engine via file
    """
    try:
      # Read existing signals
      with open(self.SIGNAL_FILE, 'r') as f:
        try:
          signals = json.load(f)
        except json.JSONDecodeError:
          signals = []

      # Add new signal
      signal_data['timestamp'] = datetime.now().isoformat()
      signals.append(signal_data)

      # Keep only last 100 signals
      signals = signals[-100:]

      # Write back
      with open(self.SIGNAL_FILE, 'w') as f:
        json.dump(signals, f)

      logging.info(f"✅ Signal queued: {signal_data['symbol']}")
      return True
    except Exception as e:
      logging.error(f"Failed to send signal: {e}")
      return False

  def get_pending_signals(self):
    """
    Get all pending signals and clear the queue
    """
    try:
      with open(self.SIGNAL_FILE, 'r+') as f:
        try:
          signals = json.load(f)
        except json.JSONDecodeError:
          signals = []

        # Clear file
        f.seek(0)
        f.truncate()
        json.dump([], f)

      return signals
    except Exception as e:
      logging.error(f"Failed to get signals: {e}")
      return []

  def update_state(self, state: dict):
    """
    Update paper trading state for UI
    """
    try:
      state['last_update'] = datetime.now().isoformat()
      with open(self.STATE_FILE, 'w') as f:
        json.dump(state, f)
    except Exception as e:
      logging.error(f"Failed to update state: {e}")

  def get_state(self):
    """
    Get current paper trading state
    """
    try:
      with open(self.STATE_FILE, 'r') as f:
        return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
      return {
        'running': False,
        'last_update': None
      }

  def get_stats_from_db(self):
    """
    Get fresh stats directly from database
    """
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Get state
    cursor.execute('SELECT * FROM trading_state WHERE id = 1')
    state = cursor.fetchone()

    # Get active positions count
    cursor.execute('SELECT COUNT(*) FROM active_positions WHERE status = "OPEN"')
    active_positions = cursor.fetchone()[0]

    # Get buying queue count
    cursor.execute('SELECT COUNT(*) FROM buying_queue WHERE status = "WAITING"')
    buying_queue = cursor.fetchone()[0]

    conn.close()

    if not state:
      return None

    total_trades = state[3]
    winning_trades = state[4]
    losing_trades = state[5]
    total_profit_loss = state[6]
    current_bankroll = state[1]
    initial_bankroll = state[2]

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    roi = ((current_bankroll - initial_bankroll) / initial_bankroll * 100)

    return {
      'current_bankroll': current_bankroll,
      'initial_bankroll': initial_bankroll,
      'total_profit_loss': total_profit_loss,
      'roi': roi,
      'total_trades': total_trades,
      'winning_trades': winning_trades,
      'losing_trades': losing_trades,
      'win_rate': win_rate,
      'active_positions': active_positions,
      'buying_queue': buying_queue,
      'last_update': datetime.now().isoformat()
    }