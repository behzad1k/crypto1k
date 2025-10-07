"""
Complete Flask Application with Integrated Crypto Pattern Monitoring
"""
import time

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
import sqlite3
import json
import os
from datetime import datetime, timedelta
from functools import wraps
import pandas as pd
from io import BytesIO
import threading
import logging
# Import our monitoring module
from monitor import CryptoPatternMonitor
from scalp_signal_analyzer import ScalpSignalAnalyzer
from live_analysis_handler import LiveAnalysisDB
from trading_position_manager import TradingPositionManager
from signal_fact_checker import SignalFactChecker
from flask_sock import Sock  # pip install flask-sock

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = 'sogol'  # IMPORTANT: Change this!
app.config['DB_PATH'] = 'crypto_signals.db'
app.config['PRIORITY_COINS_FILE'] = 'priority_coins.json'
app.config['PATTERNS_FILE'] = 'patterns.json'

# Global monitoring instance
monitor = None
monitor_thread = None
live_analyzer = None
live_db = None
trading_manager = None
fact_checker = None
sock = None


# ==================== INITIALIZATION FUNCTION ====================

def init_database():
  """Initialize database if it doesn't exist"""
  conn = sqlite3.connect(app.config['DB_PATH'])
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
            scalp_validation TEXT,
            datetime_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

  cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_symbol_datetime 
        ON pattern_signals(symbol, datetime_created)
    ''')

  conn.commit()
  conn.close()
  logging.info("✅ Database initialized")


def initialize_app():
  """Initialize all modules - called on import for gunicorn compatibility"""
  global live_analyzer, live_db, trading_manager, fact_checker, sock

  # Only initialize once
  if live_analyzer is not None:
    return

  logging.info("🚀 Initializing application...")

  # Initialize database
  init_database()
  # Initialize live analysis
  try:
    live_analyzer = ScalpSignalAnalyzer()
    live_db = LiveAnalysisDB()
    logging.info("✅ Live analysis system initialized")
  except Exception as e:
    logging.error(f"❌ Failed to initialize live analysis: {e}")

  # Initialize trading modules
  try:
    trading_manager = TradingPositionManager(db_path=app.config['DB_PATH'])
    fact_checker = SignalFactChecker(db_path=app.config['DB_PATH'])
    sock = Sock(app)
    logging.info("✅ Trading management system initialized")
  except Exception as e:
    logging.error(f"❌ Failed to initialize trading modules: {e}")
    logging.error("Trading features will not be available!")


# Initialize immediately on import (works with both gunicorn and python app.py)
initialize_app()


# ==================== AUTH ====================

def login_required(f):
  @wraps(f)
  def decorated_function(*args, **kwargs):
    if 'logged_in' not in session:
      return redirect(url_for('login'))
    return f(*args, **kwargs)

  return decorated_function


@app.route('/login', methods=['GET', 'POST'])
def login():
  if request.method == 'POST':
    data = request.get_json()
    if data.get('username') == 'iheartsogol' and data.get('password') == 'sogolpleasecomeback:(((((':
      session['logged_in'] = True
      return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
  return render_template('login.html')


@app.route('/logout')
def logout():
  session.pop('logged_in', None)
  return redirect(url_for('login'))


# ==================== ROUTES ====================

@app.route('/')
@login_required
def index():
  return render_template('index.html')


@app.route('/api/signals')
@login_required
def get_signals():
  page = int(request.args.get('page', 1))
  per_page = 10
  min_accuracy = request.args.get('minAccuracy', type=float)
  min_patterns = request.args.get('minPatterns', type=int)
  symbol = request.args.get('symbol', '').upper()

  conn = sqlite3.connect(app.config['DB_PATH'])
  conn.row_factory = sqlite3.Row
  cursor = conn.cursor()

  # Build query
  query = "SELECT * FROM pattern_signals WHERE 1=1"
  params = []

  if min_accuracy:
    query += " AND pattern_confidence >= ?"
    params.append(min_accuracy)
  if min_patterns:
    query += " AND pattern_count >= ?"
    params.append(min_patterns)
  if symbol:
    query += " AND symbol LIKE ?"
    params.append(f'%{symbol}%')

  # Get total count
  count_query = query.replace('SELECT *', 'SELECT COUNT(*)')
  cursor.execute(count_query, params)
  total = cursor.fetchone()[0]

  # Get paginated results
  query += " ORDER BY datetime_created DESC LIMIT ? OFFSET ?"
  params.extend([per_page, (page - 1) * per_page])

  cursor.execute(query, params)
  signals = []
  for row in cursor.fetchall():
    signals.append({
      'id': row['id'],
      'symbol': row['symbol'],
      'signal': row['signal'],
      'pattern_confidence': row['pattern_confidence'],
      'pattern_count': row['pattern_count'],
      'price': row['price'],
      'stop_loss': row['stop_loss'],
      'datetime_created': row['datetime_created'],
      'validity_hours': calculate_validity_hours(row['pattern_confidence'], row['pattern_count'])
    })

  conn.close()

  return jsonify({
    'signals': signals,
    'total': total,
    'pages': (total + per_page - 1) // per_page
  })


@app.route('/api/symbols/<symbol>')
@login_required
def get_symbol_history(symbol):
  conn = sqlite3.connect(app.config['DB_PATH'])
  conn.row_factory = sqlite3.Row
  cursor = conn.cursor()

  cursor.execute('''
        SELECT * FROM pattern_signals 
        WHERE symbol = ? 
        ORDER BY datetime_created DESC
    ''', (symbol,))

  history = []
  for row in cursor.fetchall():
    history.append({
      'id': row['id'],
      'signal': row['signal'],
      'pattern_confidence': row['pattern_confidence'],
      'pattern_count': row['pattern_count'],
      'price': row['price'],
      'stop_loss': row['stop_loss'],
      'datetime_created': row['datetime_created'],
      'validity_hours': calculate_validity_hours(row['pattern_confidence'], row['pattern_count'])
    })

  conn.close()
  return jsonify(history)


@app.route('/api/priority-coins', methods=['GET', 'POST'])
@login_required
def priority_coins():
  if request.method == 'POST':
    data = request.get_json()
    coins = data.get('coins', [])
    with open(app.config['PRIORITY_COINS_FILE'], 'w') as f:
      json.dump(coins, f)
    return jsonify({'success': True})
  else:
    if os.path.exists(app.config['PRIORITY_COINS_FILE']):
      with open(app.config['PRIORITY_COINS_FILE'], 'r') as f:
        coins = json.load(f)
    else:
      coins = []
    return jsonify(coins)


@app.route('/api/export')
@login_required
def export_signals():
  min_accuracy = request.args.get('minAccuracy', type=float)
  min_patterns = request.args.get('minPatterns', type=int)
  symbol = request.args.get('symbol', '').upper()

  conn = sqlite3.connect(app.config['DB_PATH'])

  query = "SELECT * FROM pattern_signals WHERE 1=1"
  params = []

  if min_accuracy:
    query += " AND pattern_confidence >= ?"
    params.append(min_accuracy)
  if min_patterns:
    query += " AND pattern_count >= ?"
    params.append(min_patterns)
  if symbol:
    query += " AND symbol LIKE ?"
    params.append(f'%{symbol}%')

  query += " ORDER BY datetime_created DESC"

  df = pd.read_sql_query(query, conn, params=params)
  conn.close()

  # Add validity hours
  df['validity_hours'] = df.apply(
    lambda row: calculate_validity_hours(row['pattern_confidence'], row['pattern_count']),
    axis=1
  )

  # Create Excel file
  output = BytesIO()
  with pd.ExcelWriter(output, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Signals')
  output.seek(0)

  return send_file(
    output,
    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    as_attachment=True,
    download_name=f'crypto_signals_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
  )


# ==================== MONITORING CONTROL ====================

@app.route('/api/monitor/status')
@login_required
def monitor_status():
  global monitor

  if monitor is None:
    return jsonify({
      'running': False,
      'message': 'Monitor not initialized',
      'symbols_processed': 0,
      'alerts_triggered': 0,
      'current_symbol': None,
      'patterns_loaded': 0,
      'last_update': None,
      'current_symbols_count': 0
    })

  # Get stats from monitor
  stats = monitor.get_stats()

  # Return consistent status
  return jsonify({
    'running': monitor.running,
    'symbols_processed': stats.get('symbols_processed', 0),
    'alerts_triggered': stats.get('alerts_triggered', 0),
    'current_symbol': stats.get('current_symbol'),
    'patterns_loaded': len(monitor.indicator_patterns) if hasattr(monitor, 'indicator_patterns') else 0,
    'last_update': stats.get('last_update'),
    'current_symbols_count': len(monitor.current_symbols) if hasattr(monitor, 'current_symbols') else 0,
    'message': 'Running' if monitor.running else 'Stopped'
  })


# REPLACE your start_monitor route with this:
@app.route('/api/monitor/start')
@login_required
def start_monitor():
  global monitor, monitor_thread

  # Check if already running
  if monitor is not None and monitor.running:
    return jsonify({
      'success': False,
      'message': 'Monitor is already running'
    })

  try:
    # Stop any existing monitor first
    if monitor is not None:
      monitor.stop()
      if monitor_thread and monitor_thread.is_alive():
        monitor_thread.join(timeout=2)

    # Create fresh monitor instance
    monitor = CryptoPatternMonitor(
      db_path=app.config['DB_PATH'],
      pattern_file=app.config['PATTERNS_FILE'],
      priority_coins_file=app.config['PRIORITY_COINS_FILE']
    )

    # Start monitoring in background thread
    monitor_thread = threading.Thread(
      target=monitor.run,
      args=(100,),
      daemon=True
    )
    monitor_thread.start()

    # Give it a moment to start
    time.sleep(0.5)

    logging.info("✅ Monitoring started successfully")
    return jsonify({
      'success': True,
      'message': 'Monitoring started'
    })

  except Exception as e:
    logging.error(f"Failed to start monitoring: {e}")
    return jsonify({
      'success': False,
      'message': f'Failed to start: {str(e)}'
    }), 500


# REPLACE your stop_monitor route with this:
@app.route('/api/monitor/stop')
@login_required
def stop_monitor():
  global monitor, monitor_thread

  if monitor is None or not monitor.running:
    return jsonify({
      'success': False,
      'message': 'Monitor is not running'
    })

  try:
    monitor.stop()

    # Wait for thread to finish (with timeout)
    if monitor_thread and monitor_thread.is_alive():
      monitor_thread.join(timeout=3)

    logging.info("⏸️ Monitoring stopped")
    return jsonify({
      'success': True,
      'message': 'Monitoring stopped'
    })

  except Exception as e:
    logging.error(f"Failed to stop monitoring: {e}")
    return jsonify({
      'success': False,
      'message': f'Failed to stop: {str(e)}'
    }), 500


# ==================== LIVE ANALYSIS ROUTES ====================


def get_signal_description(signal_name: str) -> str:
  """Get human-readable description of signal"""
  descriptions = {
    # Candlestick patterns
    'engulfing_bullish': 'Bullish engulfing pattern - strong reversal signal',
    'engulfing_bearish': 'Bearish engulfing pattern - strong reversal signal',
    'hammer': 'Hammer pattern - bullish reversal at support',
    'shooting_star': 'Shooting star - bearish reversal at resistance',
    'doji_reversal': 'Doji candle - indecision and potential reversal',
    'morning_star': 'Morning star pattern - strong bullish reversal',
    'evening_star': 'Evening star pattern - strong bearish reversal',
    'three_white_soldiers': 'Three white soldiers - strong bullish continuation',
    'three_black_crows': 'Three black crows - strong bearish continuation',

    # Moving averages
    'ma_cross_golden': 'Golden cross - 50 MA crosses above 200 MA (bullish)',
    'ma_cross_death': 'Death cross - 50 MA crosses below 200 MA (bearish)',
    'price_above_ma20': 'Price crosses above 20 MA (bullish)',
    'price_below_ma20': 'Price crosses below 20 MA (bearish)',
    'ma_ribbon_bullish': 'MA ribbon aligned bullish (5>10>20 EMAs)',
    'ma_ribbon_bearish': 'MA ribbon aligned bearish (5<10<20 EMAs)',

    # Momentum
    'rsi_oversold': 'RSI below 30 - oversold condition',
    'rsi_overbought': 'RSI above 70 - overbought condition',
    'rsi_divergence_bullish': 'Bullish RSI divergence - very strong signal',
    'rsi_divergence_bearish': 'Bearish RSI divergence - very strong signal',
    'macd_cross_bullish': 'MACD bullish crossover',
    'macd_cross_bearish': 'MACD bearish crossover',
    'macd_divergence_bullish': 'Bullish MACD divergence',
    'macd_divergence_bearish': 'Bearish MACD divergence',
    'stoch_oversold': 'Stochastic oversold (<20)',
    'stoch_overbought': 'Stochastic overbought (>80)',

    # Volume
    'volume_spike_bullish': 'Volume spike on bullish candle',
    'volume_spike_bearish': 'Volume spike on bearish candle',
    'volume_divergence_bullish': 'Bullish volume divergence',
    'volume_divergence_bearish': 'Bearish volume divergence',
    'obv_bullish': 'On-Balance Volume trending up',
    'obv_bearish': 'On-Balance Volume trending down',
    'vwap_cross_above': 'Price crosses above VWAP',
    'vwap_cross_below': 'Price crosses below VWAP',

    # Volatility
    'bollinger_squeeze': 'Bollinger Band squeeze - breakout pending',
    'bollinger_breakout_up': 'Price breaks above upper Bollinger Band',
    'bollinger_breakout_down': 'Price breaks below lower Bollinger Band',
    'bollinger_bounce_up': 'Price bounces off lower Bollinger Band',
    'bollinger_bounce_down': 'Price bounces off upper Bollinger Band',
    'atr_expansion': 'ATR expanding - increased volatility',

    # Trend
    'adx_strong_trend': 'ADX >25 - strong trend in place',
    'adx_weak_trend': 'ADX <20 - weak or no trend',
    'supertrend_bullish': 'SuperTrend indicator bullish',
    'supertrend_bearish': 'SuperTrend indicator bearish',
    'parabolic_sar_flip_bullish': 'Parabolic SAR flips bullish',
    'parabolic_sar_flip_bearish': 'Parabolic SAR flips bearish',

    # Price action
    'higher_high': 'Price making higher high',
    'lower_low': 'Price making lower low',
    'break_of_structure_bullish': 'Bullish break of structure',
    'break_of_structure_bearish': 'Bearish break of structure',
    'support_bounce': 'Price bouncing off support level',
    'resistance_rejection': 'Price rejected at resistance level',
    'support_break': 'Price breaks below support',
    'resistance_break': 'Price breaks above resistance',
  }

  return descriptions.get(signal_name, 'Technical analysis signal')


def categorize_signal(signal_name: str) -> str:
  """Categorize signal by type"""
  signal_lower = signal_name.lower()

  # Candlestick patterns
  if any(x in signal_lower for x in
         ['engulfing', 'hammer', 'star', 'doji', 'marubozu', 'soldiers', 'crows', 'piercing', 'tweezer']):
    return 'Candlestick Pattern'

  # Moving averages
  elif any(x in signal_lower for x in ['ma_', 'ema_', 'ribbon', 'moving_average']):
    return 'Moving Average'

  # Momentum indicators
  elif any(x in signal_lower for x in ['rsi', 'macd', 'stoch', 'cci', 'williams', 'momentum']):
    return 'Momentum'

  # Volume indicators
  elif any(x in signal_lower for x in ['volume', 'obv', 'vwap', 'accumulation']):
    return 'Volume'

  # Volatility indicators
  elif any(x in signal_lower for x in ['bollinger', 'atr', 'keltner', 'volatility']):
    return 'Volatility'

  # Trend indicators
  elif any(x in signal_lower for x in ['adx', 'supertrend', 'sar', 'ichimoku', 'parabolic']):
    return 'Trend'

  # Price action
  elif any(x in signal_lower for x in
           ['support', 'resistance', 'structure', 'pivot', 'fibonacci', 'break', 'bounce', 'higher', 'lower']):
    return 'Price Action'

  else:
    return 'Other'


@app.route('/api/live-analysis/analyze', methods=['POST'])
@login_required
def analyze_symbol_live():
  """
  Analyze a symbol across specified timeframes
  Request body: {
      "symbol": "BTC",
      "timeframes": ["1m", "5m", "15m", "1h"]
  }
  """
  try:
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    timeframes = data.get('timeframes', ['1m', '5m', '15m', '1h'])

    if not symbol:
      return jsonify({'error': 'Symbol required'}), 400

    # Validate timeframes
    valid_tfs = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
    timeframes = [tf for tf in timeframes if tf in valid_tfs]

    if not timeframes:
      return jsonify({'error': 'Valid timeframes required'}), 400

    logging.info(f"🔍 Analyzing {symbol} on {timeframes}")

    # Run analysis
    result = live_analyzer.analyze_symbol_all_timeframes(symbol, timeframes)

    # Save to database
    live_db.save_analysis_result(result)

    # Return formatted result
    return jsonify({
      'success': True,
      'symbol': symbol,
      'timestamp': result['timestamp'],
      'timeframes': result['timeframes']
    })

  except Exception as e:
    logging.error(f"Analysis error: {e}")
    return jsonify({'error': str(e)}), 500


@app.route('/api/live-analysis/signals/<symbol>')
@login_required
def get_live_signals(symbol):
  """Get latest signals for a symbol"""
  try:
    timeframe = request.args.get('timeframe')
    limit = int(request.args.get('limit', 50))

    signals = live_db.get_latest_signals(symbol.upper(), timeframe, limit)

    return jsonify({
      'success': True,
      'symbol': symbol,
      'signals': signals,
      'count': len(signals)
    })

  except Exception as e:
    logging.error(f"Failed to get signals: {e}")
    return jsonify({'error': str(e)}), 500


@app.route('/api/live-analysis/summary/<symbol>')
@login_required
def get_analysis_summary(symbol):
  """Get analysis summary for a symbol"""
  try:
    hours = int(request.args.get('hours', 24))
    summary = live_db.get_analysis_summary(symbol.upper(), hours)

    return jsonify({
      'success': True,
      'symbol': symbol,
      'summary': summary
    })

  except Exception as e:
    logging.error(f"Failed to get summary: {e}")
    return jsonify({'error': str(e)}), 500


@app.route('/api/live-analysis/signal-info/<signal_name>')
@login_required
def get_signal_info(signal_name):
  """Get confidence and timeframe info for a signal"""
  try:
    info = ScalpSignalAnalyzer.SIGNAL_CONFIDENCE.get(signal_name, {})

    if not info:
      return jsonify({'error': 'Signal not found'}), 404

    return jsonify({
      'success': True,
      'signal_name': signal_name,
      'confidence': info.get('confidence', 0),
      'suitable_timeframes': info.get('timeframes', []),
      'description': get_signal_description(signal_name)
    })

  except Exception as e:
    return jsonify({'error': str(e)}), 500


@app.route('/api/live-analysis/all-signals')
@login_required
def get_all_signal_definitions():
  """Get all signal definitions with confidence ratings"""
  try:
    signals = []

    for signal_name, info in ScalpSignalAnalyzer.SIGNAL_CONFIDENCE.items():
      signals.append({
        'name': signal_name,
        'confidence': info['confidence'],
        'timeframes': info['timeframes'],
        'category': categorize_signal(signal_name)
      })

    # Sort by confidence
    signals.sort(key=lambda x: x['confidence'], reverse=True)

    return jsonify({
      'success': True,
      'signals': signals,
      'total': len(signals)
    })

  except Exception as e:
    return jsonify({'error': str(e)}), 500


@app.route('/api/live-analysis/cleanup', methods=['POST'])
@login_required
def cleanup_old_signals():
  """Cleanup old signals from database"""
  try:
    days = int(request.get_json().get('days', 30))
    deleted = live_db.cleanup_old_signals(days)

    return jsonify({
      'success': True,
      'deleted': deleted,
      'message': f'Cleaned up signals older than {days} days'
    })

  except Exception as e:
    return jsonify({'error': str(e)}), 500


# ==================== TRADING POSITION ROUTES ====================

@app.route('/api/trading/positions', methods=['GET', 'POST'])
@login_required
def trading_positions():
  """Get all positions or create new position"""
  global trading_manager

  # Safety check
  if trading_manager is None:
    return jsonify({
      'success': False,
      'error': 'Trading manager not initialized. Please restart the application.'
    }), 500

  if request.method == 'GET':
    status = request.args.get('status')
    symbol = request.args.get('symbol')

    try:
      positions = trading_manager.get_all_positions(status, symbol)
      summary = trading_manager.get_position_summary()

      return jsonify({
        'success': True,
        'positions': positions,
        'summary': summary
      })
    except Exception as e:
      logging.error(f"Error getting positions: {e}")
      return jsonify({'success': False, 'error': str(e)}), 500

  elif request.method == 'POST':
    data = request.get_json()

  try:
    position_id = trading_manager.create_position(
      symbol=data['symbol'],
      entry_price=float(data['entry_price']),
      amount=float(data['amount']),
      status=data.get('status', 'waiting_for_right_time_to_enter'),
      leverage_multiplier=float(data['leverage_multiplier']) if data.get('leverage_multiplier') else None,
      liquidation_at=float(data['liquidation_at']) if data.get('liquidation_at') else None,
      break_even=float(data['break_even']) if data.get('break_even') else None,
      stop_loss=float(data['stop_loss']) if data.get('stop_loss') else None,
      stop_profit=float(data['stop_profit']) if data.get('stop_profit') else None,
      entry_reason=data.get('entry_reason'),
      notes=data.get('notes')
    )

    return jsonify({
      'success': True,
      'position_id': position_id,
      'message': 'Position created successfully'
    })

  except Exception as e:
    logging.error(f"Error creating position: {e}")
    return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trading/positions/<int:position_id>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def trading_position_detail(position_id):
  """Get, update, or delete specific position"""
  global trading_manager

  if trading_manager is None:
    return jsonify({'success': False, 'error': 'Trading manager not initialized'}), 500

  if request.method == 'GET':
    try:
      position = trading_manager.get_position(position_id)

      if not position:
        return jsonify({'success': False, 'error': 'Position not found'}), 404

      # Get signals for this position
      signals = trading_manager.get_position_signals(position_id)

      return jsonify({
        'success': True,
        'position': position,
        'signals': signals
      })
    except Exception as e:
      logging.error(f"Error getting position: {e}")
      return jsonify({'success': False, 'error': str(e)}), 500

  elif request.method == 'PUT':
    data = request.get_json()

    try:
      success = trading_manager.update_position(position_id, **data)

      if success:
        return jsonify({'success': True, 'message': 'Position updated'})
      else:
        return jsonify({'success': False, 'error': 'Update failed'}), 404
    except Exception as e:
      logging.error(f"Error updating position: {e}")
      return jsonify({'success': False, 'error': str(e)}), 500

  elif request.method == 'DELETE':
    try:
      success = trading_manager.delete_position(position_id)

      if success:
        return jsonify({'success': True, 'message': 'Position deleted'})
      else:
        return jsonify({'success': False, 'error': 'Delete failed'}), 404
    except Exception as e:
      logging.error(f"Error deleting position: {e}")
      return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trading/positions/<int:position_id>/close', methods=['POST'])
@login_required
def close_trading_position(position_id):
  """Close a position"""
  global trading_manager

  if trading_manager is None:
    return jsonify({'success': False, 'error': 'Trading manager not initialized'}), 500

  data = request.get_json()

  try:
    success = trading_manager.close_position(
      position_id,
      float(data['sold_price']),
      data.get('exit_reason')
    )

    if success:
      return jsonify({'success': True, 'message': 'Position closed'})
    else:
      return jsonify({'success': False, 'error': 'Failed to close position'}), 404
  except Exception as e:
    logging.error(f"Error closing position: {e}")
    return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trading/calculate-stops', methods=['POST'])
@login_required
def calculate_stops():
  """Calculate suggested stop loss and stop profit"""
  global trading_manager

  if trading_manager is None:
    return jsonify({'success': False, 'error': 'Trading manager not initialized'}), 500

  data = request.get_json()

  try:
    result = trading_manager.calculate_stop_loss_profit(
      signal_type=data['signal_type'],
      entry_price=float(data['entry_price']),
      confidence=int(data['confidence']),
      leverage=float(data.get('leverage', 1.0))
    )

    return jsonify({
      'success': True,
      'suggestions': result
    })

  except Exception as e:
    logging.error(f"Error calculating stops: {e}")
    return jsonify({'success': False, 'error': str(e)}), 400


# ==================== WEBSOCKET FOR LIVE MONITORING ====================

@sock.route('/ws/position/<int:position_id>')
def position_websocket(ws, position_id):
  """WebSocket for real-time position monitoring"""
  global trading_manager, fact_checker

  if trading_manager is None:
    ws.send(json.dumps({'error': 'Trading manager not initialized'}))
    return

  # Get position details
  position = trading_manager.get_position(position_id)

  if not position:
    ws.send(json.dumps({'error': 'Position not found'}))
    return

  symbol = position['symbol']
  analyzer = ScalpSignalAnalyzer()

  # Timeframes to monitor (can be customized)
  timeframes = ['1m', '5m', '15m', '1h']

  try:
    while True:
      # Fetch and analyze data
      result = analyzer.analyze_symbol_all_timeframes(symbol, timeframes)

      # Get adjusted confidences
      for tf, data in result['timeframes'].items():
        if 'error' in data:
          continue

        for signal_name, signal_data in data['signals'].items():
          adjusted_conf = fact_checker.get_adjusted_confidence(signal_name, tf)
          signal_data['adjusted_confidence'] = adjusted_conf

          # Save signal to position
          trading_manager.add_signal_to_position(
            position_id,
            signal_name,
            signal_data.get('signal', 'UNKNOWN'),
            tf,
            adjusted_conf,
            data['price']
          )

      # Send update to client
      ws.send(json.dumps({
        'timestamp': datetime.now().isoformat(),
        'position_id': position_id,
        'symbol': symbol,
        'analysis': result
      }))

      # Wait 60 seconds before next update
      # time.sleep(10)

  except Exception as e:
    logging.error(f"WebSocket error: {e}")
    ws.send(json.dumps({'error': str(e)}))


# ==================== FACT-CHECKING ROUTES ====================

@app.route('/api/fact-check/position/<int:position_id>', methods=['POST'])
@login_required
def fact_check_position(position_id):
  """Fact-check all signals for a position"""
  global trading_manager, fact_checker

  if trading_manager is None or fact_checker is None:
    return jsonify({'success': False, 'error': 'Modules not initialized'}), 500

  position = trading_manager.get_position(position_id)

  if not position:
    return jsonify({'success': False, 'error': 'Position not found'}), 404

  try:
    candles_ahead = int(request.args.get('candles_ahead', 5))

    results = fact_checker.fact_check_position_signals(
      position_id,
      position['symbol'],
      candles_ahead
    )

    return jsonify({
      'success': True,
      'results': results
    })

  except Exception as e:
    logging.error(f"Error fact-checking: {e}")
  return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/fact-check/signal-accuracy/<signal_name>')
@login_required
def get_signal_accuracy(signal_name):
  """Get accuracy stats for a specific signal"""
  global fact_checker

  if fact_checker is None:
    return jsonify({'success': False, 'error': 'Fact checker not initialized'}), 500

  timeframe = request.args.get('timeframe')

  try:
    accuracy = fact_checker.calculate_signal_accuracy(signal_name, timeframe)

    if accuracy:
      return jsonify({'success': True, 'accuracy': accuracy})
    else:
      return jsonify({
        'success': False,
        'error': 'Insufficient data',
        'message': 'Need at least 10 samples'
      }), 404
  except Exception as e:
    logging.error(f"Error getting accuracy: {e}")
    return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/fact-check/adjust-confidence', methods=['POST'])
@login_required
def adjust_signal_confidence():
  """Adjust confidence for a specific signal"""
  global fact_checker

  if fact_checker is None:
    return jsonify({'success': False, 'error': 'Fact checker not initialized'}), 500

  data = request.get_json()

  try:
    result = fact_checker.adjust_signal_confidence(
      data['signal_name'],
      data['timeframe'],
      data.get('min_samples', 10)
    )

    if result:
      return jsonify({'success': True, 'adjustment': result})
    else:
      return jsonify({
        'success': False,
        'error': 'Insufficient samples'
      }), 400
  except Exception as e:
    logging.error(f"Error adjusting confidence: {e}")
    return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/fact-check/bulk-adjust', methods=['POST'])
@login_required
def bulk_adjust_signals():
  """Adjust all signals with sufficient data"""
  global fact_checker

  if fact_checker is None:
    return jsonify({'success': False, 'error': 'Fact checker not initialized'}), 500

  data = request.get_json()
  min_samples = data.get('min_samples', 10)

  try:
    results = fact_checker.bulk_adjust_all_signals(min_samples)

    return jsonify({
      'success': True,
      'results': results
    })

  except Exception as e:
    logging.error(f"Error bulk adjusting: {e}")
  return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/fact-check/adjustments')
@login_required
def get_all_adjustments():
  """Get all signal confidence adjustments"""
  global fact_checker

  if fact_checker is None:
    return jsonify({'success': False, 'error': 'Fact checker not initialized'}), 500

  try:

    adjustments = fact_checker.get_all_adjustments()

    return jsonify({
      'success': True,
      'adjustments': adjustments,
      'count': len(adjustments)
    })
  except Exception as e:
    logging.error(f"Error getting adjustments: {e}")
    return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/fact-check/cleanup', methods=['POST'])
@login_required
def cleanup_fact_checks():
  """Clean up old fact-check records"""
  global fact_checker

  if fact_checker is None:
    return jsonify({'success': False, 'error': 'Fact checker not initialized'}), 500

  data = request.get_json()
  days = data.get('days', 90)

  try:
    deleted = fact_checker.cleanup_old_fact_checks(days)

    return jsonify({
      'success': True,
      'deleted': deleted,
      'message': f'Cleaned up records older than {days} days'
    })
  except Exception as e:
    logging.error(f"Error cleaning up: {e}")
    return jsonify({'success': False, 'error': str(e)}), 500


# ==================== HELPER FUNCTIONS ====================

def calculate_validity_hours(confidence, pattern_count):
  """Calculate signal validity based on confidence and pattern count"""
  base = 12
  confidence_factor = (confidence - 0.7) / 0.3
  pattern_factor = min(pattern_count / 200, 1.0)
  total_factor = (confidence_factor * 0.6 + pattern_factor * 0.4)
  validity = base + (48 * total_factor)
  return int(validity)


# ==================== STARTUP ====================

if __name__ == '__main__':
  # This only runs when using "python app.py" directly
  # Gunicorn will skip this and use the initialization above
  app.run(host='0.0.0.0', port=5001, debug=True)
