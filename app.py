"""
Complete Flask Application with Integrated Crypto Pattern Monitoring
"""

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
    # CHANGE THESE CREDENTIALS!
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

@app.route('/api/monitor/start')
@login_required
def start_monitor():
  global monitor, monitor_thread

  if monitor is not None and monitor.running:
    return jsonify({'success': False, 'message': 'Already running'})

  try:
    # Create monitor instance
    monitor = CryptoPatternMonitor(
      db_path=app.config['DB_PATH'],
      pattern_file=app.config['PATTERNS_FILE'],
      priority_coins_file=app.config['PRIORITY_COINS_FILE']
    )

    # Start monitoring in background thread
    monitor_thread = threading.Thread(
      target=monitor.run,
      args=(100,),  # Top 100 coins
      daemon=True
    )
    monitor_thread.start()

    logging.info("✅ Monitoring started")
    return jsonify({'success': True, 'message': 'Monitoring started'})

  except Exception as e:
    logging.error(f"Failed to start monitoring: {e}")
    return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/monitor/stop')
@login_required
def stop_monitor():
  global monitor

  if monitor is None or not monitor.running:
    return jsonify({'success': False, 'message': 'Not running'})

  try:
    monitor.stop()
    logging.info("⏸️ Monitoring stopped")
    return jsonify({'success': True, 'message': 'Monitoring stopped'})

  except Exception as e:
    logging.error(f"Failed to stop monitoring: {e}")
    return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/monitor/status')
@login_required
def monitor_status():
  global monitor

  if monitor is None:
    return jsonify({
      'running': False,
      'message': 'Monitor not initialized'
    })

  stats = monitor.get_stats()
  return jsonify(stats)




# ==================== LIVE ANALYSIS ROUTES ====================

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


# Helper functions

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
  if 'engulfing' in signal_name or 'hammer' in signal_name or 'star' in signal_name or 'doji' in signal_name or 'marubozu' in signal_name or 'soldiers' in signal_name or 'crows' in signal_name:
    return 'Candlestick Pattern'
  elif 'ma_' in signal_name or 'ema_' in signal_name or 'ribbon' in signal_name:
    return 'Moving Average'
  elif 'rsi' in signal_name or 'macd' in signal_name or 'stoch' in signal_name or 'cci' in signal_name or 'williams' in signal_name or 'momentum' in signal_name:
    return 'Momentum'
  elif 'volume' in signal_name or 'obv' in signal_name or 'vwap' in signal_name or 'accumulation' in signal_name:
    return 'Volume'
  elif 'bollinger' in signal_name or 'atr' in signal_name or 'keltner' in signal_name:
    return 'Volatility'
  elif 'adx' in signal_name or 'supertrend' in signal_name or 'sar' in signal_name or 'ichimoku' in signal_name:
    return 'Trend'
  elif 'support' in signal_name or 'resistance' in signal_name or 'structure' in signal_name or 'pivot' in signal_name or 'fibonacci' in signal_name:
    return 'Price Action'
  else:
    return 'Other'


# Add this to your app initialization section:
"""
# Import the new modules at the top of app.py:
from scalp_signal_analyzer import ScalpSignalAnalyzer
from live_analysis_handler import LiveAnalysisDB

# Initialize after other globals:
live_analyzer = ScalpSignalAnalyzer()
live_db = LiveAnalysisDB()

# Then add all the routes above before if __name__ == '__main__':
"""


# ==================== HELPER FUNCTIONS ====================

def calculate_validity_hours(confidence, pattern_count):
  """Calculate signal validity based on confidence and pattern count"""
  base = 12
  confidence_factor = (confidence - 0.7) / 0.3
  pattern_factor = min(pattern_count / 200, 1.0)
  total_factor = (confidence_factor * 0.6 + pattern_factor * 0.4)
  validity = base + (48 * total_factor)
  return int(validity)


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


# ==================== STARTUP ====================

if __name__ == '__main__':
  # Initialize database
  init_database()

  # NEW: Initialize live analysis
  # global live_analyzer, live_db
  live_analyzer = ScalpSignalAnalyzer()
  live_db = LiveAnalysisDB()
  logging.info("✅ Live analysis system initialized")

  # Run Flask app
  app.run(host='0.0.0.0', port=5001, debug=True)  # Set debug=False for production
