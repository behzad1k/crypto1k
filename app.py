"""
Complete Flask Application with Integrated Crypto Pattern Monitoring
"""
import time
from collections import defaultdict

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
from monitor import CryptoPatternMonitor
from paper_trading_engine import PaperTradingEngine
from paper_trading_manager import PaperTradingManager
from scalp_signal_analyzer import ScalpSignalAnalyzer
from live_analysis_handler import LiveAnalysisDB
from signal_combination_analyzer import SignalCombinationAnalyzer
from signal_validation_optimizer import SignalValidationOptimizer
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
fact_checker = None
signal_validation= None
combo_analyzer = None
paper_trading_engine = None
pt_manager = None
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
  global live_analyzer, live_db, fact_checker, sock, combo_analyzer, signal_validation, paper_trading_engine, pt_manager

  # Only initialize once
  if live_analyzer is not None:
    return

  logging.info("🚀 Initializing application...")

  # Initialize database
  init_database()
  # Initialize live analysis
  try:
    paper_trading_engine = PaperTradingEngine(
      db_path=app.config['DB_PATH'],
      initial_bankroll=10000.0  # Starting with $10,000
    )
    pt_manager = PaperTradingManager();
    logging.info("✅ Paper trading engine initialized")
  except Exception as e:
    logging.error(f"❌ Failed to initialize paper trading: {e}")

  try:
    live_analyzer = ScalpSignalAnalyzer()
    live_db = LiveAnalysisDB()
    fact_checker = SignalFactChecker()
    sock = Sock(app)

    logging.info("✅ Live analysis system initialized")
  except Exception as e:
    logging.error(f"❌ Failed to initialize live analysis: {e}")


  # Initialize signal validation modules
  try:
      signal_validation = SignalValidationOptimizer(db_path=app.config['DB_PATH'])
      logging.info("✅ Signal validation analyzer initialized")
  except Exception as e:
      logging.error(f"❌ Failed to initialize validation analyzer: {e}")

# Initialize signal combo modules
  try:
      combo_analyzer = SignalCombinationAnalyzer(db_path=app.config['DB_PATH'])
      logging.info("✅ Signal combination analyzer initialized")
  except Exception as e:
      logging.error(f"❌ Failed to initialize combo analyzer: {e}")
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

@app.route('/symbol/<symbol>')
@login_required
def symbol_info(symbol):
  return render_template('symbol.html', symbol=symbol.upper())


# ==================== PATTERN SIGNALS API ====================

@app.route('/api/signals')
@login_required
def get_signals():
  """Get pattern signals with pagination and filters"""
  page = int(request.args.get('page', 1))
  per_page = int(request.args.get('per_page', 10))
  min_accuracy = request.args.get('minAccuracy', type=float)
  min_patterns = request.args.get('minPatterns', type=int)
  signal_type = request.args.get('signalType')
  symbol = request.args.get('symbol', '').upper()

  conn = sqlite3.connect(app.config['DB_PATH'])
  conn.row_factory = sqlite3.Row
  cursor = conn.cursor()

  query = "SELECT * FROM pattern_signals WHERE 1=1"
  params = []

  if signal_type:
    query += " AND signal = ?"
    params.append(signal_type.upper())

  if min_accuracy:
    query += " AND pattern_confidence >= ?"
    params.append(min_accuracy)
  if min_patterns:
    query += " AND pattern_count >= ?"
    params.append(min_patterns)
  if symbol:
    query += " AND symbol LIKE ?"
    params.append(f'%{symbol}%')

    # Count total
  count_query = query.replace('SELECT *', 'SELECT COUNT(*)')
  cursor.execute(count_query, params)
  total = cursor.fetchone()[0]

  # Get paginated results
  query += " ORDER BY datetime_created DESC LIMIT ? OFFSET ?"
  params.extend([per_page, (page - 1) * per_page])

  cursor.execute(query, params)
  signals = [dict(row) for row in cursor.fetchall()]
  conn.close()

  return jsonify({
    'signals': signals,
    'total': total,
    'pages': (total + per_page - 1) // per_page
  })


@app.route('/api/symbols/<symbol>')
@login_required
def get_symbol_history(symbol):
  """Get signal history for a specific symbol"""
  conn = sqlite3.connect(app.config['DB_PATH'])
  conn.row_factory = sqlite3.Row
  cursor = conn.cursor()

  cursor.execute('''
        SELECT * FROM pattern_signals 
        WHERE symbol = ? 
        ORDER BY datetime_created DESC
    ''', (symbol,))

  history = [dict(row) for row in cursor.fetchall()]
  conn.close()
  return jsonify(history)


@app.route('/api/export')
@login_required
def export_signals():
  """Export signals to Excel"""
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


# ==================== PRIORITY COINS ====================

@app.route('/api/priority-coins', methods=['GET', 'POST'])
@login_required
def priority_coins():
  """Manage priority coins list"""
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


# ==================== MONITORING CONTROL ====================

@app.route('/api/monitor/status')
@login_required
def monitor_status():
  """Get monitoring status"""
  global monitor

  if monitor is None:
    return jsonify({
      'running': False,
            'message': 'Monitor not initialized'
    })

  stats = monitor.get_stats()
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


@app.route('/api/monitor/start')
@login_required
def start_monitor():
  """Start pattern monitoring with paper trading integration"""
  global monitor, monitor_thread, paper_trading_engine

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
      priority_coins_file=app.config['PRIORITY_COINS_FILE'],
      paper_trading_engine=paper_trading_engine
    )

    # Start monitoring in background thread WITH paper trading engine
    monitor_thread = threading.Thread(
      target=monitor.run,
      args=(100,),
      kwargs={'paper_trading_engine': paper_trading_engine},  # NEW
      daemon=True
    )
    monitor_thread.start()

    # Give it a moment to start
    time.sleep(0.5)

    logging.info("✅ Monitoring started successfully with paper trading integration")
    return jsonify({
      'success': True,
      'message': 'Monitoring started with paper trading'
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
  """Stop pattern monitoring"""
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

@app.route('/api/live-analysis/analyze', methods=['POST'])
@login_required
def analyze_symbol_live():
  """Analyze symbol across multiple timeframes"""
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
      'timeframes': result['timeframes'],
            'combinations': result['combinations']
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

@app.route('/api/live-analysis/full-signals')
@login_required
def get_full_signals():
  """Get all adjusted confidence signals"""
  global fact_checker
  try:
    signals = fact_checker.get_all_adjusted_confidence()
    return jsonify({'success': True, 'signals': signals})

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
        # 'category': categorize_signal(signal_name)
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


@app.route('/api/live-analysis/combos/<symbol>')
@login_required
def get_live_signal_combos(symbol):
  """Get latest signal combinations for a symbol"""
  try:
    timeframe = request.args.get('timeframe')
    limit = int(request.args.get('limit', 1000))

    conn = sqlite3.connect(app.config['DB_PATH'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = '''
            SELECT live.*, combos.signals_count
            FROM live_tf_combos live
            INNER JOIN tf_combos combos 
            ON live.combo_signal_name = combos.signal_name
            WHERE live.symbol = ?
        '''
    params = [symbol.upper()]

    if timeframe:
      query += ' AND timeframe = ?'
      params.append(timeframe)

    query += ' ORDER BY accuracy DESC, timestamp DESC LIMIT ?'
    params.append(limit)

    cursor.execute(query, params)
    combos = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify({
      'success': True,
      'symbol': symbol,
      'combinations': combos,
      'count': len(combos)
    })

  except Exception as e:
    logging.error(f"Failed to get combos: {e}")
    return jsonify({'error': str(e)}), 500

# ==================== WEBSOCKET FOR LIVE MONITORING ====================

@sock.route('/ws/live-analysis/<symbol>')
def symbol_websocket(ws, symbol):
  """WebSocket for real-time symbol monitoring"""
  analyzer = ScalpSignalAnalyzer()
  timeframes = ['1m', '5m', '15m', '30m', '1h', '2h']

  try:
    while True:
      # Fetch and analyze data
      result = analyzer.analyze_symbol_all_timeframes(symbol, timeframes)

      # Get adjusted confidences
      for tf, data in result['timeframes'].items():
        if 'error' in data:
          continue

      live_db.save_analysis_result(result)

      # Send update to client with combos
      ws.send(json.dumps({
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'analysis': result,
      }))

      time.sleep(50)

  except Exception as e:
    logging.error(f"WebSocket error: {e}")
    ws.send(json.dumps({'error': str(e)}))

# ==================== FACT-CHECKING ROUTES ====================

@app.route('/api/fact-check/bulk-signals', methods=['POST'])
@login_required
def bulk_fact_check_live_signals():
  """Fact-check all signals"""
  global fact_checker

  if fact_checker is None:
    return jsonify({'success': False, 'error': 'Fact checker not initialized'}), 500

  data = request.get_json()
  limit = data.get('limit', None)
  symbol = data.get('symbol', None)

  try:
    start_time = time.time()
    results = fact_checker.bulk_fact_check_live_signals(symbol=symbol, limit=limit)
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("VALIDATION WINDOW OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Time taken: {elapsed / 60:.1f} minutes")
    print(f"all: {results}")
    print(f"Total combinations: {results['total_checked']}")
    print(f"Successfully optimized: {results['correct_predictions']}")
    print(f"No data available: {results['stopped_out']}")
    print(f"\nValidation windows have been saved to the signals table")
    print("=" * 80 + "\n")

    return jsonify({'success': True, 'results': results})

  except Exception as e:
    logging.error(f"Error signal-validation: {e}")
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
      return jsonify({'success': False, 'error': 'Insufficient data'}), 404

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


# ==================== SIGNAL VALIDATION ROUTES ====================

@app.route('/api/signal-validation/bulk-validate', methods=['POST'])
@login_required
def bulk_validate_all_signals():
  """Optimize validation windows for all signals"""
  global signal_validation

  if signal_validation is None:
    return jsonify({'success': False, 'error': 'Validator not initialized'}), 500

  data = request.get_json()
  max_workers = data.get('max_workers', 10)
  limit_per_signal = data.get('limit_per_signal')

  try:
    start_time = time.time()
    results = signal_validation.optimize_all_signals(
      limit_per_signal=limit_per_signal,
      max_workers=max_workers
    )
    elapsed = time.time() - start_time

    return jsonify({
      'success': True,
      'results': results,
      'time_elapsed': elapsed
    })
  except Exception as e:
    logging.error(f"Error validating: {e}")
    return jsonify({'success': False, 'error': str(e)}), 500


# ==================== SIGNAL COMBINATION ROUTES ====================

@app.route('/api/combo-analysis/analyze', methods=['POST'])
@login_required
def analyze_signal_combinations():
  """
  Trigger bulk combination analysis
  POST body: {
      "timeframe": "1h" (optional),
      "min_samples": 20,
      "min_combo_size": 2,
      "max_combo_size": 4
  }
  """
  global combo_analyzer

  if combo_analyzer is None:
    return jsonify({
      'success': False,
      'error': 'Combination analyzer not initialized'
    }), 500

  try:
    data = request.get_json() or {}

    timeframe = data.get('timeframe')
    min_samples = int(data.get('min_samples', 20))
    min_combo_size = int(data.get('min_combo_size', 2))
    max_combo_size = int(data.get('max_combo_size', 4))

    # Validate combo sizes
    if min_combo_size < 2 or max_combo_size > 10:
      return jsonify({
        'success': False,
        'error': 'Combo size must be between 2 and 10'
      }), 400

    logging.info(f"🚀 Starting combination analysis...")
    logging.info(f"   Min samples: {min_samples}")
    logging.info(f"   Combo size: {min_combo_size}-{max_combo_size}")

    if timeframe:
      # Analyze single timeframe
      result = combo_analyzer.analyze_timeframe_combinations(
        timeframe, min_samples, min_combo_size, max_combo_size
      )
    else:
      # Analyze all timeframes
      result = combo_analyzer.analyze_all_timeframes(
        min_samples, min_combo_size, max_combo_size
      )

    return jsonify({
      'success': True,
      'result': result
    })

  except Exception as e:
    logging.error(f"❌ Combination analysis failed: {e}")
    return jsonify({
      'success': False,
      'error': str(e)
    }), 500


@app.route('/api/combo-analysis/analyze-cross-tf', methods=['POST'])
@login_required
def analyze_cross_tf_signal_combinations():
  """
  Trigger bulk combination analysis
  POST body: {
      "timeframes": ["1h"] (optional),
      "min_samples": 20,
      "min_combo_size": 2,
      "max_combo_size": 4
  }
  """
  global combo_analyzer

  if combo_analyzer is None:
    return jsonify({
      'success': False,
      'error': 'Combination analyzer not initialized'
    }), 500

  try:
    data = request.get_json() or {}

    timeframes = data.get('timeframes', ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w'])
    min_samples = int(data.get('min_samples', 20))
    min_combo_size = int(data.get('min_combo_size', 2))
    max_combo_size = int(data.get('max_combo_size', 4))

    # Validate combo sizes
    if min_combo_size < 2 or max_combo_size > 10:
      return jsonify({
        'success': False,
        'error': 'Combo size must be between 2 and 10'
      }), 400

    logging.info(f"🚀 Starting combination analysis...")
    logging.info(f"   tfs: {timeframes}")
    logging.info(f"   Min samples: {min_samples}")
    logging.info(f"   Combo size: {min_combo_size}-{max_combo_size}")

    results = combo_analyzer.analyze_cross_timeframe_combinations(
      timeframes,
      min_samples,
      min_combo_size,
      max_combo_size,
    )

    return jsonify({
      'success': True,
      'result': results
    })

  except Exception as e:
    logging.error(f"❌ Combination analysis failed: {e}")
    return jsonify({
      'success': False,
      'error': str(e)
    }), 500


# ADD THIS NEW ROUTE TO app.py AFTER THE get_active_combinations_for_symbol() function
# Around line 880 (after the existing active-combos endpoint)

@app.route('/api/combo-analysis/active-cross-tf/<symbol>')
@login_required
def get_active_cross_tf_combinations_for_symbol(symbol):
  """
  Get active cross-timeframe signal combinations for a specific symbol
  This checks which cross-TF combinations are currently active based on live_signals
  Returns top 50 combinations sorted by accuracy
  """
  global combo_analyzer

  if combo_analyzer is None:
    return jsonify({'success': False, 'error': 'Analyzer not initialized'}), 500

  try:
    conn = sqlite3.connect(app.config['DB_PATH'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get recent signals across all timeframes (last 2 hours)
    cursor.execute('''
      SELECT 
        timeframe,
        signal_name,
        timestamp,
        signal_type,
        price
      FROM live_signals
      WHERE symbol = ?
        AND timestamp >= datetime('now', '-2 hours')
      ORDER BY timestamp DESC
    ''', (symbol.upper(),))

    signals = [dict(row) for row in cursor.fetchall()]

    if not signals:
      conn.close()
      return jsonify({
        'success': True,
        'symbol': symbol,
        'combinations': [],
        'total_combos': 0,
        'debug_info': {
          'total_signals': 0,
          'message': 'No recent signals found'
        }
      })

    logging.info(f"🔍 Found {len(signals)} recent signals for {symbol}")

    # Group signals by timeframe and time window
    timeframe_signals = defaultdict(lambda: defaultdict(list))

    for signal in signals:
      timestamp = datetime.fromisoformat(signal['timestamp'])
      # Group by hour for cross-TF matching
      window = timestamp.replace(minute=0, second=0, microsecond=0)
      timeframe_signals[signal['timeframe']][window].append(signal)

    # Find matching cross-TF combinations
    matched_combos = []

    # Get all cross-TF combinations from database
    cursor.execute('''
      SELECT * FROM cross_tf_combos
      WHERE accuracy >= 60
      ORDER BY accuracy DESC, profit_factor DESC
      LIMIT 100
    ''')

    all_cross_combos = [dict(row) for row in cursor.fetchall()]

    logging.info(f"📊 Checking against {len(all_cross_combos)} cross-TF combinations")

    for combo in all_cross_combos:
      combo_signature = combo['combo_signature']
      required_timeframes = combo['timeframes'].split(',')

      # Parse signal@timeframe pairs
      signal_tf_pairs = []
      for part in combo_signature.split('+'):
        sig_name, tf = part.rsplit('@', 1)
        signal_tf_pairs.append((sig_name, tf))

      # Check if all required signals are present in their respective timeframes
      # within the same time window
      for window in set(w for tf_dict in timeframe_signals.values() for w in tf_dict.keys()):
        match_count = 0
        matched_signals = []

        for sig_name, tf in signal_tf_pairs:
          # Check if this signal exists in this timeframe at this window
          if tf in timeframe_signals and window in timeframe_signals[tf]:
            window_signals = timeframe_signals[tf][window]
            if any(s['signal_name'] == sig_name for s in window_signals):
              match_count += 1
              matched_signal = next(s for s in window_signals if s['signal_name'] == sig_name)
              matched_signals.append(matched_signal)

        # If all signals matched, add this combo
        if match_count == len(signal_tf_pairs):
          matched_combos.append({
            **combo,
            'matched_at': window.isoformat(),
            'matched_signals': matched_signals,
            'current_price': matched_signals[0]['price'] if matched_signals else None
          })
          logging.info(f"✅ MATCH: {combo_signature} at {window}")
          break  # Only count each combo once

    conn.close()

    # Sort by accuracy
    matched_combos.sort(key=lambda x: x['accuracy'], reverse=True)

    # Limit to top 50
    matched_combos = matched_combos[:50]

    logging.info(f"📊 SUMMARY:")
    logging.info(f"   Total signals: {len(signals)}")
    logging.info(f"   Matched cross-TF combos: {len(matched_combos)}")

    return jsonify({
      'success': True,
      'symbol': symbol,
      'combinations': matched_combos,
      'total_combos': len(matched_combos),
      'debug_info': {
        'total_signals': len(signals),
        'timeframes_active': list(timeframe_signals.keys()),
        'checked_combinations': len(all_cross_combos)
      }
    })

  except Exception as e:
    logging.error(f"❌ Failed to get cross-TF combinations: {e}")
    import traceback
    logging.error(traceback.format_exc())
    return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/combo-analysis/top')
@login_required
def get_top_combinations():
  """
  Get top performing signal combinations
  Query params:
      - timeframe: filter by timeframe (optional)
      - min_accuracy: minimum accuracy threshold (default 60)
      - limit: max results (default 20)
  """
  global combo_analyzer

  if combo_analyzer is None:
    return jsonify({'success': False, 'error': 'Analyzer not initialized'}), 500

  try:
    timeframe = request.args.get('timeframe')
    min_accuracy = float(request.args.get('min_accuracy', 60.0))
    limit = int(request.args.get('limit', 20))

    combinations = combo_analyzer.get_top_combinations(
      timeframe, min_accuracy, limit
    )

    return jsonify({
      'success': True,
      'combinations': combinations,
      'count': len(combinations)
    })

  except Exception as e:
    logging.error(f"Failed to get top combinations: {e}")
    return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/combo-analysis/report')
@login_required
def get_combination_report():
  """
  Get comprehensive combination analysis report
  Query params:
      - min_accuracy: minimum accuracy for top performers (default 55)
  """
  global combo_analyzer

  if combo_analyzer is None:
    return jsonify({'success': False, 'error': 'Analyzer not initialized'}), 500

  try:
    min_accuracy = float(request.args.get('min_accuracy', 55.0))

    report = combo_analyzer.generate_report(min_accuracy)

    return jsonify({
      'success': True,
      'report': report
    })

  except Exception as e:
    logging.error(f"Failed to generate report: {e}")
    return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/combo-analysis/compare/<combo_name>')
@login_required
def compare_combo_to_individual_signals(combo_name):
  """
  Compare combination accuracy to individual signal accuracies
  Query params:
      - timeframe: required
  """
  global combo_analyzer

  if combo_analyzer is None:
    return jsonify({'success': False, 'error': 'Analyzer not initialized'}), 500

  try:
    timeframe = request.args.get('timeframe')

    if not timeframe:
      return jsonify({
        'success': False,
        'error': 'Timeframe parameter required'
      }), 400

    comparison = combo_analyzer.compare_combo_to_individual(
      combo_name, timeframe
    )

    return jsonify({
      'success': True,
      'comparison': comparison
    })

  except Exception as e:
    logging.error(f"Failed to compare combination: {e}")
    return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/combo-analysis/timeframe/<timeframe>')
@login_required
def get_timeframe_combinations(timeframe):
  """
  Get all combinations for a specific timeframe
  Query params:
      - min_accuracy: minimum accuracy threshold (default 50)
      - limit: max results (default 50)
  """
  global combo_analyzer

  if combo_analyzer is None:
    return jsonify({'success': False, 'error': 'Analyzer not initialized'}), 500

  try:
    min_accuracy = float(request.args.get('min_accuracy', 50.0))
    limit = int(request.args.get('limit', 50))

    combinations = combo_analyzer.get_top_combinations(
      timeframe, min_accuracy, limit
    )

    return jsonify({
      'success': True,
      'timeframe': timeframe,
      'combinations': combinations,
      'count': len(combinations)
    })

  except Exception as e:
    logging.error(f"Failed to get timeframe combinations: {e}")
    return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/combo-analysis/details')
@login_required
def get_combination_details():
  """
  Get details for a specific combination
  Query params:
      - combo_name: required (e.g., "macd_cross_bullish+rsi_oversold")
      - timeframe: required
  """
  global combo_analyzer

  if combo_analyzer is None:
    return jsonify({'success': False, 'error': 'Analyzer not initialized'}), 500

  try:
    combo_name = request.args.get('combo_name')
    timeframe = request.args.get('timeframe')

    if not combo_name or not timeframe:
      return jsonify({
        'success': False,
        'error': 'Both combo_name and timeframe required'
      }), 400

    details = combo_analyzer.get_combination_details(combo_name, timeframe)

    if not details:
      return jsonify({
        'success': False,
        'error': 'Combination not found'
      }), 404

    return jsonify({
      'success': True,
      'details': details
    })

  except Exception as e:
    logging.error(f"Failed to get combination details: {e}")
    return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/combo-analysis/active-combos/<symbol>')
@login_required
def get_active_combinations_for_symbol(symbol):
  """
  Get active signal combinations for a specific symbol across timeframes
  This checks which combinations are currently active based on live_signals
  """
  global combo_analyzer

  if combo_analyzer is None:
    return jsonify({'success': False, 'error': 'Analyzer not initialized'}), 500

  try:
    conn = sqlite3.connect(app.config['DB_PATH'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # DEBUG 1: Check if we have ANY signals for this symbol
    cursor.execute('''
      SELECT COUNT(*) as total, 
             MIN(timestamp) as oldest, 
             MAX(timestamp) as newest
      FROM live_signals
      WHERE symbol = ?
    ''', (symbol.upper(),))

    signal_stats = cursor.fetchone()
    logging.info(f"🔍 DEBUG - Symbol: {symbol}")
    logging.info(f"   Total signals in DB: {signal_stats['total']}")
    logging.info(f"   Oldest: {signal_stats['oldest']}")
    logging.info(f"   Newest: {signal_stats['newest']}")
    logging.info(f"   now: {datetime.now() - timedelta(hours=2)}")

    # DEBUG 2: Check signals in last 2 hours
    cursor.execute('''
      SELECT COUNT(*) as count_2h
      FROM live_signals
      WHERE symbol = ?
        AND timestamp >= datetime('now', '-2 hours')
    ''', (symbol.upper(),))

    count_2h = cursor.fetchone()['count_2h']
    logging.info(f"   Signals in last 2 hours: {count_2h}")

    # DEBUG 3: Get actual signals with details
    cursor.execute('''
      SELECT 
        timeframe,
        COUNT(*) as signal_count,
        GROUP_CONCAT(signal_name) as signals,
        MAX(timestamp) as latest_timestamp
      FROM live_signals
      WHERE symbol = ?
        AND timestamp >= datetime('now', '-2 hours')
      GROUP BY timeframe, 
        CAST(strftime('%s', timestamp) / 
          CASE timeframe
            WHEN '1m' THEN 60
            WHEN '3m' THEN 180
            WHEN '5m' THEN 300
            WHEN '15m' THEN 900
            WHEN '30m' THEN 1800
            WHEN '1h' THEN 3600
            WHEN '2h' THEN 7200
            WHEN '4h' THEN 14400
            WHEN '6h' THEN 21600
            WHEN '8h' THEN 28800
            WHEN '12h' THEN 43200
            WHEN '1d' THEN 86400
            ELSE 3600
          END AS INTEGER)
      ORDER BY timeframe, latest_timestamp DESC
    ''', (symbol.upper(),))

    active_windows = cursor.fetchall()

    logging.info(f"   Active windows found: {len(active_windows)}")

    for idx, window in enumerate(active_windows):
      signals_list = window['signals'].split(',') if window['signals'] else []
      logging.info(f"   Window {idx + 1}: {window['timeframe']} - {len(signals_list)} signals")
      logging.info(f"      Signals: {signals_list}")
      logging.info(f"      Timestamp: {window['latest_timestamp']}")

    # DEBUG 4: Check tf_combos table
    cursor.execute('SELECT COUNT(*) as total FROM tf_combos')
    combo_total = cursor.fetchone()['total']
    logging.info(f"   Total combinations in tf_combos: {combo_total}")

    if combo_total == 0:
      logging.warning("⚠️  tf_combos table is EMPTY - run combination analysis first!")

    # Process combinations
    results = {}
    checked_combos = 0
    matched_combos = 0

    for window in active_windows:
      timeframe = window['timeframe']
      signals = window['signals'].split(',') if window['signals'] else []

      if len(signals) < 2:
        logging.info(f"   ⏭️  Skipping {timeframe}: only {len(signals)} signal(s)")
        continue

      # Generate combo key
      combo_key = '+'.join(sorted(set(signals)))
      checked_combos += 1

      logging.info(f"   🔍 Checking combo: {combo_key} in {timeframe}")

      # Check if this combo exists in our database
      cursor.execute('''
        SELECT * FROM tf_combos
        WHERE signal_name = ? AND timeframe = ?
      ''', (combo_key, timeframe))

      combo_data = cursor.fetchone()

      if combo_data:
        matched_combos += 1
        logging.info(f"   ✅ MATCH FOUND! Accuracy: {combo_data['accuracy']:.2f}%")

        if timeframe not in results:
          results[timeframe] = []

        results[timeframe].append({
          'combo_name': combo_key,
          'signals': signals,
          'accuracy': combo_data['accuracy'],
          'sample_count': combo_data['signals_count'],
          'profit_factor': combo_data['profit_factor'],
          'avg_price_change': combo_data['avg_price_change'],
          'combo_size': combo_data['combo_size'],
          'correct_predictions': combo_data['correct_predictions'],
          'timestamp': window['latest_timestamp']
        })
      else:
        logging.info(f"   ❌ No match in tf_combos for: {combo_key}")

    conn.close()

    logging.info(f"\n📊 SUMMARY:")
    logging.info(f"   Checked combinations: {checked_combos}")
    logging.info(f"   Matched combinations: {matched_combos}")
    logging.info(f"   Timeframes with results: {len(results)}")

    return jsonify({
      'success': True,
      'symbol': symbol,
      'timeframes': results,
      'total_combos': sum(len(combos) for combos in results.values()),
      'debug_info': {
        'total_signals': signal_stats['total'],
        'signals_last_2h': count_2h,
        'windows_found': len(active_windows),
        'combos_checked': checked_combos,
        'combos_matched': matched_combos,
        'tf_combos_total': combo_total
      }
    })

  except Exception as e:
    logging.error(f"❌ Failed to get active combinations: {e}")
    import traceback
    logging.error(traceback.format_exc())
    return jsonify({'success': False, 'error': str(e)}), 500

# ==================== PAPER TRADING ROUTES ====================
# Add these routes to app.py after the combo analysis routes

@app.route('/paper-trading')
@login_required
def paper_trading_dashboard():
  """Paper trading dashboard page"""
  return render_template('paper_trading.html')

@app.route('/paper-trading/position/<symbol>')
@login_required
def paper_trading_position_details(symbol):
  """Position details page"""
  return render_template('position_details.html', symbol=symbol.upper())

@app.route('/api/paper-trading/start', methods=['POST'])
@login_required
def start_paper_trading():
  """Start paper trading engine"""
  global paper_trading_engine

  if paper_trading_engine is None:
    return jsonify({
      'success': False,
      'error': 'Paper trading engine not initialized'
    }), 500

  try:
    paper_trading_engine.start()
    return jsonify({
      'success': True,
      'message': 'Paper trading started'
    })
  except Exception as e:
    logging.error(f"Error starting paper trading: {e}")
    return jsonify({
      'success': False,
      'error': str(e)
    }), 500

@app.route('/api/paper-trading/stop', methods=['POST'])
@login_required
def stop_paper_trading():
  """Stop paper trading engine"""
  global paper_trading_engine

  if paper_trading_engine is None:
    return jsonify({
      'success': False,
      'error': 'Paper trading engine not initialized'
    }), 500

  try:
    paper_trading_engine.stop()
    return jsonify({
      'success': True,
      'message': 'Paper trading stopped'
    })
  except Exception as e:
    logging.error(f"Error stopping paper trading: {e}")
    return jsonify({
      'success': False,
      'error': str(e)
    }), 500

@app.route('/api/paper-trading/reset', methods=['POST'])
@login_required
def reset_paper_trading():
  """Reset paper trading engine"""
  global paper_trading_engine

  if paper_trading_engine is None:
    return jsonify({
      'success': False,
      'error': 'Paper trading engine not initialized'
    }), 500

  try:
    data = request.get_json() or {}
    new_bankroll = data.get('initial_bankroll')

    success = paper_trading_engine.reset(new_bankroll)

    if success:
      return jsonify({
        'success': True,
        'message': 'Paper trading reset successfully'
      })
    else:
      return jsonify({
        'success': False,
        'error': 'Failed to reset (make sure engine is stopped)'
      }), 400

  except Exception as e:
    logging.error(f"Error resetting paper trading: {e}")
    return jsonify({
      'success': False,
      'error': str(e)
    }), 500

@app.route('/api/paper-trading/positions')
@login_required
def get_paper_trading_positions():
  """Get all active positions with current prices"""
  global paper_trading_engine

  if paper_trading_engine is None:
    return jsonify({
      'success': False,
      'error': 'Paper trading engine not initialized'
    }), 500

  try:
    positions = []

    for symbol, pos_data in paper_trading_engine.active_positions.items():
      # Get current price
      current_price = paper_trading_engine.get_current_price(symbol)

      if current_price:
        entry_price = pos_data['entry_price']
        quantity = pos_data['quantity']
        entry_fee = pos_data['entry_fee']

        # Calculate current P/L
        gross_value = quantity * current_price
        exit_fee = gross_value * (paper_trading_engine.EXCHANGE_FEE / 100)
        net_value = gross_value - exit_fee

        position_size = pos_data['position_size']
        profit_loss = net_value - (position_size - entry_fee)
        profit_loss_pct = ((current_price - entry_price) / entry_price) * 100

        positions.append({
          **pos_data,
          'current_price': current_price,
          'current_profit_loss': profit_loss,
          'current_profit_loss_pct': profit_loss_pct
        })

    return jsonify({
      'success': True,
      'positions': positions
    })

  except Exception as e:
    logging.error(f"Error getting positions: {e}")
    return jsonify({
      'success': False,
      'error': str(e)
    }), 500

@app.route('/api/paper-trading/buying-queue')
@login_required
def get_buying_queue():
  """Get buying queue with current prices"""
  global paper_trading_engine

  if paper_trading_engine is None:
    return jsonify({
      'success': False,
      'error': 'Paper trading engine not initialized'
    }), 500

  try:
    queue_items = []

    for symbol, queue_data in paper_trading_engine.buying_queue.items():
      # Get current price
      current_price = paper_trading_engine.get_current_price(symbol)

      item = {**queue_data}

      if current_price:
        item['current_price'] = current_price
        item['distance_to_target_pct'] = (
          ((current_price - queue_data['target_price']) / queue_data['target_price']) * 100
        )

      queue_items.append(item)

    return jsonify({
      'success': True,
      'queue': queue_items
    })

  except Exception as e:
    logging.error(f"Error getting buying queue: {e}")
    return jsonify({
      'success': False,
      'error': str(e)
    }), 500

@app.route('/api/paper-trading/history')
@login_required
def get_position_history():
  """Get closed position history"""
  try:
    limit = int(request.args.get('limit', 50))

    conn = sqlite3.connect(app.config['DB_PATH'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
            SELECT * FROM position_history 
            ORDER BY closed_at DESC 
            LIMIT ?
        ''', (limit,))

    history = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify({
      'success': True,
      'history': history
    })

  except Exception as e:
    logging.error(f"Error getting history: {e}")
    return jsonify({
      'success': False,
      'error': str(e)
    }), 500

@app.route('/api/paper-trading/position/<symbol>')
@login_required
def get_position_details(symbol):
  """Get detailed position information"""
  global paper_trading_engine

  if paper_trading_engine is None:
    return jsonify({
      'success': False,
      'error': 'Paper trading engine not initialized'
    }), 500

  try:
    symbol = symbol.upper()

    # Check active positions
    if symbol in paper_trading_engine.active_positions:
      pos_data = paper_trading_engine.active_positions[symbol]

      # Get current price and P/L
      current_price = paper_trading_engine.get_current_price(symbol)

      if current_price:
        entry_price = pos_data['entry_price']
        quantity = pos_data['quantity']
        entry_fee = pos_data['entry_fee']

        gross_value = quantity * current_price
        exit_fee = gross_value * (paper_trading_engine.EXCHANGE_FEE / 100)
        net_value = gross_value - exit_fee

        position_size = pos_data['position_size']
        profit_loss = net_value - (position_size - entry_fee)
        profit_loss_pct = ((current_price - entry_price) / entry_price) * 100

        return jsonify({
          'success': True,
          'position': {
            **pos_data,
            'current_price': current_price,
            'current_profit_loss': profit_loss,
            'current_profit_loss_pct': profit_loss_pct,
            'is_closed': False
          }
        })

    # Check closed positions
    conn = sqlite3.connect(app.config['DB_PATH'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
            SELECT * FROM position_history 
            WHERE symbol = ? 
            ORDER BY closed_at DESC 
            LIMIT 1
        ''', (symbol,))

    result = cursor.fetchone()
    conn.close()

    if result:
      return jsonify({
        'success': True,
        'position': {
          **dict(result),
          'is_closed': True
        }
      })

    return jsonify({
      'success': False,
      'error': 'Position not found'
    }), 404

  except Exception as e:
    logging.error(f"Error getting position details: {e}")
    return jsonify({
      'success': False,
      'error': str(e)
    }), 500

@app.route('/api/paper-trading/position/<symbol>/monitoring')
@login_required
def get_position_monitoring(symbol):
  """Get position monitoring history"""
  try:
    symbol = symbol.upper()
    limit = int(request.args.get('limit', 100))

    conn = sqlite3.connect(app.config['DB_PATH'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get position ID
    cursor.execute('''
            SELECT id FROM active_positions WHERE symbol = ?
            UNION
            SELECT id FROM (
                SELECT DISTINCT position_id as id 
                FROM position_monitoring 
                WHERE symbol = ?
                ORDER BY checked_at DESC
                LIMIT 1
            )
        ''', (symbol, symbol))

    pos_id_result = cursor.fetchone()

    if not pos_id_result:
      conn.close()
      return jsonify({
        'success': True,
        'monitoring': []
      })

    position_id = pos_id_result[0]

    # Get monitoring history
    cursor.execute('''
            SELECT * FROM position_monitoring 
            WHERE position_id = ? 
            ORDER BY checked_at DESC 
            LIMIT ?
        ''', (position_id, limit))

    monitoring = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify({
      'success': True,
      'monitoring': monitoring
    })

  except Exception as e:
    logging.error(f"Error getting monitoring history: {e}")
    return jsonify({
      'success': False,
      'error': str(e)
    }), 500


@app.route('/api/paper-trading/status')
@login_required
def get_paper_trading_status():
  """Get paper trading engine status from DATABASE"""
  try:
    # Get fresh stats from database, not memory
    stats = pt_manager.get_stats_from_db()

    if not stats:
      return jsonify({
        'success': False,
        'error': 'No trading data available'
      }), 404

    # Add running status from state file
    state = pt_manager.get_state()
    stats['running'] = state.get('running', False)

    return jsonify({
      'success': True,
      'stats': stats
    })
  except Exception as e:
    logging.error(f"Error getting status: {e}")
    return jsonify({
      'success': False,
      'error': str(e)
    }), 500


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
