"""
Complete Flask Application for Crypto Pattern Trading Dashboard
Integrated backend + frontend deployment
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
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this!
app.config['DB_PATH'] = 'crypto_signals.db'
app.config['PRIORITY_COINS_FILE'] = 'priority_coins.json'


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
    if data.get('username') == 'iheartsogol' and data.get('password') == 'sogolpleasecomeback:(((':
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


# ==================== HELPER FUNCTIONS ====================

def calculate_validity_hours(confidence, pattern_count):
  """Calculate signal validity based on confidence and pattern count"""
  # Base validity: 12-60 hours
  base = 12

  # Higher confidence = longer validity
  confidence_factor = (confidence - 0.7) / 0.3  # 0-1 range from 0.7-1.0

  # More patterns = longer validity (with diminishing returns)
  pattern_factor = min(pattern_count / 200, 1.0)  # Cap at 200 patterns

  # Combined factor
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

# Global monitor instance
monitor = None


@app.route('/api/monitor/start')
@login_required
def start_monitor():
  global monitor
  if monitor is None or not monitor.running:
    monitor = PatternMonitor(app.config['DB_PATH'])
    thread = threading.Thread(target=monitor.start_monitoring, daemon=True)
    thread.start()
    return jsonify({'success': True, 'message': 'Monitoring started'})
  return jsonify({'success': False, 'message': 'Already running'})


@app.route('/api/monitor/stop')
@login_required
def stop_monitor():
  global monitor
  if monitor and monitor.running:
    monitor.stop_monitoring()
    return jsonify({'success': True, 'message': 'Monitoring stopped'})
  return jsonify({'success': False, 'message': 'Not running'})


@app.route('/api/monitor/status')
@login_required
def monitor_status():
  global monitor
  return jsonify({
    'running': monitor.running if monitor else False
  })


# ==================== STARTUP ====================

if __name__ == '__main__':
  # Initialize database
  init_database()

  # Run Flask app
  app.run(host='0.0.0.0', port=5001, debug=True)