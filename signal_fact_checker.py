"""
Signal Fact Checker - CLEANED VERSION
- Uses validation windows from signals table
- Updates signal accuracies after fact-checking
- 0.5% minimum profit threshold
- Fast batch processing with rate limit management
- Connection pooling and caching
- ONLY works with live_signals (position_signals removed)
"""

import sqlite3
from functools import reduce
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from scalp_signal_analyzer import ScalpSignalAnalyzer
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading

logging.basicConfig(level=logging.INFO)


class RateLimiter:
    """Rate limiter to avoid hitting API limits"""

    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if we're about to exceed rate limit"""
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]

            if len(self.requests) >= self.max_requests:
                # Wait until oldest request is >60s old
                sleep_time = 60 - (now - self.requests[0]) + 0.1
                if sleep_time > 0:
                    logging.debug(f"Rate limit: sleeping {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                    self.requests = []

            self.requests.append(now)

    def is_near_limit(self, threshold=0.9):
        """Check if we're near the rate limit (90% by default)"""
        with self.lock:
            now = time.time()
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            return len(self.requests) >= (self.max_requests * threshold)

    def remaining_capacity(self):
        """Get remaining requests before hitting limit"""
        with self.lock:
            now = time.time()
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            return max(0, self.max_requests - len(self.requests))


class PriceDataCache:
    """Cache for price data to avoid redundant API calls"""

    def __init__(self, ttl_seconds=300):
        self.cache = {}
        self.ttl = ttl_seconds
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return data
                else:
                    del self.cache[key]
        return None

    def set(self, key, value):
        with self.lock:
            self.cache[key] = (value, time.time())

    def clear_old(self):
        """Clear expired entries"""
        with self.lock:
            now = time.time()
            expired = [k for k, (_, ts) in self.cache.items() if now - ts >= self.ttl]
            for k in expired:
                del self.cache[k]


class SignalFactChecker:
    """
    Fact-checker with:
    - Validation windows from signals table
    - Signal accuracy updates
    - 0.5% minimum profit threshold
    - Fast batch processing
    - Rate limit management
    - Connection pooling
    - ONLY works with live_signals
    """

    # USER REQUIREMENT: 0.5% minimum price movement
    MIN_PROFIT_THRESHOLD_PCT = 0.1

    # Stop-loss (can be overridden per signal)
    STOP_LOSS_PCT = 3.0

    # Fallback validation windows (if not in signals table)
    FALLBACK_VALIDATION_WINDOWS = {
        '1m': 10, '3m': 10, '5m': 12, '15m': 8,
        '30m': 8, '1h': 8, '2h': 6, '4h': 6,
        '6h': 8, '8h': 6, '12h': 8, '1d': 5,
        '3d': 5, '1w': 4,
    }

    # Rate limiting
    KUCOIN_RATE_LIMIT = 100
    BINANCE_RATE_LIMIT = 1200

    # Parallel processing
    MAX_WORKERS = 10

    def __init__(self, db_path: str = 'crypto_signals.db'):
        self.db_path = db_path
        self.analyzer = ScalpSignalAnalyzer()
        self.timeframe_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15,
            '30m': 30, '1h': 60, '2h': 120, '4h': 240,
            '6h': 360, '8h': 480, '12h': 720, '1d': 1440,
            '3d': 4320, '1w': 10080
        }

        self.kucoin_limiter = RateLimiter(self.KUCOIN_RATE_LIMIT)
        self.binance_limiter = RateLimiter(self.BINANCE_RATE_LIMIT)
        self.price_cache = PriceDataCache(ttl_seconds=300)

        # Connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        # Cache for validation windows
        self.validation_window_cache = {}
        self._load_validation_windows()

    def _load_validation_windows(self):
        """Load all validation windows from signals table into cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT signal_name, timeframe, validation_window
                FROM signals
            ''')

            for signal_name, timeframe, validation_window in cursor.fetchall():
                key = f"{signal_name}:{timeframe}"
                self.validation_window_cache[key] = validation_window

            logging.info(f"✅ Loaded {len(self.validation_window_cache)} validation windows from signals table")
        except sqlite3.OperationalError:
            logging.warning("⚠️  Signals table not found - using fallback validation windows")
        finally:
            conn.close()

    def get_validation_window(self, signal_name: str, timeframe: str) -> int:
        """Get validation window for signal-timeframe from signals table"""
        key = f"{signal_name}:{timeframe}"

        # Check cache first
        if key in self.validation_window_cache:
            return self.validation_window_cache[key]

        # Try to fetch from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT validation_window FROM signals
                WHERE signal_name = ? AND timeframe = ?
            ''', (signal_name, timeframe))

            result = cursor.fetchone()
            if result:
                window = result[0]
                self.validation_window_cache[key] = window
                return window
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()

        # Fallback to default
        return self.FALLBACK_VALIDATION_WINDOWS.get(timeframe, 5)

    def _get_cache_key(self, symbol: str, timestamp: datetime, timeframe: str, candles: int) -> str:
        ts_str = timestamp.strftime('%Y%m%d%H%M')
        return f"{symbol}:{timeframe}:{ts_str}:{candles}"

    def fetch_price_journey(self, symbol: str, timestamp: datetime,
                            timeframe: str, candles_ahead: int = 10) -> Optional[List[Dict]]:
        """
        Fetch with smart API selection:
        - Use Binance if KuCoin is near rate limit
        - Otherwise try KuCoin first, then Binance fallback
        """
        # Check cache
        cache_key = self._get_cache_key(symbol, timestamp, timeframe, candles_ahead)
        cached_data = self.price_cache.get(cache_key)
        if cached_data:
            return cached_data

        minutes = self.timeframe_minutes.get(timeframe, 60)
        target_time = timestamp + timedelta(minutes=minutes * (candles_ahead + 2))

        # Smart API selection: Check if KuCoin is near limit
        if self.kucoin_limiter.is_near_limit(threshold=0.85):
            logging.debug(f"KuCoin near limit ({self.kucoin_limiter.remaining_capacity()} remaining), using Binance")
            # Use Binance first if KuCoin is near limit
            binance_data = self._fetch_binance_journey(symbol, timestamp, target_time, timeframe, candles_ahead)
            if binance_data:
                self.price_cache.set(cache_key, binance_data)
                return binance_data

            # Binance failed, try KuCoin anyway
            kucoin_data = self._fetch_kucoin_journey(symbol, timestamp, target_time, timeframe, candles_ahead)
            if kucoin_data:
                self.price_cache.set(cache_key, kucoin_data)
                return kucoin_data
        else:
            # Normal flow: Try KuCoin first
            kucoin_data = self._fetch_kucoin_journey(symbol, timestamp, target_time, timeframe, candles_ahead)
            if kucoin_data:
                self.price_cache.set(cache_key, kucoin_data)
                return kucoin_data

            # KuCoin failed, fallback to Binance
            binance_data = self._fetch_binance_journey(symbol, timestamp, target_time, timeframe, candles_ahead)
            if binance_data:
                self.price_cache.set(cache_key, binance_data)
                return binance_data

        return None

    def _fetch_kucoin_journey(self, symbol: str, start_time: datetime,
                              end_time: datetime, timeframe: str,
                              candles_ahead: int) -> Optional[List[Dict]]:
        try:
            self.kucoin_limiter.wait_if_needed()

            symbol_pair = f"{symbol}-USDT"
            url = "https://api.kucoin.com/api/v1/market/candles"

            tf_map = {
                '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min',
                '30m': '30min', '1h': '1hour', '2h': '2hour', '4h': '4hour',
                '6h': '6hour', '8h': '8hour', '12h': '12hour', '1d': '1day',
                '1w': '1week'
            }

            params = {
                'symbol': symbol_pair,
                'type': tf_map.get(timeframe, '1hour'),
                'startAt': int(start_time.timestamp()),
                'endAt': int(end_time.timestamp())
            }

            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            if data.get('code') != '200000' or not data.get('data'):
                return None

            raw_candles = data['data']

            candles = []
            for candle in reversed(raw_candles):
                candles.append({
                    'timestamp': int(candle[0]),
                    'open': float(candle[1]),
                    'close': float(candle[2]),
                    'high': float(candle[3]),
                    'low': float(candle[4]),
                    'volume': float(candle[5])
                })

            if len(candles) < candles_ahead + 1:
                return None

            minutes = self.timeframe_minutes[timeframe]
            expected_start = int(start_time.timestamp())
            actual_start = candles[0]['timestamp']

            if abs(actual_start - expected_start) > minutes * 60 * 2:
                return None

            return candles

        except Exception as e:
            logging.debug(f"KuCoin failed {symbol}: {e}")
            return None

    def _fetch_binance_journey(self, symbol: str, start_time: datetime,
                               end_time: datetime, timeframe: str,
                               candles_ahead: int) -> Optional[List[Dict]]:
        try:
            self.binance_limiter.wait_if_needed()

            symbol_pair = f"{symbol}USDT"
            url = "https://api.binance.com/api/v3/klines"

            params = {
                'symbol': symbol_pair,
                'interval': timeframe,
                'startTime': int(start_time.timestamp() * 1000),
                'endTime': int(end_time.timestamp() * 1000),
                'limit': candles_ahead + 5
            }

            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            raw_candles = response.json()

            if not raw_candles or len(raw_candles) < candles_ahead + 1:
                return None

            candles = []
            for candle in raw_candles:
                candles.append({
                    'timestamp': int(candle[0] / 1000),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })

            return candles

        except Exception as e:
            logging.debug(f"Binance failed {symbol}: {e}")
            return None

    def validate_signal_with_stop_loss(self, entry_price: float, signal_type: str,
                                       candles: List[Dict],
                                       stop_loss_pct: float = None) -> Tuple[bool, str, float]:
        """
        USER REQUIREMENT: Signal successful if price moves >0.5%
        """
        if not candles or len(candles) < 2:
            return False, 'INSUFFICIENT_DATA', 0.0

        stop_loss = stop_loss_pct or self.STOP_LOSS_PCT

        if signal_type == 'BUY':
            stop_loss_price = entry_price * (1 - stop_loss / 100)

            # Check stop-loss
            for i, candle in enumerate(candles[1:], 1):
                if candle['low'] <= stop_loss_price:
                    loss_pct = ((stop_loss_price - entry_price) / entry_price) * 100
                    return False, f'STOPPED_OUT_CANDLE_{i}', loss_pct

            # Check final profit
            final_price = candles[-1]['close']
            price_change_pct = ((final_price - entry_price) / entry_price) * 100

            # USER REQUIREMENT: >0.5% = success
            if price_change_pct > self.MIN_PROFIT_THRESHOLD_PCT:
                return True, 'PROFIT_TARGET', price_change_pct
            elif price_change_pct > 0:
                return False, 'PROFIT_TOO_SMALL', price_change_pct
            else:
                return False, 'LOSS', price_change_pct

        elif signal_type == 'SELL':
            stop_loss_price = entry_price * (1 + stop_loss / 100)

            for i, candle in enumerate(candles[1:], 1):
                if candle['high'] >= stop_loss_price:
                    loss_pct = ((entry_price - stop_loss_price) / entry_price) * 100
                    return False, f'STOPPED_OUT_CANDLE_{i}', loss_pct

            final_price = candles[-1]['close']
            price_change_pct = ((entry_price - final_price) / entry_price) * 100

            if price_change_pct > self.MIN_PROFIT_THRESHOLD_PCT:
                return True, 'PROFIT_TARGET', -price_change_pct
            elif price_change_pct > 0:
                return False, 'PROFIT_TOO_SMALL', -price_change_pct
            else:
                return False, 'LOSS', -price_change_pct

        return False, 'INVALID_SIGNAL_TYPE', 0.0

    def fact_check_signal(self, signal_name: str, signal_type: str,
                          timeframe: str, detected_at: datetime,
                          price_at_detection: float, symbol: str,
                          candles_ahead: int = None,
                          stop_loss_pct: float = None) -> Optional[Dict]:
        """
        Fact-check a single signal
        """
        # Use validation window from signals table if not provided
        if candles_ahead is None:
            candles_ahead = self.get_validation_window(signal_name, timeframe)

        candles = self.fetch_price_journey(symbol, detected_at, timeframe, candles_ahead)

        if candles is None or len(candles) < 2:
            return None

        predicted_correctly, exit_reason, price_change_pct = self.validate_signal_with_stop_loss(
            price_at_detection,
            signal_type,
            candles,
            stop_loss_pct
        )

        if price_change_pct > self.MIN_PROFIT_THRESHOLD_PCT:
            actual_move = 'UP'
        elif price_change_pct < -self.MIN_PROFIT_THRESHOLD_PCT:
            actual_move = 'DOWN'
        else:
            actual_move = 'FLAT'

        return {
            'signal_name': signal_name,
            'timeframe': timeframe,
            'detected_at': detected_at,
            'price_at_detection': price_at_detection,
            'actual_move': actual_move,
            'predicted_correctly': predicted_correctly,
            'price_change_pct': price_change_pct,
            'exit_reason': exit_reason,
            'checked_at': datetime.now(),
            'candles_elapsed': len(candles) - 1,
            'validation_window': candles_ahead
        }

    def _fact_check_single_worker(self, signal: Dict) -> Optional[Dict]:
        """Worker for parallel fact-checking of live_signals"""
        try:
            result = self.fact_check_signal(
                signal['signal_name'],
                signal['signal_type'],
                signal['timeframe'],
                datetime.fromisoformat(signal['timestamp']),
                signal['price'],
                signal['symbol']
            )
            return result
        except Exception as e:
            logging.error(f"Error: {signal['symbol']}: {e}")
            return None

    def bulk_fact_check_live_signals(self, symbol: str = None,
                                     limit: int = None,
                                     use_parallel: bool = True,
                                     max_workers: int = None) -> Dict:
        """
        Main entry point for fact-checking live_signals
        Can use parallel or sequential processing
        """
        if use_parallel:
            return self._bulk_fact_check_parallel(symbol, limit, max_workers)
        else:
            return self._bulk_fact_check_sequential(symbol, limit)

    def _bulk_fact_check_parallel(self, symbol: str = None,
                                  limit: int = None,
                                  max_workers: int = None) -> Dict:
        """FAST parallel fact-checking of live_signals"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get signals that haven't been fact-checked yet
        query = '''
            SELECT ls.* 
            FROM live_signals ls
            LEFT JOIN signal_fact_checks sfc 
                ON ls.signal_name = sfc.signal_name 
                AND ls.timeframe = sfc.timeframe 
                AND ls.timestamp = sfc.detected_at
            WHERE sfc.id IS NULL
        '''
        params = []

        if symbol:
            query += ' AND ls.symbol = ?'
            params.append(symbol)

        query += ' ORDER BY ls.timestamp DESC'

        if limit:
            query += f' LIMIT {limit}'

        cursor.execute(query, params)
        signals = [dict(row) for row in cursor.fetchall()]
        conn.close()

        total_signals = len(signals)
        logging.info(f"🚀 Parallel fact-check: {total_signals} signals...")

        results = {
            'total_checked': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'stopped_out': 0,
            'accuracy': 0,
            'profit_factor': 0,
            'details': [],
            'exit_reasons': {}
        }

        if total_signals == 0:
            logging.info("No signals to check")
            return results

        workers = max_workers or self.MAX_WORKERS
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_signal = {
                executor.submit(self._fact_check_single_worker, signal): signal
                for signal in signals
            }

            completed = 0
            for future in as_completed(future_to_signal):
                completed += 1
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (total_signals - completed) / rate if rate > 0 else 0
                    logging.info(f"Progress: {completed}/{total_signals} ({completed / total_signals * 100:.1f}%) "
                                 f"- {rate:.1f}/s - ETA: {eta:.0f}s")

                result = future.result()

                if result:
                    self.save_fact_check(result)
                    results['details'].append(result)
                    results['total_checked'] += 1

                    if result['predicted_correctly']:
                        results['correct_predictions'] += 1
                    else:
                        results['incorrect_predictions'] += 1
                        if 'STOPPED_OUT' in result.get('exit_reason', ''):
                            results['stopped_out'] += 1

                    exit_reason = result.get('exit_reason', 'UNKNOWN')
                    results['exit_reasons'][exit_reason] = results['exit_reasons'].get(exit_reason, 0) + 1

        elapsed = time.time() - start_time

        if results['total_checked'] > 0:
            results['accuracy'] = (results['correct_predictions'] / results['total_checked']) * 100

            winning_sum = sum([abs(r['price_change_pct']) for r in results['details'] if r['predicted_correctly']])
            losing_sum = sum([abs(r['price_change_pct']) for r in results['details'] if not r['predicted_correctly']])

            if losing_sum > 0:
                results['profit_factor'] = winning_sum / losing_sum
            else:
                results['profit_factor'] = winning_sum if winning_sum > 0 else 0

        logging.info(
            f"✅ Done: {results['total_checked']} checks in {elapsed:.1f}s ({results['total_checked'] / elapsed:.1f}/s)")
        logging.info(f"   Accuracy: {results['accuracy']:.1f}%")
        logging.info(f"   Profit Factor: {results['profit_factor']:.2f}")
        logging.info(
            f"   Stopped: {results['stopped_out']} ({results['stopped_out'] / max(results['total_checked'], 1) * 100:.1f}%)")

        # Update signal accuracies
        self._update_signal_accuracies_from_results(results['details'])

        return results

    def _bulk_fact_check_sequential(self, symbol: str = None, limit: int = None) -> Dict:
        """Sequential fact-checking (for debugging)"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = '''
            SELECT ls.* 
            FROM live_signals ls
            LEFT JOIN signal_fact_checks sfc 
                ON ls.signal_name = sfc.signal_name 
                AND ls.timeframe = sfc.timeframe 
                AND ls.timestamp = sfc.detected_at
            WHERE sfc.id IS NULL
        '''
        params = []

        if symbol:
            query += ' AND ls.symbol = ?'
            params.append(symbol)

        query += ' ORDER BY ls.timestamp DESC'

        if limit:
            query += f' LIMIT {limit}'

        cursor.execute(query, params)
        signals = cursor.fetchall()
        conn.close()

        results = {
            'total_checked': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'stopped_out': 0,
            'accuracy': 0,
            'profit_factor': 0,
            'details': [],
            'exit_reasons': {}
        }

        for idx, signal in enumerate(signals, 1):
            if idx % 50 == 0:
                logging.info(f"Progress: {idx}/{len(signals)}")

            result = self.fact_check_signal(
                signal['signal_name'],
                signal['signal_type'],
                signal['timeframe'],
                datetime.fromisoformat(signal['timestamp']),
                signal['price'],
                signal['symbol']
            )

            if result:
                self.save_fact_check(result)
                results['details'].append(result)
                results['total_checked'] += 1

                if result['predicted_correctly']:
                    results['correct_predictions'] += 1
                else:
                    results['incorrect_predictions'] += 1
                    if 'STOPPED_OUT' in result.get('exit_reason', ''):
                        results['stopped_out'] += 1

                exit_reason = result.get('exit_reason', 'UNKNOWN')
                results['exit_reasons'][exit_reason] = results['exit_reasons'].get(exit_reason, 0) + 1

        if results['total_checked'] > 0:
            results['accuracy'] = (results['correct_predictions'] / results['total_checked']) * 100

            winning_sum = sum([abs(r['price_change_pct']) for r in results['details'] if r['predicted_correctly']])
            losing_sum = sum([abs(r['price_change_pct']) for r in results['details'] if not r['predicted_correctly']])

            if losing_sum > 0:
                results['profit_factor'] = winning_sum / losing_sum
            else:
                results['profit_factor'] = winning_sum if winning_sum > 0 else 0

        # Update signal accuracies
        self._update_signal_accuracies_from_results(results['details'])

        return results

    def save_fact_check(self, result: Dict):
        """Save fact-check result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if columns exist, add if needed
        cursor.execute("PRAGMA table_info(signal_fact_checks)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'exit_reason' not in columns:
            cursor.execute('ALTER TABLE signal_fact_checks ADD COLUMN exit_reason TEXT')
        if 'validation_window' not in columns:
            cursor.execute('ALTER TABLE signal_fact_checks ADD COLUMN validation_window INTEGER')

        cursor.execute('''
            INSERT INTO signal_fact_checks (
                signal_name, timeframe, detected_at, price_at_detection,
                actual_move, predicted_correctly, price_change_pct,
                checked_at, candles_elapsed, exit_reason, validation_window
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['signal_name'], result['timeframe'],
            result['detected_at'], result['price_at_detection'],
            result['actual_move'], result['predicted_correctly'],
            result['price_change_pct'], result['checked_at'],
            result['candles_elapsed'], result.get('exit_reason'),
            result.get('validation_window')
        ))

        conn.commit()
        conn.close()

    def _update_signal_accuracies_from_results(self, fact_check_results: List[Dict]):
        """
        Update signal accuracies in signals table based on fact-check results
        Groups by signal-timeframe and calculates accuracy
        """
        if not fact_check_results:
            return

        # Group results by signal-timeframe
        signal_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

        for result in fact_check_results:
            key = f"{result['signal_name']}:{result['timeframe']}"
            signal_stats[key]['total'] += 1
            if result['predicted_correctly']:
                signal_stats[key]['correct'] += 1

        # Update database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for key, stats in signal_stats.items():
            signal_name, timeframe = key.split(':', 1)

            # Get current stats from database
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN predicted_correctly = 1 THEN 1 ELSE 0 END) as correct
                FROM signal_fact_checks
                WHERE signal_name = ? AND timeframe = ?
            ''', (signal_name, timeframe))

            db_result = cursor.fetchone()
            if db_result and db_result[0] > 0:
                total = db_result[0]
                correct = db_result[1] or 0
                accuracy = (correct / total) * 100

                # Update signals table
                cursor.execute('''
                    UPDATE signals
                    SET signal_accuracy = ?,
                        sample_size = ?,
                        updated_at = ?
                    WHERE signal_name = ? AND timeframe = ?
                ''', (accuracy, total, datetime.now(), signal_name, timeframe))

        conn.commit()
        conn.close()

        logging.info(f"✅ Updated accuracies for {len(signal_stats)} signal-timeframe combinations")

    def calculate_signal_accuracy(self, signal_name: str, timeframe: str = None,
                                  min_samples: int = 10) -> Optional[Dict]:
        """Calculate accuracy statistics for a signal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = '''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN predicted_correctly = 1 THEN 1 ELSE 0 END) as correct,
                AVG(price_change_pct) as avg_price_change,
                AVG(CASE WHEN predicted_correctly = 1 THEN price_change_pct ELSE 0 END) as avg_win,
                AVG(CASE WHEN predicted_correctly = 0 THEN price_change_pct ELSE 0 END) as avg_loss,
                SUM(CASE WHEN exit_reason LIKE '%STOPPED_OUT%' THEN 1 ELSE 0 END) as stopped_out
            FROM signal_fact_checks
            WHERE signal_name = ?
        '''
        params = [signal_name]

        if timeframe:
            query += ' AND timeframe = ?'
            params.append(timeframe)

        cursor.execute(query, params)
        result = cursor.fetchone()
        conn.close()

        if not result or result[0] < min_samples:
            return None

        total, correct, avg_change, avg_win, avg_loss, stopped_out = result
        accuracy = (correct / total) * 100

        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        return {
            'signal_name': signal_name,
            'timeframe': timeframe,
            'total_samples': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'avg_price_change': avg_change or 0,
            'avg_win': avg_win or 0,
            'avg_loss': avg_loss or 0,
            'profit_factor': profit_factor,
            'stopped_out': stopped_out or 0,
            'stopped_out_rate': (stopped_out / total * 100) if stopped_out else 0
        }

    def adjust_signal_confidence(self, signal_name: str, timeframe: str,
                                 min_samples: int = 10) -> Optional[Dict]:
        """Adjust signal confidence based on historical accuracy"""
        accuracy_data = self.calculate_signal_accuracy(signal_name, timeframe, min_samples)

        if not accuracy_data:
            return None

        original_conf = ScalpSignalAnalyzer.SIGNAL_CONFIDENCE.get(signal_name, {}).get('confidence', 70)

        sample_weight = min(1.0, accuracy_data['total_samples'] / 500)
        base_adjusted = original_conf * (1 - sample_weight) + accuracy_data['accuracy'] * sample_weight

        profit_factor = accuracy_data.get('profit_factor', 1.0)
        if profit_factor > 2.0:
            profit_bonus = min(10, (profit_factor - 2.0) * 5)
        elif profit_factor < 1.0:
            profit_bonus = max(-15, (profit_factor - 1.0) * 15)
        else:
            profit_bonus = 0

        stopped_out_rate = accuracy_data.get('stopped_out_rate', 0)
        stop_penalty = max(0, (stopped_out_rate - 30) * 0.3) if stopped_out_rate > 30 else 0

        adjusted_conf = int(base_adjusted + profit_bonus - stop_penalty)
        adjusted_conf = max(0, min(100, adjusted_conf))

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if columns exist
        cursor.execute("PRAGMA table_info(signal_confidence_adjustments)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'profit_factor' not in columns:
            cursor.execute('ALTER TABLE signal_confidence_adjustments ADD COLUMN profit_factor REAL')
        if 'stopped_out_rate' not in columns:
            cursor.execute('ALTER TABLE signal_confidence_adjustments ADD COLUMN stopped_out_rate REAL')
        if 'sample_weight' not in columns:
            cursor.execute('ALTER TABLE signal_confidence_adjustments ADD COLUMN sample_weight REAL')

        cursor.execute('''
            INSERT OR REPLACE INTO signal_confidence_adjustments (
                signal_name, timeframe, original_confidence, adjusted_confidence,
                accuracy_rate, sample_size, profit_factor, stopped_out_rate,
                sample_weight, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_name, timeframe, original_conf, adjusted_conf,
            accuracy_data['accuracy'], accuracy_data['total_samples'],
            accuracy_data.get('profit_factor', 0), accuracy_data.get('stopped_out_rate', 0),
            sample_weight, datetime.now()
        ))

        conn.commit()
        conn.close()

        logging.info(f"✅ Adjusted {signal_name}[{timeframe}]: {original_conf}→{adjusted_conf}")

        return {
            'signal_name': signal_name,
            'timeframe': timeframe,
            'original_confidence': original_conf,
            'adjusted_confidence': adjusted_conf,
            'accuracy_rate': accuracy_data['accuracy'],
            'sample_size': accuracy_data['total_samples'],
            'profit_factor': accuracy_data.get('profit_factor', 0),
            'confidence_change': adjusted_conf - original_conf
        }

    def bulk_adjust_all_signals(self, min_samples: int = 10) -> Dict:
        """Adjust confidence for all signals with sufficient data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT DISTINCT signal_name, timeframe
            FROM signal_fact_checks
        ''')

        combinations = cursor.fetchall()
        conn.close()

        results = {
            'total_processed': 0,
            'adjusted': 0,
            'skipped_insufficient_samples': 0,
            'adjustments': []
        }

        for signal_name, timeframe in combinations:
            results['total_processed'] += 1

            adjustment = self.adjust_signal_confidence(signal_name, timeframe, min_samples)

            if adjustment:
                results['adjusted'] += 1
                results['adjustments'].append(adjustment)
            else:
                results['skipped_insufficient_samples'] += 1

        logging.info(f"✅ Bulk adjust: {results['adjusted']}/{results['total_processed']}")

        return results

    def get_adjusted_confidence(self, signal_name: str, timeframe: str) -> int:
        """Get adjusted confidence for a signal, or return original if not adjusted"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT adjusted_confidence FROM signal_confidence_adjustments
            WHERE signal_name = ? AND timeframe = ?
        ''', (signal_name, timeframe))

        result = cursor.fetchone()
        conn.close()

        if result:
            return result[0]

        return ScalpSignalAnalyzer.SIGNAL_CONFIDENCE.get(signal_name, {}).get('confidence', 70)

    def get_all_adjustments(self) -> List[Dict]:
        """Get all signal confidence adjustments"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM signal_confidence_adjustments
            ORDER BY last_updated DESC
        ''')

        adjustments = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return adjustments

    def get_all_adjusted_confidence(self) -> List[Dict]:
        """Get all adjusted confidences in a simple format for API"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT 
                signal_name,
                timeframe,
                original_confidence,
                adjusted_confidence,
                accuracy_rate,
                sample_size
            FROM signal_confidence_adjustments
            ORDER BY signal_name, timeframe
        ''')

        signals = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return signals

    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Overall statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_checks,
                SUM(CASE WHEN predicted_correctly = 1 THEN 1 ELSE 0 END) as correct,
                AVG(price_change_pct) as avg_change,
                SUM(CASE WHEN exit_reason LIKE '%STOPPED_OUT%' THEN 1 ELSE 0 END) as stopped_out
            FROM signal_fact_checks
        ''')

        overall = cursor.fetchone()

        # Exit reasons distribution
        cursor.execute('''
            SELECT exit_reason, COUNT(*) as count
            FROM signal_fact_checks
            GROUP BY exit_reason
            ORDER BY count DESC
        ''')

        exit_reasons = cursor.fetchall()

        # Top performing signals
        cursor.execute('''
            SELECT 
                signal_name, timeframe, COUNT(*) as samples,
                ROUND(AVG(CASE WHEN predicted_correctly = 1 THEN 100.0 ELSE 0.0 END), 2) as accuracy,
                ROUND(AVG(price_change_pct), 3) as avg_change
            FROM signal_fact_checks
            GROUP BY signal_name, timeframe
            HAVING samples >= 20
            ORDER BY accuracy DESC
            LIMIT 10
        ''')

        top_signals = cursor.fetchall()

        conn.close()

        return {
            'overall': {
                'total_checks': overall[0],
                'correct': overall[1],
                'accuracy': (overall[1] / overall[0] * 100) if overall[0] > 0 else 0,
                'avg_change': overall[2] or 0,
                'stopped_out': overall[3],
                'stopped_out_rate': (overall[3] / overall[0] * 100) if overall[0] > 0 else 0
            },
            'exit_reasons': {reason: count for reason, count in exit_reasons},
            'top_signals': [
                {
                    'signal': f"{signal}[{tf}]",
                    'samples': samples,
                    'accuracy': accuracy,
                    'avg_change': avg_change
                }
                for signal, tf, samples, accuracy, avg_change in top_signals
            ]
        }

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'session'):
            self.session.close()


if __name__ == "__main__":
    checker = SignalFactChecker()

    print("\n" + "=" * 80)
    print("CLEANED FACT-CHECKER - LIVE SIGNALS ONLY")
    print("=" * 80)
    print(f"\nSettings:")
    print(f"  Min Profit: {checker.MIN_PROFIT_THRESHOLD_PCT}%")
    print(f"  Stop-Loss: {checker.STOP_LOSS_PCT}%")
    print(f"  Workers: {checker.MAX_WORKERS}")
    print(f"  Validation windows: From signals table")
    print(f"  Signal accuracies: Auto-updated after fact-checking")

    print(f"\n{'=' * 80}")
    print("TEST: Fact-checking live_signals")
    print(f"{'=' * 80}\n")

    results = checker.bulk_fact_check_live_signals(limit=100, use_parallel=True)

    print("\n✅ Test complete!")
    print(f"Checked: {results['total_checked']} signals")
    print(f"Accuracy: {results['accuracy']:.1f}%")
    print(f"Signal accuracies have been updated in the signals table")