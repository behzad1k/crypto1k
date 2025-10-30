"""
Signal Combination Analyzer - FIXED SQL QUERY
Prevents duplicate row explosion in the JOIN
FIX: Removed symbol from signal_fact_checks JOIN (column doesn't exist in that table)
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set, Optional
from itertools import combinations, product
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)


class SignalCombinationAnalyzer:
    """
    FIXED: Corrected SQL JOIN to only use columns that exist in both tables
    """

    def __init__(self, db_path: str = 'crypto_signals.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Create tables for both same-timeframe and cross-timeframe combination results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Existing table for same-timeframe combinations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tf_combos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_name TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                accuracy REAL NOT NULL,
                signals_count INTEGER NOT NULL,
                correct_predictions INTEGER NOT NULL,
                avg_price_change REAL,
                profit_factor REAL,
                combo_size INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(signal_name, timeframe)
            )
        ''')

        # NEW: Table for cross-timeframe combinations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_tf_combos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                combo_signature TEXT NOT NULL,
                timeframes TEXT NOT NULL,
                signal_names TEXT NOT NULL,
                accuracy REAL NOT NULL,
                signals_count INTEGER NOT NULL,
                correct_predictions INTEGER NOT NULL,
                avg_price_change REAL,
                profit_factor REAL,
                combo_size INTEGER NOT NULL,
                num_timeframes INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(combo_signature)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_combos_accuracy 
            ON tf_combos(timeframe, accuracy DESC)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_combos_size 
            ON tf_combos(combo_size, accuracy DESC)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_cross_combos_accuracy 
            ON cross_tf_combos(accuracy DESC)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_cross_combos_size 
            ON cross_tf_combos(combo_size, num_timeframes, accuracy DESC)
        ''')

        conn.commit()
        conn.close()
        logging.info("✅ Database tables initialized (tf_combos + cross_tf_combos)")

    def get_timeframe_window_seconds(self, timeframe: str) -> int:
        """Get time window in seconds for grouping signals"""
        timeframe_seconds = {
            '1m': 60, '3m': 180, '5m': 300, '15m': 900,
            '30m': 1800, '1h': 3600, '2h': 7200, '4h': 14400,
            '6h': 21600, '8h': 28800, '12h': 43200, '1d': 86400,
            '3d': 259200, '1w': 604800
        }
        return timeframe_seconds.get(timeframe, 3600)

    # ========================================================================
    # FIXED SAME-TIMEFRAME FUNCTIONALITY
    # ========================================================================

    def find_signal_combinations(self, timeframe: str, min_combo_size: int = 2,
                                 max_combo_size: int = 5,
                                 max_combinations_per_symbol: int = 1000,
                                 strict_timing: bool = True) -> Dict[str, List[Dict]]:
        """
        FIXED: Proper JOIN without symbol column from signal_fact_checks

        The signal_fact_checks table doesn't have a 'symbol' column, so we removed it
        from the JOIN condition. We still get the symbol from live_signals.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # ⭐ FIXED SQL QUERY - Removed symbol from JOIN condition
        cursor.execute('''
            SELECT 
                ls.id,
                ls.symbol,
                ls.signal_name,
                ls.signal_type,
                ls.timestamp,
                sfc.predicted_correctly,
                sfc.price_change_pct
            FROM live_signals ls
            INNER JOIN signal_fact_checks sfc
                ON ls.signal_name = sfc.signal_name
                AND ls.timeframe = sfc.timeframe
                AND ls.timestamp = sfc.detected_at
            WHERE ls.timeframe = ?
            ORDER BY ls.symbol, ls.timestamp
        ''', (timeframe,))

        signals = [dict(row) for row in cursor.fetchall()]
        conn.close()

        # ⭐ SANITY CHECK - Verify we don't have duplicates
        signal_ids = [s['id'] for s in signals]
        unique_ids = set(signal_ids)

        if len(signal_ids) != len(unique_ids):
            logging.warning(f"⚠️  Found {len(signal_ids) - len(unique_ids)} duplicate signal IDs!")
            # Remove duplicates by keeping first occurrence
            seen_ids = set()
            signals = [s for s in signals if s['id'] not in seen_ids and not seen_ids.add(s['id'])]
            logging.info(f"   Deduplicated to {len(signals)} unique signals")

        logging.info(f"📥 Found {len(signals)} fact-checked signals for {timeframe}")

        # Group signals by symbol and time window
        window_seconds = self.get_timeframe_window_seconds(timeframe)
        signal_groups = defaultdict(list)

        for signal in signals:
            timestamp = datetime.fromisoformat(signal['timestamp'])

            if strict_timing:
                # STRICT: Round to exact candle boundary
                total_seconds = int(timestamp.timestamp())
                window_start_seconds = (total_seconds // window_seconds) * window_seconds
                window_start = datetime.fromtimestamp(window_start_seconds)
            else:
                # LOOSE: Just round to nearest minute
                window_start = timestamp.replace(second=0, microsecond=0)

            window_key = f"{signal['symbol']}_{window_start.isoformat()}"
            signal_groups[window_key].append(signal)

        logging.info(f"🗂️  Grouped into {len(signal_groups)} time windows")

        # Find combinations in each group with limit
        all_combinations = defaultdict(list)
        symbol_combo_counts = defaultdict(int)
        total_windows = len(signal_groups)
        processed_windows = 0
        skipped_windows = 0

        for window_key, window_signals in signal_groups.items():
            processed_windows += 1

            if len(window_signals) < min_combo_size:
                skipped_windows += 1
                continue

            # Extract symbol for tracking
            symbol = window_signals[0]['symbol']

            # Check if this symbol has hit the limit
            if symbol_combo_counts[symbol] >= max_combinations_per_symbol:
                skipped_windows += 1
                continue

            # Get unique signal names in this window
            signal_names = list(set(s['signal_name'] for s in window_signals))

            # OPTIMIZATION: Skip if too many unique signals (likely noise)
            if len(signal_names) > max_combo_size * 2:
                skipped_windows += 1
                continue

            # Generate combinations of different sizes
            for combo_size in range(min_combo_size, min(max_combo_size + 1, len(signal_names) + 1)):
                for combo in combinations(sorted(signal_names), combo_size):
                    combo_key = '+'.join(sorted(combo))

                    # Check if all signals in combo were present
                    combo_signals = [s for s in window_signals if s['signal_name'] in combo]

                    # STRICT: Ensure each signal in combo appears exactly once
                    signal_names_in_combo = [s['signal_name'] for s in combo_signals]
                    if len(set(signal_names_in_combo)) != len(combo):
                        continue  # Skip if duplicate signals

                    # Use the result from first signal
                    result = {
                        'predicted_correctly': combo_signals[0]['predicted_correctly'],
                        'price_change_pct': combo_signals[0]['price_change_pct'],
                        'signal_types': [s['signal_type'] for s in combo_signals],
                        'symbol': symbol,
                        'timestamp': window_signals[0]['timestamp']
                    }

                    all_combinations[combo_key].append(result)
                    symbol_combo_counts[symbol] += 1

                    # Check limit
                    if symbol_combo_counts[symbol] >= max_combinations_per_symbol:
                        break

            # Progress logging
            if processed_windows % 1000 == 0:
                logging.info(f"   Progress: {processed_windows}/{total_windows} windows | "
                           f"Combos: {len(all_combinations)} | Skipped: {skipped_windows}")

        logging.info(f"✅ Found {len(all_combinations)} unique combinations in {timeframe}")
        logging.info(f"   Processed: {processed_windows} windows | Skipped: {skipped_windows}")

        # Log top symbols by combo count
        top_symbols = sorted(symbol_combo_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_symbols:
            logging.info(f"   Top symbols by combos: {top_symbols}")

        return all_combinations

    def calculate_combination_accuracy(self, combo_name: str, results: List[Dict],
                                      min_samples: int = 10) -> Optional[Dict]:
        """Calculate accuracy metrics for a signal combination"""
        if len(results) < min_samples:
            return None

        total = len(results)
        correct = sum(1 for r in results if r['predicted_correctly'])
        accuracy = (correct / total) * 100

        # Calculate profit metrics
        winning_moves = [abs(r['price_change_pct']) for r in results if r['predicted_correctly']]
        losing_moves = [abs(r['price_change_pct']) for r in results if not r['predicted_correctly']]

        avg_win = sum(winning_moves) / len(winning_moves) if winning_moves else 0
        avg_loss = sum(losing_moves) / len(losing_moves) if losing_moves else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else avg_win

        avg_price_change = sum(r['price_change_pct'] for r in results) / total

        return {
            'combo_name': combo_name,
            'total_samples': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'avg_price_change': avg_price_change,
            'profit_factor': profit_factor,
            'combo_size': len(combo_name.split('+'))
        }

    def analyze_timeframe_combinations(self, timeframe: str, min_samples: int = 10,
                                      min_combo_size: int = 2, max_combo_size: int = 5,
                                      max_combinations_per_symbol: int = 1000,
                                      strict_timing: bool = True) -> Dict:
        """
        OPTIMIZED: Analyze signal combinations with limits
        """
        logging.info(f"\n{'=' * 80}")
        logging.info(f"Analyzing combinations for {timeframe}")
        logging.info(f"  Max combos per symbol: {max_combinations_per_symbol}")
        logging.info(f"  Strict timing: {strict_timing}")
        logging.info(f"{'=' * 80}")

        # Find all combinations
        combinations_data = self.find_signal_combinations(
            timeframe, min_combo_size, max_combo_size,
            max_combinations_per_symbol, strict_timing
        )

        results = {
            'timeframe': timeframe,
            'total_combinations': len(combinations_data),
            'analyzed': 0,
            'skipped_insufficient_samples': 0,
            'combinations': []
        }

        # Analyze each combination
        for combo_name, combo_results in combinations_data.items():
            stats = self.calculate_combination_accuracy(combo_name, combo_results, min_samples)

            if stats:
                results['analyzed'] += 1
                results['combinations'].append(stats)

                # Save to database
                self.save_combination_result(stats, timeframe)
            else:
                results['skipped_insufficient_samples'] += 1

        # Sort by accuracy
        results['combinations'].sort(key=lambda x: x['accuracy'], reverse=True)

        logging.info(f"\n✅ {timeframe} Analysis Complete:")
        logging.info(f"   Total combinations found: {results['total_combinations']}")
        logging.info(f"   Analyzed (>={min_samples} samples): {results['analyzed']}")
        logging.info(f"   Skipped (insufficient data): {results['skipped_insufficient_samples']}")

        if results['combinations']:
            top = results['combinations'][0]
            logging.info(f"\n🏆 Best Combination:")
            logging.info(f"   {top['combo_name']}")
            logging.info(f"   Accuracy: {top['accuracy']:.2f}%")
            logging.info(f"   Samples: {top['total_samples']}")
            logging.info(f"   Profit Factor: {top['profit_factor']:.2f}")

        return results

    def save_combination_result(self, stats: Dict, timeframe: str):
        """Save combination analysis results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO tf_combos 
            (signal_name, timeframe, accuracy, signals_count, correct_predictions,
             avg_price_change, profit_factor, combo_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            stats['combo_name'],
            timeframe,
            stats['accuracy'],
            stats['total_samples'],
            stats['correct_predictions'],
            stats['avg_price_change'],
            stats['profit_factor'],
            stats['combo_size']
        ))

        conn.commit()
        conn.close()

    def analyze_all_timeframes(self, min_samples: int = 10, min_combo_size: int = 2,
                               max_combo_size: int = 5, timeframes: List[str] = None,
                               max_combinations_per_symbol: int = 1000,
                               strict_timing: bool = True) -> Dict:
        """Run analysis on all timeframes with optimization"""
        if timeframes is None:
            timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',
                         '6h', '8h', '12h', '1d', '3d', '1w']

        all_results = {}

        for tf in timeframes:
            try:
                results = self.analyze_timeframe_combinations(
                    tf, min_samples, min_combo_size, max_combo_size,
                    max_combinations_per_symbol, strict_timing
                )
                all_results[tf] = results
            except Exception as e:
                logging.error(f"Error analyzing {tf}: {e}")
                all_results[tf] = {'error': str(e)}

        return all_results

    # Keeping other methods unchanged...
    def get_top_combinations(self, limit: int = 20, min_accuracy: float = 0,
                            timeframe: Optional[str] = None,
                            min_samples: int = 0) -> List[Dict]:
        """Get top performing same-timeframe combinations"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = '''
            SELECT * FROM tf_combos
            WHERE accuracy >= ? AND signals_count >= ?
        '''
        params = [min_accuracy, min_samples]

        if timeframe:
            query += ' AND timeframe = ?'
            params.append(timeframe)

        query += ' ORDER BY accuracy DESC, profit_factor DESC LIMIT ?'
        params.append(limit)

        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return results

    # ========================================================================
    # CROSS-TIMEFRAME COMBINATION ANALYSIS
    # ========================================================================

    def analyze_cross_timeframe_combinations(
        self,
        timeframes: List[str],
        min_samples: int = 10,
        min_combo_size: int = 2,
        max_combo_size: int = 4,
        max_combinations_per_symbol: int = 500
    ) -> Dict:
        """
        Find signal combinations ACROSS different timeframes
        Example: RSI oversold on 1h + MACD bullish on 15m

        Args:
            timeframes: List of timeframes to analyze (e.g., ['15m', '1h', '4h'])
            min_samples: Minimum number of occurrences required
            min_combo_size: Minimum signals in combination (must be >= 2)
            max_combo_size: Maximum signals in combination
            max_combinations_per_symbol: Limit per symbol to prevent explosion
        """
        logging.info(f"\n{'=' * 80}")
        logging.info(f"CROSS-TIMEFRAME COMBINATION ANALYSIS")
        logging.info(f"{'=' * 80}")
        logging.info(f"Timeframes: {timeframes}")
        logging.info(f"Combo size: {min_combo_size}-{max_combo_size}")
        logging.info(f"Min samples: {min_samples}")
        logging.info(f"{'=' * 80}\n")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query to get all signals from selected timeframes
        placeholders = ','.join('?' * len(timeframes))
        query = f'''
            SELECT 
                ls.id,
                ls.symbol,
                ls.signal_name,
                ls.signal_type,
                ls.timeframe,
                ls.timestamp,
                sfc.predicted_correctly,
                sfc.price_change_pct
            FROM live_signals ls
            INNER JOIN signal_fact_checks sfc
                ON ls.signal_name = sfc.signal_name
                AND ls.timeframe = sfc.timeframe
                AND ls.timestamp = sfc.detected_at
            WHERE ls.timeframe IN ({placeholders})
            ORDER BY ls.symbol, ls.timestamp
        '''

        cursor.execute(query, timeframes)
        signals = [dict(row) for row in cursor.fetchall()]
        conn.close()

        logging.info(f"📥 Fetched {len(signals)} fact-checked signals across {len(timeframes)} timeframes")

        # Group signals by symbol and approximate time window
        # Use a broader window to capture signals across timeframes
        TIME_WINDOW_MINUTES = 60  # Signals within 1 hour considered "simultaneous"

        signal_groups = defaultdict(list)

        for signal in signals:
            timestamp = datetime.fromisoformat(signal['timestamp'])
            # Round to hour for grouping
            window_start = timestamp.replace(minute=0, second=0, microsecond=0)
            window_key = f"{signal['symbol']}_{window_start.isoformat()}"
            signal_groups[window_key].append(signal)

        logging.info(f"🗂️  Grouped into {len(signal_groups)} time windows")

        # Find cross-timeframe combinations
        all_combinations = defaultdict(list)
        symbol_combo_counts = defaultdict(int)
        processed_windows = 0
        skipped_windows = 0

        for window_key, window_signals in signal_groups.items():
            processed_windows += 1

            if len(window_signals) < min_combo_size:
                skipped_windows += 1
                continue

            symbol = window_signals[0]['symbol']

            # Check symbol limit
            if symbol_combo_counts[symbol] >= max_combinations_per_symbol:
                skipped_windows += 1
                continue

            # Create signal descriptors with timeframe info
            signal_descriptors = []
            for s in window_signals:
                descriptor = f"{s['signal_name']}@{s['timeframe']}"
                signal_descriptors.append({
                    'descriptor': descriptor,
                    'signal': s
                })

            # Skip if we don't have signals from multiple timeframes
            unique_timeframes = set(s['timeframe'] for s in window_signals)
            if len(unique_timeframes) < 2:
                skipped_windows += 1
                continue

            # Generate combinations
            for combo_size in range(min_combo_size, min(max_combo_size + 1, len(signal_descriptors) + 1)):
                for combo in combinations(signal_descriptors, combo_size):
                    # Check that this combo includes multiple timeframes
                    combo_timeframes = set(c['signal']['timeframe'] for c in combo)
                    if len(combo_timeframes) < 2:
                        continue  # Skip same-timeframe combos

                    # Create combo key
                    combo_key = '+'.join(sorted(c['descriptor'] for c in combo))

                    # Use first signal's result (they should all be from same time window)
                    result = {
                        'predicted_correctly': combo[0]['signal']['predicted_correctly'],
                        'price_change_pct': combo[0]['signal']['price_change_pct'],
                        'signal_types': [c['signal']['signal_type'] for c in combo],
                        'timeframes': sorted(combo_timeframes),
                        'symbol': symbol,
                        'timestamp': window_signals[0]['timestamp']
                    }

                    all_combinations[combo_key].append(result)
                    symbol_combo_counts[symbol] += 1

                    if symbol_combo_counts[symbol] >= max_combinations_per_symbol:
                        break

            # Progress logging
            if processed_windows % 500 == 0:
                logging.info(f"   Progress: {processed_windows}/{len(signal_groups)} windows | "
                           f"Combos: {len(all_combinations)} | Skipped: {skipped_windows}")

        logging.info(f"\n✅ Found {len(all_combinations)} cross-timeframe combinations")
        logging.info(f"   Processed: {processed_windows} windows | Skipped: {skipped_windows}")

        # Calculate accuracy for each combination
        results = {
            'timeframes': timeframes,
            'total_combinations': len(all_combinations),
            'analyzed': 0,
            'skipped_insufficient_samples': 0,
            'combinations': []
        }

        for combo_name, combo_results in all_combinations.items():
            if len(combo_results) < min_samples:
                results['skipped_insufficient_samples'] += 1
                continue

            stats = self._calculate_cross_tf_accuracy(combo_name, combo_results)

            if stats:
                results['analyzed'] += 1
                results['combinations'].append(stats)

                # Save to database
                self._save_cross_tf_combination(stats, timeframes)

        # Sort by accuracy
        results['combinations'].sort(key=lambda x: x['accuracy'], reverse=True)

        logging.info(f"\n✅ Cross-Timeframe Analysis Complete:")
        logging.info(f"   Total combinations found: {results['total_combinations']}")
        logging.info(f"   Analyzed (>={min_samples} samples): {results['analyzed']}")
        logging.info(f"   Skipped (insufficient data): {results['skipped_insufficient_samples']}")

        if results['combinations']:
            top = results['combinations'][0]
            logging.info(f"\n🏆 Best Cross-TF Combination:")
            logging.info(f"   {top['combo_signature']}")
            logging.info(f"   Accuracy: {top['accuracy']:.2f}%")
            logging.info(f"   Samples: {top['signals_count']}")
            logging.info(f"   Timeframes: {top['num_timeframes']}")

        return results

    def _calculate_cross_tf_accuracy(self, combo_name: str, results: List[Dict]) -> Optional[Dict]:
        """Calculate accuracy for cross-timeframe combination"""
        total = len(results)
        correct = sum(1 for r in results if r['predicted_correctly'])
        accuracy = (correct / total) * 100

        # Calculate profit metrics
        winning_moves = [abs(r['price_change_pct']) for r in results if r['predicted_correctly']]
        losing_moves = [abs(r['price_change_pct']) for r in results if not r['predicted_correctly']]

        avg_win = sum(winning_moves) / len(winning_moves) if winning_moves else 0
        avg_loss = sum(losing_moves) / len(losing_moves) if losing_moves else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else avg_win

        avg_price_change = sum(r['price_change_pct'] for r in results) / total

        # Extract signal names and timeframes
        signal_parts = combo_name.split('+')
        signal_names = []
        timeframes_involved = set()

        for part in signal_parts:
            signal, tf = part.rsplit('@', 1)
            signal_names.append(signal)
            timeframes_involved.add(tf)

        return {
            'combo_signature': combo_name,
            'signal_names': '+'.join(signal_names),
            'timeframes': sorted(timeframes_involved),
            'signals_count': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'avg_price_change': avg_price_change,
            'profit_factor': profit_factor,
            'combo_size': len(signal_parts),
            'num_timeframes': len(timeframes_involved)
        }

    def _save_cross_tf_combination(self, stats: Dict, timeframes: List[str]):
        """Save cross-timeframe combination to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO cross_tf_combos 
            (combo_signature, timeframes, signal_names, accuracy, signals_count, 
             correct_predictions, avg_price_change, profit_factor, combo_size, num_timeframes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            stats['combo_signature'],
            ','.join(stats['timeframes']),
            stats['signal_names'],
            stats['accuracy'],
            stats['signals_count'],
            stats['correct_predictions'],
            stats['avg_price_change'],
            stats['profit_factor'],
            stats['combo_size'],
            stats['num_timeframes']
        ))

        conn.commit()
        conn.close()

    def get_top_cross_tf_combinations(
        self,
        limit: int = 20,
        min_accuracy: float = 0,
        min_samples: int = 0,
        min_timeframes: int = 2
    ) -> List[Dict]:
        """Get top performing cross-timeframe combinations"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM cross_tf_combos
            WHERE accuracy >= ? 
              AND signals_count >= ?
              AND num_timeframes >= ?
            ORDER BY accuracy DESC, profit_factor DESC 
            LIMIT ?
        ''', (min_accuracy, min_samples, min_timeframes, limit))

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return results

    def generate_report(self, min_accuracy: float = 60.0) -> Dict:
        """Generate comprehensive report of best combinations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT 
                COUNT(*) as total_combos,
                AVG(accuracy) as avg_accuracy,
                MAX(accuracy) as max_accuracy,
                SUM(signals_count) as total_samples
            FROM tf_combos
        ''')

        overall = cursor.fetchone()

        cursor.execute('''
            SELECT 
                combo_size,
                COUNT(*) as count,
                AVG(accuracy) as avg_accuracy,
                MAX(accuracy) as max_accuracy
            FROM tf_combos
            GROUP BY combo_size
            ORDER BY combo_size
        ''')

        by_size = cursor.fetchall()

        cursor.execute('''
            SELECT 
                timeframe,
                COUNT(*) as count,
                AVG(accuracy) as avg_accuracy,
                MAX(accuracy) as max_accuracy,
                signal_name
            FROM tf_combos
            WHERE accuracy = (SELECT MAX(accuracy) FROM tf_combos t2 WHERE t2.timeframe = tf_combos.timeframe)
            GROUP BY timeframe
            ORDER BY max_accuracy DESC
        ''')

        by_timeframe = cursor.fetchall()

        cursor.execute('''
            SELECT * FROM tf_combos
            WHERE accuracy >= ?
            ORDER BY accuracy DESC, profit_factor DESC
            LIMIT 20
        ''', (min_accuracy,))

        top_performers = cursor.fetchall()

        conn.close()

        return {
            'overall': {
                'total_combinations': overall[0],
                'avg_accuracy': overall[1],
                'max_accuracy': overall[2],
                'total_samples': overall[3]
            },
            'by_combo_size': [
                {
                    'combo_size': row[0],
                    'count': row[1],
                    'avg_accuracy': row[2],
                    'max_accuracy': row[3]
                }
                for row in by_size
            ],
            'by_timeframe': [
                {
                    'timeframe': row[0],
                    'count': row[1],
                    'avg_accuracy': row[2],
                    'max_accuracy': row[3],
                    'best_combo': row[4]
                }
                for row in by_timeframe
            ],
            'top_performers': [
                {
                    'signal_name': row[1],
                    'timeframe': row[2],
                    'accuracy': row[3],
                    'signals_count': row[4],
                    'profit_factor': row[7]
                }
                for row in top_performers
            ]
        }

    def compare_combo_to_individual(self, combo_name: str, timeframe: str) -> Dict:
        """Compare combination accuracy to individual signal accuracies"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get combo stats
        cursor.execute('''
            SELECT * FROM tf_combos
            WHERE signal_name = ? AND timeframe = ?
        ''', (combo_name, timeframe))

        combo_data = cursor.fetchone()
        if not combo_data:
            conn.close()
            return {'error': 'Combination not found'}

        # Get individual signal stats
        signals = combo_name.split('+')
        individual_stats = []

        for signal in signals:
            cursor.execute('''
                SELECT signal_accuracy, sample_size
                FROM signals
                WHERE signal_name = ? AND timeframe = ?
            ''', (signal, timeframe))

            signal_data = cursor.fetchone()
            if signal_data:
                individual_stats.append({
                    'signal_name': signal,
                    'accuracy': signal_data['signal_accuracy'],
                    'samples': signal_data['sample_size']
                })

        conn.close()

        return {
            'combo': dict(combo_data),
            'individual_signals': individual_stats,
            'accuracy_improvement': combo_data['accuracy'] - (
                sum(s['accuracy'] for s in individual_stats) / len(individual_stats)
                if individual_stats else 0
            )
        }

    def get_combination_details(self, combo_name: str, timeframe: str) -> Optional[Dict]:
        """Get detailed information about a specific combination"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM tf_combos
            WHERE signal_name = ? AND timeframe = ?
        ''', (combo_name, timeframe))

        result = cursor.fetchone()
        conn.close()

        if result:
            return dict(result)
        return None

    def compare_cross_tf_combo_to_individual(self, combo_signature: str) -> Dict:
        """Compare cross-timeframe combination to individual signal-timeframe pairs"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get combo stats
        cursor.execute('''
            SELECT * FROM cross_tf_combos
            WHERE combo_signature = ?
        ''', (combo_signature,))

        combo_data = cursor.fetchone()
        if not combo_data:
            conn.close()
            return {'error': 'Cross-timeframe combination not found'}

        # Parse signal@timeframe parts
        signal_parts = combo_signature.split('+')
        individual_stats = []

        for part in signal_parts:
            signal_name, timeframe = part.rsplit('@', 1)

            cursor.execute('''
                SELECT signal_accuracy, sample_size
                FROM signals
                WHERE signal_name = ? AND timeframe = ?
            ''', (signal_name, timeframe))

            signal_data = cursor.fetchone()
            if signal_data:
                individual_stats.append({
                    'signal_name': signal_name,
                    'timeframe': timeframe,
                    'accuracy': signal_data['signal_accuracy'],
                    'samples': signal_data['sample_size']
                })

        conn.close()

        avg_individual_accuracy = (
            sum(s['accuracy'] for s in individual_stats) / len(individual_stats)
            if individual_stats else 0
        )

        return {
            'combo': dict(combo_data),
            'individual_signals': individual_stats,
            'accuracy_improvement': combo_data['accuracy'] - avg_individual_accuracy
        }

    def get_cross_tf_combination_details(self, combo_signature: str) -> Optional[Dict]:
        """Get detailed information about a specific cross-timeframe combination"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM cross_tf_combos
            WHERE combo_signature = ?
        ''', (combo_signature,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return dict(result)
        return None


if __name__ == "__main__":
    analyzer = SignalCombinationAnalyzer()

    print("\n" + "=" * 80)
    print("SIGNAL COMBINATION ANALYZER - SQL JOIN FIXED")
    print("=" * 80)

    # Test the fix
    print("\n🧪 Testing SQL JOIN fix...")

    conn = sqlite3.connect('crypto_signals.db')
    cursor = conn.cursor()

    # Check table sizes
    cursor.execute('SELECT COUNT(*) FROM live_signals WHERE timeframe = "2h"')
    live_count = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM signal_fact_checks WHERE timeframe = "2h"')
    fact_count = cursor.fetchone()[0]

    conn.close()

    print(f"\n📊 Database stats for 2h timeframe:")
    print(f"   live_signals: {live_count:,} rows")
    print(f"   signal_fact_checks: {fact_count:,} rows")
    print(f"   Expected joined: ~{min(live_count, fact_count):,} rows (not {live_count * fact_count:,}!)")

    print("\n🔧 Running fixed analysis...")
    results = analyzer.analyze_timeframe_combinations(
        timeframe='2h',
        min_samples=20,
        min_combo_size=2,
        max_combo_size=4,
        max_combinations_per_symbol=500,
        strict_timing=True
    )

    print(f"\n✅ Analysis complete!")
    print(f"   Signals fetched: Should be ~{min(live_count, fact_count):,}")
    print(f"   Total combinations: {results['total_combinations']}")
    print(f"   Analyzed: {results['analyzed']}")