"""
Signal Combination Analyzer - Extended with Cross-Timeframe Analysis
Analyzes accuracy of signal combinations:
1. Same timeframe (original functionality)
2. Cross-timeframe (NEW: signals from different timeframes)
Uses fact-check results instead of live API calls for performance
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
    Analyzes signal combinations to find which signals work better together
    Supports both same-timeframe and cross-timeframe combinations
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
    # ORIGINAL SAME-TIMEFRAME FUNCTIONALITY
    # ========================================================================

    def find_signal_combinations(self, timeframe: str, min_combo_size: int = 2,
                                 max_combo_size: int = 5) -> Dict[str, List[Dict]]:
        """
        Find all signal combinations that appeared together in same timeframe
        Groups by symbol + time window
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all signals for this timeframe with fact-check results
        cursor.execute('''
            SELECT 
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

        logging.info(f"Found {len(signals)} fact-checked signals for {timeframe}")

        # Group signals by symbol and time window
        window_seconds = self.get_timeframe_window_seconds(timeframe)
        signal_groups = defaultdict(list)

        for signal in signals:
            timestamp = datetime.fromisoformat(signal['timestamp'])
            # Round timestamp to nearest window
            window_start = timestamp.replace(second=0, microsecond=0)
            window_key = f"{signal['symbol']}_{window_start.isoformat()}"
            signal_groups[window_key].append(signal)

        # Find combinations in each group
        all_combinations = defaultdict(list)

        for window_key, window_signals in signal_groups.items():
            if len(window_signals) < min_combo_size:
                continue

            # Get all signal names in this window
            signal_names = [s['signal_name'] for s in window_signals]

            # Generate combinations of different sizes
            for combo_size in range(min_combo_size, min(max_combo_size + 1, len(signal_names) + 1)):
                for combo in combinations(sorted(set(signal_names)), combo_size):
                    combo_key = '+'.join(sorted(combo))

                    # Check if all signals in combo were present
                    combo_signals = [s for s in window_signals if s['signal_name'] in combo]

                    if len(set(s['signal_name'] for s in combo_signals)) == len(combo):
                        # All signals in combo are present
                        # Use the result of the first signal (or we could use majority vote)
                        result = {
                            'predicted_correctly': combo_signals[0]['predicted_correctly'],
                            'price_change_pct': combo_signals[0]['price_change_pct'],
                            'signal_types': [s['signal_type'] for s in combo_signals]
                        }
                        all_combinations[combo_key].append(result)

        logging.info(f"Found {len(all_combinations)} unique combinations in {timeframe}")
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
                                      min_combo_size: int = 2, max_combo_size: int = 5) -> Dict:
        """
        Analyze all signal combinations for a specific timeframe
        """
        logging.info(f"\n{'=' * 80}")
        logging.info(f"Analyzing combinations for {timeframe}")
        logging.info(f"{'=' * 80}")

        # Find all combinations
        combinations_data = self.find_signal_combinations(
            timeframe, min_combo_size, max_combo_size
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
                               max_combo_size: int = 5, timeframes: List[str] = None) -> Dict:
        """Run analysis on all timeframes"""
        if timeframes is None:
            timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',
                         '6h', '8h', '12h', '1d', '3d', '1w']

        all_results = {}

        for tf in timeframes:
            try:
                results = self.analyze_timeframe_combinations(
                    tf, min_samples, min_combo_size, max_combo_size
                )
                all_results[tf] = results
            except Exception as e:
                logging.error(f"Error analyzing {tf}: {e}")
                all_results[tf] = {'error': str(e)}

        return all_results

    # ========================================================================
    # NEW: CROSS-TIMEFRAME FUNCTIONALITY
    # ========================================================================

    def find_cross_timeframe_combinations(
      self,
      timeframes: List[str],
      min_combo_size: int = 2,
      max_combo_size: int = 4,
      time_tolerance_minutes: int = 5,
      skip_existing: bool = True
    ) -> Dict[str, List[Dict]]:
      """
      Find signal combinations across different timeframes

      Args:
          timeframes: List of timeframes to analyze (e.g., ['5m', '15m', '1h'])
          min_combo_size: Minimum number of signals in combination
          max_combo_size: Maximum number of signals in combination
          time_tolerance_minutes: How close signals must be in time to be considered together
          skip_existing: If True, skip combinations already in cross_tf_combos table

      Returns:
          Dictionary mapping combination signatures to their results
      """
      import time
      start_time = time.time()

      logging.info("\n" + "=" * 80)
      logging.info("🔍 CROSS-TIMEFRAME COMBINATION FINDER")
      logging.info("=" * 80)
      logging.info(f"Timeframes: {', '.join(timeframes)}")
      logging.info(f"Combo size: {min_combo_size}-{max_combo_size}")
      logging.info(f"Time tolerance: {time_tolerance_minutes} minutes")
      logging.info(f"Skip existing: {skip_existing}")
      logging.info("=" * 80 + "\n")

      conn = sqlite3.connect(self.db_path)
      conn.row_factory = sqlite3.Row
      cursor = conn.cursor()

      # ========================================================================
      # STEP 0: Load existing combinations to skip (if enabled)
      # ========================================================================
      existing_combo_signatures = set()
      if skip_existing:
        logging.info("🔍 STEP 0/4: Loading existing combinations from database...")

        cursor.execute('''
                SELECT DISTINCT combo_signature 
                FROM cross_tf_combos
            ''')

        existing_combo_signatures = {row['combo_signature'] for row in cursor.fetchall()}

        logging.info(f"   ✓ Found {len(existing_combo_signatures):,} existing combinations to skip")
        logging.info(f"   ⏱️  Time elapsed: {time.time() - start_time:.2f}s\n")

      # ========================================================================
      # STEP 1: Fetch signals from all timeframes
      # ========================================================================
      logging.info("📥 STEP 1/4: Fetching signals from database...")
      timeframe_signals = {}
      total_signals = 0

      for idx, tf in enumerate(timeframes, 1):
        logging.info(f"   [{idx}/{len(timeframes)}] Fetching {tf} signals...")

        cursor.execute('''
                SELECT 
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
                WHERE ls.timeframe = ?
                ORDER BY ls.symbol, ls.timestamp
            ''', (tf,))

        signals = [dict(row) for row in cursor.fetchall()]
        timeframe_signals[tf] = signals
        total_signals += len(signals)

        logging.info(f"      ✓ Found {len(signals):,} signals for {tf}")

      conn.close()

      logging.info(f"\n   ✅ Total signals fetched: {total_signals:,}")
      logging.info(f"   ⏱️  Time elapsed: {time.time() - start_time:.2f}s\n")

      # ========================================================================
      # STEP 2: Organize signals by symbol
      # ========================================================================
      logging.info("🗂️  STEP 2/4: Organizing signals by symbol...")

      symbol_signals = defaultdict(lambda: defaultdict(list))
      for tf, signals in timeframe_signals.items():
        for signal in signals:
          symbol_signals[signal['symbol']][tf].append(signal)

      total_symbols = len(symbol_signals)
      logging.info(f"   ✓ Organized {total_symbols:,} unique symbols")

      # Count symbols with multi-timeframe signals
      multi_tf_symbols = [
        symbol for symbol, tf_sigs in symbol_signals.items()
        if len([tf for tf in timeframes if tf_sigs.get(tf)]) >= 2
      ]

      logging.info(f"   ✓ {len(multi_tf_symbols):,} symbols have signals in multiple timeframes")
      logging.info(f"   ⏱️  Time elapsed: {time.time() - start_time:.2f}s\n")

      # ========================================================================
      # STEP 3: Find cross-timeframe matches
      # ========================================================================
      logging.info("🔗 STEP 3/4: Finding cross-timeframe combinations...")
      all_combinations = defaultdict(list)
      time_tolerance = timedelta(minutes=time_tolerance_minutes)

      processed_symbols = 0
      symbols_with_combos = 0
      skipped_combos = 0
      new_combos = 0
      last_log_time = time.time()
      log_interval = 5  # Log every 5 seconds

      for symbol, tf_signals in symbol_signals.items():
        processed_symbols += 1

        # Log progress every 5 seconds or every 100 symbols
        current_time = time.time()
        if current_time - last_log_time >= log_interval or processed_symbols % 100 == 0:
          progress_pct = (processed_symbols / total_symbols) * 100
          elapsed = current_time - start_time
          rate = processed_symbols / elapsed if elapsed > 0 else 0
          eta = (total_symbols - processed_symbols) / rate if rate > 0 else 0

          logging.info(
            f"   Progress: {processed_symbols:,}/{total_symbols:,} symbols "
            f"({progress_pct:.1f}%) | "
            f"Rate: {rate:.1f} symbols/s | "
            f"ETA: {eta:.0f}s | "
            f"New: {new_combos:,} | Skipped: {skipped_combos:,}"
          )
          last_log_time = current_time

        # Only proceed if we have signals in multiple timeframes
        available_tfs = [tf for tf in timeframes if tf_signals.get(tf)]
        if len(available_tfs) < 2:
          continue

        # Track if this symbol produced any combos
        combos_before = len(all_combinations)

        # Find temporal matches across timeframes (with skip check)
        skipped_count = self._find_temporal_cross_tf_matches(
          symbol, tf_signals, available_tfs, time_tolerance,
          min_combo_size, max_combo_size, all_combinations,
          existing_combo_signatures  # Pass existing signatures
        )

        skipped_combos += skipped_count

        # Count if new combos were added
        new_count = len(all_combinations) - combos_before
        if new_count > 0:
          symbols_with_combos += 1
          new_combos += new_count

      logging.info(f"\n   ✅ Processed all {total_symbols:,} symbols")
      logging.info(f"   ✓ {symbols_with_combos:,} symbols produced valid combinations")
      logging.info(f"   ✓ {new_combos:,} NEW cross-timeframe combinations found")
      logging.info(f"   ⏭️  {skipped_combos:,} combinations skipped (already exist)")
      logging.info(f"   ⏱️  Time elapsed: {time.time() - start_time:.2f}s\n")

      # ========================================================================
      # STEP 4: Summary statistics
      # ========================================================================
      logging.info("📊 STEP 4/4: Generating statistics...")

      # Count combo sizes
      combo_size_distribution = defaultdict(int)
      samples_per_combo = []

      for combo_name, results in all_combinations.items():
        # Count signals in combo (split by ' + ')
        combo_size = len(combo_name.split(' + '))
        combo_size_distribution[combo_size] += 1
        samples_per_combo.append(len(results))

      if combo_size_distribution:
        logging.info(f"\n   Combination size distribution:")
        for size in sorted(combo_size_distribution.keys()):
          count = combo_size_distribution[size]
          pct = (count / len(all_combinations)) * 100 if all_combinations else 0
          logging.info(f"      {size} signals: {count:,} combinations ({pct:.1f}%)")

      if samples_per_combo:
        avg_samples = sum(samples_per_combo) / len(samples_per_combo)
        min_samples = min(samples_per_combo)
        max_samples = max(samples_per_combo)
        total_samples = sum(samples_per_combo)

        logging.info(f"\n   Sample statistics:")
        logging.info(f"      Total samples: {total_samples:,}")
        logging.info(f"      Avg samples per combo: {avg_samples:.1f}")
        logging.info(f"      Min samples: {min_samples}")
        logging.info(f"      Max samples: {max_samples}")

      # ========================================================================
      # Final summary
      # ========================================================================
      total_time = time.time() - start_time

      logging.info("\n" + "=" * 80)
      logging.info("✅ CROSS-TIMEFRAME COMBINATION FINDER COMPLETE")
      logging.info("=" * 80)
      logging.info(f"Total signals processed: {total_signals:,}")
      logging.info(f"Total symbols analyzed: {total_symbols:,}")
      logging.info(f"Symbols with multi-TF signals: {len(multi_tf_symbols):,}")
      logging.info(f"Symbols producing combinations: {symbols_with_combos:,}")
      logging.info(f"NEW combinations found: {new_combos:,}")
      logging.info(f"Existing combinations skipped: {skipped_combos:,}")
      logging.info(f"Total execution time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")
      if total_symbols > 0:
        logging.info(f"Average time per symbol: {(total_time / total_symbols) * 1000:.2f}ms")
      logging.info("=" * 80 + "\n")

      return all_combinations

    def _find_temporal_cross_tf_matches(
      self,
      symbol: str,
      tf_signals: Dict[str, List[Dict]],
      available_tfs: List[str],
      time_tolerance: timedelta,
      min_combo_size: int,
      max_combo_size: int,
      all_combinations: Dict[str, List[Dict]],
      existing_combo_signatures: set = None  # NEW parameter
    ) -> int:
      """
      Find signals that occur close together in time across different timeframes

      Returns:
          Number of combinations skipped
      """
      if existing_combo_signatures is None:
        existing_combo_signatures = set()

      skipped_count = 0

      # Get all signals sorted by time
      all_signals = []
      for tf in available_tfs:
        for signal in tf_signals[tf]:
          signal_copy = signal.copy()
          signal_copy['parsed_timestamp'] = datetime.fromisoformat(signal['timestamp'])
          all_signals.append(signal_copy)

      all_signals.sort(key=lambda x: x['parsed_timestamp'])

      # Use sliding window to find temporally close signals
      for i, anchor_signal in enumerate(all_signals):
        anchor_time = anchor_signal['parsed_timestamp']
        window_signals = [anchor_signal]

        # Look forward in time within tolerance
        for j in range(i + 1, len(all_signals)):
          candidate = all_signals[j]
          time_diff = candidate['parsed_timestamp'] - anchor_time

          if time_diff > time_tolerance:
            break

          # Must be from different timeframe
          if candidate['timeframe'] != anchor_signal['timeframe']:
            window_signals.append(candidate)

        # Generate combinations from this window
        if len(window_signals) >= min_combo_size:
          skipped = self._generate_cross_tf_combinations(
            window_signals, min_combo_size, max_combo_size,
            all_combinations, existing_combo_signatures  # Pass existing signatures
          )
          skipped_count += skipped

      return skipped_count

    def _generate_cross_tf_combinations(
      self,
      window_signals: List[Dict],
      min_combo_size: int,
      max_combo_size: int,
      all_combinations: Dict[str, List[Dict]],
      existing_combo_signatures: set = None  # NEW parameter
    ) -> int:
      """
      Generate all valid cross-timeframe combinations from a window of signals

      Returns:
          Number of combinations skipped
      """
      if existing_combo_signatures is None:
        existing_combo_signatures = set()

      skipped_count = 0

      # Ensure signals are from different timeframes
      for combo_size in range(min_combo_size, min(max_combo_size + 1, len(window_signals) + 1)):
        for combo in combinations(window_signals, combo_size):
          # Verify all signals are from different timeframes
          timeframes = [s['timeframe'] for s in combo]
          if len(set(timeframes)) != len(timeframes):
            continue  # Skip if duplicate timeframes

          # Create combination signature
          combo_parts = []
          for signal in sorted(combo, key=lambda x: x['timeframe']):
            combo_parts.append(f"{signal['signal_name']}[{signal['timeframe']}]")

          combo_signature = ' + '.join(combo_parts)

          # NEW: Skip if this combination already exists
          if combo_signature in existing_combo_signatures:
            skipped_count += 1
            continue

          # Use the prediction result from the shortest timeframe signal
          # (as it's typically the trigger)
          shortest_tf_signal = min(combo, key=lambda x: self.get_timeframe_window_seconds(x['timeframe']))

          result = {
            'predicted_correctly': shortest_tf_signal['predicted_correctly'],
            'price_change_pct': shortest_tf_signal['price_change_pct'],
            'signal_names': [s['signal_name'] for s in combo],
            'timeframes': timeframes,
            'signal_types': [s['signal_type'] for s in combo]
          }

          all_combinations[combo_signature].append(result)

      return skipped_count
    def calculate_cross_tf_accuracy(
        self,
        combo_signature: str,
        results: List[Dict],
        min_samples: int = 10
    ) -> Optional[Dict]:
        """
        Calculate accuracy metrics for a cross-timeframe combination
        """
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

        # Extract timeframes and signal names
        timeframes = sorted(set(results[0]['timeframes']))
        signal_names = results[0]['signal_names']

        return {
            'combo_signature': combo_signature,
            'timeframes': timeframes,
            'signal_names': signal_names,
            'total_samples': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'avg_price_change': avg_price_change,
            'profit_factor': profit_factor,
            'combo_size': len(signal_names),
            'num_timeframes': len(timeframes)
        }

    def save_cross_tf_combination(self, stats: Dict):
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
            ','.join(stats['signal_names']),
            stats['accuracy'],
            stats['total_samples'],
            stats['correct_predictions'],
            stats['avg_price_change'],
            stats['profit_factor'],
            stats['combo_size'],
            stats['num_timeframes']
        ))

        conn.commit()
        conn.close()

    def analyze_cross_timeframe_combinations(
        self,
        timeframes: List[str],
        min_samples: int = 10,
        min_combo_size: int = 2,
        max_combo_size: int = 4,
        time_tolerance_minutes: int = 5
    ) -> Dict:
        """
        Analyze all cross-timeframe signal combinations

        Args:
            timeframes: List of timeframes to analyze together
            min_samples: Minimum occurrences needed for analysis
            min_combo_size: Minimum signals in combination
            max_combo_size: Maximum signals in combination
            time_tolerance_minutes: Time window for matching signals
        """
        logging.info(f"\n{'=' * 80}")
        logging.info(f"Analyzing CROSS-TIMEFRAME combinations")
        logging.info(f"Timeframes: {', '.join(timeframes)}")
        logging.info(f"Time tolerance: {time_tolerance_minutes} minutes")
        logging.info(f"{'=' * 80}")

        # Find all cross-timeframe combinations
        combinations_data = self.find_cross_timeframe_combinations(
            timeframes, min_combo_size, max_combo_size, time_tolerance_minutes
        )

        results = {
            'timeframes': timeframes,
            'total_combinations': len(combinations_data),
            'analyzed': 0,
            'skipped_insufficient_samples': 0,
            'combinations': []
        }

        # Analyze each combination
        for combo_sig, combo_results in combinations_data.items():
            stats = self.calculate_cross_tf_accuracy(combo_sig, combo_results, min_samples)

            if stats:
                results['analyzed'] += 1
                results['combinations'].append(stats)

                # Save to database
                self.save_cross_tf_combination(stats)
            else:
                results['skipped_insufficient_samples'] += 1

        # Sort by accuracy
        results['combinations'].sort(key=lambda x: x['accuracy'], reverse=True)

        logging.info(f"\n✅ Cross-Timeframe Analysis Complete:")
        logging.info(f"   Total combinations found: {results['total_combinations']}")
        logging.info(f"   Analyzed (>={min_samples} samples): {results['analyzed']}")
        logging.info(f"   Skipped (insufficient data): {results['skipped_insufficient_samples']}")

        if results['combinations']:
            top = results['combinations'][0]
            logging.info(f"\n🏆 Best Cross-Timeframe Combination:")
            logging.info(f"   {top['combo_signature']}")
            logging.info(f"   Accuracy: {top['accuracy']:.2f}%")
            logging.info(f"   Samples: {top['total_samples']}")
            logging.info(f"   Profit Factor: {top['profit_factor']:.2f}")

        return results

    def analyze_all_cross_tf_patterns(
        self,
        min_samples: int = 10,
        min_combo_size: int = 2,
        max_combo_size: int = 3,
        time_tolerance_minutes: int = 5
    ) -> Dict:
        """
        Analyze common cross-timeframe patterns
        Tries various timeframe combinations
        """
        # Common timeframe patterns to analyze
        tf_patterns = [
            ['5m', '15m', '1h'],      # Short-term trend confirmation
            ['15m', '1h', '4h'],      # Medium-term confirmation
            ['1h', '4h', '1d'],       # Long-term trend alignment
            ['5m', '1h'],             # Quick scalp with hourly confirmation
            ['15m', '4h'],            # Swing setup
            ['1h', '1d'],             # Position setup
        ]

        all_results = {}

        for tf_pattern in tf_patterns:
            pattern_name = '-'.join(tf_pattern)
            logging.info(f"\n{'*' * 80}")
            logging.info(f"Analyzing pattern: {pattern_name}")
            logging.info(f"{'*' * 80}")

            try:
                results = self.analyze_cross_timeframe_combinations(
                    tf_pattern,
                    min_samples,
                    min_combo_size,
                    max_combo_size,
                    time_tolerance_minutes
                )
                all_results[pattern_name] = results
            except Exception as e:
                logging.error(f"Error analyzing {pattern_name}: {e}")
                all_results[pattern_name] = {'error': str(e)}

        return all_results

    # ========================================================================
    # QUERY AND REPORTING METHODS
    # ========================================================================

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

    def get_top_cross_tf_combinations(
        self,
        limit: int = 20,
        min_accuracy: float = 0,
        min_samples: int = 0,
        num_timeframes: Optional[int] = None
    ) -> List[Dict]:
        """Get top performing cross-timeframe combinations"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = '''
            SELECT * FROM cross_tf_combos
            WHERE accuracy >= ? AND signals_count >= ?
        '''
        params = [min_accuracy, min_samples]

        if num_timeframes:
            query += ' AND num_timeframes = ?'
            params.append(num_timeframes)

        query += ' ORDER BY accuracy DESC, profit_factor DESC LIMIT ?'
        params.append(limit)

        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return results

    def get_combination_details(self, combo_name: str, timeframe: str) -> Optional[Dict]:
        """Get detailed stats for a specific same-timeframe combination"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM tf_combos
            WHERE signal_name = ? AND timeframe = ?
        ''', (combo_name, timeframe))

        result = cursor.fetchone()
        conn.close()

        return dict(result) if result else None

    def get_cross_tf_combination_details(self, combo_signature: str) -> Optional[Dict]:
        """Get detailed stats for a specific cross-timeframe combination"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM cross_tf_combos
            WHERE combo_signature = ?
        ''', (combo_signature,))

        result = cursor.fetchone()
        conn.close()

        return dict(result) if result else None

    def compare_combo_to_individual(self, combo_name: str, timeframe: str) -> Dict:
        """
        Compare combination accuracy to individual signal accuracies
        Shows improvement/degradation from combining signals
        """
        signals = combo_name.split('+')

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get combo accuracy
        cursor.execute('''
            SELECT accuracy, signals_count FROM tf_combos
            WHERE signal_name = ? AND timeframe = ?
        ''', (combo_name, timeframe))

        combo_result = cursor.fetchone()

        if not combo_result:
            conn.close()
            return {'error': 'Combination not found'}

        combo_accuracy, combo_count = combo_result

        # Get individual signal accuracies
        individual_accuracies = []
        for signal in signals:
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN predicted_correctly = 1 THEN 1 ELSE 0 END) as correct
                FROM signal_fact_checks
                WHERE signal_name = ? AND timeframe = ?
            ''', (signal, timeframe))

            result = cursor.fetchone()
            if result and result[0] > 0:
                accuracy = (result[1] / result[0]) * 100
                individual_accuracies.append({
                    'signal': signal,
                    'accuracy': accuracy,
                    'sample_count': result[0]
                })

        conn.close()

        # Calculate improvement
        avg_individual = sum(s['accuracy'] for s in individual_accuracies) / len(individual_accuracies) if individual_accuracies else 0
        improvement = combo_accuracy - avg_individual

        return {
            'combo_name': combo_name,
            'timeframe': timeframe,
            'combo_accuracy': combo_accuracy,
            'combo_sample_count': combo_count,
            'individual_signals': individual_accuracies,
            'avg_individual_accuracy': avg_individual,
            'accuracy_improvement': improvement,
            'improved': improvement > 0
        }

    def generate_comprehensive_report(self, min_accuracy: float = 60.0) -> Dict:
        """Generate comprehensive report of both same-TF and cross-TF combinations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Same-timeframe stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total_combos,
                AVG(accuracy) as avg_accuracy,
                MAX(accuracy) as max_accuracy,
                SUM(signals_count) as total_samples
            FROM tf_combos
        ''')
        same_tf_overall = cursor.fetchone()

        # Cross-timeframe stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total_combos,
                AVG(accuracy) as avg_accuracy,
                MAX(accuracy) as max_accuracy,
                SUM(signals_count) as total_samples
            FROM cross_tf_combos
        ''')
        cross_tf_overall = cursor.fetchone()

        # Top same-TF performers
        cursor.execute('''
            SELECT * FROM tf_combos
            WHERE accuracy >= ?
            ORDER BY accuracy DESC, profit_factor DESC
            LIMIT 10
        ''', (min_accuracy,))
        top_same_tf = cursor.fetchall()

        # Top cross-TF performers
        cursor.execute('''
            SELECT * FROM cross_tf_combos
            WHERE accuracy >= ?
            ORDER BY accuracy DESC, profit_factor DESC
            LIMIT 10
        ''', (min_accuracy,))
        top_cross_tf = cursor.fetchall()

        conn.close()

        return {
            'same_timeframe': {
                'total_combinations': same_tf_overall[0],
                'avg_accuracy': same_tf_overall[1],
                'max_accuracy': same_tf_overall[2],
                'total_samples': same_tf_overall[3],
                'top_performers': [
                    {
                        'signal_name': row[1],
                        'timeframe': row[2],
                        'accuracy': row[3],
                        'signals_count': row[4],
                        'profit_factor': row[7]
                    }
                    for row in top_same_tf
                ]
            },
            'cross_timeframe': {
                'total_combinations': cross_tf_overall[0],
                'avg_accuracy': cross_tf_overall[1],
                'max_accuracy': cross_tf_overall[2],
                'total_samples': cross_tf_overall[3],
                'top_performers': [
                    {
                        'combo_signature': row[1],
                        'timeframes': row[2],
                        'signal_names': row[3],
                        'accuracy': row[4],
                        'signals_count': row[5],
                        'profit_factor': row[8]
                    }
                    for row in top_cross_tf
                ]
            }
        }

    def generate_report(self, min_accuracy: float = 60.0) -> Dict:
        """Generate comprehensive report of best combinations (same-TF only for compatibility)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Overall stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total_combos,
                AVG(accuracy) as avg_accuracy,
                MAX(accuracy) as max_accuracy,
                SUM(signals_count) as total_samples
            FROM tf_combos
        ''')

        overall = cursor.fetchone()

        # Best by combo size
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

        # Best by timeframe
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

        # Top performers
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


if __name__ == "__main__":
    analyzer = SignalCombinationAnalyzer()

    print("\n" + "=" * 80)
    print("SIGNAL COMBINATION ANALYZER - EXTENDED")
    print("Same-Timeframe + Cross-Timeframe Analysis")
    print("=" * 80)

    # Example 1: Analyze same-timeframe combinations (original functionality)
    print("\n" + "=" * 80)
    print("PART 1: SAME-TIMEFRAME ANALYSIS")
    print("=" * 80)

    same_tf_results = analyzer.analyze_timeframe_combinations(
        timeframe='1h',
        min_samples=20,
        min_combo_size=2,
        max_combo_size=4
    )

    # Example 2: Analyze cross-timeframe combinations (NEW)
    print("\n" + "=" * 80)
    print("PART 2: CROSS-TIMEFRAME ANALYSIS")
    print("=" * 80)

    cross_tf_results = analyzer.analyze_cross_timeframe_combinations(
        timeframes=['5m', '15m', '1h'],
        min_samples=15,
        min_combo_size=2,
        max_combo_size=3,
        time_tolerance_minutes=5
    )

    # Example 3: Analyze multiple cross-TF patterns
    print("\n" + "=" * 80)
    print("PART 3: ANALYZING ALL CROSS-TF PATTERNS")
    print("=" * 80)

    all_patterns = analyzer.analyze_all_cross_tf_patterns(
        min_samples=15,
        min_combo_size=2,
        max_combo_size=3,
        time_tolerance_minutes=5
    )

    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE REPORT")
    print("=" * 80)

    report = analyzer.generate_comprehensive_report(min_accuracy=55.0)

    print("\n📊 SAME-TIMEFRAME COMBINATIONS:")
    print(f"Total combinations: {report['same_timeframe']['total_combinations']}")
    print(f"Average accuracy: {report['same_timeframe']['avg_accuracy']:.2f}%")
    print(f"Maximum accuracy: {report['same_timeframe']['max_accuracy']:.2f}%")

    print("\n🔄 CROSS-TIMEFRAME COMBINATIONS:")
    print(f"Total combinations: {report['cross_timeframe']['total_combinations']}")
    print(f"Average accuracy: {report['cross_timeframe']['avg_accuracy']:.2f}%")
    print(f"Maximum accuracy: {report['cross_timeframe']['max_accuracy']:.2f}%")

    print("\n🏆 Top 5 Cross-Timeframe Combinations:")
    for i, combo in enumerate(report['cross_timeframe']['top_performers'][:5], 1):
        print(f"{i}. {combo['combo_signature']}")
        print(f"   Accuracy: {combo['accuracy']:.2f}%, "
              f"Samples: {combo['signals_count']}, "
              f"PF: {combo['profit_factor']:.2f}")