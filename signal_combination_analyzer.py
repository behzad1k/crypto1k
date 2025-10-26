"""
Signal Combination Analyzer
Analyzes accuracy of signal combinations that appear together in same timeframe
Uses fact-check results instead of live API calls for performance
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set, Optional
from itertools import combinations
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)


class SignalCombinationAnalyzer:
    """
    Analyzes signal combinations to find which signals work better together
    """

    def __init__(self, db_path: str = 'crypto_signals.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Create table for combination results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

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

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_combos_accuracy 
            ON tf_combos(timeframe, accuracy DESC)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_combos_size 
            ON tf_combos(combo_size, accuracy DESC)
        ''')

        conn.commit()
        conn.close()
        logging.info("✅ tf_combos table initialized")

    def get_timeframe_window_seconds(self, timeframe: str) -> int:
        """Get time window in seconds for grouping signals"""
        timeframe_seconds = {
            '1m': 60, '3m': 180, '5m': 300, '15m': 900,
            '30m': 1800, '1h': 3600, '2h': 7200, '4h': 14400,
            '6h': 21600, '8h': 28800, '12h': 43200, '1d': 86400,
            '3d': 259200, '1w': 604800
        }
        return timeframe_seconds.get(timeframe, 3600)

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
            logging.info(f"\n🏆 Best combination: {top['combo_name']}")
            logging.info(f"   Accuracy: {top['accuracy']:.2f}%")
            logging.info(f"   Samples: {top['total_samples']}")
            logging.info(f"   Profit Factor: {top['profit_factor']:.2f}")

        return results

    def analyze_all_timeframes(self, min_samples: int = 10,
                              min_combo_size: int = 2, max_combo_size: int = 5) -> Dict:
        """
        Analyze combinations across all timeframes
        """
        # Get all timeframes that have fact-checked signals
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT DISTINCT ls.timeframe, COUNT(*) as signal_count
            FROM live_signals ls
            INNER JOIN signal_fact_checks sfc
                ON ls.signal_name = sfc.signal_name
                AND ls.timeframe = sfc.timeframe
                AND ls.timestamp = sfc.detected_at
            GROUP BY ls.timeframe
            HAVING signal_count >= ?
            ORDER BY ls.timeframe
        ''', (min_samples * 2,))

        timeframes = [row[0] for row in cursor.fetchall()]
        conn.close()

        logging.info(f"\n{'=' * 80}")
        logging.info(f"BULK COMBINATION ANALYSIS - ALL TIMEFRAMES")
        logging.info(f"{'=' * 80}")
        logging.info(f"Timeframes to analyze: {', '.join(timeframes)}\n")

        all_results = {
            'timeframes_analyzed': len(timeframes),
            'total_combinations': 0,
            'best_overall': None,
            'by_timeframe': {}
        }

        best_accuracy = 0
        best_combo = None

        for tf in timeframes:
            tf_results = self.analyze_timeframe_combinations(
                tf, min_samples, min_combo_size, max_combo_size
            )

            all_results['by_timeframe'][tf] = tf_results
            all_results['total_combinations'] += tf_results['analyzed']

            # Track best overall
            if tf_results['combinations']:
                top_combo = tf_results['combinations'][0]
                if top_combo['accuracy'] > best_accuracy:
                    best_accuracy = top_combo['accuracy']
                    best_combo = {**top_combo, 'timeframe': tf}

        all_results['best_overall'] = best_combo

        # Print summary
        logging.info(f"\n{'=' * 80}")
        logging.info("SUMMARY - ALL TIMEFRAMES")
        logging.info(f"{'=' * 80}")
        logging.info(f"Timeframes analyzed: {all_results['timeframes_analyzed']}")
        logging.info(f"Total combinations: {all_results['total_combinations']}")

        if best_combo:
            logging.info(f"\n🏆 BEST COMBINATION OVERALL:")
            logging.info(f"   Signals: {best_combo['combo_name']}")
            logging.info(f"   Timeframe: {best_combo['timeframe']}")
            logging.info(f"   Accuracy: {best_combo['accuracy']:.2f}%")
            logging.info(f"   Samples: {best_combo['total_samples']}")
            logging.info(f"   Profit Factor: {best_combo['profit_factor']:.2f}")

        return all_results

    def save_combination_result(self, stats: Dict, timeframe: str):
        """Save combination analysis result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO tf_combos (
                signal_name, timeframe, accuracy, signals_count,
                correct_predictions, avg_price_change, profit_factor,
                combo_size, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            stats['combo_name'], timeframe, stats['accuracy'],
            stats['total_samples'], stats['correct_predictions'],
            stats['avg_price_change'], stats['profit_factor'],
            stats['combo_size'], datetime.now()
        ))

        conn.commit()
        conn.close()

    def get_top_combinations(self, timeframe: str = None, min_accuracy: float = 60.0,
                            limit: int = 20) -> List[Dict]:
        """Get top performing combinations"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = '''
            SELECT * FROM tf_combos
            WHERE accuracy >= ?
        '''
        params = [min_accuracy]

        if timeframe:
            query += ' AND timeframe = ?'
            params.append(timeframe)

        query += ' ORDER BY accuracy DESC, signals_count DESC LIMIT ?'
        params.append(limit)

        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return results

    def get_combination_details(self, combo_name: str, timeframe: str) -> Optional[Dict]:
        """Get detailed stats for a specific combination"""
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

    def generate_report(self, min_accuracy: float = 60.0) -> Dict:
        """Generate comprehensive report of best combinations"""
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
    print("SIGNAL COMBINATION ANALYZER")
    print("=" * 80)

    # Analyze all timeframes
    results = analyzer.analyze_all_timeframes(
        min_samples=20,  # Need at least 20 samples
        min_combo_size=2,  # Start with 2-signal combos
        max_combo_size=4   # Up to 4-signal combos
    )

    # Generate report
    print("\n" + "=" * 80)
    print("GENERATING REPORT...")
    print("=" * 80)

    report = analyzer.generate_report(min_accuracy=55.0)

    print("\n📊 COMBINATION ANALYSIS REPORT")
    print(f"Total combinations analyzed: {report['overall']['total_combinations']}")
    print(f"Average accuracy: {report['overall']['avg_accuracy']:.2f}%")
    print(f"Maximum accuracy: {report['overall']['max_accuracy']:.2f}%")

    print("\n📈 By Combo Size:")
    for size_data in report['by_combo_size']:
        print(f"  {size_data['combo_size']} signals: "
              f"Avg {size_data['avg_accuracy']:.2f}%, "
              f"Max {size_data['max_accuracy']:.2f}% "
              f"({size_data['count']} combinations)")

    print("\n🏆 Top 10 Combinations:")
    for i, combo in enumerate(report['top_performers'][:10], 1):
        print(f"{i}. {combo['signal_name']} [{combo['timeframe']}]")
        print(f"   Accuracy: {combo['accuracy']:.2f}%, "
              f"Samples: {combo['signals_count']}, "
              f"PF: {combo['profit_factor']:.2f}")