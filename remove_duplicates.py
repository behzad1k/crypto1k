"""
Signal Deduplication Script
Removes duplicate signals from position_signals and live_signals tables
Keeps only one signal per symbol-signal-timeframe combination within each timeframe window
"""

import sqlite3
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Timeframe windows in minutes
TIMEFRAME_WINDOWS = {
  '1m': 1,
  '3m': 3,
  '5m': 5,
  '15m': 15,
  '30m': 30,
  '1h': 60,
  '2h': 120,
  '4h': 240,
  '6h': 360,
  '8h': 480,
  '12h': 720,
  '1d': 1440,
  '3d': 4320,
  '1w': 10080
}


def deduplicate_position_signals(db_path='crypto_signals.db', dry_run=True):
  """
  Deduplicate position_signals table
  Keep only one signal per position-signal-timeframe combination within each timeframe window
  """
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  logging.info("=" * 80)
  logging.info("DEDUPLICATING position_signals TABLE")
  logging.info("=" * 80)

  # Get count before
  cursor.execute("SELECT COUNT(*) FROM position_signals")
  count_before = cursor.fetchone()[0]
  logging.info(f"Records before: {count_before:,}")

  # Get all unique combinations
  cursor.execute("""
        SELECT DISTINCT position_id, signal_name, timeframe
        FROM position_signals
        ORDER BY position_id, signal_name, timeframe
    """)

  combinations = cursor.fetchall()
  logging.info(f"Processing {len(combinations)} unique position-signal-timeframe combinations...")

  total_to_delete = 0
  ids_to_delete = []

  for idx, (position_id, signal_name, timeframe) in enumerate(combinations, 1):
    if idx % 100 == 0:
      logging.info(f"Progress: {idx}/{len(combinations)}")

    window_minutes = TIMEFRAME_WINDOWS.get(timeframe, 60)

    # Get all signals for this combination, ordered by time
    cursor.execute("""
            SELECT id, detected_at
            FROM position_signals
            WHERE position_id = ? AND signal_name = ? AND timeframe = ?
            ORDER BY detected_at ASC
        """, (position_id, signal_name, timeframe))

    signals = cursor.fetchall()

    if len(signals) <= 1:
      continue

    # Group signals into windows and keep only the first one in each window
    last_kept_time = None

    for signal_id, detected_at_str in signals:
      detected_at = datetime.fromisoformat(detected_at_str)

      if last_kept_time is None:
        # Keep the first signal
        last_kept_time = detected_at
      else:
        # Check if this signal is within the window of the last kept signal
        time_diff = (detected_at - last_kept_time).total_seconds() / 60

        if time_diff < window_minutes:
          # This is a duplicate - mark for deletion
          ids_to_delete.append(signal_id)
          total_to_delete += 1
        else:
          # This is a new window - keep it
          last_kept_time = detected_at

  logging.info(f"\nTotal signals to delete: {total_to_delete:,}")
  logging.info(f"Signals to keep: {count_before - total_to_delete:,}")
  logging.info(f"Reduction: {(total_to_delete / count_before * 100):.1f}%")

  if dry_run:
    logging.info("\n⚠️  DRY RUN MODE - No changes made")
    logging.info("Run with dry_run=False to actually delete the records")
  else:
    # Delete in batches to avoid hitting SQL limits
    batch_size = 1000
    deleted = 0

    for i in range(0, len(ids_to_delete), batch_size):
      batch = ids_to_delete[i:i + batch_size]
      placeholders = ','.join('?' * len(batch))
      cursor.execute(f"DELETE FROM position_signals WHERE id IN ({placeholders})", batch)
      deleted += len(batch)

      if deleted % 10000 == 0:
        logging.info(f"Deleted: {deleted:,}/{total_to_delete:,}")

    conn.commit()

    # Get count after
    cursor.execute("SELECT COUNT(*) FROM position_signals")
    count_after = cursor.fetchone()[0]

    logging.info(f"\n✅ Deduplication complete!")
    logging.info(f"Records after: {count_after:,}")
    logging.info(f"Deleted: {count_before - count_after:,}")

  conn.close()
  return total_to_delete


def deduplicate_live_signals(db_path='crypto_signals.db', dry_run=True):
  """
  Deduplicate live_signals table
  Keep only one signal per symbol-signal-timeframe combination within each timeframe window
  """
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  logging.info("\n" + "=" * 80)
  logging.info("DEDUPLICATING live_signals TABLE")
  logging.info("=" * 80)

  # Get count before
  cursor.execute("SELECT COUNT(*) FROM live_signals")
  count_before = cursor.fetchone()[0]
  logging.info(f"Records before: {count_before:,}")

  # Get all unique combinations
  cursor.execute("""
        SELECT DISTINCT symbol, signal_name, timeframe
        FROM live_signals
        ORDER BY symbol, signal_name, timeframe
    """)

  combinations = cursor.fetchall()
  logging.info(f"Processing {len(combinations)} unique symbol-signal-timeframe combinations...")

  total_to_delete = 0
  ids_to_delete = []

  for idx, (symbol, signal_name, timeframe) in enumerate(combinations, 1):
    if idx % 100 == 0:
      logging.info(f"Progress: {idx}/{len(combinations)}")

    window_minutes = TIMEFRAME_WINDOWS.get(timeframe, 60)

    # Get all signals for this combination, ordered by time
    cursor.execute("""
            SELECT id, timestamp
            FROM live_signals
            WHERE symbol = ? AND signal_name = ? AND timeframe = ?
            ORDER BY timestamp ASC
        """, (symbol, signal_name, timeframe))

    signals = cursor.fetchall()

    if len(signals) <= 1:
      continue

    # Group signals into windows and keep only the first one in each window
    last_kept_time = None

    for signal_id, timestamp_str in signals:
      timestamp = datetime.fromisoformat(timestamp_str)

      if last_kept_time is None:
        # Keep the first signal
        last_kept_time = timestamp
      else:
        # Check if this signal is within the window of the last kept signal
        time_diff = (timestamp - last_kept_time).total_seconds() / 60

        if time_diff < window_minutes:
          # This is a duplicate - mark for deletion
          ids_to_delete.append(signal_id)
          total_to_delete += 1
        else:
          # This is a new window - keep it
          last_kept_time = timestamp

  logging.info(f"\nTotal signals to delete: {total_to_delete:,}")
  logging.info(f"Signals to keep: {count_before - total_to_delete:,}")
  logging.info(f"Reduction: {(total_to_delete / count_before * 100):.1f}%")

  if dry_run:
    logging.info("\n⚠️  DRY RUN MODE - No changes made")
    logging.info("Run with dry_run=False to actually delete the records")
  else:
    # Delete in batches to avoid hitting SQL limits
    batch_size = 1000
    deleted = 0

    for i in range(0, len(ids_to_delete), batch_size):
      batch = ids_to_delete[i:i + batch_size]
      placeholders = ','.join('?' * len(batch))
      cursor.execute(f"DELETE FROM live_signals WHERE id IN ({placeholders})", batch)
      deleted += len(batch)

      if deleted % 10000 == 0:
        logging.info(f"Deleted: {deleted:,}/{total_to_delete:,}")

    conn.commit()

    # Get count after
    cursor.execute("SELECT COUNT(*) FROM live_signals")
    count_after = cursor.fetchone()[0]

    logging.info(f"\n✅ Deduplication complete!")
    logging.info(f"Records after: {count_after:,}")
    logging.info(f"Deleted: {count_before - count_after:,}")

  conn.close()
  return total_to_delete


def create_backup(db_path='crypto_signals.db'):
  """Create a backup of the database before deduplication"""
  import shutil
  from datetime import datetime

  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  backup_path = f"{db_path}.backup_{timestamp}"

  logging.info(f"Creating backup: {backup_path}")
  shutil.copy2(db_path, backup_path)
  logging.info(f"✅ Backup created successfully")

  return backup_path


def vacuum_database(db_path='crypto_signals.db'):
  """Vacuum database to reclaim space after deletion"""
  logging.info("\n" + "=" * 80)
  logging.info("VACUUMING DATABASE")
  logging.info("=" * 80)

  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  logging.info("Analyzing database before vacuum...")
  cursor.execute("PRAGMA page_count")
  page_count_before = cursor.fetchone()[0]
  cursor.execute("PRAGMA page_size")
  page_size = cursor.fetchone()[0]
  size_before_mb = (page_count_before * page_size) / (1024 * 1024)

  logging.info(f"Database size before: {size_before_mb:.2f} MB")
  logging.info("Running VACUUM... (this may take a while)")

  cursor.execute("VACUUM")

  cursor.execute("PRAGMA page_count")
  page_count_after = cursor.fetchone()[0]
  size_after_mb = (page_count_after * page_size) / (1024 * 1024)

  space_saved_mb = size_before_mb - size_after_mb

  logging.info(f"Database size after: {size_after_mb:.2f} MB")
  logging.info(f"Space saved: {space_saved_mb:.2f} MB ({(space_saved_mb / size_before_mb * 100):.1f}%)")

  conn.close()


def main():
  """Main execution function"""
  import argparse

  parser = argparse.ArgumentParser(description='Deduplicate signals in crypto_signals database')
  parser.add_argument('--db-path', default='crypto_signals.db', help='Path to database file')
  parser.add_argument('--no-backup', action='store_true', help='Skip creating backup')
  parser.add_argument('--dry-run', action='store_true', help='Perform dry run without deleting')
  parser.add_argument('--skip-position-signals', action='store_true', help='Skip position_signals table')
  parser.add_argument('--skip-live-signals', action='store_true', help='Skip live_signals table')
  parser.add_argument('--vacuum', action='store_true', help='Vacuum database after deduplication')

  args = parser.parse_args()

  logging.info("\n" + "=" * 80)
  logging.info("SIGNAL DEDUPLICATION SCRIPT")
  logging.info("=" * 80)
  logging.info(f"Database: {args.db_path}")
  logging.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
  logging.info("=" * 80 + "\n")

  # Create backup unless explicitly skipped
  if not args.no_backup and not args.dry_run:
    try:
      backup_path = create_backup(args.db_path)
      logging.info(f"Backup saved: {backup_path}\n")
    except Exception as e:
      logging.error(f"Failed to create backup: {e}")
      logging.error("Aborting to prevent data loss!")
      return

  # Deduplicate position_signals
  if not args.skip_position_signals:
    try:
      deduplicate_position_signals(args.db_path, dry_run=args.dry_run)
    except Exception as e:
      logging.error(f"Error deduplicating position_signals: {e}")
      import traceback
      traceback.print_exc()

  # Deduplicate live_signals
  if not args.skip_live_signals:
    try:
      deduplicate_live_signals(args.db_path, dry_run=args.dry_run)
    except Exception as e:
      logging.error(f"Error deduplicating live_signals: {e}")
      import traceback
      traceback.print_exc()

  # Vacuum database if requested
  if args.vacuum and not args.dry_run:
    try:
      vacuum_database(args.db_path)
    except Exception as e:
      logging.error(f"Error vacuuming database: {e}")

  logging.info("\n" + "=" * 80)
  logging.info("SCRIPT COMPLETE")
  logging.info("=" * 80)


if __name__ == "__main__":
  main()