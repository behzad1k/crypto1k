"""
Migration Script: Consolidate Signal Tracking
- Moves position_signals data to live_signals
- Removes trading_positions system entirely
- Simplifies the codebase to single signal tracking system
"""

import sqlite3
from datetime import datetime
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def backup_database(db_path: str):
  """Create backup before migration"""
  import shutil
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  backup_path = f"{db_path}.backup_before_trading_removal_{timestamp}"

  logging.info(f"Creating backup: {backup_path}")
  shutil.copy2(db_path, backup_path)
  logging.info(f"✅ Backup created: {backup_path}")

  return backup_path


def migrate_position_signals_to_live_signals(db_path: str):
  """Move all position_signals to live_signals table"""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  logging.info("\n" + "=" * 80)
  logging.info("STEP 1: Migrating position_signals to live_signals")
  logging.info("=" * 80)

  # Check if position_signals table exists
  cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='position_signals'
    """)

  if not cursor.fetchone():
    logging.info("⚠️  position_signals table doesn't exist - skipping migration")
    conn.close()
    return 0

  # Get count of position signals
  cursor.execute("SELECT COUNT(*) FROM position_signals")
  total_signals = cursor.fetchone()[0]

  if total_signals == 0:
    logging.info("⚠️  No signals to migrate")
    conn.close()
    return 0

  logging.info(f"Found {total_signals:,} signals to migrate")

  # Get position_signals with symbol from trading_positions
  cursor.execute("""
        SELECT 
            ps.signal_name,
            ps.signal_type,
            ps.timeframe,
            ps.confidence,
            ps.detected_at as timestamp,
            ps.price_at_detection as price,
            tp.symbol,
            NULL as strength,
            NULL as signal_value
        FROM position_signals ps
        INNER JOIN trading_positions tp ON ps.position_id = tp.id
    """)

  position_signals = cursor.fetchall()

  # Insert into live_signals (ignore duplicates)
  migrated = 0
  skipped = 0

  for signal in position_signals:
    try:
      cursor.execute("""
                INSERT OR IGNORE INTO live_signals (
                    signal_name, signal_type, timeframe, confidence,
                    timestamp, price, symbol, strength, signal_value,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (*signal, datetime.now()))

      if cursor.rowcount > 0:
        migrated += 1
      else:
        skipped += 1

    except Exception as e:
      logging.error(f"Error migrating signal: {e}")
      skipped += 1

  conn.commit()

  logging.info(f"\n✅ Migration complete:")
  logging.info(f"   Migrated: {migrated:,} signals")
  logging.info(f"   Skipped (duplicates): {skipped:,} signals")

  conn.close()
  return migrated


def drop_trading_tables(db_path: str):
  """Drop all trading-related tables"""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  logging.info("\n" + "=" * 80)
  logging.info("STEP 2: Removing trading_positions system tables")
  logging.info("=" * 80)

  tables_to_drop = [
    'position_signals',
    'trading_positions'
  ]

  for table in tables_to_drop:
    try:
      cursor.execute(f"DROP TABLE IF EXISTS {table}")
      logging.info(f"✅ Dropped table: {table}")
    except Exception as e:
      logging.error(f"❌ Error dropping {table}: {e}")

  conn.commit()
  conn.close()

  logging.info("✅ All trading tables removed")


def remove_trading_imports_and_routes(app_file_path: str):
  """Generate cleaned app.py without trading position code"""

  logging.info("\n" + "=" * 80)
  logging.info("STEP 3: Generating cleaned app.py")
  logging.info("=" * 80)

  # Lines to remove (imports)
  lines_to_remove = [
    "from trading_position_manager import TradingPositionManager",
    "trading_manager = None",
    "trading_manager = TradingPositionManager(",
  ]

  # Route prefixes to remove
  routes_to_remove = [
    "@app.route('/api/trading/",
    "@sock.route('/ws/position/",
  ]

  logging.info("⚠️  Manual cleanup required:")
  logging.info("   1. Remove 'from trading_position_manager import TradingPositionManager'")
  logging.info("   2. Remove 'trading_manager' global variable")
  logging.info("   3. Remove all @app.route('/api/trading/*') routes")
  logging.info("   4. Remove @sock.route('/ws/position/<int:position_id>') websocket")
  logging.info("   5. Remove TradingPositionManager initialization in initialize_app()")
  logging.info("   6. Delete trading_position_manager.py file")


def verify_migration(db_path: str):
  """Verify migration was successful"""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  logging.info("\n" + "=" * 80)
  logging.info("VERIFICATION")
  logging.info("=" * 80)

  # Check live_signals count
  cursor.execute("SELECT COUNT(*) FROM live_signals")
  live_count = cursor.fetchone()[0]
  logging.info(f"✅ live_signals table: {live_count:,} signals")

  # Check if trading tables still exist
  cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name IN ('trading_positions', 'position_signals')
    """)

  remaining = cursor.fetchall()
  if remaining:
    logging.warning(f"⚠️  Trading tables still exist: {[r[0] for r in remaining]}")
  else:
    logging.info("✅ All trading tables successfully removed")

  conn.close()


def vacuum_database(db_path: str):
  """Reclaim space after dropping tables"""
  logging.info("\n" + "=" * 80)
  logging.info("STEP 4: Vacuuming database")
  logging.info("=" * 80)

  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  cursor.execute("PRAGMA page_count")
  page_count_before = cursor.fetchone()[0]
  cursor.execute("PRAGMA page_size")
  page_size = cursor.fetchone()[0]
  size_before_mb = (page_count_before * page_size) / (1024 * 1024)

  logging.info(f"Database size before: {size_before_mb:.2f} MB")
  logging.info("Running VACUUM...")

  cursor.execute("VACUUM")

  cursor.execute("PRAGMA page_count")
  page_count_after = cursor.fetchone()[0]
  size_after_mb = (page_count_after * page_size) / (1024 * 1024)

  space_saved_mb = size_before_mb - size_after_mb

  logging.info(f"Database size after: {size_after_mb:.2f} MB")
  logging.info(f"Space saved: {space_saved_mb:.2f} MB")

  conn.close()


def main():
  """Run complete migration"""
  import argparse

  parser = argparse.ArgumentParser(description='Migrate and remove trading_positions system')
  parser.add_argument('--db-path', default='crypto_signals.db', help='Path to database')
  parser.add_argument('--skip-backup', action='store_true', help='Skip backup creation')
  parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')

  args = parser.parse_args()

  print("\n" + "=" * 80)
  print("TRADING POSITIONS SYSTEM REMOVAL")
  print("=" * 80)
  print(f"Database: {args.db_path}")
  print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
  print("=" * 80)

  if args.dry_run:
    print("\n⚠️  DRY RUN MODE - No changes will be made\n")

    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    cursor.execute("""
            SELECT COUNT(*) FROM position_signals ps
            INNER JOIN trading_positions tp ON ps.position_id = tp.id
        """)
    count = cursor.fetchone()[0]

    print(f"Would migrate {count:,} signals from position_signals to live_signals")
    print("Would drop tables: position_signals, trading_positions")
    print("\nRun without --dry-run to execute migration")

    conn.close()
    return

  # Create backup
  if not args.skip_backup:
    backup_path = backup_database(args.db_path)
    print(f"\n💾 Backup created: {backup_path}")
  else:
    print("\n⚠️  Skipping backup as requested")
    response = input("Are you sure you want to continue without backup? (yes/no): ")
    if response.lower() != 'yes':
      print("Migration cancelled")
      return

  try:
    # Step 1: Migrate signals
    migrated = migrate_position_signals_to_live_signals(args.db_path)

    # Step 2: Drop tables
    drop_trading_tables(args.db_path)

    # Step 3: Show manual cleanup instructions
    remove_trading_imports_and_routes('app.py')

    # Step 4: Verify
    verify_migration(args.db_path)

    # Step 5: Vacuum
    vacuum_database(args.db_path)

    print("\n" + "=" * 80)
    print("✅ MIGRATION COMPLETE")
    print("=" * 80)
    print(f"Migrated signals: {migrated:,}")
    print("\nNext steps:")
    print("1. Review and clean up app.py (remove trading routes)")
    print("2. Delete trading_position_manager.py")
    print("3. Test the application")
    print("4. If everything works, you can delete the backup file")
    print("=" * 80 + "\n")

  except Exception as e:
    logging.error(f"\n❌ Migration failed: {e}")
    import traceback
    traceback.print_exc()

    if not args.skip_backup:
      print(f"\n💾 Restore from backup: {backup_path}")

    sys.exit(1)


if __name__ == "__main__":
  main()