#!/usr/bin/env python3
"""
Fetch forward price outcomes for alerts that don't have one yet, from
on-chain OHLCV candles (GeckoTerminal). Safe to re-run — only touches alerts
missing a complete outcome, oldest first.

Usage:
  python scripts/backfill_outcomes.py [--limit 200]

Run this periodically (e.g. hourly via cron) so new alerts accumulate
outcomes as they age past 24h; running it once now will backfill the
existing history.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crypto1k.core import db, outcomes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200,
                         help="max alerts to process this run")
    args = parser.parse_args()

    db.init_db()
    result = outcomes.backfill_outcomes(limit=args.limit)
    print(f"processed={result['processed']} completed={result['completed']} "
          f"skipped={result['skipped']}")


if __name__ == "__main__":
    main()
