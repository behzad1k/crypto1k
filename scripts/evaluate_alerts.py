#!/usr/bin/env python3
"""
Print the alert-outcome evaluation report: hit rate and average forward
return per score band / alert path / symbol, plus concrete threshold
suggestions. Run scripts/backfill_outcomes.py first (or run this after the
web app has been backfilling in the background).

Usage:
  python scripts/evaluate_alerts.py [--min-bucket 3]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crypto1k.core import db, evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-bucket", type=int, default=3,
                         help="minimum alerts required before a bucket is reported")
    args = parser.parse_args()

    db.init_db()
    report = evaluation.generate_report(min_alerts_per_bucket=args.min_bucket)
    print(evaluation.format_report(report))


if __name__ == "__main__":
    main()
