"""
Manual CLI for Ingesting NASA POWER Daily Window into Vintage Store (DDL V3.1)
================================================================================
Optionally seeds Vintage 0 from canonical historical CSV if the DB is empty, then pulls
`--lookback-days` (default: 180) from NASA POWER Daily API and inserts only new dates or
changed observation hashes (`LATE_FILL_ONLY` / `VALUE_CHANGE_ONLY` / `MIXED_FILL_AND_CHANGE`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.vintage_store import (  # noqa: E402
    HCMC_LOCATION_ID,
    connect_sqlite,
    ingest_nasa_power_window,
    init_schema,
    seed_historical_csv_vintage_v0,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual NASA POWER daily ingestion into DDL V3.1 vintage SQLite DB.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "vintage" / "rainfall_vintage.db",
        help="Path to SQLite vintage database.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=180,
        help="Number of trailing calendar days to scan from NASA POWER API (default: 180).",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Explicit YYYY-MM-DD scan window start.")
    parser.add_argument("--end-date", type=str, default=None, help="Explicit YYYY-MM-DD scan window end.")
    parser.add_argument(
        "--seed-csv-if-empty",
        action="store_true",
        help="Seed historical canonical CSV as Vintage 0 first if observations_vintage is empty.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional output path to persist ingestion run summary JSON.",
    )
    args = parser.parse_args()

    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_sqlite(args.db_path, read_only=False, recursive_triggers=True)
    try:
        init_schema(conn)
        existing_count = conn.execute("SELECT COUNT(*) FROM observations_vintage").fetchone()[0]
        seed_run_id = None
        if args.seed_csv_if_empty and existing_count == 0:
            logger.info("Database is empty; seeding canonical historical CSV as Vintage 0...")
            seed_run_id = seed_historical_csv_vintage_v0(conn, location_id=HCMC_LOCATION_ID)
            seeded_rows = conn.execute("SELECT COUNT(*) FROM observations_vintage").fetchone()[0]
            logger.info("Seeded Vintage 0 complete: run_id=%s, total_rows=%d", seed_run_id, seeded_rows)

        summary = ingest_nasa_power_window(
            conn,
            lookback_days=args.lookback_days,
            start_date=args.start_date,
            end_date=args.end_date,
            location_id=HCMC_LOCATION_ID,
            run_type="manual_scan",
        )
        if seed_run_id is not None:
            summary["seed_v0_run_id"] = seed_run_id

        if args.report_json is not None:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Saved NASA ingestion audit JSON to %s", args.report_json)

        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
