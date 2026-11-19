"""Ensure `artifacts/models/demo_rainfall_vintage.db` is a hydrated SQLite binary (not a Git LFS pointer)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DB_PATH = PROJECT_ROOT / "artifacts" / "models" / "demo_rainfall_vintage.db"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "models" / "metadata.json"
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "hcmc_weather_data_daily_20000101_20260430.csv"
LFS_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ensure_demo_db")


def _read_file_head(path: Path, nbytes: int = 64) -> bytes:
    with path.open("rb") as fh:
        return fh.read(nbytes)


def is_hydrated_sqlite_db(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    size = db_path.stat().st_size
    header = _read_file_head(db_path, 64)
    if size < 100_000:
        if header.startswith(LFS_SIGNATURE):
            logger.warning("%s is an unhydrated Git LFS pointer (%d bytes).", db_path, size)
            return False
        if not header.startswith(b"SQLite format 3\x00"):
            logger.warning("%s is too small (%d bytes) and not a valid SQLite header.", db_path, size)
            return False
    return header[:16] == b"SQLite format 3\x00"


def rebuild_demo_db_fallback(db_path: Path) -> None:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Cannot rebuild demo DB cleanly: missing {RAW_CSV_PATH}")

    db_path.unlink(missing_ok=True)
    ingest_script = PROJECT_ROOT / "scripts" / "data" / "ingest_nasa_vintage.py"
    ledger_script = PROJECT_ROOT / "scripts" / "mlops" / "run_forecast_ledger.py"

    cmd_ingest = [
        sys.executable,
        str(ingest_script),
        "--db-path",
        str(db_path),
        "--seed-csv-if-empty",
        "--lookback-days",
        "210",
    ]
    res_ingest = subprocess.run(cmd_ingest, cwd=str(PROJECT_ROOT), check=False)
    if res_ingest.returncode != 0:
        logger.warning("NASA scan exited %d; attempting offline historical CSV seed only.", res_ingest.returncode)
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "from src.data.vintage_store import connect_sqlite, finalize_db_for_read_only, init_schema, seed_historical_csv_vintage_v0; "
                    f"c = connect_sqlite(Path({str(db_path)!r}), read_only=False); "
                    "init_schema(c); "
                    "seed_historical_csv_vintage_v0(c); "
                    "c.commit(); finalize_db_for_read_only(c); c.close()"
                ),
            ],
            cwd=str(PROJECT_ROOT),
            check=True,
        )

    subprocess.run(
        [sys.executable, str(ledger_script), "--db-path", str(db_path), "--issue-post-test"],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def main() -> int:
    if is_hydrated_sqlite_db(DEMO_DB_PATH):
        sha = hashlib.sha256(DEMO_DB_PATH.read_bytes()).hexdigest()
        expected = None
        if METADATA_PATH.exists():
            meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            expected = (meta.get("sha256_checksums") or {}).get("demo_rainfall_vintage.db")
        if expected and sha != expected:
            logger.warning(
                "Demo DB checksum differs from metadata.json (%d bytes, sha256=%s, expected=%s).",
                DEMO_DB_PATH.stat().st_size,
                sha,
                expected,
            )
        else:
            logger.info("Demo DB hydrated (%d bytes, sha256=%s, expected=%s).", DEMO_DB_PATH.stat().st_size, sha, expected)
        return 0

    logger.info("Hydrating %s via automated rebuild fallback...", DEMO_DB_PATH)
    rebuild_demo_db_fallback(DEMO_DB_PATH)
    if not is_hydrated_sqlite_db(DEMO_DB_PATH):
        raise RuntimeError(f"Failed to hydrate valid SQLite DB at {DEMO_DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
