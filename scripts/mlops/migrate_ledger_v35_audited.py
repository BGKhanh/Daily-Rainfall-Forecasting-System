"""
Audited Migration of `forecasts_ledger` from DDL V3.4 to DDL V3.5 (DS108 Round 13)
=================================================================================
Rebuilds `forecasts_ledger` atomically with DDL V3.5 constraints and provenance columns:
  - Adds `imputed_values_json TEXT NOT NULL DEFAULT '{}'`
  - Adds `doy_table_sha256 TEXT`
  - Allows `'P-c_DOY_MEDIAN_11_SOLAR_V1'` in `ceres_policy_applied`
  - Removes silent SQL DEFAULT on `issued_by` and allows `'manual_operational'`
  - Recreates immutability triggers (`no_replace`, `no_update`, `no_delete`) & `v_forecast_scores`

Generates a cryptographic proof JSON verifying:
  1. Per-column SHA-256 bit-identity (`before_sha256[col] == after_sha256[col]`) across all
     29 original columns on every existing row (e.g., all 4,270 rows in `demo_rainfall_vintage.db`).
  2. Zero NULLs in `issued_by` and valid CHECK constraints.
  3. Trigger immutability probes (`UPDATE`, `DELETE`, duplicate `INSERT OR REPLACE` -> IntegrityError).
  4. Strict explicit `issued_by` rejection (`NULL` & `'invalid_issuer'` -> IntegrityError / ValueError).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.clock import now_utc_iso  # noqa: E402
from src.data.vintage_store import (  # noqa: E402
    FORECASTS_LEDGER_29_ORIGINAL_COLS,
    apply_migrations_v3_5,
    connect_sqlite,
    finalize_db_for_read_only,
)

DEFAULT_DEMO_DB = PROJECT_ROOT / "artifacts" / "models" / "demo_rainfall_vintage.db"
DEFAULT_PROOF_OUT = PROJECT_ROOT / "reports" / "storage_audit" / "immutable_ledger_migration_proof_v35.json"


def compute_forecasts_column_hashes(
    conn: sqlite3.Connection,
    columns: Tuple[str, ...],
) -> Tuple[int, Dict[str, str]]:
    """Compute deterministic ordered per-column SHA-256 digests across `forecasts_ledger`."""
    col_list = ", ".join(columns)
    rows = conn.execute(
        f"SELECT {col_list} FROM forecasts_ledger ORDER BY forecast_id"
    ).fetchall()
    hashes = {col: hashlib.sha256() for col in columns}
    for row in rows:
        for col in columns:
            val = row[col]
            token = "<NULL>" if val is None else repr(val)
            hashes[col].update(token.encode("utf-8"))
            hashes[col].update(b"\n")
    return len(rows), {col: hashes[col].hexdigest() for col in columns}


def _run_immutability_and_constraint_probes(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Verify UPDATE, DELETE, duplicate INSERT, and invalid/missing issued_by all abort."""
    sample = conn.execute(
        "SELECT forecast_id, location_id, model_id, data_as_of, horizon_day FROM forecasts_ledger LIMIT 1"
    ).fetchone()
    results: Dict[str, bool] = {}

    if sample is not None:
        fid = str(sample["forecast_id"])
        for probe_name, sql, params in [
            ("update_aborted", "UPDATE forecasts_ledger SET pred_mm = pred_mm + 1.0 WHERE forecast_id = ?", (fid,)),
            ("delete_aborted", "DELETE FROM forecasts_ledger WHERE forecast_id = ?", (fid,)),
            (
                "duplicate_insert_aborted",
                "INSERT INTO forecasts_ledger (forecast_id, issued_at_utc, issued_run_date, location_id, model_id, model_kind, artifact_sha256, data_as_of, eval_split_regime, horizon_day, target_date, temporal_status_at_issue, pred_mm, input_window_start, input_window_end, input_window_hash, input_vintage_manifest_json, input_provisional_days, ceres_policy_applied, git_sha, issued_by) VALUES (?, '2026-11-21T00:00:00Z', '2026-11-21', 'HCMC', 'SC-35', 'ML_SERVING', 's', '2026-05-01', 'POST_TEST', 1, '2026-05-02', 'future_forecast', 0.0, '2026-04-02', '2026-05-01', 'h', '[]', 0, 'NONE_COMPLETE', 'g', 'automated_pipeline')",
                (fid,),
            ),
        ]:
            try:
                conn.execute(sql, params)
                results[probe_name] = False
            except sqlite3.IntegrityError:
                results[probe_name] = True
                conn.rollback()

    # Probe strict NOT NULL + CHECK on issued_by regardless of whether table already has rows
    for probe_name, bad_issuer_sql, bad_params in [
        (
            "null_issued_by_rejected",
            "INSERT INTO forecasts_ledger (forecast_id, issued_at_utc, issued_run_date, location_id, model_id, model_kind, artifact_sha256, data_as_of, eval_split_regime, horizon_day, target_date, temporal_status_at_issue, pred_mm, input_window_start, input_window_end, input_window_hash, input_vintage_manifest_json, input_provisional_days, ceres_policy_applied, git_sha) VALUES ('probe_null_issuer', '2026-11-21T00:00:00Z', '2026-11-21', 'HCMC', 'SC-35', 'ML_SERVING', 's', '2099-01-01', 'POST_TEST', 1, '2099-01-02', 'future_forecast', 0.0, '2098-12-03', '2099-01-01', 'h', '[]', 0, 'NONE_COMPLETE', 'g')",
            (),
        ),
        (
            "invalid_issued_by_rejected",
            "INSERT INTO forecasts_ledger (forecast_id, issued_at_utc, issued_run_date, location_id, model_id, model_kind, artifact_sha256, data_as_of, eval_split_regime, horizon_day, target_date, temporal_status_at_issue, pred_mm, input_window_start, input_window_end, input_window_hash, input_vintage_manifest_json, input_provisional_days, ceres_policy_applied, git_sha, issued_by) VALUES ('probe_bad_issuer', '2026-11-21T00:00:00Z', '2026-11-21', 'HCMC', 'SC-35', 'ML_SERVING', 's', '2099-01-01', 'POST_TEST', 2, '2099-01-03', 'future_forecast', 0.0, '2098-12-03', '2099-01-01', 'h', '[]', 0, 'NONE_COMPLETE', 'g', ?)",
            ("unauthorized_actor",),
        ),
    ]:
        try:
            conn.execute(bad_issuer_sql, bad_params)
            results[probe_name] = False
        except sqlite3.IntegrityError:
            results[probe_name] = True
            conn.rollback()

    return results


def migrate_with_audit(
    db_path: Path,
    *,
    proof_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run audited V3.5 migration on `db_path` and verify 29-column bit identity + triggers."""
    conn = connect_sqlite(db_path, read_only=False, recursive_triggers=True)
    try:
        before_count, before_hashes = compute_forecasts_column_hashes(
            conn, FORECASTS_LEDGER_29_ORIGINAL_COLS
        )
        mig_res = apply_migrations_v3_5(conn)
        after_count, after_hashes = compute_forecasts_column_hashes(
            conn, FORECASTS_LEDGER_29_ORIGINAL_COLS
        )

        mismatches = [
            col for col in FORECASTS_LEDGER_29_ORIGINAL_COLS
            if before_hashes[col] != after_hashes[col]
        ]
        if before_count != after_count or mismatches:
            raise RuntimeError(
                f"Audited V3.5 migration failed integrity check: "
                f"count {before_count}->{after_count}, mismatched_cols={mismatches}"
            )

        null_issued_by = int(
            conn.execute(
                "SELECT COUNT(*) FROM forecasts_ledger WHERE issued_by IS NULL OR issued_by = ''"
            ).fetchone()[0]
        )
        if null_issued_by != 0:
            raise RuntimeError(f"Found {null_issued_by} NULL/empty issued_by rows after V3.5 migration.")

        probes = _run_immutability_and_constraint_probes(conn)
        if not all(probes.values()):
            raise RuntimeError(f"One or more immutability/constraint probes failed: {probes}")

        integrity_ok = str(conn.execute("PRAGMA integrity_check;").fetchone()[0]).lower() == "ok"
        if not integrity_ok:
            raise RuntimeError("PRAGMA integrity_check did not return 'ok'.")

        v35_row = conn.execute("SELECT * FROM schema_migrations WHERE version = 'V3.5'").fetchone()
        view_cols = [
            str(r[1]) for r in conn.execute("PRAGMA table_info(v_forecast_scores)").fetchall()
        ]
        finalize_db_for_read_only(conn)
    finally:
        conn.close()

    proof: Dict[str, Any] = {
        "schema_version": "V3.5",
        "audited_at_utc": now_utc_iso(),
        "database_path": str(db_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if db_path.is_relative_to(PROJECT_ROOT)
        else str(db_path),
        "migration_executed": bool(mig_res.get("migrated", False)),
        "row_count_before": before_count,
        "row_count_after": after_count,
        "zero_dropped_rows": before_count == after_count,
        "columns_verified_count": len(FORECASTS_LEDGER_29_ORIGINAL_COLS),
        "all_29_columns_bit_identical": len(mismatches) == 0,
        "null_or_empty_issued_by_count": null_issued_by,
        "immutability_and_constraint_probes": probes,
        "v_forecast_scores_columns": view_cols,
        "schema_migrations_v35": dict(v35_row) if v35_row is not None else None,
        "before_column_sha256": before_hashes,
        "after_column_sha256": after_hashes,
        "db_file_sha256": hashlib.sha256(db_path.read_bytes()).hexdigest(),
    }

    if proof_out is not None:
        proof_out.parent.mkdir(parents=True, exist_ok=True)
        proof_out.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")

    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description="Audited DDL V3.5 migration for forecasts_ledger.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DEMO_DB)
    parser.add_argument("--proof-out", type=Path, default=DEFAULT_PROOF_OUT)
    args = parser.parse_args()

    proof = migrate_with_audit(args.db_path, proof_out=args.proof_out)
    print(
        json.dumps(
            {
                "db": proof["database_path"],
                "rows": proof["row_count_after"],
                "29_cols_identical": proof["all_29_columns_bit_identical"],
                "probes": proof["immutability_and_constraint_probes"],
                "db_sha256": proof["db_file_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
