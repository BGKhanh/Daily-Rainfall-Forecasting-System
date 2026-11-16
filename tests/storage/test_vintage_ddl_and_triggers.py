"""
Comprehensive DDL V3 Verification Tests (`src/data/vintage_store.py`)
=====================================================================
Tests:
1. 30 immutability trigger probes: (UPDATE, DELETE, INSERT OR REPLACE, REPLACE INTO,
   ON CONFLICT DO UPDATE) x 3 tables (observations_vintage, forecasts_ledger, ingestion_runs)
   x 2 SQLite modes (PRAGMA recursive_triggers=0 and 1).
2. Single-transaction atomicity, DEFERRABLE FK check at COMMIT, mid-batch ROLLBACK,
   clean FAILED `ingestion_runs` audit row, and CHECK constraint enforcement.
3. Full Vintage 0 seed parity vs DataLoader().load_data() (0.0 numeric diff, exact rows/cols).
4. Vintage revision simulation + As-Of query SHA-256 input_window_hash reconstruction
   + v_forecast_scores dual error comparison (abs_error_vs_first_mm vs abs_error_vs_latest_mm).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader  # noqa: E402
from src.data.vintage_store import (  # noqa: E402
    ALL_35_VARS,
    CERES_ONLY_8_VARS,
    FLASHFLUX_3_VARS,
    HCMC_LOCATION_ID,
    classify_completeness,
    compute_input_window_hash,
    compute_row_value_hash,
    connect_sqlite,
    derive_eval_split_regime,
    ingest_nasa_power_window,
    init_schema,
    insert_forecasts_idempotent,
    query_observations_as_of,
    read_latest_observations,
    seed_historical_csv_vintage_v0,
)
from src.evaluation.protocols import get_global_read_guard  # noqa: E402


@pytest.fixture(autouse=True)
def _allow_canonical_csv_reads():
    g = get_global_read_guard()
    was_active = g.is_active
    g.deactivate()
    yield
    if was_active:
        g.activate()


def _sample_var_dict(rain: float = 2.5, temp: float = 28.0) -> dict[str, float]:
    vals = {var: 1.0 + idx * 0.01 for idx, var in enumerate(ALL_35_VARS)}
    vals["PRECTOTCORR"] = rain
    vals["T2M"] = temp
    return vals


def _seed_one_row_for_each_table(conn: sqlite3.Connection) -> tuple[str, str, str]:
    run_id = "run_test_001"
    obs_date = "2026-04-20"
    forecast_id = "fc_test_001"
    vars_dict = _sample_var_dict()
    comp = classify_completeness(vars_dict)
    r_hash = compute_row_value_hash(vars_dict)

    conn.execute("BEGIN IMMEDIATE;")
    conn.execute(
        """
        INSERT INTO ingestion_runs (
            run_id, run_type, started_at_utc, finished_at_utc, status,
            location_id, scan_window_start, scan_window_end,
            rows_scanned, rows_inserted_v0, rows_new_vintage,
            rows_late_fill_only, rows_value_change, rows_unchanged, git_sha
        ) VALUES (?, 'historical_seed', '2026-11-16T00:00:00Z', '2026-11-16T00:00:01Z', 'SUCCESS',
                  'HCMC', ?, ?, 1, 1, 0, 0, 0, 0, 'testsha')
        """,
        (run_id, obs_date, obs_date),
    )

    cols = [
        "location_id",
        "obs_date",
        "version_num",
        "ingested_at_utc",
        "run_id",
        "age_days_at_ingest",
        "ingest_origin",
        "missing_count",
        "missing_meteo_count",
        "missing_flashflux_count",
        "missing_ceres_count",
        "completeness_regime",
        "missing_vars_json",
        "change_type_summary",
        "late_fill_count",
        "late_fill_vars_json",
        "value_change_count",
        "value_change_details_json",
        "anomaly_count",
        "quality_flags_json",
        "row_value_hash",
        *ALL_35_VARS,
    ]
    row_vals = [
        HCMC_LOCATION_ID,
        obs_date,
        0,
        "2026-11-16T00:00:00Z",
        run_id,
        210,
        "HISTORICAL_CSV_SEED",
        comp["missing_count"],
        comp["missing_meteo_count"],
        comp["missing_flashflux_count"],
        comp["missing_ceres_count"],
        comp["completeness_regime"],
        comp["missing_vars_json"],
        "INITIAL_SEED",
        0,
        "[]",
        0,
        "{}",
        0,
        "{}",
        r_hash,
        *[vars_dict[v] for v in ALL_35_VARS],
    ]
    conn.execute(
        f"INSERT INTO observations_vintage ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
        row_vals,
    )

    conn.execute(
        """
        INSERT INTO forecasts_ledger (
            forecast_id, issued_at_utc, issued_run_date, run_timezone, location_id,
            model_id, model_kind, registry_alias, registry_model_version,
            artifact_sha256, feature_builder_sha256, data_as_of, eval_split_regime,
            horizon_day, target_date, temporal_status_at_issue, pred_mm,
            input_window_start, input_window_end, input_window_hash,
            input_vintage_manifest_json, input_provisional_days,
            meteo_missing_count, flashflux_max_ffill_days, ceres_policy_applied,
            ceres_max_staleness_days, is_degraded_input, degraded_reason, git_sha
        ) VALUES (
            ?, '2026-11-16T00:00:02Z', '2026-04-23', 'Asia/Ho_Chi_Minh', 'HCMC',
            'SC-35', 'ML_SERVING', 'serving', 1,
            'artsha', 'fbsha', '2026-04-20', 'TEST',
            1, '2026-04-21', 'future_forecast', 3.5,
            '2026-03-22', '2026-04-20', 'winhash',
            '[]', 0,
            0, 0, 'NONE_COMPLETE',
            0, 0, NULL, 'testsha'
        )
        """,
        (forecast_id,),
    )
    conn.execute("COMMIT;")
    return run_id, obs_date, forecast_id


@pytest.mark.parametrize("recursive_triggers", [False, True])
def test_30_immutability_trigger_matrix(recursive_triggers: bool):
    """All 5 overwrite/mutation DML commands on all 3 immutable tables must fail with IMMUTABLE_VIOLATION."""
    conn = connect_sqlite(":memory:", recursive_triggers=recursive_triggers)
    try:
        init_schema(conn)
        run_id, obs_date, forecast_id = _seed_one_row_for_each_table(conn)

        rt_val = conn.execute("PRAGMA recursive_triggers;").fetchone()[0]
        assert rt_val == (1 if recursive_triggers else 0)

        # Attack statements per table: (attack_name, sql, params)
        attacks = {
            "observations_vintage": [
                ("UPDATE", "UPDATE observations_vintage SET PRECTOTCORR = 99.0 WHERE obs_date = ?", (obs_date,)),
                ("DELETE", "DELETE FROM observations_vintage WHERE obs_date = ?", (obs_date,)),
                (
                    "INSERT_OR_REPLACE",
                    """
                    INSERT OR REPLACE INTO observations_vintage (
                        location_id, obs_date, version_num, ingested_at_utc, run_id,
                        age_days_at_ingest, ingest_origin, completeness_regime,
                        change_type_summary, row_value_hash, PRECTOTCORR
                    ) VALUES ('HCMC', ?, 0, '2026-11-16T01:00:00Z', ?, 210, 'HISTORICAL_CSV_SEED',
                              'COMPLETE_ALL_35', 'INITIAL_SEED', 'tampered', 99.0)
                    """,
                    (obs_date, run_id),
                ),
                (
                    "REPLACE_INTO",
                    """
                    REPLACE INTO observations_vintage (
                        location_id, obs_date, version_num, ingested_at_utc, run_id,
                        age_days_at_ingest, ingest_origin, completeness_regime,
                        change_type_summary, row_value_hash, PRECTOTCORR
                    ) VALUES ('HCMC', ?, 0, '2026-11-16T01:00:00Z', ?, 210, 'HISTORICAL_CSV_SEED',
                              'COMPLETE_ALL_35', 'INITIAL_SEED', 'tampered', 99.0)
                    """,
                    (obs_date, run_id),
                ),
                (
                    "ON_CONFLICT_DO_UPDATE",
                    """
                    INSERT INTO observations_vintage (
                        location_id, obs_date, version_num, ingested_at_utc, run_id,
                        age_days_at_ingest, ingest_origin, completeness_regime,
                        change_type_summary, row_value_hash, PRECTOTCORR
                    ) VALUES ('HCMC', ?, 0, '2026-11-16T01:00:00Z', ?, 210, 'HISTORICAL_CSV_SEED',
                              'COMPLETE_ALL_35', 'INITIAL_SEED', 'tampered', 99.0)
                    ON CONFLICT(location_id, obs_date, version_num)
                    DO UPDATE SET PRECTOTCORR = excluded.PRECTOTCORR
                    """,
                    (obs_date, run_id),
                ),
            ],
            "forecasts_ledger": [
                ("UPDATE", "UPDATE forecasts_ledger SET pred_mm = 88.0 WHERE forecast_id = ?", (forecast_id,)),
                ("DELETE", "DELETE FROM forecasts_ledger WHERE forecast_id = ?", (forecast_id,)),
                (
                    "INSERT_OR_REPLACE",
                    """
                    INSERT OR REPLACE INTO forecasts_ledger (
                        forecast_id, issued_at_utc, issued_run_date, location_id,
                        model_id, model_kind, artifact_sha256, data_as_of, eval_split_regime,
                        horizon_day, target_date, temporal_status_at_issue, pred_mm,
                        input_window_start, input_window_end, input_window_hash,
                        input_vintage_manifest_json, input_provisional_days,
                        ceres_policy_applied, git_sha
                    ) VALUES (
                        ?, '2026-11-16T01:00:00Z', '2026-04-23', 'HCMC',
                        'SC-35', 'ML_SERVING', 'tampered', '2026-04-20', 'TEST',
                        1, '2026-04-21', 'future_forecast', 88.0,
                        '2026-03-22', '2026-04-20', 'winhash', '[]', 0,
                        'NONE_COMPLETE', 'testsha'
                    )
                    """,
                    (forecast_id,),
                ),
                (
                    "REPLACE_INTO",
                    """
                    REPLACE INTO forecasts_ledger (
                        forecast_id, issued_at_utc, issued_run_date, location_id,
                        model_id, model_kind, artifact_sha256, data_as_of, eval_split_regime,
                        horizon_day, target_date, temporal_status_at_issue, pred_mm,
                        input_window_start, input_window_end, input_window_hash,
                        input_vintage_manifest_json, input_provisional_days,
                        ceres_policy_applied, git_sha
                    ) VALUES (
                        'new_fc_id_unique_collision', '2026-11-16T01:00:00Z', '2026-04-23', 'HCMC',
                        'SC-35', 'ML_SERVING', 'tampered', '2026-04-20', 'TEST',
                        1, '2026-04-21', 'future_forecast', 88.0,
                        '2026-03-22', '2026-04-20', 'winhash', '[]', 0,
                        'NONE_COMPLETE', 'testsha'
                    )
                    """,
                    (),
                ),
                (
                    "ON_CONFLICT_DO_UPDATE",
                    """
                    INSERT INTO forecasts_ledger (
                        forecast_id, issued_at_utc, issued_run_date, location_id,
                        model_id, model_kind, artifact_sha256, data_as_of, eval_split_regime,
                        horizon_day, target_date, temporal_status_at_issue, pred_mm,
                        input_window_start, input_window_end, input_window_hash,
                        input_vintage_manifest_json, input_provisional_days,
                        ceres_policy_applied, git_sha
                    ) VALUES (
                        ?, '2026-11-16T01:00:00Z', '2026-04-23', 'HCMC',
                        'SC-35', 'ML_SERVING', 'tampered', '2026-04-20', 'TEST',
                        1, '2026-04-21', 'future_forecast', 88.0,
                        '2026-03-22', '2026-04-20', 'winhash', '[]', 0,
                        'NONE_COMPLETE', 'testsha'
                    )
                    ON CONFLICT(forecast_id) DO UPDATE SET pred_mm = excluded.pred_mm
                    """,
                    (forecast_id,),
                ),
            ],
            "ingestion_runs": [
                ("UPDATE", "UPDATE ingestion_runs SET status = 'FAILED' WHERE run_id = ?", (run_id,)),
                ("DELETE", "DELETE FROM ingestion_runs WHERE run_id = ?", (run_id,)),
                (
                    "INSERT_OR_REPLACE",
                    """
                    INSERT OR REPLACE INTO ingestion_runs (
                        run_id, run_type, started_at_utc, finished_at_utc, status,
                        location_id, scan_window_start, scan_window_end, git_sha
                    ) VALUES (?, 'historical_seed', '2026-11-16T01:00:00Z', '2026-11-16T01:00:01Z',
                              'FAILED', 'HCMC', '2026-04-20', '2026-04-20', 'tampered')
                    """,
                    (run_id,),
                ),
                (
                    "REPLACE_INTO",
                    """
                    REPLACE INTO ingestion_runs (
                        run_id, run_type, started_at_utc, finished_at_utc, status,
                        location_id, scan_window_start, scan_window_end, git_sha
                    ) VALUES (?, 'historical_seed', '2026-11-16T01:00:00Z', '2026-11-16T01:00:01Z',
                              'FAILED', 'HCMC', '2026-04-20', '2026-04-20', 'tampered')
                    """,
                    (run_id,),
                ),
                (
                    "ON_CONFLICT_DO_UPDATE",
                    """
                    INSERT INTO ingestion_runs (
                        run_id, run_type, started_at_utc, finished_at_utc, status,
                        location_id, scan_window_start, scan_window_end, git_sha
                    ) VALUES (?, 'historical_seed', '2026-11-16T01:00:00Z', '2026-11-16T01:00:01Z',
                              'FAILED', 'HCMC', '2026-04-20', '2026-04-20', 'tampered')
                    ON CONFLICT(run_id) DO UPDATE SET status = excluded.status
                    """,
                    (run_id,),
                ),
            ],
        }

        for table_name, table_attacks in attacks.items():
            for attack_name, sql, params in table_attacks:
                with pytest.raises(sqlite3.IntegrityError) as exc_info:
                    conn.execute(sql, params)
                assert "IMMUTABLE_VIOLATION" in str(exc_info.value), (
                    f"{table_name}.{attack_name} (recursive_triggers={recursive_triggers}) "
                    f"unexpected error: {exc_info.value}"
                )

        # Original rows remain intact
        assert conn.execute("SELECT COUNT(*) FROM observations_vintage").fetchone()[0] == 1
        assert conn.execute("SELECT PRECTOTCORR FROM observations_vintage").fetchone()[0] == 2.5
        assert conn.execute("SELECT COUNT(*) FROM forecasts_ledger").fetchone()[0] == 1
        assert conn.execute("SELECT pred_mm FROM forecasts_ledger").fetchone()[0] == 3.5
        assert conn.execute("SELECT COUNT(*) FROM ingestion_runs").fetchone()[0] == 1
        assert conn.execute("SELECT status FROM ingestion_runs").fetchone()[0] == "SUCCESS"
    finally:
        conn.close()


def test_single_transaction_fk_deferred_and_rollback_audit():
    """Verify deferred FK failure on COMMIT, clean rollback with 0 orphans, and FAILED audit log."""
    conn = connect_sqlite(":memory:", recursive_triggers=True)
    try:
        init_schema(conn)

        # 1) Orphan run_id must fail with FOREIGN KEY constraint failed at COMMIT
        conn.execute("BEGIN IMMEDIATE;")
        conn.execute(
            """
            INSERT INTO observations_vintage (
                location_id, obs_date, version_num, ingested_at_utc, run_id,
                age_days_at_ingest, ingest_origin,
                missing_count, missing_meteo_count, missing_flashflux_count, missing_ceres_count,
                completeness_regime, change_type_summary, row_value_hash
            ) VALUES ('HCMC', '2026-04-20', 0, '2026-11-16T00:00:00Z', 'nonexistent_run_id',
                      210, 'HISTORICAL_CSV_SEED', 1, 1, 0, 0,
                      'INCOMPLETE_METEO', 'INITIAL_SEED', 'h0')
            """
        )
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute("COMMIT;")
        conn.execute("ROLLBACK;")
        assert conn.execute("SELECT COUNT(*) FROM observations_vintage").fetchone()[0] == 0

        # 2) Simulated mid-batch failure rolls back partial rows and logs 1 FAILED ingestion_runs row
        dates = pd.date_range("2026-01-01", periods=100, freq="D")
        rows = []
        for idx, dt in enumerate(dates):
            row = {"DATE": dt.strftime("%Y-%m-%d"), **_sample_var_dict(rain=float(idx % 5))}
            rows.append(row)
        df_100 = pd.DataFrame(rows)

        with pytest.raises(RuntimeError, match="Simulated mid-batch failure"):
            seed_historical_csv_vintage_v0(conn, df=df_100, fail_after_rows=50)

        assert conn.execute("SELECT COUNT(*) FROM observations_vintage").fetchone()[0] == 0
        failed_runs = conn.execute("SELECT * FROM ingestion_runs").fetchall()
        assert len(failed_runs) == 1
        fr = dict(failed_runs[0])
        assert fr["status"] == "FAILED"
        assert fr["rows_inserted_v0"] == 0
        assert fr["rows_new_vintage"] == 0
        assert "Simulated mid-batch failure" in fr["error_message"]

        # 3) CHECK constraints reject PARTIAL_FAILURE and inconsistent is_degraded_input
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO ingestion_runs (
                    run_id, run_type, started_at_utc, finished_at_utc, status,
                    location_id, scan_window_start, scan_window_end, git_sha
                ) VALUES ('bad_status', 'manual_scan', 't0', 't1', 'PARTIAL_FAILURE',
                          'HCMC', '2026-04-01', '2026-04-02', 'sha')
                """
            )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO forecasts_ledger (
                    forecast_id, issued_at_utc, issued_run_date, location_id,
                    model_id, model_kind, artifact_sha256, data_as_of, eval_split_regime,
                    horizon_day, target_date, temporal_status_at_issue, pred_mm,
                    input_window_start, input_window_end, input_window_hash,
                    input_vintage_manifest_json, input_provisional_days,
                    flashflux_max_ffill_days, ceres_policy_applied, is_degraded_input, git_sha
                ) VALUES (
                    'bad_deg', 't0', '2026-04-23', 'HCMC',
                    'SC-35', 'ML_SERVING', 'sha', '2026-04-20', 'TEST',
                    1, '2026-04-21', 'future_forecast', 1.0,
                    '2026-03-22', '2026-04-20', 'wh', '[]', 0,
                    2, 'NONE_COMPLETE', 0, 'sha'
                )
                """
            )
    finally:
        conn.close()


def test_seed_historical_csv_vintage_v0_parity(tmp_path: Path):
    """Seed canonical historical CSV into SQLite and assert 0.0 numeric diff vs DataLoader().load_data()."""
    loader_df = DataLoader().load_data(rename_columns=True, exclude_year_2000=True)
    assert loader_df is not None and not loader_df.empty

    db_path = tmp_path / "vintage_seed_test.db"
    conn = connect_sqlite(db_path, recursive_triggers=True)
    try:
        init_schema(conn)
        run_id = seed_historical_csv_vintage_v0(conn, ingested_at_utc="2026-11-16T00:00:00Z")

        run_row = dict(conn.execute("SELECT * FROM ingestion_runs WHERE run_id = ?", (run_id,)).fetchone())
        assert run_row["status"] == "SUCCESS"
        assert run_row["rows_inserted_v0"] == len(loader_df)

        latest_df = read_latest_observations(
            conn,
            location_id=HCMC_LOCATION_ID,
            as_vietnamese=True,
            target_column_order=loader_df.columns,
        )
        assert list(latest_df.columns) == list(loader_df.columns)
        assert len(latest_df) == len(loader_df)
        assert (latest_df["Ngày"] == loader_df["Ngày"]).all()

        # Also assert 2001-01-01..2026-04-27 canonical slice matches identically
        slice_loader = loader_df[loader_df["Ngày"] <= "2026-04-27"]
        slice_db = latest_df[latest_df["Ngày"] <= "2026-04-27"]
        assert len(slice_db) == len(slice_loader)

        numeric_cols = [c for c in loader_df.columns if c != "Ngày"]
        max_diff = float(np.max(np.abs(latest_df[numeric_cols].to_numpy() - loader_df[numeric_cols].to_numpy())))
        assert max_diff == 0.0
    finally:
        conn.close()


def test_revision_as_of_hash_and_dual_scores():
    """Simulate v0 -> v1 NASA revision; verify As-Of input_window_hash reconstruction & v_forecast_scores."""
    conn = connect_sqlite(":memory:", recursive_triggers=True)
    try:
        init_schema(conn)

        # Build 31 days (2026-03-26 .. 2026-04-25): 30-day input window + 1 target date
        dates = pd.date_range("2026-03-26", "2026-04-25", freq="D")
        t0_ts = "2026-04-26T01:00:00Z"
        df_31 = pd.DataFrame(
            [{"DATE": d.strftime("%Y-%m-%d"), **_sample_var_dict(rain=2.0, temp=29.0)} for d in dates]
        )
        seed_historical_csv_vintage_v0(conn, df=df_31, ingested_at_utc=t0_ts)

        t1_ts = "2026-04-26T02:00:00Z"
        as_of_rows_t1 = query_observations_as_of(
            conn,
            location_id=HCMC_LOCATION_ID,
            input_window_start="2026-03-26",
            input_window_end="2026-04-24",
            issued_at_utc=t1_ts,
        )
        assert len(as_of_rows_t1) == 30
        hash_at_t1 = compute_input_window_hash(as_of_rows_t1)

        # Issue forecast for target_date = 2026-04-25 (pred_mm = 5.0; v0 PRECTOTCORR = 2.0 -> first error = 3.0)
        conn.execute("BEGIN IMMEDIATE;")
        conn.execute(
            """
            INSERT INTO forecasts_ledger (
                forecast_id, issued_at_utc, issued_run_date, location_id,
                model_id, model_kind, registry_alias, registry_model_version,
                artifact_sha256, feature_builder_sha256, data_as_of, eval_split_regime,
                horizon_day, target_date, temporal_status_at_issue, pred_mm,
                input_window_start, input_window_end, input_window_hash,
                input_vintage_manifest_json, input_provisional_days,
                ceres_policy_applied, git_sha
            ) VALUES (
                'fc_rev_001', ?, '2026-04-26', 'HCMC',
                'SC-35', 'ML_SERVING', 'serving', 1,
                'artsha', 'fbsha', '2026-04-24', 'TEST',
                1, '2026-04-25', 'past_unverified', 5.0,
                '2026-03-26', '2026-04-24', ?,
                '[]', 30,
                'NONE_COMPLETE', 'testsha'
            )
            """,
            (t1_ts, hash_at_t1),
        )
        conn.execute("COMMIT;")

        # At t2 > t1, NASA revises PRECTOTCORR on 2026-04-24 (in window) and 2026-04-25 (target) to 12.0 mm
        t2_ts = "2026-05-05T03:00:00Z"
        run_rev_id = "run_rev_002"
        rev_vars = _sample_var_dict(rain=12.0, temp=29.0)
        rev_comp = classify_completeness(rev_vars)
        rev_hash = compute_row_value_hash(rev_vars)

        conn.execute("BEGIN IMMEDIATE;")
        conn.execute(
            """
            INSERT INTO ingestion_runs (
                run_id, run_type, started_at_utc, finished_at_utc, status,
                location_id, scan_window_start, scan_window_end,
                rows_scanned, rows_inserted_v0, rows_new_vintage,
                rows_late_fill_only, rows_value_change, rows_unchanged, git_sha
            ) VALUES (?, 'manual_scan', ?, ?, 'SUCCESS', 'HCMC', '2026-04-24', '2026-04-25',
                      2, 0, 2, 0, 2, 0, 'testsha')
            """,
            (run_rev_id, t2_ts, t2_ts),
        )
        cols = [
            "location_id", "obs_date", "version_num", "ingested_at_utc", "run_id",
            "age_days_at_ingest", "ingest_origin",
            "missing_count", "missing_meteo_count", "missing_flashflux_count", "missing_ceres_count",
            "completeness_regime", "missing_vars_json",
            "change_type_summary", "late_fill_count", "late_fill_vars_json",
            "value_change_count", "value_change_details_json",
            "anomaly_count", "quality_flags_json", "row_value_hash",
            *ALL_35_VARS,
        ]
        for target_dt, age_d in (("2026-04-24", 11), ("2026-04-25", 10)):
            conn.execute(
                f"INSERT INTO observations_vintage ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
                [
                    HCMC_LOCATION_ID, target_dt, 1, t2_ts, run_rev_id,
                    age_d, "NASA_POWER_API",
                    rev_comp["missing_count"], rev_comp["missing_meteo_count"],
                    rev_comp["missing_flashflux_count"], rev_comp["missing_ceres_count"],
                    rev_comp["completeness_regime"], rev_comp["missing_vars_json"],
                    "VALUE_CHANGE_ONLY", 0, "[]",
                    1, json.dumps({"PRECTOTCORR": [2.0, 12.0]}),
                    0, "{}", rev_hash,
                    *[rev_vars[v] for v in ALL_35_VARS],
                ],
            )
        conn.execute("COMMIT;")

        # 1) Forecast row unchanged
        fc_row = dict(conn.execute("SELECT * FROM forecasts_ledger WHERE forecast_id = 'fc_rev_001'").fetchone())
        assert fc_row["pred_mm"] == 5.0
        assert fc_row["input_window_hash"] == hash_at_t1

        # 2) As-Of query at t1 reproduces exact historical hash; As-Of at t2 yields a different hash
        reconstructed_t1 = query_observations_as_of(
            conn,
            location_id=HCMC_LOCATION_ID,
            input_window_start="2026-03-26",
            input_window_end="2026-04-24",
            issued_at_utc=t1_ts,
        )
        assert compute_input_window_hash(reconstructed_t1) == hash_at_t1

        reconstructed_t2 = query_observations_as_of(
            conn,
            location_id=HCMC_LOCATION_ID,
            input_window_start="2026-03-26",
            input_window_end="2026-04-24",
            issued_at_utc=t2_ts,
        )
        assert compute_input_window_hash(reconstructed_t2) != hash_at_t1

        # 3) v_forecast_scores shows distinct first_seen vs latest errors and audit metadata
        score_row = dict(conn.execute("SELECT * FROM v_forecast_scores WHERE forecast_id = 'fc_rev_001'").fetchone())
        assert score_row["obs_first_version_num"] == 0
        assert score_row["obs_latest_version_num"] == 1
        assert score_row["obs_first_ingest_origin"] == "HISTORICAL_CSV_SEED"
        assert score_row["obs_first_age_days_at_ingest"] == 1
        assert score_row["obs_first_mm"] == 2.0
        assert score_row["obs_latest_mm"] == 12.0
        assert score_row["abs_error_vs_first_mm"] == pytest.approx(3.0)
        assert score_row["abs_error_vs_latest_mm"] == pytest.approx(7.0)
        assert score_row["obs_revision_delta_mm"] == pytest.approx(10.0)
    finally:
        conn.close()



def test_flashflux_incomplete_and_eval_split_regime():
    """Verify DDL V3.1 FLASHFLUX_INCOMPLETE precedence and deterministic derive_eval_split_regime."""
    # 1) Completeness precedence
    complete = _sample_var_dict()
    assert classify_completeness(complete)["completeness_regime"] == "COMPLETE_ALL_35"

    ceres_only_missing = {**complete, CERES_ONLY_8_VARS[0]: -999.0}
    c_res = classify_completeness(ceres_only_missing)
    assert c_res["completeness_regime"] == "MISSING_CERES_ONLY"
    assert c_res["missing_ceres_count"] == 1

    flash_missing = {**ceres_only_missing, FLASHFLUX_3_VARS[0]: float("nan")}
    f_res = classify_completeness(flash_missing)
    assert f_res["completeness_regime"] == "FLASHFLUX_INCOMPLETE"
    assert f_res["missing_flashflux_count"] == 1

    meteo_missing = {**flash_missing, "PRECTOTCORR": None}
    m_res = classify_completeness(meteo_missing)
    assert m_res["completeness_regime"] == "INCOMPLETE_METEO"
    assert m_res["missing_meteo_count"] == 1

    # 2) CHECK rejects legacy 'MISSING_ALL_RAD'
    conn = connect_sqlite(":memory:", recursive_triggers=True)
    try:
        init_schema(conn)
        conn.execute("BEGIN IMMEDIATE;")
        conn.execute(
            """
            INSERT INTO ingestion_runs (
                run_id, run_type, started_at_utc, finished_at_utc, status,
                location_id, scan_window_start, scan_window_end, git_sha
            ) VALUES ('r1', 'historical_seed', 't0', 't1', 'SUCCESS', 'HCMC', '2026-04-01', '2026-04-01', 'sha')
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO observations_vintage (
                    location_id, obs_date, version_num, ingested_at_utc, run_id,
                    age_days_at_ingest, ingest_origin,
                    missing_count, missing_meteo_count, missing_flashflux_count, missing_ceres_count,
                    completeness_regime, change_type_summary, row_value_hash
                ) VALUES ('HCMC', '2026-04-01', 0, 't0', 'r1', 10, 'HISTORICAL_CSV_SEED',
                          1, 0, 1, 0, 'MISSING_ALL_RAD', 'INITIAL_SEED', 'h')
                """
            )
        conn.execute("ROLLBACK;")
    finally:
        conn.close()

    # 3) derive_eval_split_regime canonical boundaries
    assert derive_eval_split_regime("2015-06-01") == "DEV_TRAIN"
    assert derive_eval_split_regime("2018-04-30") == "DEV_TRAIN"
    assert derive_eval_split_regime("2018-05-01") == "VALIDATION"
    assert derive_eval_split_regime("2020-04-30") == "VALIDATION"
    assert derive_eval_split_regime("2020-05-01") == "TEST"
    assert derive_eval_split_regime("2026-04-27") == "TEST"
    assert derive_eval_split_regime("2026-04-28") == "LIVE"


def test_insert_forecasts_idempotent_skips_duplicates(caplog):
    """Batch of 3 existing + 4 new forecasts inserts 4, skips 3 with log, while raw INSERT OR REPLACE fails."""
    import logging

    caplog.set_level(logging.INFO, logger="src.data.vintage_store")
    conn = connect_sqlite(":memory:", recursive_triggers=True)
    try:
        init_schema(conn)

        def _make_fc(horizon: int, data_as_of: str = "2026-05-01") -> dict:
            tgt = (pd.to_datetime(data_as_of) + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d")
            return {
                "issued_at_utc": "2026-05-04T00:00:00Z",
                "issued_run_date": "2026-05-04",
                "location_id": "HCMC",
                "model_id": "SC-35",
                "model_kind": "ML_SERVING",
                "registry_alias": "serving",
                "registry_model_version": 1,
                "artifact_sha256": "artsha",
                "feature_builder_sha256": "fbsha",
                "data_as_of": data_as_of,
                "eval_split_regime": "DEV_TRAIN",  # Intentionally wrong manual string; must be overridden to LIVE
                "horizon_day": horizon,
                "target_date": tgt,
                "temporal_status_at_issue": "future_forecast",
                "pred_mm": 1.25 * horizon,
                "input_window_start": "2026-04-02",
                "input_window_end": data_as_of,
                "input_window_hash": "winhash",
                "input_vintage_manifest_json": "[]",
                "input_provisional_days": 30,
                "ceres_policy_applied": "NONE_COMPLETE",
                "git_sha": "testsha",
            }

        ins1, skip1 = insert_forecasts_idempotent(conn, [_make_fc(h) for h in (1, 2, 3)])
        assert (ins1, skip1) == (3, 0)

        # Mixed batch: 3 duplicates (h=1,2,3) + 4 new (h=4,5,6,7)
        caplog.clear()
        ins2, skip2 = insert_forecasts_idempotent(conn, [_make_fc(h) for h in range(1, 8)])
        assert (ins2, skip2) == (4, 3)
        assert "Skipping existing forecast key" in caplog.text

        rows = conn.execute("SELECT horizon_day, eval_split_regime FROM forecasts_ledger ORDER BY horizon_day").fetchall()
        assert [r["horizon_day"] for r in rows] == [1, 2, 3, 4, 5, 6, 7]
        assert all(r["eval_split_regime"] == "LIVE" for r in rows)

        # Raw INSERT OR REPLACE must still be blocked by trigger
        with pytest.raises(sqlite3.IntegrityError, match="IMMUTABLE_VIOLATION"):
            conn.execute(
                """
                INSERT OR REPLACE INTO forecasts_ledger (
                    forecast_id, issued_at_utc, issued_run_date, location_id,
                    model_id, model_kind, artifact_sha256, data_as_of, eval_split_regime,
                    horizon_day, target_date, temporal_status_at_issue, pred_mm,
                    input_window_start, input_window_end, input_window_hash,
                    input_vintage_manifest_json, input_provisional_days,
                    ceres_policy_applied, git_sha
                ) VALUES (
                    'HCMC_SC-35_2026-05-01_h1', 't', '2026-05-04', 'HCMC',
                    'SC-35', 'ML_SERVING', 'x', '2026-05-01', 'LIVE',
                    1, '2026-05-02', 'future_forecast', 99.0,
                    '2026-04-02', '2026-05-01', 'w', '[]', 30, 'NONE_COMPLETE', 's'
                )
                """
            )
    finally:
        conn.close()


def test_read_only_wal_sqlite_and_immutable_fallback(tmp_path: Path, monkeypatch):
    """Verify connect_sqlite(read_only=True) works on WAL DBs and falls back to immutable=1 when -shm fails."""
    db_path = tmp_path / "wal_ro_test.db"
    w_conn = connect_sqlite(db_path, read_only=False, recursive_triggers=True)
    try:
        init_schema(w_conn)
        assert w_conn.execute("PRAGMA journal_mode;").fetchone()[0].lower() == "wal"
        dates = pd.date_range("2026-03-27", periods=35, freq="D")
        df_35 = pd.DataFrame([{"DATE": d.strftime("%Y-%m-%d"), **_sample_var_dict(rain=3.0)} for d in dates])
        seed_historical_csv_vintage_v0(w_conn, df=df_35)
    finally:
        w_conn.close()

    # 1) Direct read_only=True
    r_conn = connect_sqlite(db_path, read_only=True)
    try:
        assert r_conn.execute("PRAGMA query_only;").fetchone()[0] == 1
        cnt = r_conn.execute("SELECT COUNT(*) FROM v_observations_latest").fetchone()[0]
        assert cnt == 35
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            r_conn.execute("UPDATE locations SET display_name = 'Tamper'")
    finally:
        r_conn.close()

    # 2) Simulate read-only container mount blocking WAL -shm -> must fallback to ?mode=ro&immutable=1
    real_connect = sqlite3.connect
    calls: list[str] = []

    def _mock_connect(target, *args, **kwargs):
        calls.append(str(target))
        if isinstance(target, str) and target.endswith("?mode=ro"):
            raise sqlite3.OperationalError("attempt to write a readonly database")
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _mock_connect)
    fb_conn = connect_sqlite(db_path, read_only=True)
    try:
        assert any(c.endswith("?mode=ro") for c in calls)
        assert any(c.endswith("?mode=ro&immutable=1") for c in calls)
        assert fb_conn.execute("SELECT COUNT(*) FROM v_observations_latest").fetchone()[0] == 35
    finally:
        fb_conn.close()


def test_mock_nasa_api_revision_and_as_of_scores(tmp_path: Path):
    """Simulate 2 NASA POWER API scans: Scan 1 seed + forecast; Scan 2 LATE_FILL + VALUE_CHANGE revisions."""
    db_path = tmp_path / "mock_nasa_ingest.db"
    conn = connect_sqlite(db_path, read_only=False, recursive_triggers=True)
    try:
        init_schema(conn)
        dates = pd.date_range("2026-03-26", "2026-04-29", freq="D")

        def _build_nasa_json(*, ceres_fill_on_apr28: float, rain_on_apr29: float) -> dict:
            param_map: dict[str, dict[str, float]] = {var: {} for var in ALL_35_VARS}
            for idx, d in enumerate(dates):
                ymd = d.strftime("%Y%m%d")
                base = _sample_var_dict(rain=2.0 + (idx % 3), temp=29.0)
                for var in ALL_35_VARS:
                    val = base[var]
                    if d.strftime("%Y-%m-%d") == "2026-04-28" and var in CERES_ONLY_8_VARS:
                        val = ceres_fill_on_apr28
                    if d.strftime("%Y-%m-%d") == "2026-04-29" and var == "PRECTOTCORR":
                        val = rain_on_apr29
                    param_map[var][ymd] = val
            return {
                "header": {"api": {"version": "v2.10.3"}, "sources": ["MERRA2", "GEOSIT", "SYN1DEG", "FLASHFLUX"], "fill_value": -999.0},
                "properties": {"parameter": param_map},
            }

        # Scan 1: 2026-04-28 missing all 8 CERES (-999.0); 2026-04-29 PRECTOTCORR = 1.5
        t1_scan = "2026-04-30T01:00:00Z"
        s1 = ingest_nasa_power_window(
            conn,
            start_date="2026-03-26",
            end_date="2026-04-29",
            fetcher=lambda **kwargs: _build_nasa_json(ceres_fill_on_apr28=-999.0, rain_on_apr29=1.5),
            ingested_at_utc=t1_scan,
        )
        assert s1["rows_scanned"] == 35
        assert s1["rows_inserted_v0"] == 35
        assert s1["rows_new_vintage"] == 0

        apr28_v0 = dict(conn.execute("SELECT * FROM observations_vintage WHERE obs_date = '2026-04-28'").fetchone())
        assert apr28_v0["completeness_regime"] == "MISSING_CERES_ONLY"
        assert apr28_v0["missing_ceres_count"] == 8

        # Emit forecast at t1_issue using window 2026-03-26..2026-04-24 targeting 2026-04-29? Or window ending 2026-04-28!
        t1_issue = "2026-04-30T02:00:00Z"
        win_t1 = query_observations_as_of(
            conn,
            location_id="HCMC",
            input_window_start="2026-03-30",
            input_window_end="2026-04-28",
            issued_at_utc=t1_issue,
        )
        assert len(win_t1) == 30
        hash_t1 = compute_input_window_hash(win_t1)

        ins_fc, skip_fc = insert_forecasts_idempotent(
            conn,
            [
                {
                    "forecast_id": "fc_nasa_mock_001",
                    "issued_at_utc": t1_issue,
                    "issued_run_date": "2026-04-30",
                    "location_id": "HCMC",
                    "model_id": "SC-35",
                    "model_kind": "ML_SERVING",
                    "registry_alias": "serving",
                    "registry_model_version": 1,
                    "artifact_sha256": "artsha",
                    "feature_builder_sha256": "fbsha",
                    "data_as_of": "2026-04-28",
                    "horizon_day": 1,
                    "target_date": "2026-04-29",
                    "temporal_status_at_issue": "past_unverified",
                    "pred_mm": 6.0,
                    "input_window_start": "2026-03-30",
                    "input_window_end": "2026-04-28",
                    "input_window_hash": hash_t1,
                    "input_vintage_manifest_json": "[]",
                    "input_provisional_days": 30,
                    "ceres_policy_applied": "NONE_COMPLETE",
                    "git_sha": "testsha",
                }
            ],
        )
        assert (ins_fc, skip_fc) == (1, 0)

        # Scan 2: 2026-04-28 fills 8 CERES vars (-999.0 -> 4.5); 2026-04-29 PRECTOTCORR changes 1.5 -> 10.0
        t2_scan = "2026-05-15T01:00:00Z"
        s2 = ingest_nasa_power_window(
            conn,
            start_date="2026-03-26",
            end_date="2026-04-29",
            fetcher=lambda **kwargs: _build_nasa_json(ceres_fill_on_apr28=4.5, rain_on_apr29=10.0),
            ingested_at_utc=t2_scan,
        )
        assert s2["rows_scanned"] == 35
        assert s2["rows_inserted_v0"] == 0
        assert s2["rows_new_vintage"] == 2
        assert s2["rows_late_fill_only"] == 1
        assert s2["rows_value_change"] == 1
        assert s2["rows_unchanged"] == 33

        # Verify v1 rows and As-Of reconstruction
        apr28_v1 = dict(
            conn.execute("SELECT * FROM observations_vintage WHERE obs_date = '2026-04-28' AND version_num = 1").fetchone()
        )
        assert apr28_v1["change_type_summary"] == "LATE_FILL_ONLY"
        assert apr28_v1["completeness_regime"] == "COMPLETE_ALL_35"
        assert apr28_v1["late_fill_count"] == 8

        apr29_v1 = dict(
            conn.execute("SELECT * FROM observations_vintage WHERE obs_date = '2026-04-29' AND version_num = 1").fetchone()
        )
        assert apr29_v1["change_type_summary"] == "VALUE_CHANGE_ONLY"
        assert apr29_v1["value_change_count"] == 1

        # Forecast remains immutable & As-Of reproduces original hash_t1
        fc = dict(conn.execute("SELECT * FROM forecasts_ledger WHERE forecast_id = 'fc_nasa_mock_001'").fetchone())
        assert fc["pred_mm"] == 6.0
        assert fc["input_window_hash"] == hash_t1
        reconstructed_t1 = query_observations_as_of(
            conn,
            location_id="HCMC",
            input_window_start="2026-03-30",
            input_window_end="2026-04-28",
            issued_at_utc=t1_issue,
        )
        assert compute_input_window_hash(reconstructed_t1) == hash_t1

        # v_forecast_scores reflects both abs_error_vs_first_mm (|6.0 - 1.5| = 4.5) and abs_error_vs_latest_mm (|6.0 - 10.0| = 4.0)
        score = dict(conn.execute("SELECT * FROM v_forecast_scores WHERE forecast_id = 'fc_nasa_mock_001'").fetchone())
        assert score["obs_first_mm"] == pytest.approx(1.5)
        assert score["obs_latest_mm"] == pytest.approx(10.0)
        assert score["abs_error_vs_first_mm"] == pytest.approx(4.5)
        assert score["abs_error_vs_latest_mm"] == pytest.approx(4.0)
    finally:
        conn.close()
