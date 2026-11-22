"""Round 11 Step 4 (Part E) Tests: Idempotent Automated Pipeline CLI & Zero-UPDATE Ledger Contract."""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlops.automated_pipeline import (  # noqa: E402
    main as pipeline_main,
    run_ingest_window_step,
    run_issue_forecasts_step,
    run_probe_headers_step,
    run_refresh_cache_step,
)
from src.data.vintage_store import (  # noqa: E402
    ALL_35_VARS,
    compute_row_value_hash,
    connect_sqlite,
    init_schema,
)


def _seed_40_complete_days(db_path: Path, start_iso: str = "2026-04-20", n_days: int = 40) -> str:
    conn = connect_sqlite(db_path, read_only=False)
    try:
        init_schema(conn)
        conn.execute(
            """
            INSERT INTO ingestion_runs (
                run_id, run_type, started_at_utc, finished_at_utc, status, location_id,
                scan_window_start, scan_window_end, rows_scanned, rows_inserted_v0,
                git_sha, config_sha256
            ) VALUES (
                'seed_40d', 'manual_scan', '2026-06-01T00:00:00Z', '2026-06-01T00:01:00Z',
                'SUCCESS', 'HCMC', '2018-04-30', '2026-05-29', 41, 41, 'sha', 'cfg'
            )
            """
        )
        col_list = ", ".join(ALL_35_VARS)
        ph = ", ".join(["?"] * len(ALL_35_VARS))
        dev_row = {col: 1.5 for col in ALL_35_VARS}
        conn.execute(
            f"""
            INSERT INTO observations_vintage (
                location_id, obs_date, version_num, ingested_at_utc, run_id,
                age_days_at_ingest, ingest_origin, missing_count, missing_meteo_count,
                missing_flashflux_count, missing_ceres_count, completeness_regime,
                missing_vars_json, change_type_summary, late_fill_count, late_fill_vars_json,
                value_change_count, value_change_details_json, anomaly_count, quality_flags_json,
                row_value_hash, {col_list}
            ) VALUES (
                'HCMC', '2018-04-30', 0, '2018-05-01T00:00:00Z', 'seed_40d',
                1, 'HISTORICAL_CSV_SEED', 0, 0, 0, 0, 'COMPLETE_ALL_35',
                '[]', 'INITIAL_SEED', 0, '[]', 0, '{{}}', 0, '{{}}', ?, {ph}
            )
            """,
            (compute_row_value_hash(dev_row), *(dev_row[c] for c in ALL_35_VARS)),
        )
        d0 = date.fromisoformat(start_iso)
        last_iso = start_iso
        for i in range(n_days):
            d_iso = (d0 + timedelta(days=i)).isoformat()
            last_iso = d_iso
            row_dict = {col: float(1.0 + (i % 5) * 0.5) for col in ALL_35_VARS}
            rhash = compute_row_value_hash(row_dict)
            vals = tuple(row_dict[col] for col in ALL_35_VARS)
            conn.execute(
                f"""
                INSERT INTO observations_vintage (
                    location_id, obs_date, version_num, ingested_at_utc, run_id,
                    age_days_at_ingest, ingest_origin, missing_count, missing_meteo_count,
                    missing_flashflux_count, missing_ceres_count, completeness_regime,
                    missing_vars_json, change_type_summary, late_fill_count, late_fill_vars_json,
                    value_change_count, value_change_details_json, anomaly_count, quality_flags_json,
                    row_value_hash, {col_list}
                ) VALUES (
                    'HCMC', ?, 0, '2026-06-01T00:00:30Z', 'seed_40d',
                    10, 'NASA_POWER_API', 0, 0, 0, 0, 'COMPLETE_ALL_35',
                    '[]', 'INITIAL_NRT', 0, '[]', 0, '{{}}', 0, '{{}}', ?, {ph}
                )
                """,
                (d_iso, rhash, *vals),
            )
        conn.commit()
        return last_iso
    finally:
        conn.close()


def test_issue_forecasts_idempotent_skips_repeat_runs_and_executes_zero_updates(tmp_path: Path):
    db_path = tmp_path / "pipeline_fc.db"
    last_iso = _seed_40_complete_days(db_path, start_iso="2026-04-20", n_days=42)

    sql_statements: list[str] = []
    conn = connect_sqlite(db_path, read_only=False)
    conn.set_trace_callback(sql_statements.append)
    try:
        first = run_issue_forecasts_step(conn=conn, post_test_start=last_iso, end_date=last_iso)
        assert first["status"] == "SUCCESS"
        assert first["anchors_processed"] == 1
        assert first["rows_inserted"] == 35
        assert first["updates_executed"] == 0

        # Ensure config_sha256 stamped and zero SQL UPDATE statements executed
        assert not any(stmt.lstrip().upper().startswith("UPDATE") for stmt in sql_statements)
        sha_cnt = conn.execute(
            "SELECT COUNT(*) FROM forecasts_ledger WHERE config_sha256 IS NOT NULL AND config_sha256 != ''"
        ).fetchone()[0]
        assert sha_cnt == 35

        # Second identical run MUST report SKIPPED and insert 0 rows
        second = run_issue_forecasts_step(conn=conn, post_test_start=last_iso, end_date=last_iso)
        assert second["status"] == "SKIPPED"
        assert second["reason"] == "no_new_complete_anchor"
        assert second["rows_inserted"] == 0
        assert second["updates_executed"] == 0
    finally:
        conn.close()


def test_ingest_window_and_probe_headers_idempotent_and_stamp_config_sha256(tmp_path: Path):
    db_path = tmp_path / "pipeline_ing.db"
    _seed_40_complete_days(db_path, start_iso="2026-04-20", n_days=40)

    def _mock_nasa_window(*, start_date: str, end_date: str, latitude: float, longitude: float, **kwargs):
        del latitude, longitude, kwargs
        d = date.fromisoformat(start_date)
        end_d = date.fromisoformat(end_date)
        param_map = {v: {} for v in ALL_35_VARS}
        while d <= end_d:
            ymd = d.strftime("%Y%m%d")
            for v in ALL_35_VARS:
                param_map[v][ymd] = 2.5
            d += timedelta(days=1)
        return {
            "header": {"api": {"version": "v2.10.3"}, "sources": ["MERRA2"]},
            "properties": {"parameter": param_map},
        }

    def _mock_probe_day(obs_date: str, parameters: str):
        is_early = obs_date <= "2026-08-30"
        return {
            "date": obs_date,
            "api_version": "v2.10.3",
            "sources": ["MERRA2", "CERES"] if is_early else ["GEOSIT"],
            "values": {
                p: (1.0 if p == "PRECTOTCORR" else (2.0 if is_early else None))
                for p in parameters.split(",")
            },
        }

    conn = connect_sqlite(db_path, read_only=False)
    try:
        res1 = run_ingest_window_step(
            conn=conn, start_date="2026-06-01", end_date="2026-06-03", fetcher=_mock_nasa_window
        )
        assert res1["status"] == "SUCCESS"
        assert res1["rows_inserted_v0"] == 3

        res2 = run_ingest_window_step(
            conn=conn, start_date="2026-06-01", end_date="2026-06-03", fetcher=_mock_nasa_window
        )
        assert res2["status"] == "SUCCESS"
        assert res2["rows_inserted_v0"] == 0
        assert res2["rows_new_vintage"] == 0
        assert res2["rows_unchanged"] == 3

        probe = run_probe_headers_step(
            conn=conn,
            start_date="2026-08-20",
            end_date="2026-09-10",
            experiment_today="2026-11-20",
            fetcher=_mock_probe_day,
            window_fetcher=_mock_nasa_window,
        )
        assert probe["status"] == "SUCCESS"
        assert probe["run_id"] is not None
        assert len(probe["config_sha256"]) == 64
        assert probe["post_probe_ingest"] is not None
        assert probe["post_probe_ingest"]["status"] == "SUCCESS"
    finally:
        conn.close()


def test_automated_pipeline_cli_dry_run_and_refresh_cache(tmp_path: Path, capsys):
    db_path = tmp_path / "pipeline_cli.db"
    last_iso = _seed_40_complete_days(db_path, start_iso="2026-04-20", n_days=40)
    out_json = tmp_path / "out.json"

    rc = pipeline_main(
        [
            "--backend",
            "local",
            "--db-path",
            str(db_path),
            "--json-out",
            str(out_json),
            "issue-forecasts",
            "--post-test-start",
            last_iso,
            "--end-date",
            last_iso,
            "--dry-run",
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["status"] == "DRY_RUN"

    ref = run_refresh_cache_step(backend="local", db_path=db_path)
    assert ref["status"] == "SUCCESS"
    assert ref["state_key"][0] == "local"


def test_automated_pipeline_deferred_on_quota_blocked(tmp_path: Path, monkeypatch, capsys):
    """Verify HTTP 402/403/429 BLOCKED defers CLI pipeline cleanly with DEFERRED_QUOTA_BLOCKED and redacted secrets."""
    import requests

    secret_tok = "tok_pipeline_secret_555666"

    class _BlockedResp:
        status_code = 402
        text = f'{{"error":"BLOCKED: quota exceeded Bearer {secret_tok}"}}'

        @staticmethod
        def json():
            return {"error": "BLOCKED: quota exceeded"}

    monkeypatch.setenv("TURSO_DB_URL", "libsql://blocked-cli.turso.io")
    monkeypatch.setenv("TURSO_DB_TOKEN", secret_tok)
    monkeypatch.setattr(
        requests.Session,
        "post",
        lambda sess, url, json=None, timeout=None: _BlockedResp(),
    )

    out_json = tmp_path / "blocked.json"
    rc = pipeline_main(
        [
            "--backend",
            "turso_remote",
            "--json-out",
            str(out_json),
            "issue-forecasts",
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["status"] == "DEFERRED_QUOTA_BLOCKED"
    assert payload["degraded_reason"] == "TURSO_REMOTE_BLOCKED"
    assert secret_tok not in json.dumps(payload)
    assert secret_tok not in capsys.readouterr().out


def test_issue_forecasts_latest_only_issues_only_newest_anchor_and_skips_repeat(tmp_path: Path):
    """Verify `--latest-only` issues ONLY the newest complete anchor (never backfilling prior unissued dates) and skips on repeat."""
    db_path = tmp_path / "pipeline_latest_only.db"
    # Seed 2026-10-15..2026-11-17 (34 days) -> 3 meteo-complete anchors in [2026-11-15, 2026-11-17]
    last_iso = _seed_40_complete_days(db_path, start_iso="2026-10-15", n_days=34)
    assert last_iso == "2026-11-17"

    out_json = tmp_path / "latest_only_cli.json"
    rc = pipeline_main(
        [
            "--backend",
            "local",
            "--db-path",
            str(db_path),
            "--json-out",
            str(out_json),
            "issue-forecasts",
            "--post-test-start",
            "2026-11-15",
            "--end-date",
            "2026-11-17",
            "--latest-only",
        ]
    )
    assert rc == 0
    cli_res = json.loads(out_json.read_text(encoding="utf-8"))
    assert cli_res["status"] == "SUCCESS"
    assert cli_res["anchors_processed"] == 1
    assert cli_res["rows_inserted"] == 35
    assert cli_res["latest_issued_anchor"] == "2026-11-17"

    conn = connect_sqlite(db_path, read_only=False)
    try:
        counts = {
            d: int(
                conn.execute(
                    "SELECT COUNT(*) FROM forecasts_ledger WHERE data_as_of = ?",
                    (d,),
                ).fetchone()[0]
            )
            for d in ("2026-11-15", "2026-11-16", "2026-11-17")
        }
        assert counts == {"2026-11-15": 0, "2026-11-16": 0, "2026-11-17": 35}

        # Second call with latest_only=True MUST SKIP and leave prior unissued dates at 0
        repeat = run_issue_forecasts_step(
            conn=conn,
            post_test_start="2026-11-15",
            end_date="2026-11-17",
            latest_only=True,
        )
        assert repeat["status"] == "SKIPPED"
        assert repeat["reason"] == "no_new_complete_anchor"
        assert repeat["latest_complete_anchor"] == "2026-11-17"
        assert repeat["rows_inserted"] == 0
        assert (
            int(conn.execute("SELECT COUNT(*) FROM forecasts_ledger WHERE data_as_of = '2026-11-15'").fetchone()[0])
            == 0
        )

        # Empty anchor window with latest_only=True MUST SKIP cleanly without IndexError
        empty_res = run_issue_forecasts_step(
            conn=conn,
            post_test_start="2026-12-01",
            end_date="2026-12-05",
            latest_only=True,
        )
        assert empty_res["status"] == "SKIPPED"
        assert empty_res["reason"] == "no_complete_anchor_in_window"
    finally:
        conn.close()


def test_ingest_window_days_cli_validation_and_audit_db_footprint_bounds(tmp_path: Path, monkeypatch):
    """Verify `--days` CLI bounds enforcement and B-tree footprint auditor + truncated-header guards."""
    import pytest
    from scripts.audit.audit_db_storage_footprint import (
        _calc_overflow_first_page,
        _read_varint,
        inspect_sqlite_footprint,
        walk_btree_pages,
    )

    db_path = tmp_path / "days_and_audit.db"
    _seed_40_complete_days(db_path, start_iso="2026-10-15", n_days=34)

    # `--days 0` or `> 1000` must fail cleanly at CLI parse time
    with pytest.raises(SystemExit) as exc_bad:
        pipeline_main(["--backend", "local", "--db-path", str(db_path), "ingest-window", "--days", "0"])
    assert exc_bad.value.code == 2

    # `--days 14` alias resolves cleanly via patched ingest helper
    captured: list[int] = []

    def _fake_ingest(conn, *, lookback_days=0, **kwargs):
        captured.append(int(lookback_days))
        return {"status": "SUCCESS", "run_id": "mock_14d", "rows_scanned": 14}

    monkeypatch.setattr("scripts.mlops.automated_pipeline.ingest_nasa_power_window", _fake_ingest)
    rc = pipeline_main(["--backend", "local", "--db-path", str(db_path), "ingest-window", "--days", "14"])
    assert rc == 0
    assert captured == [14]

    # Footprint inspector returns exact page accounting and column sizes
    fp = inspect_sqlite_footprint(db_path)
    assert fp["active_btree_pages_walked"] == fp["expected_active_pages"]
    assert fp["table_row_counts"]["observations_vintage"] == 35

    # Binary bounds guards reject truncated / invalid inputs cleanly
    bad_file = tmp_path / "not_sqlite.db"
    bad_file.write_bytes(b"short")
    with pytest.raises(ValueError, match="hydrated SQLite"):
        inspect_sqlite_footprint(bad_file)
    with pytest.raises(ValueError, match="Truncated SQLite varint"):
        _read_varint(b"\x81", 0)
    with pytest.raises(ValueError, match="usable_size"):
        _calc_overflow_first_page(b"\x0d" + b"\x00" * 30, 0, 0x0D, 16)
    with pytest.raises(ValueError, match="Invalid or truncated SQLite"):
        walk_btree_pages(b"bad", 1, 4096, 4096)
