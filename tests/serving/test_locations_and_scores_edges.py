"""Edge-case & filter-consistency tests for /locations, /scores, and vintage_store helpers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import DEFAULT_DEMO_VINTAGE_DB_PATH, app
from src.data.vintage_store import (
    connect_sqlite,
    get_latest_precip_and_partial_explanation,
    get_locations,
    init_schema,
)


def _insert_obs_row(
    conn: sqlite3.Connection,
    obs_date: str,
    *,
    missing_meteo: int = 0,
    missing_ff: int = 0,
    missing_ceres: int = 0,
    precip: float | None = 1.5,
    missing_vars_raw: str | None = None,
) -> None:
    missing_total = missing_meteo + missing_ff + missing_ceres
    if missing_meteo > 0:
        regime = "INCOMPLETE_METEO"
    elif missing_ff > 0:
        regime = "FLASHFLUX_INCOMPLETE"
    elif missing_ceres > 0:
        regime = "MISSING_CERES_ONLY"
    else:
        regime = "COMPLETE_ALL_35"

    if missing_vars_raw is None:
        missing_vars_raw = json.dumps(["T2M_MAX", "WS2M_MIN"] if missing_meteo else [])

    conn.execute(
        """
        INSERT INTO observations_vintage (
            location_id, obs_date, version_num, ingested_at_utc, run_id,
            age_days_at_ingest, ingest_origin, missing_count, missing_meteo_count,
            missing_flashflux_count, missing_ceres_count, completeness_regime,
            missing_vars_json, change_type_summary, row_value_hash, PRECTOTCORR
        ) VALUES ('HCMC', ?, 0, '2026-11-18T00:00:00Z', 'run_1', 1, 'NASA_POWER_API',
                  ?, ?, ?, ?, ?, ?, 'INITIAL_NRT', 'hash0', ?)
        """,
        (
            obs_date,
            missing_total,
            missing_meteo,
            missing_ff,
            missing_ceres,
            regime,
            missing_vars_raw,
            precip,
        ),
    )


def test_locations_503_when_db_missing(tmp_path: Path):
    missing_db = tmp_path / "does_not_exist.db"
    with patch("src.api.main._resolve_active_db_info", return_value=(missing_db, "local_or_production_db")):
        with TestClient(app) as client:
            resp = client.get("/locations")
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "SERVER_DB_DATA_ERROR"


def test_scores_is_prospective_filter_propagates_to_block_sensitivity():
    with TestClient(app) as client:
        all_resp = client.get("/scores?eval_split_regime=POST_TEST")
        assert all_resp.status_code == 200
        all_data = all_resp.json()
        assert all_data["block_length_sensitivity"]["7"]["n_anchor_dates"] == all_data["n_origins"]

        # Bundled demo DB has is_prospective=0 retrospective POST_TEST rows, so is_prospective=1 returns 0 rows
        # and block_length_sensitivity must reflect that filter rather than computing over unfiltered rows.
        prosp_resp = client.get("/scores?eval_split_regime=POST_TEST&is_prospective=1")
        assert prosp_resp.status_code == 200
        prosp_data = prosp_resp.json()
        assert prosp_data["total_scored_rows"] == 0
        assert prosp_data["block_length_sensitivity"] == {}


def test_vintage_store_locations_active_only_and_partial_explanation_branches(tmp_path: Path):
    db_path = tmp_path / "test_edges.db"
    conn = connect_sqlite(db_path, read_only=False)
    try:
        init_schema(conn, seed_hcmc_location=True)
        conn.execute(
            """
            INSERT INTO locations (
                location_id, display_name, latitude, longitude,
                is_active, created_at_utc
            ) VALUES ('HANOI', 'Ha Noi', 21.0285, 105.8542, 0, '2026-11-18T00:00:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO ingestion_runs (
                run_id, run_type, started_at_utc, finished_at_utc, status,
                location_id, scan_window_start, scan_window_end, git_sha
            ) VALUES (
                'run_1', 'manual_scan', '2026-11-18T00:00:00Z', '2026-11-18T00:01:00Z',
                'SUCCESS', 'HCMC', '2026-08-01', '2026-11-18', 'deadbeef'
            )
            """
        )
        conn.commit()

        assert len(get_locations(conn, active_only=False)) == 2
        active_locs = get_locations(conn, active_only=True)
        assert [loc["location_id"] for loc in active_locs] == ["HCMC"]

        # Branch 1: no non-null precip rows yet
        latest_date, expl = get_latest_precip_and_partial_explanation(conn)
        assert latest_date is None
        assert expl is None

        # Branch 2: complete 35-variable row
        _insert_obs_row(conn, "2026-08-30", missing_meteo=0, missing_ff=0, missing_ceres=0, precip=4.2)
        conn.commit()
        latest_date, expl = get_latest_precip_and_partial_explanation(conn)
        assert latest_date == "2026-08-30"
        assert expl is not None and "all 35 meteorological and radiation variables" in expl

        # Branch 3: meteo complete but radiation lagging + corrupt missing_vars_json fallback
        _insert_obs_row(
            conn,
            "2026-11-10",
            missing_meteo=0,
            missing_ff=0,
            missing_ceres=8,
            precip=1.0,
            missing_vars_raw="{bad_json",
        )
        conn.commit()
        latest_date, expl = get_latest_precip_and_partial_explanation(conn)
        assert latest_date == "2026-11-10"
        assert expl is not None and "pending radiation variables: FlashFlux missing=0, CERES missing=8" in expl

        # Branch 4a: partial meteorology row with corrupt JSON falls back to count summary
        _insert_obs_row(
            conn,
            "2026-11-14",
            missing_meteo=5,
            missing_ff=3,
            missing_ceres=8,
            precip=2.1,
            missing_vars_raw="{bad_json",
        )
        conn.commit()
        latest_date, expl = get_latest_precip_and_partial_explanation(conn)
        assert latest_date == "2026-11-14"
        assert expl is not None and "5 variable(s)" in expl

        # Branch 4b: partial meteorology row with valid missing_vars_json lists variable names
        _insert_obs_row(
            conn,
            "2026-11-15",
            missing_meteo=5,
            missing_ff=3,
            missing_ceres=8,
            precip=3.32,
            missing_vars_raw=json.dumps(["T2M_MAX", "WS2M_MIN"]),
        )
        conn.commit()
        latest_date, expl = get_latest_precip_and_partial_explanation(conn, meteo_last_valid_date="2026-11-14")
        assert latest_date == "2026-11-15"
        assert expl is not None
        assert "19 of 24 meteorology variables" in expl
        assert "T2M_MAX, WS2M_MIN" in expl
        assert "2026-11-14" in expl
    finally:
        conn.close()

    assert DEFAULT_DEMO_VINTAGE_DB_PATH.exists()
