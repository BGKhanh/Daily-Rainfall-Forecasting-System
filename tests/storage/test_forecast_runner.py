"""
Automated Tests for Step 4 Forecast Runner & DB-vs-NPZ Test Parity (DDL V3.2)
=============================================================================
Verifies:
1. Canonical column ordering parity between `read_latest_observations()`, `sample_recent_history_35d.csv`,
   and `DataLoader().load_data()`.
2. Full end-to-end `forecast_runner` execution on a temporary DB seeded from tracked historical CSV
   (SC-35, SC-30, SC-31, BASELINE_PERSISTENCE, BASELINE_CLIMATOLOGY_DOY): DOY climatology table,
   idempotent ledger insertion, 100% As-Of `input_window_hash` reconstruction, `v_forecast_scores`
   evaluation, and batch-vs-single-step parity (< 1e-5 mm) — runs identically on CI and local.
3. Full 2,184-origin Test Parity (`data/vintage/rainfall_vintage.db` vs
   `data/processed/results/ml_tweedie_test_forecasts.npz`, max_abs_diff < 1e-5 mm) and Post-Test
   ledger verification when the gitignored local artifacts are present.
"""

from __future__ import annotations

import os
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
    CANONICAL_VI_COLUMN_ORDER,
    connect_sqlite,
    init_schema,
    seed_historical_csv_vintage_v0,
)
from src.evaluation.protocols import get_global_read_guard  # noqa: E402
from src.inference.forecast_runner import (  # noqa: E402
    compute_dev_train_doy_climatology,
    evaluate_post_test_retrospective_scores,
    issue_forecasts_for_anchors,
    list_complete_anchor_dates,
    verify_db_vs_npz_test_parity,
    verify_input_window_hash_reconstruction,
)


@pytest.fixture(autouse=True)
def _allow_canonical_csv_reads():
    g = get_global_read_guard()
    was_active = g.is_active
    g.deactivate()
    yield
    if was_active:
        g.activate()


def test_canonical_column_order_matches_sample_and_dataloader():
    """Ensure SQLite readers emit columns in exact DataLoader / FeatureBuilder order."""
    sample_df = pd.read_csv(PROJECT_ROOT / "artifacts" / "models" / "sample_recent_history_35d.csv")
    loader_df = DataLoader().load_data()
    assert list(CANONICAL_VI_COLUMN_ORDER) == list(sample_df.columns)
    assert list(CANONICAL_VI_COLUMN_ORDER) == list(loader_df.columns)


def test_forecast_runner_end_to_end_on_seeded_db(tmp_path: Path):
    """Run full 5-model forecast runner on a seeded tmp DB (works identically on CI without live DB)."""
    db_path = tmp_path / "runner_ci.db"
    loader_df = DataLoader().load_data(rename_columns=False)

    # Trim to minimal Dev Train (all 366 DOYs via 2016 leap year..2017) + trailing 2026 rows so test runs fast
    subset_df = loader_df[
        ((loader_df["DATE"] >= pd.Timestamp("2016-01-01")) & (loader_df["DATE"] <= pd.Timestamp("2017-12-31")))
        | (loader_df["DATE"] >= pd.Timestamp("2026-04-01"))
    ].copy().reset_index(drop=True)

    # Append 12 continuous Post-Test complete days (2026-05-01..2026-05-12) so h1..h7 have full targets on 3 anchors
    last_row = subset_df.iloc[-1].to_dict()
    ext_rows = []
    for idx, d in enumerate(pd.date_range("2026-05-01", "2026-05-12", freq="D")):
        r = dict(last_row)
        r["DATE"] = d
        r["PRECTOTCORR"] = float(round((idx % 5) * 3.2, 2))
        ext_rows.append(r)
    full_seed_df = pd.concat([subset_df, pd.DataFrame(ext_rows)], ignore_index=True)

    conn = connect_sqlite(db_path, read_only=False, recursive_triggers=True)
    try:
        init_schema(conn)
        seed_historical_csv_vintage_v0(conn, df=full_seed_df, ingested_at_utc="2026-06-01T00:00:00Z")

        doy_table, table_sha = compute_dev_train_doy_climatology(conn)
        assert len(doy_table) == 366
        assert len(table_sha) == 64

        anchors = list_complete_anchor_dates(conn, start_date="2026-05-01", end_date="2026-05-03")
        assert anchors == ["2026-05-01", "2026-05-02", "2026-05-03"]

        res = issue_forecasts_for_anchors(
            conn,
            anchor_dates=anchors,
            issued_at_utc="2026-06-02T00:00:00Z",
            issued_run_date="2026-06-02",
        )
        assert res["rows_inserted"] == 3 * 5 * 7  # 105
        assert res["rows_skipped"] == 0

        # Idempotent re-run skips all 105
        res_dup = issue_forecasts_for_anchors(
            conn,
            anchor_dates=anchors,
            issued_at_utc="2026-06-02T00:00:00Z",
            issued_run_date="2026-06-02",
        )
        assert res_dup["rows_inserted"] == 0
        assert res_dup["rows_skipped"] == 105

        hash_chk = verify_input_window_hash_reconstruction(conn, eval_split_regime="POST_TEST")
        assert hash_chk["passed"] is True
        assert hash_chk["matches"] == 3

        scores = evaluate_post_test_retrospective_scores(conn, eval_split_regime="POST_TEST")
        assert scores["total_scored_rows"] == 105
        assert scores["is_prospective_counts"] == {"0": 105}
        assert scores["temporal_status_at_issue_counts"] == {"past_observed": 105}
        assert len(scores["leaderboard_vs_latest"]) == 5
        assert all(np.isfinite(m["MAE"]) and np.isfinite(m["MASE"]) for m in scores["leaderboard_vs_latest"])
        assert len(ALL_35_VARS) == 35
    finally:
        conn.close()


def test_db_vs_npz_2184_origin_test_parity_and_demo_post_test():
    """Verify Check (a) DB-vs-NPZ parity < 1e-5 mm across 2,184 Test origins and Two-DB contract."""
    demo_db = PROJECT_ROOT / "artifacts" / "models" / "demo_rainfall_vintage.db"
    live_db = PROJECT_ROOT / "data" / "vintage" / "rainfall_vintage.db"
    npz_path = PROJECT_ROOT / "data" / "processed" / "results" / "ml_tweedie_test_forecasts.npz"
    require_parity = os.environ.get("CI_REQUIRE_PARITY") == "1"

    missing = [str(p) for p in (demo_db, npz_path) if not p.exists()]
    if missing:
        if require_parity:
            pytest.fail(f"CI_REQUIRE_PARITY=1 but required parity artifact(s) missing: {missing}")
        pytest.skip(f"Demo DB or NPZ missing locally: {missing}")

    parity = verify_db_vs_npz_test_parity(db_path=demo_db, npz_path=npz_path)
    assert parity["passed"] is True
    assert parity["origins_verified"] == 2184
    assert parity["y_true_max_abs_diff"] < 1e-5
    assert parity["build_single_step_sample_max_diff_mm"] < 1e-5
    for m_id in ("SC-35", "SC-30", "SC-31"):
        assert parity["models"][m_id]["max_abs_diff_vs_npz"] < 1e-5

    conn = connect_sqlite(demo_db, read_only=True)
    try:
        hash_chk = verify_input_window_hash_reconstruction(conn, eval_split_regime="POST_TEST")
        assert hash_chk["passed"] is True
        assert hash_chk["matches"] == 122
        scores = evaluate_post_test_retrospective_scores(conn, eval_split_regime="POST_TEST")
        assert scores["unique_origins"] == 122
        assert scores["total_scored_rows"] == 4270
        assert scores["block_bootstrap"]["block_length"] == 7
        assert scores["block_bootstrap"]["n_blocks"] == 18
        assert scores["block_bootstrap"]["n_bootstrap"] == 2000
        for pair_key in (
            "SC-35 - BASELINE_CLIMATOLOGY_DOY",
            "SC-31 - BASELINE_CLIMATOLOGY_DOY",
            "SC-30 - BASELINE_CLIMATOLOGY_DOY",
            "SC-35 - SC-31",
            "SC-35 - SC-30",
            "SC-30 - SC-31",
        ):
            assert pair_key in scores["block_bootstrap"]["pairwise_delta_mae"]
        assert len(scores["methodological_notes"]) == 3
    finally:
        conn.close()

    # Round 8 Two-DB invariant: live production DB (if present locally) must hold observations only (ledger = 0)
    if live_db.exists():
        conn_live = connect_sqlite(live_db, read_only=True)
        try:
            obs_count = int(conn_live.execute("SELECT COUNT(*) FROM observations_vintage").fetchone()[0])
            ledger_count = int(conn_live.execute("SELECT COUNT(*) FROM forecasts_ledger").fetchone()[0])
            assert obs_count >= 9449
            assert ledger_count == 0, f"Expected 0 retrospective ledger rows in live DB, got {ledger_count}"
        finally:
            conn_live.close()
