"""
CLI Runner for Step 4 Forecast Ledger & Post-Test Retrospective Evaluation
========================================================================
Executes:
1. Optional (a) 2,184-origin Test Parity audit (`v_observations_latest` DB path vs NPZ).
2. Issuing 7-horizon forecasts into `forecasts_ledger` for SC-35, SC-30, SC-31,
   BASELINE_PERSISTENCE, and BASELINE_CLIMATOLOGY_DOY over either:
   - the entire Post-Test complete window (`2026-05-01` .. `get_default_complete_anchor_date()`), or
   - a single `--anchor-date` (defaults to `get_default_complete_anchor_date()`).
3. Verifying 100% As-Of `input_window_hash` reconstruction (`query_observations_as_of`).
4. Evaluating `v_forecast_scores` on `eval_split_regime = 'POST_TEST'` across all 5 models
   (MAE, MSE, RMSE, MASE overall + h1..h7 breakdown for both `first` and `latest` targets).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.vintage_store import (  # noqa: E402
    HCMC_LOCATION_ID,
    connect_sqlite,
    get_default_complete_anchor_date,
    init_schema,
)
from src.inference.forecast_runner import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DB_PATH,
    compute_block_length_sensitivity,
    evaluate_post_test_retrospective_scores,
    issue_forecasts_for_anchors,
    list_complete_anchor_dates,
    verify_db_vs_npz_test_parity,
    verify_input_window_hash_reconstruction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_forecast_ledger")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step 4 forecast ledger issuance, parity & Post-Test scoring.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="Path to rainfall_vintage.db.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR, help="Path to packaged artifacts.")
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "results" / "ml_tweedie_test_forecasts.npz",
        help="Path to canonical ml_tweedie_test_forecasts.npz.",
    )
    parser.add_argument(
        "--verify-test-parity",
        action="store_true",
        help="Run 2,184-origin DB-vs-NPZ parity audit for SC-35, SC-30, SC-31.",
    )
    parser.add_argument(
        "--issue-post-test",
        action="store_true",
        help="Issue forecasts for all COMPLETE_ALL_35 dates from 2026-05-01 up to default complete anchor date.",
    )
    parser.add_argument(
        "--post-test-start",
        type=str,
        default="2026-05-01",
        help="Start date for Post-Test retrospective window (default: 2026-05-01).",
    )
    parser.add_argument(
        "--anchor-date",
        type=str,
        default=None,
        help="Single anchor date to issue (defaults to get_default_complete_anchor_date when --issue-post-test omitted).",
    )
    parser.add_argument(
        "--issued-by",
        type=str,
        default="manual_retrospective_backfill",
        choices=("automated_pipeline", "manual_retrospective_backfill"),
        help="Provenance issuer label stamped on forecasts_ledger rows (Decision 1).",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "storage_audit" / "post_test_retrospective_scores.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    output: Dict[str, Any] = {"db_path": str(args.db_path)}

    if args.verify_test_parity:
        logger.info("Running Check (a): 2,184-origin DB-vs-NPZ Test parity...")
        parity_res = verify_db_vs_npz_test_parity(
            db_path=args.db_path,
            npz_path=args.npz_path,
            artifact_dir=args.artifact_dir,
        )
        output["test_parity_check"] = parity_res
        if not parity_res["passed"]:
            raise AssertionError(f"DB vs NPZ test parity FAILED: {json.dumps(parity_res, indent=2)}")
        logger.info("Check (a) PASSED: max diffs = %s", {k: v["max_abs_diff_vs_npz"] for k, v in parity_res["models"].items()})

    conn = connect_sqlite(args.db_path, read_only=False, recursive_triggers=True)
    try:
        init_schema(conn)
        default_anchor = get_default_complete_anchor_date(conn, location_id=HCMC_LOCATION_ID)
        output["default_complete_anchor_date"] = default_anchor

        if args.issue_post_test:
            if default_anchor is None:
                raise RuntimeError("No COMPLETE_ALL_35 anchor date available in DB.")
            anchors = list_complete_anchor_dates(
                conn,
                start_date=args.post_test_start,
                end_date=default_anchor,
                location_id=HCMC_LOCATION_ID,
            )
            logger.info("Issuing forecasts for %d Post-Test complete anchors (%s .. %s)...", len(anchors), anchors[0], anchors[-1])
            issue_res = issue_forecasts_for_anchors(
                conn,
                anchor_dates=anchors,
                location_id=HCMC_LOCATION_ID,
                artifact_dir=args.artifact_dir,
                issued_by=args.issued_by,
            )
            output["issue_summary"] = issue_res

            hash_check = verify_input_window_hash_reconstruction(
                conn,
                eval_split_regime="POST_TEST",
                location_id=HCMC_LOCATION_ID,
            )
            output["input_window_hash_verification"] = hash_check
            if not hash_check["passed"]:
                raise AssertionError(f"input_window_hash reconstruction failed: {hash_check}")

            scores = evaluate_post_test_retrospective_scores(
                conn,
                eval_split_regime="POST_TEST",
                location_id=HCMC_LOCATION_ID,
            )
            output["post_test_evaluation"] = scores
            output["block_length_sensitivity"] = compute_block_length_sensitivity(
                conn,
                block_lengths=(7, 14, 21),
                eval_split_regime="POST_TEST",
                location_id=HCMC_LOCATION_ID,
            )
        elif args.anchor_date is not None or default_anchor is not None:
            chosen = args.anchor_date or default_anchor
            assert chosen is not None
            issue_res = issue_forecasts_for_anchors(
                conn,
                anchor_dates=[chosen],
                location_id=HCMC_LOCATION_ID,
                artifact_dir=args.artifact_dir,
                issued_by=args.issued_by,
            )
            output["issue_summary"] = issue_res
    finally:
        conn.close()

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved Step 4 report to %s", args.report_json)

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
