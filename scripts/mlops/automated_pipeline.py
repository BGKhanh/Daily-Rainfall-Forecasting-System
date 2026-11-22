"""
Idempotent Scheduled MLOps & Vintage Pipeline CLI (DS108 — Round 11 Part E)
===========================================================================
Subcommands:
- `probe-headers`   : Run per-day NASA POWER `header.sources` transition probe & stamp `ingestion_runs`
- `ingest-window`   : Incremental NASA POWER observation ingest (stamps `config_sha256`, zero duplicate v0 inserts)
- `issue-forecasts` : Issue ONLY unissued `COMPLETE_ALL_35` anchor forecasts (`SKIPPED` when none; 0 `UPDATE`s)
- `refresh-cache`   : Sync `TursoReplicaManager`, clear in-process cache, pre-warm state key, optional webhook

Supports `STORE_BACKEND` in {"local", "turso_replica", "turso_remote"} via `--backend` or env var.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import logging
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Optional, Sequence
import uuid

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.clock import today_hcmc  # noqa: E402
from src.config.settings import (  # noqa: E402
    compute_config_sha256,
    compute_operational_config_sha256,
    compute_result_config_sha256,
    load_settings,
)
from src.data.turso_backend import (  # noqa: E402
    TursoQuotaBlockedError,
    TursoRemoteConnection,
    get_replica_manager,
    open_store_connection,
    redact_secrets,
    resolve_store_backend,
)
from src.data.vintage_store import (  # noqa: E402
    DEFAULT_PROVISIONAL_WINDOW_DAYS,
    HCMC_LOCATION_ID,
    binary_search_source_transition,
    clear_db_query_cache,
    connect_sqlite,
    finalize_db_for_read_only,
    get_db_state_cache_key,
    ingest_nasa_power_window,
    init_schema,
    resolve_git_sha,
    utc_now_iso,
)
from src.inference.forecast_runner import (  # noqa: E402
    ALL_RUNNER_MODEL_IDS,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DB_PATH,
    issue_forecasts_for_anchors,
    list_complete_anchor_dates,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("automated_pipeline")

_EXPECTED_ROWS_PER_ANCHOR = len(ALL_RUNNER_MODEL_IDS) * 7


def _open_rw_store(backend: Optional[str], db_path: Optional[Path]):
    resolved = resolve_store_backend(backend)
    if resolved == "local":
        target = Path(db_path or DEFAULT_DB_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        conn = connect_sqlite(target, read_only=False)
        init_schema(conn)
        return conn, resolved, str(target)
    if resolved in ("turso_replica", "turso_remote"):
        conn = TursoRemoteConnection(read_only=False)
        return conn, resolved, "turso_primary"
    raise ValueError(f"Unsupported backend: {resolved!r}")


def run_probe_headers_step(
    *,
    conn: Any,
    location_id: str = HCMC_LOCATION_ID,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    experiment_today: Optional[str] = None,
    dry_run: bool = False,
    auto_ingest: bool = True,
    ingest_lookback_days: int = 150,
    fetcher: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    window_fetcher: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Probe NASA POWER `header.sources` transition cutoffs and append an `audit_probe` row."""
    from scripts.audit.probe_nasa_header_sources import (
        _classify_header_signature,
        _classify_meteo_source,
        _classify_radiation_source,
        build_sparse_probe_dates,
        fetch_nasa_single_day,
    )

    today_d = today_hcmc()
    exp_today = experiment_today or today_d.isoformat()
    end_iso = end_date or (today_d - timedelta(days=1)).isoformat()
    start_iso = start_date or (today_d - timedelta(days=ingest_lookback_days)).isoformat()
    cfg_sha = compute_config_sha256()
    op_cfg_sha = compute_operational_config_sha256()

    sparse_dates = build_sparse_probe_dates(start_iso=start_iso, end_iso=end_iso, step_days=10)
    session = requests.Session() if fetcher is None else None
    try:
        def _probe_day(obs_d: str, params: str) -> Dict[str, Any]:
            if fetcher is not None:
                return fetcher(obs_d, params)
            assert session is not None
            return fetch_nasa_single_day(session, obs_d, params)

        meteo_sparse = [_probe_day(d, "PRECTOTCORR") for d in sparse_dates]
        rad_sparse = [_probe_day(d, "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN") for d in sparse_dates]

        meteo_pub_sigs = {
            _classify_header_signature(r) for r in meteo_sparse if r["values"].get("PRECTOTCORR") is not None
        }
        rad_pub_sigs = {
            _classify_header_signature(r)
            for r in rad_sparse
            if r["values"].get("ALLSKY_SFC_SW_DWN") is not None
            or r["values"].get("CLRSKY_SFC_SW_DWN") is not None
        }
        header_changes_meteo = len(meteo_pub_sigs) > 1
        header_changes_rad = len(rad_pub_sigs) > 1
        meteo_transition, _ = binary_search_source_transition(
            meteo_sparse,
            lambda d: _probe_day(d, "PRECTOTCORR"),
            classifier=_classify_header_signature if header_changes_meteo else _classify_meteo_source,
        )
        rad_transition, _ = binary_search_source_transition(
            rad_sparse,
            lambda d: _probe_day(d, "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN"),
            classifier=_classify_header_signature if header_changes_rad else _classify_radiation_source,
        )
        meteo_pub_to_unpub, _ = binary_search_source_transition(
            meteo_sparse,
            lambda d: _probe_day(d, "PRECTOTCORR"),
            classifier=lambda r: "PUBLISHED" if r["values"].get("PRECTOTCORR") is not None else "UNPUBLISHED",
            from_label="PUBLISHED",
            to_label="UNPUBLISHED",
        )
        ceres_pub_to_unpub, _ = binary_search_source_transition(
            rad_sparse,
            lambda d: _probe_day(d, "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN"),
            classifier=lambda r: "SYN1DEG_PRESENT" if r["values"].get("CLRSKY_SFC_SW_DWN") is not None else "SYN1DEG_MISSING",
            from_label="SYN1DEG_PRESENT",
            to_label="SYN1DEG_MISSING",
        )
    finally:
        if session is not None:
            session.close()

    from datetime import date as _date

    today_ref = _date.fromisoformat(exp_today)
    meteo_last = meteo_transition["last_from_date"] if (header_changes_meteo and meteo_transition) else None
    rad_last = (
        rad_transition["last_from_date"]
        if (header_changes_rad and rad_transition)
        else (ceres_pub_to_unpub["last_from_date"] if ceres_pub_to_unpub else None)
    )
    if header_changes_meteo and meteo_last:
        finality_mode = "source_header_transition"
        cutoff_date = meteo_last
        lag_d = (today_ref - _date.fromisoformat(meteo_last)).days
        finality_rule = (
            f"FINALIZED when PRECTOTCORR IS NOT NULL and obs_date <= {meteo_last} "
            f"(verified MERRA2 -> GEOSIT header transition cutoff, lag={lag_d}d vs {exp_today})."
        )
    else:
        finality_mode = "provisional_window_fallback"
        cutoff_date = (today_ref - timedelta(days=DEFAULT_PROVISIONAL_WINDOW_DAYS)).isoformat()
        finality_rule = (
            f"header.sources uniform; fallback applies FINALIZED when completeness_regime == 'COMPLETE_ALL_35' "
            f"and age > {DEFAULT_PROVISIONAL_WINDOW_DAYS} days."
        )

    cutoffs_payload = {
        "experiment_date": exp_today,
        "api_version": (meteo_sparse[0].get("api_version") if meteo_sparse else "v2.10.3") or "v2.10.3",
        "header_sources_varies_per_day_meteo": header_changes_meteo,
        "header_sources_varies_per_day_radiation": header_changes_rad,
        "meteo_merra2_last_date": meteo_last,
        "rad_syn1deg_last_date": rad_last,
        "meteo_last_published_date": meteo_pub_to_unpub["last_from_date"] if meteo_pub_to_unpub else None,
        "finality_mode": finality_mode,
        "finality_cutoff_date": cutoff_date,
        "finality_rule": finality_rule,
        "observed_unique_header_sources": [
            list(s) for s in sorted({tuple(r.get("sources") or ()) for r in (meteo_sparse + rad_sparse)})
        ],
    }

    run_id = None
    ingest_summary = None
    if not dry_run:
        prev = conn.execute(
            """
            SELECT scan_window_start, scan_window_end, nasa_api_version, latest_valid_by_group_json
            FROM ingestion_runs
            WHERE location_id = ? AND status = 'SUCCESS' AND latest_valid_by_group_json IS NOT NULL
            ORDER BY julianday(finished_at_utc) DESC, rowid DESC
            LIMIT 1
            """,
            (location_id,),
        ).fetchone()
        merged_json = json.loads(str(prev["latest_valid_by_group_json"])) if (prev and prev["latest_valid_by_group_json"]) else {}
        merged_json["source_transition_cutoffs"] = cutoffs_payload
        now_utc = utc_now_iso()
        run_id = f"auto_probe_{uuid.uuid4().hex[:12]}"
        conn.execute(
            """
            INSERT INTO ingestion_runs (
                run_id, run_type, started_at_utc, finished_at_utc, status, location_id,
                scan_window_start, scan_window_end, nasa_api_version, nasa_sources_header_json,
                rows_scanned, rows_inserted_v0, rows_new_vintage, rows_late_fill_only,
                rows_value_change, rows_unchanged, rows_unpublished_skipped,
                latest_valid_by_group_json, revision_stats_json, git_sha, config_sha256, operational_config_sha256, error_message
            ) VALUES (?, 'audit_probe', ?, ?, 'SUCCESS', ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0, 0, ?, ?, ?, ?, ?, NULL)
            """,
            (
                run_id,
                now_utc,
                now_utc,
                location_id,
                str(prev["scan_window_start"]) if prev else start_iso,
                str(prev["scan_window_end"]) if prev else end_iso,
                cutoffs_payload["api_version"],
                json.dumps(cutoffs_payload["observed_unique_header_sources"]),
                json.dumps(merged_json, ensure_ascii=False),
                json.dumps({"pipeline": "automated_pipeline.probe-headers"}),
                resolve_git_sha(),
                cfg_sha,
                op_cfg_sha,
            ),
        )
        conn.commit()
        if auto_ingest:
            ingest_summary = run_ingest_window_step(
                conn=conn,
                lookback_days=ingest_lookback_days,
                start_date=cutoff_date if header_changes_meteo and cutoff_date else start_iso,
                end_date=end_iso,
                location_id=location_id,
                fetcher=window_fetcher,
            )
        if not isinstance(conn, TursoRemoteConnection):
            finalize_db_for_read_only(conn)

    return {
        "command": "probe-headers",
        "status": "DRY_RUN" if dry_run else "SUCCESS",
        "run_id": run_id,
        "config_sha256": cfg_sha,
        "finality_cutoff_date": cutoff_date,
        "finality_mode": finality_mode,
        "post_probe_ingest": ingest_summary,
    }


def run_ingest_window_step(
    *,
    conn: Any,
    lookback_days: int = 150,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    location_id: str = HCMC_LOCATION_ID,
    fetcher: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Idempotently ingest NASA POWER daily window, stamping config_sha256."""
    cfg_sha = compute_config_sha256()
    op_cfg_sha = compute_operational_config_sha256()
    summary = ingest_nasa_power_window(
        conn,
        lookback_days=lookback_days,
        start_date=start_date,
        end_date=end_date,
        location_id=location_id,
        run_type="manual_scan",
        fetcher=fetcher,
    )
    return {
        "command": "ingest-window",
        "status": summary.get("status", "SUCCESS"),
        "run_id": summary.get("run_id"),
        "config_sha256": cfg_sha,
        "operational_config_sha256": op_cfg_sha,
        "rows_scanned": summary.get("rows_scanned", 0),
        "rows_inserted_v0": summary.get("rows_inserted_v0", 0),
        "rows_new_vintage": summary.get("rows_new_vintage", 0),
        "rows_unchanged": summary.get("rows_unchanged", 0),
    }


def _skipped_issue_forecasts_result(
    reason: str,
    cfg_sha: str,
    *,
    latest_complete_anchor: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "command": "issue-forecasts",
        "status": "SKIPPED",
        "reason": reason,
        "anchors_processed": 0,
        "rows_inserted": 0,
        "updates_executed": 0,
        "config_sha256": cfg_sha,
    }
    if latest_complete_anchor is not None:
        out["latest_complete_anchor"] = latest_complete_anchor
    return out


def run_issue_forecasts_step(
    *,
    conn: Any,
    location_id: str = HCMC_LOCATION_ID,
    post_test_start: str = "2026-05-01",
    end_date: Optional[str] = None,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    dry_run: bool = False,
    latest_only: bool = False,
) -> Dict[str, Any]:
    """
    Issue 5-model x 7-horizon forecasts ONLY for unissued meteo-complete anchor dates.
    When `latest_only=True`, inspects ONLY the single newest complete anchor (`complete_anchors[-1]`)
    without backfilling older unissued anchors. Logs `SKIPPED` when nothing new is ready.
    """
    cfg_sha = compute_config_sha256()
    res_cfg_sha = compute_result_config_sha256()
    effective_end = end_date or today_hcmc().isoformat()
    from src.data.vintage_store import METEO_24_VARS

    complete_anchors = list_complete_anchor_dates(
        conn,
        location_id=location_id,
        start_date=post_test_start,
        end_date=effective_end,
        required_vars=METEO_24_VARS,
    )
    if not complete_anchors:
        logger.info("SKIPPED issue-forecasts: no meteo-complete anchors found in [%s, %s].", post_test_start, effective_end)
        return _skipped_issue_forecasts_result("no_complete_anchor_in_window", cfg_sha)

    counts_rows = conn.execute(
        """
        SELECT data_as_of, COUNT(*) AS cnt
        FROM forecasts_ledger
        WHERE location_id = ?
          AND data_as_of >= ?
          AND data_as_of <= ?
        GROUP BY data_as_of
        """,
        (location_id, post_test_start, effective_end),
    ).fetchall()
    issued_counts = {str(r["data_as_of"]): int(r["cnt"]) for r in counts_rows}
    if latest_only:
        latest_anchor = complete_anchors[-1]
        if issued_counts.get(latest_anchor, 0) >= _EXPECTED_ROWS_PER_ANCHOR:
            logger.info(
                "SKIPPED issue-forecasts (--latest-only): newest complete anchor %s is already issued.",
                latest_anchor,
            )
            return _skipped_issue_forecasts_result(
                "no_new_complete_anchor", cfg_sha, latest_complete_anchor=latest_anchor
            )
        pending_anchors = [latest_anchor]
    else:
        pending_anchors = [
            a for a in complete_anchors if issued_counts.get(a, 0) < _EXPECTED_ROWS_PER_ANCHOR
        ]

    if not pending_anchors:
        logger.info(
            "SKIPPED issue-forecasts: all %d complete anchors up to %s are already issued.",
            len(complete_anchors),
            complete_anchors[-1],
        )
        return _skipped_issue_forecasts_result(
            "no_new_complete_anchor", cfg_sha, latest_complete_anchor=complete_anchors[-1]
        )

    if dry_run:
        return {
            "command": "issue-forecasts",
            "status": "DRY_RUN",
            "pending_anchors": pending_anchors,
            "anchors_processed": len(pending_anchors),
            "rows_inserted": 0,
            "updates_executed": 0,
            "config_sha256": cfg_sha,
        }

    issued_res = issue_forecasts_for_anchors(
        conn,
        anchor_dates=pending_anchors,
        location_id=location_id,
        artifact_dir=artifact_dir,
        issued_by="automated_pipeline",
    )
    if not isinstance(conn, TursoRemoteConnection):
        finalize_db_for_read_only(conn)
    return {
        "command": "issue-forecasts",
        "status": "SUCCESS",
        "issued_by": "automated_pipeline",
        "anchors_processed": issued_res["anchors_processed"],
        "rows_inserted": issued_res["rows_inserted"],
        "rows_skipped": issued_res["rows_skipped"],
        "updates_executed": 0,
        "latest_issued_anchor": pending_anchors[-1],
        "config_sha256": cfg_sha,
        "result_config_sha256": res_cfg_sha,
    }


def run_refresh_cache_step(
    *,
    backend: Optional[str] = None,
    db_path: Optional[Path] = None,
    webhook_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Sync local replica (when enabled), flush in-process query cache, and warm state key."""
    resolved = resolve_store_backend(backend)
    sync_info: Optional[Dict[str, Any]] = None
    clear_db_query_cache()

    if resolved == "turso_replica":
        sync_info = get_replica_manager().sync_now()
    conn, _ = open_store_connection(db_path=db_path, read_only=True, backend=resolved)
    try:
        state_key = get_db_state_cache_key(conn, resolved)
    finally:
        conn.close()

    webhook_status: Optional[int] = None
    if webhook_url:
        try:
            r = requests.get(webhook_url, timeout=15)
            webhook_status = r.status_code
        except requests.RequestException as exc:
            safe_exc = redact_secrets(str(exc), extra_secrets=[webhook_url])
            raise RuntimeError(f"Webhook refresh failed: {safe_exc}") from exc

    return {
        "command": "refresh-cache",
        "status": "SUCCESS",
        "backend": resolved,
        "state_key": list(state_key),
        "replica_sync": sync_info,
        "webhook_status": webhook_status,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Idempotent DS108 Scheduled MLOps & Vintage Pipeline CLI.")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["local", "turso_replica", "turso_remote"],
        help="Store backend override (default: STORE_BACKEND env or runtime_config.toml).",
    )
    parser.add_argument("--db-path", type=Path, default=None, help="Local SQLite DB path when backend=local.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional file path to write JSON summary.")

    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_probe = sub.add_parser("probe-headers", help="Probe NASA POWER header sources transition cutoffs.")
    p_probe.add_argument("--start-date", type=str, default=None)
    p_probe.add_argument("--end-date", type=str, default=None)
    p_probe.add_argument("--experiment-today", type=str, default=None)
    p_probe.add_argument("--dry-run", action="store_true")

    p_ing = sub.add_parser("ingest-window", help="Ingest incremental NASA POWER observations.")
    p_ing.add_argument(
        "--days",
        "--lookback-days",
        dest="lookback_days",
        type=int,
        default=settings.operational.short_scan_window_days,
    )
    p_ing.add_argument("--start-date", type=str, default=None)
    p_ing.add_argument("--end-date", type=str, default=None)

    p_fc = sub.add_parser("issue-forecasts", help="Issue forecasts strictly for new meteo-complete anchors.")
    p_fc.add_argument("--post-test-start", type=str, default="2026-05-01")
    p_fc.add_argument("--end-date", type=str, default=None)
    p_fc.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    p_fc.add_argument("--dry-run", action="store_true")
    p_fc.add_argument(
        "--latest-only",
        action="store_true",
        help="Inspect and issue only for the newest meteo-complete anchor without historical backfill.",
    )

    p_ref = sub.add_parser("refresh-cache", help="Sync Turso replica & refresh in-process cache state.")
    p_ref.add_argument("--webhook-url", type=str, default=None)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.subcommand == "ingest-window" and not (1 <= int(args.lookback_days) <= 1000):
        parser.error("--days / --lookback-days must be between 1 and 1000.")
    try:
        if args.subcommand == "refresh-cache":
            out = run_refresh_cache_step(
                backend=args.backend, db_path=args.db_path, webhook_url=args.webhook_url
            )
        else:
            conn, _, _ = _open_rw_store(args.backend, args.db_path)
            try:
                if args.subcommand == "probe-headers":
                    out = run_probe_headers_step(
                        conn=conn,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        experiment_today=args.experiment_today,
                        dry_run=args.dry_run,
                    )
                elif args.subcommand == "ingest-window":
                    out = run_ingest_window_step(
                        conn=conn,
                        lookback_days=args.lookback_days,
                        start_date=args.start_date,
                        end_date=args.end_date,
                    )
                elif args.subcommand == "issue-forecasts":
                    out = run_issue_forecasts_step(
                        conn=conn,
                        post_test_start=args.post_test_start,
                        end_date=args.end_date,
                        artifact_dir=args.artifact_dir,
                        dry_run=args.dry_run,
                        latest_only=args.latest_only,
                    )
                else:
                    raise ValueError(f"Unknown subcommand: {args.subcommand}")
            finally:
                conn.close()
    except TursoQuotaBlockedError as exc:
        safe_err = redact_secrets(str(exc))
        logger.warning("DEFERRED_QUOTA_BLOCKED (%s): %s", args.subcommand, safe_err)
        out = {
            "command": args.subcommand,
            "status": "DEFERRED_QUOTA_BLOCKED",
            "degraded_reason": "TURSO_REMOTE_BLOCKED",
            "error": safe_err,
        }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
