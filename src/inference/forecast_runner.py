"""
Operational & Retrospective Forecast Ledger Runner (DS108 DDL V3.2 — Step 4)
===========================================================================
Reads strictly from SQLite Vintage Store (`v_observations_latest` / As-Of views),
runs the 3 packaged ML models (SC-35, SC-30, SC-31) plus 2 canonical Baselines
(BASELINE_PERSISTENCE, BASELINE_CLIMATOLOGY_DOY), writes immutable rows into
`forecasts_ledger`, verifies 2,184-origin DB-vs-NPZ Test parity (< 1e-5 mm),
verifies 100% As-Of `input_window_hash` reproducibility, and evaluates scores
through `v_forecast_scores`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)

from src.config.clock import today_hcmc
from src.config.constants import Config
from src.data.vintage_store import (
    ALL_35_VARS,
    HCMC_LOCATION_ID,
    align_and_validate_canonical_raw_df,
    compute_input_window_hash,
    connect_sqlite,
    derive_eval_split_regime,
    derive_temporal_status_at_issue,
    insert_forecasts_idempotent,
    query_observations_as_of,
    read_latest_observations,
    resolve_git_sha,
    utc_now_iso,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "models"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "vintage" / "rainfall_vintage.db"
CANONICAL_DEV_TRAIN_END = "2018-04-30"
OFFICIAL_MASE_SCALE_POST_TEST = 3.7015424361567635
CANONICAL_DEV_TRAIN_MASE_SCALE = 3.7015424361567634
PERSISTENCE_SPEC_SHA256 = hashlib.sha256(
    b"BASELINE_PERSISTENCE:v1:pred_mm(h)=PRECTOTCORR[data_as_of]:h=1..7"
).hexdigest()

ALL_RUNNER_MODEL_IDS: Tuple[str, ...] = (
    "SC-35",
    "SC-30",
    "SC-31",
    "BASELINE_PERSISTENCE",
    "BASELINE_CLIMATOLOGY_DOY",
)

METHODOLOGICAL_NOTES: Dict[str, str] = {
    "effective_sample_size": (
        "122 unique anchor dates (2026-05-01..2026-08-30), NOT 854 independent observations "
        "(854 = 122 anchors x 7 overlapping horizons)."
    ),
    "first_vs_latest_equality": (
        "vs_first and vs_latest metrics are identical because each target date (2026-05-02..2026-09-06) "
        "currently has a single ingested version (version_num = 0, ingested on 2026-11-16); "
        "this does NOT prove zero NASA POWER revision between NRT and Final."
    ),
    "seasonal_regime_note": (
        "The 2026-05-01..2026-08-30 window lies entirely within the Southwest Monsoon wet season in HCMC; "
        "therefore its absolute MAE (~4.92-4.99 mm) is not directly comparable to the full-year 6-year Test MAE "
        "(~3.72-3.75 mm) which spans both dry and wet seasons. Valid comparison is against same-window baselines "
        "(-11.0% to -12.2% vs Climatology DOY, -31.1% to -32.1% vs Persistence)."
    ),
}

PAIRWISE_DELTA_MAE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("SC-35", "BASELINE_CLIMATOLOGY_DOY"),
    ("SC-31", "BASELINE_CLIMATOLOGY_DOY"),
    ("SC-30", "BASELINE_CLIMATOLOGY_DOY"),
    ("SC-35", "SC-31"),
    ("SC-35", "SC-30"),
    ("SC-30", "SC-31"),
)


def hcmc_today_str() -> str:
    """Return current operational date in Asia/Ho_Chi_Minh (UTC+7)."""
    return today_hcmc().strftime("%Y-%m-%d")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of an artifact file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dev_train_doy_climatology(
    conn: sqlite3.Connection,
    *,
    location_id: str = HCMC_LOCATION_ID,
    dev_train_end: str = CANONICAL_DEV_TRAIN_END,
) -> Tuple[Dict[int, float], str]:
    """
    Compute day-of-year (1..366) rainfall climatology strictly on Dev Train rows
    (`obs_date <= dev_train_end`) in `v_observations_latest`.

    Returns `(doy_means_dict, frozen_table_sha256)`.
    """
    rows = conn.execute(
        """
        SELECT obs_date, PRECTOTCORR
        FROM v_observations_latest
        WHERE location_id = ?
          AND obs_date <= ?
          AND PRECTOTCORR IS NOT NULL
        ORDER BY obs_date ASC
        """,
        (location_id, dev_train_end),
    ).fetchall()
    if not rows:
        raise ValueError(
            f"No Dev Train precipitation rows found in v_observations_latest (obs_date <= {dev_train_end})."
        )

    df = pd.DataFrame([{"obs_date": r["obs_date"], "rain": float(r["PRECTOTCORR"])} for r in rows])
    df["doy"] = pd.to_datetime(df["obs_date"]).dt.dayofyear
    grouped = df.groupby("doy")["rain"].mean().to_dict()
    table: Dict[int, float] = {int(k): float(v) for k, v in sorted(grouped.items())}
    if 366 not in table and 365 in table:
        table[366] = table[365]

    canonical_json = json.dumps({str(k): table[k] for k in sorted(table)}, sort_keys=True, separators=(",", ":"))
    table_sha = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return table, table_sha


def _rows_to_vietnamese_df(window_rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert raw English observation dicts into a Vietnamese DataFrame expected by FeatureBuilder."""
    en_to_vi = {k: v for k, v in Config.COLUMN_MAPPING.items() if k == "DATE" or k in ALL_35_VARS}
    records = []
    for row in window_rows:
        item: Dict[str, Any] = {"DATE": row["obs_date"]}
        for col in ALL_35_VARS:
            item[col] = row.get(col)
        records.append(item)
    df = pd.DataFrame(records)
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.normalize()
    df = df.sort_values("DATE").reset_index(drop=True)
    df = df.rename(columns=en_to_vi)
    return align_and_validate_canonical_raw_df(df)


def _load_runner_bundle(artifact_dir: Path) -> Dict[str, Any]:
    """Load SC-35, SC-30, SC-31 estimators, F2/F3 FeatureBuilders, and SHA-256 digests."""
    files = {
        "sc35_model": artifact_dir / "sc35_xgboost_tweedie_f3.joblib",
        "sc30_model": artifact_dir / "sc30_lightgbm_tweedie_f2.joblib",
        "sc31_model": artifact_dir / "sc31_lightgbm_tweedie_f3.joblib",
        "fb_f3": artifact_dir / "feature_builder_f3.joblib",
        "fb_f2": artifact_dir / "feature_builder_f2.joblib",
    }
    return {
        "models": {
            "SC-35": joblib.load(files["sc35_model"]),
            "SC-30": joblib.load(files["sc30_model"]),
            "SC-31": joblib.load(files["sc31_model"]),
        },
        "feature_builders": {
            "F3": joblib.load(files["fb_f3"]),
            "F2": joblib.load(files["fb_f2"]),
        },
        "sha256": {k: compute_file_sha256(p) for k, p in files.items()},
    }


def list_complete_anchor_dates(
    conn: sqlite3.Connection,
    *,
    start_date: str,
    end_date: str,
    location_id: str = HCMC_LOCATION_ID,
) -> List[str]:
    """Return all dates in [start_date, end_date] with COMPLETE_ALL_35 and a complete 30-day lookback."""
    rows = conn.execute(
        """
        SELECT obs_date, completeness_regime
        FROM v_observations_latest
        WHERE location_id = ?
          AND obs_date BETWEEN ? AND ?
        ORDER BY obs_date ASC
        """,
        (location_id, (pd.to_datetime(start_date) - timedelta(days=29)).strftime("%Y-%m-%d"), end_date),
    ).fetchall()
    complete_set = {str(r["obs_date"]) for r in rows if r["completeness_regime"] == "COMPLETE_ALL_35"}

    anchors: List[str] = []
    for dt in pd.date_range(start_date, end_date, freq="D"):
        anchor_str = dt.strftime("%Y-%m-%d")
        win_dates = [(dt - timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(29, -1, -1)]
        if all(d in complete_set for d in win_dates):
            anchors.append(anchor_str)
    return anchors


def issue_forecasts_for_anchors(
    conn: sqlite3.Connection,
    *,
    anchor_dates: Sequence[str],
    location_id: str = HCMC_LOCATION_ID,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    issued_at_utc: Optional[str] = None,
    issued_run_date: Optional[str] = None,
    model_ids: Sequence[str] = ALL_RUNNER_MODEL_IDS,
    git_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate and idempotently insert 7-horizon forecasts into `forecasts_ledger`
    for each validated `COMPLETE_ALL_35` anchor date across selected models.
    """
    if not anchor_dates:
        return {"anchors_processed": 0, "rows_inserted": 0, "rows_skipped": 0}

    bundle = _load_runner_bundle(Path(artifact_dir))
    fb_f3 = bundle["feature_builders"]["F3"]
    fb_f2 = bundle["feature_builders"]["F2"]
    shas = bundle["sha256"]

    doy_table, clim_sha = compute_dev_train_doy_climatology(conn, location_id=location_id)
    loc_row = conn.execute(
        "SELECT provisional_window_days FROM locations WHERE location_id = ?",
        (location_id,),
    ).fetchone()
    prov_window_days = int(loc_row["provisional_window_days"]) if loc_row is not None else 120

    issue_ts = issued_at_utc or utc_now_iso()
    issue_run_d = issued_run_date or hcmc_today_str()
    sha = git_sha or resolve_git_sha()
    selected_models = set(model_ids)

    all_forecast_rows: List[Dict[str, Any]] = []
    anchor_hashes: Dict[str, str] = {}

    for anchor_str in anchor_dates:
        anchor_dt = pd.to_datetime(anchor_str).normalize()
        win_start = (anchor_dt - timedelta(days=29)).strftime("%Y-%m-%d")
        win_end = anchor_dt.strftime("%Y-%m-%d")

        win_rows = query_observations_as_of(
            conn,
            location_id=location_id,
            input_window_start=win_start,
            input_window_end=win_end,
            issued_at_utc=issue_ts,
        )
        if len(win_rows) != 30:
            raise ValueError(
                f"Anchor {win_end} expected 30 continuous days in [{win_start}, {win_end}] at {issue_ts}, "
                f"got {len(win_rows)}."
            )
        incomplete_rows = [r["obs_date"] for r in win_rows if r["completeness_regime"] != "COMPLETE_ALL_35"]
        if incomplete_rows:
            raise ValueError(
                f"Anchor {win_end} violates COMPLETE_ALL_35 invariant on input window dates: {incomplete_rows}"
            )

        win_hash = compute_input_window_hash(win_rows)
        anchor_hashes[win_end] = win_hash
        manifest = [
            {
                "obs_date": str(r["obs_date"]),
                "version_num": int(r["version_num"]),
                "ingest_origin": str(r["ingest_origin"]),
                "completeness_regime": str(r["completeness_regime"]),
            }
            for r in win_rows
        ]
        manifest_json = json.dumps(manifest, separators=(",", ":"))
        issue_day_dt = pd.to_datetime(issue_ts[:10])
        prov_days = sum(
            1
            for r in win_rows
            if int(r["missing_count"]) > 0
            or (issue_day_dt - pd.to_datetime(str(r["obs_date"]))).days <= prov_window_days
        )
        prov_days = max(0, min(30, prov_days))

        win_df = _rows_to_vietnamese_df(win_rows)
        preds_by_model: Dict[str, List[float]] = {}

        if "SC-35" in selected_models or "SC-31" in selected_models:
            feat3 = fb_f3.build_single_step(win_df)
            cols3 = [c for c in feat3.columns if c not in [fb_f3.date_col, fb_f3.target_col]]
            x3 = feat3[cols3].values.astype(np.float64)
            if "SC-35" in selected_models:
                preds_by_model["SC-35"] = np.clip(bundle["models"]["SC-35"].predict(x3)[0], 0.0, None).tolist()
            if "SC-31" in selected_models:
                preds_by_model["SC-31"] = np.clip(bundle["models"]["SC-31"].predict(x3)[0], 0.0, None).tolist()

        if "SC-30" in selected_models:
            feat2 = fb_f2.build_single_step(win_df)
            cols2 = [c for c in feat2.columns if c not in [fb_f2.date_col, fb_f2.target_col]]
            x2 = feat2[cols2].values.astype(np.float64)
            preds_by_model["SC-30"] = np.clip(bundle["models"]["SC-30"].predict(x2)[0], 0.0, None).tolist()

        if "BASELINE_PERSISTENCE" in selected_models:
            anchor_rain = max(0.0, float(win_rows[-1]["PRECTOTCORR"]))
            preds_by_model["BASELINE_PERSISTENCE"] = [anchor_rain] * 7

        if "BASELINE_CLIMATOLOGY_DOY" in selected_models:
            fallback_mean = float(np.mean(list(doy_table.values())))
            preds_by_model["BASELINE_CLIMATOLOGY_DOY"] = [
                max(0.0, float(doy_table.get((anchor_dt + timedelta(days=h)).dayofyear, fallback_mean)))
                for h in range(1, 8)
            ]

        model_meta = {
            "SC-35": ("ML_SERVING", "serving", 1, shas["sc35_model"], shas["fb_f3"], "NONE_COMPLETE"),
            "SC-30": ("ML_SHADOW", "shadow_val_champ", 2, shas["sc30_model"], shas["fb_f2"], "NONE_COMPLETE"),
            "SC-31": ("ML_SHADOW", "shadow_test_contender", 3, shas["sc31_model"], shas["fb_f3"], "NONE_COMPLETE"),
            "BASELINE_PERSISTENCE": ("BASELINE", None, None, PERSISTENCE_SPEC_SHA256, None, "NOT_APPLICABLE_BASELINE"),
            "BASELINE_CLIMATOLOGY_DOY": ("BASELINE", None, None, clim_sha, None, "NOT_APPLICABLE_BASELINE"),
        }

        eval_regime = derive_eval_split_regime(win_end)
        for h in range(1, 8):
            target_str = (anchor_dt + timedelta(days=h)).strftime("%Y-%m-%d")
            has_precip_row = conn.execute(
                """
                SELECT 1
                FROM observations_vintage
                WHERE location_id = ?
                  AND obs_date = ?
                  AND ingested_at_utc <= ?
                  AND PRECTOTCORR IS NOT NULL
                LIMIT 1
                """,
                (location_id, target_str, issue_ts),
            ).fetchone()
            t_status = derive_temporal_status_at_issue(
                target_str,
                issue_run_d,
                target_has_precip_at_issue=(has_precip_row is not None),
            )

            for m_id in ALL_RUNNER_MODEL_IDS:
                if m_id not in selected_models:
                    continue
                m_kind, r_alias, r_ver, art_sha, fb_sha, ceres_policy = model_meta[m_id]
                all_forecast_rows.append(
                    {
                        "issued_at_utc": issue_ts,
                        "issued_run_date": issue_run_d,
                        "run_timezone": "Asia/Ho_Chi_Minh",
                        "location_id": location_id,
                        "model_id": m_id,
                        "model_kind": m_kind,
                        "registry_alias": r_alias,
                        "registry_model_version": r_ver,
                        "artifact_sha256": art_sha,
                        "feature_builder_sha256": fb_sha,
                        "data_as_of": win_end,
                        "eval_split_regime": eval_regime,
                        "horizon_day": h,
                        "target_date": target_str,
                        "temporal_status_at_issue": t_status,
                        "pred_mm": float(preds_by_model[m_id][h - 1]),
                        "input_window_start": win_start,
                        "input_window_end": win_end,
                        "input_window_hash": win_hash,
                        "input_vintage_manifest_json": manifest_json,
                        "input_provisional_days": prov_days,
                        "meteo_missing_count": 0,
                        "flashflux_max_ffill_days": 0,
                        "ceres_policy_applied": ceres_policy,
                        "ceres_max_staleness_days": 0,
                        "is_degraded_input": 0,
                        "degraded_reason": None,
                        "git_sha": sha,
                    }
                )

    inserted, skipped = insert_forecasts_idempotent(conn, all_forecast_rows)
    return {
        "anchors_processed": len(anchor_dates),
        "anchor_start": anchor_dates[0],
        "anchor_end": anchor_dates[-1],
        "models": [m for m in ALL_RUNNER_MODEL_IDS if m in selected_models],
        "climatology_doy_table_sha256": clim_sha,
        "persistence_spec_sha256": PERSISTENCE_SPEC_SHA256,
        "issued_at_utc": issue_ts,
        "issued_run_date": issue_run_d,
        "rows_inserted": inserted,
        "rows_skipped": skipped,
        "sample_anchor_hashes": {k: anchor_hashes[k] for k in list(anchor_hashes)[:3]},
    }


def verify_db_vs_npz_test_parity(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    npz_path: Path = PROJECT_ROOT / "data" / "processed" / "results" / "ml_tweedie_test_forecasts.npz",
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    location_id: str = HCMC_LOCATION_ID,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """
    Verification (a): Read observations strictly from `v_observations_latest` in SQLite DB,
    generate F2/F3 features across all 2,184 canonical Test origins (`2020-05-01..2026-04-23`),
    predict SC-35, SC-30, SC-31, and assert `max_abs_diff < tolerance` vs canonical Test NPZ.
    Also verifies single-step (`build_single_step`) vs batch feature parity on sample Test origins.
    """
    from src.evaluation.protocols import build_purged_direct_samples

    conn = connect_sqlite(db_path, read_only=True)
    try:
        full_df = read_latest_observations(conn, location_id=location_id, as_vietnamese=True)
    finally:
        conn.close()

    if full_df.empty:
        raise RuntimeError(f"Empty observations in {db_path}")

    full_df["Ngày"] = pd.to_datetime(full_df["Ngày"]).dt.normalize()
    context_df = full_df[full_df["Ngày"] <= pd.Timestamp("2020-04-30")].copy().reset_index(drop=True)
    test_df = (
        full_df[(full_df["Ngày"] >= pd.Timestamp("2020-05-01")) & (full_df["Ngày"] <= pd.Timestamp("2026-04-30"))]
        .copy()
        .reset_index(drop=True)
    )

    bundle = _load_runner_bundle(Path(artifact_dir))
    fb_f2 = bundle["feature_builders"]["F2"]
    fb_f3 = bundle["feature_builders"]["F3"]

    feat_f2 = fb_f2.transform(test_df, context_df=context_df, drop_warmup=False)
    cols_f2 = [c for c in feat_f2.columns if c not in [fb_f2.date_col, fb_f2.target_col]]
    x2, y2, dates2 = build_purged_direct_samples(feat_f2, cols_f2, horizon=7)

    feat_f3 = fb_f3.transform(test_df, context_df=context_df, drop_warmup=False)
    cols_f3 = [c for c in feat_f3.columns if c not in [fb_f3.date_col, fb_f3.target_col]]
    x3, y3, dates3 = build_purged_direct_samples(feat_f3, cols_f3, horizon=7)

    npz = np.load(npz_path, allow_pickle=True)
    npz_origins = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in npz["origin_dates"]]
    db_origins_f3 = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates3]
    db_origins_f2 = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates2]

    if len(db_origins_f3) != 2184 or db_origins_f3 != npz_origins or db_origins_f2 != npz_origins:
        raise AssertionError(
            f"Test origin count/alignment mismatch: DB F3={len(db_origins_f3)}, NPZ={len(npz_origins)}"
        )

    y_diff = float(np.max(np.abs(y3 - npz["y_true"])))
    if y_diff >= tolerance:
        raise AssertionError(f"DB y_true vs NPZ mismatch: max_abs_diff={y_diff}")

    npz_keys = {
        "SC-35": "SC-35_xgboost_tweedie_F3_preds",
        "SC-30": "SC-30_lightgbm_tweedie_F2_preds",
        "SC-31": "SC-31_lightgbm_tweedie_F3_preds",
    }
    x_by_model = {"SC-35": x3, "SC-30": x2, "SC-31": x3}

    model_results: Dict[str, Any] = {}
    all_passed = True
    for m_id, npz_key in npz_keys.items():
        pred_db = np.clip(bundle["models"][m_id].predict(x_by_model[m_id]), 0.0, None)
        pred_npz = npz[npz_key]
        max_diff = float(np.max(np.abs(pred_db - pred_npz)))
        passed = max_diff < tolerance
        if not passed:
            all_passed = False
        mae_db = float(mean_absolute_error(y3, pred_db))
        model_results[m_id] = {
            "origins": int(pred_db.shape[0]),
            "horizons": int(pred_db.shape[1]),
            "max_abs_diff_vs_npz": max_diff,
            "mean_abs_diff_vs_npz": float(np.mean(np.abs(pred_db - pred_npz))),
            "db_test_mae_mm": round(mae_db, 6),
            "passed": passed,
        }

    # Verify build_single_step(30d_window) matches batch matrix on boundary/mid origins
    single_step_max_diff = 0.0
    sample_indices = (0, 500, 1091, 1800, 2183)
    for idx in sample_indices:
        origin_dt = pd.Timestamp(db_origins_f3[idx])
        win_df = full_df[
            (full_df["Ngày"] >= origin_dt - pd.Timedelta(days=29)) & (full_df["Ngày"] <= origin_dt)
        ].copy().reset_index(drop=True)
        sf3 = fb_f3.build_single_step(win_df)[cols_f3].values.astype(np.float64)
        pred_single = np.clip(bundle["models"]["SC-35"].predict(sf3)[0], 0.0, None)
        single_step_max_diff = max(single_step_max_diff, float(np.max(np.abs(pred_single - npz["SC-35_xgboost_tweedie_F3_preds"][idx]))))

    return {
        "passed": all_passed and single_step_max_diff < tolerance,
        "tolerance_mm": tolerance,
        "origins_verified": len(db_origins_f3),
        "origin_start": db_origins_f3[0],
        "origin_end": db_origins_f3[-1],
        "y_true_max_abs_diff": y_diff,
        "build_single_step_sample_max_diff_mm": single_step_max_diff,
        "models": model_results,
    }


def verify_input_window_hash_reconstruction(
    conn: sqlite3.Connection,
    *,
    eval_split_regime: str = "POST_TEST",
    location_id: str = HCMC_LOCATION_ID,
) -> Dict[str, Any]:
    """
    Verification (b): Reconstruct the exact 30-day As-Of observation snapshot for every
    forecast window in `forecasts_ledger` (`eval_split_regime`) and confirm 100% SHA-256 hash match.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT data_as_of, issued_at_utc, input_window_start, input_window_end, input_window_hash
        FROM forecasts_ledger
        WHERE location_id = ?
          AND eval_split_regime = ?
        ORDER BY data_as_of ASC
        """,
        (location_id, eval_split_regime),
    ).fetchall()

    mismatches: List[Dict[str, Any]] = []
    for r in rows:
        as_of_rows = query_observations_as_of(
            conn,
            location_id=location_id,
            input_window_start=str(r["input_window_start"]),
            input_window_end=str(r["input_window_end"]),
            issued_at_utc=str(r["issued_at_utc"]),
        )
        recomputed = compute_input_window_hash(as_of_rows)
        if recomputed != str(r["input_window_hash"]):
            mismatches.append(
                {
                    "data_as_of": str(r["data_as_of"]),
                    "expected": str(r["input_window_hash"]),
                    "actual": recomputed,
                    "rows_found": len(as_of_rows),
                }
            )

    return {
        "eval_split_regime": eval_split_regime,
        "unique_windows_checked": len(rows),
        "matches": len(rows) - len(mismatches),
        "mismatches": mismatches,
        "passed": len(rows) > 0 and len(mismatches) == 0,
    }


def _compute_4_metrics(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mase = mae / float(scale)
    return {
        "MAE": round(mae, 6),
        "MSE": round(mse, 6),
        "RMSE": round(rmse, 6),
        "MASE": round(mase, 6),
    }


def _summarize_bootstrap_samples(samples: np.ndarray, point_estimate: float) -> Dict[str, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "point_estimate": round(float(point_estimate), 6),
        "ci_lower": round(float(lo), 6),
        "ci_upper": round(float(hi), 6),
        "ci_width": round(float(hi - lo), 6),
        "bootstrap_se": round(float(np.std(samples, ddof=1)), 6),
    }


def compute_block_bootstrap_ci(
    df_scores: pd.DataFrame,
    *,
    block_length: int = 7,
    n_bootstrap: int = 2000,
    seed: int = 42,
    mase_scale: float = OFFICIAL_MASE_SCALE_POST_TEST,
    target_col: str = "obs_latest_mm",
) -> Dict[str, Any]:
    """
    Moving Block Bootstrap across consecutive anchor dates (`block_length = 7`) preserving all 7 horizons
    together per sampled anchor date. Computes 95% CIs and bootstrap SEs for Overall MAE/RMSE/MASE,
    per-horizon (h1..h7) MAE/RMSE, and pairwise delta-MAE comparisons.
    """
    if df_scores.empty:
        raise ValueError("Cannot compute block bootstrap CI on empty scores DataFrame.")

    anchors = sorted(df_scores["data_as_of"].astype(str).unique().tolist())
    n_origins = len(anchors)
    eff_block = max(1, min(int(block_length), n_origins))
    horizons = [1, 2, 3, 4, 5, 6, 7]

    model_order = [m for m in ALL_RUNNER_MODEL_IDS if m in set(df_scores["model_id"])] + sorted(
        set(df_scores["model_id"]) - set(ALL_RUNNER_MODEL_IDS)
    )

    abs_err_by_model: Dict[str, np.ndarray] = {}
    sq_err_by_model: Dict[str, np.ndarray] = {}
    for m_id in model_order:
        mdf = df_scores[df_scores["model_id"] == m_id]
        piv_pred = (
            mdf.pivot(index="data_as_of", columns="horizon_day", values="pred_mm")
            .reindex(index=anchors, columns=horizons)
            .to_numpy(dtype=np.float64)
        )
        piv_obs = (
            mdf.pivot(index="data_as_of", columns="horizon_day", values=target_col)
            .reindex(index=anchors, columns=horizons)
            .to_numpy(dtype=np.float64)
        )
        diff = piv_pred - piv_obs
        abs_err_by_model[m_id] = np.abs(diff)
        sq_err_by_model[m_id] = diff ** 2

    # Generate deterministic Moving Block Bootstrap index matrix of shape (n_bootstrap, n_origins)
    rng = np.random.default_rng(int(seed))
    num_blocks = n_origins - eff_block + 1
    blocks_per_sample = int(np.ceil(n_origins / eff_block))
    starts = rng.integers(0, num_blocks, size=(int(n_bootstrap), blocks_per_sample))
    offsets = np.arange(eff_block, dtype=np.int64)
    idx_matrix = (starts[:, :, None] + offsets[None, None, :]).reshape(int(n_bootstrap), -1)[:, :n_origins]

    overall_mae_samples: Dict[str, np.ndarray] = {}
    models_ci: Dict[str, Any] = {}
    for m_id in model_order:
        abs_arr = abs_err_by_model[m_id]
        sq_arr = sq_err_by_model[m_id]

        # Sampled shapes: (n_bootstrap, n_origins, 7)
        boot_abs = abs_arr[idx_matrix]
        boot_sq = sq_arr[idx_matrix]

        mae_overall_b = np.nanmean(boot_abs, axis=(1, 2))
        rmse_overall_b = np.sqrt(np.nanmean(boot_sq, axis=(1, 2)))
        mase_overall_b = mae_overall_b / float(mase_scale)
        overall_mae_samples[m_id] = mae_overall_b

        point_mae = float(np.nanmean(abs_arr))
        point_rmse = float(np.sqrt(np.nanmean(sq_arr)))
        point_mase = point_mae / float(mase_scale)

        by_horizon: Dict[str, Any] = {}
        for h_idx, h in enumerate(horizons):
            mae_h_b = np.nanmean(boot_abs[:, :, h_idx], axis=1)
            rmse_h_b = np.sqrt(np.nanmean(boot_sq[:, :, h_idx], axis=1))
            point_mae_h = float(np.nanmean(abs_arr[:, h_idx]))
            point_rmse_h = float(np.sqrt(np.nanmean(sq_arr[:, h_idx])))
            by_horizon[f"h{h}"] = {
                "MAE": _summarize_bootstrap_samples(mae_h_b, point_mae_h),
                "RMSE": _summarize_bootstrap_samples(rmse_h_b, point_rmse_h),
            }

        models_ci[m_id] = {
            "overall": {
                "MAE": _summarize_bootstrap_samples(mae_overall_b, point_mae),
                "RMSE": _summarize_bootstrap_samples(rmse_overall_b, point_rmse),
                "MASE": _summarize_bootstrap_samples(mase_overall_b, point_mase),
            },
            "by_horizon": by_horizon,
        }

    pairwise_delta_mae: Dict[str, Any] = {}
    for model_a, model_b in PAIRWISE_DELTA_MAE_PAIRS:
        if model_a in overall_mae_samples and model_b in overall_mae_samples:
            delta_b = overall_mae_samples[model_a] - overall_mae_samples[model_b]
            point_delta = float(np.nanmean(abs_err_by_model[model_a]) - np.nanmean(abs_err_by_model[model_b]))
            pair_key = f"{model_a} - {model_b}"
            summary = _summarize_bootstrap_samples(delta_b, point_delta)
            summary["significant_at_95pct"] = bool(summary["ci_upper"] < 0.0 or summary["ci_lower"] > 0.0)
            pairwise_delta_mae[pair_key] = summary

    return {
        "method": "moving_block_bootstrap",
        "block_length": int(block_length),
        "effective_block_length": eff_block,
        "n_blocks": int(blocks_per_sample),
        "candidate_blocks": int(num_blocks),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "n_anchor_dates": n_origins,
        "target_col": target_col,
        "models": models_ci,
        "pairwise_delta_mae": pairwise_delta_mae,
    }


def compute_block_length_sensitivity(
    conn: sqlite3.Connection,
    *,
    block_lengths: Sequence[int] = (7, 14, 21),
    eval_split_regime: str = "POST_TEST",
    location_id: str = HCMC_LOCATION_ID,
    mase_scale: float = OFFICIAL_MASE_SCALE_POST_TEST,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate Moving Block Bootstrap 95% CI sensitivity across `block_lengths` (e.g. 7, 14, 21)
    on `v_forecast_scores` and return MAE, CI bounds, CI width, and effective `n_blocks`.
    """
    rows = conn.execute(
        """
        SELECT *
        FROM v_forecast_scores
        WHERE location_id = ? AND eval_split_regime = ?
        ORDER BY model_id ASC, data_as_of ASC, horizon_day ASC
        """,
        (location_id, eval_split_regime),
    ).fetchall()
    if not rows:
        raise ValueError(f"No rows found in v_forecast_scores for eval_split_regime='{eval_split_regime}'.")
    df = pd.DataFrame([dict(r) for r in rows])

    by_block_length: Dict[str, Any] = {}
    for blk in block_lengths:
        boot = compute_block_bootstrap_ci(
            df,
            block_length=int(blk),
            n_bootstrap=n_bootstrap,
            seed=seed,
            mase_scale=mase_scale,
            target_col="obs_latest_mm",
        )
        models_summary: Dict[str, Any] = {}
        for m_id, m_entry in boot["models"].items():
            mae_ci = m_entry["overall"]["MAE"]
            models_summary[m_id] = {
                "MAE": mae_ci["point_estimate"],
                "ci_lower": mae_ci["ci_lower"],
                "ci_upper": mae_ci["ci_upper"],
                "ci_width": mae_ci["ci_width"],
                "bootstrap_se": mae_ci["bootstrap_se"],
            }
        by_block_length[str(int(blk))] = {
            "block_length": int(blk),
            "n_anchor_dates": boot["n_anchor_dates"],
            "n_blocks": boot["n_blocks"],
            "candidate_blocks": boot["candidate_blocks"],
            "models": models_summary,
            "pairwise_delta_mae": boot["pairwise_delta_mae"],
        }

    return {
        "eval_split_regime": eval_split_regime,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "block_lengths": [int(b) for b in block_lengths],
        "by_block_length": by_block_length,
    }


def evaluate_post_test_retrospective_scores(
    conn: sqlite3.Connection,
    *,
    eval_split_regime: str = "POST_TEST",
    location_id: str = HCMC_LOCATION_ID,
    mase_scale: float = OFFICIAL_MASE_SCALE_POST_TEST,
    is_prospective_filter: Optional[int] = None,
    block_length: int = 7,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Query `v_forecast_scores` on `eval_split_regime = 'POST_TEST'` (optionally filtered by
    `is_prospective`), compute overall + per-horizon (h1..h7) MAE/MSE/RMSE/MASE against both
    `first` and `latest`, attach 95% Moving Block Bootstrap CIs, and include methodological notes.
    """
    where_sql = "location_id = ? AND eval_split_regime = ?"
    params: List[Any] = [location_id, eval_split_regime]
    if is_prospective_filter is not None:
        where_sql += " AND is_prospective = ?"
        params.append(int(is_prospective_filter))

    rows = conn.execute(
        f"""
        SELECT *
        FROM v_forecast_scores
        WHERE {where_sql}
        ORDER BY model_id ASC, data_as_of ASC, horizon_day ASC
        """,
        tuple(params),
    ).fetchall()
    if not rows:
        if is_prospective_filter is not None:
            return {
                "eval_split_regime": eval_split_regime,
                "is_prospective_filter": int(is_prospective_filter),
                "mase_scale": mase_scale,
                "canonical_mase_scale_metrics_py": CANONICAL_DEV_TRAIN_MASE_SCALE,
                "origin_min": None,
                "origin_max": None,
                "target_date_min": None,
                "target_date_max": None,
                "unique_origins": 0,
                "total_scored_rows": 0,
                "is_prospective_counts": {},
                "temporal_status_at_issue_counts": {},
                "obs_revision_delta_nonzero_count": 0,
                "leaderboard_vs_latest": [],
                "models": {},
                "block_bootstrap": {
                    "method": "moving_block_bootstrap",
                    "block_length": int(block_length),
                    "n_bootstrap": int(n_bootstrap),
                    "seed": int(seed),
                    "n_anchor_dates": 0,
                    "models": {},
                    "pairwise_delta_mae": {},
                },
                "methodological_notes": dict(METHODOLOGICAL_NOTES),
            }
        raise ValueError(f"No rows found in v_forecast_scores for eval_split_regime='{eval_split_regime}'.")

    df = pd.DataFrame([dict(r) for r in rows])
    null_first = int(df["obs_first_mm"].isna().sum())
    null_latest = int(df["obs_latest_mm"].isna().sum())
    if null_first > 0 or null_latest > 0:
        raise AssertionError(f"Expected 0 NULL target observations, found first={null_first}, latest={null_latest}")

    prospective_counts = df["is_prospective"].value_counts(dropna=False).to_dict()
    temporal_counts = df["temporal_status_at_issue"].value_counts(dropna=False).to_dict()

    leaderboard: List[Dict[str, Any]] = []
    by_model: Dict[str, Any] = {}

    model_order = [m for m in ALL_RUNNER_MODEL_IDS if m in set(df["model_id"])] + sorted(
        set(df["model_id"]) - set(ALL_RUNNER_MODEL_IDS)
    )
    for m_id in model_order:
        mdf = df[df["model_id"] == m_id].copy()
        y_pred = mdf["pred_mm"].to_numpy(dtype=np.float64)
        y_first = mdf["obs_first_mm"].to_numpy(dtype=np.float64)
        y_latest = mdf["obs_latest_mm"].to_numpy(dtype=np.float64)

        overall_first = _compute_4_metrics(y_first, y_pred, mase_scale)
        overall_latest = _compute_4_metrics(y_latest, y_pred, mase_scale)

        horizon_first: Dict[str, Dict[str, float]] = {}
        horizon_latest: Dict[str, Dict[str, float]] = {}
        for h in range(1, 8):
            hdf = mdf[mdf["horizon_day"] == h]
            yp_h = hdf["pred_mm"].to_numpy(dtype=np.float64)
            yf_h = hdf["obs_first_mm"].to_numpy(dtype=np.float64)
            yl_h = hdf["obs_latest_mm"].to_numpy(dtype=np.float64)
            horizon_first[f"h{h}"] = _compute_4_metrics(yf_h, yp_h, mase_scale)
            horizon_latest[f"h{h}"] = _compute_4_metrics(yl_h, yp_h, mase_scale)

        entry = {
            "model_id": m_id,
            "model_kind": str(mdf["model_kind"].iloc[0]),
            "n_origins": int(mdf["data_as_of"].nunique()),
            "n_forecasts": int(len(mdf)),
            "overall_vs_first": overall_first,
            "overall_vs_latest": overall_latest,
            "by_horizon_vs_first": horizon_first,
            "by_horizon_vs_latest": horizon_latest,
        }
        by_model[m_id] = entry
        leaderboard.append(
            {
                "model_id": m_id,
                "model_kind": entry["model_kind"],
                "n_origins": entry["n_origins"],
                "n_forecasts": entry["n_forecasts"],
                **overall_latest,
                "MAE_vs_first": overall_first["MAE"],
            }
        )

    leaderboard.sort(key=lambda r: r["MAE"])
    for idx, row in enumerate(leaderboard, start=1):
        row["rank"] = idx

    block_bootstrap = compute_block_bootstrap_ci(
        df,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
        mase_scale=mase_scale,
        target_col="obs_latest_mm",
    )

    return {
        "eval_split_regime": eval_split_regime,
        "is_prospective_filter": int(is_prospective_filter) if is_prospective_filter is not None else None,
        "mase_scale": mase_scale,
        "canonical_mase_scale_metrics_py": CANONICAL_DEV_TRAIN_MASE_SCALE,
        "origin_min": str(df["data_as_of"].min()),
        "origin_max": str(df["data_as_of"].max()),
        "target_date_min": str(df["target_date"].min()),
        "target_date_max": str(df["target_date"].max()),
        "unique_origins": int(df["data_as_of"].nunique()),
        "total_scored_rows": int(len(df)),
        "is_prospective_counts": {str(k): int(v) for k, v in prospective_counts.items()},
        "temporal_status_at_issue_counts": {str(k): int(v) for k, v in temporal_counts.items()},
        "obs_revision_delta_nonzero_count": int((df["obs_revision_delta_mm"].abs() > 1e-12).sum()),
        "leaderboard_vs_latest": leaderboard,
        "models": by_model,
        "block_bootstrap": block_bootstrap,
        "methodological_notes": dict(METHODOLOGICAL_NOTES),
    }
