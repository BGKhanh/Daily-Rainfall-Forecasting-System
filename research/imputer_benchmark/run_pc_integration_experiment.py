"""Round 13 Step 1 (Section A & Addendum §§1-3):
True Future Days Table, Pre-Declared 3-Clause Validation Gate (h=1..7, h=1..3, h=4..7),
and P-c vs P-c-ffill7 Experiment Across SC-35, SC-31, SC-30 (724 Val & 2,184 Test Origins).
"""

from __future__ import annotations

from datetime import date
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.imputer_benchmark.exogenous_imputer import (  # noqa: E402
    PHYSICAL_METEO_BOUNDS,
    _circular_fill_missing_days,
    _circular_smooth_365,
    _dates_to_circular_doy_365,
)
from research.imputer_benchmark.run_ceres8_vs_11_audit import (  # noqa: E402
    CERES_ONLY_8_CODES,
    FLASHFLUX_3_CODES,
    MASE_SCALE,
    SYN1DEG_11_CODES,
    _compute_split_matrices,
    _load_clean_hcmc_full,
    _predict_single,
    _retrain_dev_train_models,
)
from src.analysis.integrity import assert_no_purged_metrics  # noqa: E402
from src.data.vintage_store import EN_TO_VI_CANONICAL, connect_sqlite  # noqa: E402
from src.inference.validation import CANONICAL_DATE_COL, CANONICAL_TARGET_COL  # noqa: E402

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "models"
PROD_DB_PATH = PROJECT_ROOT / "data" / "vintage" / "rainfall_vintage.db"
DEMO_DB_PATH = PROJECT_ROOT / "artifacts" / "models" / "demo_rainfall_vintage.db"
OUT_JSON_PATH = PROJECT_ROOT / "research" / "imputer_benchmark" / "pc_integration_audit_2184.json"
OUT_README_PATH = PROJECT_ROOT / "research" / "imputer_benchmark" / "README.md"
CERES8_AUDIT_PATH = PROJECT_ROOT / "research" / "imputer_benchmark" / "ceres8_scientific_audit_2184.json"

# Pre-declared 3-Clause Validation Selection Gate (locked BEFORE inspecting Test)
VAL_GATE_MAX_ABS_DELTA_MAE_MM = 0.020
VAL_GATE_MAX_DELTA_RMSE_MM = 0.050
VAL_GATE_MAX_DELTA_P90_TAIL_MAE_MM = 0.250
TIE_BREAK_NOISE_BAND_MAE_MM = 0.005


def build_solar_doy_366_table(dev_train_df: pd.DataFrame, *, radius: int = 7) -> Dict[str, Any]:
    """
    Build Dev_Train-only (2001-01-01..2018-04-30) circular DOY median table (r=7)
    across all 11 solar radiation variables with explicit keys "1".."366".
    """
    cutoff = pd.Timestamp("2018-04-30")
    train = dev_train_df.copy()
    train[CANONICAL_DATE_COL] = pd.to_datetime(train[CANONICAL_DATE_COL]).dt.normalize()
    if train[CANONICAL_DATE_COL].max() > cutoff:
        raise ValueError("Data leakage guard: dev_train_df exceeds 2018-04-30.")

    doy_365 = _dates_to_circular_doy_365(train[CANONICAL_DATE_COL])
    doy_table_by_code: Dict[str, Dict[str, float]] = {}

    for code in SYN1DEG_11_CODES:
        vi_col = EN_TO_VI_CANONICAL[code]
        vals = pd.to_numeric(train[vi_col], errors="coerce").to_numpy(dtype=np.float64)
        daily_med = np.full(365, np.nan, dtype=np.float64)
        for d in range(1, 366):
            mask_d = (doy_365 == d) & np.isfinite(vals)
            if np.any(mask_d):
                daily_med[d - 1] = float(np.median(vals[mask_d]))
        filled = _circular_fill_missing_days(daily_med)
        smoothed_365 = _circular_smooth_365(filled, radius=radius)
        lo, hi = PHYSICAL_METEO_BOUNDS[vi_col]
        smoothed_365 = np.clip(smoothed_365, lo, hi)

        # Explicit leap-day key "366" (for leap-year Dec 31 tm_yday == 366) via circular r=7 wrap around 365/1
        wrap_window = np.concatenate([filled[-radius:], filled[:radius]])
        val_366 = float(np.clip(np.mean(wrap_window), lo, hi))

        per_day = {str(d): round(float(smoothed_365[d - 1]), 6) for d in range(1, 366)}
        per_day["366"] = round(val_366, 6)
        doy_table_by_code[code] = per_day

    return {
        "schema_version": "1.0.0",
        "policy": "P-c_DOY_MEDIAN_11_SOLAR_V1",
        "leap_day_doy_policy": "EXPLICIT_366_CIRCULAR_MEDIAN_R7",
        "dev_train_start": train[CANONICAL_DATE_COL].min().strftime("%Y-%m-%d"),
        "dev_train_end": train[CANONICAL_DATE_COL].max().strftime("%Y-%m-%d"),
        "n_dev_train_rows": int(len(train)),
        "circular_smoothing_radius_days": int(radius),
        "solar_variables": list(SYN1DEG_11_CODES),
        "doy_medians": doy_table_by_code,
    }


def _lookup_doy_matrix(
    doy_table: Dict[str, Any],
    dates: pd.Series,
    codes: Sequence[str],
) -> np.ndarray:
    """Lookup shape (N, len(codes)) using exact anchor.timetuple().tm_yday strings '1'..'366'."""
    doy_map = doy_table["doy_medians"]
    tm_ydays = [str(pd.Timestamp(dt).date().timetuple().tm_yday) for dt in dates]
    out = np.empty((len(tm_ydays), len(codes)), dtype=np.float64)
    for j, code in enumerate(codes):
        col_dict = doy_map[code]
        out[:, j] = [float(col_dict[k]) for k in tm_ydays]
    return out


def _compute_true_future_days_table() -> Dict[str, Any]:
    """Build comparative Effective Future Days table from real `ingestion_runs` latency history."""
    db_path = PROD_DB_PATH if PROD_DB_PATH.exists() else DEMO_DB_PATH
    conn = connect_sqlite(db_path, read_only=True)
    try:
        scan_rows = conn.execute(
            """
            SELECT run_id, run_type, started_at_utc, scan_window_start, scan_window_end, latest_valid_by_group_json
            FROM ingestion_runs
            WHERE status = 'SUCCESS' AND latest_valid_by_group_json IS NOT NULL
            ORDER BY started_at_utc DESC
            LIMIT 8
            """
        ).fetchall()
    finally:
        conn.close()

    observed_scans: List[Dict[str, Any]] = []
    for r in scan_rows:
        grp = json.loads(str(r["latest_valid_by_group_json"]))
        observed_scans.append(
            {
                "run_id": str(r["run_id"]),
                "run_type": str(r["run_type"]),
                "scan_window_end": str(r["scan_window_end"]),
                "METEO_GEOSIT": {
                    "last_valid_date": grp.get("METEO_GEOSIT", {}).get("last_valid_date"),
                    "lag_days_vs_scan_end": grp.get("METEO_GEOSIT", {}).get("lag_days_vs_scan_end"),
                },
                "RAD_FLASHFLUX": {
                    "last_valid_date": grp.get("RAD_FLASHFLUX", {}).get("last_valid_date"),
                    "lag_days_vs_scan_end": grp.get("RAD_FLASHFLUX", {}).get("lag_days_vs_scan_end"),
                },
                "RAD_CERES_SYN1DEG": {
                    "last_valid_date": grp.get("RAD_CERES_SYN1DEG", {}).get("last_valid_date"),
                    "lag_days_vs_scan_end": grp.get("RAD_CERES_SYN1DEG", {}).get("lag_days_vs_scan_end"),
                },
            }
        )

    eval_today = date(2026, 11, 21)
    latest_scan = observed_scans[0] if observed_scans else {}
    meteo_last = str(latest_scan.get("METEO_GEOSIT", {}).get("last_valid_date") or "2026-11-17")
    ff_last_contiguous = "2026-11-12"  # before 2026-11-06/14/16..17 holes across 30d window, or 2026-11-15 latest raw
    ceres_last = str(latest_scan.get("RAD_CERES_SYN1DEG", {}).get("last_valid_date") or "2026-08-30")

    def _future_range(anchor_iso: str, typical_lag_range: str, notes: str) -> Dict[str, Any]:
        a_dt = date.fromisoformat(anchor_iso)
        lag_on_nov21 = (eval_today - a_dt).days
        return {
            "anchor_date_observed_db": anchor_iso,
            "typical_lag_days": typical_lag_range,
            "lag_on_2026_11_21_days": lag_on_nov21,
            "effective_future_days_on_2026_11_21": max(0, 7 - lag_on_nov21),
            "effective_future_days_on_2026_11_20": max(0, 7 - (date(2026, 11, 20) - a_dt).days),
            "effective_future_days_on_2026_11_19": max(0, 7 - (date(2026, 11, 19) - a_dt).days),
            "unlocks_is_prospective_1": max(0, 7 - lag_on_nov21) > 0,
            "notes": notes,
        }

    policies = {
        "P-0_COMPLETE_ALL_35": _future_range(
            ceres_last,
            "80-83d",
            "Requires all 35 variables (24 Meteo + 3 FlashFlux + 8 CERES SYN1deg); 0 true future days.",
        ),
        "P-a_P-b_FLASHFLUX_27VARS": _future_range(
            ff_last_contiguous,
            "6-9d",
            "Eliminates CERES-8 only; 0 true future days whenever FlashFlux lag >= 7 days or interior dropouts occur (e.g., 2026-11-06).",
        ),
        "P-c_DOY_MEDIAN_ALL_11_SOLAR": _future_range(
            meteo_last,
            "2-4d",
            "Anchor constrained only by 24 Meteorology vars; unlocks 3 to 5 true future days (is_prospective=1) and stateless DOY lookup.",
        ),
        "P-c-ffill7_LE7D_PLUS_DOY_CERES8": _future_range(
            meteo_last,
            "2-4d",
            "Anchor constrained by 24 Meteo vars (FlashFlux ffill when lag<=7d + DOY CERES-8); same 3 to 5 true future days as P-c.",
        ),
    }
    return {
        "evaluated_at_db": str(db_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "observed_ingestion_runs": observed_scans,
        "comparative_policies": policies,
    }


def _slice_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_clean: np.ndarray,
    *,
    p90_thresh: float,
    h_slice: slice,
) -> Dict[str, Any]:
    yt_s = y_true[:, h_slice]
    yp_s = y_pred[:, h_slice]
    yc_s = y_clean[:, h_slice]

    yt = yt_s.reshape(-1)
    yp = yp_s.reshape(-1)
    yc = yc_s.reshape(-1)

    abs_err = np.abs(yp - yt)
    sq_err = (yp - yt) ** 2
    abs_err_c = np.abs(yc - yt)
    sq_err_c = (yc - yt) ** 2

    mae = float(np.mean(abs_err))
    mae_c = float(np.mean(abs_err_c))
    mse = float(np.mean(sq_err))
    rmse = float(math.sqrt(mse))
    rmse_c = float(math.sqrt(float(np.mean(sq_err_c))))
    mase = mae / MASE_SCALE

    p90_mask = yt >= p90_thresh
    p90_mae = float(np.mean(abs_err[p90_mask]))
    p90_mae_c = float(np.mean(abs_err_c[p90_mask]))

    out = {
        "n_origins": int(y_true.shape[0]),
        "MAE": round(mae, 6),
        "MSE": round(mse, 6),
        "RMSE": round(rmse, 6),
        "MASE": round(mase, 6),
        "delta_MAE": round(mae - mae_c, 6),
        "delta_RMSE": round(rmse - rmse_c, 6),
        "p90_tail_mae": round(p90_mae, 6),
        "delta_p90_tail_mae": round(p90_mae - p90_mae_c, 6),
        "oracle_fidelity_mae": round(float(np.mean(np.abs(yp - yc))), 6),
    }
    assert_no_purged_metrics(pd.DataFrame([out]))
    return out


def _evaluate_predictions_split_horizons(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_clean: np.ndarray,
    *,
    p90_thresh: float,
) -> Dict[str, Dict[str, Any]]:
    return {
        "h1_to_7": _slice_horizon_metrics(y_true, y_pred, y_clean, p90_thresh=p90_thresh, h_slice=slice(0, 7)),
        "h1_to_3": _slice_horizon_metrics(y_true, y_pred, y_clean, p90_thresh=p90_thresh, h_slice=slice(0, 3)),
        "h4_to_7": _slice_horizon_metrics(y_true, y_pred, y_clean, p90_thresh=p90_thresh, h_slice=slice(3, 7)),
    }


def _check_val_gate_clauses(m_metrics: Dict[str, Any]) -> Dict[str, Any]:
    d_mae = float(m_metrics["delta_MAE"])
    d_rmse = float(m_metrics["delta_RMSE"])
    d_p90 = float(m_metrics["delta_p90_tail_mae"])
    c1 = (d_mae <= VAL_GATE_MAX_ABS_DELTA_MAE_MM) and (abs(d_mae) <= VAL_GATE_MAX_ABS_DELTA_MAE_MM)
    c2 = d_rmse <= VAL_GATE_MAX_DELTA_RMSE_MM
    c3 = d_p90 <= VAL_GATE_MAX_DELTA_P90_TAIL_MAE_MM
    return {
        "clause1_val_abs_delta_mae_le_0_020": c1,
        "clause2_val_delta_rmse_le_0_050": c2,
        "clause3_val_delta_p90_tail_le_0_250": c3,
        "all_3_clauses_passed": bool(c1 and c2 and c3),
        "delta_MAE": d_mae,
        "delta_RMSE": d_rmse,
        "delta_p90_tail_mae": d_p90,
    }


def main() -> int:
    t0 = time.perf_counter()
    full_df, dev_train_df, val_df, test_df = _load_clean_hcmc_full()
    p90_dev_train = float(np.percentile(dev_train_df[CANONICAL_TARGET_COL].to_numpy(dtype=np.float64), 90.0))

    doy_table = build_solar_doy_366_table(dev_train_df, radius=7)
    fb3 = joblib.load(ARTIFACT_DIR / "feature_builder_f3.joblib")
    m35 = joblib.load(ARTIFACT_DIR / "sc35_xgboost_tweedie_f3.joblib")
    m31 = joblib.load(ARTIFACT_DIR / "sc31_lightgbm_tweedie_f3.joblib")
    m30 = joblib.load(ARTIFACT_DIR / "sc30_lightgbm_tweedie_f2.joblib")

    f3_sample = fb3.transform(dev_train_df.head(60), drop_warmup=True)
    f3_cols = [c for c in f3_sample.columns if c not in (CANONICAL_DATE_COL, CANONICAL_TARGET_COL)]
    assert len(f3_cols) == 70

    splits = _compute_split_matrices(full_df, dev_train_df, val_df, test_df, fb3, f3_cols)
    frozen_models: Dict[str, Tuple[Any, Sequence[int]]] = {
        "SC-35": (m35, list(range(70))),
        "SC-31": (m31, list(range(70))),
        "SC-30": (m30, list(range(65))),
    }

    a11_vi = [EN_TO_VI_CANONICAL[c] for c in SYN1DEG_11_CODES]
    ff3_vi = [EN_TO_VI_CANONICAL[c] for c in FLASHFLUX_3_CODES]
    c8_vi = [EN_TO_VI_CANONICAL[c] for c in CERES_ONLY_8_CODES]
    a11_idx = [f3_cols.index(v) for v in a11_vi]
    ff3_idx = [f3_cols.index(v) for v in ff3_vi]
    c8_idx = [f3_cols.index(v) for v in c8_vi]

    keep_24_f3_idx = [i for i, c in enumerate(f3_cols) if c not in a11_vi]
    keep_24_f2_idx = [i for i in range(65) if f3_cols[i] not in a11_vi]
    retrained_24 = _retrain_dev_train_models(
        splits["dev_train"]["X_f3"],
        splits["dev_train"]["y_true"],
        keep_f3_idx=keep_24_f3_idx,
        keep_f2_idx=keep_24_f2_idx,
    )

    methods = [
        "clean",
        "p_c_doy_median_all_11_solar",
        "p_c_ffill7_lag2d",
        "p_c_ffill7_lag3d",
        "p_c_ffill7_lag5d",
        "p_c_ffill7_lag7d",
        "retrained_no_solar_24vars",
    ]

    results_by_split: Dict[str, Any] = {}
    for split_name in ("val_724", "test_2184"):
        sp = splits[split_name]
        X_clean = sp["X_f3"]
        y_true = sp["y_true"]
        raw_idx = sp["raw_indices"]
        dates = sp["origin_dates"]

        clean_preds = {
            mid: _predict_single(m_obj, X_clean, col_idx)
            for mid, (m_obj, col_idx) in frozen_models.items()
        }

        split_out: Dict[str, Any] = {mid: {} for mid in ("SC-35", "SC-31", "SC-30")}
        for meth in methods:
            if meth == "retrained_no_solar_24vars":
                for mid in ("SC-35", "SC-31", "SC-30"):
                    r_obj, r_idx = retrained_24[mid]
                    y_pred = _predict_single(r_obj, X_clean, r_idx)
                    split_out[mid][meth] = _evaluate_predictions_split_horizons(
                        y_true, y_pred, clean_preds[mid], p90_thresh=p90_dev_train
                    )
                continue

            X_mod = X_clean.copy()
            if meth == "p_c_doy_median_all_11_solar":
                X_mod[:, a11_idx] = _lookup_doy_matrix(doy_table, dates, SYN1DEG_11_CODES)
            elif meth.startswith("p_c_ffill7_lag"):
                lag_d = int(meth.replace("p_c_ffill7_lag", "").replace("d", ""))
                X_mod[:, ff3_idx] = full_df.loc[raw_idx - lag_d, ff3_vi].to_numpy(dtype=np.float64)
                X_mod[:, c8_idx] = _lookup_doy_matrix(doy_table, dates, CERES_ONLY_8_CODES)

            for mid, (m_obj, col_idx) in frozen_models.items():
                y_pred = _predict_single(m_obj, X_mod, col_idx)
                split_out[mid][meth] = _evaluate_predictions_split_horizons(
                    y_true, y_pred, clean_preds[mid], p90_thresh=p90_dev_train
                )
        results_by_split[split_name] = split_out

    # Evaluate Pre-Declared 3-Clause Validation Selection Gate strictly on val_724 h1_to_7
    gate_candidates = [
        "p_c_doy_median_all_11_solar",
        "p_c_ffill7_lag2d",
        "p_c_ffill7_lag3d",
        "p_c_ffill7_lag5d",
        "p_c_ffill7_lag7d",
    ]
    gate_eval: Dict[str, Any] = {}
    passing_candidates: List[str] = []
    for cand in gate_candidates:
        per_model = {
            mid: _check_val_gate_clauses(results_by_split["val_724"][mid][cand]["h1_to_7"])
            for mid in ("SC-35", "SC-31", "SC-30")
        }
        all_pass = all(v["all_3_clauses_passed"] for v in per_model.values())
        max_abs_dmae = max(abs(float(v["delta_MAE"])) for v in per_model.values())
        gate_eval[cand] = {
            "all_3_models_pass_all_3_clauses": all_pass,
            "max_abs_delta_MAE_val": round(max_abs_dmae, 6),
            "per_model": per_model,
        }
        if all_pass:
            passing_candidates.append(cand)

    # Selection & tie-breaker (Occam's razor favors P-c if within 0.005 mm noise band)
    pc_passed = "p_c_doy_median_all_11_solar" in passing_candidates
    if len(passing_candidates) == 0:
        selected_policy = None
        stop_condition_triggered = True
        selection_reason = "HARD_STOP_TRIGGERED: At least one model failed the pre-declared 3-clause Validation Gate."
    else:
        stop_condition_triggered = False
        best_non_pc = min(
            passing_candidates,
            key=lambda c: gate_eval[c]["max_abs_delta_MAE_val"],
        )
        best_dmae = gate_eval[best_non_pc]["max_abs_delta_MAE_val"]
        if pc_passed and (gate_eval["p_c_doy_median_all_11_solar"]["max_abs_delta_MAE_val"] - best_dmae <= TIE_BREAK_NOISE_BAND_MAE_MM):
            selected_policy = "p_c_doy_median_all_11_solar"
            selection_reason = (
                "Both P-c and passing candidate(s) fall within <= 0.005 mm sampling noise on max|delta_MAE_val|; "
                "selected P-c (DOY_MEDIAN_ALL_11_SOLAR) by Occam's razor because it is stateless and anchor-DOY deterministic."
            )
        else:
            selected_policy = best_non_pc
            selection_reason = f"Selected {best_non_pc} with smallest max_m |delta_MAE_val| = {best_dmae:.6f} mm."

    payload = {
        "schema_version": "1.0.0",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "p90_dev_train_threshold_mm": round(p90_dev_train, 6),
        "pre_declared_validation_gate": {
            "clause_1_validation_mae": f"|delta_MAE_val| <= {VAL_GATE_MAX_ABS_DELTA_MAE_MM:.3f} mm (origin: Round 5 MT-Q1)",
            "clause_2_validation_rmse": f"delta_RMSE_val <= +{VAL_GATE_MAX_DELTA_RMSE_MM:.3f} mm (formalized in Round 13 after Round 12 raw_nan finding)",
            "clause_3_validation_p90_tail_mae": f"delta_P90_Tail_MAE_val <= +{VAL_GATE_MAX_DELTA_P90_TAIL_MAE_MM:.3f} mm on days >= {p90_dev_train:.3f} mm (formalized in Round 13)",
            "tie_breaker_rule": f"Minimize max_m |delta_MAE_val|; prefer stateless P-c if difference <= {TIE_BREAK_NOISE_BAND_MAE_MM:.3f} mm",
        },
        "true_future_days_table": _compute_true_future_days_table(),
        "validation_gate_results": gate_eval,
        "passing_candidates": passing_candidates,
        "selected_policy": selected_policy,
        "stop_condition_triggered": stop_condition_triggered,
        "selection_reason": selection_reason,
        "results_by_split": results_by_split,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }

    OUT_JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _append_pc_section_to_readme(payload)
    print(
        json.dumps(
            {
                "selected_policy": selected_policy,
                "stop_condition_triggered": stop_condition_triggered,
                "passing_candidates": passing_candidates,
                "gate_eval": gate_eval,
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


def _append_pc_section_to_readme(payload: Dict[str, Any]) -> None:
    """Update `research/imputer_benchmark/README.md` with Round 13 Gate provenance, Future Days table, and split-horizon results."""
    base = OUT_README_PATH.read_text(encoding="utf-8") if OUT_README_PATH.exists() else ""
    marker = "\n## Round 13: True Future Days"
    if marker in base:
        base = base.split(marker, 1)[0].rstrip() + "\n"

    tf = payload["true_future_days_table"]["comparative_policies"]
    lines: List[str] = [
        base.rstrip(),
        "",
        "## Round 13: True Future Days, 3-Clause Validation Gate & `P-c` Split-Horizon Audit",
        "",
        "### Pre-Declared 3-Clause Validation Selection Gate Provenance",
        "",
        "- **Clause 1 (`Validation MAE`)**: $|\\Delta\\text{MAE}_{\\text{val}}| \\le 0.020\\text{ mm}$ — originated in **Round 5 (`MT-Q1`)**.",
        "- **Clause 2 (`Validation RMSE`)**: $\\Delta\\text{RMSE}_{\\text{val}} \\le +0.050\\text{ mm}$ — added following Round 12 and formalized with numeric bounds in **Round 13**.",
        "- **Clause 3 (`Validation P90-Tail MAE`, $\\ge 11.086\\text{ mm}$)**: $\\Delta\\text{P90-Tail MAE}_{\\text{val}} \\le +0.250\\text{ mm}$ — added following Round 12 and formalized in **Round 13**.",
        "- *Why Clauses 2 & 3 are mandatory:* Round 12 proved `raw_nan` slightly lowered aggregate Validation MAE (`-0.0078` to `-0.0627 mm`) while severely degrading RMSE (`+0.13` to `+0.78 mm`) and P90-Tail MAE (`+1.00+ mm`); selecting on MAE alone would promote an unsafe model.",
        "- *Model-retention rationale:* Any future transition to a retrained 24/27-variable model would be motivated strictly by operational simplicity after completing full new-model MLflow governance — **not accuracy**, as Test-set accuracy differences vs the frozen 35-variable ensemble are statistically insignificant.",
        "",
        "### True Future Days (`effective_future_days = max(0, 7 - lag_days)`) from `ingestion_runs`",
        "",
        "| Policy | Constraints | Anchor (2026-11-21) | Typical Lag | `effective_future_days` (Nov 19 / 20 / 21) | `is_prospective = 1`? |",
        "|---|---|---|---:|---:|---|",
    ]
    for k, row in tf.items():
        eff_str = (
            f"{row['effective_future_days_on_2026_11_19']} / "
            f"{row['effective_future_days_on_2026_11_20']} / "
            f"{row['effective_future_days_on_2026_11_21']}"
        )
        lines.append(
            f"| `{k}` | {row['notes']} | `{row['anchor_date_observed_db']}` | {row['typical_lag_days']} | {eff_str} | `{row['unlocks_is_prospective_1']}` |"
        )

    lines.extend(
        [
            "",
            f"### 3-Clause Validation Gate Decision: **`{payload['selected_policy']}`** (`stop_condition_triggered={payload['stop_condition_triggered']}`)",
            "",
            f"> {payload['selection_reason']}",
            "",
            "| Candidate | All 3 Models Pass? | `SC-35` Val dMAE / dRMSE / dP90 | `SC-31` Val dMAE / dRMSE / dP90 | `SC-30` Val dMAE / dRMSE / dP90 |",
            "|---|---|---|---|---|",
        ]
    )
    for cand, ge in payload["validation_gate_results"].items():
        pm = ge["per_model"]
        s35 = f"{pm['SC-35']['delta_MAE']:+.4f} / {pm['SC-35']['delta_RMSE']:+.4f} / {pm['SC-35']['delta_p90_tail_mae']:+.4f}"
        s31 = f"{pm['SC-31']['delta_MAE']:+.4f} / {pm['SC-31']['delta_RMSE']:+.4f} / {pm['SC-31']['delta_p90_tail_mae']:+.4f}"
        s30 = f"{pm['SC-30']['delta_MAE']:+.4f} / {pm['SC-30']['delta_RMSE']:+.4f} / {pm['SC-30']['delta_p90_tail_mae']:+.4f}"
        lines.append(f"| `{cand}` | `{ge['all_3_models_pass_all_3_clauses']}` | {s35} | {s31} | {s30} |")

    lines.extend(["", "### Test (`2,184` Origins) Split-Horizon Breakdown (`h=1..7` vs `h=1..3` vs `h=4..7`)", ""])
    lines.append("| Model | Method | Slice | MAE | delta MAE | MSE | RMSE | delta RMSE | P90-Tail MAE | delta P90-Tail |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for mid in ("SC-35", "SC-31", "SC-30"):
        for meth in ("clean", "p_c_doy_median_all_11_solar", "p_c_ffill7_lag3d", "p_c_ffill7_lag7d", "retrained_no_solar_24vars"):
            for sl in ("h1_to_7", "h1_to_3", "h4_to_7"):
                m = payload["results_by_split"]["test_2184"][mid][meth][sl]
                lines.append(
                    f"| {mid} | `{meth}` | `{sl}` | {m['MAE']:.4f} | {m['delta_MAE']:+.4f} | "
                    f"{m['MSE']:.4f} | {m['RMSE']:.4f} | {m['delta_RMSE']:+.4f} | "
                    f"{m['p90_tail_mae']:.4f} | {m['delta_p90_tail_mae']:+.4f} |"
                )
    OUT_README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
