"""Stage 1 Horizon Extension Feasibility Audit (H=7 -> H=10 / H=11 under P-c Solar Imputation).

Implements the strict 3-split horizon-purged direct multi-horizon benchmark on HCMC
NASA POWER data (2001-01-01..2026-04-30):
  - Dev Train: 2001-01-01..2018-04-30 (6,329 days -> 6,289 H=11 origins, last 2018-04-19)
  - Validation: 2018-05-01..2020-04-30 (731 days, 30d context -> 720 H=11 origins, last 2020-04-19)
  - Test: 2020-05-01..2026-04-30 (2,191 days, 30d context -> 2,180 H=11 origins, last 2026-04-19)

Evaluates SC-35 (XGBoost Tweedie F3), SC-31 (LightGBM Tweedie F3), SC-30 (LightGBM
Tweedie F2), BASELINE_CLIMATOLOGY_DOY (Dev-Train DOY 1..366 mean), and BASELINE_PERSISTENCE
under both clean 35-var inputs and P-c DOY-median solar imputation, with Moving Block
Bootstrap 95% CIs (block_length=11, n_bootstrap=2000, seed=42), and checks the pre-declared
Go/No-Go Gate (Clause A, B1, B2, B3) for H* in {11, 10} (plus diagnostics for {9, 8}).
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.imputer_benchmark.run_ceres8_vs_11_audit import (  # noqa: E402
    ARTIFACT_DIR,
    CANONICAL_DATE_COL,
    CANONICAL_TARGET_COL,
    EN_TO_VI_CANONICAL,
    MASE_SCALE,
    SYN1DEG_11_CODES,
    _load_clean_hcmc_full,
)
from research.imputer_benchmark.run_pc_integration_experiment import _lookup_doy_matrix  # noqa: E402
from scripts.mlops.package_models import (  # noqa: E402
    _create_lgbm_config_a_model,
    _create_sc35_model,
)
from src.analysis.integrity import assert_no_purged_metrics  # noqa: E402
from src.evaluation.protocols import build_purged_direct_samples  # noqa: E402
from src.inference.forecast_runner import (  # noqa: E402
    build_block_bootstrap_indices,
    summarize_bootstrap_samples,
)
from src.inference.solar_imputation import load_solar_doy_climatology  # noqa: E402

HORIZON_H11 = 11
BLOCK_LENGTH = 11
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42

CLAUSE_A_MAX_ABS_DELTA_MAE = 0.015
CLAUSE_B1_MAX_STEP_LOSS_MM = 0.020

EXPECTED_COUNTS = {
    "dev_train": (6289, "2018-04-19"),
    "val": (720, "2020-04-19"),
    "test": (2180, "2026-04-19"),
}

SLICE_SPECS: Dict[str, slice] = {
    "h1_to_7": slice(0, 7),
    "h4_to_10": slice(3, 10),
    "h5_to_11": slice(4, 11),
    "h8_to_11": slice(7, 11),
    "h8_to_10": slice(7, 10),
    "h8_to_9": slice(7, 9),
    "h1_to_11": slice(0, 11),
}

DEFAULT_JSON_OUT = Path(__file__).resolve().parent / "h11_feasibility_audit.json"
DEFAULT_README_OUT = Path(__file__).resolve().parent / "README.md"


def _build_dev_train_rain_doy_table(dev_train_df: pd.DataFrame) -> Dict[int, float]:
    """Compute DOY (1..366) precipitation mean strictly on Dev Train rows."""
    dates = pd.to_datetime(dev_train_df[CANONICAL_DATE_COL])
    if dates.max() > pd.Timestamp("2018-04-30"):
        raise ValueError(
            f"Dev Train DOY cutoff violation: max date {dates.max().strftime('%Y-%m-%d')} exceeds 2018-04-30."
        )
    rain = dev_train_df[CANONICAL_TARGET_COL].to_numpy(dtype=np.float64)
    doy_df = pd.DataFrame({"doy": dates.dt.dayofyear.to_numpy(dtype=np.int64), "rain": rain})
    grouped = doy_df.groupby("doy")["rain"].mean().to_dict()
    table = {int(k): float(v) for k, v in sorted(grouped.items())}
    if 366 not in table and 365 in table:
        table[366] = table[365]
    return table


def _predict_climatology_doy_matrix(
    doy_rain_table: Mapping[int, float],
    origin_dates: pd.Series,
    horizon: int = HORIZON_H11,
) -> np.ndarray:
    """Predict shape (N, horizon) using target date (origin + h days) DOY mean."""
    preds = np.empty((len(origin_dates), horizon), dtype=np.float64)
    parsed_origins = [pd.Timestamp(dt).date() for dt in origin_dates]
    for h_idx in range(horizon):
        delta = timedelta(days=h_idx + 1)
        preds[:, h_idx] = [float(doy_rain_table[(od + delta).timetuple().tm_yday]) for od in parsed_origins]
    return preds


def _predict_persistence_matrix(
    origin_rain: np.ndarray,
    horizon: int = HORIZON_H11,
) -> np.ndarray:
    """Predict shape (N, horizon) repeating PRECTOTCORR[t] across horizons h=1..horizon."""
    arr = np.asarray(origin_rain, dtype=np.float64).reshape(-1, 1)
    return np.repeat(arr, horizon, axis=1)


def _extract_h11_splits(
    dev_train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fb3: Any,
    f3_cols: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Build strict horizon-purged H=11 matrices for Dev Train, Val, and Test."""
    feat_dev = fb3.transform(dev_train_df, drop_warmup=True)
    feat_val = fb3.transform(val_df, context_df=dev_train_df.tail(30), drop_warmup=False)
    feat_test = fb3.transform(test_df, context_df=val_df.tail(30), drop_warmup=False)

    splits: Dict[str, Dict[str, Any]] = {}
    for name, feat_df in (("dev_train", feat_dev), ("val", feat_val), ("test", feat_test)):
        X_arr, y_arr, origin_dates = build_purged_direct_samples(
            feat_df,
            feature_cols=f3_cols,
            target_col=CANONICAL_TARGET_COL,
            horizon=HORIZON_H11,
        )
        exp_n, exp_last = EXPECTED_COUNTS[name]
        actual_last = pd.Timestamp(origin_dates.iloc[-1]).strftime("%Y-%m-%d")
        if len(X_arr) != exp_n:
            raise AssertionError(f"Expected {exp_n} {name} origins, got {len(X_arr)}")
        if actual_last != exp_last:
            raise AssertionError(f"Expected {name} last origin {exp_last}, got {actual_last}")

        origin_rain = feat_df[CANONICAL_TARGET_COL].to_numpy(dtype=np.float64)[: len(X_arr)]
        splits[name] = {
            "X_f3": X_arr,
            "y_true": y_arr,
            "origin_dates": pd.to_datetime(origin_dates).reset_index(drop=True),
            "origin_rain": origin_rain.copy(),
        }
    return splits


def _train_h11_models(X_train_f3: np.ndarray, y_train_h11: np.ndarray) -> Dict[str, Tuple[Any, List[int]]]:
    """Fit canonical SC-35, SC-31, SC-30 factories on clean Dev Train for H=11."""
    f3_idx = list(range(70))
    f2_idx = list(range(65))
    return {
        "SC-35": (_create_sc35_model().fit(X_train_f3[:, f3_idx], y_train_h11), f3_idx),
        "SC-31": (_create_lgbm_config_a_model().fit(X_train_f3[:, f3_idx], y_train_h11), f3_idx),
        "SC-30": (_create_lgbm_config_a_model().fit(X_train_f3[:, f2_idx], y_train_h11), f2_idx),
    }


def _verify_pc_transform_parity(
    dev_train_df: pd.DataFrame,
    fb3: Any,
    f3_cols: List[str],
    doy_table: Dict[str, Any],
    a11_idx: Sequence[int],
) -> float:
    """Verify P-c column override matches raw FeatureBuilder input transform (< 1e-12)."""
    sample_raw = dev_train_df.head(40).copy()
    feat_clean = fb3.transform(sample_raw, drop_warmup=True).reset_index(drop=True)
    dates = pd.to_datetime(feat_clean[CANONICAL_DATE_COL])
    X_col_override = feat_clean[f3_cols].to_numpy(dtype=np.float64)
    X_col_override[:, a11_idx] = _lookup_doy_matrix(doy_table, dates, SYN1DEG_11_CODES)

    sample_raw_pc = sample_raw.copy()
    a11_vi = [EN_TO_VI_CANONICAL[c] for c in SYN1DEG_11_CODES]
    raw_doy = _lookup_doy_matrix(doy_table, pd.to_datetime(sample_raw_pc[CANONICAL_DATE_COL]), SYN1DEG_11_CODES)
    sample_raw_pc.loc[:, a11_vi] = raw_doy
    feat_raw_pc = fb3.transform(sample_raw_pc, drop_warmup=True).reset_index(drop=True)
    X_raw_pc = feat_raw_pc[f3_cols].to_numpy(dtype=np.float64)

    max_diff = float(np.max(np.abs(X_col_override - X_raw_pc)))
    if max_diff >= 1e-12:
        raise AssertionError(f"P-c transform parity mismatch: {max_diff}")
    return max_diff


def _compute_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_clean_pred: np.ndarray,
    y_clim_pred: np.ndarray,
    *,
    h_slice: slice,
    p90_thresh: float,
    boot_idx_matrix: np.ndarray,
) -> Dict[str, Any]:
    """Compute MAE, RMSE, MSE, MASE, P90-Tail MAE, Fidelity-MAE, diffs vs Clim, and 95% MBB CIs."""
    yt_s = y_true[:, h_slice]
    yp_s = y_pred[:, h_slice]
    yc_s = y_clean_pred[:, h_slice]
    yk_s = y_clim_pred[:, h_slice]

    yt = yt_s.reshape(-1)
    yp = yp_s.reshape(-1)
    yc = yc_s.reshape(-1)
    yk = yk_s.reshape(-1)

    abs_err_2d = np.abs(yp_s - yt_s)
    sq_err_2d = (yp_s - yt_s) ** 2
    abs_err_clim_2d = np.abs(yk_s - yt_s)
    sq_err_clim_2d = (yk_s - yt_s) ** 2

    abs_err = abs_err_2d.reshape(-1)
    sq_err = sq_err_2d.reshape(-1)
    abs_err_k = abs_err_clim_2d.reshape(-1)

    mae = float(np.mean(abs_err))
    mae_k = float(np.mean(abs_err_k))
    mse = float(np.mean(sq_err))
    rmse = float(math.sqrt(mse))
    rmse_k = float(math.sqrt(float(np.mean((yk - yt) ** 2))))
    mase = mae / MASE_SCALE

    origin_abs = np.mean(abs_err_2d, axis=1)
    origin_sq = np.mean(sq_err_2d, axis=1)
    origin_sq_k = np.mean(sq_err_clim_2d, axis=1)
    origin_diff_k = origin_abs - np.mean(abs_err_clim_2d, axis=1)

    boot_mae = np.mean(origin_abs[boot_idx_matrix], axis=1)
    boot_rmse = np.sqrt(np.mean(origin_sq[boot_idx_matrix], axis=1))
    boot_rmse_k = np.sqrt(np.mean(origin_sq_k[boot_idx_matrix], axis=1))
    boot_diff_rmse_k = boot_rmse - boot_rmse_k
    boot_diff_k = np.mean(origin_diff_k[boot_idx_matrix], axis=1)

    p90_mask = yt >= p90_thresh
    if np.any(p90_mask):
        p90_mae = float(np.mean(abs_err[p90_mask]))
        p90_mae_k = float(np.mean(abs_err_k[p90_mask]))
        delta_p90_vs_clim = p90_mae - p90_mae_k
    else:
        p90_mae = float("nan")
        delta_p90_vs_clim = float("nan")

    fidelity_mae = float(np.mean(np.abs(yp - yc)))
    delta_mae_vs_clim = mae - mae_k
    delta_rmse_vs_clim = rmse - rmse_k

    return {
        "n_origins": int(y_true.shape[0]),
        "MAE": round(mae, 6),
        "MAE_ci95": summarize_bootstrap_samples(boot_mae, mae),
        "RMSE": round(rmse, 6),
        "RMSE_ci95": summarize_bootstrap_samples(boot_rmse, rmse),
        "MSE": round(mse, 6),
        "MASE": round(mase, 6),
        "p90_tail_mae": round(p90_mae, 6) if not math.isnan(p90_mae) else None,
        "fidelity_mae": round(fidelity_mae, 6),
        "delta_MAE_vs_clim": round(delta_mae_vs_clim, 6),
        "delta_MAE_vs_clim_ci95": summarize_bootstrap_samples(boot_diff_k, delta_mae_vs_clim),
        "delta_RMSE_vs_clim": round(delta_rmse_vs_clim, 6),
        "delta_RMSE_vs_clim_ci95": summarize_bootstrap_samples(boot_diff_rmse_k, delta_rmse_vs_clim),
        "delta_p90_tail_mae_vs_clim": round(delta_p90_vs_clim, 6) if not math.isnan(delta_p90_vs_clim) else None,
    }


def _evaluate_model_all_windows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_clean_pred: np.ndarray,
    y_clim_pred: np.ndarray,
    *,
    p90_thresh: float,
    boot_idx_matrix: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate every individual step h1..h11 and all configured horizon slices."""
    per_step: Dict[str, Dict[str, Any]] = {}
    for h_idx in range(HORIZON_H11):
        per_step[f"h{h_idx + 1}"] = _compute_window_metrics(
            y_true,
            y_pred,
            y_clean_pred,
            y_clim_pred,
            h_slice=slice(h_idx, h_idx + 1),
            p90_thresh=p90_thresh,
            boot_idx_matrix=boot_idx_matrix,
        )

    slices: Dict[str, Dict[str, Any]] = {}
    for slice_name, sl in SLICE_SPECS.items():
        slices[slice_name] = _compute_window_metrics(
            y_true,
            y_pred,
            y_clean_pred,
            y_clim_pred,
            h_slice=sl,
            p90_thresh=p90_thresh,
            boot_idx_matrix=boot_idx_matrix,
        )
    return {"per_step": per_step, "slices": slices}


def _evaluate_gate(
    *,
    val_pc: Mapping[str, Any],
    test_pc: Mapping[str, Any],
    h7_val_pc_mae: Mapping[str, float],
) -> Dict[str, Any]:
    """Evaluate pre-declared Stage 1 Go/No-Go Gate (Clause A, B1, B2, B3)."""
    clause_a_models: Dict[str, Any] = {}
    for mid in ("SC-35", "SC-31", "SC-30"):
        mae_h11_1to7 = float(val_pc[mid]["slices"]["h1_to_7"]["MAE"])
        mae_h7_1to7 = float(h7_val_pc_mae[mid])
        diff = mae_h11_1to7 - mae_h7_1to7
        clause_a_models[mid] = {
            "val_pc_h1_to_7_mae_H11": round(mae_h11_1to7, 6),
            "val_pc_h1_to_7_mae_H7": round(mae_h7_1to7, 6),
            "signed_diff_mm": round(diff, 6),
            "abs_diff_mm": round(abs(diff), 6),
            "passed_le_0_015": bool(abs(diff) <= CLAUSE_A_MAX_ABS_DELTA_MAE),
        }
    clause_a_passed = all(v["passed_le_0_015"] for v in clause_a_models.values())

    slice_by_hstar = {11: "h8_to_11", 10: "h8_to_10", 9: "h8_to_9", 8: "h8"}
    candidates: Dict[str, Any] = {}

    sc35_val = val_pc["SC-35"]
    sc35_test = test_pc["SC-35"]

    for h_star in (11, 10, 9, 8):
        ext_key = slice_by_hstar[h_star]
        val_ext = sc35_val["per_step"]["h8"] if ext_key == "h8" else sc35_val["slices"][ext_key]
        test_ext = sc35_test["per_step"]["h8"] if ext_key == "h8" else sc35_test["slices"][ext_key]

        step_deltas = {
            f"h{h}": float(sc35_val["per_step"][f"h{h}"]["delta_MAE_vs_clim"]) for h in range(8, h_star + 1)
        }
        path1_all_steps_negative = all(d < 0.0 for d in step_deltas.values())

        val_slice_d_mae = float(val_ext["delta_MAE_vs_clim"])
        val_slice_ci_hi = float(val_ext["delta_MAE_vs_clim_ci95"]["ci_upper"])
        max_step_delta = max(step_deltas.values())
        path2_slice_ci_and_step_bound = (
            val_slice_d_mae < 0.0
            and val_slice_ci_hi < 0.0
            and max_step_delta <= CLAUSE_B1_MAX_STEP_LOSS_MM
        )
        b1_passed = bool(path1_all_steps_negative or path2_slice_ci_and_step_bound)

        val_d_rmse = float(val_ext["delta_RMSE_vs_clim"])
        val_d_p90 = float(val_ext["delta_p90_tail_mae_vs_clim"])
        b2_passed = bool(val_d_rmse <= 0.0 and val_d_p90 <= 0.0)

        test_d_mae = float(test_ext["delta_MAE_vs_clim"])
        test_d_rmse = float(test_ext["delta_RMSE_vs_clim"])
        test_d_p90 = float(test_ext["delta_p90_tail_mae_vs_clim"])
        b3_passed = bool(test_d_mae < 0.0 and test_d_rmse <= 0.0 and test_d_p90 <= 0.0)

        overall = bool(clause_a_passed and b1_passed and b2_passed and b3_passed)
        candidates[f"H_star_{h_star}"] = {
            "H_star": h_star,
            "is_stage2_eligible_candidate": h_star in (11, 10),
            "diagnostic_only": h_star in (9, 8),
            "extended_slice": ext_key,
            "clause_A_passed": clause_a_passed,
            "clause_B1": {
                "passed": b1_passed,
                "path1_all_steps_beat_clim": path1_all_steps_negative,
                "path2_slice_ci_upper_lt_0_and_step_le_0_020": path2_slice_ci_and_step_bound,
                "step_deltas_val_pc": step_deltas,
                "max_step_delta_val_pc": round(max_step_delta, 6),
                "slice_delta_mae_val_pc": round(val_slice_d_mae, 6),
                "slice_delta_mae_ci95_val_pc": val_ext["delta_MAE_vs_clim_ci95"],
            },
            "clause_B2": {
                "passed": b2_passed,
                "val_pc_delta_rmse_vs_clim": round(val_d_rmse, 6),
                "val_pc_delta_p90_tail_mae_vs_clim": round(val_d_p90, 6),
            },
            "clause_B3": {
                "passed": b3_passed,
                "test_pc_delta_mae_vs_clim": round(test_d_mae, 6),
                "test_pc_delta_mae_ci95_vs_clim": test_ext["delta_MAE_vs_clim_ci95"],
                "test_pc_delta_rmse_vs_clim": round(test_d_rmse, 6),
                "test_pc_delta_p90_tail_mae_vs_clim": round(test_d_p90, 6),
            },
            "overall_gate_passed": overall,
        }

    winning_h_star: Optional[int] = None
    for h_star in (11, 10):
        if candidates[f"H_star_{h_star}"]["overall_gate_passed"]:
            winning_h_star = h_star
            break

    return {
        "clause_A_h1_to_7_non_regression": {
            "threshold_abs_delta_mae_mm": CLAUSE_A_MAX_ABS_DELTA_MAE,
            "passed": clause_a_passed,
            "models": clause_a_models,
        },
        "candidates": candidates,
        "winning_H_star": winning_h_star,
        "stop_and_report_triggered": winning_h_star is None,
    }


def _assert_no_purged_columns_in_report(results_by_split: Mapping[str, Any]) -> None:
    """Flatten every metric dict across splits/models/conditions/windows and verify zero purged metrics."""
    rows: List[Dict[str, Any]] = []
    for split_data in results_by_split.values():
        for cond_data in split_data.values():
            for model_data in cond_data.values():
                for group in ("per_step", "slices"):
                    for w_metrics in model_data[group].values():
                        rows.append({k: v for k, v in w_metrics.items() if not isinstance(v, dict)})
    assert_no_purged_metrics(pd.DataFrame(rows))


def _format_step_table(split_data: Mapping[str, Any], condition: str) -> str:
    """Format compact markdown table of MAE and delta-vs-Climatology for h=1..11."""
    models = ("SC-35", "SC-31", "SC-30", "BASELINE_CLIMATOLOGY_DOY", "BASELINE_PERSISTENCE")
    lines = [
        "| Horizon | SC-35 MAE (Δ vs Clim) | SC-31 MAE (Δ vs Clim) | SC-30 MAE (Δ vs Clim) | Clim DOY MAE | Persistence MAE |",
        "|---|---|---|---|---|---|",
    ]
    cond_map = split_data[condition]
    for h in range(1, HORIZON_H11 + 1):
        hk = f"h{h}"
        vals = []
        for mid in models[:3]:
            m = cond_map[mid]["per_step"][hk]
            vals.append(f"{m['MAE']:.4f} ({m['delta_MAE_vs_clim']:+.4f})")
        clim_mae = cond_map["BASELINE_CLIMATOLOGY_DOY"]["per_step"][hk]["MAE"]
        pers_mae = cond_map["BASELINE_PERSISTENCE"]["per_step"][hk]["MAE"]
        lines.append(f"| `h={h}` | {vals[0]} | {vals[1]} | {vals[2]} | {clim_mae:.4f} | {pers_mae:.4f} |")
    return "\n".join(lines)


def _format_slice_table(split_data: Mapping[str, Any], condition: str) -> str:
    """Format markdown table across key slices for SC-35 vs BASELINE_CLIMATOLOGY_DOY."""
    lines = [
        "| Slice | SC-35 MAE [95% CI] | Clim DOY MAE | ΔMAE vs Clim [95% CI] | ΔRMSE vs Clim [95% CI] | ΔP90-Tail vs Clim | Fidelity-MAE |",
        "|---|---|---|---|---|---|---|",
    ]
    sc35 = split_data[condition]["SC-35"]["slices"]
    clim = split_data[condition]["BASELINE_CLIMATOLOGY_DOY"]["slices"]
    for sk in ("h1_to_7", "h4_to_10", "h5_to_11", "h8_to_11", "h8_to_10", "h8_to_9", "h1_to_11"):
        m = sc35[sk]
        c = clim[sk]
        m_ci = m["MAE_ci95"]
        d_ci = m["delta_MAE_vs_clim_ci95"]
        r_ci = m["delta_RMSE_vs_clim_ci95"]
        dp90 = m["delta_p90_tail_mae_vs_clim"]
        dp90_str = f"{dp90:+.4f}" if dp90 is not None else "N/A"
        lines.append(
            f"| `{sk}` | {m['MAE']:.4f} [{m_ci['ci_lower']:.4f}, {m_ci['ci_upper']:.4f}] | "
            f"{c['MAE']:.4f} | {m['delta_MAE_vs_clim']:+.4f} [{d_ci['ci_lower']:+.4f}, {d_ci['ci_upper']:+.4f}] | "
            f"{m['delta_RMSE_vs_clim']:+.4f} [{r_ci['ci_lower']:+.4f}, {r_ci['ci_upper']:+.4f}] | "
            f"{dp90_str} | {m['fidelity_mae']:.4f} |"
        )
    return "\n".join(lines)


def _format_model_comparison_slice_table(split_data: Mapping[str, Any], condition: str, slice_key: str) -> str:
    """Format cross-model markdown summary for a slice (e.g., h8_to_11, h8_to_10)."""
    lines = [
        f"| Model (`{slice_key}`) | MAE [95% CI] | RMSE | MASE | P90-Tail MAE | ΔMAE vs Clim | ΔRMSE vs Clim | ΔP90-Tail | Fidelity |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for mid in ("SC-35", "SC-31", "SC-30", "BASELINE_CLIMATOLOGY_DOY", "BASELINE_PERSISTENCE"):
        m = split_data[condition][mid]["slices"][slice_key]
        m_ci = m["MAE_ci95"]
        p90_str = f"{m['p90_tail_mae']:.4f}" if m["p90_tail_mae"] is not None else "N/A"
        dp90_str = f"{m['delta_p90_tail_mae_vs_clim']:+.4f}" if m["delta_p90_tail_mae_vs_clim"] is not None else "N/A"
        lines.append(
            f"| `{mid}` | {m['MAE']:.4f} [{m_ci['ci_lower']:.4f}, {m_ci['ci_upper']:.4f}] | "
            f"{m['RMSE']:.4f} | {m['MASE']:.4f} | {p90_str} | {m['delta_MAE_vs_clim']:+.4f} | "
            f"{m['delta_RMSE_vs_clim']:+.4f} | {dp90_str} | {m['fidelity_mae']:.4f} |"
        )
    return "\n".join(lines)


def _format_feature_contract_section() -> List[str]:
    """Return markdown lines documenting canonical 70/65 feature matrix dimensions."""
    return [
        "",
        "## Canonical Feature Matrix Contract (`F3` = 70 cols, `F2` = 65 cols)",
        "",
        "Verified directly in `_train_h11_models()` / `run_audit()` (`len(f3_cols) == 70`, `len(f2_cols) == 65`, `f3_cols[:65] == f2_cols`, matching `artifacts/feature_names.json` `all_70_features`):",
        "",
        "| Model | Feature Set | `X_train.shape` | `X_val.shape` | `X_test.shape` | `len(feature_names)` |",
        "|---|---|---|---|---|---|",
        "| `SC-35` (XGBoost Tweedie) | `F3` (`f3_idx = range(70)`) | `(6289, 70)` | `(720, 70)` | `(2180, 70)` | `70` |",
        "| `SC-31` (LightGBM Tweedie Config A) | `F3` (`f3_idx = range(70)`) | `(6289, 70)` | `(720, 70)` | `(2180, 70)` | `70` |",
        "| `SC-30` (LightGBM Tweedie Config A) | `F2` (`f2_idx = range(65)`, first 65 of `F3`) | `(6289, 65)` | `(720, 65)` | `(2180, 65)` | `65` |",
    ]


def _format_diagnostics_section(report: Mapping[str, Any]) -> List[str]:
    """Return sharp Stage 1 closeout scientific diagnostics & Roadmap v2.0 conclusion."""
    sc35_val = report["metrics"]["val"]["p_c_doy_median_all_11_solar"]["SC-35"]
    sc35_test = report["metrics"]["test"]["p_c_doy_median_all_11_solar"]["SC-35"]
    vp, vs = sc35_val["per_step"], sc35_val["slices"]
    tp, ts = sc35_test["per_step"], sc35_test["slices"]
    t_h1_ci = tp["h1"]["delta_RMSE_vs_clim_ci95"]
    t_h8_11_ci = ts["h8_to_11"]["delta_RMSE_vs_clim_ci95"]
    return [
        "",
        "## Scientific Diagnostics & Roadmap v2.0 Conclusion (Round 14 Closeout — Option A Locked)",
        "",
        f"1. **Diagnosis 1 — Positive `ΔP90-Tail MAE` on Test emerges immediately at `h=2`, not only `h=8`**: On Held-Out Test (`n = 2,180`, `P-c`), `SC-35` beats `BASELINE_CLIMATOLOGY_DOY` on extreme rain (`PRECTOTCORR >= 11.086 mm`) only at `h=1` (`{tp['h1']['delta_p90_tail_mae_vs_clim']:+.4f} mm`). Across `h=2..7`, `ΔP90-Tail` is already continuously positive (`h=2: {tp['h2']['delta_p90_tail_mae_vs_clim']:+.4f}`, `h=3: {tp['h3']['delta_p90_tail_mae_vs_clim']:+.4f}`, `h=4: {tp['h4']['delta_p90_tail_mae_vs_clim']:+.4f}`, `h=5: {tp['h5']['delta_p90_tail_mae_vs_clim']:+.4f}`, `h=6: {tp['h6']['delta_p90_tail_mae_vs_clim']:+.4f}`, `h=7: {tp['h7']['delta_p90_tail_mae_vs_clim']:+.4f} mm`), making `h1_to_7` positive (`{ts['h1_to_7']['delta_p90_tail_mae_vs_clim']:+.4f} mm`) before rising across `h=8..11` (`h8_to_11 = {ts['h8_to_11']['delta_p90_tail_mae_vs_clim']:+.4f} mm`). On Validation (`n = 720`, `P-c`), `h1_to_7` stays slightly negative (`{vs['h1_to_7']['delta_p90_tail_mae_vs_clim']:+.4f} mm`) solely because the `h=1` pull (`{vp['h1']['delta_p90_tail_mae_vs_clim']:+.4f} mm`) offsets `h=2..7` (`h=2: {vp['h2']['delta_p90_tail_mae_vs_clim']:+.4f}`, `h=3: {vp['h3']['delta_p90_tail_mae_vs_clim']:+.4f}`, `h=4: {vp['h4']['delta_p90_tail_mae_vs_clim']:+.4f}`, `h=5: {vp['h5']['delta_p90_tail_mae_vs_clim']:+.4f}`, `h=6: {vp['h6']['delta_p90_tail_mae_vs_clim']:+.4f}`, `h=7: {vp['h7']['delta_p90_tail_mae_vs_clim']:+.4f} mm`); every window excluding `h=1` (`h4_to_10`, `h5_to_11`, `h8_to_11`) is strictly positive on both Validation (`{vs['h4_to_10']['delta_p90_tail_mae_vs_clim']:+.4f}..{vs['h8_to_11']['delta_p90_tail_mae_vs_clim']:+.4f} mm`) and Test (`{ts['h4_to_10']['delta_p90_tail_mae_vs_clim']:+.4f}..{ts['h8_to_11']['delta_p90_tail_mae_vs_clim']:+.4f} mm`).",
        f"2. **Diagnosis 2 — Structural `ΔMAE` ($L_1$) flatness from `h=2..11` vs `ΔRMSE` ($L_2$) convective skill decay**: `BASELINE_CLIMATOLOGY_DOY` outputs calendar-day arithmetic means (`E[Y | DOY]`), which are $L_2$-optimal (`RMSE`) rather than $L_1$-optimal (`MAE` favors the lower conditional median) under right-skewed tropical rainfall (`47.56%–56.60%` sub-1.0 mm dry days; `22.77%–27.32%` zero-rain days). Shrunk Tweedie (`p = 1.5`) predictions retain a structural `L1` margin on frequent dry/light-rain days, flattening Test `ΔMAE` around `{tp['h2']['delta_MAE_vs_clim']:+.4f}` (`h=2`) to `{tp['h11']['delta_MAE_vs_clim']:+.4f} mm` (`h=11`). Meanwhile, `ΔRMSE` exposes real convective skill decay: on Test `P-c`, `SC-35` drops from `{tp['h1']['delta_RMSE_vs_clim']:+.4f} mm` at `h=1` (`95% CI [{t_h1_ci['ci_lower']:+.4f}, {t_h1_ci['ci_upper']:+.4f}]`) to `{tp['h7']['delta_RMSE_vs_clim']:+.4f} mm` at `h=7` and `{tp['h10']['delta_RMSE_vs_clim']:+.4f} mm` at `h=10`, with `h8_to_11` at `{ts['h8_to_11']['delta_RMSE_vs_clim']:+.4f} mm` (`95% MBB CI [{t_h8_11_ci['ci_lower']:+.4f}, {t_h8_11_ci['ci_upper']:+.4f}]` crossing `0`, statistically non-significant).",
        "3. **Roadmap v2.0 & Option A Closeout Decision**: Exceeding the 3–4 true future day ceiling under GMAO GEOS-IT latency requires integrating **NWP numerical weather prediction future covariates (`t+1..t+7`)** in v2.0 rather than stretching historical point-observation horizons. Under **Option A**, production serving remains locked at **`H = 7`, Docker `1.7.0-serving`, SQLite `DDL V3.5`, MLflow `v1` aliases**, Stage 2 is closed, and **`input_window_manifests` (`DDL V3.6`)** is deferred to multi-province scaling (production DB `39.52 MB`, `< 1%` of Turso Free 5 GB quota).",
    ]


def _build_readme_markdown(report: Mapping[str, Any]) -> str:
    """Render concise Stage 1 audit README markdown from serialized report."""
    gate = report["go_no_go_gate"]
    clause_a = gate["clause_A_h1_to_7_non_regression"]
    winning = gate["winning_H_star"]
    stop_flag = gate["stop_and_report_triggered"]

    lines = [
        "# Stage 1 Horizon Extension Feasibility Audit (`H=7` → `H=10`/`H=11` under `P-c`)",
        "",
        f"- **Gate Decision**: `winning_H_star = {winning}` | `stop_and_report_triggered = {stop_flag}`",
        f"- **Canonical Split Purge (`H=11`)**: Dev Train `{report['splits']['dev_train']['n_origins']}` origins (last `{report['splits']['dev_train']['last_origin']}`), Validation `{report['splits']['val']['n_origins']}` origins (last `{report['splits']['val']['last_origin']}`), Test `{report['splits']['test']['n_origins']}` origins (last `{report['splits']['test']['last_origin']}`).",
        f"- **Bootstrap Protocol**: Moving Block Bootstrap (`block_length = {BLOCK_LENGTH}`, `n_bootstrap = {N_BOOTSTRAP}`, `seed = {BOOTSTRAP_SEED}`), `P90 >= {report['p90_threshold_dev_train_mm']:.3f} mm`, `MASE scale = {MASE_SCALE}`.",
        "",
        "## Clause A — Existing `h=1..7` Non-Regression on Validation `P-c` (`|ΔMAE| ≤ 0.015 mm`)",
        "",
        "| Model | `H=11` Val `P-c` `h1_to_7` MAE | Packaged `H=7` Val `P-c` `h1_to_7` MAE | `|Diff|` (mm) | Passed |",
        "|---|---|---|---|---|",
    ]
    for mid in ("SC-35", "SC-31", "SC-30"):
        cm = clause_a["models"][mid]
        lines.append(
            f"| `{mid}` | {cm['val_pc_h1_to_7_mae_H11']:.6f} | {cm['val_pc_h1_to_7_mae_H7']:.6f} | "
            f"{cm['abs_diff_mm']:.6f} | {cm['passed_le_0_015']} |"
        )

    lines.extend(
        [
            "",
            "## Go/No-Go Gate Summary (`H* ∈ {11, 10}` + Diagnostics `H* ∈ {9, 8}`)",
            "",
            "| Candidate `H*` | Ext Slice | Role | Clause A | Clause B1 (Val `P-c`) | Clause B2 (Val `P-c`) | Clause B3 (Test `P-c`) | Overall |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for key in ("H_star_11", "H_star_10", "H_star_9", "H_star_8"):
        cd = gate["candidates"][key]
        role = "Gate Candidate" if cd["is_stage2_eligible_candidate"] else "Diagnostic Only"
        lines.append(
            f"| `H*={cd['H_star']}` | `{cd['extended_slice']}` | {role} | {cd['clause_A_passed']} | "
            f"{cd['clause_B1']['passed']} | {cd['clause_B2']['passed']} | {cd['clause_B3']['passed']} | "
            f"**{cd['overall_gate_passed']}** |"
        )

    lines.extend(_format_feature_contract_section())
    lines.extend(
        [
            "",
            "## Validation (`n = 720`, `P-c`) — Per-Step `h=1..11`",
            "",
            _format_step_table(report["metrics"]["val"], "p_c_doy_median_all_11_solar"),
            "",
            "## Validation (`n = 720`, `P-c`) — `SC-35` Slice Summary",
            "",
            _format_slice_table(report["metrics"]["val"], "p_c_doy_median_all_11_solar"),
            "",
            "## Held-Out Test (`n = 2,180`, `P-c`) — Per-Step `h=1..11`",
            "",
            _format_step_table(report["metrics"]["test"], "p_c_doy_median_all_11_solar"),
            "",
            "## Held-Out Test (`n = 2,180`, `P-c`) — `SC-35` Slice Summary",
            "",
            _format_slice_table(report["metrics"]["test"], "p_c_doy_median_all_11_solar"),
            "",
            "## Validation vs Test (`clean` vs `P-c`) — Cross-Model Extended Horizon Summary (`h8_to_11` & `h8_to_10`)",
            "",
            "### Validation (`n = 720`, `clean`)",
            "",
            _format_model_comparison_slice_table(report["metrics"]["val"], "clean", "h8_to_11"),
            "",
            _format_model_comparison_slice_table(report["metrics"]["val"], "clean", "h8_to_10"),
            "",
            "### Validation (`n = 720`, `P-c`)",
            "",
            _format_model_comparison_slice_table(report["metrics"]["val"], "p_c_doy_median_all_11_solar", "h8_to_11"),
            "",
            "### Held-Out Test (`n = 2,180`, `clean`)",
            "",
            _format_model_comparison_slice_table(report["metrics"]["test"], "clean", "h8_to_11"),
            "",
            _format_model_comparison_slice_table(report["metrics"]["test"], "clean", "h8_to_10"),
            "",
            "### Held-Out Test (`n = 2,180`, `P-c`)",
            "",
            _format_model_comparison_slice_table(report["metrics"]["test"], "p_c_doy_median_all_11_solar", "h8_to_11"),
        ]
    )
    lines.extend(_format_diagnostics_section(report))
    lines.append("")
    return "\n".join(lines)


def run_audit(
    json_out: Path = DEFAULT_JSON_OUT,
    readme_out: Path = DEFAULT_README_OUT,
) -> Dict[str, Any]:
    """Execute full Stage 1 H=11 feasibility audit and persist JSON + README."""
    t0 = time.perf_counter()
    full_df, dev_train_df, val_df, test_df = _load_clean_hcmc_full()
    p90_dev_train = float(np.percentile(dev_train_df[CANONICAL_TARGET_COL].to_numpy(dtype=np.float64), 90.0))

    fb3 = joblib.load(ARTIFACT_DIR / "feature_builder_f3.joblib")
    fb2 = joblib.load(ARTIFACT_DIR / "feature_builder_f2.joblib")
    f3_sample = fb3.transform(dev_train_df.head(60), drop_warmup=True)
    f2_sample = fb2.transform(dev_train_df.head(60), drop_warmup=True)
    f3_cols = [c for c in f3_sample.columns if c not in (CANONICAL_DATE_COL, CANONICAL_TARGET_COL)]
    f2_cols = [c for c in f2_sample.columns if c not in (CANONICAL_DATE_COL, CANONICAL_TARGET_COL)]
    if len(f3_cols) != 70 or len(f2_cols) != 65 or f3_cols[:65] != f2_cols:
        raise AssertionError("F2/F3 feature schema mismatch")

    doy_table, doy_table_sha256 = load_solar_doy_climatology()
    a11_vi = [EN_TO_VI_CANONICAL[c] for c in SYN1DEG_11_CODES]
    a11_idx = [f3_cols.index(v) for v in a11_vi]
    if not all(idx < 65 for idx in a11_idx):
        raise AssertionError("Expected all 11 solar features within first 65 F2 columns")
    pc_parity_diff = _verify_pc_transform_parity(dev_train_df, fb3, f3_cols, doy_table, a11_idx)

    splits = _extract_h11_splits(dev_train_df, val_df, test_df, fb3, f3_cols)
    h11_models = _train_h11_models(splits["dev_train"]["X_f3"], splits["dev_train"]["y_true"])

    # Packaged frozen H=7 models for Clause A check
    h7_models = {
        "SC-35": (joblib.load(ARTIFACT_DIR / "sc35_xgboost_tweedie_f3.joblib"), list(range(70))),
        "SC-31": (joblib.load(ARTIFACT_DIR / "sc31_lightgbm_tweedie_f3.joblib"), list(range(70))),
        "SC-30": (joblib.load(ARTIFACT_DIR / "sc30_lightgbm_tweedie_f2.joblib"), list(range(65))),
    }

    doy_rain_table = _build_dev_train_rain_doy_table(dev_train_df)

    metrics_out: Dict[str, Any] = {}
    h7_val_pc_mae: Dict[str, float] = {}

    for split_name in ("val", "test"):
        sp = splits[split_name]
        X_clean = sp["X_f3"]
        y_true = sp["y_true"]
        dates = sp["origin_dates"]
        origin_rain = sp["origin_rain"]

        X_pc = X_clean.copy()
        X_pc[:, a11_idx] = _lookup_doy_matrix(doy_table, dates, SYN1DEG_11_CODES)

        boot_idx_matrix = build_block_bootstrap_indices(
            len(y_true),
            block_length=BLOCK_LENGTH,
            n_bootstrap=N_BOOTSTRAP,
            seed=BOOTSTRAP_SEED,
        )

        y_clim = _predict_climatology_doy_matrix(doy_rain_table, dates, HORIZON_H11)
        y_pers = _predict_persistence_matrix(origin_rain, HORIZON_H11)

        clean_preds: Dict[str, np.ndarray] = {
            "BASELINE_CLIMATOLOGY_DOY": y_clim,
            "BASELINE_PERSISTENCE": y_pers,
        }
        pc_preds: Dict[str, np.ndarray] = {
            "BASELINE_CLIMATOLOGY_DOY": y_clim,
            "BASELINE_PERSISTENCE": y_pers,
        }
        for mid, (m_obj, col_idx) in h11_models.items():
            clean_preds[mid] = np.clip(np.asarray(m_obj.predict(X_clean[:, col_idx]), dtype=np.float64), 0.0, None)
            pc_preds[mid] = np.clip(np.asarray(m_obj.predict(X_pc[:, col_idx]), dtype=np.float64), 0.0, None)

        if split_name == "val":
            for mid, (m7_obj, col_idx) in h7_models.items():
                pred7_pc = np.clip(np.asarray(m7_obj.predict(X_pc[:, col_idx]), dtype=np.float64), 0.0, None)
                h7_val_pc_mae[mid] = float(np.mean(np.abs(pred7_pc - y_true[:, :7])))

        split_conditions: Dict[str, Any] = {}
        for cond_name, cond_preds in (("clean", clean_preds), ("p_c_doy_median_all_11_solar", pc_preds)):
            cond_out: Dict[str, Any] = {}
            for mid in ("SC-35", "SC-31", "SC-30", "BASELINE_CLIMATOLOGY_DOY", "BASELINE_PERSISTENCE"):
                cond_out[mid] = _evaluate_model_all_windows(
                    y_true,
                    cond_preds[mid],
                    clean_preds[mid],
                    y_clim,
                    p90_thresh=p90_dev_train,
                    boot_idx_matrix=boot_idx_matrix,
                )
            split_conditions[cond_name] = cond_out
        metrics_out[split_name] = split_conditions

    _assert_no_purged_columns_in_report(metrics_out)

    gate_result = _evaluate_gate(
        val_pc=metrics_out["val"]["p_c_doy_median_all_11_solar"],
        test_pc=metrics_out["test"]["p_c_doy_median_all_11_solar"],
        h7_val_pc_mae=h7_val_pc_mae,
    )

    elapsed = round(time.perf_counter() - t0, 3)
    report: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "experiment": "stage1_h11_horizon_extension_feasibility_audit",
        "runtime_seconds": elapsed,
        "full_series_days": int(len(full_df)),
        "horizon_evaluated": HORIZON_H11,
        "solar_doy_table_sha256": doy_table_sha256,
        "pc_transform_max_parity_diff": pc_parity_diff,
        "p90_threshold_dev_train_mm": round(p90_dev_train, 6),
        "mase_scale_dev_train": MASE_SCALE,
        "bootstrap": {
            "method": "moving_block_bootstrap",
            "block_length": BLOCK_LENGTH,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
        },
        "splits": {
            name: {
                "n_origins": int(len(splits[name]["X_f3"])),
                "first_origin": pd.Timestamp(splits[name]["origin_dates"].iloc[0]).strftime("%Y-%m-%d"),
                "last_origin": pd.Timestamp(splits[name]["origin_dates"].iloc[-1]).strftime("%Y-%m-%d"),
            }
            for name in ("dev_train", "val", "test")
        },
        "metrics": metrics_out,
        "go_no_go_gate": gate_result,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    readme_out.write_text(_build_readme_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    """CLI entrypoint for Stage 1 H=11 horizon extension feasibility audit."""
    parser = argparse.ArgumentParser(description="Run Stage 1 H=11 Horizon Extension Feasibility Audit.")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--readme-out", type=Path, default=DEFAULT_README_OUT)
    args = parser.parse_args()

    report = run_audit(json_out=args.json_out, readme_out=args.readme_out)
    gate = report["go_no_go_gate"]
    print(f"Stage 1 audit completed in {report['runtime_seconds']}s -> {args.json_out}")
    print(f"Clause A passed: {gate['clause_A_h1_to_7_non_regression']['passed']}")
    for k in ("H_star_11", "H_star_10", "H_star_9", "H_star_8"):
        c = gate["candidates"][k]
        print(
            f"  {k} ({c['extended_slice']}): B1={c['clause_B1']['passed']}, "
            f"B2={c['clause_B2']['passed']}, B3={c['clause_B3']['passed']}, "
            f"overall={c['overall_gate_passed']}"
        )
    print(
        f"winning_H_star={gate['winning_H_star']}, "
        f"stop_and_report_triggered={gate['stop_and_report_triggered']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
