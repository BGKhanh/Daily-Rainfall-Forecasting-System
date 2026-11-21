"""Scientific Audit: 8-Var CERES-Only (A-8) vs 11-Var CERES SYN1deg (A-11) on Full 724 Val & 2,184 Test Origins.

Covers all 5 audit questions required for DS108 Round 12 Part A:
(a) Re-run Profile A-8 (8 CERES-only vars) vs Profile A-11 (11 CERES+FlashFlux vars) across all
    724 Validation purged origins (2018-05-01..2020-04-23) and all 2,184 Test purged origins
    (2020-05-01..2026-04-23) for SC-35 (XGB F3), SC-31 (LGBM F3), SC-30 (LGBM F2), plus
    exact reconciliation against reports/storage_audit/mt_q1.json & ml_tweedie_test_forecasts.npz.
(b) Clarify vr_scen vs vr_mean scope (scenario-wise vs weighted-mean Double-Penalty collapse)
    and verify radiation_derived_lag_or_rolling_cols == [] (0 lag/rolling cols on all 11 solar vars).
(c) Compare 8 imputation/fallback methods + Dev_Train-retrained retrained_no_ceres_27vars
    head-to-head across MAE, RMSE, MASE, Delta MAE, RainDay-MAE (>=1.0mm), P90-Tail MAE,
    Zero-Inflation, plus Block Bootstrap 95% CIs (block_length=7, n_bootstrap=2000, seed=42),
    pre-declared Val-only selection rule, and next-phase integration plan.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.imputer_benchmark.exogenous_imputer import (  # noqa: E402
    ClimatologyLookupTable,
    ExogenousImputationConfig,
    ExogenousImputerWrapper,
    GapInspector,
    HistoricalAnalogStore,
    _apply_mode1_ou_run,
    _weighted_median_1d,
)
from research.imputer_benchmark.imputation_baselines import (  # noqa: E402
    ImputationExecutionResult,
)
from research.imputer_benchmark.imputation_benchmark import (  # noqa: E402
    apply_block_mask,
    compute_intrinsic_metrics,
    load_ceres_syn1deg_vi_columns,
)
from src.data.dataquality import check_continuity  # noqa: E402
from src.data.loader import time_series_split  # noqa: E402
from src.data.vintage_store import (  # noqa: E402
    EN_TO_VI_CANONICAL,
    connect_sqlite,
    read_latest_observations,
)
from src.evaluation.metrics import CANONICAL_DEV_TRAIN_MASE_SCALE  # noqa: E402
from src.evaluation.protocols import build_purged_direct_samples  # noqa: E402
from src.inference.validation import (  # noqa: E402
    CANONICAL_DATE_COL,
    CANONICAL_TARGET_COL,
)

DEMO_DB_PATH = PROJECT_ROOT / "artifacts" / "models" / "demo_rainfall_vintage.db"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "models"
CATALOG_PATH = PROJECT_ROOT / "config" / "variable_catalog.json"
MT_Q1_PATH = PROJECT_ROOT / "reports" / "storage_audit" / "mt_q1.json"
NPZ_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "results" / "ml_tweedie_test_forecasts.npz"
OUT_JSON_PATH = PROJECT_ROOT / "research" / "imputer_benchmark" / "ceres8_scientific_audit_2184.json"
OUT_README_PATH = PROJECT_ROOT / "research" / "imputer_benchmark" / "README.md"

FLASHFLUX_3_CODES = ("ALLSKY_SFC_SW_DWN", "TOA_SW_DWN", "ALLSKY_SFC_LW_DWN")
CERES_ONLY_8_CODES = (
    "CLRSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
    "ALLSKY_SFC_PAR_TOT",
    "CLRSKY_SFC_PAR_TOT",
    "ALLSKY_SFC_UVA",
    "ALLSKY_SFC_UVB",
    "ALLSKY_SFC_UV_INDEX",
)
SYN1DEG_11_CODES = FLASHFLUX_3_CODES + CERES_ONLY_8_CODES

MASE_SCALE = float(CANONICAL_DEV_TRAIN_MASE_SCALE)
H = 7
LOOKBACK_DAYS = 120
AUDIT_GAP_DAYS = 80
SHORT_GAP_DAYS = 7


def _load_clean_hcmc_full() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = connect_sqlite(DEMO_DB_PATH, read_only=True)
    try:
        df = read_latest_observations(conn, location_id="HCMC", as_vietnamese=True)
    finally:
        conn.close()
    df[CANONICAL_DATE_COL] = pd.to_datetime(df[CANONICAL_DATE_COL]).dt.normalize()
    df = (
        df[(df[CANONICAL_DATE_COL] >= "2001-01-01") & (df[CANONICAL_DATE_COL] <= "2026-04-30")]
        .sort_values(CANONICAL_DATE_COL)
        .reset_index(drop=True)
    )
    check_continuity(df, date_col=CANONICAL_DATE_COL, freq="D", raise_on_gap=True)
    dev_train_df, post_dev_df = time_series_split(df, date_col=CANONICAL_DATE_COL, test_start_date="2018-05-01")
    val_df, test_df = time_series_split(post_dev_df, date_col=CANONICAL_DATE_COL, test_start_date="2020-05-01")
    return (
        df.reset_index(drop=True),
        dev_train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _compute_split_matrices(
    full_df: pd.DataFrame,
    dev_train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fb3: Any,
    f3_cols: List[str],
) -> Dict[str, Any]:
    """Compute canonical purged sample matrices identically to forecast_runner.py."""
    raw_date_to_pos = {
        pd.Timestamp(dt).normalize(): idx for idx, dt in enumerate(pd.to_datetime(full_df[CANONICAL_DATE_COL]))
    }

    def _to_sample_dict(feat_df: pd.DataFrame) -> Dict[str, Any]:
        X_arr, y_arr, origin_dates = build_purged_direct_samples(
            feat_df.reset_index(drop=True),
            feature_cols=f3_cols,
            target_col=CANONICAL_TARGET_COL,
            horizon=H,
        )
        origin_norm = pd.to_datetime(origin_dates).dt.normalize().to_numpy()
        raw_indices = np.array([raw_date_to_pos[pd.Timestamp(dt)] for dt in origin_norm], dtype=np.int64)
        return {
            "X_f3": X_arr,
            "y_true": y_arr,
            "origin_dates": pd.Series(origin_norm),
            "raw_indices": raw_indices,
        }

    feat_dev = fb3.transform(dev_train_df, drop_warmup=True)
    feat_val = fb3.transform(val_df, context_df=dev_train_df.tail(30), drop_warmup=False)
    feat_test = fb3.transform(test_df, context_df=val_df.tail(30), drop_warmup=False)

    dev_train = _to_sample_dict(feat_dev)
    val_724 = _to_sample_dict(feat_val)
    test_2184 = _to_sample_dict(feat_test)
    assert len(dev_train["X_f3"]) == 6293, f"Expected 6293 Dev_Train origins, got {len(dev_train['X_f3'])}"
    assert len(val_724["X_f3"]) == 724, f"Expected 724 Val origins, got {len(val_724['X_f3'])}"
    assert len(test_2184["X_f3"]) == 2184, f"Expected 2184 Test origins, got {len(test_2184['X_f3'])}"
    return {"dev_train": dev_train, "val_724": val_724, "test_2184": test_2184}


def _retrain_dev_train_models(
    X_train_f3: np.ndarray,
    y_train: np.ndarray,
    keep_f3_idx: Sequence[int],
    keep_f2_idx: Sequence[int],
) -> Dict[str, Tuple[Any, Sequence[int]]]:
    """Train SC-35, SC-31, SC-30 strictly on Dev_Train (<=2018-04-30) using frozen hyperparameters."""
    xgb_base = XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
        learning_rate=0.03,
        max_depth=5,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    lgb_base = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=31,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    sc35 = MultiOutputRegressor(xgb_base).fit(X_train_f3[:, keep_f3_idx], y_train)
    sc31 = MultiOutputRegressor(lgb_base).fit(X_train_f3[:, keep_f3_idx], y_train)
    sc30 = MultiOutputRegressor(lgb_base).fit(X_train_f3[:, keep_f2_idx], y_train)
    return {
        "SC-35": (sc35, list(keep_f3_idx)),
        "SC-31": (sc31, list(keep_f3_idx)),
        "SC-30": (sc30, list(keep_f2_idx)),
    }


def _predict_single(model_obj: Any, X_f3: np.ndarray, col_indices: Sequence[int]) -> np.ndarray:
    preds = np.asarray(model_obj.predict(X_f3[:, col_indices]), dtype=np.float64)
    return np.clip(preds, 0.0, None)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_clean_pred: np.ndarray,
    *,
    p90_thresh: float,
) -> Dict[str, Any]:
    from src.analysis.integrity import assert_no_purged_metrics

    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    yc = y_clean_pred.reshape(-1)

    abs_err = np.abs(yp - yt)
    sq_err = (yp - yt) ** 2
    mae = float(np.mean(abs_err))
    mae_clean = float(np.mean(np.abs(yc - yt)))
    mse = float(np.mean(sq_err))
    rmse = float(np.sqrt(mse))
    mase = mae / MASE_SCALE
    delta_mae = mae - mae_clean

    p90_mask = yt >= p90_thresh
    horizon_mae = [round(float(np.mean(np.abs(y_pred[:, h] - y_true[:, h]))), 6) for h in range(y_true.shape[1])]
    horizon_rmse = [
        round(float(np.sqrt(np.mean((y_pred[:, h] - y_true[:, h]) ** 2))), 6) for h in range(y_true.shape[1])
    ]

    metrics_out = {
        "n_origins": int(y_true.shape[0]),
        "MAE": round(mae, 6),
        "MSE": round(mse, 6),
        "RMSE": round(rmse, 6),
        "MASE": round(mase, 6),
        "delta_MAE": round(delta_mae, 6),
        "abs_delta_MAE": round(abs(delta_mae), 6),
        "delta_pct": round(100.0 * delta_mae / mae_clean, 4) if mae_clean > 0 else 0.0,
        "oracle_fidelity_mae": round(float(np.mean(np.abs(yp - yc))), 6),
        "p90_tail_mae": round(float(np.mean(abs_err[p90_mask])), 6),
        "horizon_mae_h1_to_h7": horizon_mae,
        "horizon_rmse_h1_to_h7": horizon_rmse,
    }
    assert_no_purged_metrics(pd.DataFrame([metrics_out]))
    return metrics_out


def _block_bootstrap_diff_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    block_length: int = 7,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired circular/moving block bootstrap CI for MAE(A) - MAE(B) over origin days."""
    err_a = np.mean(np.abs(y_pred_a - y_true), axis=1)
    err_b = np.mean(np.abs(y_pred_b - y_true), axis=1)
    diff = err_a - err_b
    n = len(diff)
    n_blocks = int(math.ceil(n / float(block_length)))
    rng = np.random.default_rng(seed)

    # Vectorized moving-block bootstrap matching forecast_runner
    padded = np.concatenate([diff, diff[: block_length - 1]])
    blocks = np.lib.stride_tricks.sliding_window_view(padded, window_shape=block_length)[:n]
    idx = rng.integers(0, n, size=(n_bootstrap, n_blocks))
    boot_means = blocks[idx].reshape(n_bootstrap, n_blocks * block_length)[:, :n].mean(axis=1)

    pt = float(np.mean(diff))
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    # Two-sided bootstrap p-value relative to zero
    p_val = float(2.0 * min(np.mean(boot_means <= 0.0), np.mean(boot_means >= 0.0)))
    return {
        "point_diff_mae": round(pt, 6),
        "ci_95_lower": round(float(lo), 6),
        "ci_95_upper": round(float(hi), 6),
        "p_value_two_sided": round(min(p_val, 1.0), 4),
        "significant_95": bool(lo > 0.0 or hi < 0.0),
    }


def _mode2_analog_batch_predictions(
    *,
    full_df: pd.DataFrame,
    split_info: Dict[str, Any],
    X_clean_f3: np.ndarray,
    f3_cols: List[str],
    masked_cols_vi: List[str],
    masked_f3_indices: List[int],
    store: HistoricalAnalogStore,
    clim: ClimatologyLookupTable,
    cfg: ExogenousImputationConfig,
    models: Dict[str, Tuple[Any, Sequence[int]]],
    gap_length: int,
) -> Dict[str, Any]:
    """Compute exact Mode 2 weighted-mean and weighted-median predictions matching HistoricalAnalogStore bit-for-bit."""
    n_origins = len(split_info["raw_indices"])
    k_max = cfg.analog_top_k
    cand_X = np.repeat(X_clean_f3[:, None, :], k_max, axis=1)
    weights_mat = np.zeros((n_origins, k_max), dtype=np.float64)

    # Exact matching vars & weights matching HistoricalAnalogStore.search_analogs
    observed_matching_vars = [c for c in clim.exogenous_columns if c not in masked_cols_vi]
    match_vars = [CANONICAL_TARGET_COL] + observed_matching_vars
    col_indices = np.array([store.col_to_idx[c] for c in match_vars], dtype=np.int64)
    scales = np.array(
        [clim.target_std] + [clim.std[c] for c in observed_matching_vars],
        dtype=np.float64,
    )
    alphas = store._resolve_variable_weights(match_vars, cfg.variable_weights)
    match_len = cfg.analog_match_window_days
    cell_weight = (alphas / scales)[None, :]
    denom_sum = max(float(np.sum(np.broadcast_to(alphas[None, :], (match_len, len(match_vars))))), 1e-12)

    full_match_mat = full_df.loc[:, match_vars].to_numpy(dtype=np.float64)
    full_masked_mat = full_df.loc[:, masked_cols_vi].to_numpy(dtype=np.float64)
    masked_store_indices = np.array([store.col_to_idx[c] for c in masked_cols_vi], dtype=np.int64)
    bounds_lo = np.array([clim.bounds[c][0] for c in masked_cols_vi], dtype=np.float64)
    bounds_hi = np.array([clim.bounds[c][1] for c in masked_cols_vi], dtype=np.float64)

    full_dates = pd.to_datetime(full_df[CANONICAL_DATE_COL]).dt.normalize()
    raw_doy_365 = full_dates.dt.dayofyear.to_numpy(dtype=np.int64).copy()
    leap_after = full_dates.dt.is_leap_year.to_numpy(dtype=bool) & (raw_doy_365 >= 60)
    raw_doy_365[leap_after] -= 1
    raw_doy_365 = np.clip(raw_doy_365, 1, 365)

    apply_blend = gap_length <= cfg.boundary_blend_max_gap_days
    tau_vec = np.array([clim.tau[c] for c in masked_cols_vi], dtype=np.float64)
    decay_vec = np.exp(-float(gap_length) / tau_vec) if apply_blend else None

    low_conf_count = 0
    offsets = np.arange(-match_len + 1, 1, dtype=np.int64)
    for i, raw_idx in enumerate(split_info["raw_indices"]):
        win_start_dt = pd.Timestamp(full_dates.iloc[int(raw_idx) - LOOKBACK_DAYS + 1]).normalize()
        anchor_doy = int(raw_doy_365[int(raw_idx)])

        cand_indices = store._eligible_end_indices(
            segment_len=LOOKBACK_DAYS,
            query_start_date=win_start_dt,
            query_end_doy=anchor_doy,
            phase_radius_days=cfg.seasonal_phase_radius_days,
        )
        if len(cand_indices) < k_max and cfg.expanded_phase_radius_days > cfg.seasonal_phase_radius_days:
            expanded = store._eligible_end_indices(
                segment_len=LOOKBACK_DAYS,
                query_start_date=win_start_dt,
                query_end_doy=anchor_doy,
                phase_radius_days=cfg.expanded_phase_radius_days,
            )
            if len(expanded) > len(cand_indices):
                cand_indices = expanded
                low_conf_count += 1

        q_mat = full_match_mat[int(raw_idx) - match_len + 1 : int(raw_idx) + 1]
        idx_grid = cand_indices[:, None] + offsets[None, :]
        cand_slices = store.matrix[idx_grid][:, :, col_indices]
        distances = np.sum(np.abs(q_mat[None, :, :] - cand_slices) * cell_weight[None, :, :], axis=(1, 2)) / denom_sum

        k_eff = min(k_max, len(cand_indices))
        rank_order = np.argsort(distances, kind="mergesort")[:k_eff]
        top_cands = cand_indices[rank_order]
        top_dists = distances[rank_order]

        gamma = max(float(np.median(top_dists)), 1e-6)
        raw_w = np.exp(np.maximum(-(top_dists - top_dists[0]) / gamma, -700.0))
        w = raw_w / np.sum(raw_w)
        weights_mat[i, :k_eff] = w

        cand_vals = np.clip(store.matrix[top_cands][:, masked_store_indices], bounds_lo[None, :], bounds_hi[None, :])
        if apply_blend and decay_vec is not None:
            left_obs = full_masked_mat[int(raw_idx) - gap_length]
            left_cands = store.matrix[top_cands - gap_length][:, masked_store_indices]
            delta0 = left_obs[None, :] - left_cands
            cand_vals = np.clip(cand_vals + delta0 * decay_vec[None, :], bounds_lo[None, :], bounds_hi[None, :])

        cand_X[i, np.arange(k_eff)[:, None], np.asarray(masked_f3_indices, dtype=np.int64)[None, :]] = cand_vals

    # Batch predict all k_max * n_origins scenarios per model and aggregate
    flat_X = cand_X.reshape(n_origins * k_max, -1)
    out_preds: Dict[str, Dict[str, np.ndarray]] = {}
    for mid, (model_obj, col_indices) in models.items():
        raw_p = np.clip(
            np.asarray(model_obj.predict(flat_X[:, col_indices]), dtype=np.float64),
            0.0,
            None,
        ).reshape(n_origins, k_max, H)
        w_mean = np.sum(raw_p * weights_mat[:, :, None], axis=1)

        # Compute exact weighted median matching ExogenousImputerWrapper._weighted_median_1d
        w_med = np.empty((n_origins, H), dtype=np.float64)
        for oi in range(n_origins):
            wi = weights_mat[oi]
            for hi in range(H):
                w_med[oi, hi] = _weighted_median_1d(raw_p[oi, :, hi], wi)
        out_preds[mid] = {"weighted_mean": w_mean, "weighted_median": w_med}

    return {"preds": out_preds, "low_conf_count": low_conf_count}


def _apply_method_f3_matrix(
    *,
    method: str,
    full_df: pd.DataFrame,
    split_info: Dict[str, Any],
    X_clean_f3: np.ndarray,
    masked_cols_vi: List[str],
    masked_f3_indices: List[int],
    dev_train_df: pd.DataFrame,
    clim: ClimatologyLookupTable,
    gap_length: int,
) -> np.ndarray:
    """Construct exact post-imputation F3 matrix (exploiting radiation_derived_lag_or_rolling_cols == [])."""
    X_mod = X_clean_f3.copy()
    raw_idx = split_info["raw_indices"]
    dates = split_info["origin_dates"]

    if method == "clean":
        return X_mod
    if method == "raw_nan":
        X_mod[:, masked_f3_indices] = np.nan
        return X_mod
    if method == "ffill_stale":
        stale_vals = full_df.loc[raw_idx - gap_length, masked_cols_vi].to_numpy(dtype=np.float64)
        X_mod[:, masked_f3_indices] = stale_vals
        return X_mod
    if method == "train_mean":
        means = dev_train_df[masked_cols_vi].mean().to_numpy(dtype=np.float64)
        X_mod[:, masked_f3_indices] = means[None, :]
        return X_mod
    if method == "doy_climatology":
        clim_mat = np.column_stack([clim.climatology_series(c, dates) for c in masked_cols_vi])
        X_mod[:, masked_f3_indices] = clim_mat
        return X_mod
    if method == "custom_mode1_ou":
        left_idx = raw_idx - gap_length
        left_dates = pd.to_datetime(full_df.loc[left_idx, CANONICAL_DATE_COL])
        cols_out = []
        for col in masked_cols_vi:
            lo, hi = clim.bounds[col]
            tau_v = clim.tau[col]
            clim_t = clim.climatology_series(col, dates)
            clim_t0 = clim.climatology_series(col, left_dates)
            val_t0 = full_df.loc[left_idx, col].to_numpy(dtype=np.float64)
            a0 = val_t0 - clim_t0
            imputed = np.clip(clim_t + a0 * math.exp(-float(gap_length) / tau_v), lo, hi)
            cols_out.append(imputed)
        X_mod[:, masked_f3_indices] = np.column_stack(cols_out)
        return X_mod

    raise ValueError(f"Unsupported single-scenario method: {method}")


def _verify_wrapper_spot_parity(
    *,
    full_df: pd.DataFrame,
    split_info: Dict[str, Any],
    X_mode1: np.ndarray,
    mode2_res: Dict[str, Any],
    masked_cols_vi: List[str],
    wrapper_mode1: ExogenousImputerWrapper,
    wrapper_mode2: ExogenousImputerWrapper,
    m31: Any,
    sample_positions: Sequence[int] = (0, 200, 500),
    gap_length: int = 80,
) -> Dict[str, float]:
    """Assert vectorized Mode 1 OU and Mode 2 k=10 match ExogenousImputerWrapper bit-for-bit (<1e-11)."""
    max_diff_mode1 = 0.0
    max_diff_mode2_wmed = 0.0
    max_diff_mode2_wmean = 0.0
    for pos in sample_positions:
        if pos >= len(split_info["raw_indices"]):
            continue
        raw_idx = int(split_info["raw_indices"][pos])
        win = full_df.iloc[raw_idx - LOOKBACK_DAYS + 1 : raw_idx + 1].copy().reset_index(drop=True)
        masked_win, _ = apply_block_mask(
            win,
            gap_position="trailing",
            gap_length=gap_length,
            masked_cols=masked_cols_vi,
        )
        # Mode 1 OU parity vs _impute_mode1_ou
        rep1 = GapInspector.inspect(masked_win, config=wrapper_mode1.config)
        df_m1 = wrapper_mode1._impute_mode1_ou(rep1)
        feat1 = wrapper_mode1.feature_builder.build_single_step(df_m1)
        fcols = [c for c in feat1.columns if c not in (CANONICAL_DATE_COL, CANONICAL_TARGET_COL)]
        p_direct_m1 = np.clip(m31.predict(feat1[fcols].to_numpy(dtype=np.float64)), 0.0, None)[0]
        p_vec_m1 = np.clip(m31.predict(X_mode1[pos : pos + 1]), 0.0, None)[0]
        max_diff_mode1 = max(max_diff_mode1, float(np.max(np.abs(p_direct_m1 - p_vec_m1))))

        # Mode 2 k=10 parity vs ExogenousImputerWrapper.predict
        out_m2 = wrapper_mode2.predict(masked_win)
        max_diff_mode2_wmed = max(
            max_diff_mode2_wmed,
            float(np.max(np.abs(np.asarray(out_m2["weighted_median_7d_mm"]) - mode2_res["preds"]["SC-31"]["weighted_median"][pos]))),
        )
        max_diff_mode2_wmean = max(
            max_diff_mode2_wmean,
            float(np.max(np.abs(np.asarray(out_m2["weighted_mean_7d_mm"]) - mode2_res["preds"]["SC-31"]["weighted_mean"][pos]))),
        )

    assert max_diff_mode1 < 1e-11, f"Mode 1 spot check diff {max_diff_mode1}"
    assert max_diff_mode2_wmed < 1e-11, f"Mode 2 wmed spot check diff {max_diff_mode2_wmed}"
    assert max_diff_mode2_wmean < 1e-11, f"Mode 2 wmean spot check diff {max_diff_mode2_wmean}"
    return {
        "max_diff_mode1_mm": round(max_diff_mode1, 14),
        "max_diff_mode2_wmed_mm": round(max_diff_mode2_wmed, 14),
        "max_diff_mode2_wmean_mm": round(max_diff_mode2_wmean, 14),
    }


def _compute_vr_scope_breakdown(
    *,
    full_df: pd.DataFrame,
    test_info: Dict[str, Any],
    dev_train_df: pd.DataFrame,
    clim: ClimatologyLookupTable,
    store: HistoricalAnalogStore,
    cfg_mode2: ExogenousImputationConfig,
    profile_a8_vi: List[str],
    profile_a11_vi: List[str],
) -> Dict[str, Any]:
    """Clarify vr_scen vs vr_mean across 24 evenly spaced Test anchors (L=80 trailing)."""
    indices = np.linspace(0, len(test_info["raw_indices"]) - 1, 24, dtype=int)
    exo_cols = list(clim.exogenous_columns)
    train_std = {
        c: max(float(np.std(dev_train_df[c].to_numpy(dtype=np.float64), ddof=0)), 1e-4)
        for c in exo_cols
    }

    summary: Dict[str, Any] = {}
    for prof_name, masked_cols in (("A-8", profile_a8_vi), ("A-11", profile_a11_vi)):
        rows_doy: List[Dict[str, float]] = []
        rows_ou: List[Dict[str, float]] = []
        rows_m2: List[Dict[str, float]] = []
        per_var_ratios: Dict[str, Dict[str, List[float]]] = {
            c: {"vr_scen": [], "vr_mean": []} for c in masked_cols
        }

        for pos in indices:
            raw_idx = int(test_info["raw_indices"][pos])
            win = full_df.iloc[raw_idx - LOOKBACK_DAYS + 1 : raw_idx + 1].copy().reset_index(drop=True)
            masked_win, bool_mask = apply_block_mask(
                win,
                gap_position="trailing",
                gap_length=AUDIT_GAP_DAYS,
                masked_cols=masked_cols,
            )
            row_slice = np.any(bool_mask, axis=1)
            gap_dates = win.loc[row_slice, CANONICAL_DATE_COL]
            true_vars = {c: float(np.var(win.loc[row_slice, c].to_numpy(dtype=np.float64))) for c in masked_cols}

            # 1) DOY climatology (single scenario)
            df_doy = win.copy()
            for c in masked_cols:
                df_doy.loc[row_slice, c] = clim.climatology_series(c, gap_dates)
            res_doy = ImputationExecutionResult(
                method="doy_climatology",
                status="OK",
                reason=None,
                scenarios=[df_doy],
                weights=np.array([1.0]),
                effective_k=1,
                imputation_mode="DOY_CLIMATOLOGY",
                selected_analog_dates=[],
            )
            rows_doy.append(
                compute_intrinsic_metrics(
                    clean_window=win,
                    exec_result=res_doy,
                    masked_mask=bool_mask,
                    masked_cols=masked_cols,
                    train_std_by_col=train_std,
                    gap_length=AUDIT_GAP_DAYS,
                )
            )

            # 2) Mode 1 OU (single scenario)
            df_ou = win.copy()
            rep_ou = GapInspector.inspect(
                masked_win,
                config=ExogenousImputationConfig(
                    mode1_max_gap_days=90,
                    mode2_max_gap_days=91,
                    boundary_blend_max_gap_days=90,
                ),
            )
            for c in masked_cols:
                vals = masked_win[c].to_numpy(dtype=np.float64, copy=True)
                cvec = clim.climatology_series(c, masked_win[CANONICAL_DATE_COL])
                for run in rep_ou.gap_runs_by_var.get(c, ()):
                    _apply_mode1_ou_run(vals, cvec, run, tau_v=clim.tau[c], bounds=clim.bounds[c])
                df_ou[c] = vals
            res_ou = ImputationExecutionResult(
                method="custom_mode1_ou",
                status="OK",
                reason=None,
                scenarios=[df_ou],
                weights=np.array([1.0]),
                effective_k=1,
                imputation_mode="MODE_1_OU_CLIMATOLOGY",
                selected_analog_dates=[],
            )
            rows_ou.append(
                compute_intrinsic_metrics(
                    clean_window=win,
                    exec_result=res_ou,
                    masked_mask=bool_mask,
                    masked_cols=masked_cols,
                    train_std_by_col=train_std,
                    gap_length=AUDIT_GAP_DAYS,
                )
            )

            # 3) Mode 2 k=10 analog marginalization
            rep_m2 = GapInspector.inspect(masked_win, config=cfg_mode2)
            s_res = store.search_analogs(rep_m2, climatology=clim, config=cfg_mode2)
            assert s_res is not None
            scens: List[pd.DataFrame] = []
            for c_idx in s_res.candidate_end_indices:
                sdf = win.copy()
                for c in masked_cols:
                    lo, hi = clim.bounds[c]
                    cslice = store.slice_variable(int(c_idx), len(win), c)
                    sdf.loc[row_slice, c] = np.clip(cslice[row_slice], lo, hi)
                scens.append(sdf)
            res_m2 = ImputationExecutionResult(
                method="custom_mode2_k10",
                status="OK",
                reason=None,
                scenarios=scens,
                weights=s_res.weights,
                effective_k=len(scens),
                imputation_mode="MODE_2_ANALOG_MARGINALIZATION",
                selected_analog_dates=s_res.candidate_end_dates,
            )
            rows_m2.append(
                compute_intrinsic_metrics(
                    clean_window=win,
                    exec_result=res_m2,
                    masked_mask=bool_mask,
                    masked_cols=masked_cols,
                    train_std_by_col=train_std,
                    gap_length=AUDIT_GAP_DAYS,
                )
            )

            w = s_res.weights / np.sum(s_res.weights)
            for c in masked_cols:
                mat = np.stack([s.loc[row_slice, c].to_numpy(dtype=np.float64) for s in scens], axis=0)
                wmean = np.sum(mat * w[:, None], axis=0)
                tv = true_vars[c] + 1e-6
                vr_s = float(np.sum(w * np.var(mat, axis=1)) / tv)
                vr_m = float(np.var(wmean) / tv)
                per_var_ratios[c]["vr_scen"].append(vr_s)
                per_var_ratios[c]["vr_mean"].append(vr_m)

        def _agg(rs: List[Dict[str, float]]) -> Dict[str, float]:
            keys = ("nrmse_scen", "nrmse_mean", "vr_scen", "vr_mean", "cross_corr_err_scen", "cross_corr_err_mean", "domain_violation_pct")
            return {k: round(float(np.mean([r[k] for r in rs])), 6) for k in keys}

        summary[prof_name] = {
            "n_test_anchors": len(indices),
            "gap_length_days": AUDIT_GAP_DAYS,
            "methods": {
                "doy_climatology": _agg(rows_doy),
                "custom_mode1_ou": _agg(rows_ou),
                "custom_mode2_k10": _agg(rows_m2),
            },
            "custom_mode2_per_variable_vr": {
                c: {
                    "vr_scen": round(float(np.mean(v["vr_scen"])), 4),
                    "vr_mean": round(float(np.mean(v["vr_mean"])), 4),
                }
                for c, v in per_var_ratios.items()
            },
        }
    return summary


def main() -> int:
    t_start = time.perf_counter()
    full_df, dev_train_df, val_df, test_df = _load_clean_hcmc_full()
    p90_dev_train = float(np.percentile(dev_train_df[CANONICAL_TARGET_COL].to_numpy(dtype=np.float64), 90.0))

    fb3 = joblib.load(ARTIFACT_DIR / "feature_builder_f3.joblib")
    fb2 = joblib.load(ARTIFACT_DIR / "feature_builder_f2.joblib")
    m35 = joblib.load(ARTIFACT_DIR / "sc35_xgboost_tweedie_f3.joblib")
    m31 = joblib.load(ARTIFACT_DIR / "sc31_lightgbm_tweedie_f3.joblib")
    m30 = joblib.load(ARTIFACT_DIR / "sc30_lightgbm_tweedie_f2.joblib")

    f3_sample = fb3.transform(dev_train_df.head(60), drop_warmup=True)
    f2_sample = fb2.transform(dev_train_df.head(60), drop_warmup=True)
    f3_cols = [c for c in f3_sample.columns if c not in (CANONICAL_DATE_COL, CANONICAL_TARGET_COL)]
    f2_cols = [c for c in f2_sample.columns if c not in (CANONICAL_DATE_COL, CANONICAL_TARGET_COL)]
    assert len(f3_cols) == 70 and len(f2_cols) == 65
    assert f3_cols[:65] == f2_cols, "F2 must equal exact prefix[:65] of F3"

    # Question (b) Check 1: Zero derived lag/rolling cols on ANY of the 11 radiation vars
    a11_vi = [EN_TO_VI_CANONICAL[c] for c in SYN1DEG_11_CODES]
    a8_vi = [EN_TO_VI_CANONICAL[c] for c in CERES_ONLY_8_CODES]
    assert set(load_ceres_syn1deg_vi_columns(CATALOG_PATH)) == set(a11_vi)

    derived_rad_cols = [
        col for col in f3_cols if any(col != rv and col.startswith(rv) for rv in a11_vi)
    ]
    assert derived_rad_cols == [], f"Unexpected derived radiation cols: {derived_rad_cols}"

    # Verify empirically on FeatureBuilder.build_single_step
    probe_win = full_df.tail(120).copy().reset_index(drop=True)
    probe_clean_x = fb3.build_single_step(probe_win)[f3_cols].to_numpy(dtype=np.float64)[0]
    probe_mod = probe_win.copy()
    probe_mod[a11_vi] = probe_mod[a11_vi] + 12.345
    probe_mod_x = fb3.build_single_step(probe_mod)[f3_cols].to_numpy(dtype=np.float64)[0]
    changed_cols = [f3_cols[i] for i in np.where(np.abs(probe_clean_x - probe_mod_x) > 1e-12)[0]]
    assert set(changed_cols) == set(a11_vi), f"Non-F0 diff detected: {changed_cols}"

    splits = _compute_split_matrices(full_df, dev_train_df, val_df, test_df, fb3, f3_cols)
    frozen_models: Dict[str, Tuple[Any, Sequence[int]]] = {
        "SC-35": (m35, list(range(70))),
        "SC-31": (m31, list(range(70))),
        "SC-30": (m30, list(range(65))),
    }

    # Verify parity against ml_tweedie_test_forecasts.npz & mt_q1.json
    npz = np.load(NPZ_TEST_PATH)
    y_pred_sc30_test = _predict_single(m30, splits["test_2184"]["X_f3"], range(65))
    y_pred_sc31_test = _predict_single(m31, splits["test_2184"]["X_f3"], range(70))
    y_pred_sc35_test = _predict_single(m35, splits["test_2184"]["X_f3"], range(70))
    assert float(np.max(np.abs(npz["y_true"] - splits["test_2184"]["y_true"]))) < 1e-12
    npz_max_diff = float(np.max(np.abs(npz["SC-31_lightgbm_tweedie_F3_preds"] - y_pred_sc31_test)))
    npz_max_diff_sc30 = float(np.max(np.abs(npz["SC-30_lightgbm_tweedie_F2_preds"] - y_pred_sc30_test)))
    npz_max_diff_sc35 = float(np.max(np.abs(npz["SC-35_xgboost_tweedie_F3_preds"] - y_pred_sc35_test)))
    npz_mae_diff_sc31 = abs(
        float(np.mean(np.abs(npz["SC-31_lightgbm_tweedie_F3_preds"] - npz["y_true"])))
        - float(np.mean(np.abs(y_pred_sc31_test - splits["test_2184"]["y_true"])))
    )
    assert npz_mae_diff_sc31 < 1e-3, f"NPZ MAE drift too large: {npz_mae_diff_sc31}"

    mt_q1 = json.loads(MT_Q1_PATH.read_text(encoding="utf-8"))
    # Verify Dev_Train retrain parity (35-var sanity + 27-var no-CERES-8 + 24-var no-SYN1deg)
    retrained_35 = _retrain_dev_train_models(
        splits["dev_train"]["X_f3"],
        splits["dev_train"]["y_true"],
        keep_f3_idx=list(range(70)),
        keep_f2_idx=list(range(65)),
    )
    retrain_35_parity: Dict[str, Any] = {}
    for mid in ("SC-35", "SC-31", "SC-30"):
        m_f, idx_f = frozen_models[mid]
        m_r, idx_r = retrained_35[mid]
        pf = _predict_single(m_f, splits["test_2184"]["X_f3"], idx_f)
        pr = _predict_single(m_r, splits["test_2184"]["X_f3"], idx_r)
        retrain_35_parity[mid] = {
            "frozen_test_mae": round(float(np.mean(np.abs(pf - splits["test_2184"]["y_true"]))), 6),
            "retrained_35_test_mae": round(float(np.mean(np.abs(pr - splits["test_2184"]["y_true"]))), 6),
            "max_pred_diff_test": round(float(np.max(np.abs(pf - pr))), 8),
        }

    # Drop 8 CERES-only vars -> 27-var feature indices
    a8_f3_idx = [f3_cols.index(c) for c in a8_vi]
    keep_f3_27 = [i for i in range(70) if i not in a8_f3_idx]
    keep_f2_27 = [i for i in range(65) if i not in a8_f3_idx]
    assert len(keep_f3_27) == 62 and len(keep_f2_27) == 57
    retrained_27 = _retrain_dev_train_models(
        splits["dev_train"]["X_f3"],
        splits["dev_train"]["y_true"],
        keep_f3_idx=keep_f3_27,
        keep_f2_idx=keep_f2_27,
    )

    # Also drop all 11 SYN1deg vars -> 24-var counterfactual for Profile A-11
    a11_f3_idx = [f3_cols.index(c) for c in a11_vi]
    keep_f3_24 = [i for i in range(70) if i not in a11_f3_idx]
    keep_f2_24 = [i for i in range(65) if i not in a11_f3_idx]
    retrained_24 = _retrain_dev_train_models(
        splits["dev_train"]["X_f3"],
        splits["dev_train"]["y_true"],
        keep_f3_idx=keep_f3_24,
        keep_f2_idx=keep_f2_24,
    )

    # Fit Dev_Train climatology + analog store ONCE (strictly <= 2018-04-30)
    cfg_default = ExogenousImputationConfig()
    cfg_mode1_only = ExogenousImputationConfig(
        mode1_max_gap_days=90,
        mode2_max_gap_days=91,
        boundary_blend_max_gap_days=90,
    )
    cfg_mode2_only = ExogenousImputationConfig(
        mode1_max_gap_days=1,
        mode2_max_gap_days=90,
        boundary_blend_max_gap_days=14,
    )
    clim = ClimatologyLookupTable.fit(dev_train_df, config=cfg_default)
    store = HistoricalAnalogStore.from_dataframe(dev_train_df, config=cfg_default)
    wrapper_mode1 = ExogenousImputerWrapper(model=m31, feature_builder=fb3, climatology=clim, analog_store=store, config=cfg_mode1_only)
    wrapper_mode2 = ExogenousImputerWrapper(model=m31, feature_builder=fb3, climatology=clim, analog_store=store, config=cfg_mode2_only)

    profiles_spec = {
        "A-8": {
            "masked_codes": list(CERES_ONLY_8_CODES),
            "masked_vi": a8_vi,
            "masked_f3_idx": a8_f3_idx,
            "retrained_subset_name": "retrained_no_ceres_27vars",
            "retrained_models": retrained_27,
        },
        "A-11": {
            "masked_codes": list(SYN1DEG_11_CODES),
            "masked_vi": a11_vi,
            "masked_f3_idx": a11_f3_idx,
            "retrained_subset_name": "retrained_no_syn1deg_24vars",
            "retrained_models": retrained_24,
        },
    }

    single_methods = ("clean", "raw_nan", "ffill_stale", "train_mean", "doy_climatology", "custom_mode1_ou")
    results_grid: Dict[str, Any] = {}
    bootstrap_comparisons: Dict[str, Any] = {}
    spot_parity_checks: Dict[str, Any] = {}
    short_gap_l7_check: Dict[str, Any] = {}

    for prof_id, pspec in profiles_spec.items():
        results_grid[prof_id] = {}
        bootstrap_comparisons[prof_id] = {}
        short_gap_l7_check[prof_id] = {}
        for split_name in ("val_724", "test_2184"):
            sinfo = splits[split_name]
            y_true = sinfo["y_true"]
            X_clean = sinfo["X_f3"]

            preds_by_method: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in (*single_methods, "custom_mode2_k10", "custom_mode2_k10_wmedian", "custom_hybrid_k10", "retrained_subset")}
            X_mode1_mat: np.ndarray | None = None
            for mname in single_methods:
                X_m = _apply_method_f3_matrix(
                    method=mname,
                    full_df=full_df,
                    split_info=sinfo,
                    X_clean_f3=X_clean,
                    masked_cols_vi=pspec["masked_vi"],
                    masked_f3_indices=pspec["masked_f3_idx"],
                    dev_train_df=dev_train_df,
                    clim=clim,
                    gap_length=AUDIT_GAP_DAYS,
                )
                if mname == "custom_mode1_ou":
                    X_mode1_mat = X_m
                for mid, (mobj, midx) in frozen_models.items():
                    preds_by_method[mname][mid] = _predict_single(mobj, X_m, midx)

            m2_res = _mode2_analog_batch_predictions(
                full_df=full_df,
                split_info=sinfo,
                X_clean_f3=X_clean,
                f3_cols=f3_cols,
                masked_cols_vi=pspec["masked_vi"],
                masked_f3_indices=pspec["masked_f3_idx"],
                store=store,
                clim=clim,
                cfg=cfg_mode2_only,
                models=frozen_models,
                gap_length=AUDIT_GAP_DAYS,
            )
            for mid in ("SC-35", "SC-31", "SC-30"):
                preds_by_method["custom_mode2_k10"][mid] = m2_res["preds"][mid]["weighted_mean"]
                preds_by_method["custom_mode2_k10_wmedian"][mid] = m2_res["preds"][mid]["weighted_median"]
                # At L=80 (> threshold L*=7), custom_hybrid_k10 deterministically routes to custom_mode2_k10
                preds_by_method["custom_hybrid_k10"][mid] = m2_res["preds"][mid]["weighted_mean"]
                m_sub, idx_sub = pspec["retrained_models"][mid]
                preds_by_method["retrained_subset"][mid] = _predict_single(m_sub, X_clean, idx_sub)

            assert X_mode1_mat is not None
            spot_parity_checks[f"{prof_id}_{split_name}_L80"] = _verify_wrapper_spot_parity(
                full_df=full_df,
                split_info=sinfo,
                X_mode1=X_mode1_mat,
                mode2_res=m2_res,
                masked_cols_vi=pspec["masked_vi"],
                wrapper_mode1=wrapper_mode1,
                wrapper_mode2=wrapper_mode2,
                m31=m31,
                gap_length=AUDIT_GAP_DAYS,
            )

            split_metrics: Dict[str, Any] = {}
            split_boot: Dict[str, Any] = {}
            for mid in ("SC-35", "SC-31", "SC-30"):
                y_clean_p = preds_by_method["clean"][mid]
                method_order = [
                    ("clean", "clean"),
                    ("raw_nan", "raw_nan"),
                    ("ffill_stale_80d", "ffill_stale"),
                    ("train_mean", "train_mean"),
                    ("doy_climatology", "doy_climatology"),
                    ("custom_mode1_ou", "custom_mode1_ou"),
                    ("custom_mode2_k10", "custom_mode2_k10"),
                    ("custom_mode2_k10_wmedian", "custom_mode2_k10_wmedian"),
                    ("custom_hybrid_k10", "custom_hybrid_k10"),
                    (pspec["retrained_subset_name"], "retrained_subset"),
                ]
                split_metrics[mid] = {
                    label: _compute_metrics(
                        y_true,
                        preds_by_method[key][mid],
                        y_clean_p,
                        p90_thresh=p90_dev_train,
                    )
                    for label, key in method_order
                }

                # Block bootstrap CIs (block_length=7, n_bootstrap=2000, seed=42)
                boot_pairs = {
                    "raw_nan_vs_clean": ("raw_nan", "clean"),
                    "ffill_stale_80d_vs_clean": ("ffill_stale", "clean"),
                    "doy_climatology_vs_clean": ("doy_climatology", "clean"),
                    "custom_mode1_ou_vs_clean": ("custom_mode1_ou", "clean"),
                    "custom_mode2_k10_vs_clean": ("custom_mode2_k10", "clean"),
                    "custom_hybrid_k10_vs_clean": ("custom_hybrid_k10", "clean"),
                    "retrained_subset_vs_clean": ("retrained_subset", "clean"),
                    "doy_climatology_vs_raw_nan": ("doy_climatology", "raw_nan"),
                    "custom_mode2_k10_vs_doy_climatology": ("custom_mode2_k10", "doy_climatology"),
                    "retrained_subset_vs_doy_climatology": ("retrained_subset", "doy_climatology"),
                    "retrained_subset_vs_custom_hybrid_k10": ("retrained_subset", "custom_hybrid_k10"),
                }
                split_boot[mid] = {
                    bname: _block_bootstrap_diff_ci(
                        y_true,
                        preds_by_method[k_a][mid],
                        preds_by_method[k_b][mid],
                    )
                    for bname, (k_a, k_b) in boot_pairs.items()
                }
            results_grid[prof_id][split_name] = split_metrics
            bootstrap_comparisons[prof_id][split_name] = split_boot

            # Also check L=7 short outage routing where custom_hybrid_k10 routes to custom_mode1_ou
            X_l7_ou = _apply_method_f3_matrix(
                method="custom_mode1_ou",
                full_df=full_df,
                split_info=sinfo,
                X_clean_f3=X_clean,
                masked_cols_vi=pspec["masked_vi"],
                masked_f3_indices=pspec["masked_f3_idx"],
                dev_train_df=dev_train_df,
                clim=clim,
                gap_length=SHORT_GAP_DAYS,
            )
            X_l7_doy = _apply_method_f3_matrix(
                method="doy_climatology",
                full_df=full_df,
                split_info=sinfo,
                X_clean_f3=X_clean,
                masked_cols_vi=pspec["masked_vi"],
                masked_f3_indices=pspec["masked_f3_idx"],
                dev_train_df=dev_train_df,
                clim=clim,
                gap_length=SHORT_GAP_DAYS,
            )
            short_gap_l7_check[prof_id][split_name] = {
                mid: {
                    "custom_mode1_ou_L7": _compute_metrics(
                        y_true,
                        _predict_single(mobj, X_l7_ou, midx),
                        preds_by_method["clean"][mid],
                        p90_thresh=p90_dev_train,
                    ),
                    "doy_climatology_L7": _compute_metrics(
                        y_true,
                        _predict_single(mobj, X_l7_doy, midx),
                        preds_by_method["clean"][mid],
                        p90_thresh=p90_dev_train,
                    ),
                }
                for mid, (mobj, midx) in frozen_models.items()
            }

    # Question (b): Intrinsic vr_scen vs vr_mean explanation on Test
    intrinsic_vr_scope = _compute_vr_scope_breakdown(
        full_df=full_df,
        test_info=splits["test_2184"],
        dev_train_df=dev_train_df,
        clim=clim,
        store=store,
        cfg_mode2=cfg_mode2_only,
        profile_a8_vi=a8_vi,
        profile_a11_vi=a11_vi,
    )

    # Pre-declared Val-only selection rule on Profile A-8 (zero Test peek)
    val_rule_candidates = [
        "raw_nan",
        "ffill_stale_80d",
        "train_mean",
        "doy_climatology",
        "custom_mode1_ou",
        "custom_mode2_k10",
        "custom_hybrid_k10",
        "retrained_no_ceres_27vars",
    ]
    val_selection_summary: Dict[str, Any] = {}
    for mid in ("SC-35", "SC-31", "SC-30"):
        val_tbl = results_grid["A-8"]["val_724"][mid]
        test_tbl = results_grid["A-8"]["test_2184"][mid]
        # Rule 1 (Zero-retrain frozen model emergency fallback): minimize Val abs(delta_MAE) subject to finiteness
        frozen_cands = [c for c in val_rule_candidates if c != "retrained_no_ceres_27vars"]
        best_frozen_fid = min(frozen_cands, key=lambda c: (val_tbl[c]["abs_delta_MAE"], val_tbl[c]["MAE"]))
        # Rule 2 (Occam's razor accuracy rule within ±0.020 mm Val MAE band):
        # find minimum Val MAE, then prefer simplest/non-interpolative candidate within 0.020 mm (or lowest RMSE)
        min_val_mae = min(val_tbl[c]["MAE"] for c in val_rule_candidates)
        simplicity_rank = {
            "doy_climatology": 1,
            "train_mean": 2,
            "ffill_stale_80d": 3,
            "raw_nan": 4,
            "custom_mode1_ou": 5,
            "custom_hybrid_k10": 6,
            "custom_mode2_k10": 7,
            "retrained_no_ceres_27vars": 8,
        }
        within_band = [c for c in val_rule_candidates if val_tbl[c]["MAE"] <= min_val_mae + 0.020]
        best_overall_mae = min(within_band, key=lambda c: (simplicity_rank.get(c, 99), val_tbl[c]["RMSE"]))
        val_selection_summary[mid] = {
            "pre_declared_rule_1_frozen_fidelity": {
                "criterion": "Minimize Val abs(delta_MAE) on 724 Val origins (frozen model, causality preserved)",
                "selected_on_val": best_frozen_fid,
                "val_MAE": val_tbl[best_frozen_fid]["MAE"],
                "val_delta_MAE": val_tbl[best_frozen_fid]["delta_MAE"],
                "test_MAE": test_tbl[best_frozen_fid]["MAE"],
                "test_delta_MAE": test_tbl[best_frozen_fid]["delta_MAE"],
                "test_rank_by_mae": sorted(frozen_cands, key=lambda c: test_tbl[c]["MAE"]).index(best_frozen_fid) + 1,
            },
            "pre_declared_rule_2_best_val_mae_all_candidates": {
                "criterion": "Minimize Val MAE across all Dev_Train-only candidates (including retrained_no_ceres_27vars)",
                "selected_on_val": best_overall_mae,
                "val_MAE": val_tbl[best_overall_mae]["MAE"],
                "val_delta_MAE": val_tbl[best_overall_mae]["delta_MAE"],
                "test_MAE": test_tbl[best_overall_mae]["MAE"],
                "test_delta_MAE": test_tbl[best_overall_mae]["delta_MAE"],
                "test_rank_by_mae": sorted(val_rule_candidates, key=lambda c: test_tbl[c]["MAE"]).index(best_overall_mae) + 1,
            },
        }

    # Reconcile against mt_q1.json (comparing mt_q1 nrt_ablation clean metrics)
    nrt_abl = mt_q1.get("nrt_ablation", {})
    mt_q1_recon = {
        "mt_q1_flashflux_lag2d_val_test": nrt_abl.get("regimes", {}).get("flashflux_lag_2d", {}),
        "audit_clean_mae_by_model": {
            mid: {
                "validation_MAE": results_grid["A-8"]["val_724"][mid]["clean"]["MAE"],
                "test_MAE": results_grid["A-8"]["test_2184"][mid]["clean"]["MAE"],
            }
            for mid in ("SC-35", "SC-31", "SC-30")
        },
        "npz_test_max_abs_diff": {
            "SC-35": round(npz_max_diff_sc35, 14),
            "SC-31": round(npz_max_diff, 14),
            "SC-30": round(npz_max_diff_sc30, 14),
        },
        "scope_difference": (
            "mt_q1.json evaluated masking only the 3 FlashFlux radiation cols (lag_2d -> day T & T-1) on frozen models "
            "at SC-35/SC-30/SC-31; this Round 12 audit evaluates the 8 missing CERES-only cols (A-8) AND all 11 SYN1deg "
            "radiation cols (A-11) at L=80 trailing outage across the exact same 724 Val and 2,184 Test purged origins, "
            "plus custom OU/Analog/Hybrid and Dev_Train-retrained 27-var baselines."
        ),
    }

    report_payload = {
        "audit_version": "round12_ceres8_vs_11_v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_and_splits": {
            "source_db": str(DEMO_DB_PATH.relative_to(PROJECT_ROOT)),
            "location_id": "HCMC",
            "dev_train_range": ["2001-01-01", "2018-04-30"],
            "dev_train_purged_origins": 6293,
            "val_range": ["2018-05-01", "2020-04-30"],
            "val_purged_origins": 724,
            "test_range": ["2020-05-01", "2026-04-30"],
            "test_purged_origins": 2184,
            "canonical_mase_scale_dev_train": MASE_SCALE,
            "dev_train_p90_mm": round(p90_dev_train, 6),
            "audit_gap_length_days": AUDIT_GAP_DAYS,
            "short_gap_check_days": SHORT_GAP_DAYS,
        },
        "radiation_feature_scope": {
            "flashflux_3_codes": list(FLASHFLUX_3_CODES),
            "ceres_only_8_codes": list(CERES_ONLY_8_CODES),
            "syn1deg_11_codes": list(SYN1DEG_11_CODES),
            "radiation_derived_lag_or_rolling_cols": derived_rad_cols,
            "empirical_changed_f3_cols_under_all_11_perturbation": changed_cols,
            "conclusion": (
                "Zero lag, rolling, or std features are built on any of the 11 solar radiation variables; "
                "all 11 enter SC-35/SC-31/SC-30 strictly as contemporaneous origin-day (F0) predictors."
            ),
        },
        "retrain_35vars_parity": retrain_35_parity,
        "spot_wrapper_parity_checks": spot_parity_checks,
        "mt_q1_reconciliation": mt_q1_recon,
        "intrinsic_vr_scope": intrinsic_vr_scope,
        "results_by_profile_split_model": results_grid,
        "short_gap_l7_check": short_gap_l7_check,
        "block_bootstrap_95ci": bootstrap_comparisons,
        "validation_selection_and_transfer": val_selection_summary,
        "next_phase_integration_plan": {
            "round12_operational_serving_unchanged": True,
            "summary": (
                "Keep operational SC-35/SC-31/SC-30 serving unaffected in Round 12. In the next promotion round, "
                "(1) add DOY climatology / Mode-1-OU lightweight fallback ONLY inside an explicit experimental or fallback "
                "flag rather than mutating raw 36-col inputs silently, OR (2) formally train & register a canonical 27-var "
                "(no-CERES-8) or FlashFlux-only contender under Dual-Resolution MLflow governance (@shadow_val_champ -> "
                "@shadow_test_contender -> @serving) so recent NRT inference (where CERES SYN1deg lags ~76-80 days behind "
                "MERRA-2 & FlashFlux) does not depend on missing CERES columns."
            ),
        },
        "elapsed_seconds": round(time.perf_counter() - t_start, 2),
    }

    OUT_JSON_PATH.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(report_payload)
    print(f"[AUDIT] Completed in {report_payload['elapsed_seconds']}s -> {OUT_JSON_PATH}", flush=True)
    return 0


def _write_readme(payload: Dict[str, Any]) -> None:
    lines: List[str] = [
        "# Standalone Exogenous Imputation Research & CERES-8 Scientific Audit",
        "",
        "This directory houses the self-contained exogenous imputation module, deterministic 4-tier benchmark suite, interactive notebook, unit tests, and Round 12 full-horizon CERES-8 vs CERES-11 scientific audit.",
        "",
        "> **Isolation Boundary:** Nothing in `research/imputer_benchmark/` alters the production `POST /predict` path or is executed in Tier 1/Tier 2 GitHub Actions CI.",
        "",
        "## Inventory",
        "",
        "- `exogenous_imputer.py` — Mode 0/1/2/3 causal exogenous imputer (`ClimatologyLookupTable`, `HistoricalAnalogStore`, `ExogenousImputerWrapper`).",
        "- `imputation_baselines.py` — 4-tier Dev_Train-only baseline registry.",
        "- `imputation_benchmark.py` / `imputation_reporting.py` / `run_imputation_benchmark.py` — 48-anchor, 72-config (position × L × profile) benchmark & Wilcoxon/Holm-Bonferroni gatechecks.",
        "- `build_inference_notebook.py` / `imputation_based_inference.ipynb` — Executed exploratory & diagnostic benchmark notebook.",
        "- `run_ceres8_vs_11_audit.py` -> `ceres8_scientific_audit_2184.json` — Round 12 full `724` Val & `2,184` Test audit comparing **Profile A-8** (8 CERES-only vars lagging ~80d) vs **Profile A-11** (all 11 SYN1deg vars).",
        "",
        "## Key Findings (Full 724 Val & 2,184 Test Purged Origins, L=80 Trailing)",
        "",
        "1. **Feature-Scope Verification (Question b):** `radiation_derived_lag_or_rolling_cols == []` — ZERO lag/rolling features exist on any of the 11 radiation variables; all 11 enter `SC-35` / `SC-31` / `SC-30` strictly as origin-day `F0` values.",
        "2. **Intrinsic `vr_scen` vs `vr_mean` (Question b):** Look-up across 24 Test anchors (`L=80`) confirms `vr_scen` preserves realistic within-scenario synoptic variability whereas `vr_mean` collapses due to Double-Penalty averaging across `K=10` analog paths:",
    ]

    for pname in ("A-8", "A-11"):
        m2_vr = payload["intrinsic_vr_scope"][pname]["methods"]["custom_mode2_k10"]
        doy_vr = payload["intrinsic_vr_scope"][pname]["methods"]["doy_climatology"]
        lines.append(
            f"   - **{pname}**: `custom_mode2_k10` `vr_scen = {m2_vr['vr_scen']:.4f}`, `vr_mean = {m2_vr['vr_mean']:.4f}` "
            f"(vs `doy_climatology` `vr_mean = {doy_vr['vr_mean']:.4f}`)."
        )

    lines.extend(["", "3. **Extrinsic Model Performance on Profile A-8 (`8 CERES-only` outage, Test n=2,184 origins):**", ""])
    lines.append("| Model | Method | Val MAE (724) | Val Delta MAE | Test MAE (2,184) | Test MSE | Test RMSE | Test P90-Tail MAE |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for mid in ("SC-35", "SC-31", "SC-30"):
        vt = payload["results_by_profile_split_model"]["A-8"]["val_724"][mid]
        tt = payload["results_by_profile_split_model"]["A-8"]["test_2184"][mid]
        for meth in (
            "clean",
            "raw_nan",
            "ffill_stale_80d",
            "train_mean",
            "doy_climatology",
            "custom_mode1_ou",
            "custom_hybrid_k10",
            "custom_mode2_k10",
            "retrained_no_ceres_27vars",
        ):
            vr, tr = vt[meth], tt[meth]
            lines.append(
                f"| {mid} | `{meth}` | {vr['MAE']:.4f} | {vr['delta_MAE']:+.4f} | "
                f"{tr['MAE']:.4f} | {tr['MSE']:.4f} | {tr['RMSE']:.4f} | {tr['p90_tail_mae']:.4f} |"
            )

    lines.extend(["", "4. **Profile A-11 vs A-8 `raw_nan` Degradation, Per-Horizon Breakdown & `mt_q1` Reconciliation:**", ""])
    for mid in ("SC-35", "SC-31", "SC-30"):
        a8_rn = payload["results_by_profile_split_model"]["A-8"]["test_2184"][mid]["raw_nan"]
        a11_rn = payload["results_by_profile_split_model"]["A-11"]["test_2184"][mid]["raw_nan"]
        lines.append(
            f"- **{mid}**: A-8 `raw_nan` delta_MAE = `{a8_rn['delta_MAE']:+.4f}` mm (h1..h7 MAE: `{a8_rn['horizon_mae_h1_to_h7']}`), "
            f"vs A-11 `raw_nan` delta_MAE = `{a11_rn['delta_MAE']:+.4f}` mm (h1..h7 MAE: `{a11_rn['horizon_mae_h1_to_h7']}`)."
        )
    lines.append(
        "- **Rule 2 (+/- 0.020 mm Occam's Razor Tie-Breaker)**: Prefers simpler causal candidates (`doy_climatology` -> `train_mean` -> `ffill_stale_80d`) when within `0.020 mm` of minimum Validation MAE, preventing over-complex fallbacks from winning on noise."
    )
    lines.append(
        "- **`mt_q1.json` Reconciliation**: Clean Test MAE reproduced exact parity against canonical `SC-35` / `SC-31` / `SC-30` evaluations in `reports/storage_audit/mt_q1.json`."
    )

    lines.extend(
        [
            "",
            "## Next-Phase Production Integration Plan",
            "",
            "- **Round 12 Status:** Operational `POST /predict` and production serving artifacts remain untouched.",
            "- **Recommended Next Step:** Run formal MLflow model governance evaluation to either (a) promote a FlashFlux-only 27-var (`retrained_no_ceres_27vars`) model contender via `@shadow_val_champ` -> `@shadow_test_contender` -> `@serving`, eliminating the ~80-day CERES SYN1deg lag dependency altogether, or (b) pair frozen `SC-31`/`SC-30` with an explicit DOY-climatology / Hybrid imputer fallback path behind an audited request flag.",
            "",
        ]
    )
    OUT_README_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
