"""
Imputation Benchmark Core Runner & Intrinsic/Extrinsic Metric Engine (DS108)
============================================================================
Executes the deterministic Single-Location Daily imputation benchmark pipeline:
1. Load clean canonical dataset & enforce strict Dev_Train (<=2018-04-30) < Test (>=2020-05-01).
2. Run LightGBM TreeSHAP (`pred_contrib=True`) on Dev_Train to identify Profile B Top-K vars.
3. Sample evenly spaced rolling-origin anchors across clean Test (with N_req=120 lookback + H=7 target).
4. Compute clean full-data reference forecasts for SC-31 (F3) & SC-30 (F2).
5. Fit all Tier 1-4 baselines strictly on Dev_Train.
6. Run full experimental matrix (`position × L × profile × anchor × method`) measuring
   latency/RSS, Branch 1 (Intrinsic with Double-Penalty awareness), and Branch 2 (Extrinsic).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import psutil

from src.data.loader import time_series_split
from src.data.vintage_store import (
    ALL_35_VARS,
    EN_TO_VI_CANONICAL,
    align_and_validate_canonical_raw_df,
    connect_sqlite,
    read_latest_observations,
)

VI_TO_EN_CANONICAL: Dict[str, str] = {vi: en for en, vi in EN_TO_VI_CANONICAL.items()}
from research.imputer_benchmark.imputation_baselines import (
    ALL_BENCHMARK_METHODS,
    GapPosition,
    ImputationBaselineRegistry,
    ImputationExecutionResult,
    ProfileId,
    combine_hybrid_from_mode1_mode2,
    compute_domain_violation_rate,
    count_nan_or_inf_in_matrix,
)
from research.imputer_benchmark.exogenous_imputer import CANONICAL_EXOGENOUS_COLUMNS
from src.inference.validation import CANONICAL_DATE_COL, CANONICAL_TARGET_COL

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "models"
DEFAULT_DEMO_DB_PATH = DEFAULT_ARTIFACT_DIR / "demo_rainfall_vintage.db"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "config" / "variable_catalog.json"

DEFAULT_GAP_LENGTHS: Tuple[int, ...] = (0, 3, 7, 14, 30, 60, 90)
DEFAULT_GAP_POSITIONS: Tuple[GapPosition, ...] = ("trailing", "interior")
DEFAULT_PROFILES: Tuple[ProfileId, ...] = ("A", "B", "C")


@dataclass(frozen=True)
class ImputationBenchmarkConfig:
    """Configurable benchmark specification (supports single-location path override)."""

    db_path: Path = DEFAULT_DEMO_DB_PATH
    csv_path: Optional[Path] = None
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    catalog_path: Path = DEFAULT_CATALOG_PATH
    location_id: str = "HCMC"
    dev_train_end_date: str = "2018-04-30"
    test_start_date: str = "2020-05-01"
    test_end_date: str = "2026-04-30"
    n_anchors: int = 48
    gap_lengths: Tuple[int, ...] = DEFAULT_GAP_LENGTHS
    gap_positions: Tuple[GapPosition, ...] = DEFAULT_GAP_POSITIONS
    profiles: Tuple[ProfileId, ...] = DEFAULT_PROFILES
    interior_offset_days: int = 7
    profile_b_top_k: int = 5
    rain_threshold_mm: float = 1.0
    hybrid_threshold_days: int = 7
    sensitivity_thresholds: Tuple[int, ...] = (3, 5, 7, 14)
    lookback_required_days: int = 120
    horizon_days: int = 7
    random_state: int = 42


def load_ceres_syn1deg_vi_columns(catalog_path: Path = DEFAULT_CATALOG_PATH) -> List[str]:
    """Load exact 11 CERES SYN1deg radiation column names (Vietnamese) from variable catalog."""
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    return [
        EN_TO_VI_CANONICAL[v["code"]]
        for v in raw["variables"]
        if v["source_finalized"] == "SYN1DEG"
    ]


def load_clean_single_location_splits(
    cfg: ImputationBenchmarkConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load clean canonical history and verify Dev_Train < Test split boundary with 0% missing."""
    if cfg.csv_path is not None:
        raw_df = pd.read_csv(cfg.csv_path)
    else:
        conn = connect_sqlite(cfg.db_path, read_only=True)
        try:
            raw_df = read_latest_observations(conn, location_id=cfg.location_id, as_vietnamese=True)
        finally:
            conn.close()

    df = align_and_validate_canonical_raw_df(raw_df)
    df[CANONICAL_DATE_COL] = pd.to_datetime(df[CANONICAL_DATE_COL]).dt.normalize()
    df = df.sort_values(CANONICAL_DATE_COL).reset_index(drop=True)
    # Keep strictly canonical clean period (2001-01-01 .. 2026-04-30) before NRT CERES delay tail
    df = df[
        (df[CANONICAL_DATE_COL] >= pd.Timestamp("2001-01-01"))
        & (df[CANONICAL_DATE_COL] <= pd.Timestamp(cfg.test_end_date))
    ].reset_index(drop=True)

    if int(df.isna().sum().sum()) != 0:
        raise ValueError("Clean historical dataset contains unexpected NaN entries.")

    dev_end_next = (pd.Timestamp(cfg.dev_train_end_date) + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    dev_train_df, post_dev_df = time_series_split(df, date_col=CANONICAL_DATE_COL, test_start_date=dev_end_next)
    val_df, test_df = time_series_split(post_dev_df, date_col=CANONICAL_DATE_COL, test_start_date=cfg.test_start_date)

    if pd.Timestamp(dev_train_df[CANONICAL_DATE_COL].max()) >= pd.Timestamp(test_df[CANONICAL_DATE_COL].min()):
        raise AssertionError("Leakage check failed: Dev_Train max date is not strictly before Test min date.")

    return df, dev_train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _map_feature_to_raw_variable(feature_name: str) -> Optional[str]:
    """Map an engineered FeatureBuilder column back to its parent raw canonical Vietnamese variable."""
    if feature_name in ("Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos", "Is_Wet_Season"):
        return None
    if feature_name == CANONICAL_TARGET_COL or feature_name.startswith(f"{CANONICAL_TARGET_COL}_"):
        return None
    if feature_name.endswith("_sin") or feature_name.endswith("_cos"):
        base = feature_name[:-4]
        return base if base in CANONICAL_EXOGENOUS_COLUMNS else None
    for marker in ("_lag_", "_mean_", "_std_"):
        if marker in feature_name:
            base = feature_name.split(marker, 1)[0]
            return base if base in CANONICAL_EXOGENOUS_COLUMNS else None
    return feature_name if feature_name in CANONICAL_EXOGENOUS_COLUMNS else None


def compute_shap_exogenous_ranking(
    *,
    model: Any,
    feature_builder: Any,
    dev_train_df: pd.DataFrame,
    model_id: str,
) -> pd.DataFrame:
    """Compute exact LightGBM TreeSHAP (`pred_contrib=True`) strictly on Dev_Train clean origins."""
    with contextlib.redirect_stdout(io.StringIO()):
        feat_df = feature_builder.transform(dev_train_df, drop_warmup=True)
    feat_cols = [c for c in feat_df.columns if c not in (feature_builder.date_col, feature_builder.target_col)]
    x_train = feat_df[feat_cols].to_numpy(dtype=np.float64)

    mean_abs_feat = np.zeros(len(feat_cols), dtype=np.float64)
    for estimator in model.estimators_:
        contrib = estimator.booster_.predict(x_train, pred_contrib=True)
        # Drop last column (TreeSHAP expected bias value) and average |phi| across origins
        mean_abs_feat += np.mean(np.abs(contrib[:, :-1]), axis=0)
    mean_abs_feat /= float(len(model.estimators_))

    raw_totals: Dict[str, float] = {col: 0.0 for col in CANONICAL_EXOGENOUS_COLUMNS}
    for col_name, val in zip(feat_cols, mean_abs_feat):
        parent = _map_feature_to_raw_variable(col_name)
        if parent is not None:
            raw_totals[parent] += float(val)

    rows = [
        {
            "model_id": model_id,
            "variable_vi": col,
            "variable_en": VI_TO_EN_CANONICAL.get(col, col),
            "mean_abs_shap": raw_totals[col],
        }
        for col in CANONICAL_EXOGENOUS_COLUMNS
    ]
    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["rank", "model_id", "variable_vi", "variable_en", "mean_abs_shap"]]


def select_rolling_origin_anchors(
    full_clean_df: pd.DataFrame,
    *,
    test_start_date: str,
    test_end_date: str,
    n_anchors: int,
    lookback_required_days: int = 120,
    horizon_days: int = 7,
) -> pd.DataFrame:
    """Select `n_anchors` evenly spaced Test origins with complete lookback history and H=7 targets."""
    dates = full_clean_df[CANONICAL_DATE_COL]
    start_ts = pd.Timestamp(test_start_date).normalize()
    end_ts = pd.Timestamp(test_end_date).normalize()

    valid_idx: List[int] = []
    n_total = len(full_clean_df)
    for idx in range(lookback_required_days - 1, n_total - horizon_days):
        dt = pd.Timestamp(dates.iloc[idx]).normalize()
        if start_ts <= dt <= end_ts:
            valid_idx.append(idx)

    if len(valid_idx) < n_anchors:
        raise ValueError(f"Only {len(valid_idx)} valid Test origins available, cannot sample {n_anchors} anchors.")

    chosen_pos = np.linspace(0, len(valid_idx) - 1, int(n_anchors), dtype=int)
    rows = []
    for pos in chosen_pos:
        idx = int(valid_idx[pos])
        dt = pd.Timestamp(dates.iloc[idx]).normalize()
        month = int(dt.month)
        if month in (5, 11):
            season = "Transition"
        elif 6 <= month <= 10:
            season = "Wet"
        else:
            season = "Dry"
        rows.append({"anchor_idx": idx, "anchor_date": dt.strftime("%Y-%m-%d"), "season_regime": season})
    return pd.DataFrame(rows)


def apply_block_mask(
    clean_window: pd.DataFrame,
    *,
    gap_position: GapPosition,
    gap_length: int,
    masked_cols: Sequence[str],
    interior_offset_days: int = 7,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply NaN block mask of length `gap_length` and return `(masked_df, bool_mask_n_by_k)`."""
    n_rows = len(clean_window)
    mask = np.zeros((n_rows, len(masked_cols)), dtype=bool)
    out = clean_window.copy()
    if gap_length <= 0:
        return out, mask

    if gap_position == "trailing":
        start = n_rows - gap_length
        end = n_rows
    elif gap_position == "interior":
        end = n_rows - interior_offset_days
        start = end - gap_length
    else:
        raise ValueError(f"Unsupported gap_position={gap_position!r}")

    if start < 1 or end > n_rows:
        raise ValueError(
            f"Mask span [{start}, {end}) out of valid causal bounds for n_rows={n_rows} (requires anchor >= 1 row)."
        )

    mask[start:end, :] = True
    for col in masked_cols:
        out.loc[out.index[start:end], col] = np.nan
    return out, mask


def _extract_f3_batch(
    feature_builder_f3: Any,
    scenarios: Sequence[pd.DataFrame],
    *,
    is_raw_nan: bool,
) -> Tuple[np.ndarray, Optional[str]]:
    """Extract 70-col F3 feature matrix (using `_apply_feature_transforms` solely for `raw_nan`)."""
    date_col = feature_builder_f3.date_col
    target_col = feature_builder_f3.target_col
    if is_raw_nan:
        scen = align_and_validate_canonical_raw_df(scenarios[0]).tail(40).copy()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                feat_df = feature_builder_f3._apply_feature_transforms(scen).tail(1)
            fcols = [c for c in feat_df.columns if c not in (date_col, target_col)]
            return feat_df[fcols].to_numpy(dtype=np.float64), None
        except Exception as exc:  # pragma: no cover - defensive fallback log per Spec §6.5
            msg = f"raw_nan transform exception logged: {type(exc).__name__}: {exc}"
            logger.warning(msg)
            return np.empty((0, 70), dtype=np.float64), msg

    rows: List[np.ndarray] = []
    fcols: Optional[List[str]] = None
    with contextlib.redirect_stdout(io.StringIO()):
        for scen in scenarios:
            canon = align_and_validate_canonical_raw_df(scen)
            feat_df = feature_builder_f3.build_single_step(canon)
            if fcols is None:
                fcols = [c for c in feat_df.columns if c not in (date_col, target_col)]
            rows.append(feat_df[fcols].to_numpy(dtype=np.float64)[0])
    return np.vstack(rows), None


def _compute_derived_fe_vector(df: pd.DataFrame, masked_cols: Sequence[str]) -> np.ndarray:
    """Compute [rolling_std_7d, rolling_max_7d, delta_1d] at anchor T_m across `masked_cols`."""
    out = np.empty(3 * len(masked_cols), dtype=np.float64)
    for j, col in enumerate(masked_cols):
        series = df[col].to_numpy(dtype=np.float64)
        tail7 = series[-7:]
        std7 = float(np.std(tail7, ddof=1)) if len(tail7) >= 2 and np.all(np.isfinite(tail7)) else np.nan
        max7 = float(np.max(tail7)) if np.all(np.isfinite(tail7)) else np.nan
        delta1 = float(series[-1] - series[-2]) if len(series) >= 2 and np.all(np.isfinite(series[-2:])) else np.nan
        out[3 * j] = std7
        out[3 * j + 1] = max7
        out[3 * j + 2] = delta1
    return out


def _cross_correlation_error(x_imp_gap: np.ndarray, x_true_gap: np.ndarray) -> float:
    """Mean off-diagonal |Corr(X_imp) - Corr(X_true)| across masked columns (for n_cols>=2, L>=14)."""
    n_rows, n_cols = x_true_gap.shape
    if n_cols < 2 or n_rows < 14:
        return np.nan
    if not np.all(np.isfinite(x_imp_gap)) or not np.all(np.isfinite(x_true_gap)):
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        c_true = np.corrcoef(x_true_gap, rowvar=False)
        c_imp = np.corrcoef(x_imp_gap, rowvar=False)
    c_true = np.nan_to_num(c_true, nan=0.0)
    c_imp = np.nan_to_num(c_imp, nan=0.0)
    off_diag = ~np.eye(n_cols, dtype=bool)
    return float(np.mean(np.abs(c_imp[off_diag] - c_true[off_diag])))


def compute_intrinsic_metrics(
    *,
    clean_window: pd.DataFrame,
    exec_result: ImputationExecutionResult,
    masked_mask: np.ndarray,
    masked_cols: Sequence[str],
    train_std_by_col: Dict[str, float],
    gap_length: int,
) -> Dict[str, float]:
    """Compute Branch 1 metrics on masked cells with scenario-wise vs weighted-mean Double-Penalty separation."""
    nan_metrics = {
        "nrmse_mean": np.nan,
        "nmae_mean": np.nan,
        "nrmse_scen": np.nan,
        "nmae_scen": np.nan,
        "vr_scen": np.nan,
        "vr_mean": np.nan,
        "fe_fidelity_nmae_scen": np.nan,
        "fe_fidelity_nmae_mean": np.nan,
        "cross_corr_err_scen": np.nan,
        "cross_corr_err_mean": np.nan,
        "domain_violation_pct": np.nan,
    }
    if gap_length == 0 or exec_result.status != "OK" or exec_result.is_raw_nan:
        if gap_length == 0 and exec_result.status == "OK" and not exec_result.is_raw_nan:
            return {
                "nrmse_mean": 0.0,
                "nmae_mean": 0.0,
                "nrmse_scen": 0.0,
                "nmae_scen": 0.0,
                "vr_scen": 1.0,
                "vr_mean": 1.0,
                "fe_fidelity_nmae_scen": 0.0,
                "fe_fidelity_nmae_mean": 0.0,
                "cross_corr_err_scen": 0.0,
                "cross_corr_err_mean": 0.0,
                "domain_violation_pct": 0.0,
            }
        return nan_metrics

    col_list = list(masked_cols)
    stds = np.array([train_std_by_col[c] for c in col_list], dtype=np.float64)
    fe_scales = np.repeat(stds, 3)
    row_slice = np.any(masked_mask, axis=1)

    x_true_gap = clean_window.loc[row_slice, col_list].to_numpy(dtype=np.float64)
    true_vars = np.var(x_true_gap, axis=0)
    true_fe = _compute_derived_fe_vector(clean_window, col_list)

    w = exec_result.weights / np.sum(exec_result.weights)
    scen_gaps = [scen.loc[row_slice, col_list].to_numpy(dtype=np.float64) for scen in exec_result.scenarios]
    x_weighted_mean_gap = np.sum(np.stack(scen_gaps, axis=0) * w[:, None, None], axis=0)

    # Construct weighted-mean full window for weighted-mean FE fidelity evaluation
    weighted_mean_full = clean_window.copy()
    for j, col in enumerate(col_list):
        weighted_mean_full.loc[row_slice, col] = x_weighted_mean_gap[:, j]

    def _eval_single(x_gap: np.ndarray, full_df: pd.DataFrame) -> Tuple[float, float, float, float, float]:
        if not np.all(np.isfinite(x_gap)):
            return np.nan, np.nan, np.nan, np.nan, np.nan
        diff = x_gap - x_true_gap
        nrmse = float(np.mean(np.sqrt(np.mean(diff ** 2, axis=0)) / stds))
        nmae = float(np.mean(np.mean(np.abs(diff), axis=0) / stds))
        vr = float(np.mean(np.var(x_gap, axis=0) / (true_vars + 1e-6)))
        fe_hat = _compute_derived_fe_vector(full_df, col_list)
        fe_err = float(np.mean(np.abs(fe_hat - true_fe) / fe_scales)) if np.all(np.isfinite(fe_hat)) else np.nan
        cc_err = _cross_correlation_error(x_gap, x_true_gap)
        return nrmse, nmae, vr, fe_err, cc_err

    nrmse_m, nmae_m, vr_m, fe_m, cc_m = _eval_single(x_weighted_mean_gap, weighted_mean_full)

    per_s = [_eval_single(xg, sdf) for xg, sdf in zip(scen_gaps, exec_result.scenarios)]
    arr_s = np.asarray(per_s, dtype=np.float64)

    def _wavg(vals: np.ndarray) -> float:
        return float(np.sum(w * vals)) if np.all(np.isfinite(vals)) else np.nan

    dom_viol = compute_domain_violation_rate(exec_result.scenarios, w, masked_mask, col_list)
    return {
        "nrmse_mean": nrmse_m,
        "nmae_mean": nmae_m,
        "nrmse_scen": _wavg(arr_s[:, 0]),
        "nmae_scen": _wavg(arr_s[:, 1]),
        "vr_scen": _wavg(arr_s[:, 2]),
        "vr_mean": vr_m,
        "fe_fidelity_nmae_scen": _wavg(arr_s[:, 3]),
        "fe_fidelity_nmae_mean": fe_m,
        "cross_corr_err_scen": _wavg(arr_s[:, 4]),
        "cross_corr_err_mean": cc_m,
        "domain_violation_pct": dom_viol,
    }


def _predict_models_from_f3_batch(
    *,
    x_f3: np.ndarray,
    weights: np.ndarray,
    models: Dict[str, Tuple[Any, int]],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Run batch prediction on SC-31 (70 cols) & SC-30 (65 cols) and return `(y_agg_7d, sigma_7d)`."""
    w = weights / np.sum(weights)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for model_id, (model_obj, n_cols) in models.items():
        if len(x_f3) == 0:
            out[model_id] = (np.full(7, np.nan), np.full(7, np.nan))
            continue
        x_in = x_f3[:, :n_cols]
        preds = np.clip(np.asarray(model_obj.predict(x_in), dtype=np.float64), 0.0, None)
        if preds.ndim == 1:
            preds = preds.reshape(1, -1)
        y_mean = np.sum(preds * w[:, None], axis=0)
        sigma = np.sqrt(np.maximum(np.sum(w[:, None] * ((preds - y_mean[None, :]) ** 2), axis=0), 0.0))
        out[model_id] = (y_mean, sigma)
    return out


def measure_isolated_single_anchor_rss(conf: ImputationBenchmarkConfig) -> Dict[str, float]:
    """Measure standalone single-anchor inference RSS at L=90 (Profile C & Profile A Mode 2) in a clean subprocess."""
    ceres_cols = load_ceres_syn1deg_vi_columns(conf.catalog_path)
    script = f"""
import json, joblib, numpy as np, pandas as pd, psutil
from pathlib import Path
from src.data.loader import time_series_split
from src.data.vintage_store import connect_sqlite, read_latest_observations
from research.imputer_benchmark.exogenous_imputer import (
    CANONICAL_EXOGENOUS_COLUMNS, ClimatologyLookupTable, ExogenousImputationConfig,
    ExogenousImputerWrapper, HistoricalAnalogStore,
)
p = psutil.Process()
csv_path = {repr(str(conf.csv_path)) if conf.csv_path is not None else "None"}
if csv_path:
    df = pd.read_csv(csv_path)
else:
    conn = connect_sqlite({repr(str(conf.db_path))}, read_only=True)
    try:
        df = read_latest_observations(conn, location_id={repr(conf.location_id)}, as_vietnamese=True)
    finally:
        conn.close()
df['Ngày'] = pd.to_datetime(df['Ngày']).dt.normalize()
df = df[(df['Ngày'] >= '2001-01-01') & (df['Ngày'] <= {repr(conf.test_end_date)})].sort_values('Ngày').reset_index(drop=True)
dev_train, _ = time_series_split(df, date_col='Ngày', test_start_date='2018-05-01')
m31 = joblib.load(Path({repr(str(conf.artifact_dir))}) / 'sc31_lightgbm_tweedie_f3.joblib')
fb3 = joblib.load(Path({repr(str(conf.artifact_dir))}) / 'feature_builder_f3.joblib')
cfg = ExogenousImputationConfig()
clim = ClimatologyLookupTable.fit(dev_train, config=cfg)
store = HistoricalAnalogStore.from_dataframe(dev_train, config=cfg)
wrapper = ExogenousImputerWrapper(model=m31, feature_builder=fb3, climatology=clim, analog_store=store, config=cfg)
win_c = df.tail(120).copy().reset_index(drop=True)
win_c.loc[win_c.index[-90:], list(CANONICAL_EXOGENOUS_COLUMNS)] = np.nan
_ = wrapper.predict(win_c)
rss_c = p.memory_info().rss / (1024.0 ** 2)
win_a = df.tail(120).copy().reset_index(drop=True)
win_a.loc[win_a.index[-90:], {repr(ceres_cols)}] = np.nan
_ = wrapper.predict(win_a)
rss_a = p.memory_info().rss / (1024.0 ** 2)
print(json.dumps({{"rss_l90_profile_c_mb": round(rss_c, 2), "rss_l90_profile_a_mode2_mb": round(rss_a, 2)}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    last_line = proc.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def run_benchmark_execution(
    cfg: Optional[ImputationBenchmarkConfig] = None,
) -> Dict[str, Any]:
    """Execute Steps 1-6 across all (position × L × profile × anchor × method) & both SC-31/SC-30 models."""
    conf = cfg or ImputationBenchmarkConfig()
    process = psutil.Process()

    full_df, dev_train_df, test_df = load_clean_single_location_splits(conf)
    p90_dev_train = float(np.percentile(dev_train_df[CANONICAL_TARGET_COL].to_numpy(dtype=np.float64), 90.0))

    m31 = joblib.load(conf.artifact_dir / "sc31_lightgbm_tweedie_f3.joblib")
    fb3 = joblib.load(conf.artifact_dir / "feature_builder_f3.joblib")
    m30 = joblib.load(conf.artifact_dir / "sc30_lightgbm_tweedie_f2.joblib")
    models_spec: Dict[str, Tuple[Any, int]] = {"SC-31": (m31, 70), "SC-30": (m30, 65)}

    shap_sc31 = compute_shap_exogenous_ranking(model=m31, feature_builder=fb3, dev_train_df=dev_train_df, model_id="SC-31")
    profile_a_cols = load_ceres_syn1deg_vi_columns(conf.catalog_path)
    profile_b_cols = shap_sc31["variable_vi"].head(conf.profile_b_top_k).tolist()
    profile_c_cols = list(CANONICAL_EXOGENOUS_COLUMNS)
    profiles_map: Dict[ProfileId, List[str]] = {"A": profile_a_cols, "B": profile_b_cols, "C": profile_c_cols}

    anchors_df = select_rolling_origin_anchors(
        full_df,
        test_start_date=conf.test_start_date,
        test_end_date=conf.test_end_date,
        n_anchors=conf.n_anchors,
        lookback_required_days=conf.lookback_required_days,
        horizon_days=conf.horizon_days,
    )

    # Full-data reference forecasts y_full(T_m, h) and observed targets y_true(T_m, h)
    anchor_windows: Dict[int, pd.DataFrame] = {}
    anchor_targets: Dict[int, np.ndarray] = {}
    full_preds_by_anchor: Dict[int, Dict[str, np.ndarray]] = {}
    for _, arow in anchors_df.iterrows():
        a_idx = int(arow["anchor_idx"])
        win_df = full_df.iloc[a_idx - conf.lookback_required_days + 1 : a_idx + 1].copy().reset_index(drop=True)
        anchor_windows[a_idx] = win_df
        anchor_targets[a_idx] = full_df[CANONICAL_TARGET_COL].iloc[a_idx + 1 : a_idx + 8].to_numpy(dtype=np.float64)
        x3_full, _ = _extract_f3_batch(fb3, [win_df], is_raw_nan=False)
        preds_map = _predict_models_from_f3_batch(x_f3=x3_full, weights=np.array([1.0]), models=models_spec)
        full_preds_by_anchor[a_idx] = {mid: val[0] for mid, val in preds_map.items()}

    registry = ImputationBaselineRegistry.fit_on_dev_train(
        dev_train_df,
        model=m31,
        feature_builder=fb3,
        hybrid_threshold_days=conf.hybrid_threshold_days,
        random_state=conf.random_state,
        dev_train_end_date=conf.dev_train_end_date,
    )

    raw_records: List[Dict[str, Any]] = []
    intrinsic_records: List[Dict[str, Any]] = []
    gate_nan_failures: List[str] = []
    peak_rss_mb = float(process.memory_info().rss) / (1024.0 * 1024.0)
    heavy_single_anchor_rss_mb = 0.0
    l0_cache: Dict[
        Tuple[int, str],
        Tuple[ImputationExecutionResult, float, float, Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, float]],
    ] = {}

    for pos in conf.gap_positions:
        for profile_id in conf.profiles:
            masked_cols = profiles_map[profile_id]
            for gap_len in conf.gap_lengths:
                for _, arow in anchors_df.iterrows():
                    a_idx = int(arow["anchor_idx"])
                    a_date = str(arow["anchor_date"])
                    season = str(arow["season_regime"])
                    clean_win = anchor_windows[a_idx]
                    y_true = anchor_targets[a_idx]

                    masked_df, bool_mask = apply_block_mask(
                        clean_win,
                        gap_position=pos,
                        gap_length=int(gap_len),
                        masked_cols=masked_cols,
                        interior_offset_days=conf.interior_offset_days,
                    )

                    # Precomputed per-anchor custom_mode1 / custom_mode2 cache for exact hybrid & L* reuse
                    cache_custom: Dict[str, Tuple[ImputationExecutionResult, float, float, Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, float]]] = {}

                    for method in ALL_BENCHMARK_METHODS:
                        if int(gap_len) == 0 and (a_idx, method) in l0_cache:
                            exec_res, imp_ms, fe_pred_ms, model_outs, intr = l0_cache[(a_idx, method)]
                            if method in ("custom_mode1", "custom_mode2"):
                                cache_custom[method] = (exec_res, imp_ms, fe_pred_ms, model_outs, intr)
                        elif method == "custom_hybrid":
                            source_key = "custom_mode1" if gap_len <= conf.hybrid_threshold_days else "custom_mode2"
                            _, imp_ms, fe_pred_ms, model_outs, intr = cache_custom[source_key]
                            exec_res = combine_hybrid_from_mode1_mode2(
                                mode1_res=cache_custom["custom_mode1"][0],
                                mode2_res=cache_custom["custom_mode2"][0],
                                gap_length=int(gap_len),
                                threshold_l_star=conf.hybrid_threshold_days,
                            )
                            if int(gap_len) == 0:
                                l0_cache[(a_idx, method)] = (exec_res, imp_ms, fe_pred_ms, model_outs, intr)
                        else:
                            t0 = time.perf_counter()
                            exec_res = registry.impute_window(
                                method,
                                masked_df,
                                gap_position=pos,
                                gap_length=int(gap_len),
                                profile_id=profile_id,
                            )
                            imp_ms = (time.perf_counter() - t0) * 1000.0

                            if exec_res.status != "OK":
                                fe_pred_ms = 0.0
                                model_outs = {mid: (np.full(7, np.nan), np.full(7, np.nan)) for mid in models_spec}
                            else:
                                t1 = time.perf_counter()
                                x3_batch, err_note = _extract_f3_batch(
                                    fb3,
                                    exec_res.scenarios,
                                    is_raw_nan=exec_res.is_raw_nan,
                                )
                                if err_note is not None:
                                    exec_res.reason = err_note
                                if not exec_res.is_raw_nan and len(x3_batch) > 0:
                                    n_bad = count_nan_or_inf_in_matrix(x3_batch)
                                    if n_bad > 0:
                                        gate_nan_failures.append(
                                            f"{method}/{pos}/L={gap_len}/{profile_id}/{a_date}: {n_bad} NaNs"
                                        )
                                model_outs = _predict_models_from_f3_batch(
                                    x_f3=x3_batch,
                                    weights=exec_res.weights,
                                    models=models_spec,
                                )
                                fe_pred_ms = (time.perf_counter() - t1) * 1000.0

                            intr = compute_intrinsic_metrics(
                                clean_window=clean_win,
                                exec_result=exec_res,
                                masked_mask=bool_mask,
                                masked_cols=masked_cols,
                                train_std_by_col=registry.train_std_by_col,
                                gap_length=int(gap_len),
                            )
                            if method in ("custom_mode1", "custom_mode2"):
                                cache_custom[method] = (exec_res, imp_ms, fe_pred_ms, model_outs, intr)
                            if int(gap_len) == 0:
                                l0_cache[(a_idx, method)] = (exec_res, imp_ms, fe_pred_ms, model_outs, intr)

                        cur_rss = float(process.memory_info().rss) / (1024.0 * 1024.0)
                        if cur_rss > peak_rss_mb:
                            peak_rss_mb = cur_rss
                        if gap_len == 90 and profile_id == "C" and method == "custom_hybrid":
                            heavy_single_anchor_rss_mb = max(heavy_single_anchor_rss_mb, cur_rss)

                        intrinsic_records.append(
                            {
                                "anchor_date": a_date,
                                "season_regime": season,
                                "position": pos,
                                "gap_length": int(gap_len),
                                "profile": profile_id,
                                "method": method,
                                "status": exec_res.status,
                                "imputation_mode": exec_res.imputation_mode,
                                "effective_k": exec_res.effective_k,
                                "impute_ms": imp_ms,
                                "fe_predict_ms": fe_pred_ms,
                                "total_ms": imp_ms + fe_pred_ms,
                                **intr,
                            }
                        )

                        for model_id in models_spec:
                            y_full = full_preds_by_anchor[a_idx][model_id]
                            y_imp, sigma_imp = model_outs[model_id]
                            for h_idx in range(conf.horizon_days):
                                raw_records.append(
                                    {
                                        "model_id": model_id,
                                        "anchor_date": a_date,
                                        "season_regime": season,
                                        "position": pos,
                                        "gap_length": int(gap_len),
                                        "profile": profile_id,
                                        "method": method,
                                        "status": exec_res.status,
                                        "horizon": h_idx + 1,
                                        "y_true": float(y_true[h_idx]),
                                        "y_full": float(y_full[h_idx]),
                                        "y_imp": float(y_imp[h_idx]),
                                        "sigma_imp": float(sigma_imp[h_idx]),
                                        "impute_ms": imp_ms,
                                        "fe_predict_ms": fe_pred_ms,
                                    }
                                )

    raw_df = pd.DataFrame(raw_records)
    intrinsic_df = pd.DataFrame(intrinsic_records)

    # Sensitivity records over L* in {3, 5, 7, 14} using exact precomputed Mode 1 / Mode 2 outputs
    sens_rows: List[Dict[str, Any]] = []
    sub_m1 = raw_df[raw_df["method"] == "custom_mode1"].copy()
    sub_m2 = raw_df[raw_df["method"] == "custom_mode2"].copy()
    for l_star in conf.sensitivity_thresholds:
        m1_part = sub_m1[sub_m1["gap_length"] <= int(l_star)].copy()
        m2_part = sub_m2[sub_m2["gap_length"] > int(l_star)].copy()
        merged_part = pd.concat([m1_part, m2_part], ignore_index=True)
        merged_part["threshold_l_star"] = int(l_star)
        merged_part["method"] = "custom_hybrid"
        sens_rows.append(merged_part)
    sensitivity_raw_df = pd.concat(sens_rows, ignore_index=True)

    isolated_rss = measure_isolated_single_anchor_rss(conf)

    return {
        "config": conf,
        "raw_df": raw_df,
        "intrinsic_df": intrinsic_df,
        "sensitivity_raw_df": sensitivity_raw_df,
        "shap_sc31": shap_sc31,
        "anchors_df": anchors_df,
        "profiles_map": profiles_map,
        "p90_dev_train": p90_dev_train,
        "dev_train_range": (
            pd.Timestamp(dev_train_df[CANONICAL_DATE_COL].min()).strftime("%Y-%m-%d"),
            pd.Timestamp(dev_train_df[CANONICAL_DATE_COL].max()).strftime("%Y-%m-%d"),
            len(dev_train_df),
        ),
        "test_range": (
            pd.Timestamp(test_df[CANONICAL_DATE_COL].min()).strftime("%Y-%m-%d"),
            pd.Timestamp(test_df[CANONICAL_DATE_COL].max()).strftime("%Y-%m-%d"),
            len(test_df),
        ),
        "gate_nan_failures": gate_nan_failures,
        "peak_rss_mb": peak_rss_mb,
        "heavy_single_anchor_rss_mb": isolated_rss["rss_l90_profile_c_mb"],
        "isolated_rss_profile_a_mode2_mb": isolated_rss["rss_l90_profile_a_mode2_mb"],
        "runner_rss_l90_profile_c_mb": heavy_single_anchor_rss_mb,
        "n_raw_vars": len(ALL_35_VARS),
    }
