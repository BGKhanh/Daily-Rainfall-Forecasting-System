"""
Standalone Inference-Time Exogenous Imputation & Marginalization Wrapper (DS108)
==============================================================================
Self-contained library module for exogenous-variable gap handling (Mode 0/1/2/3):
- Strictly isolated from operational `POST /predict`, `WeatherHistoryProvider`,
  `StandaloneRainfallPredictor.predict` default path, and `automated_pipeline.py`.
- Fits climatology & analog reference strictly on canonical `Dev_Train` (<= 2018-04-30).
- Routes completed 36-column scenario DataFrames through `align_and_validate_canonical_raw_df`
  and frozen `FeatureBuilder.build_single_step` (Single Source of Truth).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.data.dataquality import check_continuity
from src.data.loader import time_series_split
from src.data.vintage_store import (
    CANONICAL_VI_COLUMN_ORDER,
    EN_TO_VI_CANONICAL,
    align_and_validate_canonical_raw_df,
)
from src.inference.validation import (
    CANONICAL_DATE_COL,
    CANONICAL_NUMERIC_COLUMNS,
    CANONICAL_TARGET_COL,
    PHYSICAL_METEO_BOUNDS,
    SENTINEL_THRESHOLD,
)

CANONICAL_EXOGENOUS_COLUMNS: Tuple[str, ...] = tuple(
    col for col in CANONICAL_VI_COLUMN_ORDER if col not in (CANONICAL_DATE_COL, CANONICAL_TARGET_COL)
)

_DEW_POINT_COL = "Điểm sương 2m"
_TEMP_2M_COL = "Nhiệt độ 2m"


@dataclass(frozen=True)
class ExogenousImputationConfig:
    """Configuration parameters for standalone exogenous imputation and marginalization."""

    mode1_max_gap_days: int = 7
    mode2_max_gap_days: int = 90
    boundary_blend_max_gap_days: int = 14
    circular_smoothing_radius: int = 7
    analog_top_k: int = 10
    analog_match_window_days: int = 30
    seasonal_phase_radius_days: int = 5
    expanded_phase_radius_days: int = 15
    feature_warmup_days: int = 30
    aggregation_strategy: Literal["weighted_median", "weighted_mean"] = "weighted_median"
    variable_weights: Optional[Mapping[str, float]] = None
    dev_train_end_date: str = "2018-04-30"
    sentinel_threshold: float = SENTINEL_THRESHOLD

    def __post_init__(self) -> None:
        if self.mode1_max_gap_days < 1:
            raise ValueError("mode1_max_gap_days must be >= 1.")
        if self.mode2_max_gap_days <= self.mode1_max_gap_days:
            raise ValueError("mode2_max_gap_days must exceed mode1_max_gap_days.")
        if self.boundary_blend_max_gap_days < self.mode1_max_gap_days:
            raise ValueError("boundary_blend_max_gap_days must be >= mode1_max_gap_days.")
        if self.circular_smoothing_radius < 0 or self.circular_smoothing_radius > 60:
            raise ValueError("circular_smoothing_radius must be in [0, 60].")
        if self.analog_top_k < 1:
            raise ValueError("analog_top_k must be >= 1.")
        if self.analog_match_window_days < 1:
            raise ValueError("analog_match_window_days must be >= 1.")
        if self.seasonal_phase_radius_days < 0:
            raise ValueError("seasonal_phase_radius_days must be >= 0.")
        if self.expanded_phase_radius_days < self.seasonal_phase_radius_days:
            raise ValueError("expanded_phase_radius_days must be >= seasonal_phase_radius_days.")
        if self.feature_warmup_days < 1:
            raise ValueError("feature_warmup_days must be >= 1.")
        if self.aggregation_strategy not in ("weighted_median", "weighted_mean"):
            raise ValueError(f"Unsupported aggregation_strategy={self.aggregation_strategy!r}.")
        if self.variable_weights is not None:
            for key, weight in self.variable_weights.items():
                if not np.isfinite(float(weight)) or float(weight) < 0.0:
                    raise ValueError(f"variable_weights[{key!r}] must be finite and >= 0; got {weight}.")


def _dates_to_circular_doy_365(dates: Union[pd.Series, pd.DatetimeIndex]) -> np.ndarray:
    """
    Map timestamps onto circular day-of-year indices in `{1, ..., 365}`.

    Leap day Feb 29 (doy=60) shares seasonal phase index 59 with Feb 28 and shifts subsequent
    days Mar 1..Dec 31 by -1 so calendar seasons never drift on leap years.
    """
    dt_index = pd.DatetimeIndex(dates)
    doy = dt_index.dayofyear.to_numpy(dtype=np.int64).copy()
    after_feb28 = dt_index.is_leap_year & (doy >= 60)
    doy[after_feb28] = doy[after_feb28] - 1
    return np.clip(doy, 1, 365)


def _circular_doy_distance(d1: Union[int, np.ndarray], d2: int) -> Union[int, np.ndarray]:
    """Shortest distance around the 365-day annual circle in days."""
    diff = np.abs(np.asarray(d1, dtype=np.int64) - int(d2))
    out = np.minimum(diff, 365 - diff)
    return int(out) if np.ndim(d1) == 0 else out


def _circular_fill_missing_days(daily_medians: np.ndarray) -> np.ndarray:
    """Fill unobserved DOY slots on synthetic sparse training series around the 365-day ring."""
    arr = daily_medians.astype(np.float64).copy()
    valid_idx = np.flatnonzero(np.isfinite(arr))
    if len(valid_idx) == 0:
        return np.zeros(365, dtype=np.float64)
    if len(valid_idx) == 365:
        return arr
    xp = np.concatenate([valid_idx - 365, valid_idx, valid_idx + 365]).astype(np.float64)
    fp = np.concatenate([arr[valid_idx], arr[valid_idx], arr[valid_idx]]).astype(np.float64)
    return np.interp(np.arange(365, dtype=np.float64), xp, fp)


def _circular_smooth_365(daily_values: np.ndarray, radius: int) -> np.ndarray:
    """Circular moving average C_v(d) with symmetric radius r on the 365-day year."""
    if radius <= 0:
        return daily_values.astype(np.float64).copy()
    base = np.asarray(daily_values, dtype=np.float64)
    out = np.empty(365, dtype=np.float64)
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    for idx in range(365):
        out[idx] = float(np.mean(base[(idx + offsets) % 365]))
    return out


def _prepare_canonical_sorted_df(
    df: pd.DataFrame,
    *,
    sentinel_threshold: float = SENTINEL_THRESHOLD,
) -> pd.DataFrame:
    """Align DataFrame to 36 canonical Vietnamese columns and normalize sorted dates/sentinels."""
    aligned = align_and_validate_canonical_raw_df(df)
    aligned[CANONICAL_DATE_COL] = pd.to_datetime(aligned[CANONICAL_DATE_COL], errors="coerce")
    if aligned[CANONICAL_DATE_COL].isna().any():
        raise ValueError("Invalid or unparseable dates encountered in 'Ngày' column.")
    aligned[CANONICAL_DATE_COL] = aligned[CANONICAL_DATE_COL].dt.normalize()
    aligned = aligned.sort_values(CANONICAL_DATE_COL).reset_index(drop=True)
    if aligned[CANONICAL_DATE_COL].duplicated().any():
        raise ValueError("Duplicate dates encountered in input meteorological sequence.")

    for col in CANONICAL_NUMERIC_COLUMNS:
        vals = pd.to_numeric(aligned[col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        vals[~np.isfinite(vals)] = np.nan
        vals[vals <= sentinel_threshold] = np.nan
        aligned[col] = vals
    return aligned


def _prepare_dev_train_frame(
    train_df: pd.DataFrame,
    *,
    min_rows: int,
    config: ExogenousImputationConfig,
    enforce_dev_train_cutoff: bool,
    caller: str,
) -> pd.DataFrame:
    """Validate and prepare training DataFrame strictly honoring Dev_Train split boundary."""
    if train_df is None or not isinstance(train_df, pd.DataFrame) or len(train_df) < min_rows:
        raise ValueError(f"{caller}() requires at least {min_rows} rows of Dev_Train history.")

    df = _prepare_canonical_sorted_df(train_df, sentinel_threshold=config.sentinel_threshold)
    if enforce_dev_train_cutoff:
        cutoff_inclusive = pd.Timestamp(config.dev_train_end_date).normalize()
        next_split_start = (cutoff_inclusive + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        _, post_train_df = time_series_split(df, date_col=CANONICAL_DATE_COL, test_start_date=next_split_start)
        if len(post_train_df) > 0:
            max_dt = pd.Timestamp(post_train_df[CANONICAL_DATE_COL].max()).normalize()
            raise ValueError(
                f"[DATA_LEAKAGE_GUARD] {caller}() received observations through {max_dt.strftime('%Y-%m-%d')}, "
                f"which exceeds canonical Dev_Train cutoff {cutoff_inclusive.strftime('%Y-%m-%d')}."
            )
    return df


@dataclass
class ClimatologyLookupTable:
    """Seasonal circular climatology C_v(d) and OU anomaly statistics fit strictly on Dev_Train."""

    exogenous_columns: Tuple[str, ...]
    climatology: Dict[str, np.ndarray]
    std: Dict[str, float]
    iqr: Dict[str, float]
    rho1: Dict[str, float]
    tau: Dict[str, float]
    bounds: Dict[str, Tuple[float, float]]
    target_std: float
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    n_train_rows: int
    circular_smoothing_radius: int

    @classmethod
    def fit(
        cls,
        train_df: pd.DataFrame,
        *,
        config: Optional[ExogenousImputationConfig] = None,
        enforce_dev_train_cutoff: bool = True,
    ) -> "ClimatologyLookupTable":
        """Fit circular median climatology and OU persistence constants strictly on Dev_Train."""
        cfg = config or ExogenousImputationConfig()
        df = _prepare_dev_train_frame(
            train_df,
            min_rows=30,
            config=cfg,
            enforce_dev_train_cutoff=enforce_dev_train_cutoff,
            caller="ClimatologyLookupTable.fit",
        )

        doy_365 = _dates_to_circular_doy_365(df[CANONICAL_DATE_COL])
        consec_mask = (df[CANONICAL_DATE_COL].diff().dt.days == 1).to_numpy(dtype=bool)

        climatology: Dict[str, np.ndarray] = {}
        std_map: Dict[str, float] = {}
        iqr_map: Dict[str, float] = {}
        rho1_map: Dict[str, float] = {}
        tau_map: Dict[str, float] = {}
        bounds_map: Dict[str, Tuple[float, float]] = {}

        for col in CANONICAL_EXOGENOUS_COLUMNS:
            vals = df[col].to_numpy(dtype=np.float64)
            daily_med = np.full(365, np.nan, dtype=np.float64)
            for d in range(1, 366):
                mask_d = (doy_365 == d) & np.isfinite(vals)
                if np.any(mask_d):
                    daily_med[d - 1] = float(np.median(vals[mask_d]))

            filled_med = _circular_fill_missing_days(daily_med)
            smooth_clim = _circular_smooth_365(filled_med, radius=cfg.circular_smoothing_radius)
            lo, hi = PHYSICAL_METEO_BOUNDS[col]
            smooth_clim = np.clip(smooth_clim, lo, hi)
            climatology[col] = smooth_clim
            bounds_map[col] = (float(lo), float(hi))

            anomalies = vals - smooth_clim[doy_365 - 1]
            finite_anom = anomalies[np.isfinite(anomalies)]
            if len(finite_anom) > 1:
                sigma = max(float(np.std(finite_anom, ddof=0)), 1e-4)
                q75, q25 = np.percentile(finite_anom, [75.0, 25.0])
                iqr_val = max(float(q75 - q25), 1e-4)
            else:
                sigma = 1.0
                iqr_val = 1.0
            std_map[col] = sigma
            iqr_map[col] = iqr_val

            pair_mask = consec_mask.copy()
            pair_mask[0] = False
            prev_vals = np.roll(anomalies, 1)
            valid_pairs = pair_mask & np.isfinite(anomalies) & np.isfinite(prev_vals)
            raw_rho = 0.05
            if int(np.sum(valid_pairs)) >= 3:
                x_prev, x_curr = prev_vals[valid_pairs], anomalies[valid_pairs]
                if np.std(x_prev) > 1e-12 and np.std(x_curr) > 1e-12:
                    corr = float(np.corrcoef(x_prev, x_curr)[0, 1])
                    if np.isfinite(corr):
                        raw_rho = corr

            clipped_rho = float(np.clip(raw_rho, 0.05, 0.98))
            rho1_map[col] = clipped_rho
            tau_map[col] = float(-1.0 / math.log(clipped_rho))

        target_vals = df[CANONICAL_TARGET_COL].to_numpy(dtype=np.float64)
        finite_target = target_vals[np.isfinite(target_vals)]
        target_std = max(float(np.std(finite_target, ddof=0)) if len(finite_target) > 1 else 1.0, 1e-4)

        return cls(
            exogenous_columns=CANONICAL_EXOGENOUS_COLUMNS,
            climatology=climatology,
            std=std_map,
            iqr=iqr_map,
            rho1=rho1_map,
            tau=tau_map,
            bounds=bounds_map,
            target_std=target_std,
            train_start_date=pd.Timestamp(df[CANONICAL_DATE_COL].iloc[0]).normalize(),
            train_end_date=pd.Timestamp(df[CANONICAL_DATE_COL].iloc[-1]).normalize(),
            n_train_rows=len(df),
            circular_smoothing_radius=cfg.circular_smoothing_radius,
        )

    def climatology_value(self, variable: str, dt: Union[str, pd.Timestamp]) -> float:
        """Return smoothed Dev_Train climatology C_v(d_t) for `variable` on date `dt`."""
        ts = pd.Timestamp(dt).normalize()
        doy = int(_dates_to_circular_doy_365(pd.DatetimeIndex([ts]))[0])
        return float(self.climatology[variable][doy - 1])

    def climatology_series(self, variable: str, dates: Union[pd.Series, pd.DatetimeIndex]) -> np.ndarray:
        """Vectorized lookup of circular climatology C_v(d_t) across a date series."""
        doys = _dates_to_circular_doy_365(dates)
        return self.climatology[variable][doys - 1].astype(np.float64, copy=True)


MODE_0_CLEAN = "MODE_0_CLEAN"
MODE_1_OU_CLIMATOLOGY = "MODE_1_OU_CLIMATOLOGY"
MODE_2_ANALOG_MARGINALIZATION = "MODE_2_ANALOG_MARGINALIZATION"
MODE_3_UNCONDITIONAL_SEASONAL_FALLBACK = "MODE_3_UNCONDITIONAL_SEASONAL_FALLBACK"

ImputationMode = Literal[
    "MODE_0_CLEAN",
    "MODE_1_OU_CLIMATOLOGY",
    "MODE_2_ANALOG_MARGINALIZATION",
    "MODE_3_UNCONDITIONAL_SEASONAL_FALLBACK",
]


@dataclass(frozen=True)
class GapRun:
    """Contiguous missing interval inside the inspected history tail for one exogenous variable."""

    variable: str
    start_idx: int
    end_idx: int
    length: int
    gap_type: Literal["trailing", "interior", "leading", "all_missing"]
    left_anchor_idx: Optional[int]
    right_anchor_idx: Optional[int]


@dataclass
class GapInspectionReport:
    """Structured diagnostic output produced by `GapInspector.inspect`."""

    df_aligned: pd.DataFrame
    mode: ImputationMode
    max_gap_length: int
    missing_variables: List[str]
    gap_runs_by_var: Dict[str, List[GapRun]]
    observed_matching_variables: List[str]
    inspected_start_date: pd.Timestamp
    inspected_end_date: pd.Timestamp
    inspected_rows: int


def _extract_contiguous_runs(variable: str, missing_mask: np.ndarray) -> List[GapRun]:
    """Identify contiguous boolean missing intervals and classify anchor endpoints."""
    n = len(missing_mask)
    runs: List[GapRun] = []
    idx = 0
    while idx < n:
        if not missing_mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and missing_mask[idx]:
            idx += 1
        end = idx - 1
        left_anchor = start - 1 if start > 0 else None
        right_anchor = end + 1 if end < n - 1 else None
        if left_anchor is None and right_anchor is None:
            gap_type: Literal["trailing", "interior", "leading", "all_missing"] = "all_missing"
        elif right_anchor is None:
            gap_type = "trailing"
        elif left_anchor is None:
            gap_type = "leading"
        else:
            gap_type = "interior"
        runs.append(
            GapRun(
                variable=variable,
                start_idx=start,
                end_idx=end,
                length=end - start + 1,
                gap_type=gap_type,
                left_anchor_idx=left_anchor,
                right_anchor_idx=right_anchor,
            )
        )
    return runs


class GapInspector:
    """Inspect causal input history window, validate target/calendar continuity, and route mode."""

    @staticmethod
    def inspect(
        history_df: pd.DataFrame,
        *,
        config: Optional[ExogenousImputationConfig] = None,
        min_history_days: int = 35,
    ) -> GapInspectionReport:
        cfg = config or ExogenousImputationConfig()
        if history_df is None or not isinstance(history_df, pd.DataFrame):
            raise ValueError("history_df must be a pandas DataFrame.")

        aligned_full = _prepare_canonical_sorted_df(history_df, sentinel_threshold=cfg.sentinel_threshold)
        req_tail_len = max(cfg.mode2_max_gap_days, cfg.analog_match_window_days) + cfg.feature_warmup_days
        min_req = max(int(min_history_days), cfg.feature_warmup_days + 1)
        if len(aligned_full) < min_req:
            raise ValueError(
                f"Exogenous imputer requires at least {min_req} history days; got {len(aligned_full)}."
            )

        inspected = aligned_full.tail(min(len(aligned_full), req_tail_len)).copy().reset_index(drop=True)
        check_continuity(inspected, date_col=CANONICAL_DATE_COL, freq="D", raise_on_gap=True)

        target_vals = inspected[CANONICAL_TARGET_COL].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(target_vals)):
            bad_count = int(np.sum(~np.isfinite(target_vals)))
            raise ValueError(
                f"Target '{CANONICAL_TARGET_COL}' contains {bad_count} missing or non-finite value(s); "
                "exogenous imputer never imputes target precipitation."
            )
        if np.any(target_vals < 0.0):
            raise ValueError(f"Target '{CANONICAL_TARGET_COL}' contains negative value(s).")

        gap_runs_by_var: Dict[str, List[GapRun]] = {}
        missing_vars: List[str] = []
        max_gap = 0

        for col in CANONICAL_EXOGENOUS_COLUMNS:
            miss_mask = ~np.isfinite(inspected[col].to_numpy(dtype=np.float64))
            if np.any(miss_mask):
                runs = _extract_contiguous_runs(col, miss_mask)
                gap_runs_by_var[col] = runs
                missing_vars.append(col)
                max_gap = max(max_gap, max(r.length for r in runs))

        match_len = min(cfg.analog_match_window_days, len(inspected))
        match_slice = inspected.tail(match_len)
        observed_matching_vars = [
            col
            for col in CANONICAL_EXOGENOUS_COLUMNS
            if bool(np.any(np.isfinite(match_slice[col].to_numpy(dtype=np.float64))))
        ]

        if max_gap == 0:
            mode: ImputationMode = MODE_0_CLEAN
        elif max_gap <= cfg.mode1_max_gap_days:
            mode = MODE_1_OU_CLIMATOLOGY
        elif max_gap <= cfg.mode2_max_gap_days and len(observed_matching_vars) > 0:
            mode = MODE_2_ANALOG_MARGINALIZATION
        else:
            mode = MODE_3_UNCONDITIONAL_SEASONAL_FALLBACK

        return GapInspectionReport(
            df_aligned=inspected,
            mode=mode,
            max_gap_length=max_gap,
            missing_variables=missing_vars,
            gap_runs_by_var=gap_runs_by_var,
            observed_matching_variables=observed_matching_vars,
            inspected_start_date=pd.Timestamp(inspected[CANONICAL_DATE_COL].iloc[0]).normalize(),
            inspected_end_date=pd.Timestamp(inspected[CANONICAL_DATE_COL].iloc[-1]).normalize(),
            inspected_rows=len(inspected),
        )


def _extract_feature_matrix(feature_builder: Any, scenarios: Sequence[pd.DataFrame]) -> np.ndarray:
    """Materialize post-fill feature rows via frozen `FeatureBuilder.build_single_step`."""
    date_col = getattr(feature_builder, "date_col", CANONICAL_DATE_COL)
    target_col = getattr(feature_builder, "target_col", CANONICAL_TARGET_COL)
    rows: List[np.ndarray] = []
    expected_cols: Optional[List[str]] = None

    for scenario_df in scenarios:
        canonical_df = align_and_validate_canonical_raw_df(scenario_df)
        feat_df = feature_builder.build_single_step(canonical_df)
        feat_cols = [c for c in feat_df.columns if c not in (date_col, target_col)]
        if expected_cols is None:
            expected_cols = feat_cols
        rows.append(feat_df[expected_cols].to_numpy(dtype=np.float64)[0])

    return np.vstack(rows)


def _predict_batch_7d(model: Any, x_batch: np.ndarray) -> np.ndarray:
    """Invoke frozen model once per scenario batch, verify finiteness, and clamp >= 0.0 mm/day."""
    raw = np.asarray(model.predict(x_batch), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != 7:
        raise ValueError(f"Expected 7-horizon prediction output, got shape {raw.shape}.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("Frozen model returned non-finite (NaN or Inf) forecast predictions.")
    return np.clip(raw, 0.0, None)


def _weighted_median_1d(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median over normalized weights w_k >= 0."""
    if len(values) == 1:
        return float(values[0])
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    sorted_w = weights[order]
    total_w = float(np.sum(sorted_w))
    if total_w <= 0.0:
        return float(np.median(sorted_vals))
    cdf = np.cumsum(sorted_w / total_w)
    idx = min(max(int(np.searchsorted(cdf, 0.5, side="left")), 0), len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def _aggregate_scenario_predictions(
    preds_k_by_7: np.ndarray,
    weights: np.ndarray,
    strategy: Literal["weighted_median", "weighted_mean"],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Compute weighted mean, weighted median, imputation std uncertainty, and chosen forecast."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    weighted_mean = np.sum(preds_k_by_7 * w[:, None], axis=0)
    weighted_median = np.array(
        [_weighted_median_1d(preds_k_by_7[:, h], w) for h in range(preds_k_by_7.shape[1])],
        dtype=np.float64,
    )
    variance = np.sum(w[:, None] * ((preds_k_by_7 - weighted_mean[None, :]) ** 2), axis=0)
    uncertainty = np.sqrt(np.maximum(variance, 0.0))
    chosen = weighted_median if strategy == "weighted_median" else weighted_mean
    return chosen.tolist(), weighted_mean.tolist(), weighted_median.tolist(), uncertainty.tolist()


def _enforce_dew_point_coupling(
    out: pd.DataFrame,
    imputed_mask: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Enforce T2MDEW <= T2M on imputed cells while keeping non-imputed observed cells intact."""
    t2m = out[_TEMP_2M_COL].to_numpy(dtype=np.float64, copy=True)
    dew = out[_DEW_POINT_COL].to_numpy(dtype=np.float64, copy=True)
    t2m_imp = imputed_mask.get(_TEMP_2M_COL, np.zeros(len(out), dtype=bool))
    dew_imp = imputed_mask.get(_DEW_POINT_COL, np.zeros(len(out), dtype=bool))
    lo_t2m, hi_t2m = PHYSICAL_METEO_BOUNDS[_TEMP_2M_COL]
    lo_dew, hi_dew = PHYSICAL_METEO_BOUNDS[_DEW_POINT_COL]
    viol = np.isfinite(t2m) & np.isfinite(dew) & (dew > t2m)
    if np.any(viol):
        only_t2m_imp = viol & t2m_imp & (~dew_imp)
        t2m[only_t2m_imp] = np.clip(dew[only_t2m_imp], lo_t2m, hi_t2m)
        clamp_dew = (dew > t2m) & dew_imp
        dew[clamp_dew] = np.clip(t2m[clamp_dew], lo_dew, hi_dew)
        out[_TEMP_2M_COL] = t2m
        out[_DEW_POINT_COL] = dew
    return out


def _apply_mode1_ou_run(
    vals: np.ndarray,
    clim_vec: np.ndarray,
    run: GapRun,
    *,
    tau_v: float,
    bounds: Tuple[float, float],
) -> None:
    """Mode 1 Ornstein-Uhlenbeck trailing decay or two-sided hyperbolic interior bridge."""
    lo, hi = bounds
    if run.gap_type == "trailing" and run.left_anchor_idx is not None:
        t0 = run.left_anchor_idx
        a0 = vals[t0] - clim_vec[t0]
        for t in range(run.start_idx, run.end_idx + 1):
            dt = float(t - t0)
            vals[t] = float(np.clip(clim_vec[t] + a0 * math.exp(-dt / tau_v), lo, hi))
    elif run.gap_type == "interior" and run.left_anchor_idx is not None and run.right_anchor_idx is not None:
        t0, t1 = run.left_anchor_idx, run.right_anchor_idx
        l_gap = float(t1 - t0)
        a0, a1 = vals[t0] - clim_vec[t0], vals[t1] - clim_vec[t1]
        ratio_total = l_gap / tau_v
        if ratio_total > 500.0:
            for t in range(run.start_idx, run.end_idx + 1):
                vals[t] = float(
                    np.clip(
                        clim_vec[t]
                        + a0 * math.exp(-float(t - t0) / tau_v)
                        + a1 * math.exp(-float(t1 - t) / tau_v),
                        lo,
                        hi,
                    )
                )
        else:
            denom = math.sinh(ratio_total)
            for t in range(run.start_idx, run.end_idx + 1):
                w0 = math.sinh(float(t1 - t) / tau_v) / denom
                w1 = math.sinh(float(t - t0) / tau_v) / denom
                vals[t] = float(np.clip(clim_vec[t] + w0 * a0 + w1 * a1, lo, hi))
    elif run.gap_type == "leading" and run.right_anchor_idx is not None:
        t1 = run.right_anchor_idx
        a1 = vals[t1] - clim_vec[t1]
        for t in range(run.start_idx, run.end_idx + 1):
            vals[t] = float(np.clip(clim_vec[t] + a1 * math.exp(-float(t1 - t) / tau_v), lo, hi))
    else:
        for t in range(run.start_idx, run.end_idx + 1):
            vals[t] = float(np.clip(clim_vec[t], lo, hi))


def _apply_left_anchor_boundary_blend(
    vals: np.ndarray,
    cand_slice: np.ndarray,
    run: GapRun,
    *,
    tau_v: float,
    bounds: Tuple[float, float],
) -> None:
    """Apply fast-decaying left-anchor offset blend on candidate slice entries for short Mode 2 gaps."""
    if run.left_anchor_idx is None:
        return
    t0 = run.left_anchor_idx
    delta0 = float(vals[t0] - cand_slice[t0])
    lo, hi = bounds
    for t in range(run.start_idx, run.end_idx + 1):
        dt = float(t - t0)
        vals[t] = float(np.clip(cand_slice[t] + delta0 * math.exp(-dt / tau_v), lo, hi))


@dataclass
class AnalogSearchResult:
    """Ranked historical analogs and adaptive kernel weights from `HistoricalAnalogStore`."""

    candidate_end_indices: np.ndarray
    candidate_end_dates: List[str]
    distances: np.ndarray
    weights: np.ndarray
    phase_radius_used: int
    phase_expanded: bool


class HistoricalAnalogStore:
    """Dev_Train historical analog repository enabling causal Mode 2 multivariate search."""

    def __init__(
        self,
        *,
        dates: pd.DatetimeIndex,
        date_ordinals_days: np.ndarray,
        doy_365: np.ndarray,
        matrix: np.ndarray,
        col_to_idx: Dict[str, int],
        row_all_finite: np.ndarray,
        clean_prefix: np.ndarray,
    ) -> None:
        self.dates = dates
        self.date_ordinals_days = date_ordinals_days
        self.doy_365 = doy_365
        self.matrix = matrix
        self.col_to_idx = col_to_idx
        self.row_all_finite = row_all_finite
        self.clean_prefix = clean_prefix

    @classmethod
    def from_dataframe(
        cls,
        train_df: pd.DataFrame,
        *,
        config: Optional[ExogenousImputationConfig] = None,
        enforce_dev_train_cutoff: bool = True,
    ) -> "HistoricalAnalogStore":
        """Build causal Dev_Train analog store (strictly <= 2018-04-30)."""
        cfg = config or ExogenousImputationConfig()
        df = _prepare_dev_train_frame(
            train_df,
            min_rows=35,
            config=cfg,
            enforce_dev_train_cutoff=enforce_dev_train_cutoff,
            caller="HistoricalAnalogStore.from_dataframe",
        )

        dates = pd.DatetimeIndex(df[CANONICAL_DATE_COL])
        epoch = pd.Timestamp("1970-01-01")
        date_ordinals_days = (dates - epoch).days.to_numpy(dtype=np.int64)
        doy_365 = _dates_to_circular_doy_365(dates)
        matrix = df.loc[:, list(CANONICAL_NUMERIC_COLUMNS)].to_numpy(dtype=np.float64)
        col_to_idx = {col: idx for idx, col in enumerate(CANONICAL_NUMERIC_COLUMNS)}
        row_all_finite = np.all(np.isfinite(matrix), axis=1)
        clean_prefix = np.concatenate([[0], np.cumsum(row_all_finite.astype(np.int64))])

        return cls(
            dates=dates,
            date_ordinals_days=date_ordinals_days,
            doy_365=doy_365,
            matrix=matrix,
            col_to_idx=col_to_idx,
            row_all_finite=row_all_finite,
            clean_prefix=clean_prefix,
        )

    def slice_variable(self, cand_end_idx: int, segment_len: int, variable: str) -> np.ndarray:
        """Return contiguous `segment_len`-day historical vector ending at `cand_end_idx`."""
        cand_start = int(cand_end_idx) - segment_len + 1
        return self.matrix[cand_start : int(cand_end_idx) + 1, self.col_to_idx[variable]]

    def _eligible_end_indices(
        self,
        *,
        segment_len: int,
        query_start_date: pd.Timestamp,
        query_end_doy: int,
        phase_radius_days: int,
    ) -> np.ndarray:
        """Return candidate indices `c` ending clean continuous `segment_len`-day slices strictly < query_start_date."""
        n = len(self.dates)
        if segment_len < 1 or n < segment_len:
            return np.empty(0, dtype=np.int64)

        cand = np.arange(segment_len - 1, n, dtype=np.int64)
        query_start_ordinal = int((pd.Timestamp(query_start_date).normalize() - pd.Timestamp("1970-01-01")).days)
        cand = cand[self.date_ordinals_days[cand] < query_start_ordinal]
        if len(cand) == 0:
            return cand

        starts = cand - (segment_len - 1)
        cont_mask = (self.date_ordinals_days[cand] - self.date_ordinals_days[starts]) == (segment_len - 1)
        clean_mask = (self.clean_prefix[cand + 1] - self.clean_prefix[starts]) == segment_len
        cand = cand[cont_mask & clean_mask]
        if len(cand) == 0:
            return cand

        return cand[_circular_doy_distance(self.doy_365[cand], query_end_doy) <= int(phase_radius_days)]

    @staticmethod
    def _resolve_variable_weights(
        match_vars: Sequence[str],
        custom_weights: Optional[Mapping[str, float]],
    ) -> np.ndarray:
        if not custom_weights:
            return np.full(len(match_vars), 1.0 / float(len(match_vars)), dtype=np.float64)
        normalized_map = {EN_TO_VI_CANONICAL.get(k, k): float(v) for k, v in custom_weights.items()}
        alphas = np.array([normalized_map.get(col, 1.0) for col in match_vars], dtype=np.float64)
        alpha_sum = float(np.sum(alphas))
        if alpha_sum <= 0.0:
            return np.full(len(match_vars), 1.0 / float(len(match_vars)), dtype=np.float64)
        return alphas / alpha_sum

    def search_analogs(
        self,
        report: GapInspectionReport,
        *,
        climatology: ClimatologyLookupTable,
        config: ExogenousImputationConfig,
    ) -> Optional[AnalogSearchResult]:
        """Find Top-K seasonal multivariate analogs using weighted L1 z-score distance."""
        df = report.df_aligned
        segment_len = len(df)
        match_len = min(config.analog_match_window_days, segment_len)
        query_end_doy = int(_dates_to_circular_doy_365(pd.DatetimeIndex([report.inspected_end_date]))[0])

        phase_radius = config.seasonal_phase_radius_days
        phase_expanded = False
        cand_indices = self._eligible_end_indices(
            segment_len=segment_len,
            query_start_date=report.inspected_start_date,
            query_end_doy=query_end_doy,
            phase_radius_days=phase_radius,
        )
        if len(cand_indices) < config.analog_top_k and config.expanded_phase_radius_days > phase_radius:
            expanded = self._eligible_end_indices(
                segment_len=segment_len,
                query_start_date=report.inspected_start_date,
                query_end_doy=query_end_doy,
                phase_radius_days=config.expanded_phase_radius_days,
            )
            if len(expanded) > len(cand_indices):
                cand_indices = expanded
                phase_radius = config.expanded_phase_radius_days
                phase_expanded = True

        if len(cand_indices) == 0:
            return None

        match_vars = [CANONICAL_TARGET_COL] + list(report.observed_matching_variables)
        col_indices = np.array([self.col_to_idx[c] for c in match_vars], dtype=np.int64)
        scales = np.array(
            [climatology.target_std] + [climatology.std[c] for c in report.observed_matching_variables],
            dtype=np.float64,
        )
        alphas = self._resolve_variable_weights(match_vars, config.variable_weights)

        query_match = df.tail(match_len).loc[:, match_vars].to_numpy(dtype=np.float64)
        obs_mask = np.isfinite(query_match)
        active_weight_per_day = np.sum(obs_mask * alphas[None, :], axis=1)
        if not np.any(active_weight_per_day > 0.0):
            return None

        # Normalize per observed cell mass so partially missing matching windows remain strictly weighted.
        cell_weight = np.where(obs_mask, (alphas / scales)[None, :], 0.0)
        safe_query = np.where(obs_mask, query_match, 0.0)
        denom_sum = max(float(np.sum(np.where(obs_mask, alphas[None, :], 0.0))), 1e-12)

        distances = np.empty(len(cand_indices), dtype=np.float64)
        for i, c_idx in enumerate(cand_indices):
            hist_slice = self.matrix[c_idx - match_len + 1 : c_idx + 1, :][:, col_indices]
            abs_diff = np.where(obs_mask, np.abs(safe_query - hist_slice), 0.0)
            distances[i] = float(np.sum(abs_diff * cell_weight) / denom_sum)

        k_eff = min(config.analog_top_k, len(cand_indices))
        rank_order = np.argsort(distances, kind="mergesort")[:k_eff]
        top_cand = cand_indices[rank_order]
        top_dists = distances[rank_order]

        gamma = max(float(np.median(top_dists)), 1e-6)
        raw_w = np.exp(np.maximum(-(top_dists - top_dists[0]) / gamma, -700.0))
        weights = raw_w / np.sum(raw_w)

        return AnalogSearchResult(
            candidate_end_indices=top_cand,
            candidate_end_dates=[self.dates[idx].strftime("%Y-%m-%d") for idx in top_cand],
            distances=top_dists,
            weights=weights,
            phase_radius_used=phase_radius,
            phase_expanded=phase_expanded,
        )


class ExogenousImputerWrapper:
    """Standalone inference-time exogenous imputer supporting Modes 0, 1, 2, and 3."""

    def __init__(
        self,
        *,
        model: Any,
        feature_builder: Any,
        climatology: ClimatologyLookupTable,
        analog_store: Optional[HistoricalAnalogStore] = None,
        config: Optional[ExogenousImputationConfig] = None,
    ) -> None:
        self.model = model
        self.feature_builder = feature_builder
        self.climatology = climatology
        self.analog_store = analog_store
        self.config = config or ExogenousImputationConfig()

    @staticmethod
    def _build_prediction_response(
        *,
        predictions_7d_mm: List[float],
        weighted_mean_7d_mm: List[float],
        weighted_median_7d_mm: List[float],
        imputation_uncertainty_7d_mm: List[float],
        mode: ImputationMode,
        report: GapInspectionReport,
        analog_mean_distance: Optional[float],
        effective_k: int,
        low_confidence_warning: bool,
        selected_analog_dates: List[str],
        phase_radius_used: Optional[int],
    ) -> Dict[str, Any]:
        return {
            "predictions_7d_mm": predictions_7d_mm,
            "weighted_mean_7d_mm": weighted_mean_7d_mm,
            "weighted_median_7d_mm": weighted_median_7d_mm,
            "imputation_uncertainty_7d_mm": imputation_uncertainty_7d_mm,
            "imputation_mode": mode,
            "max_gap_length": report.max_gap_length,
            "missing_variables": list(report.missing_variables),
            "analog_mean_distance": analog_mean_distance,
            "effective_k": effective_k,
            "low_confidence_warning": low_confidence_warning,
            "selected_analog_dates": selected_analog_dates,
            "phase_radius_used": phase_radius_used,
            "data_as_of": report.inspected_end_date.strftime("%Y-%m-%d"),
        }

    def _fill_climatology_fallback(self, df: pd.DataFrame, report: GapInspectionReport) -> pd.DataFrame:
        out = df.copy()
        dates = out[CANONICAL_DATE_COL]
        imputed_mask: Dict[str, np.ndarray] = {}
        for col in report.missing_variables:
            vals = out[col].to_numpy(dtype=np.float64, copy=True)
            clim_vec = self.climatology.climatology_series(col, dates)
            miss_mask = ~np.isfinite(vals)
            lo, hi = self.climatology.bounds[col]
            vals[miss_mask] = np.clip(clim_vec[miss_mask], lo, hi)
            out[col] = vals
            imputed_mask[col] = miss_mask
        return _enforce_dew_point_coupling(out, imputed_mask)

    def _predict_single_scenario(
        self,
        scenario_df: pd.DataFrame,
        *,
        mode: ImputationMode,
        report: GapInspectionReport,
        effective_k: int,
        low_confidence_warning: bool,
    ) -> Dict[str, Any]:
        x_batch = _extract_feature_matrix(self.feature_builder, [scenario_df])
        preds = _predict_batch_7d(self.model, x_batch)[0].tolist()
        return self._build_prediction_response(
            predictions_7d_mm=preds,
            weighted_mean_7d_mm=list(preds),
            weighted_median_7d_mm=list(preds),
            imputation_uncertainty_7d_mm=[0.0] * 7,
            mode=mode,
            report=report,
            analog_mean_distance=None,
            effective_k=effective_k,
            low_confidence_warning=low_confidence_warning,
            selected_analog_dates=[],
            phase_radius_used=None,
        )

    def _predict_mode3_fallback(self, report: GapInspectionReport) -> Dict[str, Any]:
        fallback_df = self._fill_climatology_fallback(report.df_aligned, report)
        return self._predict_single_scenario(
            fallback_df,
            mode=MODE_3_UNCONDITIONAL_SEASONAL_FALLBACK,
            report=report,
            effective_k=0,
            low_confidence_warning=True,
        )

    def predict(
        self,
        raw_history_df: pd.DataFrame,
        *,
        min_history_days: int = 35,
    ) -> Dict[str, Any]:
        """Run Mode 0/1/2/3 exogenous imputation followed by frozen FeatureBuilder + batch model predict."""
        report = GapInspector.inspect(raw_history_df, config=self.config, min_history_days=min_history_days)

        if report.mode == MODE_0_CLEAN:
            return self._predict_single_scenario(
                report.df_aligned,
                mode=MODE_0_CLEAN,
                report=report,
                effective_k=1,
                low_confidence_warning=False,
            )

        if report.mode == MODE_1_OU_CLIMATOLOGY:
            return self._predict_single_scenario(
                self._impute_mode1_ou(report),
                mode=MODE_1_OU_CLIMATOLOGY,
                report=report,
                effective_k=1,
                low_confidence_warning=False,
            )

        if report.mode == MODE_2_ANALOG_MARGINALIZATION and self.analog_store is not None:
            return self._predict_mode2_analog(report)

        return self._predict_mode3_fallback(report)

    def transform_mode1_ou(
        self,
        raw_history_df: pd.DataFrame,
        *,
        min_history_days: int = 35,
    ) -> Tuple[pd.DataFrame, GapInspectionReport]:
        """Inspect and apply Mode 1 Ornstein-Uhlenbeck anomaly bridge/decay imputation."""
        report = GapInspector.inspect(raw_history_df, config=self.config, min_history_days=min_history_days)
        if report.mode == MODE_0_CLEAN:
            return report.df_aligned.copy(), report
        if report.mode != MODE_1_OU_CLIMATOLOGY:
            raise ValueError(
                f"Expected MODE_1_OU_CLIMATOLOGY (1 <= max_gap <= {self.config.mode1_max_gap_days}); "
                f"got {report.mode} (max_gap={report.max_gap_length})."
            )
        return self._impute_mode1_ou(report), report

    def _impute_mode1_ou(self, report: GapInspectionReport) -> pd.DataFrame:
        out = report.df_aligned.copy()
        dates = out[CANONICAL_DATE_COL]
        imputed_mask: Dict[str, np.ndarray] = {}

        for col, runs in report.gap_runs_by_var.items():
            vals = out[col].to_numpy(dtype=np.float64, copy=True)
            clim_vec = self.climatology.climatology_series(col, dates)
            tau_v = self.climatology.tau[col]
            bounds = self.climatology.bounds[col]
            var_mask = ~np.isfinite(vals)
            for run in runs:
                _apply_mode1_ou_run(vals, clim_vec, run, tau_v=tau_v, bounds=bounds)
            out[col] = vals
            imputed_mask[col] = var_mask

        return _enforce_dew_point_coupling(out, imputed_mask)

    def _materialize_from_report(
        self,
        report: GapInspectionReport,
    ) -> Optional[Tuple[List[pd.DataFrame], np.ndarray, Dict[str, Any]]]:
        if self.analog_store is None:
            raise ValueError("analog_store is required for Mode 2 analog marginalization.")

        search_res = self.analog_store.search_analogs(report, climatology=self.climatology, config=self.config)
        if search_res is None:
            return None

        base_df = report.df_aligned
        n_win = len(base_df)
        apply_blend = report.max_gap_length <= self.config.boundary_blend_max_gap_days
        missing_masks = {
            col: ~np.isfinite(base_df[col].to_numpy(dtype=np.float64)) for col in report.missing_variables
        }

        scenarios: List[pd.DataFrame] = []
        for c_idx in search_res.candidate_end_indices:
            scen_df = base_df.copy()
            for col in report.missing_variables:
                cand_slice = self.analog_store.slice_variable(int(c_idx), n_win, col)
                vals = scen_df[col].to_numpy(dtype=np.float64, copy=True)
                miss_mask = missing_masks[col]
                bounds = self.climatology.bounds[col]
                lo, hi = bounds

                vals[miss_mask] = np.clip(cand_slice[miss_mask], lo, hi)
                if apply_blend:
                    tau_v = self.climatology.tau[col]
                    for run in report.gap_runs_by_var.get(col, ()):
                        if run.length <= self.config.boundary_blend_max_gap_days:
                            _apply_left_anchor_boundary_blend(
                                vals,
                                cand_slice,
                                run,
                                tau_v=tau_v,
                                bounds=bounds,
                            )

                scen_df[col] = vals

            scenarios.append(_enforce_dew_point_coupling(scen_df, missing_masks))

        meta = {
            "selected_analog_dates": list(search_res.candidate_end_dates),
            "analog_distances": search_res.distances.tolist(),
            "analog_mean_distance": float(np.sum(search_res.weights * search_res.distances)),
            "effective_k": len(scenarios),
            "phase_radius_used": search_res.phase_radius_used,
            "phase_expanded": search_res.phase_expanded,
        }
        return scenarios, search_res.weights, meta

    def materialize_mode2_scenarios(
        self,
        raw_history_df: pd.DataFrame,
        *,
        min_history_days: int = 35,
    ) -> Tuple[List[pd.DataFrame], np.ndarray, Dict[str, Any]]:
        """Public seam returning completed Mode 2 scenario DataFrames, kernel weights, and metadata."""
        report = GapInspector.inspect(raw_history_df, config=self.config, min_history_days=min_history_days)
        if report.mode != MODE_2_ANALOG_MARGINALIZATION:
            raise ValueError(
                f"Expected MODE_2_ANALOG_MARGINALIZATION; got {report.mode} (max_gap={report.max_gap_length})."
            )
        materialized = self._materialize_from_report(report)
        if materialized is None:
            raise ValueError("No valid historical analogs available inside Dev_Train for Mode 2 query.")
        return materialized

    def _predict_mode2_analog(self, report: GapInspectionReport) -> Dict[str, Any]:
        materialized = self._materialize_from_report(report)
        if materialized is None:
            return self._predict_mode3_fallback(report)

        scenarios, weights, meta = materialized
        x_batch = _extract_feature_matrix(self.feature_builder, scenarios)
        preds_k_by_7 = _predict_batch_7d(self.model, x_batch)
        chosen, w_mean, w_med, unc = _aggregate_scenario_predictions(
            preds_k_by_7,
            weights,
            self.config.aggregation_strategy,
        )
        low_conf = bool(meta["phase_expanded"] or meta["effective_k"] < self.config.analog_top_k)
        return self._build_prediction_response(
            predictions_7d_mm=chosen,
            weighted_mean_7d_mm=w_mean,
            weighted_median_7d_mm=w_med,
            imputation_uncertainty_7d_mm=unc,
            mode=MODE_2_ANALOG_MARGINALIZATION,
            report=report,
            analog_mean_distance=meta["analog_mean_distance"],
            effective_k=meta["effective_k"],
            low_confidence_warning=low_conf,
            selected_analog_dates=meta["selected_analog_dates"],
            phase_radius_used=meta["phase_radius_used"],
        )
