"""
Baseline & Ablation Imputation Suite for Standalone Exogenous Benchmark (DS108)
===============================================================================
Provides 4 tiers (11 identifiers) of causal, deterministic inference-time exogenous
imputers evaluated strictly under Dev_Train (<= 2018-04-30) fit/transform separation:

- Tier 1: `raw_nan`, `sklearn_mean`, `sklearn_median`
- Tier 2: `ffill`, `linear_interp` (interior-only)
- Tier 3: `doy_climatology`, `sklearn_iterative` (Profile A/B only), `sklearn_knn`
- Tier 4: `custom_mode1`, `custom_mode2`, `custom_hybrid`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data.loader import time_series_split
from src.data.vintage_store import CANONICAL_VI_COLUMN_ORDER, align_and_validate_canonical_raw_df
from research.imputer_benchmark.exogenous_imputer import (
    CANONICAL_EXOGENOUS_COLUMNS,
    ClimatologyLookupTable,
    ExogenousImputationConfig,
    ExogenousImputerWrapper,
    GapInspectionReport,
    GapInspector,
    HistoricalAnalogStore,
    _enforce_dew_point_coupling,
)
from src.inference.validation import (
    CANONICAL_DATE_COL,
    PHYSICAL_METEO_BOUNDS,
)

GapPosition = Literal["trailing", "interior"]
ProfileId = Literal["A", "B", "C"]

ALL_BENCHMARK_METHODS: Tuple[str, ...] = (
    "raw_nan",
    "sklearn_mean",
    "sklearn_median",
    "ffill",
    "linear_interp",
    "doy_climatology",
    "sklearn_iterative",
    "sklearn_knn",
    "custom_mode1",
    "custom_mode2",
    "custom_hybrid",
)

METHOD_TIERS: Dict[str, str] = {
    "raw_nan": "Tier 1 (Default Floor)",
    "sklearn_mean": "Tier 1 (Static Floor)",
    "sklearn_median": "Tier 1 (Static Floor)",
    "ffill": "Tier 2 (TS Heuristic)",
    "linear_interp": "Tier 2 (TS Heuristic)",
    "doy_climatology": "Tier 3 (Strong Baseline)",
    "sklearn_iterative": "Tier 3 (Strong Baseline)",
    "sklearn_knn": "Tier 3 (Strong Baseline)",
    "custom_mode1": "Tier 4 (Proposed / Ablation)",
    "custom_mode2": "Tier 4 (Proposed / Ablation)",
    "custom_hybrid": "Tier 4 (Proposed Hybrid)",
}

_DEW_POINT_COL = "Điểm sương 2m"
_TEMP_2M_COL = "Nhiệt độ 2m"


@dataclass
class ImputationExecutionResult:
    """Standardized output produced by any Tier 1-4 imputation method on one masked window."""

    method: str
    status: Literal["OK", "NOT_APPLICABLE"]
    reason: Optional[str]
    scenarios: List[pd.DataFrame]
    weights: np.ndarray
    effective_k: int
    imputation_mode: str
    selected_analog_dates: List[str]
    is_raw_nan: bool = False


def validate_dev_train_only(
    train_df: pd.DataFrame,
    *,
    dev_train_end_date: str = "2018-04-30",
    caller: str = "BaselineSuite",
) -> pd.DataFrame:
    """Align DataFrame and fail closed if any training observation post-dates `Dev_Train`."""
    aligned = align_and_validate_canonical_raw_df(train_df)
    aligned[CANONICAL_DATE_COL] = pd.to_datetime(aligned[CANONICAL_DATE_COL]).dt.normalize()
    aligned = aligned.sort_values(CANONICAL_DATE_COL).reset_index(drop=True)
    cutoff = pd.Timestamp(dev_train_end_date).normalize()
    next_split = (cutoff + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    _, post_train = time_series_split(aligned, date_col=CANONICAL_DATE_COL, test_start_date=next_split)
    if len(post_train) > 0:
        raise ValueError(
            f"[DATA_LEAKAGE_GUARD] {caller} received training dates through "
            f"{pd.Timestamp(post_train[CANONICAL_DATE_COL].max()).strftime('%Y-%m-%d')} "
            f"(cutoff={cutoff.strftime('%Y-%m-%d')})."
        )
    return aligned


def compute_domain_violation_rate(
    df_list: Sequence[pd.DataFrame],
    weights: Optional[np.ndarray],
    masked_mask: np.ndarray,
    masked_cols: Sequence[str],
) -> float:
    """Compute weighted % of masked cells violating `PHYSICAL_METEO_BOUNDS` or `T2MDEW > T2M`."""
    if len(df_list) == 0 or len(masked_cols) == 0 or not np.any(masked_mask):
        return 0.0
    w = (
        np.ones(len(df_list), dtype=np.float64) / float(len(df_list))
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    w = w / np.sum(w)

    rates: List[float] = []
    check_dew_pair = _DEW_POINT_COL in masked_cols or _TEMP_2M_COL in masked_cols
    col_to_m_idx = {c: i for i, c in enumerate(masked_cols)}

    for df in df_list:
        total_cells = 0
        bad_cells = 0
        for col in masked_cols:
            m_idx = col_to_m_idx[col]
            col_mask = masked_mask[:, m_idx]
            n_col = int(np.sum(col_mask))
            if n_col == 0:
                continue
            vals = df[col].to_numpy(dtype=np.float64)[col_mask]
            lo, hi = PHYSICAL_METEO_BOUNDS[col]
            total_cells += n_col
            bad_col = (~np.isfinite(vals)) | (vals < lo - 1e-9) | (vals > hi + 1e-9)
            if check_dew_pair and col == _DEW_POINT_COL:
                t2m_vals = df[_TEMP_2M_COL].to_numpy(dtype=np.float64)[col_mask]
                bad_col = bad_col | (vals > t2m_vals + 1e-9)
            bad_cells += int(np.sum(bad_col))
        rates.append((100.0 * bad_cells / total_cells) if total_cells > 0 else 0.0)

    return float(np.sum(w * np.asarray(rates, dtype=np.float64)))


def count_nan_or_inf_in_matrix(x: np.ndarray) -> int:
    """Count non-finite entries in a NumPy feature matrix."""
    arr = np.asarray(x, dtype=np.float64)
    return int(np.sum(~np.isfinite(arr)))


class ImputationBaselineRegistry:
    """Stateful container holding all Dev_Train-fitted Tier 1-4 baselines & custom imputers."""

    def __init__(
        self,
        *,
        train_df: pd.DataFrame,
        train_std_by_col: Dict[str, float],
        exo_cols: Tuple[str, ...],
        mean_imputer: SimpleImputer,
        median_imputer: SimpleImputer,
        iterative_imputer: IterativeImputer,
        knn_scaler: StandardScaler,
        knn_imputer: KNNImputer,
        climatology: ClimatologyLookupTable,
        analog_store: HistoricalAnalogStore,
        wrapper_mode1: ExogenousImputerWrapper,
        wrapper_mode2: ExogenousImputerWrapper,
        wrapper_hybrid: ExogenousImputerWrapper,
        random_state: int = 42,
    ) -> None:
        self.train_start_date = pd.Timestamp(train_df[CANONICAL_DATE_COL].min()).normalize()
        self.train_end_date = pd.Timestamp(train_df[CANONICAL_DATE_COL].max()).normalize()
        self.n_train_rows = len(train_df)
        self.train_std_by_col = train_std_by_col
        self.exo_cols = exo_cols
        self.mean_imputer = mean_imputer
        self.median_imputer = median_imputer
        self.iterative_imputer = iterative_imputer
        self.knn_scaler = knn_scaler
        self.knn_imputer = knn_imputer
        self.climatology = climatology
        self.analog_store = analog_store
        self.wrapper_mode1 = wrapper_mode1
        self.wrapper_mode2 = wrapper_mode2
        self.wrapper_hybrid = wrapper_hybrid
        self.random_state = random_state

    @classmethod
    def fit_on_dev_train(
        cls,
        train_df: pd.DataFrame,
        *,
        model: Any,
        feature_builder: Any,
        hybrid_threshold_days: int = 7,
        random_state: int = 42,
        dev_train_end_date: str = "2018-04-30",
    ) -> "ImputationBaselineRegistry":
        """Fit all Tier 1-4 baselines and custom imputer state strictly on `Dev_Train`."""
        clean_train = validate_dev_train_only(
            train_df,
            dev_train_end_date=dev_train_end_date,
            caller="ImputationBaselineRegistry.fit_on_dev_train",
        )
        exo_cols = CANONICAL_EXOGENOUS_COLUMNS
        x_train = clean_train.loc[:, list(exo_cols)].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(x_train)):
            raise ValueError("Dev_Train contains non-finite exogenous values; expected clean 0% missing Dev_Train.")

        train_std_by_col = {
            col: max(float(np.std(x_train[:, i], ddof=0)), 1e-4)
            for i, col in enumerate(exo_cols)
        }

        mean_imp = SimpleImputer(strategy="mean").fit(x_train)
        median_imp = SimpleImputer(strategy="median").fit(x_train)
        iter_imp = IterativeImputer(max_iter=10, random_state=random_state).fit(x_train)
        knn_scaler = StandardScaler().fit(x_train)
        knn_imp = KNNImputer(n_neighbors=5).fit(knn_scaler.transform(x_train))

        cfg_default = ExogenousImputationConfig(
            mode1_max_gap_days=hybrid_threshold_days,
            mode2_max_gap_days=90,
            boundary_blend_max_gap_days=14,
            dev_train_end_date=dev_train_end_date,
        )
        cfg_mode1_only = ExogenousImputationConfig(
            mode1_max_gap_days=90,
            mode2_max_gap_days=91,
            boundary_blend_max_gap_days=90,
            dev_train_end_date=dev_train_end_date,
        )
        cfg_mode2_only = ExogenousImputationConfig(
            mode1_max_gap_days=1,
            mode2_max_gap_days=90,
            boundary_blend_max_gap_days=14,
            dev_train_end_date=dev_train_end_date,
        )

        clim = ClimatologyLookupTable.fit(clean_train, config=cfg_default)
        store = HistoricalAnalogStore.from_dataframe(clean_train, config=cfg_default)

        return cls(
            train_df=clean_train,
            train_std_by_col=train_std_by_col,
            exo_cols=exo_cols,
            mean_imputer=mean_imp,
            median_imputer=median_imp,
            iterative_imputer=iter_imp,
            knn_scaler=knn_scaler,
            knn_imputer=knn_imp,
            climatology=clim,
            analog_store=store,
            wrapper_mode1=ExogenousImputerWrapper(
                model=model,
                feature_builder=feature_builder,
                climatology=clim,
                analog_store=store,
                config=cfg_mode1_only,
            ),
            wrapper_mode2=ExogenousImputerWrapper(
                model=model,
                feature_builder=feature_builder,
                climatology=clim,
                analog_store=store,
                config=cfg_mode2_only,
            ),
            wrapper_hybrid=ExogenousImputerWrapper(
                model=model,
                feature_builder=feature_builder,
                climatology=clim,
                analog_store=store,
                config=cfg_default,
            ),
            random_state=random_state,
        )

    def _replace_exogenous_matrix(
        self,
        base_df: pd.DataFrame,
        x_imputed: np.ndarray,
    ) -> pd.DataFrame:
        """Replace missing cells with `x_imputed` matrix while preserving observed cells 100%."""
        out = base_df.copy()
        for j, col in enumerate(self.exo_cols):
            orig = out[col].to_numpy(dtype=np.float64, copy=True)
            miss = ~np.isfinite(orig)
            if np.any(miss):
                orig[miss] = x_imputed[miss, j]
                out[col] = orig
        return out

    def _run_doy_climatology(self, base_df: pd.DataFrame) -> pd.DataFrame:
        out = base_df.copy()
        dates = out[CANONICAL_DATE_COL]
        for col in self.exo_cols:
            orig = out[col].to_numpy(dtype=np.float64, copy=True)
            miss = ~np.isfinite(orig)
            if np.any(miss):
                orig[miss] = self.climatology.climatology_series(col, dates)[miss]
                out[col] = orig
        return out

    def _run_ffill(self, base_df: pd.DataFrame) -> pd.DataFrame:
        out = base_df.copy()
        for col in self.exo_cols:
            if out[col].isna().any():
                out[col] = out[col].ffill()
        return out

    def _run_linear_interp(self, base_df: pd.DataFrame) -> pd.DataFrame:
        out = base_df.copy()
        for col in self.exo_cols:
            if out[col].isna().any():
                out[col] = out[col].interpolate(method="linear", limit_area="inside")
        return out

    def _run_custom_wrapper(
        self,
        wrapper: ExogenousImputerWrapper,
        method: str,
        masked_df: pd.DataFrame,
        gap_length: int,
    ) -> ImputationExecutionResult:
        if gap_length == 0:
            aligned = align_and_validate_canonical_raw_df(masked_df).copy()
            return ImputationExecutionResult(
                method=method,
                status="OK",
                reason=None,
                scenarios=[aligned],
                weights=np.array([1.0], dtype=np.float64),
                effective_k=1,
                imputation_mode="MODE_0_CLEAN",
                selected_analog_dates=[],
            )

        report: GapInspectionReport = GapInspector.inspect(masked_df, config=wrapper.config)
        if report.mode == "MODE_1_OU_CLIMATOLOGY":
            filled_df, _ = wrapper.transform_mode1_ou(masked_df)
            return ImputationExecutionResult(
                method=method,
                status="OK",
                reason=None,
                scenarios=[filled_df],
                weights=np.array([1.0], dtype=np.float64),
                effective_k=1,
                imputation_mode=report.mode,
                selected_analog_dates=[],
            )
        if report.mode == "MODE_2_ANALOG_MARGINALIZATION":
            scenarios, weights, meta = wrapper.materialize_mode2_scenarios(masked_df)
            return ImputationExecutionResult(
                method=method,
                status="OK",
                reason=None,
                scenarios=scenarios,
                weights=weights,
                effective_k=int(meta["effective_k"]),
                imputation_mode=report.mode,
                selected_analog_dates=list(meta["selected_analog_dates"]),
            )
        # Mode 3 unconditional seasonal fallback
        fallback_df = self._run_doy_climatology(report.df_aligned)
        missing_masks = {
            c: ~np.isfinite(report.df_aligned[c].to_numpy(dtype=np.float64))
            for c in report.missing_variables
        }
        for col, m_mask in missing_masks.items():
            lo, hi = PHYSICAL_METEO_BOUNDS[col]
            vals = fallback_df[col].to_numpy(dtype=np.float64, copy=True)
            vals[m_mask] = np.clip(vals[m_mask], lo, hi)
            fallback_df[col] = vals
        fallback_df = _enforce_dew_point_coupling(fallback_df, missing_masks)
        return ImputationExecutionResult(
            method=method,
            status="OK",
            reason=None,
            scenarios=[fallback_df],
            weights=np.array([1.0], dtype=np.float64),
            effective_k=0,
            imputation_mode="MODE_3_UNCONDITIONAL_SEASONAL_FALLBACK",
            selected_analog_dates=[],
        )

    def impute_window(
        self,
        method: str,
        masked_df: pd.DataFrame,
        *,
        gap_position: GapPosition,
        gap_length: int,
        profile_id: ProfileId,
    ) -> ImputationExecutionResult:
        """Apply one Tier 1-4 imputation method on a single masked lookback window."""
        if method == "linear_interp" and gap_position == "trailing" and gap_length > 0:
            return ImputationExecutionResult(
                method=method,
                status="NOT_APPLICABLE",
                reason="linear_interp requires both boundary endpoints and is invalid for causal trailing gaps.",
                scenarios=[],
                weights=np.empty(0, dtype=np.float64),
                effective_k=0,
                imputation_mode="NOT_APPLICABLE",
                selected_analog_dates=[],
            )
        if method == "sklearn_iterative" and profile_id == "C" and gap_length > 0:
            return ImputationExecutionResult(
                method=method,
                status="NOT_APPLICABLE",
                reason="IterativeImputer cannot regress across exogenous columns when all 34 exogenous variables are masked (Profile C).",
                scenarios=[],
                weights=np.empty(0, dtype=np.float64),
                effective_k=0,
                imputation_mode="NOT_APPLICABLE",
                selected_analog_dates=[],
            )

        canonical_masked = align_and_validate_canonical_raw_df(masked_df).copy()
        canonical_masked[CANONICAL_DATE_COL] = pd.to_datetime(canonical_masked[CANONICAL_DATE_COL]).dt.normalize()
        canonical_masked = canonical_masked.sort_values(CANONICAL_DATE_COL).reset_index(drop=True)

        if method == "raw_nan":
            return ImputationExecutionResult(
                method=method,
                status="OK",
                reason=None,
                scenarios=[canonical_masked],
                weights=np.array([1.0], dtype=np.float64),
                effective_k=1,
                imputation_mode="RAW_NAN_PASSTHROUGH",
                selected_analog_dates=[],
                is_raw_nan=True,
            )

        # Strict L=0 clean identity short-circuit for every imputation method
        if gap_length == 0:
            return ImputationExecutionResult(
                method=method,
                status="OK",
                reason=None,
                scenarios=[canonical_masked],
                weights=np.array([1.0], dtype=np.float64),
                effective_k=1,
                imputation_mode="CLEAN_L0",
                selected_analog_dates=[],
            )

        if method == "sklearn_mean":
            x_raw = canonical_masked.loc[:, list(self.exo_cols)].to_numpy(dtype=np.float64)
            filled_df = self._replace_exogenous_matrix(canonical_masked, self.mean_imputer.transform(x_raw))
        elif method == "sklearn_median":
            x_raw = canonical_masked.loc[:, list(self.exo_cols)].to_numpy(dtype=np.float64)
            filled_df = self._replace_exogenous_matrix(canonical_masked, self.median_imputer.transform(x_raw))
        elif method == "ffill":
            filled_df = self._run_ffill(canonical_masked)
        elif method == "linear_interp":
            filled_df = self._run_linear_interp(canonical_masked)
        elif method == "doy_climatology":
            filled_df = self._run_doy_climatology(canonical_masked)
        elif method == "sklearn_iterative":
            x_raw = canonical_masked.loc[:, list(self.exo_cols)].to_numpy(dtype=np.float64)
            filled_df = self._replace_exogenous_matrix(canonical_masked, self.iterative_imputer.transform(x_raw))
        elif method == "sklearn_knn":
            x_raw = canonical_masked.loc[:, list(self.exo_cols)].to_numpy(dtype=np.float64)
            x_scaled = self.knn_scaler.transform(x_raw)
            x_imp_scaled = self.knn_imputer.transform(x_scaled)
            x_imp = self.knn_scaler.inverse_transform(x_imp_scaled)
            filled_df = self._replace_exogenous_matrix(canonical_masked, x_imp)
        elif method == "custom_mode1":
            return self._run_custom_wrapper(self.wrapper_mode1, method, canonical_masked, gap_length)
        elif method == "custom_mode2":
            return self._run_custom_wrapper(self.wrapper_mode2, method, canonical_masked, gap_length)
        elif method == "custom_hybrid":
            return self._run_custom_wrapper(self.wrapper_hybrid, method, canonical_masked, gap_length)
        else:
            raise ValueError(f"Unknown benchmark method: {method!r}")

        return ImputationExecutionResult(
            method=method,
            status="OK",
            reason=None,
            scenarios=[filled_df.loc[:, list(CANONICAL_VI_COLUMN_ORDER)]],
            weights=np.array([1.0], dtype=np.float64),
            effective_k=1,
            imputation_mode="BASELINE_DETERMINISTIC",
            selected_analog_dates=[],
        )


def combine_hybrid_from_mode1_mode2(
    *,
    mode1_res: ImputationExecutionResult,
    mode2_res: ImputationExecutionResult,
    gap_length: int,
    threshold_l_star: int,
) -> ImputationExecutionResult:
    """Deterministic hybrid router selector reusing precomputed Mode 1 and Mode 2 results for threshold sweeps."""
    chosen = mode1_res if gap_length <= threshold_l_star else mode2_res
    return ImputationExecutionResult(
        method="custom_hybrid",
        status=chosen.status,
        reason=chosen.reason,
        scenarios=chosen.scenarios,
        weights=chosen.weights,
        effective_k=chosen.effective_k,
        imputation_mode=chosen.imputation_mode,
        selected_analog_dates=chosen.selected_analog_dates,
    )
