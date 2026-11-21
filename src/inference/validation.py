"""
Serving Input Validation Gate (DS108)
=====================================
Enforces operational data contracts before FeatureBuilder or model inference runs.
Prevents silent imputation of missing sensors, NASA POWER sentinel (-999) leaks,
stale CERES radiation feeds, non-continuous daily histories, and >7-day stale runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.clock import today_hcmc
from src.data.dataquality import check_continuity

CANONICAL_DATE_COL = "Ngày"
CANONICAL_TARGET_COL = "Lượng mưa"
MIN_HISTORY_ROWS = 30
MAX_OPERATIONAL_LAG_DAYS = 7
RADIATION_STALENESS_WINDOW = 5
SENTINEL_THRESHOLD = -900.0

RADIATION_STALENESS_COLUMNS: Tuple[str, ...] = (
    "Bức xạ sóng ngắn bề mặt",
    "Bức xạ sóng dài xuống",
)

# Physical bounds calibrated conservatively against Ho Chi Minh City (10.8N, 106.7E)
# historical ranges (2001-2026) to reject sensor dropouts, units shifts, and sentinels.
PHYSICAL_METEO_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Lượng mưa": (0.0, 500.0),
    "Bức xạ sóng ngắn bề mặt": (0.0, 15.0),
    "Bức xạ sóng ngắn trời quang": (0.0, 15.0),
    "Bức xạ trực tiếp pháp tuyến": (0.0, 20.0),
    "Bức xạ khuếch tán": (0.0, 15.0),
    "Bức xạ đỉnh khí quyển": (0.0, 20.0),
    "Bức xạ quang hợp tổng": (0.0, 10.0),
    "Bức xạ quang hợp trời quang": (0.0, 10.0),
    "Bức xạ UVA": (0.0, 5.0),
    "Bức xạ UVB": (0.0, 1.0),
    "Chỉ số UV": (0.0, 25.0),
    "Bức xạ sóng dài xuống": (1.0, 20.0),
    "Nhiệt độ 2m": (5.0, 50.0),
    "Điểm sương 2m": (-10.0, 45.0),
    "Nhiệt độ bầu ướt 2m": (0.0, 45.0),
    "Nhiệt độ bề mặt đất": (5.0, 65.0),
    "Biên độ nhiệt 2m": (0.0, 35.0),
    "Nhiệt độ tối đa 2m": (5.0, 55.0),
    "Nhiệt độ tối thiểu 2m": (0.0, 45.0),
    "Độ ẩm tuyệt đối 2m": (0.0, 60.0),
    "Độ ẩm tương đối 2m": (0.0, 100.0),
    "Áp suất bề mặt": (80.0, 110.0),
    "Tốc độ gió 2m": (0.0, 60.0),
    "Tốc độ gió tối đa 2m": (0.0, 80.0),
    "Tốc độ gió tối thiểu 2m": (0.0, 50.0),
    "Biên độ gió 2m": (0.0, 60.0),
    "Hướng gió 2m": (0.0, 360.0),
    "Tốc độ gió 10m": (0.0, 80.0),
    "Tốc độ gió tối đa 10m": (0.0, 100.0),
    "Tốc độ gió tối thiểu 10m": (0.0, 60.0),
    "Biên độ gió 10m": (0.0, 80.0),
    "Hướng gió 10m": (0.0, 360.0),
    "Độ ẩm đất bề mặt": (0.0, 1.0),
    "Độ ẩm đất vùng rễ": (0.0, 1.0),
    "Độ ẩm đất mặt cắt": (0.0, 1.0),
}

CANONICAL_NUMERIC_COLUMNS: Tuple[str, ...] = tuple(PHYSICAL_METEO_BOUNDS.keys())


class ServingValidationError(ValueError):
    """Structured ValueError carrying a canonical serving error code."""

    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[{code}] {message}")


@dataclass(frozen=True)
class ValidatedServingContext:
    """Normalized DataFrame and metadata produced by validate_serving_input."""

    df: pd.DataFrame
    data_as_of: pd.Timestamp
    effective_run_date: pd.Timestamp
    lag_days: int
    effective_future_days: int
    imputation_outcome: Optional[Any] = None


def resolve_required_numeric_columns(
    feature_builder: Optional[Any] = None,
    target_col: str = CANONICAL_TARGET_COL,
) -> List[str]:
    """Resolve required meteorological numeric columns from FeatureBuilder or canonical defaults."""
    if feature_builder is None:
        return list(CANONICAL_NUMERIC_COLUMNS)

    cols: List[str] = []
    imputation_stats = getattr(feature_builder, "_imputation_stats", None) or {}
    fill_values = imputation_stats.get("fill_values")
    if isinstance(fill_values, dict):
        cols.extend(fill_values.keys())

    config = getattr(feature_builder, "config", None)
    circular_cols = getattr(config, "circular_wind_columns", ()) if config is not None else ()
    for col in circular_cols:
        if col not in cols:
            cols.append(col)

    if target_col not in cols:
        cols.append(target_col)

    return cols or list(CANONICAL_NUMERIC_COLUMNS)


def _validate_numeric_column(series: pd.Series, col_name: str) -> None:
    """Validate finiteness, NASA POWER sentinel (-999), and physical bounds."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        bad_count = int(numeric.isna().sum())
        raise ServingValidationError(
            "INVALID_OR_MISSING_METEO_INPUT",
            f"Column '{col_name}' contains {bad_count} missing or non-numeric value(s).",
            {"column": col_name, "invalid_count": bad_count},
        )

    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ServingValidationError(
            "INVALID_OR_MISSING_METEO_INPUT",
            f"Column '{col_name}' contains non-finite (Inf/-Inf) values.",
            {"column": col_name},
        )

    if (values <= SENTINEL_THRESHOLD).any():
        bad_val = float(values[values <= SENTINEL_THRESHOLD][0])
        raise ServingValidationError(
            "INVALID_OR_MISSING_METEO_INPUT",
            f"Column '{col_name}' contains sentinel fill value ({bad_val}).",
            {"column": col_name, "value": bad_val},
        )

    bounds = PHYSICAL_METEO_BOUNDS.get(col_name)
    if bounds is not None:
        low, high = bounds
        out_of_bounds = (values < low) | (values > high)
        if out_of_bounds.any():
            bad_val = float(values[out_of_bounds][0])
            raise ServingValidationError(
                "INVALID_OR_MISSING_METEO_INPUT",
                f"Column '{col_name}' value {bad_val} violates physical bounds [{low}, {high}].",
                {"column": col_name, "value": bad_val, "bounds": [low, high]},
            )


def _validate_radiation_freshness(df: pd.DataFrame) -> None:
    """Reject frozen radiation sensors where trailing 5 non-null days share a single constant value."""
    tail_df = df.tail(RADIATION_STALENESS_WINDOW)
    if len(tail_df) < RADIATION_STALENESS_WINDOW:
        return

    for col in RADIATION_STALENESS_COLUMNS:
        if col not in tail_df.columns:
            continue
        series = pd.to_numeric(tail_df[col], errors="coerce")
        if int(series.notna().sum()) >= RADIATION_STALENESS_WINDOW and series.nunique(dropna=True) <= 1:
            const_val = float(series.iloc[-1])
            raise ServingValidationError(
                "STALE_RADIATION_OR_LAG",
                (
                    f"Radiation column '{col}' is frozen/constant across the last "
                    f"{RADIATION_STALENESS_WINDOW} consecutive days (value={const_val})."
                ),
                {"column": col, "window": RADIATION_STALENESS_WINDOW, "value": const_val},
            )


def _validate_solar_pre_imputation(series: pd.Series, col_name: str) -> bool:
    """Validate non-null solar cells (no non-numeric strings, Inf, -999 sentinel, or out-of-bounds); return True if any NaN exists."""
    raw_notna = series.notna()
    numeric = pd.to_numeric(series, errors="coerce")
    coerced_bad = raw_notna & numeric.isna()
    if coerced_bad.any():
        bad_count = int(coerced_bad.sum())
        raise ServingValidationError(
            "INVALID_OR_MISSING_METEO_INPUT",
            f"Column '{col_name}' contains {bad_count} non-numeric value(s).",
            {"column": col_name, "invalid_count": bad_count},
        )

    non_null_vals = numeric.dropna().to_numpy(dtype=np.float64)
    if non_null_vals.size > 0:
        if not np.isfinite(non_null_vals).all():
            raise ServingValidationError(
                "INVALID_OR_MISSING_METEO_INPUT",
                f"Column '{col_name}' contains non-finite (Inf/-Inf) values.",
                {"column": col_name},
            )
        if (non_null_vals <= SENTINEL_THRESHOLD).any():
            bad_val = float(non_null_vals[non_null_vals <= SENTINEL_THRESHOLD][0])
            raise ServingValidationError(
                "INVALID_OR_MISSING_METEO_INPUT",
                f"Column '{col_name}' contains sentinel fill value ({bad_val}).",
                {"column": col_name, "value": bad_val},
            )
        bounds = PHYSICAL_METEO_BOUNDS.get(col_name)
        if bounds is not None:
            low, high = bounds
            out_of_bounds = (non_null_vals < low) | (non_null_vals > high)
            if out_of_bounds.any():
                bad_val = float(non_null_vals[out_of_bounds][0])
                raise ServingValidationError(
                    "INVALID_OR_MISSING_METEO_INPUT",
                    f"Column '{col_name}' value {bad_val} violates physical bounds [{low}, {high}].",
                    {"column": col_name, "value": bad_val, "bounds": [low, high]},
                )

    return bool(numeric.isna().any())


def validate_serving_input(
    raw_history_df: pd.DataFrame,
    run_date: Optional[str] = None,
    *,
    date_col: str = CANONICAL_DATE_COL,
    target_col: str = CANONICAL_TARGET_COL,
    feature_builder: Optional[Any] = None,
    required_columns: Optional[Iterable[str]] = None,
    min_rows: int = MIN_HISTORY_ROWS,
    max_lag_days: int = MAX_OPERATIONAL_LAG_DAYS,
    enforce_lag_limit: bool = True,
    allow_solar_doy_imputation: bool = True,
) -> ValidatedServingContext:
    """
    Validate raw meteorological history before feature construction or inference.

    Raises ServingValidationError (a ValueError subclass) with one of:
    - INSUFFICIENT_HISTORY
    - DATE_CONTINUITY_VIOLATION
    - INVALID_OR_MISSING_METEO_INPUT
    - STALE_RADIATION_OR_LAG
    """
    if raw_history_df is None or not isinstance(raw_history_df, pd.DataFrame):
        raise ServingValidationError(
            "INSUFFICIENT_HISTORY",
            "History payload must be a non-empty tabular DataFrame.",
        )

    if len(raw_history_df) < min_rows:
        raise ServingValidationError(
            "INSUFFICIENT_HISTORY",
            (
                f"History DataFrame must contain at least {min_rows} continuous days "
                f"(got {len(raw_history_df)} rows). Warmup lookback requires {min_rows - 1} days + 1 origin day."
            ),
            {"row_count": len(raw_history_df), "min_rows": min_rows},
        )

    df = raw_history_df.copy()
    if required_columns is None:
        from src.data.vintage_store import align_and_validate_canonical_raw_df

        try:
            df = align_and_validate_canonical_raw_df(df)
        except ValueError as exc:
            raise ServingValidationError(
                "INVALID_OR_MISSING_METEO_INPUT",
                str(exc),
                {"reason": "canonical_column_alignment_failed"},
            ) from exc

    if date_col not in df.columns:
        raise ServingValidationError(
            "INVALID_OR_MISSING_METEO_INPUT",
            f"Missing required date column '{date_col}'.",
            {"missing_columns": [date_col]},
        )

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    if parsed_dates.isna().any():
        bad_count = int(parsed_dates.isna().sum())
        raise ServingValidationError(
            "DATE_CONTINUITY_VIOLATION",
            f"Date column '{date_col}' contains {bad_count} unparseable/NaT entry(ies).",
            {"invalid_date_count": bad_count},
        )
    df[date_col] = parsed_dates.dt.normalize()

    if df[date_col].duplicated().any():
        dup_dates = [d.strftime("%Y-%m-%d") for d in df.loc[df[date_col].duplicated(), date_col].unique()[:5]]
        raise ServingValidationError(
            "DATE_CONTINUITY_VIOLATION",
            f"Duplicate date(s) detected in history sequence: {dup_dates}.",
            {"duplicate_dates": dup_dates},
        )

    df = df.sort_values(date_col).reset_index(drop=True)
    try:
        check_continuity(df, date_col=date_col, freq="D", raise_on_gap=True)
    except ValueError as exc:
        raise ServingValidationError(
            "DATE_CONTINUITY_VIOLATION",
            f"Daily history contains calendar gaps: {exc}",
        ) from exc

    req_cols = list(required_columns) if required_columns is not None else resolve_required_numeric_columns(
        feature_builder=feature_builder,
        target_col=target_col,
    )
    missing_cols = [c for c in req_cols if c not in df.columns]
    if missing_cols:
        raise ServingValidationError(
            "INVALID_OR_MISSING_METEO_INPUT",
            f"Missing required meteorological column(s): {missing_cols}",
            {"missing_columns": missing_cols},
        )

    imputation_outcome = None
    if allow_solar_doy_imputation and required_columns is None:
        from src.inference.solar_imputation import (
            SOLAR_11_VI_COLS,
            apply_serving_solar_doy_fallback,
        )

        solar_set = set(SOLAR_11_VI_COLS)
        # 1. Fail-fast on all 24 meteorological variables (never imputed)
        for col in req_cols:
            if col not in solar_set:
                _validate_numeric_column(df[col], col)

        # 2. Check non-null solar cells for sentinels/bounds and detect missing solar cells
        has_missing_solar = False
        for col in req_cols:
            if col in solar_set and _validate_solar_pre_imputation(df[col], col):
                has_missing_solar = True

        # 3. Reject frozen non-null constant radiation sensors BEFORE filling missing solar cells
        _validate_radiation_freshness(df)

        # 4. Selectively apply P-c (Dev_Train 366-day DOY median) ONLY to missing solar cells
        if has_missing_solar:
            imputation_outcome = apply_serving_solar_doy_fallback(df, date_col=date_col)
            df = imputation_outcome.filled_df

    for col in req_cols:
        _validate_numeric_column(df[col], col)

    _validate_radiation_freshness(df)

    data_as_of = pd.Timestamp(df[date_col].max()).normalize()
    if run_date is not None:
        effective_run_date = pd.to_datetime(run_date, errors="coerce")
        if pd.isna(effective_run_date):
            raise ServingValidationError(
                "INVALID_OR_MISSING_METEO_INPUT",
                f"Invalid run_date format '{run_date}'; expected YYYY-MM-DD.",
                {"run_date": run_date},
            )
        effective_run_date = pd.Timestamp(effective_run_date).normalize()
    else:
        effective_run_date = pd.Timestamp(today_hcmc()).normalize() if enforce_lag_limit else data_as_of

    lag_days = int((effective_run_date - data_as_of).days)
    if lag_days < 0:
        raise ServingValidationError(
            "DATE_CONTINUITY_VIOLATION",
            (
                f"run_date ({effective_run_date.strftime('%Y-%m-%d')}) cannot precede "
                f"latest history date data_as_of ({data_as_of.strftime('%Y-%m-%d')})."
            ),
            {
                "run_date": effective_run_date.strftime("%Y-%m-%d"),
                "data_as_of": data_as_of.strftime("%Y-%m-%d"),
                "lag_days": lag_days,
            },
        )

    if enforce_lag_limit and lag_days > max_lag_days:
        raise ServingValidationError(
            "STALE_RADIATION_OR_LAG",
            (
                f"History data is stale by {lag_days} days (max allowed operational lag is "
                f"{max_lag_days} days; data_as_of={data_as_of.strftime('%Y-%m-%d')}, "
                f"run_date={effective_run_date.strftime('%Y-%m-%d')})."
            ),
            {
                "lag_days": lag_days,
                "max_lag_days": max_lag_days,
                "data_as_of": data_as_of.strftime("%Y-%m-%d"),
                "run_date": effective_run_date.strftime("%Y-%m-%d"),
            },
        )

    return ValidatedServingContext(
        df=df,
        data_as_of=data_as_of,
        effective_run_date=effective_run_date,
        lag_days=lag_days,
        effective_future_days=max(0, 7 - lag_days),
        imputation_outcome=imputation_outcome,
    )

