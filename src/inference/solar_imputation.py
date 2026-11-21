"""
Serving-Time Solar Radiation DOY Fallback Imputation (P-c, DS108 Round 13)
==========================================================================
Pure-runtime fallback implementing policy ``P-c_DOY_MEDIAN_11_SOLAR_V1``:
- Loads Dev_Train-only (2001-01-01..2018-04-30) circular r=7 DOY median table
  (`artifacts/models/solar_doy_climatology_dev_train.json`) with explicit keys ``"1".."366"``.
- Never mutates caller DataFrames or database observations (`observations_vintage`).
- Selectively fills ONLY ``NaN``/``NULL`` cells across the 11 solar variables while
  preserving every present NASA POWER FLASHFlux / CERES measurement verbatim.
- Leaves all 24 GEOS-IT meteorological variables strictly untouched (fail-fast contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.constants import Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOLAR_DOY_CLIMATOLOGY_PATH = PROJECT_ROOT / "artifacts" / "models" / "solar_doy_climatology_dev_train.json"

PC_CERES_POLICY_NAME = "P-c_DOY_MEDIAN_11_SOLAR_V1"
PC_LEAP_DAY_DOY_POLICY = "EXPLICIT_366_CIRCULAR_MEDIAN_R7"

SOLAR_11_EN_VARS: Tuple[str, ...] = (
    "ALLSKY_SFC_SW_DWN",
    "TOA_SW_DWN",
    "ALLSKY_SFC_LW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
    "ALLSKY_SFC_PAR_TOT",
    "CLRSKY_SFC_PAR_TOT",
    "ALLSKY_SFC_UVA",
    "ALLSKY_SFC_UVB",
    "ALLSKY_SFC_UV_INDEX",
)

METEO_24_EN_VARS: Tuple[str, ...] = (
    "T2M",
    "T2MDEW",
    "T2MWET",
    "TS",
    "T2M_RANGE",
    "T2M_MAX",
    "T2M_MIN",
    "QV2M",
    "RH2M",
    "PRECTOTCORR",
    "PS",
    "WS2M",
    "WS2M_MAX",
    "WS2M_MIN",
    "WS2M_RANGE",
    "WD2M",
    "WS10M",
    "WS10M_MAX",
    "WS10M_MIN",
    "WS10M_RANGE",
    "WD10M",
    "GWETTOP",
    "GWETROOT",
    "GWETPROF",
)

SOLAR_11_VI_COLS: Tuple[str, ...] = tuple(Config.COLUMN_MAPPING[c] for c in SOLAR_11_EN_VARS)
METEO_24_VI_COLS: Tuple[str, ...] = tuple(Config.COLUMN_MAPPING[c] for c in METEO_24_EN_VARS)
SOLAR_VI_TO_EN: Dict[str, str] = {Config.COLUMN_MAPPING[c]: c for c in SOLAR_11_EN_VARS}
SOLAR_EN_TO_VI: Dict[str, str] = {c: Config.COLUMN_MAPPING[c] for c in SOLAR_11_EN_VARS}


@dataclass(frozen=True)
class SolarImputationOutcome:
    """Result returned by `apply_serving_solar_doy_fallback` on an in-memory history window."""

    filled_df: pd.DataFrame
    is_degraded_input: bool
    ceres_policy_applied: str
    imputed_values: Dict[str, float]
    imputed_variables: List[str]
    doy_table_sha256: Optional[str]
    window_imputed_cell_count: int


@lru_cache(maxsize=4)
def _load_solar_doy_climatology_cached(resolved_path: str) -> Tuple[Dict[str, Any], str]:
    path_obj = Path(resolved_path)
    raw_bytes = path_obj.read_bytes()
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    payload = json.loads(raw_bytes.decode("utf-8"))
    doy_medians = payload.get("doy_medians")
    if not isinstance(doy_medians, dict):
        raise ValueError(f"Invalid solar DOY table at {path_obj}: missing 'doy_medians' mapping.")
    expected_keys = {str(d) for d in range(1, 367)}
    for code in SOLAR_11_EN_VARS:
        per_day = doy_medians.get(code)
        if not isinstance(per_day, dict) or set(per_day.keys()) != expected_keys:
            raise ValueError(
                f"Invalid solar DOY table at {path_obj}: variable {code} must define explicit keys '1'..'366'."
            )
    return payload, sha256_hex


def load_solar_doy_climatology(path: Optional[Path] = None) -> Tuple[Dict[str, Any], str]:
    """Load validated 366-day solar DOY median table and its SHA-256 checksum."""
    target = (path or DEFAULT_SOLAR_DOY_CLIMATOLOGY_PATH).resolve()
    return _load_solar_doy_climatology_cached(str(target))


def apply_serving_solar_doy_fallback(
    df_history: pd.DataFrame,
    *,
    doy_table: Optional[Dict[str, Any]] = None,
    doy_table_sha256: Optional[str] = None,
    date_col: str = "Ngày",
) -> SolarImputationOutcome:
    """
    Fill ONLY ``NaN`` / missing cells across the 11 solar columns using Dev_Train DOY medians (1..366).

    - Works purely in-memory on a copy of `df_history`; caller DataFrame is never modified.
    - Non-null solar values (including present FLASHFlux values) are preserved untouched.
    - Anchor-day imputed values (`imputed_values`, sorted by English variable code) reflect
      ONLY the anchor row (latest date in `df_history`).
    - Rejects sentinels (`<= -900.0`) rather than silently overwriting them unless pre-normalized.
    """
    if doy_table is None or doy_table_sha256 is None:
        loaded_table, loaded_sha = load_solar_doy_climatology()
        if doy_table is None:
            doy_table = loaded_table
        if doy_table_sha256 is None:
            doy_table_sha256 = loaded_sha

    doy_medians: Dict[str, Dict[str, float]] = doy_table["doy_medians"]
    out_df = df_history.copy()
    active_date_col = date_col if date_col in out_df.columns else ("DATE" if "DATE" in out_df.columns else date_col)
    parsed_dates = pd.to_datetime(out_df[active_date_col], errors="raise").dt.normalize()
    out_df[active_date_col] = parsed_dates
    out_df = out_df.sort_values(active_date_col).reset_index(drop=True)

    doy_keys = [str(ts.timetuple().tm_yday) for ts in pd.to_datetime(out_df[active_date_col])]
    anchor_idx = len(out_df) - 1

    imputed_at_anchor: Dict[str, float] = {}
    window_imputed_cells = 0

    for code in SOLAR_11_EN_VARS:
        vi_col = SOLAR_EN_TO_VI[code]
        target_col_name = vi_col if vi_col in out_df.columns else (code if code in out_df.columns else None)
        if target_col_name is None:
            continue

        raw_notna = out_df[target_col_name].notna()
        series = pd.to_numeric(out_df[target_col_name], errors="coerce")
        if (raw_notna & series.isna()).any():
            raise ValueError(f"Column '{target_col_name}' contains non-numeric solar value(s).")
        non_null = series.dropna().to_numpy(dtype=np.float64)
        if non_null.size > 0:
            if not np.isfinite(non_null).all():
                raise ValueError(f"Column '{target_col_name}' contains non-finite (Inf/-Inf) solar value(s).")
            if (non_null <= -900.0).any():
                bad_val = float(non_null[non_null <= -900.0][0])
                raise ValueError(f"Column '{target_col_name}' contains unnormalized sentinel fill value ({bad_val}).")
        missing_mask = series.isna()
        if not missing_mask.any():
            continue

        out_series = series.to_numpy(dtype=np.float64, copy=True)
        code_table = doy_medians[code]
        for idx in np.flatnonzero(missing_mask.to_numpy()):
            d_key = doy_keys[int(idx)]
            fill_val = round(float(code_table[d_key]), 6)
            out_series[int(idx)] = fill_val
            window_imputed_cells += 1
            if int(idx) == anchor_idx:
                imputed_at_anchor[code] = fill_val

        out_df[target_col_name] = out_series

    sorted_imputed = {k: imputed_at_anchor[k] for k in sorted(imputed_at_anchor.keys())}
    is_degraded = window_imputed_cells > 0
    return SolarImputationOutcome(
        filled_df=out_df,
        is_degraded_input=is_degraded,
        ceres_policy_applied=PC_CERES_POLICY_NAME if is_degraded else "NONE",
        imputed_values=sorted_imputed,
        imputed_variables=list(sorted_imputed.keys()),
        doy_table_sha256=doy_table_sha256 if is_degraded else None,
        window_imputed_cell_count=window_imputed_cells,
    )
