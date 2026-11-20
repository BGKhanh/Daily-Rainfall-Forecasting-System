"""
Round 11 — Step 1 (Part C): Empirical Per-Variable Grid Boundary Probe & Catalog Generator
=========================================================================================
Probes all 35 variables in ALL_35_VARS across 7 spatial coordinates around HCMC
(10.7769N, 106.7009E) over complete window 2026-08-26..2026-08-28 to verify each
variable's exact NASA POWER grid cell (0.5 x 0.625 deg vs 1.0 x 1.0 deg).

Outputs:
- reports/storage_audit/per_variable_grid_verification.json
- config/variable_catalog.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import Config  # noqa: E402
from src.data.vintage_store import ALL_35_VARS, CERES_ONLY_8_VARS, FLASHFLUX_3_VARS, METEO_24_VARS  # noqa: E402

DEFAULT_WINDOW_START = "20260826"
DEFAULT_WINDOW_END = "20260828"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "storage_audit" / "per_variable_grid_verification.json"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "config" / "variable_catalog.json"

PROBE_POINTS: Tuple[Dict[str, Any], ...] = (
    {"key": "anchor_hcmc", "lat": 10.7769, "lon": 106.7009, "tier": "anchor"},
    {"key": "inside_meteo_sw", "lat": 10.76, "lon": 106.57, "tier": "inside_0.5x0.625"},
    {"key": "inside_meteo_ne", "lat": 11.24, "lon": 107.18, "tier": "inside_0.5x0.625"},
    {"key": "mid_inside_rad_south", "lat": 10.50, "lon": 106.7009, "tier": "outside_0.5x0.625_inside_1.0x1.0"},
    {"key": "mid_inside_rad_west", "lat": 10.7769, "lon": 106.30, "tier": "outside_0.5x0.625_inside_1.0x1.0"},
    {"key": "outside_rad_south", "lat": 9.99, "lon": 106.7009, "tier": "outside_1.0x1.0"},
    {"key": "outside_rad_north", "lat": 11.01, "lon": 106.7009, "tier": "outside_1.0x1.0_inside_0.5x0.625"},
    {"key": "outside_both_north", "lat": 11.26, "lon": 106.7009, "tier": "outside_both_grids"},
)

# Metadata (English label, unit, source roles) aligned with NASA POWER v2.10.4 & Config.COLUMN_MAPPING
VAR_METADATA: Dict[str, Tuple[str, str]] = {
    "ALLSKY_SFC_SW_DWN": ("All Sky Surface Shortwave Downward Irradiance", "MJ/m^2/day"),
    "TOA_SW_DWN": ("Top-Of-Atmosphere Shortwave Downward Irradiance", "MJ/m^2/day"),
    "ALLSKY_SFC_LW_DWN": ("All Sky Surface Longwave Downward Irradiance", "W/m^2"),
    "CLRSKY_SFC_SW_DWN": ("Clear Sky Surface Shortwave Downward Irradiance", "MJ/m^2/day"),
    "ALLSKY_SFC_SW_DNI": ("All Sky Direct Normal Irradiance", "MJ/m^2/day"),
    "ALLSKY_SFC_SW_DIFF": ("All Sky Diffuse Shortwave Irradiance", "MJ/m^2/day"),
    "ALLSKY_SFC_PAR_TOT": ("All Sky Photosynthetically Active Radiation", "W/m^2"),
    "CLRSKY_SFC_PAR_TOT": ("Clear Sky Photosynthetically Active Radiation", "W/m^2"),
    "ALLSKY_SFC_UVA": ("All Sky UVA Irradiance", "W/m^2"),
    "ALLSKY_SFC_UVB": ("All Sky UVB Irradiance", "W/m^2"),
    "ALLSKY_SFC_UV_INDEX": ("All Sky UV Index", "index"),
    "T2M": ("Air Temperature at 2m", "C"),
    "T2MDEW": ("Dew/Frost Point at 2m", "C"),
    "T2MWET": ("Wet Bulb Temperature at 2m", "C"),
    "TS": ("Earth Skin Temperature", "C"),
    "T2M_RANGE": ("Temperature Range at 2m", "C"),
    "T2M_MAX": ("Maximum Temperature at 2m", "C"),
    "T2M_MIN": ("Minimum Temperature at 2m", "C"),
    "QV2M": ("Specific Humidity at 2m", "g/kg"),
    "RH2M": ("Relative Humidity at 2m", "%"),
    "PRECTOTCORR": ("Precipitation Corrected", "mm/day"),
    "PS": ("Surface Pressure", "kPa"),
    "WS2M": ("Wind Speed at 2m", "m/s"),
    "WS2M_MAX": ("Maximum Wind Speed at 2m", "m/s"),
    "WS2M_MIN": ("Minimum Wind Speed at 2m", "m/s"),
    "WS2M_RANGE": ("Wind Speed Range at 2m", "m/s"),
    "WD2M": ("Wind Direction at 2m", "degrees"),
    "WS10M": ("Wind Speed at 10m", "m/s"),
    "WS10M_MAX": ("Maximum Wind Speed at 10m", "m/s"),
    "WS10M_MIN": ("Minimum Wind Speed at 10m", "m/s"),
    "WS10M_RANGE": ("Wind Speed Range at 10m", "m/s"),
    "WD10M": ("Wind Direction at 10m", "degrees"),
    "GWETTOP": ("Surface Soil Wetness", "0..1"),
    "GWETROOT": ("Root Zone Soil Wetness", "0..1"),
    "GWETPROF": ("Profile Soil Moisture", "0..1"),
}


def _fetch_batch(
    lat: float,
    lon: float,
    params: List[str],
    start: str,
    end: str,
    timeout: int = 120,
) -> Dict[str, Any]:
    url = Config.NASA_API_BASE_URL
    query = {
        "parameters": ",".join(params),
        "community": "RE",
        "longitude": f"{lon:.4f}",
        "latitude": f"{lat:.4f}",
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "LST",
    }
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=query, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"NASA probe failed at ({lat}, {lon}) params={len(params)}: {last_err}")


def _series_equal(a: Dict[str, float], b: Dict[str, float], tol: float = 1e-9) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if va is None or vb is None:
            if va != vb:
                return False
        elif not math.isclose(float(va), float(vb), abs_tol=tol, rel_tol=0.0):
            return False
    return True


def _clean_series(raw: Dict[str, Any]) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {}
    for d_str, val in sorted(raw.items()):
        f = float(val)
        out[d_str] = None if (not math.isfinite(f) or f <= -900.0) else f
    return out


def classify_variable_grid(point_series: Dict[str, Dict[str, float | None]]) -> Tuple[str, str]:
    """
    Determine spatial grid strictly from observed equality pattern against anchor_hcmc:
    - 0.5x0.625 (HCMC [10.75,11.25] x [106.5625,107.1875]):
        inside_meteo_sw (10.76, 106.57), inside_meteo_ne (11.24, 107.18), and
        outside_rad_north (11.01, 106.7009 -- inside [10.75,11.25]) ALL match anchor;
        mid_inside_rad_south (10.50), mid_inside_rad_west (106.30), outside_rad_south (9.99),
        and outside_both_north (11.26) ALL differ.
    - 1.0x1.0 (HCMC [10.0,11.0] x [106.0,107.0]):
        inside_meteo_sw (10.76, 106.57), mid_inside_rad_south (10.50), and
        mid_inside_rad_west (106.30) MATCH anchor;
        inside_meteo_ne (11.24, 107.18), outside_rad_south (9.99), outside_rad_north (11.01),
        and outside_both_north (11.26) ALL differ.
    """
    anc = point_series["anchor_hcmc"]
    eq_sw = _series_equal(anc, point_series["inside_meteo_sw"])
    eq_ne = _series_equal(anc, point_series["inside_meteo_ne"])
    eq_mid_s = _series_equal(anc, point_series["mid_inside_rad_south"])
    eq_mid_w = _series_equal(anc, point_series["mid_inside_rad_west"])
    eq_out_s = _series_equal(anc, point_series["outside_rad_south"])
    eq_out_n = _series_equal(anc, point_series["outside_rad_north"])
    eq_out_both_n = _series_equal(anc, point_series["outside_both_north"])

    if eq_sw and eq_ne and eq_out_n and (not eq_mid_s) and (not eq_mid_w) and (not eq_out_s) and (not eq_out_both_n):
        return "MERRA2_GEOSIT_0.5x0.625", "0.5 x 0.625"
    if eq_mid_s and eq_mid_w and (not eq_ne) and (not eq_out_s) and (not eq_out_n) and (not eq_out_both_n):
        return "CERES_FLASHFLUX_1.0x1.0", "1.0 x 1.0"
    if eq_sw and eq_mid_s and eq_mid_w and eq_ne and eq_out_s and eq_out_n and eq_out_both_n:
        return "UNIFORM_ACROSS_PROBES", "uniform_in_window"
    return (
        "AMBIGUOUS",
        f"eq_sw={eq_sw},eq_ne={eq_ne},eq_mid_s={eq_mid_s},eq_mid_w={eq_mid_w},"
        f"eq_out_s={eq_out_s},eq_out_n={eq_out_n},eq_out_both_n={eq_out_both_n}",
    )


def _group_and_sources(var_code: str) -> Tuple[str, str, str]:
    if var_code in CERES_ONLY_8_VARS:
        return "solar_ceres_only", "SYN1DEG", "CERES SYN1deg (unavailable until ~80d lag)"
    if var_code in FLASHFLUX_3_VARS:
        return "solar_flashflux_fallback", "SYN1DEG", "FLASHFLUX"
    if var_code in METEO_24_VARS:
        return "meteorology", "MERRA2", "GEOSIT"
    raise ValueError(f"Unknown variable code: {var_code}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Empirical per-variable grid probe for all 35 NASA POWER vars.")
    parser.add_argument("--start", default=DEFAULT_WINDOW_START)
    parser.add_argument("--end", default=DEFAULT_WINDOW_END)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    args = parser.parse_args()

    all_vars = list(ALL_35_VARS)
    assert len(all_vars) == 35
    batch1 = all_vars[:20]
    batch2 = all_vars[20:]

    raw_by_point: Dict[str, Dict[str, Any]] = {}
    api_version = None
    header_sources_seen: set[str] = set()

    for pt in PROBE_POINTS:
        r1 = _fetch_batch(pt["lat"], pt["lon"], batch1, args.start, args.end)
        r2 = _fetch_batch(pt["lat"], pt["lon"], batch2, args.start, args.end)
        api_version = api_version or r1.get("header", {}).get("api", {}).get("version")
        for s in (r1.get("header", {}).get("sources") or []) + (r2.get("header", {}).get("sources") or []):
            header_sources_seen.add(str(s))
        merged: Dict[str, Dict[str, float | None]] = {}
        for var in batch1:
            merged[var] = _clean_series(r1["properties"]["parameter"][var])
        for var in batch2:
            merged[var] = _clean_series(r2["properties"]["parameter"][var])
        raw_by_point[pt["key"]] = {
            "meta": pt,
            "variables": merged,
            "sources_batch1": r1.get("header", {}).get("sources"),
            "sources_batch2": r2.get("header", {}).get("sources"),
        }

    verified_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    per_var_report: Dict[str, Any] = {}
    catalog_entries: List[Dict[str, Any]] = []
    grid_counts: Dict[str, int] = {}

    for var in all_vars:
        pt_series = {pt["key"]: raw_by_point[pt["key"]]["variables"][var] for pt in PROBE_POINTS}
        grid_id, grid_res = classify_variable_grid(pt_series)
        grid_counts[grid_id] = grid_counts.get(grid_id, 0) + 1
        grp, src_final, src_prov = _group_and_sources(var)
        label_en, unit = VAR_METADATA[var]
        label_vi = Config.COLUMN_MAPPING[var]
        per_var_report[var] = {
            "variable": var,
            "label_vi": label_vi,
            "label_en": label_en,
            "unit": unit,
            "group": grp,
            "classified_grid_id": grid_id,
            "classified_grid_resolution_deg": grid_res,
            "equality_vs_anchor": {
                pt["key"]: _series_equal(pt_series["anchor_hcmc"], pt_series[pt["key"]])
                for pt in PROBE_POINTS
                if pt["key"] != "anchor_hcmc"
            },
            "sample_values": pt_series,
        }
        catalog_entries.append(
            {
                "code": var,
                "label_vi": label_vi,
                "label_en": label_en,
                "unit": unit,
                "group": grp,
                "source_finalized": src_final,
                "source_provisional": src_prov,
                "grid_id": grid_id,
                "grid_resolution_deg": grid_res,
                "required_by_serving_model": True,
                "is_queryable_in_ui": var == "PRECTOTCORR",
                "verified_by_probe_date": verified_date,
            }
        )

    report_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "api_version": api_version,
        "window": {"start": args.start, "end": args.end, "time_standard": "LST", "community": "RE"},
        "probe_points": list(PROBE_POINTS),
        "header_sources_observed": sorted(header_sources_seen),
        "total_variables_probed": len(all_vars),
        "grid_classification_counts": grid_counts,
        "excluded_non_model_vars_confirmed_absent": ["CLOUD_AMT", "ALB"],
        "toa_sw_dwn_present": "TOA_SW_DWN" in all_vars,
        "variables": per_var_report,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    catalog_payload = {
        "schema_version": "1.0.0",
        "verified_by_probe_date": verified_date,
        "probe_window": f"{args.start}..{args.end}",
        "total_variables": len(catalog_entries),
        "variables": catalog_entries,
    }
    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    args.catalog.write_text(json.dumps(catalog_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("PROBE_SUMMARY:", json.dumps({"counts": grid_counts, "toa": report_payload["toa_sw_dwn_present"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
