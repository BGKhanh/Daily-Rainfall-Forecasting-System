"""
65-Unit Spherical Polygon Area vs. Official Pre-Merger Administrative Area Audit (DS108)
======================================================================================
Compares spherical polygon areas computed from `frontend/public/map/vietnam_63_provinces.geojson`
(geoBoundaries VNM-ADM1 + Nguyen Duy Liem 2025 islands) and cached in
`frontend/public/map/province_grid_overlap.json` against official pre-merger Vietnamese
provincial areas (General Statistics Office of Vietnam / Nien giam Thong ke / Bo TN&MT).

Flags every administrative unit with |polygon_area - official_area| / official_area > 5.0%
and documents the geometric investigation of Can Tho (`1,826.91 km2` simplified polygon vs
`1,440.4 km2` official GSO area, +26.83%).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
GEOJSON_PATH = REPO_ROOT / "frontend" / "public" / "map" / "vietnam_63_provinces.geojson"
OVERLAP_PATH = REPO_ROOT / "frontend" / "public" / "map" / "province_grid_overlap.json"
OUT_PATH = REPO_ROOT / "reports" / "storage_audit" / "province_area_official_comparison_65.json"

# Official pre-merger (before July 2025 provincial mergers) natural land areas in km^2
# Source: General Statistics Office of Vietnam (GSO / Nien giam Thong ke Viet Nam)
# For island districts (Hoang Sa, Truong Sa), figures represent emerged land area (~10 km^2, ~20 km^2)
# vs the simplified maritime envelope polygons included in Free-GIS-Data.
OFFICIAL_AREAS_KM2: Dict[str, Dict[str, Any]] = {
    "AN_GIANG": {"official_km2": 3536.83, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "BARIA_VUNGTAU": {"official_km2": 1982.60, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "BAC_GIANG": {"official_km2": 3851.40, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "BAC_KAN": {"official_km2": 4859.96, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "BAC_LIEU": {"official_km2": 2667.88, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "BAC_NINH": {"official_km2": 822.70, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "BEN_TRE": {"official_km2": 2379.70, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "BINH_DINH": {"official_km2": 6066.40, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "BINH_DUONG": {"official_km2": 2694.64, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "BINH_PHUOC": {"official_km2": 6873.56, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "BINH_THUAN": {"official_km2": 7942.60, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "CA_MAU": {"official_km2": 5274.51, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "CANTHO": {"official_km2": 1440.40, "unit_type": "municipality", "source": "GSO Statistical Yearbook"},
    "CAO_BANG": {"official_km2": 6700.30, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "DANANG": {"official_km2": 1284.90, "unit_type": "municipality", "source": "GSO Statistical Yearbook"},
    "DAK_LAK": {"official_km2": 13125.37, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "DAK_NONG": {"official_km2": 6514.33, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "DIEN_BIEN": {"official_km2": 9539.93, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "DONG_NAI": {"official_km2": 5863.62, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "DONG_THAP": {"official_km2": 3382.28, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "GIA_LAI": {"official_km2": 15510.99, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "HA_GIANG": {"official_km2": 7929.50, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "HA_NAM": {"official_km2": 860.90, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "HANOI": {"official_km2": 3358.90, "unit_type": "municipality", "source": "GSO Statistical Yearbook"},
    "HA_TINH": {"official_km2": 5994.50, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "HAI_DUONG": {"official_km2": 1668.20, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "HAIPHONG": {"official_km2": 1522.50, "unit_type": "municipality", "source": "GSO Statistical Yearbook"},
    "HAU_GIANG": {"official_km2": 1622.20, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "HCMC": {"official_km2": 2095.00, "unit_type": "municipality", "source": "GSO Statistical Yearbook"},
    "HOA_BINH": {"official_km2": 4591.00, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "HOANG_SA": {
        "official_km2": 10.00,
        "unit_type": "island_maritime_envelope",
        "source": "Official emerged islet area ~10 km^2 (district admin zone 305 km^2); GeoJSON stores maritime bounding envelope",
    },
    "HUNG_YEN": {"official_km2": 930.20, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "KHANH_HOA": {"official_km2": 5199.60, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "KIEN_GIANG": {"official_km2": 6352.00, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "KON_TUM": {"official_km2": 9674.20, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "LAI_CHAU": {"official_km2": 9068.73, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "LAM_DONG": {"official_km2": 9781.20, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "LANG_SON": {"official_km2": 8310.20, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "LAO_CAI": {"official_km2": 6364.00, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "LONG_AN": {"official_km2": 4494.80, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "NAM_DINH": {"official_km2": 1668.80, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "NGHE_AN": {"official_km2": 16493.70, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "NINH_BINH": {"official_km2": 1411.80, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "NINH_THUAN": {"official_km2": 3355.30, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "PHU_THO": {"official_km2": 3534.60, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "PHU_YEN": {"official_km2": 5026.00, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "QUANG_BINH": {"official_km2": 8065.30, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "QUANG_NAM": {"official_km2": 10574.90, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "QUANG_NGAI": {"official_km2": 5155.30, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "QUANG_NINH": {"official_km2": 6177.70, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "QUANG_TRI": {"official_km2": 4739.80, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "SOC_TRANG": {"official_km2": 3298.20, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "SON_LA": {"official_km2": 14123.50, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "TAY_NINH": {"official_km2": 4041.70, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "THAI_BINH": {"official_km2": 1584.60, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "THAI_NGUYEN": {"official_km2": 3536.40, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "THANH_HOA": {"official_km2": 11114.70, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "THUA_THIEN_HUE": {"official_km2": 5048.20, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "TIEN_GIANG": {"official_km2": 2556.40, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "TRA_VINH": {"official_km2": 2390.80, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "TRUONG_SA": {
        "official_km2": 20.00,
        "unit_type": "island_maritime_envelope",
        "source": "Official emerged coral-island land area < 20 km^2 (maritime waters ~160,000-180,000 km^2); GeoJSON stores simplified maritime envelope",
    },
    "TUYEN_QUANG": {"official_km2": 5867.90, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "VINH_LONG": {"official_km2": 1525.70, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "VINH_PHUC": {"official_km2": 1236.00, "unit_type": "province", "source": "GSO Statistical Yearbook"},
    "YEN_BAI": {"official_km2": 6887.70, "unit_type": "province", "source": "GSO Statistical Yearbook"},
}


def _count_vertices(geometry: Dict[str, Any]) -> int:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if gtype == "Polygon":
        return sum(len(ring) for ring in coords)
    if gtype == "MultiPolygon":
        return sum(len(ring) for poly in coords for ring in poly)
    return 0


def _load_grid_overlap_helpers() -> Any:
    import importlib.util

    mod_path = REPO_ROOT / "src" / "data" / "grid_overlap.py"
    spec = importlib.util.spec_from_file_location("grid_overlap_standalone", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {mod_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["grid_overlap_standalone"] = module
    spec.loader.exec_module(module)
    return module


def run_audit() -> Dict[str, Any]:
    grid_mod = _load_grid_overlap_helpers()
    _geometry_bbox = grid_mod._geometry_bbox
    _geometry_spherical_area_km2 = grid_mod._geometry_spherical_area_km2

    geo = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    overlap = json.loads(OVERLAP_PATH.read_text(encoding="utf-8"))
    overlap_units = overlap["units"]

    features = geo["features"]
    if len(features) != 65:
        raise ValueError(f"Expected 65 features in GeoJSON, found {len(features)}")

    records: List[Dict[str, Any]] = []
    flagged_gt_5pct: List[Dict[str, Any]] = []
    max_overlap_parity_diff_km2 = 0.0

    for feat in features:
        props = feat["properties"]
        loc_id = str(props["location_id"])
        geom = feat["geometry"]
        computed_km2 = round(_geometry_spherical_area_km2(geom), 2)
        cached_km2 = float(overlap_units[loc_id]["polygon_area_km2"])
        parity_diff = round(abs(computed_km2 - cached_km2), 4)
        max_overlap_parity_diff_km2 = max(max_overlap_parity_diff_km2, parity_diff)

        ref = OFFICIAL_AREAS_KM2[loc_id]
        official_km2 = float(ref["official_km2"])
        diff_km2 = round(computed_km2 - official_km2, 2)
        rel_diff_pct = round(((computed_km2 - official_km2) / official_km2) * 100.0, 2)
        exceeds_5pct = abs(rel_diff_pct) > 5.0

        row = {
            "location_id": loc_id,
            "display_name": props["display_name"],
            "unit_type": ref["unit_type"],
            "polygon_area_km2": computed_km2,
            "cached_overlap_polygon_area_km2": cached_km2,
            "cache_parity_diff_km2": parity_diff,
            "official_area_km2": official_km2,
            "diff_km2": diff_km2,
            "rel_diff_pct": rel_diff_pct,
            "exceeds_5pct_threshold": exceeds_5pct,
            "vertex_count": _count_vertices(geom),
            "official_source": ref["source"],
        }
        records.append(row)
        if exceeds_5pct:
            flagged_gt_5pct.append(row)

    records.sort(key=lambda r: abs(r["rel_diff_pct"]), reverse=True)
    flagged_gt_5pct.sort(key=lambda r: abs(r["rel_diff_pct"]), reverse=True)

    mainland_only = [r for r in records if r["unit_type"] != "island_maritime_envelope"]
    mainland_flagged = [r for r in mainland_only if r["exceeds_5pct_threshold"]]

    by_id = {r["location_id"]: r for r in records}
    can_feat = next(f for f in features if f["properties"]["location_id"] == "CANTHO")
    hau_feat = next(f for f in features if f["properties"]["location_id"] == "HAU_GIANG")
    can_bbox = _geometry_bbox(can_feat["geometry"])
    hau_bbox = _geometry_bbox(hau_feat["geometry"])

    combined_polygon_km2 = round(by_id["CANTHO"]["polygon_area_km2"] + by_id["HAU_GIANG"]["polygon_area_km2"], 2)
    combined_official_km2 = round(by_id["CANTHO"]["official_area_km2"] + by_id["HAU_GIANG"]["official_area_km2"], 2)
    combined_rel_diff_pct = round(((combined_polygon_km2 - combined_official_km2) / combined_official_km2) * 100.0, 2)

    can_tho_investigation = {
        "location_id": "CANTHO",
        "polygon_area_km2": by_id["CANTHO"]["polygon_area_km2"],
        "official_area_km2": by_id["CANTHO"]["official_area_km2"],
        "rel_diff_pct": by_id["CANTHO"]["rel_diff_pct"],
        "vertex_count": by_id["CANTHO"]["vertex_count"],
        "bbox_lat_lon": {
            "lat_min": round(can_bbox[0], 4),
            "lat_max": round(can_bbox[1], 4),
            "lon_min": round(can_bbox[2], 4),
            "lon_max": round(can_bbox[3], 4),
        },
        "adjacent_hau_giang": {
            "polygon_area_km2": by_id["HAU_GIANG"]["polygon_area_km2"],
            "official_area_km2": by_id["HAU_GIANG"]["official_area_km2"],
            "rel_diff_pct": by_id["HAU_GIANG"]["rel_diff_pct"],
            "vertex_count": by_id["HAU_GIANG"]["vertex_count"],
            "bbox_lat_lon": {
                "lat_min": round(hau_bbox[0], 4),
                "lat_max": round(hau_bbox[1], 4),
                "lon_min": round(hau_bbox[2], 4),
                "lon_max": round(hau_bbox[3], 4),
            },
        },
        "combined_cantho_plus_hau_giang": {
            "polygon_area_km2": combined_polygon_km2,
            "official_area_km2": combined_official_km2,
            "rel_diff_pct": combined_rel_diff_pct,
        },
        "root_cause_conclusion": (
            "geoBoundaries VNM-ADM1 simplified polygon for Can Tho (54 vertices) generalizes the "
            "2004 Can Tho / Hau Giang administrative split southward into Hau Giang: Can Tho is "
            f"+{by_id['CANTHO']['diff_km2']:.2f} km2 ({by_id['CANTHO']['rel_diff_pct']:+.2f}%) oversized "
            f"while neighboring Hau Giang is {by_id['HAU_GIANG']['diff_km2']:.2f} km2 "
            f"({by_id['HAU_GIANG']['rel_diff_pct']:+.2f}%) undersized, yet their combined polygon area "
            f"({combined_polygon_km2:.2f} km2) agrees within {combined_rel_diff_pct:+.2f}% of the official "
            f"combined total ({combined_official_km2:.2f} km2). Consequently, UI panels must label "
            "computed values as 'Polygon area (from simplified boundary data)' rather than official survey area."
        ),
    }

    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "total_units": len(records),
        "threshold_pct": 5.0,
        "max_cache_parity_diff_km2": max_overlap_parity_diff_km2,
        "summary": {
            "total_flagged_gt_5pct": len(flagged_gt_5pct),
            "mainland_units_count": len(mainland_only),
            "mainland_flagged_gt_5pct_count": len(mainland_flagged),
            "mainland_flagged_ids": [r["location_id"] for r in mainland_flagged],
            "island_maritime_envelopes": ["HOANG_SA", "TRUONG_SA"],
            "hcmc_polygon_km2": by_id["HCMC"]["polygon_area_km2"],
            "hcmc_official_km2": by_id["HCMC"]["official_area_km2"],
            "hcmc_rel_diff_pct": by_id["HCMC"]["rel_diff_pct"],
        },
        "can_tho_investigation": can_tho_investigation,
        "flagged_units_gt_5pct": flagged_gt_5pct,
        "units": records,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    res = run_audit()
    print(json.dumps(
        {
            "total_units": res["total_units"],
            "max_cache_parity_diff_km2": res["max_cache_parity_diff_km2"],
            "mainland_flagged_count": res["summary"]["mainland_flagged_gt_5pct_count"],
            "mainland_flagged_ids": res["summary"]["mainland_flagged_ids"],
            "can_tho": res["can_tho_investigation"],
        },
        indent=2,
        ensure_ascii=False,
    ))
