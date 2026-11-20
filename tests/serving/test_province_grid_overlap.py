"""Unit tests for offline spherical province-to-grid intersection percentages."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.grid_overlap import (  # noqa: E402
    compute_all_provinces_grid_overlap,
    intersecting_ceres_cells_for_box,
)

OVERLAP_JSON_PATH = PROJECT_ROOT / "frontend" / "public" / "map" / "province_grid_overlap.json"
GEOJSON_PATH = PROJECT_ROOT / "frontend" / "public" / "map" / "vietnam_63_provinces.geojson"


def test_province_grid_overlap_sums_100_percent_and_hcmc_crosses_4_ceres_cells():
    assert OVERLAP_JSON_PATH.exists(), f"Missing generated overlap JSON at {OVERLAP_JSON_PATH}"
    data = json.loads(OVERLAP_JSON_PATH.read_text(encoding="utf-8"))
    assert data["unit_count"] == 65
    units = data["units"]
    assert len(units) == 65

    for loc_id, unit in units.items():
        assert unit["polygon_area_km2"] > 0.0, f"{loc_id} must have positive spherical area"
        m_sum = sum(c["pct_of_province_area"] for c in unit["merra2_grid"]["intersecting_cells"])
        c_sum = sum(c["pct_of_province_area"] for c in unit["ceres_grid"]["intersecting_cells"])
        assert 99.95 <= m_sum <= 100.05, f"{loc_id} MERRA-2 overlap sum={m_sum}"
        assert 99.95 <= c_sum <= 100.05, f"{loc_id} CERES overlap sum={c_sum}"

    # Verify HCMC active MERRA-2 cell [10.75, 11.25] x [106.5625, 107.1875] straddles 4 distinct 1x1 CERES cells
    hcmc = units["HCMC"]
    rep_merra = hcmc["merra2_grid"]["representative_cell"]
    assert rep_merra is not None
    assert rep_merra["lat_min"] == 10.75
    assert rep_merra["lat_max"] == 11.25
    assert rep_merra["lon_min"] == 106.5625
    assert rep_merra["lon_max"] == 107.1875

    # HCMC active MERRA-2 cell [10.75, 11.25] x [106.5625, 107.1875] intersects 4 distinct 1.0x1.0 CERES cells
    box_ceres = set(
        intersecting_ceres_cells_for_box(
            rep_merra["lat_min"],
            rep_merra["lat_max"],
            rep_merra["lon_min"],
            rep_merra["lon_max"],
        )
    )
    assert box_ceres == {(10.0, 106.0), (10.0, 107.0), (11.0, 106.0), (11.0, 107.0)}
    assert len(box_ceres) == 4
    # HCMC representative MERRA-2 cell's own coordinate-bound check; polygon itself also intersects 3 CERES cells
    assert len(hcmc["ceres_grid"]["intersecting_cells"]) >= 3


def test_compute_all_provinces_matches_disk_artifact():
    fresh = compute_all_provinces_grid_overlap(GEOJSON_PATH)
    disk = json.loads(OVERLAP_JSON_PATH.read_text(encoding="utf-8"))
    assert fresh["unit_count"] == disk["unit_count"] == 65
    assert fresh["units"]["HCMC"]["polygon_area_km2"] == disk["units"]["HCMC"]["polygon_area_km2"]
