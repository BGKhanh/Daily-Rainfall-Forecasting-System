"""
Exact Geographic Grid-Scale & Province Polygon Overlap Calculator (DS108)
=========================================================================
Computes true spherical polygon areas (km^2), cos(phi)-scaled grid cell dimensions,
and Sutherland-Hodgman clipped polygon-cell intersection areas across all 65 pre-merger
Vietnamese units and the two NASA POWER grids actually used:
- Meteorology (MERRA-2 / GEOS-IT): 0.5 deg lat x 0.625 deg lon
- Radiation (CERES SYN1deg / FLASHFlux): 1.0 deg lat x 1.0 deg lon
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEG_LAT = 111.32

Point2D = Tuple[float, float]  # (lon, lat)


def spherical_ring_area_km2(ring: Sequence[Sequence[float]]) -> float:
    """
    Compute surface area (km^2) of a closed geographic ring [(lon, lat), ...]
    on a sphere of radius R = 6371.0088 km using Girard / Chamberlin-Duquette formula:
        A = (R^2 / 2) * |sum (lon_{i+1} - lon_{i-1}) * sin(lat_i)|
    Exact for arbitrary polygons without assuming flat Cartesian distortion.
    """
    pts: List[Point2D] = [(float(p[0]), float(p[1])) for p in ring]
    if len(pts) < 3:
        return 0.0
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    n = len(pts)
    if n < 3:
        return 0.0

    total = 0.0
    for i in range(n):
        lon_prev = math.radians(pts[(i - 1) % n][0])
        lon_next = math.radians(pts[(i + 1) % n][0])
        lat_curr = math.radians(pts[i][1])
        total += (lon_next - lon_prev) * math.sin(lat_curr)
    return abs(total) * (EARTH_RADIUS_KM ** 2) / 2.0


def _clip_polygon_edge(
    poly: List[Point2D],
    is_inside: Any,
    intersect: Any,
) -> List[Point2D]:
    if not poly:
        return []
    out: List[Point2D] = []
    prev = poly[-1]
    prev_in = is_inside(prev)
    for curr in poly:
        curr_in = is_inside(curr)
        if curr_in:
            if not prev_in:
                out.append(intersect(prev, curr))
            out.append(curr)
        elif prev_in:
            out.append(intersect(prev, curr))
        prev = curr
        prev_in = curr_in
    return out


def clip_ring_to_rect(
    ring: Sequence[Sequence[float]],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> List[Point2D]:
    """Clip a 2D polygon ring to an axis-aligned rectangle [lon_min, lon_max] x [lat_min, lat_max]."""
    pts: List[Point2D] = [(float(p[0]), float(p[1])) for p in ring]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return []

    def clip_x_min(p1: Point2D, p2: Point2D) -> Point2D:
        dx = p2[0] - p1[0]
        t = (lon_min - p1[0]) / dx if abs(dx) > 1e-15 else 0.0
        return (lon_min, p1[1] + t * (p2[1] - p1[1]))

    def clip_x_max(p1: Point2D, p2: Point2D) -> Point2D:
        dx = p2[0] - p1[0]
        t = (lon_max - p1[0]) / dx if abs(dx) > 1e-15 else 0.0
        return (lon_max, p1[1] + t * (p2[1] - p1[1]))

    def clip_y_min(p1: Point2D, p2: Point2D) -> Point2D:
        dy = p2[1] - p1[1]
        t = (lat_min - p1[1]) / dy if abs(dy) > 1e-15 else 0.0
        return (p1[0] + t * (p2[0] - p1[0]), lat_min)

    def clip_y_max(p1: Point2D, p2: Point2D) -> Point2D:
        dy = p2[1] - p1[1]
        t = (lat_max - p1[1]) / dy if abs(dy) > 1e-15 else 0.0
        return (p1[0] + t * (p2[0] - p1[0]), lat_max)

    clipped = _clip_polygon_edge(pts, lambda p: p[0] >= lon_min, clip_x_min)
    clipped = _clip_polygon_edge(clipped, lambda p: p[0] <= lon_max, clip_x_max)
    clipped = _clip_polygon_edge(clipped, lambda p: p[1] >= lat_min, clip_y_min)
    clipped = _clip_polygon_edge(clipped, lambda p: p[1] <= lat_max, clip_y_max)
    return clipped


def rect_spherical_area_km2(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> float:
    """Exact spherical zone area between [lat_min, lat_max] and [lon_min, lon_max]."""
    dlon = math.radians(lon_max - lon_min)
    s_diff = math.sin(math.radians(lat_max)) - math.sin(math.radians(lat_min))
    return abs((EARTH_RADIUS_KM ** 2) * dlon * s_diff)


def cell_dimensions_km(lat_deg: float, dlat_deg: float, dlon_deg: float) -> Tuple[float, float, float]:
    """Return `(height_km, width_km, spherical_cell_area_km2)` centered at `lat_deg`."""
    cos_phi = max(0.0, math.cos(math.radians(lat_deg)))
    height_km = dlat_deg * KM_PER_DEG_LAT
    width_km = dlon_deg * 111.3195 * cos_phi
    lat_min = lat_deg - dlat_deg / 2.0
    lat_max = lat_deg + dlat_deg / 2.0
    area_km2 = rect_spherical_area_km2(lat_min, lat_max, 0.0, dlon_deg)
    return round(height_km, 2), round(width_km, 2), round(area_km2, 2)


def _iter_polygons(geometry: Dict[str, Any]) -> Iterable[Tuple[List[Sequence[float]], List[List[Sequence[float]]]]]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if gtype == "Polygon" and coords:
        yield coords[0], coords[1:]
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                yield poly[0], poly[1:]


def _geometry_spherical_area_km2(geometry: Dict[str, Any]) -> float:
    total = 0.0
    for outer, holes in _iter_polygons(geometry):
        total += spherical_ring_area_km2(outer)
        for hole in holes:
            total -= spherical_ring_area_km2(hole)
    return max(0.0, total)


def _geometry_clipped_area_km2(
    geometry: Dict[str, Any],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> float:
    total = 0.0
    for outer, holes in _iter_polygons(geometry):
        c_outer = clip_ring_to_rect(outer, lon_min, lon_max, lat_min, lat_max)
        if len(c_outer) >= 3:
            total += spherical_ring_area_km2(c_outer)
            for hole in holes:
                c_hole = clip_ring_to_rect(hole, lon_min, lon_max, lat_min, lat_max)
                if len(c_hole) >= 3:
                    total -= spherical_ring_area_km2(c_hole)
    return max(0.0, total)


def _geometry_bbox(geometry: Dict[str, Any]) -> Tuple[float, float, float, float]:
    lons: List[float] = []
    lats: List[float] = []
    for outer, _ in _iter_polygons(geometry):
        for pt in outer:
            lons.append(float(pt[0]))
            lats.append(float(pt[1]))
    return min(lats), max(lats), min(lons), max(lons)


def _merra_cell_for_point(lat: float, lon: float) -> Tuple[float, float, float, float]:
    c_lat = round(lat / 0.5) * 0.5
    c_lon = round(lon / 0.625) * 0.625
    return c_lat - 0.25, c_lat + 0.25, c_lon - 0.3125, c_lon + 0.3125


def _ceres_cell_for_point(lat: float, lon: float) -> Tuple[float, float, float, float]:
    f_lat = float(math.floor(lat))
    f_lon = float(math.floor(lon))
    return f_lat, f_lat + 1.0, f_lon, f_lon + 1.0


def intersecting_ceres_cells_for_box(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> List[Tuple[float, float]]:
    """Return distinct (lat_min, lon_min) corners of 1.0 x 1.0 deg cells intersecting an interior-open box."""
    lat0 = math.floor(lat_min)
    lat1 = math.floor(lat_max - 1e-9)
    lon0 = math.floor(lon_min)
    lon1 = math.floor(lon_max - 1e-9)
    return [(float(la), float(lo)) for la in range(lat0, lat1 + 1) for lo in range(lon0, lon1 + 1)]


def _intersecting_cells_for_geometry(
    geometry: Dict[str, Any],
    *,
    grid_kind: str,
    poly_area_km2: float,
    rep_lat: float,
    rep_lon: float,
) -> List[Dict[str, Any]]:
    lat_min_g, lat_max_g, lon_min_g, lon_max_g = _geometry_bbox(geometry)

    if grid_kind == "merra2":
        dlat, dlon = 0.5, 0.625
        # MERRA-2 centers sit at k*0.5 lat -> edges at k*0.5 - 0.25; lon centers at m*0.625 -> edges at m*0.625 - 0.3125
        k_start = math.floor((lat_min_g + 0.25) / dlat) - 1
        k_end = math.ceil((lat_max_g + 0.25) / dlat) + 1
        m_start = math.floor((lon_min_g + 0.3125) / dlon) - 1
        m_end = math.ceil((lon_max_g + 0.3125) / dlon) + 1
        rep_bounds = _merra_cell_for_point(rep_lat, rep_lon)
        cell_iter = [
            (
                round(k * dlat - 0.25, 6),
                round(k * dlat + 0.25, 6),
                round(m * dlon - 0.3125, 6),
                round(m * dlon + 0.3125, 6),
            )
            for k in range(k_start, k_end + 1)
            for m in range(m_start, m_end + 1)
        ]
    else:
        dlat, dlon = 1.0, 1.0
        k_start = math.floor(lat_min_g) - 1
        k_end = math.floor(lat_max_g) + 1
        m_start = math.floor(lon_min_g) - 1
        m_end = math.floor(lon_max_g) + 1
        rep_bounds = _ceres_cell_for_point(rep_lat, rep_lon)
        cell_iter = [
            (float(k), float(k + 1), float(m), float(m + 1))
            for k in range(k_start, k_end + 1)
            for m in range(m_start, m_end + 1)
        ]

    cells: List[Dict[str, Any]] = []
    for c_lat0, c_lat1, c_lon0, c_lon1 in cell_iter:
        if c_lat1 < lat_min_g or c_lat0 > lat_max_g or c_lon1 < lon_min_g or c_lon0 > lon_max_g:
            continue
        inter_km2 = _geometry_clipped_area_km2(geometry, c_lat0, c_lat1, c_lon0, c_lon1)
        is_rep = (
            abs(c_lat0 - rep_bounds[0]) < 1e-6
            and abs(c_lat1 - rep_bounds[1]) < 1e-6
            and abs(c_lon0 - rep_bounds[2]) < 1e-6
            and abs(c_lon1 - rep_bounds[3]) < 1e-6
        )
        if inter_km2 <= 1e-5 and not is_rep:
            continue
        cell_area = rect_spherical_area_km2(c_lat0, c_lat1, c_lon0, c_lon1)
        c_mid_lat = (c_lat0 + c_lat1) / 2.0
        h_km, w_km, _ = cell_dimensions_km(c_mid_lat, dlat, dlon)
        pct_prov = (inter_km2 / poly_area_km2 * 100.0) if poly_area_km2 > 0 else 0.0
        pct_cell = (inter_km2 / cell_area * 100.0) if cell_area > 0 else 0.0
        cells.append(
            {
                "cell_id": f"{grid_kind}_{c_lat0:.4f}_{c_lon0:.4f}",
                "lat_min": c_lat0,
                "lat_max": c_lat1,
                "lon_min": c_lon0,
                "lon_max": c_lon1,
                "cell_area_km2": round(cell_area, 3),
                "cell_height_km": h_km,
                "cell_width_km": w_km,
                "intersection_area_km2": round(inter_km2, 4),
                "pct_of_province_area": round(pct_prov, 4),
                "pct_of_cell_area": round(pct_cell, 4),
                "contains_representative_point": is_rep,
            }
        )

    cells.sort(key=lambda c: (-c["pct_of_province_area"], -int(c["contains_representative_point"]), c["cell_id"]))
    return cells


def _lookup_rep_coords(display_name: str, fallback_lat: float, fallback_lon: float) -> Tuple[float, float]:
    from src.config.constants import Config

    def _norm(s: str) -> str:
        import unicodedata

        nfkd = unicodedata.normalize("NFKD", s.replace("Đ", "D").replace("đ", "d"))
        ascii_s = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
        return " ".join(ascii_s.lower().replace("-", " ").split())

    target_norm = _norm(display_name)
    aliases = {
        "ho chi minh city": "tp ho chi minh",
        "hoang sa island": "hoang sa",
        "truong sa island": "truong sa",
    }
    lookup_key = aliases.get(target_norm, target_norm)
    for k, coords in Config.province_coordinates.items():
        if _norm(k) == lookup_key:
            return float(coords[0]), float(coords[1])
    return float(fallback_lat), float(fallback_lon)


def compute_all_provinces_grid_overlap(
    geojson_path: Path,
) -> Dict[str, Any]:
    """Compute grid cell overlap and spherical dimensions for all 65 pre-merger GeoJSON units."""
    geo = json.loads(geojson_path.read_text(encoding="utf-8"))
    units: Dict[str, Any] = {}

    for feat in geo.get("features", []):
        props = feat.get("properties") or {}
        loc_id = str(props["location_id"])
        display_name = str(props["display_name"])
        cent_lat = float(props.get("centroid_lat", 0.0))
        cent_lon = float(props.get("centroid_lon", 0.0))
        rep_lat, rep_lon = _lookup_rep_coords(display_name, cent_lat, cent_lon)
        geom = feat.get("geometry") or {}
        poly_area_km2 = _geometry_spherical_area_km2(geom)

        merra_cells = _intersecting_cells_for_geometry(
            geom,
            grid_kind="merra2",
            poly_area_km2=poly_area_km2,
            rep_lat=rep_lat,
            rep_lon=rep_lon,
        )
        ceres_cells = _intersecting_cells_for_geometry(
            geom,
            grid_kind="ceres",
            poly_area_km2=poly_area_km2,
            rep_lat=rep_lat,
            rep_lon=rep_lon,
        )

        m_h, m_w, m_area = cell_dimensions_km(rep_lat, 0.5, 0.625)
        c_h, c_w, c_area = cell_dimensions_km(rep_lat, 1.0, 1.0)
        rep_merra = next((c for c in merra_cells if c["contains_representative_point"]), merra_cells[0] if merra_cells else None)
        rep_ceres = next((c for c in ceres_cells if c["contains_representative_point"]), ceres_cells[0] if ceres_cells else None)

        units[loc_id] = {
            "location_id": loc_id,
            "display_name": display_name,
            "representative_point": {"lat": round(rep_lat, 4), "lon": round(rep_lon, 4)},
            "polygon_area_km2": round(poly_area_km2, 2),
            "merra2_grid": {
                "resolution_deg": "0.5 x 0.625",
                "representative_cell": rep_merra,
                "cell_dimensions_km_at_phi": {"height_km": m_h, "width_km": m_w, "area_km2": m_area},
                "intersecting_cell_count": len(merra_cells),
                "pct_sum": round(sum(c["pct_of_province_area"] for c in merra_cells), 4),
                "intersecting_cells": merra_cells,
            },
            "ceres_grid": {
                "resolution_deg": "1.0 x 1.0",
                "representative_cell": rep_ceres,
                "cell_dimensions_km_at_phi": {"height_km": c_h, "width_km": c_w, "area_km2": c_area},
                "intersecting_cell_count": len(ceres_cells),
                "pct_sum": round(sum(c["pct_of_province_area"] for c in ceres_cells), 4),
                "intersecting_cells": ceres_cells,
            },
        }

    return {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "earth_radius_km": EARTH_RADIUS_KM,
        "grids": {
            "merra2_geosit": {"dlat_deg": 0.5, "dlon_deg": 0.625},
            "ceres_flashflux": {"dlat_deg": 1.0, "dlon_deg": 1.0},
        },
        "unit_count": len(units),
        "units": units,
    }
