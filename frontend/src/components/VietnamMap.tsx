import React, { useCallback, useLayoutEffect, useMemo, useRef, useState } from "react";
import { EN, stripDiacriticsForSearch } from "../i18n/en";
import type { LocationItem, ProvinceFeature, ProvinceGeoJson } from "../types/api";

interface VietnamMapProps {
  geoJson: ProvinceGeoJson | null;
  mapError: string | null;
  locations: LocationItem[];
  selectedLocationIds: string[];
  focusedLocationId: string;
  onToggleLocation: (locationId: string) => void;
  onRemoveLocation?: (locationId: string) => void;
  onClearSelection?: () => void;
  onResetSelection: () => void;
}

const MIN_ZOOM = 1.0;
const MAX_ZOOM = 6.0;

const LON_MIN = 101.8;
const LON_MAX = 117.3;
const LAT_MIN = 6.6;
const LAT_MAX = 23.6;
const COS_MID_LAT = Math.cos((15.0 * Math.PI) / 180);
const SVG_W = 540;
const SVG_H = Math.round(SVG_W * ((LAT_MAX - LAT_MIN) / ((LON_MAX - LON_MIN) * COS_MID_LAT)));

const MERRA2_LAT_STEP = 0.5;
const MERRA2_LON_STEP = 0.625;

// Verified HCMC MERRA-2 grid cell [10.75N, 11.25N] x [106.5625E, 107.1875E]
const HCMC_CELL_KEY = `${Math.round(11.0 / MERRA2_LAT_STEP)}_${Math.round(106.875 / MERRA2_LON_STEP)}`;

interface GridCellDef {
  key: string;
  latMin: number;
  latMax: number;
  lonMin: number;
  lonMax: number;
}

function projectCoord(lon: number, lat: number): [number, number] {
  const x = ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * SVG_W;
  const y = SVG_H - ((lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * SVG_H;
  return [Number(x.toFixed(2)), Number(y.toFixed(2))];
}

function buildMerra2Lattice(): GridCellDef[] {
  const cells: GridCellDef[] = [];
  const latIdxMin = Math.floor(LAT_MIN / MERRA2_LAT_STEP);
  const latIdxMax = Math.ceil(LAT_MAX / MERRA2_LAT_STEP);
  const lonIdxMin = Math.floor(LON_MIN / MERRA2_LON_STEP);
  const lonIdxMax = Math.ceil(LON_MAX / MERRA2_LON_STEP);
  for (let i = latIdxMin; i <= latIdxMax; i++) {
    const cLat = i * MERRA2_LAT_STEP;
    const latMin = cLat - 0.25;
    const latMax = cLat + 0.25;
    if (latMax < LAT_MIN || latMin > LAT_MAX) continue;
    for (let j = lonIdxMin; j <= lonIdxMax; j++) {
      const cLon = j * MERRA2_LON_STEP;
      const lonMin = cLon - 0.3125;
      const lonMax = cLon + 0.3125;
      if (lonMax < LON_MIN || lonMin > LON_MAX) continue;
      cells.push({
        key: `${i}_${j}`,
        latMin,
        latMax,
        lonMin,
        lonMax,
      });
    }
  }
  return cells;
}

const MERRA2_LATTICE = buildMerra2Lattice();

function pointInRing(lon: number, lat: number, ring: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0];
    const yi = ring[i][1];
    const xj = ring[j][0];
    const yj = ring[j][1];
    const intersect = yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function getFeatureOuterRings(feature: ProvinceFeature): number[][][] {
  if (feature.geometry.type === "Polygon") {
    const coords = feature.geometry.coordinates as number[][][];
    return coords.length ? [coords[0]] : [];
  }
  const polys = feature.geometry.coordinates as number[][][][];
  return polys.filter((p) => p.length > 0).map((p) => p[0]);
}

function getIntersectingCellKeys(feature: ProvinceFeature): Set<string> {
  const keys = new Set<string>();
  const rings = getFeatureOuterRings(feature);
  for (const ring of rings) {
    let minLon = Infinity;
    let maxLon = -Infinity;
    let minLat = Infinity;
    let maxLat = -Infinity;
    for (const [lon, lat] of ring) {
      if (lon < minLon) minLon = lon;
      if (lon > maxLon) maxLon = lon;
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
      const li = Math.round(lat / MERRA2_LAT_STEP);
      const lj = Math.round(lon / MERRA2_LON_STEP);
      keys.add(`${li}_${lj}`);
    }
    for (const cell of MERRA2_LATTICE) {
      if (cell.lonMax < minLon || cell.lonMin > maxLon || cell.latMax < minLat || cell.latMin > maxLat) {
        continue;
      }
      if (keys.has(cell.key)) continue;
      // Check cell corners and center inside polygon ring
      const samples: Array<[number, number]> = [
        [(cell.lonMin + cell.lonMax) / 2, (cell.latMin + cell.latMax) / 2],
        [cell.lonMin + 0.05, cell.latMin + 0.05],
        [cell.lonMax - 0.05, cell.latMax - 0.05],
        [cell.lonMin + 0.05, cell.latMax - 0.05],
        [cell.lonMax - 0.05, cell.latMin + 0.05],
      ];
      if (samples.some(([sx, sy]) => pointInRing(sx, sy, ring))) {
        keys.add(cell.key);
      }
    }
  }
  return keys;
}

function ringsToSvgPath(rings: number[][][]): string {
  return rings
    .map((ring) => {
      if (!ring.length) return "";
      const pts = ring.map(([lon, lat], idx) => {
        const [x, y] = projectCoord(lon, lat);
        return `${idx === 0 ? "M" : "L"}${x},${y}`;
      });
      return `${pts.join("")}Z`;
    })
    .join(" ");
}

function featureToSvgPath(feature: ProvinceFeature): string {
  if (feature.geometry.type === "Polygon") {
    return ringsToSvgPath(feature.geometry.coordinates as number[][][]);
  }
  const polys = feature.geometry.coordinates as number[][][][];
  return polys.map((poly) => ringsToSvgPath(poly)).join(" ");
}

export const VietnamMap: React.FC<VietnamMapProps> = ({
  geoJson,
  mapError,
  locations,
  selectedLocationIds,
  focusedLocationId,
  onToggleLocation,
  onRemoveLocation,
  onClearSelection,
  onResetSelection: _onResetSelection,
}) => {
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [hoveredFeature, setHoveredFeature] = useState<ProvinceFeature | null>(null);
  const [showBoundaries, setShowBoundaries] = useState<boolean>(true);
  const [showGridCells, setShowGridCells] = useState<boolean>(true);
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const dragRef = useRef<{ active: boolean; lastX: number; lastY: number; moved: boolean }>({
    active: false,
    lastX: 0,
    lastY: 0,
    moved: false,
  });
  const pinchDistRef = useRef<number | null>(null);

  const clampZoom = (z: number) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, Number(z.toFixed(2))));

  const clampPan = (px: number, py: number, z: number) => {
    if (z <= 1.0) return { x: 0, y: 0 };
    const maxPanX = (SVG_W * (z - 1)) / 2;
    const maxPanY = (SVG_H * (z - 1)) / 2;
    return {
      x: Math.max(-maxPanX, Math.min(maxPanX, px)),
      y: Math.max(-maxPanY, Math.min(maxPanY, py)),
    };
  };

  const applyZoomFactor = (factor: number) => {
    setZoom((prevZ) => {
      const nextZ = clampZoom(prevZ * factor);
      setPan((prevPan) => clampPan(prevPan.x, prevPan.y, nextZ));
      return nextZ;
    });
  };

  const resetView = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };

  const fitToSelection = () => {
    const selectedFeats = features.filter((f) => selectedSet.has(f.properties.location_id));
    if (selectedFeats.length === 0) {
      resetView();
      return;
    }
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    const pushLonLat = (lon: number, lat: number) => {
      const [px, py] = projectCoord(lon, lat);
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    };
    for (const feat of selectedFeats) {
      const polys =
        feat.geometry.type === "Polygon"
          ? [feat.geometry.coordinates as number[][][]]
          : (feat.geometry.coordinates as number[][][][]);
      for (const poly of polys) {
        for (const ring of poly) {
          for (const [lo, la] of ring) {
            pushLonLat(lo, la);
          }
        }
      }
    }
    if (!Number.isFinite(minX)) {
      resetView();
      return;
    }
    const bw = Math.max(24, maxX - minX);
    const bh = Math.max(24, maxY - minY);
    const fitZ = clampZoom(Math.max(1.2, Math.min(SVG_W / (bw * 1.4), SVG_H / (bh * 1.4))));
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const targetPan = clampPan(fitZ * (SVG_W / 2 - cx), fitZ * (SVG_H / 2 - cy), fitZ);
    setZoom(fitZ);
    setPan(targetPan);
  };

  const activeLocationsById = useMemo(() => {
    const map = new Map<string, LocationItem>();
    for (const loc of locations) {
      if (loc.is_active) {
        map.set(loc.location_id, loc);
      }
    }
    return map;
  }, [locations]);

  const features = geoJson?.features ?? [];
  const selectedSet = useMemo(() => new Set(selectedLocationIds), [selectedLocationIds]);

  const cellKeysByLocId = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const f of features) {
      map.set(f.properties.location_id, getIntersectingCellKeys(f));
    }
    return map;
  }, [features]);

  const highlightedCellKeys = useMemo(() => {
    const combined = new Set<string>();
    for (const locId of selectedLocationIds) {
      const cells = cellKeysByLocId.get(locId);
      if (cells) {
        for (const k of cells) combined.add(k);
      }
    }
    return combined;
  }, [selectedLocationIds, cellKeysByLocId]);

  const filteredFeatures = useMemo(() => {
    const q = stripDiacriticsForSearch(searchQuery);
    if (!q) return features;
    return features.filter((f) => {
      const nameClean = stripDiacriticsForSearch(f.properties.display_name);
      const idClean = stripDiacriticsForSearch(f.properties.location_id);
      return nameClean.includes(q) || idClean.includes(q);
    });
  }, [features, searchQuery]);

  const featureByLocId = useMemo(() => {
    const map = new Map<string, ProvinceFeature>();
    for (const f of features) map.set(f.properties.location_id, f);
    return map;
  }, [features]);

  const activeLocationsList = useMemo(
    () => Array.from(activeLocationsById.values()),
    [activeLocationsById]
  );

  const selectedScrollRef = useRef<HTMLDivElement | null>(null);
  const [scrollStats, setScrollStats] = useState<{
    hiddenBelow: number;
    visible: number;
    total: number;
  }>({ hiddenBelow: 0, visible: selectedLocationIds.length, total: selectedLocationIds.length });

  const recomputeScrollStats = useCallback(() => {
    const total = selectedLocationIds.length;
    const container = selectedScrollRef.current;
    if (!container || container.clientHeight === 0) {
      const visible = Math.min(total, 5);
      setScrollStats({
        hiddenBelow: Math.max(0, total - visible),
        visible,
        total,
      });
      return;
    }
    const rows = Array.from(container.querySelectorAll("tbody tr")) as HTMLTableRowElement[];
    if (rows.length === 0) {
      setScrollStats({ hiddenBelow: 0, visible: 0, total: 0 });
      return;
    }
    const viewTop = container.scrollTop;
    const viewBottom = viewTop + container.clientHeight;
    let hiddenBelow = 0;
    let hiddenAbove = 0;
    for (const row of rows) {
      const rowTop = row.offsetTop;
      const rowBottom = rowTop + row.offsetHeight;
      if (rowBottom > viewBottom + 2) {
        hiddenBelow += 1;
      } else if (rowBottom <= viewTop + 1) {
        hiddenAbove += 1;
      }
    }
    const visible = Math.max(0, rows.length - hiddenBelow - hiddenAbove);
    setScrollStats({ hiddenBelow, visible, total: rows.length });
  }, [selectedLocationIds]);

  useLayoutEffect(() => {
    recomputeScrollStats();
    const container = selectedScrollRef.current;
    if (!container || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => recomputeScrollStats());
    observer.observe(container);
    return () => observer.disconnect();
  }, [recomputeScrollStats]);

  return (
    <section className="panel map-panel" aria-label={EN.map.heading}>
      <div className="panel-header-row">
        <div>
          <h2>{EN.map.heading}</h2>
          <p className="panel-subtitle">{EN.map.subtitle}</p>
        </div>
        <div className="map-layer-toggles">
          <label className="layer-toggle-label">
            <input
              type="checkbox"
              checked={showBoundaries}
              onChange={(e) => setShowBoundaries(e.target.checked)}
            />
            <span>{EN.map.toggleBoundaries}</span>
          </label>
          <label className="layer-toggle-label">
            <input
              type="checkbox"
              checked={showGridCells}
              onChange={(e) => setShowGridCells(e.target.checked)}
            />
            <span>{EN.map.toggleGridCells}</span>
          </label>
        </div>
      </div>

      <div className="map-workspace">
        <div className="map-canvas-col">
          {mapError ? (
            <div className="info-banner">{mapError}</div>
          ) : !geoJson ? (
            <div className="empty-box">{EN.map.loadingMap}</div>
          ) : (
            <div
              className="map-svg-wrap"
              tabIndex={0}
              onWheel={(e) => {
                if (e.ctrlKey || e.metaKey) {
                  e.preventDefault();
                  applyZoomFactor(e.deltaY < 0 ? 1.2 : 1 / 1.2);
                }
              }}
              onKeyDown={(e) => {
                if (e.target !== e.currentTarget) return;
                if (e.key === "+" || e.key === "=") {
                  e.preventDefault();
                  applyZoomFactor(1.25);
                } else if (e.key === "-") {
                  e.preventDefault();
                  applyZoomFactor(1 / 1.25);
                } else if (e.key === "0") {
                  e.preventDefault();
                  resetView();
                } else if (e.key === "ArrowLeft") {
                  e.preventDefault();
                  setPan((p) => clampPan(p.x + 28, p.y, zoom));
                } else if (e.key === "ArrowRight") {
                  e.preventDefault();
                  setPan((p) => clampPan(p.x - 28, p.y, zoom));
                } else if (e.key === "ArrowUp") {
                  e.preventDefault();
                  setPan((p) => clampPan(p.x, p.y + 28, zoom));
                } else if (e.key === "ArrowDown") {
                  e.preventDefault();
                  setPan((p) => clampPan(p.x, p.y - 28, zoom));
                }
              }}
              onPointerDown={(e) => {
                if (zoom <= 1.0) return;
                dragRef.current = { active: true, lastX: e.clientX, lastY: e.clientY, moved: false };
              }}
              onPointerMove={(e) => {
                if (!dragRef.current.active) return;
                const dx = e.clientX - dragRef.current.lastX;
                const dy = e.clientY - dragRef.current.lastY;
                if (Math.abs(dx) > 2 || Math.abs(dy) > 2) dragRef.current.moved = true;
                dragRef.current.lastX = e.clientX;
                dragRef.current.lastY = e.clientY;
                setPan((p) => clampPan(p.x + dx, p.y + dy, zoom));
              }}
              onPointerUp={() => {
                dragRef.current.active = false;
              }}
              onTouchMove={(e) => {
                if (e.touches.length === 2) {
                  const t0 = e.touches[0];
                  const t1 = e.touches[1];
                  const dist = Math.hypot(t1.clientX - t0.clientX, t1.clientY - t0.clientY);
                  if (pinchDistRef.current !== null && pinchDistRef.current > 0) {
                    applyZoomFactor(dist / pinchDistRef.current);
                  }
                  pinchDistRef.current = dist;
                }
              }}
              onTouchEnd={() => {
                pinchDistRef.current = null;
              }}
            >
              <div className="map-zoom-controls" data-testid="map-zoom-controls">
                <button
                  type="button"
                  className="zoom-btn"
                  aria-label={EN.map.zoomInAria}
                  onClick={() => applyZoomFactor(1.3)}
                  disabled={zoom >= MAX_ZOOM}
                >
                  {EN.map.zoomInLabel}
                </button>
                <button
                  type="button"
                  className="zoom-btn"
                  aria-label={EN.map.zoomOutAria}
                  onClick={() => applyZoomFactor(1 / 1.3)}
                  disabled={zoom <= MIN_ZOOM}
                >
                  {EN.map.zoomOutLabel}
                </button>
                <button
                  type="button"
                  className="zoom-btn zoom-btn--fit"
                  aria-label={EN.map.fitSelectionAria}
                  onClick={fitToSelection}
                  data-testid="map-fit-selection-btn"
                >
                  {EN.map.fitSelectionLabel}
                </button>
                <button
                  type="button"
                  className="zoom-btn zoom-btn--reset"
                  onClick={resetView}
                  disabled={zoom <= MIN_ZOOM && pan.x === 0 && pan.y === 0}
                >
                  {EN.map.zoomResetLabel}
                </button>
                <span className="zoom-level-pill" data-testid="map-zoom-level">
                  {EN.map.zoomLevelText(zoom)}
                </span>
              </div>

              <svg
                className="vietnam-svg"
                viewBox={`0 0 ${SVG_W} ${SVG_H}`}
                role="img"
                aria-label={EN.map.svgAriaLabel}
              >
                <g
                  data-testid="map-zoom-viewport"
                  transform={`translate(${(SVG_W / 2 + pan.x).toFixed(2)} ${(SVG_H / 2 + pan.y).toFixed(2)}) scale(${zoom}) translate(${(-SVG_W / 2).toFixed(2)} ${(-SVG_H / 2).toFixed(2)})`}
                >
                  {showBoundaries && (
                    <g data-testid="map-boundaries-layer">
                      {features.map((feature) => {
                        const locId = feature.properties.location_id;
                        const isSupported = activeLocationsById.has(locId);
                        const isSelected = selectedSet.has(locId);
                        const isFocused = locId === focusedLocationId;
                        const statusLabel = isSupported ? EN.map.statusSupported : EN.map.statusUnsupported;
                        const ariaName = `${feature.properties.display_name}, ${statusLabel}`;

                        return (
                          <path
                            key={feature.properties.polygon_code}
                            d={featureToSvgPath(feature)}
                            vectorEffect="non-scaling-stroke"
                            tabIndex={0}
                            role="button"
                            aria-label={ariaName}
                            aria-pressed={isSelected}
                            data-location-id={locId}
                            data-supported={isSupported ? "true" : "false"}
                            className={[
                              "map-province",
                              isSupported ? "map-province--supported" : "map-province--unsupported",
                              isSelected ? "map-province--selected" : "",
                              isFocused ? "map-province--focused" : "",
                            ]
                              .filter(Boolean)
                              .join(" ")}
                            onMouseEnter={() => setHoveredFeature(feature)}
                            onMouseLeave={() =>
                              setHoveredFeature((prev) =>
                                prev?.properties.polygon_code === feature.properties.polygon_code ? null : prev
                              )
                            }
                            onFocus={() => setHoveredFeature(feature)}
                            onClick={() => {
                              if (dragRef.current.moved) {
                                dragRef.current.moved = false;
                                return;
                              }
                              setHoveredFeature(feature);
                              onToggleLocation(locId);
                            }}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                setHoveredFeature(feature);
                                onToggleLocation(locId);
                              }
                            }}
                          >
                            <title>{ariaName}</title>
                          </path>
                        );
                      })}
                    </g>
                  )}

                  {showGridCells && (
                    <g data-testid="map-grid-layer" className="map-grid-layer">
                      {MERRA2_LATTICE.map((cell) => {
                        const [x1, y1] = projectCoord(cell.lonMin, cell.latMax);
                        const [x2, y2] = projectCoord(cell.lonMax, cell.latMin);
                        const isHighlighted = highlightedCellKeys.has(cell.key);
                        const isModelCell = cell.key === HCMC_CELL_KEY;
                        return (
                          <rect
                            key={cell.key}
                            x={x1}
                            y={y1}
                            width={Math.max(1, Number((x2 - x1).toFixed(2)))}
                            height={Math.max(1, Number((y2 - y1).toFixed(2)))}
                            vectorEffect="non-scaling-stroke"
                            data-grid-highlighted={isHighlighted ? "true" : "false"}
                            className={[
                              "map-grid-lattice-cell",
                              isHighlighted ? "map-grid-lattice-cell--active" : "",
                              isModelCell ? "map-grid-lattice-cell--model" : "",
                            ]
                              .filter(Boolean)
                              .join(" ")}
                          >
                            {isHighlighted && (
                              <title>
                                {`${EN.map.hoverGridHint}: [${cell.latMin.toFixed(2)}N, ${cell.latMax.toFixed(2)}N] x [${cell.lonMin.toFixed(4)}E, ${cell.lonMax.toFixed(4)}E]`}
                              </title>
                            )}
                          </rect>
                        );
                      })}
                    </g>
                  )}

                  {activeLocationsList.map((loc) => {
                    const [mx, my] = projectCoord(loc.longitude, loc.latitude);
                    return (
                      <g key={loc.location_id} className="map-marker-group">
                        <circle cx={mx} cy={my} r={5.5 / zoom} className="map-marker-halo" />
                        <circle cx={mx} cy={my} r={3.2 / zoom} className="map-marker-dot" />
                        <title>{`${loc.display_name} (${loc.latitude.toFixed(4)}N, ${loc.longitude.toFixed(4)}E)`}</title>
                      </g>
                    );
                  })}
                </g>
              </svg>

              {hoveredFeature && (
                <div className="map-tooltip" role="status" data-testid="map-tooltip">
                  <strong>{hoveredFeature.properties.display_name}</strong>
                  {" — "}
                  <span>
                    {activeLocationsById.has(hoveredFeature.properties.location_id)
                      ? EN.map.statusAvailable
                      : EN.map.statusNotAvailableYet}
                  </span>
                </div>
              )}
            </div>
          )}

          <div className="map-marker-note">
            <div>
              <strong>{EN.map.markerLabel}:</strong> {EN.map.markerNote}
            </div>
            <div className="muted-inline">{EN.map.gridCellVerifiedDetail}</div>
            <div className="muted-inline">{EN.map.gridFormulaNote}</div>
          </div>

          <div className="map-legend" aria-label={EN.map.legendTitle}>
            <span className="legend-item">
              <span className="legend-swatch legend-swatch--supported" />
              {EN.map.legendSupported}
            </span>
            <span className="legend-item">
              <span className="legend-swatch legend-swatch--unsupported" />
              {EN.map.legendUnsupported}
            </span>
            <span className="legend-item">
              <span className="legend-swatch legend-swatch--grid" />
              {EN.map.legendHighlightedCell}
            </span>
            <div className="legend-meta">
              <div>{EN.map.legendVersion}</div>
              <div>{EN.map.attributionSummary}</div>
            </div>
          </div>
        </div>

        <div className="map-sidebar-col">
          <div className="selected-locations-box" data-testid="selected-locations-table">
            <div className="selected-header-row">
              <div className="selected-title-group">
                <span className="list-caption">{EN.map.selectedTableHeading}</span>
                <span className="pill-badge pill-badge--active" data-testid="selected-count-badge">
                  {EN.map.selectedCountBadge(selectedLocationIds.length)}
                </span>
              </div>
              {onClearSelection && (
                <button
                  type="button"
                  className="btn-chip"
                  onClick={onClearSelection}
                >
                  {EN.map.clearSelectionBtn}
                </button>
              )}
            </div>
            <div className="selected-table-wrap">
              <div
                className="selected-table-scroll"
                ref={selectedScrollRef}
                onScroll={recomputeScrollStats}
              >
                <table className="data-table data-table--compact sticky-head-table">
                  <thead>
                    <tr>
                      <th>{EN.map.selectedColLocation}</th>
                      <th>{EN.map.selectedColStatus}</th>
                      <th className="num">{EN.map.selectedColCells}</th>
                      <th aria-label={EN.map.clearSelectionBtn} />
                    </tr>
                  </thead>
                  <tbody>
                    {selectedLocationIds.map((locId) => {
                      const feat = featureByLocId.get(locId);
                      const disp = feat?.properties.display_name ?? activeLocationsById.get(locId)?.display_name ?? locId;
                      const isAvail = activeLocationsById.has(locId);
                      const cellCount = cellKeysByLocId.get(locId)?.size ?? 0;
                      return (
                        <tr
                          key={locId}
                          className={locId === focusedLocationId ? "row-focused" : ""}
                        >
                          <td>{disp}</td>
                          <td>
                            <span className={`pill-badge ${isAvail ? "pill-badge--active" : ""}`}>
                              {isAvail ? EN.map.statusAvailable : EN.map.statusNotAvailableYet}
                            </span>
                          </td>
                          <td className="num">{cellCount}</td>
                          <td className="num">
                            <button
                              type="button"
                              className="deselect-x-btn"
                              aria-label={EN.map.deselectBtnLabel(disp)}
                              onClick={() => (onRemoveLocation ? onRemoveLocation(locId) : onToggleLocation(locId))}
                            >
                              {EN.map.deselectSymbol}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {scrollStats.hiddenBelow > 0 && (
                <div className="selected-scroll-more-hint" data-testid="selected-scroll-more-badge">
                  {EN.map.selectedScrollMore(scrollStats.hiddenBelow, scrollStats.visible, scrollStats.total)}
                </div>
              )}
            </div>
          </div>

          <div className="search-field-group">
            <label htmlFor="province-search-input">{EN.map.searchLabel}</label>
            <input
              id="province-search-input"
              type="search"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={EN.map.searchPlaceholder}
            />
          </div>

          <div className="province-list-wrap">
            <div className="list-caption">{EN.map.listHeading}</div>
            {filteredFeatures.length === 0 ? (
              <div className="empty-box">{EN.map.noSearchMatches}</div>
            ) : (
              <ul className="province-list" aria-label={EN.map.listHeading}>
                {filteredFeatures.map((feature) => {
                  const locId = feature.properties.location_id;
                  const isSupported = activeLocationsById.has(locId);
                  const isSelected = selectedSet.has(locId);
                  return (
                    <li key={feature.properties.polygon_code}>
                      <button
                        type="button"
                        className={[
                          "province-list-btn",
                          isSelected ? "province-list-btn--selected" : "",
                        ]
                          .filter(Boolean)
                          .join(" ")}
                        onClick={() => {
                          setHoveredFeature(feature);
                          onToggleLocation(locId);
                        }}
                      >
                        <span>{feature.properties.display_name}</span>
                        <span className={`pill-badge ${isSupported ? "pill-badge--active" : ""}`}>
                          {isSupported ? EN.map.statusAvailable : EN.map.statusNotAvailableYet}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};
