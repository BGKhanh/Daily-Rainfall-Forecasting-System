import React, { useMemo, useState } from "react";
import { EN, stripDiacriticsForSearch } from "../i18n/en";
import type { LocationItem, ProvinceFeature, ProvinceGeoJson } from "../types/api";

interface VietnamMapProps {
  geoJson: ProvinceGeoJson | null;
  mapError: string | null;
  locations: LocationItem[];
  selectedLocationIds: string[];
  focusedLocationId: string;
  onToggleLocation: (locationId: string) => void;
  onResetSelection: () => void;
}

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
  onResetSelection,
}) => {
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [hoveredFeature, setHoveredFeature] = useState<ProvinceFeature | null>(null);
  const [showBoundaries, setShowBoundaries] = useState<boolean>(true);
  const [showGridCells, setShowGridCells] = useState<boolean>(true);

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
          {selectedLocationIds.length > 1 || focusedLocationId !== "HCMC" ? (
            <button type="button" className="btn-chip" onClick={onResetSelection}>
              {EN.map.resetSelectionBtn}
            </button>
          ) : null}
        </div>
      </div>

      <div className="map-workspace">
        <div className="map-canvas-col">
          {mapError ? (
            <div className="info-banner">{mapError}</div>
          ) : !geoJson ? (
            <div className="empty-box">{EN.map.loadingMap}</div>
          ) : (
            <div className="map-svg-wrap">
              <svg
                className="vietnam-svg"
                viewBox={`0 0 ${SVG_W} ${SVG_H}`}
                role="img"
                aria-label={EN.map.svgAriaLabel}
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
                      <circle cx={mx} cy={my} r={5.5} className="map-marker-halo" />
                      <circle cx={mx} cy={my} r={3.2} className="map-marker-dot" />
                      <title>{`${loc.display_name} (${loc.latitude.toFixed(4)}N, ${loc.longitude.toFixed(4)}E)`}</title>
                    </g>
                  );
                })}
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
            <div className="list-caption">{EN.map.selectedTableHeading}</div>
            <div className="table-scroll">
              <table className="data-table data-table--compact">
                <thead>
                  <tr>
                    <th>{EN.map.selectedColLocation}</th>
                    <th>{EN.map.selectedColStatus}</th>
                    <th className="num">{EN.map.selectedColCells}</th>
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
                      </tr>
                    );
                  })}
                </tbody>
              </table>
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
