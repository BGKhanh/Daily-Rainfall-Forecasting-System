import React, { useEffect, useMemo, useState } from "react";
import variableCatalogRaw from "../data/variableCatalog.json";
import { EN } from "../i18n/en";
import type {
  ProvinceFeature,
  ProvinceGeoJson,
  ProvinceGridOverlapData,
  VariableCatalogData,
} from "../types/api";

interface GridScalePanelProps {
  geoJson: ProvinceGeoJson | null;
  gridOverlap: ProvinceGridOverlapData | null;
  focusedLocationId: string;
}

const VARIABLE_CATALOG = variableCatalogRaw as VariableCatalogData;

function getBboxFromFeatureAndCells(
  feature: ProvinceFeature | undefined,
  merraCells: Array<{ lat_min: number; lat_max: number; lon_min: number; lon_max: number }>,
  ceresCells: Array<{ lat_min: number; lat_max: number; lon_min: number; lon_max: number }>
): { latMin: number; latMax: number; lonMin: number; lonMax: number } {
  let latMin = Infinity;
  let latMax = -Infinity;
  let lonMin = Infinity;
  let lonMax = -Infinity;

  const pushPt = (lon: number, lat: number) => {
    if (lon < lonMin) lonMin = lon;
    if (lon > lonMax) lonMax = lon;
    if (lat < latMin) latMin = lat;
    if (lat > latMax) latMax = lat;
  };

  if (feature) {
    const polys =
      feature.geometry.type === "Polygon"
        ? [feature.geometry.coordinates as number[][][]]
        : (feature.geometry.coordinates as number[][][][]);
    for (const poly of polys) {
      for (const ring of poly) {
        for (const [lo, la] of ring) {
          pushPt(lo, la);
        }
      }
    }
  }

  for (const c of [...merraCells, ...ceresCells]) {
    pushPt(c.lon_min, c.lat_min);
    pushPt(c.lon_max, c.lat_max);
  }

  if (!Number.isFinite(latMin)) {
    return { latMin: 9.8, latMax: 11.4, lonMin: 105.8, lonMax: 107.4 };
  }

  const dLat = Math.max(0.2, (latMax - latMin) * 0.08);
  const dLon = Math.max(0.2, (lonMax - lonMin) * 0.08);
  return {
    latMin: latMin - dLat,
    latMax: latMax + dLat,
    lonMin: lonMin - dLon,
    lonMax: lonMax + dLon,
  };
}

function projectRingsToPath(
  feature: ProvinceFeature,
  project: (lon: number, lat: number) => [number, number]
): string {
  const polys =
    feature.geometry.type === "Polygon"
      ? [feature.geometry.coordinates as number[][][]]
      : (feature.geometry.coordinates as number[][][][]);

  const segs: string[] = [];
  for (const poly of polys) {
    for (const ring of poly) {
      if (!ring.length) continue;
      const cmds = ring.map(([lon, lat], idx) => {
        const [x, y] = project(lon, lat);
        return `${idx === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      });
      segs.push(`${cmds.join("")}Z`);
    }
  }
  return segs.join(" ");
}

export const GridScalePanel: React.FC<GridScalePanelProps> = ({
  geoJson,
  gridOverlap,
  focusedLocationId,
}) => {
  const [selectedLocId, setSelectedLocId] = useState<string>(focusedLocationId || "HCMC");
  const [showOverlapPct, setShowOverlapPct] = useState<boolean>(false);

  useEffect(() => {
    if (focusedLocationId) {
      setSelectedLocId(focusedLocationId);
    }
  }, [focusedLocationId]);

  const sortedUnits = useMemo(() => {
    const features = geoJson?.features ?? [];
    if (features.length > 0) {
      return [...features]
        .map((f) => ({
          location_id: f.properties.location_id,
          display_name: f.properties.display_name,
        }))
        .sort((a, b) => a.display_name.localeCompare(b.display_name));
    }
    if (gridOverlap?.units) {
      return Object.values(gridOverlap.units)
        .map((u) => ({ location_id: u.location_id, display_name: u.display_name }))
        .sort((a, b) => a.display_name.localeCompare(b.display_name));
    }
    return [{ location_id: "HCMC", display_name: EN.forecast.defaultLocationName }];
  }, [geoJson, gridOverlap]);

  const unitOverlap = gridOverlap?.units?.[selectedLocId] ?? null;
  const feature = useMemo(
    () => (geoJson?.features ?? []).find((f) => f.properties.location_id === selectedLocId),
    [geoJson, selectedLocId]
  );

  const svgGeometry = useMemo(() => {
    const merraCells = unitOverlap?.merra2_grid.intersecting_cells ?? [];
    const ceresCells = unitOverlap?.ceres_grid.intersecting_cells ?? [];
    const bbox = getBboxFromFeatureAndCells(feature, merraCells, ceresCells);
    const midLatRad = (((bbox.latMin + bbox.latMax) / 2) * Math.PI) / 180;
    const cosPhi = Math.max(0.25, Math.cos(midLatRad));
    const spanX = (bbox.lonMax - bbox.lonMin) * cosPhi;
    const spanY = bbox.latMax - bbox.latMin;

    const svgW = 460;
    const svgH = Math.max(240, Math.min(440, Math.round(svgW * (spanY / Math.max(0.1, spanX)))));
    const project = (lon: number, lat: number): [number, number] => [
      ((lon - bbox.lonMin) / Math.max(1e-6, bbox.lonMax - bbox.lonMin)) * svgW,
      svgH - ((lat - bbox.latMin) / Math.max(1e-6, bbox.latMax - bbox.latMin)) * svgH,
    ];

    return {
      svgW,
      svgH,
      cosPhi,
      polyPath: feature ? projectRingsToPath(feature, project) : "",
      merraCells,
      ceresCells,
      project,
    };
  }, [feature, unitOverlap]);

  const merraVars = useMemo(
    () => VARIABLE_CATALOG.variables.filter((v) => v.grid_id === "MERRA2_GEOSIT_0.5x0.625"),
    []
  );
  const ceresVars = useMemo(
    () => VARIABLE_CATALOG.variables.filter((v) => v.grid_id === "CERES_FLASHFLUX_1.0x1.0"),
    []
  );

  const combinedCells = useMemo(() => {
    if (!unitOverlap) return [];
    return [
      ...unitOverlap.merra2_grid.intersecting_cells.map((c) => ({
        gridLabel: "MERRA-2 / GEOS-IT (0.5° × 0.625°)",
        ...c,
      })),
      ...unitOverlap.ceres_grid.intersecting_cells.map((c) => ({
        gridLabel: "CERES / FLASHFlux (1.0° × 1.0°)",
        ...c,
      })),
    ];
  }, [unitOverlap]);

  const displayName =
    unitOverlap?.display_name ?? feature?.properties.display_name ?? selectedLocId;

  return (
    <section
      className="panel grid-scale-panel"
      aria-label={EN.gridScale.heading}
      data-testid="grid-scale-panel"
    >
      <div className="panel-header-row">
        <div>
          <h2>{EN.gridScale.heading}</h2>
          <p className="panel-subtitle">{EN.gridScale.subtitle}</p>
        </div>

        <div className="grid-scale-controls">
          <div className="grid-select-wrap">
            <label htmlFor="grid-scale-province-select">{EN.gridScale.provinceSelectLabel}</label>
            <select
              id="grid-scale-province-select"
              aria-label={EN.gridScale.provinceSelectAria}
              value={selectedLocId}
              onChange={(e) => setSelectedLocId(e.target.value)}
            >
              {sortedUnits.map((u) => (
                <option key={u.location_id} value={u.location_id}>
                  {u.display_name}
                </option>
              ))}
            </select>
          </div>
          <label className="layer-toggle-label">
            <input
              type="checkbox"
              checked={showOverlapPct}
              onChange={(e) => setShowOverlapPct(e.target.checked)}
            />
            <span>{EN.gridScale.showOverlapToggle}</span>
          </label>
        </div>
      </div>

      {!unitOverlap ? (
        <div className="empty-box">{EN.gridScale.loadingOverlap}</div>
      ) : (
        <>
          <div className="grid-scale-workspace">
            <div className="grid-overlay-svg-col">
              <svg
                className="grid-overlay-svg"
                viewBox={`0 0 ${svgGeometry.svgW} ${svgGeometry.svgH}`}
                role="img"
                aria-label={EN.gridScale.svgAriaLabel(displayName)}
                data-testid="grid-scale-svg"
              >
                {svgGeometry.ceresCells.map((cell) => {
                  const [x1, y1] = svgGeometry.project(cell.lon_min, cell.lat_max);
                  const [x2, y2] = svgGeometry.project(cell.lon_max, cell.lat_min);
                  return (
                    <rect
                      key={cell.cell_id}
                      data-grid-type="ceres"
                      x={x1}
                      y={y1}
                      width={Math.max(1, x2 - x1)}
                      height={Math.max(1, y2 - y1)}
                      vectorEffect="non-scaling-stroke"
                      className={[
                        "overlay-ceres-cell",
                        cell.contains_representative_point ? "overlay-ceres-cell--rep" : "",
                      ]
                        .filter(Boolean)
                        .join(" ")}
                    >
                      <title>{`CERES 1.0x1.0 [${cell.lat_min}N..${cell.lat_max}N] x [${cell.lon_min}E..${cell.lon_max}E] (${cell.pct_of_province_area.toFixed(1)}%)`}</title>
                    </rect>
                  );
                })}

                {svgGeometry.merraCells.map((cell) => {
                  const [x1, y1] = svgGeometry.project(cell.lon_min, cell.lat_max);
                  const [x2, y2] = svgGeometry.project(cell.lon_max, cell.lat_min);
                  return (
                    <rect
                      key={cell.cell_id}
                      data-grid-type="merra2"
                      x={x1}
                      y={y1}
                      width={Math.max(1, x2 - x1)}
                      height={Math.max(1, y2 - y1)}
                      vectorEffect="non-scaling-stroke"
                      className={[
                        "overlay-merra-cell",
                        cell.contains_representative_point ? "overlay-merra-cell--rep" : "",
                      ]
                        .filter(Boolean)
                        .join(" ")}
                    >
                      <title>{`MERRA-2 0.5x0.625 [${cell.lat_min}N..${cell.lat_max}N] x [${cell.lon_min}E..${cell.lon_max}E] (${cell.pct_of_province_area.toFixed(1)}%)`}</title>
                    </rect>
                  );
                })}

                {svgGeometry.polyPath && (
                  <path
                    d={svgGeometry.polyPath}
                    vectorEffect="non-scaling-stroke"
                    className="overlay-province-polygon"
                  />
                )}

                {(() => {
                  const [rx, ry] = svgGeometry.project(
                    unitOverlap.representative_point.lon,
                    unitOverlap.representative_point.lat
                  );
                  return (
                    <g>
                      <circle cx={rx} cy={ry} r={6} className="map-marker-halo" />
                      <circle cx={rx} cy={ry} r={3.5} className="map-marker-dot" />
                    </g>
                  );
                })()}
              </svg>

              <div className="map-legend">
                <span className="legend-item">
                  <span className="legend-swatch legend-swatch--poly-outline" />
                  {EN.gridScale.legendProvince}
                </span>
                <span className="legend-item">
                  <span className="legend-swatch legend-swatch--merra-rep" />
                  {EN.gridScale.legendMerraRep}
                </span>
                <span className="legend-item">
                  <span className="legend-swatch legend-swatch--merra-all" />
                  {EN.gridScale.legendMerraAll}
                </span>
                <span className="legend-item">
                  <span className="legend-swatch legend-swatch--ceres-all" />
                  {EN.gridScale.legendCeresAll}
                </span>
              </div>
            </div>

            <div className="grid-metrics-col">
              <div className="metric-grid">
                <div className="metric-card">
                  <div className="metric-label">{EN.gridScale.metricLatitudeLabel}</div>
                  <div className="metric-value">
                    {`${unitOverlap.representative_point.lat.toFixed(4)}°N · ${unitOverlap.polygon_area_km2.toFixed(1)} ${EN.gridScale.metricAreaUnit}`}
                  </div>
                  <div className="metric-sub">
                    {`cos(φ) = ${Math.cos((unitOverlap.representative_point.lat * Math.PI) / 180).toFixed(4)}`}
                  </div>
                </div>

                <div className="metric-card">
                  <div className="metric-label">{EN.gridScale.metricMerraLabel}</div>
                  <div className="metric-value">
                    {`${unitOverlap.merra2_grid.cell_dimensions_km_at_phi.height_km} × ${unitOverlap.merra2_grid.cell_dimensions_km_at_phi.width_km} km`}
                  </div>
                  <div className="metric-sub">
                    {`${unitOverlap.merra2_grid.cell_dimensions_km_at_phi.area_km2.toFixed(0)} ${EN.gridScale.metricAreaUnit} · ${unitOverlap.merra2_grid.intersecting_cell_count} cells`}
                  </div>
                </div>

                <div className="metric-card">
                  <div className="metric-label">{EN.gridScale.metricCeresLabel}</div>
                  <div className="metric-value">
                    {`${unitOverlap.ceres_grid.cell_dimensions_km_at_phi.height_km} × ${unitOverlap.ceres_grid.cell_dimensions_km_at_phi.width_km} km`}
                  </div>
                  <div className="metric-sub">
                    {`${unitOverlap.ceres_grid.cell_dimensions_km_at_phi.area_km2.toFixed(0)} ${EN.gridScale.metricAreaUnit} · ${unitOverlap.ceres_grid.intersecting_cell_count} cells`}
                  </div>
                </div>
              </div>

              {showOverlapPct && (
                <div className="subpanel overlap-breakdown-panel" data-testid="overlap-breakdown-table">
                  <h4>{EN.gridScale.overlapTableHeading}</h4>
                  <div className="table-scroll overlap-table-scroll">
                    <table className="data-table data-table--compact">
                      <thead>
                        <tr>
                          <th>{EN.gridScale.overlapColGrid}</th>
                          <th>{EN.gridScale.overlapColBounds}</th>
                          <th className="num">{EN.gridScale.overlapColInterKm2}</th>
                          <th className="num">{EN.gridScale.overlapColPctProv}</th>
                          <th className="num">{EN.gridScale.overlapColPctCell}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {combinedCells.map((c) => (
                          <tr key={c.cell_id}>
                            <td>
                              {c.gridLabel}
                              {c.contains_representative_point && (
                                <span className="pill-badge pill-badge--active inline-badge">
                                  {EN.gridScale.repYesBadge}
                                </span>
                              )}
                            </td>
                            <td className="mono">
                              {`[${c.lat_min.toFixed(2)}, ${c.lat_max.toFixed(2)}] × [${c.lon_min.toFixed(4)}, ${c.lon_max.toFixed(4)}]`}
                            </td>
                            <td className="num">{c.intersection_area_km2.toFixed(1)}</td>
                            <td className="num">{`${c.pct_of_province_area.toFixed(2)}%`}</td>
                            <td className="num">{`${c.pct_of_cell_area.toFixed(2)}%`}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <div className="subpanel timeline-notes-panel">
                <h4>{EN.gridScale.timelineHeading}</h4>
                <ul className="timeline-list">
                  <li>{EN.gridScale.timelineCeres}</li>
                  <li>{EN.gridScale.timelineMerra}</li>
                  <li>{EN.gridScale.timelineFaqNote}</li>
                </ul>
              </div>
            </div>
          </div>

          <details className="catalog-details" data-testid="variable-catalog-section">
            <summary>{EN.gridScale.catalogHeading}</summary>
            <div className="catalog-groups">
              <div className="subpanel">
                <h4>{EN.gridScale.catalogGroupMerra}</h4>
                <div className="table-scroll">
                  <table className="data-table data-table--compact">
                    <thead>
                      <tr>
                        <th>{EN.gridScale.catalogHeaders.code}</th>
                        <th>{EN.gridScale.catalogHeaders.name}</th>
                        <th>{EN.gridScale.catalogHeaders.unit}</th>
                        <th>{EN.gridScale.catalogHeaders.finalized}</th>
                        <th>{EN.gridScale.catalogHeaders.provisional}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {merraVars.map((v) => (
                        <tr key={v.code}>
                          <td className="mono">{v.code}</td>
                          <td>{v.label_en}</td>
                          <td>{v.unit}</td>
                          <td>{v.source_finalized}</td>
                          <td>{v.source_provisional}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="subpanel">
                <h4>{EN.gridScale.catalogGroupCeres}</h4>
                <div className="table-scroll">
                  <table className="data-table data-table--compact">
                    <thead>
                      <tr>
                        <th>{EN.gridScale.catalogHeaders.code}</th>
                        <th>{EN.gridScale.catalogHeaders.name}</th>
                        <th>{EN.gridScale.catalogHeaders.unit}</th>
                        <th>{EN.gridScale.catalogHeaders.finalized}</th>
                        <th>{EN.gridScale.catalogHeaders.provisional}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ceresVars.map((v) => (
                        <tr key={v.code}>
                          <td className="mono">{v.code}</td>
                          <td>{v.label_en}</td>
                          <td>{v.unit}</td>
                          <td>{v.source_finalized}</td>
                          <td>{v.source_provisional}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </details>
        </>
      )}
    </section>
  );
};
