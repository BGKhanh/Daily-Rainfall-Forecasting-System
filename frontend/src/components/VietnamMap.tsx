import React, { useMemo, useState } from "react";
import { EN, normalizeAccentInsensitive } from "../i18n/en";
import type { LocationItem, ProvinceFeature, ProvinceGeoJson } from "../types/api";

interface VietnamMapProps {
  geoJson: ProvinceGeoJson | null;
  mapError: string | null;
  locations: LocationItem[];
  selectedLocationId: string;
  onSelectSupportedLocation: (locationId: string) => void;
}

// Bounding box covering mainland Vietnam (63 pre-merger provinces) + Hoang Sa Island + Truong Sa Island
const LON_MIN = 101.8;
const LON_MAX = 117.3;
const LAT_MIN = 6.6;
const LAT_MAX = 23.6;
const COS_MID_LAT = Math.cos((15.0 * Math.PI) / 180);
const SVG_W = 540;
const SVG_H = Math.round(SVG_W * ((LAT_MAX - LAT_MIN) / ((LON_MAX - LON_MIN) * COS_MID_LAT)));

// Empirically verified MERRA-2 0.5 deg x 0.625 deg grid cell around HCMC (10.7769N, 106.7009E)
const VERIFIED_GRID_CELL = {
  latMin: 10.75,
  latMax: 11.25,
  lonMin: 106.5625,
  lonMax: 107.1875,
};

function projectCoord(lon: number, lat: number): [number, number] {
  const x = ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * SVG_W;
  const y = SVG_H - ((lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * SVG_H;
  return [Number(x.toFixed(2)), Number(y.toFixed(2))];
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
  selectedLocationId,
  onSelectSupportedLocation,
}) => {
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [hoveredFeature, setHoveredFeature] = useState<ProvinceFeature | null>(null);
  const [unsupportedProvince, setUnsupportedProvince] = useState<string | null>(null);

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

  const filteredFeatures = useMemo(() => {
    const q = normalizeAccentInsensitive(searchQuery);
    if (!q) return features;
    return features.filter((f) => {
      const nameNorm = normalizeAccentInsensitive(f.properties.display_name);
      const idNorm = normalizeAccentInsensitive(f.properties.location_id);
      return nameNorm.includes(q) || idNorm.includes(q);
    });
  }, [features, searchQuery]);

  const handleActivateFeature = (feature: ProvinceFeature) => {
    setHoveredFeature(feature);
    const activeLoc = activeLocationsById.get(feature.properties.location_id);
    if (activeLoc) {
      setUnsupportedProvince(null);
      onSelectSupportedLocation(activeLoc.location_id);
    } else {
      setUnsupportedProvince(feature.properties.display_name);
    }
  };

  const gridTopLeft = projectCoord(VERIFIED_GRID_CELL.lonMin, VERIFIED_GRID_CELL.latMax);
  const gridBottomRight = projectCoord(VERIFIED_GRID_CELL.lonMax, VERIFIED_GRID_CELL.latMin);

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
      </div>

      {unsupportedProvince && (
        <div className="info-banner" role="status" data-testid="unsupported-location-banner">
          <strong>{EN.map.unsupportedNotice(unsupportedProvince)}</strong>{" "}
          <span>{EN.map.unsupportedActionHint}</span>
        </div>
      )}

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
                {features.map((feature) => {
                  const locId = feature.properties.location_id;
                  const isSupported = activeLocationsById.has(locId);
                  const isSelected = locId === selectedLocationId && isSupported;
                  const statusLabel = isSupported ? EN.map.statusSupported : EN.map.statusUnsupported;
                  const ariaName = `${feature.properties.display_name}, ${statusLabel}`;

                  return (
                    <path
                      key={feature.properties.polygon_code}
                      d={featureToSvgPath(feature)}
                      tabIndex={0}
                      role="button"
                      aria-label={ariaName}
                      data-location-id={locId}
                      data-supported={isSupported ? "true" : "false"}
                      className={[
                        "map-province",
                        isSupported ? "map-province--supported" : "map-province--unsupported",
                        isSelected ? "map-province--selected" : "",
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
                      onClick={() => handleActivateFeature(feature)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          handleActivateFeature(feature);
                        }
                      }}
                    >
                      <title>{ariaName}</title>
                    </path>
                  );
                })}

                {activeLocationsList.some((l) => l.location_id === "HCMC") && (
                  <rect
                    x={gridTopLeft[0]}
                    y={gridTopLeft[1]}
                    width={Math.max(4, gridBottomRight[0] - gridTopLeft[0])}
                    height={Math.max(4, gridBottomRight[1] - gridTopLeft[1])}
                    className="map-grid-cell-box"
                    aria-label={EN.map.legendGridCell}
                  >
                    <title>{EN.map.gridCellVerifiedDetail}</title>
                  </rect>
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
                      ? EN.map.statusSupported
                      : EN.map.statusUnsupported}
                  </span>
                </div>
              )}
            </div>
          )}

          <div className="map-marker-note">
            <strong>{EN.map.markerLabel}:</strong> {EN.map.markerNote}{" "}
            <span className="muted-inline">{EN.map.gridCellVerifiedDetail}</span>
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
              {EN.map.legendGridCell}
            </span>
            <div className="legend-meta">
              <div>{EN.map.legendVersion}</div>
              <div>{EN.map.attributionSummary}</div>
            </div>
          </div>
        </div>

        <div className="map-sidebar-col">
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

          <div className="province-list-header">{EN.map.listHeading}</div>
          <div className="province-list" role="list" aria-label={EN.map.listHeading}>
            {filteredFeatures.length === 0 ? (
              <div className="empty-box">{EN.map.noSearchMatches}</div>
            ) : (
              filteredFeatures.map((feature) => {
                const locId = feature.properties.location_id;
                const isSupported = activeLocationsById.has(locId);
                const isSelected = locId === selectedLocationId && isSupported;
                const statusText = isSupported ? EN.map.statusSupported : EN.map.statusUnsupported;
                return (
                  <button
                    key={feature.properties.polygon_code}
                    type="button"
                    className={[
                      "province-list-btn",
                      isSupported ? "province-list-btn--supported" : "",
                      isSelected ? "province-list-btn--selected" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    aria-label={`${feature.properties.display_name}, ${statusText}`}
                    onClick={() => handleActivateFeature(feature)}
                    onFocus={() => setHoveredFeature(feature)}
                  >
                    <span className="province-name">{feature.properties.display_name}</span>
                    <span className={`province-pill ${isSupported ? "pill-supported" : "pill-unsupported"}`}>
                      {statusText}
                    </span>
                  </button>
                );
              })
            )}
          </div>
        </div>
      </div>
    </section>
  );
};
