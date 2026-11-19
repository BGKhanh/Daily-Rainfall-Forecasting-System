import React, { useMemo, useState } from "react";
import { EN, formatReadableDate, mapApiErrorToFriendly, type FriendlyErrorView } from "../i18n/en";
import type { ApiClientError } from "../api/client";
import type {
  HealthResponse,
  LocationItem,
  ObservationSeriesPoint,
  PredictResponse,
  ProvinceGeoJson,
} from "../types/api";
import { VietnamMap } from "./VietnamMap";
import { computeTicksMultiplesOf5 } from "../utils/chart";

interface AnchorForecastSectionProps {
  health: HealthResponse | null;
  locations: LocationItem[];
  selectedLocationIds: string[];
  focusedLocationId: string;
  onToggleLocation: (locationId: string) => void;
  onResetSelection: () => void;
  geoJson: ProvinceGeoJson | null;
  mapError: string | null;
  prediction: PredictResponse | null;
  loading: boolean;
  errorValue: unknown;
  priorObservations: ObservationSeriesPoint[];
  showPrior14d: boolean;
  onTogglePrior14d: (enabled: boolean) => void;
}

interface TimelinePoint {
  date: string;
  observedMm: number | null;
  forecastMm: number | null;
  statusLabel: string;
  isPriorContext: boolean;
}

const OBSERVED_COLOR = "#0f1e2e";
const FORECAST_COLOR = "#7c3aed";

function buildPolylinePath(points: Array<{ x: number; y: number } | null>): string {
  let d = "";
  let inRun = false;
  for (const pt of points) {
    if (!pt) {
      inRun = false;
      continue;
    }
    d += inRun ? ` L${pt.x.toFixed(1)},${pt.y.toFixed(1)}` : ` M${pt.x.toFixed(1)},${pt.y.toFixed(1)}`;
    inRun = true;
  }
  return d.trim();
}

export const AnchorForecastSection: React.FC<AnchorForecastSectionProps> = ({
  health,
  locations,
  selectedLocationIds,
  focusedLocationId,
  onToggleLocation,
  onResetSelection,
  geoJson,
  mapError,
  prediction,
  loading,
  errorValue,
  priorObservations,
  showPrior14d,
  onTogglePrior14d,
}) => {
  const [activePointIdx, setActivePointIdx] = useState<number | null>(null);

  const activeLocationsById = useMemo(() => {
    const map = new Map<string, LocationItem>();
    for (const loc of locations) {
      if (loc.is_active) map.set(loc.location_id, loc);
    }
    return map;
  }, [locations]);

  const focusedFeature = useMemo(
    () => (geoJson?.features ?? []).find((f) => f.properties.location_id === focusedLocationId),
    [geoJson, focusedLocationId]
  );

  const focusedDisplayName =
    focusedFeature?.properties.display_name ??
    activeLocationsById.get(focusedLocationId)?.display_name ??
    focusedLocationId;

  const isFocusedSupported = activeLocationsById.has(focusedLocationId);

  const friendlyError: FriendlyErrorView | null = useMemo(() => {
    if (!errorValue) return null;
    const errObj = errorValue as Partial<ApiClientError>;
    return mapApiErrorToFriendly({
      code: errObj.code,
      message: errObj.message,
      details: errObj.details,
      status: errObj.status,
    });
  }, [errorValue]);

  const dailyForecasts = !friendlyError && prediction ? prediction.daily_forecasts : [];

  const timelinePoints: TimelinePoint[] = useMemo(() => {
    const pts: TimelinePoint[] = [];
    const anchorLimit = prediction?.anchor_date ?? "";
    if (showPrior14d && priorObservations.length > 0) {
      for (const p of priorObservations) {
        if (anchorLimit && p.date > anchorLimit) continue;
        pts.push({
          date: p.date,
          observedMm: p.precip_mm,
          forecastMm: null,
          statusLabel: EN.forecast.statusLabels.past_observed,
          isPriorContext: true,
        });
      }
    }
    for (const d of dailyForecasts) {
      pts.push({
        date: d.target_date,
        observedMm: d.observed_mm,
        forecastMm: d.predictions["SC-35"],
        statusLabel: EN.forecast.statusLabels[d.status] ?? d.status,
        isPriorContext: false,
      });
    }
    return pts;
  }, [priorObservations, showPrior14d, dailyForecasts, prediction?.anchor_date]);

  const maxChartVal = useMemo(() => {
    let maxV = 0;
    for (const pt of timelinePoints) {
      if (pt.observedMm !== null && pt.observedMm > maxV) maxV = pt.observedMm;
      if (pt.forecastMm !== null && pt.forecastMm > maxV) maxV = pt.forecastMm;
    }
    return maxV;
  }, [timelinePoints]);

  const yTicks = useMemo(() => computeTicksMultiplesOf5(maxChartVal), [maxChartVal]);
  const yMax = yTicks[yTicks.length - 1] || 15;
  const inspectedPt = activePointIdx !== null ? timelinePoints[activePointIdx] ?? null : null;

  const anchorIso = prediction?.anchor_date ?? health?.default_complete_anchor_date ?? "";
  const todayIso = prediction?.run_date ?? health?.scan_window_end ?? "";

  const svgW = 700;
  const svgH = 260;
  const padL = 48;
  const padR = 30;
  const padT = 18;
  const padB = 46;
  const plotW = svgW - padL - padR;
  const plotH = svgH - padT - padB;
  const yToPx = (val: number) => padT + plotH - (Math.min(val, yMax) / yMax) * plotH;
  const xAtIdx = (idx: number) =>
    timelinePoints.length <= 1 ? padL + plotW / 2 : padL + (idx / (timelinePoints.length - 1)) * plotW;

  const observedPathD = buildPolylinePath(
    timelinePoints.map((pt, idx) => (pt.observedMm !== null ? { x: xAtIdx(idx), y: yToPx(pt.observedMm) } : null))
  );
  const forecastPathD = buildPolylinePath(
    timelinePoints.map((pt, idx) => (pt.forecastMm !== null ? { x: xAtIdx(idx), y: yToPx(pt.forecastMm) } : null))
  );

  const labelStep = timelinePoints.length <= 8 ? 1 : Math.ceil(timelinePoints.length / 6);

  return (
    <div className="forecast-layout">
      <VietnamMap
        geoJson={geoJson}
        mapError={mapError}
        locations={locations}
        selectedLocationIds={selectedLocationIds}
        focusedLocationId={focusedLocationId}
        onToggleLocation={onToggleLocation}
        onResetSelection={onResetSelection}
      />

      <section className="panel result-dock" aria-label={EN.forecast.chartHeading} data-testid="result-dock">
        {!isFocusedSupported ? (
          <div className="info-banner" role="status" data-testid="forecast-unavailable-panel">
            <h3>{EN.forecast.unavailableHeading(focusedDisplayName)}</h3>
            <p className="unavailable-headline">{EN.forecast.unavailableMessage}</p>
            <p>{EN.forecast.unavailableDetail}</p>
          </div>
        ) : friendlyError ? (
          <div className="error-state-panel" role="alert" data-testid="anchor-error-panel">
            <h3>{EN.forecast.errorHeading(formatReadableDate(anchorIso))}</h3>
            <p className="error-summary">{friendlyError.summary}</p>
            <details className="technical-details">
              <summary>{EN.errors.technicalDetailsTitle}</summary>
              <div className="technical-details-body">
                <div>{friendlyError.code}</div>
                <div>{friendlyError.message}</div>
              </div>
            </details>
          </div>
        ) : loading && !prediction ? (
          <div className="empty-box">{EN.forecast.loadingResult}</div>
        ) : !prediction ? (
          <div className="empty-box">{EN.forecast.loadingResult}</div>
        ) : (
          <>
            <div className="dock-header">
              <div>
                <h3>
                  {EN.forecast.mainHeading(formatReadableDate(anchorIso), formatReadableDate(todayIso))}
                </h3>
                <div className="dock-meta-row">
                  <span>
                    {EN.forecast.selectedLocationLabel}: <strong>{focusedDisplayName}</strong>
                  </span>
                </div>
              </div>
              <label className="prior-toggle-label">
                <input
                  type="checkbox"
                  checked={showPrior14d}
                  onChange={(e) => onTogglePrior14d(e.target.checked)}
                />
                <span>{EN.forecast.priorObservationsToggle}</span>
              </label>
            </div>

            {prediction.lag_days > 0 && (
              <div className="retrospective-banner" role="status" data-testid="retrospective-banner">
                <span>{EN.forecast.retrospectiveBanner(prediction.lag_days)}</span>{" "}
                <span
                  className="tooltip-badge"
                  title={EN.forecast.technicalBannerTooltip}
                  aria-label={EN.forecast.technicalBannerTooltip}
                >
                  i
                </span>
              </div>
            )}

            <div className="chart-toolbar">
              <div className="line-legend">
                <span className="legend-item">
                  <span className="line-swatch line-swatch--forecast" />
                  {EN.forecast.legendForecast}
                </span>
                <span className="legend-item">
                  <span className="line-swatch line-swatch--observed" />
                  {EN.forecast.legendObserved}
                </span>
              </div>
              <span className="chart-hint">{EN.forecast.chartHoverHint}</span>
            </div>

            <div className="chart-wrap">
              <svg
                className="forecast-svg"
                viewBox={`0 0 ${svgW} ${svgH}`}
                role="img"
                aria-label={EN.forecast.chartHeading}
                data-testid="forecast-line-chart"
              >
                {yTicks.map((tick) => {
                  const y = yToPx(tick);
                  return (
                    <g key={tick}>
                      <line x1={padL} y1={y} x2={svgW - padR} y2={y} className="chart-grid-line" />
                      <text x={padL - 8} y={y + 4} textAnchor="end" className="chart-axis-text">
                        {tick}
                      </text>
                    </g>
                  );
                })}
                <text x={14} y={padT + plotH / 2} transform={`rotate(-90 14 ${padT + plotH / 2})`} textAnchor="middle" className="chart-axis-title">
                  {EN.forecast.chartYAxisLabel}
                </text>

                {observedPathD && (
                  <path
                    d={observedPathD}
                    data-series="observed"
                    fill="none"
                    stroke={OBSERVED_COLOR}
                    strokeWidth={2.4}
                  />
                )}

                {forecastPathD && (
                  <path
                    d={forecastPathD}
                    data-series="forecast"
                    fill="none"
                    stroke={FORECAST_COLOR}
                    strokeWidth={2.6}
                    strokeDasharray="6 4"
                  />
                )}

                {timelinePoints.map((pt, idx) => {
                  const cx = xAtIdx(idx);
                  const isLast = idx === timelinePoints.length - 1;
                  const tooCloseToLast =
                    !isLast &&
                    labelStep > 1 &&
                    timelinePoints.length - 1 - idx < Math.max(2, Math.floor(labelStep * 0.5));
                  const showX = isLast || (idx % labelStep === 0 && !tooCloseToLast);
                  const anchor = isLast ? "end" : idx === 0 ? "start" : "middle";
                  return (
                    <g key={`${pt.isPriorContext ? "prior" : "fwd"}_${pt.date}`}>
                      {showX && (
                        <text x={cx} y={svgH - 12} textAnchor={anchor} className="chart-axis-text">
                          {formatReadableDate(pt.date)}
                        </text>
                      )}
                      {pt.observedMm !== null && (
                        <circle
                          cx={cx}
                          cy={yToPx(pt.observedMm)}
                          r={3.8}
                          fill={OBSERVED_COLOR}
                          data-series="observed"
                          onMouseEnter={() => setActivePointIdx(idx)}
                          onFocus={() => setActivePointIdx(idx)}
                          onClick={() => setActivePointIdx(idx)}
                        >
                          <title>{`${pt.date} — ${EN.forecast.cardLabels.observed}: ${pt.observedMm.toFixed(2)}`}</title>
                        </circle>
                      )}
                      {pt.forecastMm !== null && (
                        <circle
                          cx={cx}
                          cy={yToPx(pt.forecastMm)}
                          r={4.2}
                          fill={FORECAST_COLOR}
                          data-series="forecast"
                          onMouseEnter={() => setActivePointIdx(idx)}
                          onFocus={() => setActivePointIdx(idx)}
                          onClick={() => setActivePointIdx(idx)}
                        >
                          <title>{`${pt.date} — ${EN.forecast.cardLabels.forecast}: ${pt.forecastMm.toFixed(2)}`}</title>
                        </circle>
                      )}
                    </g>
                  );
                })}
              </svg>

              {inspectedPt && (
                <div className="chart-readout" role="status" data-testid="chart-readout">
                  <strong>{`${EN.forecast.hoverReadoutPrefix} · ${formatReadableDate(inspectedPt.date)} (${inspectedPt.date})`}</strong>
                  <span>{`${EN.forecast.cardLabels.observed}: ${inspectedPt.observedMm !== null ? `${inspectedPt.observedMm.toFixed(2)} ${EN.forecast.mmUnit}` : EN.forecast.nullValue}`}</span>
                  <span>{`${EN.forecast.cardLabels.forecast}: ${inspectedPt.forecastMm !== null ? `${inspectedPt.forecastMm.toFixed(2)} ${EN.forecast.mmUnit}` : EN.forecast.nullValue}`}</span>
                </div>
              )}
            </div>

            <div className="desktop-forecast-table table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>{EN.forecast.tableHeaders.readableDate}</th>
                    <th>{EN.forecast.tableHeaders.isoDate}</th>
                    <th>{EN.forecast.tableHeaders.status}</th>
                    <th>{EN.forecast.tableHeaders.observed}</th>
                    <th>{EN.forecast.tableHeaders.forecast}</th>
                  </tr>
                </thead>
                <tbody>
                  {dailyForecasts.map((day) => (
                    <tr key={day.target_date}>
                      <td title={day.target_date}>{formatReadableDate(day.target_date)}</td>
                      <td className="mono">{day.target_date}</td>
                      <td>{EN.forecast.statusLabels[day.status]}</td>
                      <td className="num">
                        {day.observed_mm !== null ? day.observed_mm.toFixed(2) : EN.forecast.nullValue}
                      </td>
                      <td className="num">{day.predictions["SC-35"].toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mobile-day-cards" data-testid="mobile-day-cards">
              {dailyForecasts.map((day) => (
                <article key={day.target_date} className="day-card">
                  <div className="day-card-header">
                    <strong>{formatReadableDate(day.target_date)}</strong>
                    <span className="day-card-iso">{day.target_date}</span>
                    <span className="day-card-status">{EN.forecast.statusLabels[day.status]}</span>
                  </div>
                  <dl className="day-card-metrics">
                    <div>
                      <dt>{EN.forecast.cardLabels.observed}</dt>
                      <dd>{day.observed_mm !== null ? day.observed_mm.toFixed(2) : EN.forecast.nullValue}</dd>
                    </div>
                    <div>
                      <dt>{EN.forecast.cardLabels.forecast}</dt>
                      <dd>{day.predictions["SC-35"].toFixed(2)}</dd>
                    </div>
                  </dl>
                </article>
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  );
};
