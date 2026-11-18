import React, { useMemo, useState } from "react";
import { EN, formatReadableDate, mapApiErrorToFriendly, type FriendlyErrorView } from "../i18n/en";
import type { ApiClientError } from "../api/client";
import type {
  DailyForecastItem,
  HealthResponse,
  LocationItem,
  ModelKey5,
  PredictResponse,
  ProvinceGeoJson,
} from "../types/api";
import { VietnamMap } from "./VietnamMap";

interface AnchorForecastSectionProps {
  health: HealthResponse | null;
  locations: LocationItem[];
  selectedLocationId: string;
  onSelectLocation: (locationId: string) => void;
  geoJson: ProvinceGeoJson | null;
  mapError: string | null;
  prediction: PredictResponse | null;
  loading: boolean;
  errorValue: unknown;
  selectedAnchor: string;
  onSelectAnchor: (anchorDate: string) => void;
}

type OptionalSeriesKey = ModelKey5;

const TOGGLEABLE_SERIES: Array<{ key: OptionalSeriesKey; color: string }> = [
  { key: "SC-35", color: "#0B5CAD" },
  { key: "BASELINE_CLIMATOLOGY_DOY", color: "#0D9488" },
  { key: "SC-30", color: "#6D28D9" },
  { key: "SC-31", color: "#D97706" },
  { key: "BASELINE_PERSISTENCE", color: "#64748B" },
];

const OBSERVED_COLOR = "#0F1E2E";

function computeTicksMultiplesOf5(maxVal: number): number[] {
  const ceil5 = Math.max(15, Math.ceil(maxVal / 5) * 5);
  const step = ceil5 <= 25 ? 5 : Math.ceil(ceil5 / 5 / 5) * 5;
  const ticks: number[] = [];
  for (let v = 0; v <= ceil5; v += step) {
    ticks.push(v);
  }
  if (ticks[ticks.length - 1] < ceil5) {
    ticks.push(ceil5);
  }
  return ticks;
}

export const AnchorForecastSection: React.FC<AnchorForecastSectionProps> = ({
  health,
  locations,
  selectedLocationId,
  onSelectLocation,
  geoJson,
  mapError,
  prediction,
  loading,
  errorValue,
  selectedAnchor,
  onSelectAnchor,
}) => {
  const [draftAnchor, setDraftAnchor] = useState<string>(selectedAnchor || "2026-08-30");
  const [visibleSeries, setVisibleSeries] = useState<Record<OptionalSeriesKey, boolean>>({
    "SC-35": true,
    BASELINE_CLIMATOLOGY_DOY: true,
    "SC-30": false,
    "SC-31": false,
    BASELINE_PERSISTENCE: false,
  });
  const [activeDayIndex, setActiveDayIndex] = useState<number | null>(null);

  React.useEffect(() => {
    if (selectedAnchor) {
      setDraftAnchor(selectedAnchor);
    }
  }, [selectedAnchor]);

  const minAnchor = health?.anchor_range?.min ?? "2001-02-04";
  const maxAnchor = health?.anchor_range?.max ?? "2026-08-30";

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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (draftAnchor.trim()) {
      onSelectAnchor(draftAnchor.trim());
    }
  };

  const applyPreset = (dateStr: string) => {
    setDraftAnchor(dateStr);
    onSelectAnchor(dateStr);
  };

  const toggleSeries = (key: OptionalSeriesKey) => {
    setVisibleSeries((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const selectedLocationObj = locations.find((l) => l.location_id === selectedLocationId);
  const daily: DailyForecastItem[] = !friendlyError && prediction ? prediction.daily_forecasts : [];

  const maxChartVal = useMemo(() => {
    let maxV = 0;
    for (const day of daily) {
      if (day.observed_mm !== null && day.observed_mm > maxV) {
        maxV = day.observed_mm;
      }
      for (const s of TOGGLEABLE_SERIES) {
        if (visibleSeries[s.key]) {
          const val = day.predictions[s.key] ?? 0;
          if (val > maxV) maxV = val;
        }
      }
    }
    return maxV;
  }, [daily, visibleSeries]);

  const yTicks = useMemo(() => computeTicksMultiplesOf5(maxChartVal), [maxChartVal]);
  const yMax = yTicks[yTicks.length - 1] || 15;
  const inspectedDay = activeDayIndex !== null ? daily[activeDayIndex] ?? null : null;

  return (
    <div className="forecast-layout">
      <VietnamMap
        geoJson={geoJson}
        mapError={mapError}
        locations={locations}
        selectedLocationId={selectedLocationId}
        onSelectSupportedLocation={onSelectLocation}
      />

      <section className="panel anchor-control-panel" aria-label={EN.forecast.anchorSectionTitle}>
        <form className="anchor-form" onSubmit={handleSubmit}>
          <div className="anchor-field-wrap">
            <label htmlFor="anchor-date-input">
              {EN.forecast.anchorLabel}{" "}
              <span
                className="tooltip-badge"
                title={EN.forecast.anchorTooltip}
                aria-label={EN.forecast.anchorTooltip}
              >
                ?
              </span>
            </label>
            <div className="anchor-input-row">
              <input
                id="anchor-date-input"
                type="date"
                value={draftAnchor}
                min={minAnchor}
                max={health?.data_snapshot_date ?? maxAnchor}
                onChange={(e) => setDraftAnchor(e.target.value)}
              />
              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? EN.forecast.loadingResult : EN.forecast.runButton}
              </button>
            </div>
            <p className="field-hint">
              {EN.forecast.anchorHelpRange(formatReadableDate(minAnchor), formatReadableDate(maxAnchor))}
            </p>
          </div>

          <div className="preset-group" aria-label={EN.forecast.quickPresetsLabel}>
            <span className="preset-caption">{EN.forecast.quickPresetsLabel}</span>
            <div className="preset-buttons">
              <button
                type="button"
                className="btn-chip"
                onClick={() => applyPreset(maxAnchor)}
              >
                {EN.forecast.presetLatest}
              </button>
              <button
                type="button"
                className="btn-chip"
                onClick={() => applyPreset(EN.forecast.presetWetAnchor)}
              >
                {EN.forecast.presetWet}
              </button>
              <button
                type="button"
                className="btn-chip"
                onClick={() => applyPreset(EN.forecast.presetDryAnchor)}
              >
                {EN.forecast.presetDry}
              </button>
            </div>
          </div>
        </form>
      </section>

      <section className="panel result-dock" aria-label={EN.forecast.chartHeading} data-testid="result-dock">
        {friendlyError ? (
          <div className="error-state-panel" role="alert" data-testid="anchor-error-panel">
            <h3>{EN.forecast.errorHeading(formatReadableDate(selectedAnchor || draftAnchor))}</h3>
            <p className="error-summary">{friendlyError.summary}</p>
            <details className="technical-details">
              <summary>{EN.errors.technicalDetailsTitle}</summary>
              <div className="technical-details-body">
                <div>{friendlyError.code}</div>
                <div>{friendlyError.message}</div>
                {Object.keys(friendlyError.details).length > 0 && (
                  <pre>{JSON.stringify(friendlyError.details, null, 2)}</pre>
                )}
              </div>
            </details>
          </div>
        ) : loading && !prediction ? (
          <div className="empty-box">{EN.forecast.loadingResult}</div>
        ) : !prediction ? (
          <div className="empty-box">{EN.forecast.chartEmpty}</div>
        ) : (
          <>
            <div className="dock-header">
              <div>
                <h3>
                  {EN.forecast.dockTitle(
                    formatReadableDate(prediction.anchor_date),
                    prediction.effective_future_days === 0
                  )}
                </h3>
                <div className="dock-meta-row">
                  <span>
                    {EN.forecast.selectedLocationLabel}:{" "}
                    <strong>{selectedLocationObj?.display_name ?? EN.forecast.defaultLocationName}</strong>
                  </span>
                  <span>
                    {EN.forecast.splitRegimeLabel}: <strong>{prediction.anchor_split_regime}</strong>
                  </span>
                  <span>
                    {EN.forecast.effectiveFutureLabel}: <strong>{prediction.effective_future_days}</strong>
                  </span>
                </div>
              </div>
            </div>

            {prediction.effective_future_days === 0 && (
              <div className="retrospective-banner" role="status" data-testid="retrospective-banner">
                <strong>{EN.forecast.retrospectiveBanner}</strong>{" "}
                {prediction.future_forecast_notice && <span>{prediction.future_forecast_notice}</span>}
              </div>
            )}

            <div className="series-toggle-bar" role="group" aria-label={EN.forecast.seriesTogglesLabel}>
              <span className="series-toggle-title">{EN.forecast.seriesTogglesLabel}</span>
              <span className="series-pill series-pill--locked">
                <span className="series-dot" style={{ backgroundColor: OBSERVED_COLOR }} />
                {EN.forecast.seriesLabels.observed}
              </span>
              {TOGGLEABLE_SERIES.map((s) => (
                <button
                  key={s.key}
                  type="button"
                  aria-pressed={visibleSeries[s.key]}
                  onClick={() => toggleSeries(s.key)}
                  className={`series-pill ${visibleSeries[s.key] ? "series-pill--on" : ""}`}
                >
                  <span className="series-dot" style={{ backgroundColor: s.color }} />
                  {EN.forecast.seriesLabels[s.key]}
                </button>
              ))}
            </div>

            <div className="chart-container">
              <div className="chart-header-row">
                <h4>{EN.forecast.chartHeading}</h4>
                <span className="chart-hint">{EN.forecast.chartHoverHint}</span>
              </div>

              <svg
                className="forecast-chart-svg"
                viewBox="0 0 640 240"
                role="img"
                aria-label={EN.forecast.chartHeading}
              >
                {/* Horizontal gridlines & y-axis ticks (multiples of 5) */}
                {yTicks.map((tick) => {
                  const yPos = 195 - (tick / yMax) * 165;
                  return (
                    <g key={tick}>
                      <line x1={48} x2={624} y1={yPos} y2={yPos} className="chart-grid-line" />
                      <text x={40} y={yPos + 4} textAnchor="end" className="chart-axis-text">
                        {tick}
                      </text>
                    </g>
                  );
                })}
                <line x1={48} x2={48} y1={24} y2={195} className="chart-axis-line" />
                <line x1={48} x2={624} y1={195} y2={195} className="chart-axis-line" />

                {daily.map((day, idx) => {
                  const slotW = (624 - 56) / Math.max(1, daily.length);
                  const cx = 56 + idx * slotW + slotW / 2;

                  const activeBars: Array<{ key: string; val: number | null; color: string }> = [
                    { key: "observed", val: day.observed_mm, color: OBSERVED_COLOR },
                    ...TOGGLEABLE_SERIES.filter((s) => visibleSeries[s.key]).map((s) => ({
                      key: s.key,
                      val: day.predictions[s.key] ?? 0,
                      color: s.color,
                    })),
                  ];
                  const barW = Math.min(16, Math.max(7, Math.floor((slotW - 14) / activeBars.length)));
                  const totalGroupW = activeBars.length * (barW + 2) - 2;
                  const startX = cx - totalGroupW / 2;
                  const isHighlighted = activeDayIndex === idx;

                  return (
                    <g
                      key={day.target_date}
                      tabIndex={0}
                      role="button"
                      aria-label={`${formatReadableDate(day.target_date)} ${day.predictions["SC-35"].toFixed(1)} ${EN.forecast.millimeterUnit}`}
                      className="chart-day-group"
                      onMouseEnter={() => setActiveDayIndex(idx)}
                      onFocus={() => setActiveDayIndex(idx)}
                      onClick={() => setActiveDayIndex(idx)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          setActiveDayIndex(idx);
                        }
                      }}
                    >
                      <rect
                        x={56 + idx * slotW + 2}
                        y={24}
                        width={slotW - 4}
                        height={171}
                        className={isHighlighted ? "chart-slot-highlight" : "chart-slot-hit"}
                      />
                      {activeBars.map((bar, bIdx) => {
                        if (bar.val === null) return null;
                        const hPx = Math.max(2, (bar.val / yMax) * 165);
                        const bx = startX + bIdx * (barW + 2);
                        const by = 195 - hPx;
                        return (
                          <rect
                            key={bar.key}
                            x={bx}
                            y={by}
                            width={barW}
                            height={hPx}
                            rx={2}
                            fill={bar.color}
                          />
                        );
                      })}
                      <text x={cx} y={214} textAnchor="middle" className="chart-x-label">
                        {formatReadableDate(day.target_date).replace(/ \d{4}$/, "")}
                      </text>
                    </g>
                  );
                })}
              </svg>

              {inspectedDay && (
                <div className="chart-readout" role="status" data-testid="chart-readout">
                  <strong>{formatReadableDate(inspectedDay.target_date)}</strong> (
                  <span>{inspectedDay.target_date}</span>) —{" "}
                  <span>
                    {EN.forecast.seriesLabels.observed}:{" "}
                    {inspectedDay.observed_mm !== null
                      ? `${inspectedDay.observed_mm.toFixed(2)} ${EN.forecast.millimeterUnit}`
                      : EN.forecast.nullValue}
                  </span>
                  {" · "}
                  <span>
                    {EN.forecast.seriesLabels["SC-35"]}:{" "}
                    {inspectedDay.predictions["SC-35"].toFixed(2)} {EN.forecast.millimeterUnit}
                  </span>
                  {" · "}
                  <span>
                    {EN.forecast.seriesLabels.BASELINE_CLIMATOLOGY_DOY}:{" "}
                    {inspectedDay.predictions.BASELINE_CLIMATOLOGY_DOY.toFixed(2)} {EN.forecast.millimeterUnit}
                  </span>
                </div>
              )}
            </div>

            {/* Desktop table */}
            <div className="desktop-only-table">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>{EN.forecast.tableHeaders.horizon}</th>
                    <th>{EN.forecast.tableHeaders.readableDate}</th>
                    <th>{EN.forecast.tableHeaders.isoDate}</th>
                    <th>{EN.forecast.tableHeaders.status}</th>
                    <th>{EN.forecast.tableHeaders.observed}</th>
                    <th>{EN.forecast.tableHeaders.sc35}</th>
                    <th>{EN.forecast.tableHeaders.clim}</th>
                    <th>{EN.forecast.tableHeaders.sc30}</th>
                    <th>{EN.forecast.tableHeaders.sc31}</th>
                    <th>{EN.forecast.tableHeaders.pers}</th>
                  </tr>
                </thead>
                <tbody>
                  {daily.map((day) => (
                    <tr key={day.target_date}>
                      <td className="num">{EN.forecast.horizonDayLabel(day.horizon_day)}</td>
                      <td title={day.target_date}>{formatReadableDate(day.target_date)}</td>
                      <td className="mono">{day.target_date}</td>
                      <td>{EN.forecast.statusLabels[day.status]}</td>
                      <td className="num">
                        {day.observed_mm !== null ? day.observed_mm.toFixed(2) : EN.forecast.nullValue}
                      </td>
                      <td className="num">{day.predictions["SC-35"].toFixed(2)}</td>
                      <td className="num">{day.predictions.BASELINE_CLIMATOLOGY_DOY.toFixed(2)}</td>
                      <td className="num">{day.predictions["SC-30"].toFixed(2)}</td>
                      <td className="num">{day.predictions["SC-31"].toFixed(2)}</td>
                      <td className="num">{day.predictions.BASELINE_PERSISTENCE.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Mobile 375px vertical day cards (every numeric field visible, zero hidden columns or overflow) */}
            <div className="mobile-day-cards" data-testid="mobile-day-cards">
              {daily.map((day) => (
                <article key={day.target_date} className="day-card">
                  <div className="day-card-header">
                    <strong>{formatReadableDate(day.target_date)}</strong>
                    <span className="day-card-iso">{day.target_date}</span>
                    <span className="day-card-status">{EN.forecast.statusLabels[day.status]}</span>
                  </div>
                  <dl className="day-card-metrics">
                    <div>
                      <dt>{EN.forecast.tableHeaders.observed}</dt>
                      <dd>{day.observed_mm !== null ? day.observed_mm.toFixed(2) : EN.forecast.nullValue}</dd>
                    </div>
                    <div>
                      <dt>{EN.forecast.tableHeaders.sc35}</dt>
                      <dd>{day.predictions["SC-35"].toFixed(2)}</dd>
                    </div>
                    <div>
                      <dt>{EN.forecast.tableHeaders.clim}</dt>
                      <dd>{day.predictions.BASELINE_CLIMATOLOGY_DOY.toFixed(2)}</dd>
                    </div>
                    <div>
                      <dt>{EN.forecast.tableHeaders.sc30}</dt>
                      <dd>{day.predictions["SC-30"].toFixed(2)}</dd>
                    </div>
                    <div>
                      <dt>{EN.forecast.tableHeaders.sc31}</dt>
                      <dd>{day.predictions["SC-31"].toFixed(2)}</dd>
                    </div>
                    <div>
                      <dt>{EN.forecast.tableHeaders.pers}</dt>
                      <dd>{day.predictions.BASELINE_PERSISTENCE.toFixed(2)}</dd>
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
