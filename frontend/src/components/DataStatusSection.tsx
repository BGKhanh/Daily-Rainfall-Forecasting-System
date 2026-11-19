import React, { useEffect, useMemo, useRef, useState } from "react";
import { fetchObservations, type ApiClientError } from "../api/client";
import { EN, formatReadableDate, mapApiErrorToFriendly } from "../i18n/en";
import { computeTicksMultiplesOf5 } from "../utils/chart";
import type {
  HealthResponse,
  LocationItem,
  ObservationGranularity,
  ObservationSeriesPoint,
  ObservationsQueryResponse,
  ProvinceGeoJson,
} from "../types/api";

interface DataStatusSectionProps {
  health: HealthResponse | null;
  locations: LocationItem[];
  geoJson: ProvinceGeoJson | null;
}

const FINALIZED_COLOR = "#1d4ed8";
const PROVISIONAL_COLOR = "#d97706";
const FORECAST_COLOR = "#7c3aed";
const PAGE_SIZE = 15;

function diffCalendarDays(startIso: string, endIso: string): number {
  const s = Date.parse(`${startIso}T00:00:00Z`);
  const e = Date.parse(`${endIso}T00:00:00Z`);
  if (Number.isNaN(s) || Number.isNaN(e)) return 0;
  return Math.floor((e - s) / 86400000) + 1;
}

function resolveGranularityForSpan(
  startIso: string,
  endIso: string,
  requested: ObservationGranularity
): { granularity: ObservationGranularity; switched: boolean } {
  if (!startIso || !endIso) return { granularity: requested, switched: false };
  const span = diffCalendarDays(startIso, endIso);
  if (span > 2555 && requested !== "monthly") {
    return { granularity: "monthly", switched: true };
  }
  if (span > 366 && requested === "daily") {
    return { granularity: "weekly", switched: true };
  }
  return { granularity: requested, switched: false };
}

function csvCell(value: string): string {
  const sanitized = value.replace(/[\r\n]+/g, " ");
  const formulaSafe = /^[=+\-@\t]/.test(sanitized) ? `'${sanitized}` : sanitized;
  return `"${formulaSafe.replace(/"/g, '""')}"`;
}

export const DataStatusSection: React.FC<DataStatusSectionProps> = ({
  health,
  locations,
  geoJson,
}) => {
  const [locationId, setLocationId] = useState<string>("HCMC");
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");
  const [granularity, setGranularity] = useState<ObservationGranularity>("daily");
  const [metricMode, setMetricMode] = useState<"total" | "mean">("total");
  const [autoSwitched, setAutoSwitched] = useState<boolean>(false);

  const [data, setData] = useState<ObservationsQueryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [errorValue, setErrorValue] = useState<unknown>(null);
  const [page, setPage] = useState<number>(1);
  const reqSeqRef = useRef<number>(0);

  const locationOptions = useMemo(() => {
    const map = new Map<string, string>();
    for (const loc of locations) {
      map.set(loc.location_id, loc.display_name);
    }
    for (const feat of geoJson?.features ?? []) {
      if (!map.has(feat.properties.location_id)) {
        map.set(feat.properties.location_id, feat.properties.display_name);
      }
    }
    if (!map.has("HCMC")) {
      map.set("HCMC", EN.forecast.defaultLocationName);
    }
    return Array.from(map.entries())
      .map(([id, name]) => ({ location_id: id, display_name: name }))
      .sort((a, b) => (a.location_id === "HCMC" ? -1 : b.location_id === "HCMC" ? 1 : a.display_name.localeCompare(b.display_name)));
  }, [locations, geoJson]);

  const runQuery = async (opts: {
    loc: string;
    start?: string;
    end?: string;
    gran: ObservationGranularity;
    noticeSwitched?: boolean;
  }) => {
    const seq = ++reqSeqRef.current;
    setLoading(true);
    setErrorValue(null);
    setAutoSwitched(Boolean(opts.noticeSwitched));
    try {
      const res = await fetchObservations({
        locationId: opts.loc,
        startDate: opts.start || undefined,
        endDate: opts.end || undefined,
        granularity: opts.gran,
      });
      if (seq !== reqSeqRef.current) return;
      setData(res);
      setStartDate(res.start_date);
      setEndDate(res.end_date);
      setGranularity(res.granularity);
      setPage(1);
    } catch (err) {
      if (seq !== reqSeqRef.current) return;
      setErrorValue(err);
    } finally {
      if (seq === reqSeqRef.current) setLoading(false);
    }
  };

  useEffect(() => {
    void runQuery({ loc: "HCMC", gran: "daily" });
    return () => {
      reqSeqRef.current += 1;
    };
  }, []);

  const handleApplyRange = (e: React.FormEvent) => {
    e.preventDefault();
    const resolved = resolveGranularityForSpan(startDate, endDate, granularity);
    setGranularity(resolved.granularity);
    void runQuery({
      loc: locationId,
      start: startDate,
      end: endDate,
      gran: resolved.granularity,
      noticeSwitched: resolved.switched,
    });
  };

  const applyQuickPreset = (kind: "90d" | "2y" | "full") => {
    const maxD = data?.valid_range.max_date ?? health?.scan_window_end ?? "2026-11-19";
    const minD = data?.valid_range.min_date ?? "2001-01-01";
    const endDt = new Date(`${maxD}T00:00:00Z`);
    if (kind === "90d") {
      const startDt = new Date(endDt.getTime() - 89 * 86400000);
      const s = startDt.toISOString().slice(0, 10);
      setStartDate(s);
      setEndDate(maxD);
      setGranularity("daily");
      void runQuery({ loc: locationId, start: s, end: maxD, gran: "daily" });
    } else if (kind === "2y") {
      const startDt = new Date(endDt.getTime() - 729 * 86400000);
      const s = startDt.toISOString().slice(0, 10);
      setStartDate(s);
      setEndDate(maxD);
      setGranularity("weekly");
      void runQuery({ loc: locationId, start: s, end: maxD, gran: "weekly" });
    } else {
      setStartDate(minD);
      setEndDate(maxD);
      setGranularity("monthly");
      void runQuery({ loc: locationId, start: minD, end: maxD, gran: "monthly" });
    }
  };

  const series = data?.series ?? [];
  const overlay = data?.forecast_overlay ?? [];

  const getValue = (pt: ObservationSeriesPoint): number | null => {
    if (pt.status === "NOT_PUBLISHED") return null;
    if (data?.granularity === "daily") return pt.precip_mm;
    return metricMode === "total" ? pt.precip_total_mm : pt.precip_daily_mean_mm;
  };

  const handleDownloadCsv = () => {
    if (!data || !series.length) return;
    const valCol =
      data.granularity === "daily"
        ? "precip_mm"
        : metricMode === "total"
          ? "precip_total_mm"
          : "precip_daily_mean_mm";
    const lines = [
      "# Source: NASA POWER Daily API v2.10.4 (PRECTOTCORR)",
      `# Location: ${data.display_name} (${data.location_id}) | Granularity: ${data.granularity} | Range: ${data.start_date}..${data.end_date}`,
      `date,period_end,${valCol},published_days_count,calendar_days_count,status`,
    ];
    for (const row of series) {
      const v = getValue(row);
      lines.push(
        [
          csvCell(row.date),
          csvCell(row.period_end),
          v !== null ? v.toFixed(4) : "",
          String(row.published_days_count),
          String(row.calendar_days_count),
          csvCell(row.status),
        ].join(",")
      );
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${data.location_id.toLowerCase()}_observations_${data.granularity}_${data.start_date}_${data.end_date}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const friendlyErr = useMemo(() => {
    if (!errorValue) return null;
    const errObj = errorValue as Partial<ApiClientError>;
    return mapApiErrorToFriendly({
      code: errObj.code,
      message: errObj.message,
      details: errObj.details,
      status: errObj.status,
    });
  }, [errorValue]);

  const dateToIndex = useMemo(() => {
    const m = new Map<string, number>();
    series.forEach((pt, idx) => m.set(pt.date, idx));
    return m;
  }, [series]);

  const maxVal = useMemo(() => {
    let mv = 0;
    for (const pt of series) {
      const v = getValue(pt);
      if (v !== null && v > mv) mv = v;
    }
    if (data?.granularity === "daily") {
      for (const f of overlay) {
        if (dateToIndex.has(f.date) && f.pred_mm > mv) mv = f.pred_mm;
      }
    }
    return mv;
  }, [series, overlay, data?.granularity, metricMode, dateToIndex]);

  const yTicks = computeTicksMultiplesOf5(maxVal);
  const yMax = yTicks[yTicks.length - 1] || 15;
  const svgW = 740;
  const svgH = 260;
  const padL = 52;
  const padR = 30;
  const padT = 18;
  const padB = 46;
  const plotW = svgW - padL - padR;
  const plotH = svgH - padT - padB;
  const yToPx = (val: number) => padT + plotH - (Math.min(val, yMax) / yMax) * plotH;
  const xAtIdx = (idx: number) =>
    series.length <= 1 ? padL + plotW / 2 : padL + (idx / (series.length - 1)) * plotW;

  // Build non-gap segmented paths + contiguous NOT_PUBLISHED gap bands
  const segmentPaths = useMemo(() => {
    const paths: Array<{ status: "FINALIZED" | "PROVISIONAL"; d: string; key: string }> = [];
    const gapBands: Array<{ startIdx: number; endIdx: number; key: string }> = [];

    let curRunStatus: "FINALIZED" | "PROVISIONAL" | null = null;
    let curRunCoords: Array<{ x: number; y: number }> = [];
    let gapStart: number | null = null;

    const flushRun = (idxMarker: number) => {
      if (curRunStatus && curRunCoords.length >= 1) {
        const d = curRunCoords
          .map((p, k) => `${k === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
          .join(" ");
        paths.push({ status: curRunStatus, d, key: `${curRunStatus}_${idxMarker}` });
      }
      curRunStatus = null;
      curRunCoords = [];
    };

    for (let i = 0; i < series.length; i++) {
      const pt = series[i];
      const val = getValue(pt);
      if (pt.status === "NOT_PUBLISHED" || val === null) {
        flushRun(i);
        if (gapStart === null) gapStart = i;
        continue;
      }
      if (gapStart !== null) {
        gapBands.push({ startIdx: gapStart, endIdx: i - 1, key: `gap_${gapStart}` });
        gapStart = null;
      }
      const coord = { x: xAtIdx(i), y: yToPx(val) };
      if (curRunStatus === pt.status) {
        curRunCoords.push(coord);
      } else {
        const prevValid = curRunCoords.length > 0 ? curRunCoords[curRunCoords.length - 1] : null;
        flushRun(i);
        curRunStatus = pt.status;
        curRunCoords = prevValid ? [prevValid, coord] : [coord];
      }
    }
    flushRun(series.length);
    if (gapStart !== null) {
      gapBands.push({ startIdx: gapStart, endIdx: series.length - 1, key: `gap_${gapStart}` });
    }
    return { paths, gapBands };
  }, [series, metricMode, yMax]);

  const forecastOverlayPathD = useMemo(() => {
    if (data?.granularity !== "daily" || !overlay.length) return "";
    const pts: Array<{ x: number; y: number }> = [];
    for (const f of overlay) {
      const idx = dateToIndex.get(f.date);
      if (idx !== undefined) {
        pts.push({ x: xAtIdx(idx), y: yToPx(f.pred_mm) });
      }
    }
    if (pts.length === 0) return "";
    return pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  }, [data?.granularity, overlay, dateToIndex, yMax]);

  const totalPages = Math.max(1, Math.ceil(series.length / PAGE_SIZE));
  const pagedRows = useMemo(() => {
    const rev = [...series].reverse();
    const startIdx = (page - 1) * PAGE_SIZE;
    return rev.slice(startIdx, startIdx + PAGE_SIZE);
  }, [series, page]);

  const yAxisTitle =
    data?.granularity === "daily"
      ? EN.dataRetrieval.yAxisDaily
      : metricMode === "total"
        ? EN.dataRetrieval.yAxisPeriodTotal
        : EN.dataRetrieval.yAxisDailyMean;

  const xTickStep = series.length <= 6 ? 1 : Math.ceil(series.length / 5);

  return (
    <section className="panel" aria-label={EN.dataRetrieval.heading}>
      <h2>{EN.dataRetrieval.heading}</h2>
      <p className="panel-subtitle">{EN.dataRetrieval.subtitle}</p>

      <form className="obs-controls-form" onSubmit={handleApplyRange}>
        <div className="obs-controls-grid">
          <div className="control-field">
            <label htmlFor="obs-location-select">{EN.dataRetrieval.locationLabel}</label>
            <select
              id="obs-location-select"
              value={locationId}
              onChange={(e) => {
                const nextLoc = e.target.value;
                setLocationId(nextLoc);
                void runQuery({ loc: nextLoc, start: startDate, end: endDate, gran: granularity });
              }}
            >
              {locationOptions.map((opt) => (
                <option key={opt.location_id} value={opt.location_id}>
                  {opt.display_name}
                </option>
              ))}
            </select>
          </div>

          <div className="control-field">
            <label htmlFor="obs-start-date">{EN.dataRetrieval.startDateLabel}</label>
            <input
              id="obs-start-date"
              type="date"
              value={startDate}
              min={data?.valid_range.min_date ?? "2001-01-01"}
              max={data?.valid_range.max_date ?? health?.scan_window_end ?? "2026-11-19"}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>

          <div className="control-field">
            <label htmlFor="obs-end-date">{EN.dataRetrieval.endDateLabel}</label>
            <input
              id="obs-end-date"
              type="date"
              value={endDate}
              min={data?.valid_range.min_date ?? "2001-01-01"}
              max={data?.valid_range.max_date ?? health?.scan_window_end ?? "2026-11-19"}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>

          <div className="control-field">
            <label htmlFor="obs-granularity-select">{EN.dataRetrieval.granularityLabel}</label>
            <select
              id="obs-granularity-select"
              value={granularity}
              onChange={(e) => {
                const requested = e.target.value as ObservationGranularity;
                const resolved = resolveGranularityForSpan(startDate, endDate, requested);
                setGranularity(resolved.granularity);
                void runQuery({
                  loc: locationId,
                  start: startDate,
                  end: endDate,
                  gran: resolved.granularity,
                  noticeSwitched: resolved.switched,
                });
              }}
            >
              <option value="daily">{EN.dataRetrieval.granularityOptions.daily}</option>
              <option value="weekly">{EN.dataRetrieval.granularityOptions.weekly}</option>
              <option value="monthly">{EN.dataRetrieval.granularityOptions.monthly}</option>
            </select>
          </div>

          {granularity !== "daily" && (
            <div className="control-field">
              <label htmlFor="obs-metric-mode">{EN.dataRetrieval.metricModeLabel}</label>
              <select
                id="obs-metric-mode"
                value={metricMode}
                onChange={(e) => setMetricMode(e.target.value as "total" | "mean")}
              >
                <option value="total">{EN.dataRetrieval.metricTotalOption}</option>
                <option value="mean">{EN.dataRetrieval.metricMeanOption}</option>
              </select>
            </div>
          )}
        </div>

        <div className="obs-actions-row">
          <div className="preset-buttons">
            <button type="button" className="btn-chip" onClick={() => applyQuickPreset("90d")}>
              {EN.dataRetrieval.preset90Days}
            </button>
            <button type="button" className="btn-chip" onClick={() => applyQuickPreset("2y")}>
              {EN.dataRetrieval.preset2Years}
            </button>
            <button type="button" className="btn-chip" onClick={() => applyQuickPreset("full")}>
              {EN.dataRetrieval.presetFullHistory}
            </button>
          </div>
          <div className="obs-submit-group">
            <button type="submit" className="btn-primary" disabled={loading}>
              {EN.dataRetrieval.runQueryButton}
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={handleDownloadCsv}
              disabled={!data || !series.length}
            >
              {EN.dataRetrieval.downloadCsv}
            </button>
          </div>
        </div>
      </form>

      {autoSwitched && (
        <div className="info-banner" role="status">
          {EN.dataRetrieval.autoSwitchedNotice}
        </div>
      )}

      {friendlyErr && (
        <div className="error-state-panel" role="alert">
          <p className="error-summary">{friendlyErr.summary}</p>
          <details className="technical-details">
            <summary>{EN.errors.technicalDetailsTitle}</summary>
            <div className="technical-details-body">
              <div>{friendlyErr.code}</div>
              <div>{friendlyErr.message}</div>
            </div>
          </details>
        </div>
      )}

      {loading && !data ? (
        <div className="empty-box">{EN.dataRetrieval.loadingQuery}</div>
      ) : data && !data.has_data ? (
        <div className="info-banner" role="status" data-testid="observations-empty-location">
          <strong>{EN.dataRetrieval.emptyLocationTitle}:</strong>{" "}
          <span>{data.empty_reason || EN.dataRetrieval.emptyLocationMessage}</span>
        </div>
      ) : data ? (
        <>
          <div className="metric-grid">
            <div className="metric-card">
              <div className="metric-label">{EN.dataRetrieval.summaryTotalLabel}</div>
              <div className="metric-value">{data.summary.total_precip_mm.toFixed(1)}</div>
              <div className="metric-sub">{`${formatReadableDate(data.start_date)} — ${formatReadableDate(data.end_date)}`}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">{EN.dataRetrieval.summaryRainDaysLabel}</div>
              <div className="metric-value">{data.summary.rain_days_ge_1mm}</div>
              <div className="metric-sub">{`${data.summary.published_days} / ${data.summary.calendar_days} ${EN.dataRetrieval.daysUnit}`}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">{EN.dataRetrieval.summaryBreakdownLabel}</div>
              <div className="metric-value">
                {`${data.summary.finalized_days} / ${data.summary.provisional_days} / ${data.summary.not_published_days}`}
              </div>
              <div className="metric-sub">
                {data.finality_cutoff_date
                  ? `${EN.dataRetrieval.cutoffLabel}: ${formatReadableDate(data.finality_cutoff_date)}`
                  : ""}
              </div>
            </div>
          </div>

          <div className="subpanel">
            <div className="panel-header-row">
              <h3>{EN.dataRetrieval.chartHeading}</h3>
              <span
                className="tooltip-badge"
                title={EN.dataRetrieval.finalityTooltipText}
                aria-label={EN.dataRetrieval.finalityTooltipLabel}
              >
                i
              </span>
            </div>

            <div className="line-legend">
              <span className="legend-item">
                <span className="line-swatch line-swatch--finalized" />
                {EN.dataRetrieval.legendFinalized}
              </span>
              <span className="legend-item">
                <span className="line-swatch line-swatch--provisional" />
                {EN.dataRetrieval.legendProvisional}
              </span>
              <span className="legend-item">
                <span className="line-swatch line-swatch--forecast" />
                {EN.dataRetrieval.legendForecast}
              </span>
              <span className="legend-item">
                <span className="legend-swatch legend-swatch--gap" />
                {EN.dataRetrieval.legendNotPublished}
              </span>
            </div>

            <div className="chart-wrap">
              <svg
                className="forecast-svg"
                viewBox={`0 0 ${svgW} ${svgH}`}
                role="img"
                aria-label={EN.dataRetrieval.chartHeading}
                data-testid="observations-chart"
              >
                {segmentPaths.gapBands.map((gb) => {
                  const xStart = xAtIdx(gb.startIdx) - 3;
                  const xEnd = xAtIdx(gb.endIdx) + 3;
                  return (
                    <rect
                      key={gb.key}
                      data-gap-band="NOT_PUBLISHED"
                      x={Math.max(padL, xStart)}
                      y={padT}
                      width={Math.max(6, xEnd - Math.max(padL, xStart))}
                      height={plotH}
                      className="obs-gap-band"
                    >
                      <title>{EN.dataRetrieval.legendNotPublished}</title>
                    </rect>
                  );
                })}

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
                <text
                  x={14}
                  y={padT + plotH / 2}
                  transform={`rotate(-90 14 ${padT + plotH / 2})`}
                  textAnchor="middle"
                  className="chart-axis-title"
                >
                  {yAxisTitle}
                </text>

                {segmentPaths.paths.map((seg) => (
                  <path
                    key={seg.key}
                    data-segment={seg.status}
                    d={seg.d}
                    fill="none"
                    stroke={seg.status === "FINALIZED" ? FINALIZED_COLOR : PROVISIONAL_COLOR}
                    strokeWidth={2.4}
                  />
                ))}

                {forecastOverlayPathD && (
                  <path
                    data-segment="FORECAST"
                    d={forecastOverlayPathD}
                    fill="none"
                    stroke={FORECAST_COLOR}
                    strokeWidth={2.6}
                    strokeDasharray="6 4"
                  />
                )}

                {series.map((pt, idx) => {
                  const isLast = idx === series.length - 1;
                  const tooCloseToLast =
                    !isLast &&
                    xTickStep > 1 &&
                    series.length - 1 - idx < Math.max(2, Math.floor(xTickStep * 0.45));
                  if (!isLast && (idx % xTickStep !== 0 || tooCloseToLast)) return null;
                  const anchor = isLast ? "end" : idx === 0 ? "start" : "middle";
                  return (
                    <text
                      key={`lbl_${pt.date}`}
                      x={xAtIdx(idx)}
                      y={svgH - 12}
                      textAnchor={anchor}
                      className="chart-axis-text"
                    >
                      {formatReadableDate(pt.date)}
                    </text>
                  );
                })}
              </svg>
            </div>
          </div>

          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>{EN.dataRetrieval.tableHeaders.period}</th>
                  <th>{EN.dataRetrieval.tableHeaders.periodEnd}</th>
                  <th>{EN.dataRetrieval.tableHeaders.status}</th>
                  <th className="num">{yAxisTitle}</th>
                  <th className="num">{EN.dataRetrieval.tableHeaders.publishedDays}</th>
                </tr>
              </thead>
              <tbody>
                {pagedRows.map((row) => {
                  const v = getValue(row);
                  return (
                    <tr key={`${row.date}_${row.period_end}`}>
                      <td title={row.date}>{formatReadableDate(row.date)}</td>
                      <td className="mono">{row.period_end}</td>
                      <td>
                        <span className={`pill-badge pill-badge--${row.status.toLowerCase()}`}>
                          {EN.dataRetrieval.statusLabels[row.status]}
                        </span>
                      </td>
                      <td className="num">{v !== null ? v.toFixed(2) : EN.forecast.nullValue}</td>
                      <td className="num">{`${row.published_days_count} / ${row.calendar_days_count}`}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="pagination-bar">
            <span>{EN.dataRetrieval.paginationInfo(page, totalPages, series.length)}</span>
            <div className="pagination-buttons">
              <button
                type="button"
                className="btn-chip"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                {EN.dataRetrieval.prevPage}
              </button>
              <button
                type="button"
                className="btn-chip"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              >
                {EN.dataRetrieval.nextPage}
              </button>
            </div>
          </div>
        </>
      ) : null}

      {health && (
        <div className="subpanel status-footnote-panel">
          <h4>{EN.dataRetrieval.scanWindowHeading}</h4>
          <div className="footnote-metrics">
            <span>
              <strong>{EN.dataRetrieval.scanWindowRangeLabel}:</strong>{" "}
              {health.scan_window_start && health.scan_window_end
                ? `${formatReadableDate(health.scan_window_start)} — ${formatReadableDate(health.scan_window_end)} (${health.scan_window_total_days} ${EN.dataRetrieval.daysUnit})`
                : EN.forecast.nullValue}
            </span>
            <span>
              <strong>{EN.dataRetrieval.latestPrecipLabel}:</strong>{" "}
              {formatReadableDate(health.latest_precip_obs_date)}
            </span>
            <span>
              <strong>{EN.dataRetrieval.latestGroupLabel}:</strong>{" "}
              {formatReadableDate(health.data_latency?.groups?.METEO_GEOSIT?.last_valid_date)}
            </span>
          </div>
          {health.latest_partial_obs_explanation && (
            <p className="panel-subtitle">{health.latest_partial_obs_explanation}</p>
          )}
        </div>
      )}
    </section>
  );
};
