import React from "react";
import { EN, formatReadableDate, formatUtcDateTime } from "../i18n/en";
import type { HealthResponse } from "../types/api";

interface DataStatusSectionProps {
  health: HealthResponse | null;
}

const FEED_ORDER = ["METEO_GEOSIT", "RAD_FLASHFLUX", "RAD_CERES_SYN1DEG"] as const;

export const DataStatusSection: React.FC<DataStatusSectionProps> = ({ health }) => {
  const scanStart = health?.scan_window_start ?? health?.data_latency?.scan_window_start ?? null;
  const scanEnd = health?.scan_window_end ?? health?.data_latency?.scan_window_end ?? null;
  const scanDays = health?.scan_window_total_days ?? health?.data_latency?.scan_window_total_days ?? null;
  const groups = health?.data_latency?.groups ?? {};

  return (
    <section className="panel" aria-label={EN.dataStatus.heading}>
      <h2>{EN.dataStatus.heading}</h2>
      <p className="panel-subtitle">{EN.dataStatus.subtitle}</p>

      <div className="metric-grid">
        <div className="metric-card">
          <div className="metric-label">{EN.dataStatus.scanWindowRangeLabel}</div>
          <div className="metric-value">
            {scanStart && scanEnd
              ? `${formatReadableDate(scanStart)} — ${formatReadableDate(scanEnd)}`
              : EN.dataStatus.notAvailable}
          </div>
          {scanStart && scanEnd && (
            <div className="metric-sub">{`${scanStart} .. ${scanEnd}`}</div>
          )}
        </div>
        <div className="metric-card">
          <div className="metric-label">{EN.dataStatus.scanWindowDaysLabel}</div>
          <div className="metric-value">
            {scanDays !== null ? `${scanDays} ${EN.dataStatus.daysUnit}` : EN.dataStatus.notAvailable}
          </div>
          <div className="metric-sub">
            {health?.data_latency?.last_ingest_at_utc
              ? formatUtcDateTime(health.data_latency.last_ingest_at_utc)
              : EN.dataStatus.notAvailable}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">{EN.dataStatus.scanWindowSnapshotLabel}</div>
          <div className="metric-value">{formatReadableDate(health?.data_snapshot_date)}</div>
          <div className="metric-sub">{health?.data_snapshot_date ?? EN.dataStatus.notAvailable}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">{EN.dataStatus.scanWindowCompleteLabel}</div>
          <div className="metric-value">{formatReadableDate(health?.default_complete_anchor_date)}</div>
          <div className="metric-sub">{health?.default_complete_anchor_date ?? EN.dataStatus.notAvailable}</div>
        </div>
      </div>

      <h3>{EN.dataStatus.definitionsHeading}</h3>
      <div className="two-col-grid">
        <div className="subpanel">
          <h4>{EN.dataStatus.latestPrecipTitle}</h4>
          <p className="panel-subtitle">{EN.dataStatus.latestPrecipDesc}</p>
          <div className="highlight-date" title={health?.latest_precip_obs_date ?? undefined}>
            {formatReadableDate(health?.latest_precip_obs_date)}
          </div>
          {health?.latest_precip_obs_date && (
            <div className="mono-sub">{health.latest_precip_obs_date}</div>
          )}
        </div>

        <div className="subpanel">
          <h4>{EN.dataStatus.latestGroupTitle}</h4>
          <p className="panel-subtitle">{EN.dataStatus.latestGroupDesc}</p>
          <ul className="group-date-list">
            {FEED_ORDER.map((groupKey) => {
              const info = groups[groupKey];
              return (
                <li key={groupKey}>
                  <span>{groupKey}</span>
                  <strong title={info?.last_valid_date ?? undefined}>
                    {formatReadableDate(info?.last_valid_date)}
                  </strong>
                </li>
              );
            })}
          </ul>
        </div>
      </div>

      {health?.latest_partial_obs_explanation && (
        <div className="info-banner" role="note" data-testid="partial-obs-explanation">
          <strong>{EN.dataStatus.partialExplanationTitle}:</strong>{" "}
          <span>{health.latest_partial_obs_explanation}</span>
        </div>
      )}

      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>{EN.dataStatus.groupTableHeaders.group}</th>
              <th>{EN.dataStatus.groupTableHeaders.lastValid}</th>
              <th>{EN.dataStatus.groupTableHeaders.lagDays}</th>
              <th>{EN.dataStatus.groupTableHeaders.validDays}</th>
            </tr>
          </thead>
          <tbody>
            {FEED_ORDER.map((groupKey) => {
              const g = groups[groupKey];
              return (
                <tr key={groupKey}>
                  <td>{EN.dataStatus.groupLabels[groupKey] ?? groupKey}</td>
                  <td title={g?.last_valid_date ?? undefined}>
                    {formatReadableDate(g?.last_valid_date)}
                  </td>
                  <td className="num">
                    {g?.lag_days_vs_scan_end !== null && g?.lag_days_vs_scan_end !== undefined
                      ? `${g.lag_days_vs_scan_end} ${EN.dataStatus.daysUnit}`
                      : EN.dataStatus.notAvailable}
                  </td>
                  <td className="num">
                    {g?.valid_days_in_window !== null && g?.valid_days_in_window !== undefined
                      ? g.valid_days_in_window
                      : EN.dataStatus.notAvailable}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
};
