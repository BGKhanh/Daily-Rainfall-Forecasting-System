import React from "react";
import { EN, formatReadableDate, mapApiErrorToFriendly } from "../i18n/en";
import type { ApiClientError } from "../api/client";
import type { ScoresResponse } from "../types/api";

interface ScoresSectionProps {
  scores: ScoresResponse | null;
  loading: boolean;
  errorValue: unknown;
}

const MODEL_ORDER = [
  "SC-31",
  "SC-30",
  "SC-35",
  "BASELINE_CLIMATOLOGY_DOY",
  "BASELINE_PERSISTENCE",
] as const;

const PAIR_ORDER = [
  "SC-35 - BASELINE_CLIMATOLOGY_DOY",
  "SC-31 - BASELINE_CLIMATOLOGY_DOY",
  "SC-30 - BASELINE_CLIMATOLOGY_DOY",
  "SC-35 - SC-31",
  "SC-35 - SC-30",
  "SC-30 - SC-31",
] as const;

const SENSITIVITY_BLOCKS = ["7", "14", "21"] as const;

export const ScoresSection: React.FC<ScoresSectionProps> = ({ scores, loading, errorValue }) => {
  if (errorValue) {
    const errObj = errorValue as Partial<ApiClientError>;
    const friendly = mapApiErrorToFriendly({
      code: errObj.code,
      message: errObj.message,
      details: errObj.details,
      status: errObj.status,
    });
    return (
      <section className="panel" aria-label={EN.scores.heading}>
        <h2>{EN.scores.heading}</h2>
        <div className="error-state-panel" role="alert">
          <p className="error-summary">{friendly.summary}</p>
          <details className="technical-details">
            <summary>{EN.errors.technicalDetailsTitle}</summary>
            <div className="technical-details-body">
              <div>{friendly.code}</div>
              <div>{friendly.message}</div>
            </div>
          </details>
        </div>
      </section>
    );
  }

  if (loading || !scores) {
    return (
      <section className="panel" aria-label={EN.scores.heading}>
        <h2>{EN.scores.heading}</h2>
        <div className="empty-box">{EN.scores.loadingText}</div>
      </section>
    );
  }

  const sensitivity = scores.block_length_sensitivity ?? {};

  return (
    <section className="panel" aria-label={EN.scores.heading}>
      <h2>{EN.scores.heading}</h2>
      <p className="panel-subtitle">{EN.scores.subtitle}</p>

      <div className="metric-grid">
        <div className="metric-card">
          <div className="metric-label">{EN.scores.metaOrigins}</div>
          <div className="metric-value">{scores.n_origins}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">{EN.scores.metaForecasts}</div>
          <div className="metric-value">{scores.n_forecasts_per_model}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">{EN.scores.metaWindow}</div>
          <div className="metric-value">
            {`${formatReadableDate(scores.origin_min)} — ${formatReadableDate(scores.origin_max)}`}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">{EN.scores.metaMaseScale}</div>
          <div className="metric-value">{`${scores.mase_scale.toFixed(4)} ${EN.forecast.millimeterUnit}`}</div>
        </div>
      </div>

      <h3>{EN.scores.modelsHeading}</h3>
      <div className="table-scroll">
        <table className="data-table">
          <caption>{EN.scores.modelsCaption}</caption>
          <thead>
            <tr>
              <th>{EN.scores.modelHeaders.model}</th>
              <th>{EN.scores.modelHeaders.role}</th>
              <th>{EN.scores.modelHeaders.mae}</th>
              <th>{EN.scores.modelHeaders.maeCi}</th>
              <th>{EN.scores.modelHeaders.rmse}</th>
              <th>{EN.scores.modelHeaders.mase}</th>
              <th>{EN.scores.modelHeaders.maseCi}</th>
            </tr>
          </thead>
          <tbody>
            {MODEL_ORDER.filter((m) => scores.models[m]).map((modelId) => {
              const entry = scores.models[modelId];
              const ciOverall = entry.confidence_intervals_95pct?.overall;
              return (
                <tr key={modelId}>
                  <td>
                    <strong>{modelId}</strong>
                  </td>
                  <td>{entry.model_kind}</td>
                  <td className="num">{entry.overall_vs_latest.MAE.toFixed(3)}</td>
                  <td className="num">
                    {ciOverall
                      ? `[${ciOverall.MAE.ci_lower.toFixed(3)}, ${ciOverall.MAE.ci_upper.toFixed(3)}]`
                      : EN.forecast.nullValue}
                  </td>
                  <td className="num">{entry.overall_vs_latest.RMSE.toFixed(3)}</td>
                  <td className="num">{entry.overall_vs_latest.MASE.toFixed(3)}</td>
                  <td className="num">
                    {ciOverall
                      ? `[${ciOverall.MASE.ci_lower.toFixed(3)}, ${ciOverall.MASE.ci_upper.toFixed(3)}]`
                      : EN.forecast.nullValue}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h3>{EN.scores.pairwiseHeading}</h3>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>{EN.scores.pairwiseHeaders.pair}</th>
              <th>{EN.scores.pairwiseHeaders.delta}</th>
              <th>{EN.scores.pairwiseHeaders.ci}</th>
              <th>{EN.scores.pairwiseHeaders.width}</th>
              <th>{EN.scores.pairwiseHeaders.decision}</th>
            </tr>
          </thead>
          <tbody>
            {PAIR_ORDER.filter((p) => scores.pairwise_delta_mae_ci[p]).map((pairKey) => {
              const ci = scores.pairwise_delta_mae_ci[pairKey];
              return (
                <tr key={pairKey}>
                  <td>{pairKey}</td>
                  <td className="num">{ci.point_estimate.toFixed(3)}</td>
                  <td className="num">{`[${ci.ci_lower.toFixed(3)}, ${ci.ci_upper.toFixed(3)}]`}</td>
                  <td className="num">{ci.ci_width.toFixed(3)}</td>
                  <td>{ci.decision_label ?? EN.forecast.nullValue}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {Object.keys(sensitivity).length > 0 && (
        <>
          <h3>{EN.scores.sensitivityHeading}</h3>
          <p className="panel-subtitle">{EN.scores.sensitivitySubtitle}</p>
          <div className="table-scroll">
            <table className="data-table" data-testid="sensitivity-table">
              <thead>
                <tr>
                  <th>{EN.scores.sensitivityHeaders.blockLength}</th>
                  <th>{EN.scores.sensitivityHeaders.blocks}</th>
                  <th>{EN.scores.sensitivityHeaders.sc35Mae}</th>
                  <th>{EN.scores.sensitivityHeaders.sc31Mae}</th>
                  <th>{EN.scores.sensitivityHeaders.sc30Mae}</th>
                  <th>{EN.scores.sensitivityHeaders.sc30vs31}</th>
                </tr>
              </thead>
              <tbody>
                {SENSITIVITY_BLOCKS.filter((b) => sensitivity[b]).map((blkKey) => {
                  const row = sensitivity[blkKey];
                  const sc35 = row.models["SC-35"];
                  const sc31 = row.models["SC-31"];
                  const sc30 = row.models["SC-30"];
                  const pair3031 = row.pairwise_delta_mae["SC-30 - SC-31"];
                  return (
                    <tr key={blkKey}>
                      <td className="num">{`${row.block_length} ${EN.dataStatus.daysUnit}`}</td>
                      <td className="num">{row.n_blocks}</td>
                      <td className="num">
                        {sc35
                          ? `${sc35.MAE.toFixed(3)} [${sc35.ci_lower.toFixed(3)}, ${sc35.ci_upper.toFixed(3)}]`
                          : EN.forecast.nullValue}
                      </td>
                      <td className="num">
                        {sc31
                          ? `${sc31.MAE.toFixed(3)} [${sc31.ci_lower.toFixed(3)}, ${sc31.ci_upper.toFixed(3)}]`
                          : EN.forecast.nullValue}
                      </td>
                      <td className="num">
                        {sc30
                          ? `${sc30.MAE.toFixed(3)} [${sc30.ci_lower.toFixed(3)}, ${sc30.ci_upper.toFixed(3)}]`
                          : EN.forecast.nullValue}
                      </td>
                      <td className="num">
                        {pair3031
                          ? `${pair3031.point_estimate.toFixed(3)} [${pair3031.ci_lower.toFixed(3)}, ${pair3031.ci_upper.toFixed(3)}] (${pair3031.ci_width.toFixed(3)})`
                          : EN.forecast.nullValue}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      <h3>{EN.scores.notesHeading}</h3>
      <div className="notes-grid" data-testid="methodological-notes">
        {Object.entries(scores.methodological_notes).map(([noteKey, noteText]) => (
          <div key={noteKey} className="note-card">
            <h4>{EN.scores.noteTitles[noteKey] ?? noteKey}</h4>
            <p>{noteText}</p>
          </div>
        ))}
      </div>
    </section>
  );
};
