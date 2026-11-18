import React from "react";
import type { ScoresResponse } from "../types/api";

interface Props {
  scores: ScoresResponse | null;
  loading: boolean;
  errorBanner: string | null;
}

const FIXED_MODEL_ORDER = [
  "SC-35",
  "SC-30",
  "SC-31",
  "BASELINE_CLIMATOLOGY_DOY",
  "BASELINE_PERSISTENCE",
];

const MODEL_LABELS: Record<string, string> = {
  "SC-35": "SC-35 — XGBoost Tweedie F3 (@serving)",
  "SC-30": "SC-30 — LightGBM Tweedie F2 (@shadow_val_champ)",
  "SC-31": "SC-31 — LightGBM Tweedie F3 (@shadow_test_contender)",
  BASELINE_CLIMATOLOGY_DOY: "BASELINE_CLIMATOLOGY_DOY (Trung bình DOY DevTrain)",
  BASELINE_PERSISTENCE: "BASELINE_PERSISTENCE (Mưa ngày neo)",
};

export const ScoresSection: React.FC<Props> = ({ scores, loading, errorBanner }) => {
  if (errorBanner) {
    return (
      <section id="bang-diem" className="panel" aria-labelledby="heading-scores">
        <h2 id="heading-scores">2. Bảng điểm Hậu-Test (`POST_TEST`)</h2>
        <div className="alert alert-error" role="alert">
          {errorBanner}
        </div>
      </section>
    );
  }

  if (loading && !scores) {
    return (
      <section id="bang-diem" className="panel" aria-labelledby="heading-scores">
        <h2 id="heading-scores">2. Bảng điểm Hậu-Test (`POST_TEST`)</h2>
        <div className="skeleton-box" aria-busy="true">
          Đang tải bảng điểm & khoảng tin cậy Moving Block Bootstrap...
        </div>
      </section>
    );
  }

  if (!scores) return null;

  const modelKeys = FIXED_MODEL_ORDER.filter((m) => m in scores.models);
  const mlPairs = Object.entries(scores.pairwise_delta_mae_ci).filter(
    ([pair]) => !pair.includes("BASELINE_")
  );
  const baselinePairs = Object.entries(scores.pairwise_delta_mae_ci).filter(([pair]) =>
    pair.includes("BASELINE_")
  );
  const hasAnyNonSignificantMlPair = mlPairs.some(([, ci]) => ci.significant_at_95pct === false);

  return (
    <section id="bang-diem" className="panel" aria-labelledby="heading-scores">
      <div className="panel-header">
        <div>
          <h2 id="heading-scores">2. Bảng điểm Hậu-Test ({scores.eval_split_regime})</h2>
          <p className="panel-subtitle">
            Khoảng tin cậy bootstrap khối {scores.block_bootstrap.block_length} ngày (
            {scores.block_bootstrap.n_blocks} khối/lần lấy mẫu, B={scores.block_bootstrap.n_bootstrap},
            N={scores.n_origins} ngày neo từ {scores.origin_min} đến {scores.origin_max}).
          </p>
        </div>
        <span className="badge badge-neutral">
          Cỡ mẫu hiệu dụng: {scores.n_origins} ngày neo (không xếp hạng 3 mô hình ML)
        </span>
      </div>

      <h3>2.1. MAE, RMSE, MASE tổng hợp theo thứ tự cố định</h3>
      <p className="helper-note">
        Ba mô hình ML hiển thị theo thứ tự cố định (không gắn thứ hạng 1/2/3).
      </p>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>Mô hình</th>
              <th className="num">MAE (mm) [95% CI]</th>
              <th className="num">RMSE (mm) [95% CI]</th>
              <th className="num">MASE [95% CI]</th>
            </tr>
          </thead>
          <tbody>
            {modelKeys.map((mId) => {
              const entry = scores.models[mId];
              const ov = entry.confidence_intervals_95pct.overall;
              return (
                <tr key={mId}>
                  <td>{MODEL_LABELS[mId] ?? mId}</td>
                  <td className="num mono">
                    <strong>{entry.overall_vs_latest.MAE.toFixed(4)}</strong>{" "}
                    {ov && (
                      <span className="ci-range">
                        [{ov.MAE.ci_lower.toFixed(3)}, {ov.MAE.ci_upper.toFixed(3)}]
                      </span>
                    )}
                  </td>
                  <td className="num mono">
                    {entry.overall_vs_latest.RMSE.toFixed(4)}{" "}
                    {ov && (
                      <span className="ci-range">
                        [{ov.RMSE.ci_lower.toFixed(3)}, {ov.RMSE.ci_upper.toFixed(3)}]
                      </span>
                    )}
                  </td>
                  <td className="num mono">
                    {entry.overall_vs_latest.MASE.toFixed(4)}{" "}
                    {ov && (
                      <span className="ci-range">
                        [{ov.MASE.ci_lower.toFixed(3)}, {ov.MASE.ci_upper.toFixed(3)}]
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h3>2.2. MAE theo từng bước horizon (h1..h7, mm)</h3>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>Mô hình</th>
              {["h1", "h2", "h3", "h4", "h5", "h6", "h7"].map((h) => (
                <th key={h} className="num">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {modelKeys.map((mId) => {
              const byH = scores.models[mId].by_horizon_vs_latest;
              return (
                <tr key={mId}>
                  <td className="mono">{mId}</td>
                  {["h1", "h2", "h3", "h4", "h5", "h6", "h7"].map((h) => (
                    <td key={h} className="num mono">
                      {byH[h]?.MAE?.toFixed(3) ?? "—"}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h3>2.3. Hiệu số cặp ΔMAE & Kiểm định khác biệt (95% CI)</h3>
      {hasAnyNonSignificantMlPair && (
        <div className="alert alert-warn" role="note">
          <strong>Khác biệt giữa các mô hình ML không có ý nghĩa thống kê</strong> ở các cặp có
          khoảng tin cậy 95% chứa 0 (cần đọc đủ từng cặp bên dưới; không dựa vào chênh lệch điểm để
          xếp hạng 3 mô hình ML).
        </div>
      )}
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>Cặp so sánh (A − B)</th>
              <th className="num">ΔMAE (mm)</th>
              <th className="num">95% CI</th>
              <th>Kết luận thống kê (95%)</th>
            </tr>
          </thead>
          <tbody>
            {[...baselinePairs, ...mlPairs].map(([pair, ci]) => (
              <tr key={pair}>
                <td className="mono">{pair}</td>
                <td className="num mono">
                  {ci.point_estimate >= 0 ? `+${ci.point_estimate.toFixed(4)}` : ci.point_estimate.toFixed(4)}
                </td>
                <td className="num mono">
                  [{ci.ci_lower.toFixed(4)}, {ci.ci_upper.toFixed(4)}]
                </td>
                <td>
                  {ci.significant_at_95pct
                    ? "Khác biệt có ý nghĩa thống kê (CI không chứa 0)"
                    : "Không có ý nghĩa thống kê (CI chứa 0)"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>2.4. Ghi chú phương pháp luận bắt buộc</h3>
      <ul className="notes-list">
        {Object.entries(scores.methodological_notes).map(([key, text]) => (
          <li key={key}>
            <code className="note-key">{key}</code>: {text}
          </li>
        ))}
      </ul>
    </section>
  );
};
