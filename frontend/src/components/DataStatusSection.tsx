import React from "react";
import type { HealthResponse } from "../types/api";

interface Props {
  health: HealthResponse | null;
}

const GROUP_LABELS: Record<string, string> = {
  METEO_GEOSIT: "Khí tượng & Thủy văn (24 biến GEOS-IT / MERRA-2)",
  RAD_FLASHFLUX: "Bức xạ Cận Thời Gian Thực (3 biến FLASHFlux)",
  RAD_CERES_SYN1DEG: "Bức xạ Chuẩn Nghiên Cứu (8 biến CERES SYN1deg)",
};

export const DataStatusSection: React.FC<Props> = ({ health }) => {
  if (!health) return null;

  const groups = health.data_latency?.groups ?? {};

  return (
    <section id="tinh-trang-du-lieu" className="panel" aria-labelledby="heading-data">
      <div className="panel-header">
        <div>
          <h2 id="heading-data">3. Tình trạng dữ liệu NASA POWER & Bản chụp DB</h2>
          <p className="panel-subtitle">
            Giám sát độ trễ từng nhóm biến đầu vào và ngày neo <code>COMPLETE_ALL_35</code> hợp lệ mới nhất.
          </p>
        </div>
        <span className="badge badge-neutral">
          Vai trò DB: {health.db_role} ({health.journal_mode ?? "n/a"})
        </span>
      </div>

      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Ngày neo đủ 35 biến mới nhất</div>
          <div className="kpi-value mono">{health.default_complete_anchor_date ?? "—"}</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Ngày quan sát mới nhất trong DB (Snapshot)</div>
          <div className="kpi-value mono">{health.data_snapshot_date ?? "—"}</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Ngày quét NASA POWER gần nhất</div>
          <div className="kpi-value mono">{health.data_latency?.scan_window_end ?? "—"}</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Thời điểm cập nhật gần nhất (UTC)</div>
          <div className="kpi-value mono small">
            {health.data_latency?.last_ingest_at_utc ?? "—"}
          </div>
        </div>
      </div>

      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>Nhóm nguồn dữ liệu</th>
              <th className="num">Ngày hợp lệ mới nhất</th>
              <th className="num">Độ trễ (ngày)</th>
              <th className="num">Số ngày đủ trong cửa sổ quét</th>
            </tr>
          </thead>
          <tbody>
            {["METEO_GEOSIT", "RAD_FLASHFLUX", "RAD_CERES_SYN1DEG"].map((gKey) => {
              const g = groups[gKey];
              return (
                <tr key={gKey}>
                  <td>{GROUP_LABELS[gKey] ?? gKey}</td>
                  <td className="num mono">{g?.last_valid_date ?? "—"}</td>
                  <td className="num mono">
                    {g?.lag_days_vs_scan_end !== null && g?.lag_days_vs_scan_end !== undefined
                      ? `${g.lag_days_vs_scan_end} ngày`
                      : "—"}
                  </td>
                  <td className="num mono">{g?.valid_days_in_window ?? "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="alert alert-warn" role="note">
        <strong>Giới hạn độ trễ bức xạ vệ tinh:</strong> Nhóm CERES SYN1deg trễ khoảng 78–80 ngày
        so với ngày quét gần nhất (và mô hình ML yêu cầu đủ cả 35 biến <code>COMPLETE_ALL_35</code> trong 30
        ngày liên tiếp), nên dự báo thời gian thực chưa được hỗ trợ ở phiên bản hiện tại.
      </div>
    </section>
  );
};
