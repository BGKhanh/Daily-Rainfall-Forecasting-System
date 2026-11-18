import React, { useMemo, useState } from "react";
import type {
  DayForecastStatus,
  HealthResponse,
  ModelKey5,
  PredictResponse,
  SplitRegime,
} from "../types/api";

interface Props {
  health: HealthResponse | null;
  prediction: PredictResponse | null;
  loading: boolean;
  errorBanner: string | null;
  selectedAnchor: string;
  onSelectAnchor: (anchorDate: string) => void;
}

type SeriesKey = "observed_mm" | ModelKey5;

interface SeriesMeta {
  key: SeriesKey;
  label: string;
  color: string;
  dashed?: boolean;
  defaultVisible: boolean;
}

const SERIES_CONFIG: SeriesMeta[] = [
  { key: "observed_mm", label: "Quan sát thực tế", color: "#0F766E", defaultVisible: true },
  { key: "SC-35", label: "SC-35 (@serving)", color: "#0B5CAD", defaultVisible: true },
  {
    key: "BASELINE_CLIMATOLOGY_DOY",
    label: "Climatology DOY",
    color: "#64748B",
    dashed: true,
    defaultVisible: true,
  },
  { key: "SC-30", label: "SC-30 (Shadow Val)", color: "#D97706", defaultVisible: false },
  { key: "SC-31", label: "SC-31 (Shadow Test)", color: "#7C3AED", defaultVisible: false },
  {
    key: "BASELINE_PERSISTENCE",
    label: "Persistence",
    color: "#94A3B8",
    dashed: true,
    defaultVisible: false,
  },
];

const STATUS_LABELS: Record<DayForecastStatus, string> = {
  past_observed: "đã có quan sát",
  today: "hôm nay",
  future_forecast: "ngày tương lai",
  past_unverified: "chưa có quan sát",
};

const SPLIT_REGIME_LABELS: Record<SplitRegime, string> = {
  DEV_TRAIN: "DEV_TRAIN — mô hình đã học dữ liệu này",
  VALIDATION: "VALIDATION — dùng để chọn mô hình",
  TEST: "TEST — đã dùng để đánh giá, kết quả đã được xem",
  POST_TEST: "POST_TEST — chưa dùng để chọn mô hình",
};

export const AnchorForecastSection: React.FC<Props> = ({
  health,
  prediction,
  loading,
  errorBanner,
  selectedAnchor,
  onSelectAnchor,
}) => {
  const [inputDate, setInputDate] = useState<string>(selectedAnchor);
  const [visibleSeries, setVisibleSeries] = useState<Record<SeriesKey, boolean>>({
    observed_mm: true,
    "SC-35": true,
    BASELINE_CLIMATOLOGY_DOY: true,
    "SC-30": false,
    "SC-31": false,
    BASELINE_PERSISTENCE: false,
  });

  React.useEffect(() => {
    setInputDate(selectedAnchor);
  }, [selectedAnchor]);

  const toggleSeries = (key: SeriesKey) => {
    setVisibleSeries((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const chartData = useMemo(() => {
    const days = prediction?.daily_forecasts ?? [];
    let maxVal = 10;
    for (const d of days) {
      if (visibleSeries.observed_mm && d.observed_mm !== null) {
        maxVal = Math.max(maxVal, d.observed_mm);
      }
      for (const s of SERIES_CONFIG) {
        if (s.key !== "observed_mm" && visibleSeries[s.key]) {
          maxVal = Math.max(maxVal, d.predictions[s.key] ?? 0);
        }
      }
    }
    return {
      days,
      yMax: Math.ceil((maxVal * 1.15) / 5) * 5 || 15,
    };
  }, [prediction, visibleSeries]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputDate) {
      onSelectAnchor(inputDate);
    }
  };

  const svgW = 680;
  const svgH = 260;
  const padL = 48;
  const padR = 42;
  const padT = 18;
  const padB = 34;
  const plotW = svgW - padL - padR;
  const plotH = svgH - padT - padB;

  const getX = (idx: number, count: number) =>
    count <= 1 ? padL + plotW / 2 : padL + (idx / (count - 1)) * plotW;
  const getY = (val: number, yMax: number) =>
    padT + plotH - Math.min(1, Math.max(0, val / yMax)) * plotH;

  return (
    <section id="theo-ngay-neo" className="panel" aria-labelledby="heading-anchor">
      <div className="panel-header">
        <div>
          <h2 id="heading-anchor">1. Chuỗi 7 ngày sau ngày neo (TP. Hồ Chí Minh)</h2>
          <p className="panel-subtitle">
            Cửa sổ đầu vào 30–35 ngày đạt đủ 35 biến (<code>COMPLETE_ALL_35</code>), xuất chuỗi 7 ngày sau ngày neo.
          </p>
        </div>
        {prediction && (
          <span className={`badge badge-regime-${prediction.anchor_split_regime.toLowerCase()}`}>
            {SPLIT_REGIME_LABELS[prediction.anchor_split_regime]}
          </span>
        )}
      </div>

      <form className="anchor-controls" onSubmit={handleSubmit}>
        <label className="control-label" htmlFor="anchor-date-input">
          Chọn ngày neo:
          <input
            id="anchor-date-input"
            type="date"
            value={inputDate}
            min={health?.anchor_range?.min ?? undefined}
            max={health?.anchor_range?.max ?? undefined}
            onChange={(e) => setInputDate(e.target.value)}
            className="date-input"
          />
        </label>
        <button type="submit" className="btn btn-primary" disabled={loading || !inputDate}>
          {loading ? "Đang tải..." : "Xem chuỗi 7 ngày sau ngày neo"}
        </button>
        {health?.default_complete_anchor_date && (
          <button
            type="button"
            className="btn btn-secondary"
            disabled={loading}
            onClick={() => onSelectAnchor(health.default_complete_anchor_date!)}
          >
            Ngày neo mới nhất ({health.default_complete_anchor_date})
          </button>
        )}
        <button
          type="button"
          className="btn btn-outline-warn"
          disabled={loading}
          onClick={() => {
            setInputDate("2026-10-15");
            onSelectAnchor("2026-10-15");
          }}
        >
          Thử ngày khuyết CERES (2026-10-15)
        </button>
      </form>

      {health?.anchor_range && (
        <p className="helper-note">
          Phạm vi ngày neo hợp lệ trong DB: <strong>{health.anchor_range.min}</strong> đến{" "}
          <strong>{health.anchor_range.max}</strong>.
        </p>
      )}

      {errorBanner && (
        <div className="alert alert-error" role="alert">
          {errorBanner}
        </div>
      )}

      {prediction && prediction.effective_future_days === 0 && (
        <div className="alert alert-warn" role="status">
          <strong>Đây là kết quả hồi cứu, không phải dự báo tương lai.</strong> Ngày chạy{" "}
          <code>{prediction.run_date}</code> cách ngày neo <code>{prediction.anchor_date}</code>{" "}
          {prediction.lag_days} ngày (<code>effective_future_days = 0</code>).
        </div>
      )}

      {loading && !prediction && (
        <div className="skeleton-box" aria-busy="true">
          Đang tính chuỗi 7 ngày sau ngày neo...
        </div>
      )}

      {prediction && (
        <>
          <div className="series-toggles" role="group" aria-label="Bật tắt đường biểu diễn">
            {SERIES_CONFIG.map((s) => (
              <label key={s.key} className="toggle-pill">
                <input
                  type="checkbox"
                  checked={visibleSeries[s.key]}
                  onChange={() => toggleSeries(s.key)}
                />
                <span className="swatch" style={{ backgroundColor: s.color }} />
                <span>{s.label}</span>
              </label>
            ))}
          </div>

          <div className="chart-wrap">
            <svg
              viewBox={`0 0 ${svgW} ${svgH}`}
              className="rain-chart"
              role="img"
              aria-label="Biểu đồ chuỗi 7 ngày sau ngày neo (mm)"
            >
              {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
                const yVal = Math.round(chartData.yMax * frac * 10) / 10;
                const yPos = getY(yVal, chartData.yMax);
                return (
                  <g key={frac}>
                    <line x1={padL} y1={yPos} x2={svgW - padR} y2={yPos} stroke="#E2E8F0" />
                    <text
                      x={padL - 8}
                      y={yPos + 4}
                      textAnchor="end"
                      fontSize="11"
                      fill="#64748B"
                    >
                      {yVal}
                    </text>
                  </g>
                );
              })}

              <text x={12} y={padT + 10} fontSize="11" fill="#475569">
                mm
              </text>

              {chartData.days.map((d, idx) => (
                <text
                  key={d.target_date}
                  x={getX(idx, chartData.days.length)}
                  y={svgH - 10}
                  textAnchor="middle"
                  fontSize="11"
                  fill="#475569"
                >
                  h{d.horizon_day} ({d.target_date.slice(5)})
                </text>
              ))}

              {SERIES_CONFIG.filter((s) => visibleSeries[s.key]).map((s) => {
                const pts = chartData.days
                  .map((d, idx) => {
                    const val =
                      s.key === "observed_mm" ? d.observed_mm : d.predictions[s.key];
                    if (val === null || val === undefined) return null;
                    return {
                      x: getX(idx, chartData.days.length),
                      y: getY(val, chartData.yMax),
                      val,
                      date: d.target_date,
                    };
                  })
                  .filter((p): p is NonNullable<typeof p> => p !== null);

                if (pts.length === 0) return null;
                const pathD = pts
                  .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
                  .join(" ");

                return (
                  <g key={s.key}>
                    <path
                      d={pathD}
                      fill="none"
                      stroke={s.color}
                      strokeWidth={s.key === "SC-35" || s.key === "observed_mm" ? 2.6 : 1.8}
                      strokeDasharray={s.dashed ? "5 4" : undefined}
                    />
                    {pts.map((p) => (
                      <circle
                        key={`${s.key}-${p.date}`}
                        cx={p.x}
                        cy={p.y}
                        r={s.key === "observed_mm" ? 4.2 : 3.4}
                        fill={s.color}
                        stroke="#FFFFFF"
                        strokeWidth={1.2}
                      >
                        <title>{`${s.label} (${p.date}): ${p.val.toFixed(2)} mm`}</title>
                      </circle>
                    ))}
                  </g>
                );
              })}
            </svg>
          </div>

          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Bước</th>
                  <th>Ngày đích</th>
                  <th>Trạng thái</th>
                  <th className="num">Quan sát (mm)</th>
                  <th className="num">SC-35 (@serving)</th>
                  <th className="num">SC-30</th>
                  <th className="num">SC-31</th>
                  <th className="num">Climatology DOY</th>
                  <th className="num">Persistence</th>
                </tr>
              </thead>
              <tbody>
                {prediction.daily_forecasts.map((row) => (
                  <tr key={row.target_date}>
                    <td className="mono">h{row.horizon_day}</td>
                    <td className="mono">{row.target_date}</td>
                    <td>
                      <span className={`status-pill status-${row.status}`}>
                        {STATUS_LABELS[row.status]}
                      </span>
                    </td>
                    <td className="num mono">
                      {row.observed_mm !== null ? row.observed_mm.toFixed(2) : "—"}
                    </td>
                    <td className="num mono strong">{row.predictions["SC-35"].toFixed(2)}</td>
                    <td className="num mono">{row.predictions["SC-30"].toFixed(2)}</td>
                    <td className="num mono">{row.predictions["SC-31"].toFixed(2)}</td>
                    <td className="num mono">
                      {row.predictions["BASELINE_CLIMATOLOGY_DOY"].toFixed(2)}
                    </td>
                    <td className="num mono">
                      {row.predictions["BASELINE_PERSISTENCE"].toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
};
