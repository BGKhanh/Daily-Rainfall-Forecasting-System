import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { HealthResponse, PredictResponse, ScoresResponse } from "../types/api";

const MOCK_HEALTH: HealthResponse = {
  status: "healthy",
  service: "HCMC Rainfall Forecasting Engine",
  version: "1.2.0",
  predictor_loaded: true,
  model_source: "local",
  weather_backend: "local",
  db_role: "bundled_demo_db",
  default_complete_anchor_date: "2026-08-30",
  data_snapshot_date: "2026-11-15",
  anchor_range: { min: "2001-02-04", max: "2026-08-30" },
  data_latency: {
    last_ingest_run_id: "nasa_manual_scan_1",
    last_ingest_at_utc: "2026-11-18T05:17:49.000000Z",
    scan_window_end: "2026-11-18",
    groups: {
      METEO_GEOSIT: { last_valid_date: "2026-11-14", lag_days_vs_scan_end: 4, valid_days_in_window: 206 },
      RAD_FLASHFLUX: { last_valid_date: "2026-11-13", lag_days_vs_scan_end: 5, valid_days_in_window: 204 },
      RAD_CERES_SYN1DEG: { last_valid_date: "2026-08-30", lag_days_vs_scan_end: 80, valid_days_in_window: 130 },
    },
  },
  journal_mode: "delete",
};

const MOCK_PREDICT: PredictResponse = {
  anchor_date: "2026-08-30",
  anchor_split_regime: "POST_TEST",
  run_date: "2026-11-18",
  run_timezone: "Asia/Ho_Chi_Minh",
  effective_future_days: 0,
  future_forecast_notice: "Không có ngày tương lai (effective_future_days = 0).",
  db_role: "bundled_demo_db",
  run_timestamp: "2026-11-18T05:00:00.000000Z",
  model_source: "local",
  history_backend: "sqlite",
  data_as_of: "2026-08-30",
  lag_days: 80,
  forecast_horizon_mode: "retrospective_hindcast_only",
  anchor_completeness_regime: "COMPLETE_ALL_35",
  daily_forecasts: [
    {
      target_date: "2026-08-31",
      horizon_day: 1,
      status: "past_observed",
      observed_mm: 5.2,
      predictions: {
        "SC-35": 4.8,
        "SC-30": 4.9,
        "SC-31": 4.7,
        BASELINE_CLIMATOLOGY_DOY: 6.1,
        BASELINE_PERSISTENCE: 2.0,
      },
    },
  ],
};

const MOCK_SCORES: ScoresResponse = {
  location_id: "HCMC",
  db_role: "bundled_demo_db",
  eval_split_regime: "POST_TEST",
  is_prospective_filter: null,
  mase_scale: 3.701542,
  origin_min: "2026-05-01",
  origin_max: "2026-08-30",
  target_date_min: "2026-05-02",
  target_date_max: "2026-09-06",
  n_origins: 122,
  n_forecasts_per_model: 854,
  total_scored_rows: 4270,
  models: {
    "SC-35": {
      model_id: "SC-35",
      model_kind: "ML_SERVING",
      n_origins: 122,
      n_forecasts: 854,
      overall_vs_first: { MAE: 4.992, MSE: 68.0, RMSE: 8.248, MASE: 1.348 },
      overall_vs_latest: { MAE: 4.992, MSE: 68.0, RMSE: 8.248, MASE: 1.348 },
      by_horizon_vs_first: { h1: { MAE: 4.4, MSE: 62.0, RMSE: 7.9, MASE: 1.2 } },
      by_horizon_vs_latest: { h1: { MAE: 4.4, MSE: 62.0, RMSE: 7.9, MASE: 1.2 } },
      confidence_intervals_95pct: {
        block_length: 7,
        n_blocks: 18,
        overall: {
          MAE: { point_estimate: 4.992, ci_lower: 4.13, ci_upper: 6.28, ci_width: 2.15, bootstrap_se: 0.54, significant_at_95pct: null },
          RMSE: { point_estimate: 8.248, ci_lower: 6.32, ci_upper: 10.38, ci_width: 4.06, bootstrap_se: 1.03, significant_at_95pct: null },
          MASE: { point_estimate: 1.348, ci_lower: 1.11, ci_upper: 1.7, ci_width: 0.59, bootstrap_se: 0.15, significant_at_95pct: null },
        },
        by_horizon: {},
      },
    },
    "SC-31": {
      model_id: "SC-31",
      model_kind: "ML_SHADOW",
      n_origins: 122,
      n_forecasts: 854,
      overall_vs_first: { MAE: 4.923, MSE: 68.6, RMSE: 8.283, MASE: 1.33 },
      overall_vs_latest: { MAE: 4.923, MSE: 68.6, RMSE: 8.283, MASE: 1.33 },
      by_horizon_vs_first: { h1: { MAE: 4.3, MSE: 62.0, RMSE: 7.9, MASE: 1.16 } },
      by_horizon_vs_latest: { h1: { MAE: 4.3, MSE: 62.0, RMSE: 7.9, MASE: 1.16 } },
      confidence_intervals_95pct: {
        block_length: 7,
        n_blocks: 18,
        overall: {
          MAE: { point_estimate: 4.923, ci_lower: 4.01, ci_upper: 6.28, ci_width: 2.27, bootstrap_se: 0.56, significant_at_95pct: null },
          RMSE: { point_estimate: 8.283, ci_lower: 6.32, ci_upper: 10.44, ci_width: 4.12, bootstrap_se: 1.05, significant_at_95pct: null },
          MASE: { point_estimate: 1.33, ci_lower: 1.08, ci_upper: 1.7, ci_width: 0.62, bootstrap_se: 0.15, significant_at_95pct: null },
        },
        by_horizon: {},
      },
    },
  },
  pairwise_delta_mae_ci: {
    "SC-35 - SC-31": {
      point_estimate: 0.0699,
      ci_lower: -0.019,
      ci_upper: 0.1644,
      ci_width: 0.1834,
      bootstrap_se: 0.0468,
      significant_at_95pct: false,
    },
  },
  block_bootstrap: {
    method: "moving_block_bootstrap",
    block_length: 7,
    n_blocks: 18,
    n_bootstrap: 2000,
    seed: 42,
    n_anchor_dates: 122,
  },
  methodological_notes: {
    effective_sample_size: "122 unique anchor dates",
    first_vs_latest_equality: "single ingested version",
    seasonal_regime_note: "wet season monsoon window",
  },
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("HCMC Rainfall Forecasting UI v1", () => {
  it("renders retrospective banner when effective_future_days=0 and never contains forbidden phrasing", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/health")) {
        return Promise.resolve(new Response(JSON.stringify(MOCK_HEALTH), { status: 200 }));
      }
      if (url.includes("/scores")) {
        return Promise.resolve(new Response(JSON.stringify(MOCK_SCORES), { status: 200 }));
      }
      return Promise.resolve(new Response(JSON.stringify(MOCK_PREDICT), { status: 200 }));
    });

    const { container } = render(<App />);

    await waitFor(() => {
      expect(
        screen.getByText(/Đây là kết quả hồi cứu, không phải dự báo tương lai\./i)
      ).toBeInTheDocument();
    });

    expect(
      screen.getByText(/Khác biệt giữa các mô hình ML không có ý nghĩa thống kê/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/Khoảng tin cậy bootstrap khối 7 ngày/i)).toBeInTheDocument();
    expect(container.textContent).not.toMatch(/dự báo 7 ngày tới/i);
  });

  it("maps HTTP 422 INCOMPLETE_RADIATION_AT_ANCHOR and HTTP 503 SERVER_DB_DATA_ERROR properly", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/health")) {
        return Promise.resolve(new Response(JSON.stringify(MOCK_HEALTH), { status: 200 }));
      }
      if (url.includes("/scores")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              error: { code: "SERVER_DB_DATA_ERROR", message: "DB failed", details: {} },
            }),
            { status: 503 }
          )
        );
      }
      const bodyStr = typeof init?.body === "string" ? init.body : "";
      if (bodyStr.includes("2026-10-15")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              error: {
                code: "INCOMPLETE_RADIATION_AT_ANCHOR",
                message: "MISSING_CERES_ONLY",
                details: { anchor_date: "2026-10-15" },
              },
            }),
            { status: 422 }
          )
        );
      }
      return Promise.resolve(new Response(JSON.stringify(MOCK_PREDICT), { status: 200 }));
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/Lỗi máy chủ dữ liệu, thử lại sau/i)).toBeInTheDocument();
    });

    const test422Btn = screen.getByRole("button", { name: /Thử ngày khuyết CERES/i });
    fireEvent.click(test422Btn);

    await waitFor(() => {
      expect(screen.getByText(/Ngày neo không hợp lệ: MISSING_CERES_ONLY/i)).toBeInTheDocument();
    });
  });

  it("shows cold start waiting banner when /health exceeds threshold", async () => {
    vi.useFakeTimers();
    try {
      vi.spyOn(globalThis, "fetch").mockImplementation(() => new Promise(() => {}));
      render(<App />);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(3100);
      });
      expect(screen.getByText(/Máy chủ đang khởi động, vui lòng chờ/i)).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});
