import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { EN } from "../i18n/en";
import type {
  HealthResponse,
  LocationsResponse,
  PredictResponse,
  ProvinceGeoJson,
  ScoresResponse,
} from "../types/api";

const MOCK_GEOJSON: ProvinceGeoJson = {
  type: "FeatureCollection",
  metadata: { version: "Pre-merger 63 provinces + Hoang Sa Island + Truong Sa Island", feature_count: 4 },
  features: [
    {
      type: "Feature",
      properties: {
        polygon_code: "VN_HCMC",
        location_id: "HCMC",
        display_name: "Ho Chi Minh City",
        source_iso: "VN-SG",
        source: "geoBoundaries",
        centroid_lat: 10.7769,
        centroid_lon: 106.7009,
      },
      geometry: {
        type: "Polygon",
        coordinates: [[[106.4, 10.4], [107.0, 10.4], [107.0, 11.1], [106.4, 11.1], [106.4, 10.4]]],
      },
    },
    {
      type: "Feature",
      properties: {
        polygon_code: "VN_HANOI",
        location_id: "HANOI",
        display_name: "Ha Noi",
        source_iso: "VN-HN",
        source: "geoBoundaries",
        centroid_lat: 21.0285,
        centroid_lon: 105.8542,
      },
      geometry: {
        type: "Polygon",
        coordinates: [[[105.5, 20.8], [106.0, 20.8], [106.0, 21.3], [105.5, 21.3], [105.5, 20.8]]],
      },
    },
    {
      type: "Feature",
      properties: {
        polygon_code: "VN_HOANG_SA",
        location_id: "HOANG_SA",
        display_name: "Hoang Sa Island",
        source_iso: "",
        source: "Free-GIS-Data",
        centroid_lat: 16.5333,
        centroid_lon: 111.6,
      },
      geometry: {
        type: "Polygon",
        coordinates: [[[111.2, 16.2], [111.8, 16.2], [111.8, 16.8], [111.2, 16.8], [111.2, 16.2]]],
      },
    },
    {
      type: "Feature",
      properties: {
        polygon_code: "VN_TRUONG_SA",
        location_id: "TRUONG_SA",
        display_name: "Truong Sa Island",
        source_iso: "",
        source: "Free-GIS-Data",
        centroid_lat: 8.6439,
        centroid_lon: 111.9197,
      },
      geometry: {
        type: "Polygon",
        coordinates: [[[111.5, 8.3], [112.2, 8.3], [112.2, 8.9], [111.5, 8.9], [111.5, 8.3]]],
      },
    },
  ],
};

const MOCK_LOCATIONS: LocationsResponse = {
  db_role: "bundled_demo_db",
  locations: [
    {
      location_id: "HCMC",
      display_name: "Ho Chi Minh City",
      latitude: 10.7769,
      longitude: 106.7009,
      meteo_grid_note: "MERRA-2 / GEOS-IT 0.5° x 0.625°",
      rad_grid_note: "CERES SYN1deg / FLASHFlux 1.0° x 1.0°",
      time_standard: "LST",
      is_active: true,
    },
  ],
};

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
  scan_window_start: "2026-04-23",
  scan_window_end: "2026-11-18",
  scan_window_total_days: 210,
  latest_precip_obs_date: "2026-11-15",
  latest_partial_obs_explanation:
    "On 2026-11-15, precipitation (3.32 mm) and 19 of 24 meteorology variables are present, while 5 remain missing (T2M_MAX, WS2M_MIN, WS2M_RANGE, WS10M_MIN, GWETPROF), so METEO_GEOSIT latest valid day is 2026-11-14.",
  anchor_range: { min: "2001-02-04", max: "2026-08-30" },
  data_latency: {
    last_ingest_run_id: "nasa_manual_scan_1",
    last_ingest_at_utc: "2026-11-18T05:17:49.000000Z",
    scan_window_start: "2026-04-23",
    scan_window_end: "2026-11-18",
    scan_window_total_days: 210,
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
  future_forecast_notice:
    "This is a retrospective replay (effective_future_days = 0): latest complete anchor is over 7 days before today.",
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
    "SC-30": {
      model_id: "SC-30",
      model_kind: "ML_SHADOW",
      n_origins: 122,
      n_forecasts: 854,
      overall_vs_first: { MAE: 4.98, MSE: 68.2, RMSE: 8.26, MASE: 1.345 },
      overall_vs_latest: { MAE: 4.98, MSE: 68.2, RMSE: 8.26, MASE: 1.345 },
      by_horizon_vs_first: { h1: { MAE: 4.35, MSE: 62.0, RMSE: 7.9, MASE: 1.18 } },
      by_horizon_vs_latest: { h1: { MAE: 4.35, MSE: 62.0, RMSE: 7.9, MASE: 1.18 } },
      confidence_intervals_95pct: {
        block_length: 7,
        n_blocks: 18,
        overall: {
          MAE: { point_estimate: 4.98, ci_lower: 4.1, ci_upper: 6.26, ci_width: 2.16, bootstrap_se: 0.54, significant_at_95pct: null },
          RMSE: { point_estimate: 8.26, ci_lower: 6.3, ci_upper: 10.4, ci_width: 4.1, bootstrap_se: 1.04, significant_at_95pct: null },
          MASE: { point_estimate: 1.345, ci_lower: 1.1, ci_upper: 1.69, ci_width: 0.59, bootstrap_se: 0.15, significant_at_95pct: null },
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
      ci_upper: 0.164,
      ci_width: 0.183,
      bootstrap_se: 0.047,
      significant_at_95pct: false,
      decision_label: "Not statistically significant (95% CI includes 0)",
    },
    "SC-30 - SC-31": {
      point_estimate: 0.057,
      ci_lower: -0.002,
      ci_upper: 0.118,
      ci_width: 0.12,
      bootstrap_se: 0.031,
      significant_at_95pct: false,
      decision_label: "Borderline, not adjusted for multiple comparisons",
    },
  },
  block_bootstrap: {
    method: "moving_block_bootstrap",
    block_length: 7,
    n_blocks: 18,
    candidate_blocks: 116,
    n_bootstrap: 2000,
    seed: 42,
    n_anchor_dates: 122,
  },
  block_length_sensitivity: {
    "7": {
      block_length: 7,
      n_anchor_dates: 122,
      n_blocks: 18,
      candidate_blocks: 116,
      models: {
        "SC-35": { MAE: 4.992, ci_lower: 4.13, ci_upper: 6.28, ci_width: 2.15, bootstrap_se: 0.54 },
        "SC-31": { MAE: 4.923, ci_lower: 4.01, ci_upper: 6.28, ci_width: 2.27, bootstrap_se: 0.56 },
        "SC-30": { MAE: 4.98, ci_lower: 4.1, ci_upper: 6.26, ci_width: 2.16, bootstrap_se: 0.54 },
      },
      pairwise_delta_mae: {
        "SC-30 - SC-31": {
          point_estimate: 0.057,
          ci_lower: -0.002,
          ci_upper: 0.118,
          ci_width: 0.12,
          bootstrap_se: 0.031,
          significant_at_95pct: false,
        },
      },
    },
    "14": {
      block_length: 14,
      n_anchor_dates: 122,
      n_blocks: 9,
      candidate_blocks: 109,
      models: {
        "SC-35": { MAE: 4.992, ci_lower: 3.9, ci_upper: 6.5, ci_width: 2.6, bootstrap_se: 0.65 },
        "SC-31": { MAE: 4.923, ci_lower: 3.8, ci_upper: 6.48, ci_width: 2.68, bootstrap_se: 0.67 },
        "SC-30": { MAE: 4.98, ci_lower: 3.85, ci_upper: 6.49, ci_width: 2.64, bootstrap_se: 0.66 },
      },
      pairwise_delta_mae: {
        "SC-30 - SC-31": {
          point_estimate: 0.057,
          ci_lower: -0.01,
          ci_upper: 0.13,
          ci_width: 0.14,
          bootstrap_se: 0.035,
          significant_at_95pct: false,
        },
      },
    },
    "21": {
      block_length: 21,
      n_anchor_dates: 122,
      n_blocks: 6,
      candidate_blocks: 102,
      models: {
        "SC-35": { MAE: 4.992, ci_lower: 3.7, ci_upper: 6.7, ci_width: 3.0, bootstrap_se: 0.75 },
        "SC-31": { MAE: 4.923, ci_lower: 3.65, ci_upper: 6.68, ci_width: 3.03, bootstrap_se: 0.76 },
        "SC-30": { MAE: 4.98, ci_lower: 3.68, ci_upper: 6.69, ci_width: 3.01, bootstrap_se: 0.75 },
      },
      pairwise_delta_mae: {
        "SC-30 - SC-31": {
          point_estimate: 0.057,
          ci_lower: -0.02,
          ci_upper: 0.145,
          ci_width: 0.165,
          bootstrap_se: 0.04,
          significant_at_95pct: false,
        },
      },
    },
  },
  methodological_notes: {
    effective_sample_size: "122 unique anchor dates (2026-05-01..2026-08-30), NOT 854 independent observations.",
    first_vs_latest_equality: "vs_first and vs_latest metrics are identical because version_num = 0.",
    seasonal_regime_note: "The 2026-05-01..2026-08-30 window lies within the wet season.",
  },
};

function installFetchMock(options?: {
  locations?: LocationsResponse;
  onPredict?: (body: Record<string, unknown>) => Response;
}) {
  const requestedUrls: string[] = [];
  const predictCalls: Array<Record<string, unknown>> = [];

  const mockFn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    requestedUrls.push(url);

    // Zero external requests rule
    if (/^https?:\/\//i.test(url) && !url.startsWith("http://localhost") && !url.startsWith("http://127.0.0.1")) {
      throw new Error(`External network request prohibited: ${url}`);
    }

    if (url.endsWith("/health")) {
      return new Response(JSON.stringify(MOCK_HEALTH), { status: 200 });
    }
    if (url.endsWith("/locations")) {
      return new Response(JSON.stringify(options?.locations ?? MOCK_LOCATIONS), { status: 200 });
    }
    if (url.startsWith("/scores")) {
      return new Response(JSON.stringify(MOCK_SCORES), { status: 200 });
    }
    if (url.endsWith("/map/vietnam_63_provinces.geojson")) {
      return new Response(JSON.stringify(MOCK_GEOJSON), { status: 200 });
    }
    if (url.endsWith("/map/polygon_to_location.json")) {
      return new Response(JSON.stringify({ administrative_version: "pre-merger", polygons: {} }), { status: 200 });
    }
    if (url.endsWith("/predict")) {
      const body = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : {};
      predictCalls.push(body);
      if (options?.onPredict) {
        return options.onPredict(body);
      }
      return new Response(JSON.stringify(MOCK_PREDICT), { status: 200 });
    }
    return new Response("Not Found", { status: 404 });
  });

  vi.stubGlobal("fetch", mockFn);
  return { requestedUrls, predictCalls, mockFn };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const VIETNAMESE_SCRIPT_RE =
  /[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]/;

describe("Round 9b English App Shell & Map Integration", () => {
  it("renders English retrospective replay, unambiguous dates, zero banned phrases, zero Vietnamese text, and zero external requests", async () => {
    const { requestedUrls } = installFetchMock();
    const { container } = render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("retrospective-banner")).toBeInTheDocument();
    });

    // Title, banner, date format "30 Aug 2026" and observed value
    expect(screen.getByText("7 days after anchor date 30 Aug 2026 (retrospective replay)")).toBeInTheDocument();
    expect(screen.getByText(EN.forecast.retrospectiveBanner)).toBeInTheDocument();
    expect(screen.getAllByText("5.20").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Hoang Sa Island").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Truong Sa Island").length).toBeGreaterThanOrEqual(1);

    // Check every rendered text node across all 4 tabs for Vietnamese script & banned retrospective phrases
    for (const tabLabel of [
      EN.tabs.forecast,
      EN.tabs.data,
      EN.tabs.scores,
      EN.tabs.governance,
    ]) {
      fireEvent.click(screen.getByRole("button", { name: tabLabel }));
      await waitFor(() => {
        const text = container.textContent ?? "";
        expect(VIETNAMESE_SCRIPT_RE.test(text)).toBe(false);
        const sanitized = text.toLowerCase().replace(/not a measuring station/g, "");
        expect(sanitized).not.toContain("next 7 days");
        expect(sanitized).not.toContain("station");
      });
    }

    // Also inspect Tab 1 retrospective dock does not say "7-day forecast"
    fireEvent.click(screen.getByRole("button", { name: EN.tabs.forecast }));
    const dockText = screen.getByTestId("result-dock").textContent?.toLowerCase() ?? "";
    expect(dockText).not.toContain("7-day forecast");

    // Same-origin fetches only
    for (const url of requestedUrls) {
      expect(url.startsWith("/")).toBe(true);
    }
  });

  it("hides previous result on error and renders friendly message + expandable technical details", async () => {
    let failMode = false;
    installFetchMock({
      onPredict: (body) => {
        if (failMode || body.anchor_date === "2026-10-15") {
          return new Response(
            JSON.stringify({
              error: {
                code: "INCOMPLETE_RADIATION_AT_ANCHOR",
                message: "Anchor 2026-10-15 lacks CERES data.",
                details: { anchor_date: "2026-10-15", completeness_regime: "MISSING_CERES_ONLY" },
              },
            }),
            { status: 422 }
          );
        }
        return new Response(JSON.stringify(MOCK_PREDICT), { status: 200 });
      },
    });

    render(<App />);
    await waitFor(() => {
      expect(screen.getByTestId("retrospective-banner")).toBeInTheDocument();
    });

    failMode = true;
    const dateInput = screen.getByLabelText(/^Anchor date/i, { selector: "input" }) as HTMLInputElement;
    fireEvent.change(dateInput, { target: { value: "2026-10-15" } });
    fireEvent.click(screen.getByRole("button", { name: EN.forecast.runButton }));

    await waitFor(() => {
      expect(screen.getByTestId("anchor-error-panel")).toBeInTheDocument();
    });

    expect(screen.queryByTestId("retrospective-banner")).toBeNull();
    expect(screen.getByText("Result for anchor date 15 Oct 2026")).toBeInTheDocument();
    expect(screen.getByText(EN.errors.byCode.INCOMPLETE_RADIATION_AT_ANCHOR)).toBeInTheDocument();
    expect(screen.getByText(EN.errors.technicalDetailsTitle)).toBeInTheDocument();
  });

  it("clicking an unsupported province shows neutral info message and fires zero /predict requests", async () => {
    const { predictCalls } = installFetchMock();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("retrospective-banner")).toBeInTheDocument();
    });
    const baselinePredictCalls = predictCalls.length;

    const hanoiPath = screen.getAllByRole("button", { name: "Ha Noi, not supported" })[0];
    fireEvent.click(hanoiPath);

    await waitFor(() => {
      expect(screen.getByTestId("unsupported-location-banner")).toBeInTheDocument();
    });
    expect(
      screen.getByText(EN.map.unsupportedNotice("Ha Noi"))
    ).toBeInTheDocument();
    expect(predictCalls.length).toBe(baselinePredictCalls);
  });

  it("dynamically marks a second location from GET /locations as supported without frontend code changes", async () => {
    const { predictCalls } = installFetchMock({
      locations: {
        db_role: "local_or_production_db",
        locations: [
          ...MOCK_LOCATIONS.locations,
          {
            location_id: "HANOI",
            display_name: "Ha Noi",
            latitude: 21.0285,
            longitude: 105.8542,
            meteo_grid_note: "MERRA-2 / GEOS-IT 0.5° x 0.625°",
            rad_grid_note: "CERES SYN1deg / FLASHFlux 1.0° x 1.0°",
            time_standard: "LST",
            is_active: true,
          },
        ],
      },
    });

    render(<App />);
    await waitFor(() => {
      expect(screen.getAllByRole("button", { name: "Ha Noi, supported" }).length).toBeGreaterThanOrEqual(1);
    });

    fireEvent.click(screen.getAllByRole("button", { name: "Ha Noi, supported" })[0]);
    await waitFor(() => {
      expect(predictCalls.some((c) => c.location_id === "HANOI")).toBe(true);
    });
  });

  it("supports accent-insensitive search, 375px vertical day cards, and Tab 2 & Tab 3 details", async () => {
    Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: 375 });
    installFetchMock();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("mobile-day-cards")).toBeInTheDocument();
    });

    // Verify zero horizontal overflow at 375px
    expect(document.documentElement.scrollWidth).toBeLessThanOrEqual(window.innerWidth);

    // At 375px, every numeric field is visible in vertical cards
    const mobileCards = within(screen.getByTestId("mobile-day-cards"));
    expect(mobileCards.getByText("5.20")).toBeInTheDocument();
    expect(mobileCards.getByText("4.80")).toBeInTheDocument();
    expect(mobileCards.getByText("6.10")).toBeInTheDocument();
    expect(mobileCards.getByText("4.90")).toBeInTheDocument();
    expect(mobileCards.getByText("4.70")).toBeInTheDocument();
    expect(mobileCards.getByText("2.00")).toBeInTheDocument();

    // Accent-insensitive search for "Hồ Chí Minh" matches Ho Chi Minh City & filters others
    const searchInput = screen.getByLabelText(EN.map.searchLabel);
    fireEvent.change(searchInput, { target: { value: "Hồ Chí Minh" } });
    const listBtns = within(screen.getByRole("list", { name: EN.map.listHeading })).getAllByRole("button");
    expect(listBtns.length).toBe(1);
    expect(listBtns[0]).toHaveTextContent("Ho Chi Minh City");

    // Tab 2 shows dynamic scan window & partial 15 Nov explanation from /health
    fireEvent.click(screen.getByRole("button", { name: EN.tabs.data }));
    expect(screen.getByTestId("partial-obs-explanation")).toHaveTextContent("T2M_MAX");
    expect(screen.getByText("210 days")).toBeInTheDocument();

    // Tab 3 shows SC-30 - SC-31 borderline label & 7/14/21 sensitivity table without ML rank numbers
    fireEvent.click(screen.getByRole("button", { name: EN.tabs.scores }));
    expect(screen.getByText("Borderline, not adjusted for multiple comparisons")).toBeInTheDocument();
    expect(screen.getByTestId("sensitivity-table")).toHaveTextContent("21 days");
  });

  it("enforces single i18n resource file (no hardcoded JSX text literals in components or App.tsx)", () => {
    const rawModules = import.meta.glob(["../App.tsx", "../components/*.tsx"], {
      query: "?raw",
      eager: true,
      import: "default",
    }) as Record<string, string>;

    expect(Object.keys(rawModules).length).toBeGreaterThanOrEqual(6);

    // Scan for multi-word English text directly between JSX tags (`>Some literal text<`)
    // and hardcoded user-facing JSX attribute literals (`title="..."`, `placeholder="..."`, `aria-label="..."`, `alt="..."`)
    const jsxLiteralTextRe = />\s*[A-Za-z][A-Za-z0-9 ,.;:()/-]{2,}\s*</g;
    const jsxLiteralAttrRe = /\b(?:title|placeholder|aria-label|alt)="[A-Za-z][^"]{1,}"/g;
    const violations: string[] = [];
    for (const [modPath, content] of Object.entries(rawModules)) {
      for (const m of content.match(jsxLiteralTextRe) ?? []) {
        violations.push(`${modPath}: ${m.trim()}`);
      }
      for (const m of content.match(jsxLiteralAttrRe) ?? []) {
        violations.push(`${modPath}: ${m.trim()}`);
      }
    }
    expect(violations).toEqual([]);
  });
});
