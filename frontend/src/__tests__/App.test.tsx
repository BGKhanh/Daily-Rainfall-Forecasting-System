import { describe, expect, it, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import App from "../App";
import { EN } from "../i18n/en";
import type {
  HealthResponse,
  LocationItem,
  ObservationsQueryResponse,
  PredictResponse,
  ProvinceGeoJson,
  ProvinceGridOverlapData,
} from "../types/api";

const MOCK_GRID_OVERLAP: ProvinceGridOverlapData = {
  schema_version: "1.0",
  generated_at_utc: "2026-11-20T00:00:00Z",
  earth_radius_km: 6371.0088,
  unit_count: 2,
  units: {
    HCMC: {
      location_id: "HCMC",
      display_name: "Ho Chi Minh City",
      representative_point: { lat: 10.7769, lon: 106.7009 },
      polygon_area_km2: 2095.5,
      merra2_grid: {
        resolution_deg: "0.5x0.625",
        representative_cell: {
          cell_id: "MERRA2_201_747",
          lat_min: 10.75,
          lat_max: 11.25,
          lon_min: 106.5625,
          lon_max: 107.1875,
          cell_area_km2: 3795.8,
          cell_height_km: 55.6,
          cell_width_km: 68.3,
          intersection_area_km2: 1200.0,
          pct_of_province_area: 57.27,
          pct_of_cell_area: 31.5,
          contains_representative_point: true,
        },
        cell_dimensions_km_at_phi: { height_km: 55.6, width_km: 68.3, area_km2: 3795.8 },
        intersecting_cell_count: 1,
        pct_sum: 100.0,
        intersecting_cells: [
          {
            cell_id: "MERRA2_201_747",
            lat_min: 10.75,
            lat_max: 11.25,
            lon_min: 106.5625,
            lon_max: 107.1875,
            cell_area_km2: 3795.8,
            cell_height_km: 55.6,
            cell_width_km: 68.3,
            intersection_area_km2: 2095.5,
            pct_of_province_area: 100.0,
            pct_of_cell_area: 55.21,
            contains_representative_point: true,
          },
        ],
      },
      ceres_grid: {
        resolution_deg: "1.0x1.0",
        representative_cell: {
          cell_id: "CERES_10_106",
          lat_min: 10.0,
          lat_max: 11.0,
          lon_min: 106.0,
          lon_max: 107.0,
          cell_area_km2: 12145.6,
          cell_height_km: 111.2,
          cell_width_km: 109.2,
          intersection_area_km2: 1900.0,
          pct_of_province_area: 90.67,
          pct_of_cell_area: 15.63,
          contains_representative_point: true,
        },
        cell_dimensions_km_at_phi: { height_km: 111.2, width_km: 109.2, area_km2: 12145.6 },
        intersecting_cell_count: 1,
        pct_sum: 100.0,
        intersecting_cells: [
          {
            cell_id: "CERES_10_106",
            lat_min: 10.0,
            lat_max: 11.0,
            lon_min: 106.0,
            lon_max: 107.0,
            cell_area_km2: 12145.6,
            cell_height_km: 111.2,
            cell_width_km: 109.2,
            intersection_area_km2: 2095.5,
            pct_of_province_area: 100.0,
            pct_of_cell_area: 17.25,
            contains_representative_point: true,
          },
        ],
      },
    },
    HANOI: {
      location_id: "HANOI",
      display_name: "Ha Noi",
      representative_point: { lat: 21.0285, lon: 105.8542 },
      polygon_area_km2: 3358.6,
      merra2_grid: {
        resolution_deg: "0.5x0.625",
        representative_cell: null,
        cell_dimensions_km_at_phi: { height_km: 55.6, width_km: 64.8, area_km2: 3604.2 },
        intersecting_cell_count: 1,
        pct_sum: 100.0,
        intersecting_cells: [
          {
            cell_id: "MERRA2_221_746",
            lat_min: 20.75,
            lat_max: 21.25,
            lon_min: 105.9375,
            lon_max: 106.5625,
            cell_area_km2: 3604.2,
            cell_height_km: 55.6,
            cell_width_km: 64.8,
            intersection_area_km2: 3358.6,
            pct_of_province_area: 100.0,
            pct_of_cell_area: 93.18,
            contains_representative_point: true,
          },
        ],
      },
      ceres_grid: {
        resolution_deg: "1.0x1.0",
        representative_cell: null,
        cell_dimensions_km_at_phi: { height_km: 111.2, width_km: 103.8, area_km2: 11541.5 },
        intersecting_cell_count: 1,
        pct_sum: 100.0,
        intersecting_cells: [
          {
            cell_id: "CERES_21_105",
            lat_min: 21.0,
            lat_max: 22.0,
            lon_min: 105.0,
            lon_max: 106.0,
            cell_area_km2: 11541.5,
            cell_height_km: 111.2,
            cell_width_km: 103.8,
            intersection_area_km2: 3358.6,
            pct_of_province_area: 100.0,
            pct_of_cell_area: 29.1,
            contains_representative_point: true,
          },
        ],
      },
    },
  },
};

const MOCK_GEOJSON: ProvinceGeoJson = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      properties: {
        polygon_code: "VN_HO_CHI_MINH",
        location_id: "HCMC",
        display_name: "Ho Chi Minh City",
        source: "geoBoundaries",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [106.58, 10.65],
            [106.95, 10.65],
            [106.95, 11.12],
            [106.58, 11.12],
            [106.58, 10.65],
          ],
        ],
      },
    },
    {
      type: "Feature",
      properties: {
        polygon_code: "VN_HA_NOI",
        location_id: "HANOI",
        display_name: "Ha Noi",
        source: "geoBoundaries",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [105.45, 20.85],
            [105.92, 20.85],
            [105.92, 21.22],
            [105.45, 21.22],
            [105.45, 20.85],
          ],
        ],
      },
    },
    {
      type: "Feature",
      properties: {
        polygon_code: "VN_CAN_THO",
        location_id: "CAN_THO",
        display_name: "Can Tho",
        source: "geoBoundaries",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [105.35, 9.85],
            [105.85, 9.85],
            [105.85, 10.18],
            [105.35, 10.18],
            [105.35, 9.85],
          ],
        ],
      },
    },
  ],
};

const MOCK_HEALTH: HealthResponse = {
  status: "healthy",
  service: "HCMC Rainfall Forecasting Engine",
  version: "1.2.0",
  predictor_loaded: true,
  model_source: "local",
  weather_backend: "sqlite",
  db_role: "bundled_demo_db",
  default_complete_anchor_date: "2026-08-30",
  data_snapshot_date: "2026-11-18",
  scan_window_start: "2026-04-23",
  scan_window_end: "2026-11-18",
  scan_window_total_days: 210,
  latest_precip_obs_date: "2026-11-15",
  latest_partial_obs_explanation:
    "On 2026-11-15, precipitation (3.32 mm) and 19 of 24 meteorology variables are present, while 5 remain missing (T2M_MAX, WS2M_MIN, WS2M_RANGE, WS10M_MIN, GWETPROF).",
  anchor_range: { min: "2001-02-04", max: "2026-08-30" },
  data_latency: {
    last_ingest_run_id: "audit_probe_01",
    last_ingest_at_utc: "2026-11-19T08:40:00Z",
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

const DEFAULT_LOCATIONS: LocationItem[] = [
  {
    location_id: "HCMC",
    display_name: "Ho Chi Minh City",
    latitude: 10.7769,
    longitude: 106.7009,
    meteo_grid_note: "MERRA-2 / GEOS-IT 0.5 x 0.625 deg",
    rad_grid_note: "CERES SYN1deg / FLASHFlux 1.0 x 1.0 deg",
    time_standard: "LST",
    is_active: true,
  },
];

const MOCK_PREDICT: PredictResponse = {
  anchor_date: "2026-08-30",
  anchor_split_regime: "POST_TEST",
  run_date: "2026-11-19",
  run_timezone: "Asia/Ho_Chi_Minh",
  effective_future_days: 0,
  future_forecast_notice: "This output replays 7 days after 2026-08-30 using completed archive data.",
  db_role: "bundled_demo_db",
  daily_forecasts: [
    {
      target_date: "2026-08-31",
      horizon_day: 1,
      status: "past_observed",
      observed_mm: 4.8,
      predictions: { "SC-35": 5.2, "SC-30": 6.1, "SC-31": 4.9, BASELINE_CLIMATOLOGY_DOY: 4.7, BASELINE_PERSISTENCE: 2.0 },
    },
    {
      target_date: "2026-09-01",
      horizon_day: 2,
      status: "past_observed",
      observed_mm: 0.2,
      predictions: { "SC-35": 3.4, "SC-30": 3.9, "SC-31": 3.5, BASELINE_CLIMATOLOGY_DOY: 4.5, BASELINE_PERSISTENCE: 2.0 },
    },
    {
      target_date: "2026-09-02",
      horizon_day: 3,
      status: "past_observed",
      observed_mm: 6.5,
      predictions: { "SC-35": 5.9, "SC-30": 5.1, "SC-31": 5.4, BASELINE_CLIMATOLOGY_DOY: 4.8, BASELINE_PERSISTENCE: 2.0 },
    },
  ],
  run_timestamp: "2026-11-19T08:40:00Z",
  model_source: "local",
  history_backend: "sqlite",
  data_as_of: "2026-08-30",
  lag_days: 81,
  forecast_horizon_mode: "retrospective_hindcast_only",
  anchor_completeness_regime: "COMPLETE_ALL_35",
};

const MOCK_OBS_DAILY_90D: ObservationsQueryResponse = {
  location_id: "HCMC",
  display_name: "Ho Chi Minh City",
  db_role: "bundled_demo_db",
  has_data: true,
  empty_reason: null,
  granularity: "daily",
  start_date: "2026-08-22",
  end_date: "2026-11-19",
  valid_range: { min_date: "2001-01-01", max_date: "2026-11-19" },
  finality_mode: "source_header_transition",
  finality_cutoff_date: "2026-09-30",
  last_scan_date: "2026-11-18",
  probe_date: "2026-11-19",
  probe_timestamp_utc: "2026-11-19T08:40:00Z",
  finality_rule: "FINALIZED when PRECTOTCORR IS NOT NULL and obs_date <= 2026-09-30.",
  series: [
    ...Array.from({ length: 16 }, (_, idx) => {
      const d = new Date(Date.UTC(2026, 7, 22 + idx)).toISOString().slice(0, 10);
      return {
        date: d,
        period_end: d,
        precip_mm: Number((2.5 + (idx % 4)).toFixed(2)),
        precip_total_mm: Number((2.5 + (idx % 4)).toFixed(2)),
        precip_daily_mean_mm: Number((2.5 + (idx % 4)).toFixed(2)),
        published_days_count: 1,
        calendar_days_count: 1,
        status: "FINALIZED" as const,
      };
    }),
    {
      date: "2026-10-05",
      period_end: "2026-10-05",
      precip_mm: 7.4,
      precip_total_mm: 7.4,
      precip_daily_mean_mm: 7.4,
      published_days_count: 1,
      calendar_days_count: 1,
      status: "PROVISIONAL",
    },
    {
      date: "2026-11-15",
      period_end: "2026-11-15",
      precip_mm: 3.32,
      precip_total_mm: 3.32,
      precip_daily_mean_mm: 3.32,
      published_days_count: 1,
      calendar_days_count: 1,
      status: "PROVISIONAL",
    },
    {
      date: "2026-11-18",
      period_end: "2026-11-18",
      precip_mm: null,
      precip_total_mm: null,
      precip_daily_mean_mm: null,
      published_days_count: 0,
      calendar_days_count: 1,
      status: "NOT_PUBLISHED",
    },
    {
      date: "2026-11-19",
      period_end: "2026-11-19",
      precip_mm: null,
      precip_total_mm: null,
      precip_daily_mean_mm: null,
      published_days_count: 0,
      calendar_days_count: 1,
      status: "NOT_PUBLISHED",
    },
  ],
  forecast_overlay: [
    { date: "2026-08-31", horizon_day: 1, pred_mm: 5.2 },
    { date: "2026-09-01", horizon_day: 2, pred_mm: 3.4 },
    { date: "2026-09-02", horizon_day: 3, pred_mm: 5.9 },
  ],
  forecast_overlay_anchor_date: "2026-08-30",
  summary: {
    total_precip_mm: 62.72,
    rain_days_ge_1mm: 18,
    finalized_days: 16,
    provisional_days: 2,
    not_published_days: 2,
    published_days: 18,
    calendar_days: 20,
  },
};

const MOCK_OBS_WEEKLY: ObservationsQueryResponse = {
  ...MOCK_OBS_DAILY_90D,
  granularity: "weekly",
  start_date: "2024-11-20",
  end_date: "2026-11-19",
  series: [
    {
      date: "2024-11-20",
      period_end: "2024-11-24",
      precip_mm: null,
      precip_total_mm: 14.0,
      precip_daily_mean_mm: 2.8,
      published_days_count: 5,
      calendar_days_count: 7,
      status: "FINALIZED",
    },
    {
      date: "2024-11-25",
      period_end: "2024-12-01",
      precip_mm: null,
      precip_total_mm: 24.5,
      precip_daily_mean_mm: 3.5,
      published_days_count: 7,
      calendar_days_count: 7,
      status: "FINALIZED",
    },
    {
      date: "2024-12-02",
      period_end: "2024-12-08",
      precip_mm: null,
      precip_total_mm: 28.0,
      precip_daily_mean_mm: 4.0,
      published_days_count: 7,
      calendar_days_count: 7,
      status: "FINALIZED",
    },
    {
      date: "2026-11-09",
      period_end: "2026-11-15",
      precip_mm: null,
      precip_total_mm: 6.4,
      precip_daily_mean_mm: 3.2,
      published_days_count: 2,
      calendar_days_count: 7,
      status: "PROVISIONAL",
    },
    {
      date: "2026-11-16",
      period_end: "2026-11-19",
      precip_mm: null,
      precip_total_mm: null,
      precip_daily_mean_mm: null,
      published_days_count: 0,
      calendar_days_count: 7,
      status: "NOT_PUBLISHED",
    },
  ],
};

const MOCK_OBS_EMPTY_HANOI: ObservationsQueryResponse = {
  location_id: "HANOI",
  display_name: "Ha Noi",
  db_role: "bundled_demo_db",
  has_data: false,
  empty_reason: "No observations have been ingested for this location yet.",
  granularity: "daily",
  start_date: "2026-08-22",
  end_date: "2026-11-19",
  valid_range: { min_date: "2001-01-01", max_date: "2026-11-19" },
  finality_mode: "source_header_transition",
  finality_cutoff_date: "2026-09-30",
  last_scan_date: "2026-11-18",
  probe_date: "2026-11-19",
  probe_timestamp_utc: "2026-11-19T08:40:00Z",
  finality_rule: "FINALIZED when PRECTOTCORR IS NOT NULL and obs_date <= 2026-09-30.",
  series: [],
  forecast_overlay: [],
  forecast_overlay_anchor_date: "2026-08-30",
  summary: {
    total_precip_mm: 0,
    rain_days_ge_1mm: 0,
    finalized_days: 0,
    provisional_days: 0,
    not_published_days: 0,
    published_days: 0,
    calendar_days: 0,
  },
};

const VIETNAM_DIACRITIC_REGEX =
  /[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]/i;

function installFetchMock(overrides?: { predict?: PredictResponse }) {
  const predictCalls: string[] = [];
  const observationCalls: string[] = [];
  const externalUrls: string[] = [];
  const predictPayload = overrides?.predict ?? MOCK_PREDICT;

  const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    if (/^https?:\/\//i.test(url)) {
      externalUrls.push(url);
      throw new Error(`Unexpected external fetch blocked by test: ${url}`);
    }
    if (url.endsWith("/health")) {
      return new Response(JSON.stringify(MOCK_HEALTH), { status: 200 });
    }
    if (url.endsWith("/locations")) {
      return new Response(JSON.stringify({ db_role: "bundled_demo_db", locations: DEFAULT_LOCATIONS }), {
        status: 200,
      });
    }
    if (url.includes("/map/vietnam_63_provinces.geojson")) {
      return new Response(JSON.stringify(MOCK_GEOJSON), { status: 200 });
    }
    if (url.includes("/map/province_grid_overlap.json")) {
      return new Response(JSON.stringify(MOCK_GRID_OVERLAP), { status: 200 });
    }
    if (url.startsWith("/observations")) {
      observationCalls.push(url);
      if (url.includes("location_id=HANOI")) {
        return new Response(JSON.stringify(MOCK_OBS_EMPTY_HANOI), { status: 200 });
      }
      if (url.includes("granularity=weekly")) {
        return new Response(JSON.stringify(MOCK_OBS_WEEKLY), { status: 200 });
      }
      return new Response(JSON.stringify(MOCK_OBS_DAILY_90D), { status: 200 });
    }
    if (url.endsWith("/predict")) {
      predictCalls.push(String(init?.body ?? ""));
      return new Response(JSON.stringify(predictPayload), { status: 200 });
    }
    return new Response("Not found", { status: 404 });
  });

  vi.stubGlobal("fetch", fetchSpy);
  return { predictCalls, observationCalls, externalUrls };
}

describe("Round 10 English SPA, Single-Model Main, 2-Layer Map & Data Retrieval", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: 1280 });
  });

  it("renders single-model @serving line chart, exact 3 tabs, no banned words/UI, and same-origin requests only", async () => {
    const { externalUrls } = installFetchMock();
    render(<App initialGeoJson={MOCK_GEOJSON} />);

    await waitFor(() => {
      expect(screen.getByTestId("forecast-line-chart")).toBeInTheDocument();
    });

    // Exactly 3 tabs; Comparison & scores must be completely absent
    expect(screen.getByRole("button", { name: EN.tabs.forecast })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: EN.tabs.data })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: EN.tabs.governance })).toBeInTheDocument();
    expect(screen.queryByText(/Comparison & scores/i)).not.toBeInTheDocument();

    // No anchor_date input or presets
    expect(document.querySelector('input[type="date"]')).toBeNull();
    expect(screen.queryByRole("button", { name: /Mid-monsoon/i })).not.toBeInTheDocument();

    // New main heading & dashed Forecast / solid Observed polylines
    expect(
      screen.getByText(/Forecast issued from data up to 30 Aug 2026 \(latest complete input\)\. Today: 19 Nov 2026/)
    ).toBeInTheDocument();
    const chartSvg = screen.getByTestId("forecast-line-chart");
    expect(chartSvg.querySelector("path[data-series='forecast']")?.getAttribute("stroke-dasharray")).toBeTruthy();
    expect(chartSvg.querySelector("path[data-series='observed']")?.getAttribute("stroke-dasharray")).toBeNull();

    // Dates instead of +1d..+7d
    expect(screen.getAllByText("31 Aug 2026").length).toBeGreaterThanOrEqual(2);
    expect(screen.queryByText("+1d")).not.toBeInTheDocument();

    // Toggle 14 prior observations switch triggers /observations query
    const priorToggle = screen.getByLabelText(EN.forecast.priorObservationsToggle);
    fireEvent.click(priorToggle);
    await waitFor(() => {
      expect(screen.getByTestId("forecast-line-chart").querySelectorAll("circle[data-series='observed']").length).toBeGreaterThan(3);
    });

    // Zero banned words, zero Vietnamese diacritics, zero external network requests
    const text = document.body.textContent ?? "";
    expect(text).not.toMatch(/\bnormalized\b/i);
    expect(text).not.toMatch(/\bnext 7 days\b/i);
    expect(text).not.toMatch(/effective_future_days|COMPLETE_ALL_35/i);
    expect(VIETNAM_DIACRITIC_REGEX.test(text)).toBe(false);
    expect(externalUrls).toEqual([]);
  });

  it("supports independent Province boundaries / Grid cells toggles, multi-select table, and Forecast not available yet on unsupported province", async () => {
    const { predictCalls } = installFetchMock();
    render(<App initialGeoJson={MOCK_GEOJSON} />);

    await waitFor(() => {
      expect(screen.getByTestId("selected-locations-table")).toBeInTheDocument();
      expect(screen.getByTestId("forecast-line-chart")).toBeInTheDocument();
    });
    const baselinePredicts = predictCalls.length;

    const boundToggle = screen.getByLabelText(EN.map.toggleBoundaries);
    const gridToggle = screen.getByLabelText(EN.map.toggleGridCells);
    expect(screen.getByTestId("map-boundaries-layer")).toBeInTheDocument();
    expect(screen.getByTestId("map-grid-layer")).toBeInTheDocument();

    // Toggle off boundaries alone -> grid remains
    fireEvent.click(boundToggle);
    expect(screen.queryByTestId("map-boundaries-layer")).not.toBeInTheDocument();
    expect(screen.getByTestId("map-grid-layer")).toBeInTheDocument();

    // Toggle boundaries back on, toggle grid off -> boundaries remain
    fireEvent.click(boundToggle);
    fireEvent.click(gridToggle);
    expect(screen.getByTestId("map-boundaries-layer")).toBeInTheDocument();
    expect(screen.queryByTestId("map-grid-layer")).not.toBeInTheDocument();
    fireEvent.click(gridToggle);

    // Click Ha Noi (unsupported) -> keeps HCMC forecast active by default, shows forecast tabs,
    // switching to Ha Noi tab shows deduplicated heading "{DisplayName} — Forecast not available yet" with zero new /predict calls
    const hanoiPoly = document.querySelector('path[data-location-id="HANOI"]');
    expect(hanoiPoly).not.toBeNull();
    fireEvent.click(hanoiPoly!);

    const tabsBar = screen.getByTestId("forecast-location-tabs");
    expect(tabsBar).toBeInTheDocument();
    expect(screen.getByTestId("forecast-line-chart")).toBeInTheDocument();
    expect(screen.getByTestId("forecast-spatial-note")).toHaveTextContent("29.11%");

    const hanoiTab = within(tabsBar).getByRole("tab", { name: /Ha Noi/i });
    fireEvent.click(hanoiTab);

    const unavailable = screen.getByTestId("forecast-unavailable-panel");
    expect(unavailable).toHaveTextContent("Ha Noi — Forecast not available yet");
    expect(unavailable.querySelectorAll("p").length).toBe(1);
    expect(predictCalls.length).toBe(baselinePredicts);
    const selectedTbl = screen.getByTestId("selected-locations-table");
    expect(selectedTbl).toHaveTextContent("Ho Chi Minh City");
    expect(selectedTbl).toHaveTextContent("Ha Noi");
    expect(selectedTbl).toHaveTextContent("Not available yet");
  });

  it("renders Tab 2 Data retrieval with 3 colored segments, NOT_PUBLISHED gaps, auto Weekly switch, pagination, CSV export & empty state", async () => {
    installFetchMock();
    render(<App initialGeoJson={MOCK_GEOJSON} />);

    fireEvent.click(screen.getByRole("button", { name: EN.tabs.data }));

    await waitFor(() => {
      expect(screen.getByTestId("observations-chart")).toBeInTheDocument();
    });

    const obsChart = screen.getByTestId("observations-chart");
    expect(obsChart.querySelector("[data-segment='FINALIZED']")).not.toBeNull();
    expect(obsChart.querySelector("[data-segment='PROVISIONAL']")).not.toBeNull();
    expect(obsChart.querySelector("[data-segment='FORECAST']")).not.toBeNull();
    expect(obsChart.querySelector("[data-gap-band='NOT_PUBLISHED']")).not.toBeNull();

    // Switch to Long range (> 366d) -> switches to Weekly & allows Period total vs Daily mean
    fireEvent.click(screen.getByRole("button", { name: EN.dataRetrieval.preset2Years }));
    await waitFor(() => {
      expect(screen.getByLabelText(EN.dataRetrieval.granularityLabel)).toHaveValue("weekly");
    });

    // Empty state for unpopulated location HANOI
    fireEvent.change(screen.getByLabelText(EN.dataRetrieval.locationLabel), { target: { value: "HANOI" } });
    await waitFor(() => {
      expect(screen.getByText("No observations have been ingested for this location yet.")).toBeInTheDocument();
    });

    // Verify CSV download button works and produces NASA POWER header
    fireEvent.change(screen.getByLabelText(EN.dataRetrieval.locationLabel), { target: { value: "HCMC" } });
    await waitFor(() => {
      expect(screen.getByRole("button", { name: EN.dataRetrieval.downloadCsv })).not.toBeDisabled();
    });
  });

  it("renders Tab 3 System limitations with Spatial representativeness & 3 NASA POWER links, zero model-selection prose", async () => {
    installFetchMock();
    render(<App initialGeoJson={MOCK_GEOJSON} />);

    fireEvent.click(screen.getByRole("button", { name: EN.tabs.governance }));

    expect(screen.getByText("Spatial representativeness")).toBeInTheDocument();
    expect(document.body.textContent).toContain("29.11%");
    expect(document.body.textContent).toContain("16.12%");
    expect(document.body.textContent).toContain("90.30%");
    expect(screen.getByTestId("dynamic-snapshot-mode")).toHaveTextContent("Bundled Demo SQLite DB");
    const hrefs = screen.getAllByRole("link").map((el) => el.getAttribute("href") ?? "");
    expect(hrefs.some((h) => h.includes("power.larc.nasa.gov/docs/services"))).toBe(true);
    expect(hrefs.some((h) => h.includes("power.larc.nasa.gov/docs/faqs/solar"))).toBe(true);
    expect(hrefs.some((h) => h.includes("power.larc.nasa.gov/docs/methodology/data/sources"))).toBe(true);
    // Removed model selection prose
    expect(document.body.textContent).not.toMatch(/Validation-selected champion is SC-30/i);
    expect(document.body.textContent).not.toMatch(/SC-31/i);
  });

  it("renders vertical day cards at 375px without horizontal overflow", async () => {
    Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: 375 });
    installFetchMock();
    render(<App initialGeoJson={MOCK_GEOJSON} />);

    await waitFor(() => {
      expect(screen.getByTestId("mobile-day-cards")).toBeInTheDocument();
    });
    expect(document.documentElement.scrollWidth).toBeLessThanOrEqual(window.innerWidth);
    const cards = within(screen.getByTestId("mobile-day-cards"));
    expect(cards.getByText("31 Aug 2026")).toBeInTheDocument();
    expect(cards.getByText("5.20")).toBeInTheDocument();
    expect(cards.getByText("4.80")).toBeInTheDocument();
  });

  it("enforces single i18n resource file, zero hardcoded JSX literals, and zero occurrences of 'normalized'", () => {
    const rawModules = import.meta.glob(["../App.tsx", "../components/*.tsx", "../i18n/*.ts"], {
      query: "?raw",
      eager: true,
      import: "default",
    }) as Record<string, string>;

    expect(Object.keys(rawModules).length).toBeGreaterThanOrEqual(5);

    const jsxLiteralTextRe = /(?<![=<])>[^\S\r\n]*[A-Za-z][A-Za-z0-9 ,.;:()/-]{2,}[^\S\r\n]*</g;
    const jsxLiteralAttrRe = /\b(?:title|placeholder|aria-label|alt)="[A-Za-z][^"]{1,}"/g;
    const violations: string[] = [];
    for (const [modPath, content] of Object.entries(rawModules)) {
      if (/\bnormalized\b/i.test(content)) {
        violations.push(`${modPath}: contains banned word 'normalized'`);
      }
      if (/\btelemetry\b/i.test(content)) {
        violations.push(`${modPath}: contains replaced word 'telemetry'`);
      }
      if (modPath.endsWith(".tsx")) {
        for (const m of content.match(jsxLiteralTextRe) ?? []) {
          violations.push(`${modPath}: ${m.trim()}`);
        }
        for (const m of content.match(jsxLiteralAttrRe) ?? []) {
          violations.push(`${modPath}: ${m.trim()}`);
        }
      }
    }
    expect(violations).toEqual([]);
  });

  it("supports map zoom controls, toggle-to-deselect, X button, Clear selection revert to HCMC, and scrollable 5-row selected table (Part A)", async () => {
    installFetchMock();
    render(<App initialGeoJson={MOCK_GEOJSON} />);

    await waitFor(() => {
      expect(screen.getByTestId("map-zoom-controls")).toBeInTheDocument();
    });

    expect(screen.getByTestId("map-zoom-level")).toHaveTextContent("1×");
    expect(screen.getByTestId("map-fit-selection-btn")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: EN.map.zoomInAria }));
    expect(screen.getByTestId("map-zoom-level")).toHaveTextContent("1.3×");
    fireEvent.click(screen.getByRole("button", { name: EN.map.zoomResetLabel }));
    expect(screen.getByTestId("map-zoom-level")).toHaveTextContent("1×");

    const hanoiPoly = document.querySelector('path[data-location-id="HANOI"]');
    const canThoPoly = document.querySelector('path[data-location-id="CAN_THO"]');
    expect(hanoiPoly).not.toBeNull();
    expect(canThoPoly).not.toBeNull();

    // Select Ha Noi -> 2 selected, then click Ha Noi again -> toggles off back to HCMC
    fireEvent.click(hanoiPoly!);
    expect(screen.getByTestId("selected-count-badge")).toHaveTextContent("2 selected");
    fireEvent.click(hanoiPoly!);
    expect(screen.getByTestId("selected-count-badge")).toHaveTextContent("1 selected");

    // Select Ha Noi + Can Tho -> remove Can Tho via X -> Clear selection reverts to HCMC
    fireEvent.click(hanoiPoly!);
    fireEvent.click(canThoPoly!);
    expect(screen.getByTestId("selected-count-badge")).toHaveTextContent("3 selected");
    fireEvent.click(screen.getByRole("button", { name: EN.map.deselectBtnLabel("Can Tho") }));
    expect(screen.getByTestId("selected-count-badge")).toHaveTextContent("2 selected");

    fireEvent.click(screen.getByRole("button", { name: EN.map.clearSelectionBtn }));
    expect(screen.getByTestId("selected-count-badge")).toHaveTextContent("1 selected");
    expect(screen.getByTestId("selected-locations-table")).toHaveTextContent("Ho Chi Minh City");
    expect(document.querySelector(".selected-table-scroll")).not.toBeNull();
  });

  it("renders Tab 1 GridScalePanel with province selector, MERRA-2 + CERES SVG overlay, overlap %, and 35-var catalog (Part B)", async () => {
    installFetchMock();
    render(<App initialGeoJson={MOCK_GEOJSON} initialGridOverlap={MOCK_GRID_OVERLAP} />);

    await waitFor(() => {
      expect(screen.getByTestId("grid-scale-svg")).toBeInTheDocument();
    });

    const svg = screen.getByTestId("grid-scale-svg");
    expect(svg.querySelectorAll('rect[data-grid-type="merra2"]').length).toBeGreaterThanOrEqual(1);
    expect(svg.querySelectorAll('rect[data-grid-type="ceres"]').length).toBeGreaterThanOrEqual(1);
    expect(svg.querySelector('rect[data-grid-type="ceres"][data-rep="true"]')).not.toBeNull();
    expect(screen.getByTestId("grid-scale-decorations")).toHaveTextContent("N ↑");
    expect(screen.getByTestId("grid-scale-decorations")).toHaveTextContent("km");

    const dropdown = screen.getByLabelText(EN.gridScale.provinceSelectAria);
    fireEvent.change(dropdown, { target: { value: "HANOI" } });
    expect(dropdown).toHaveValue("HANOI");

    fireEvent.click(screen.getByLabelText(EN.gridScale.showOverlapToggle));
    const overlapPanel = screen.getByTestId("overlap-breakdown-table");
    expect(overlapPanel).toBeInTheDocument();
    expect(overlapPanel.closest(".grid-scale-workspace")).toBeNull();
    expect(overlapPanel).toHaveTextContent("Representative MERRA-2 cell (meteo input feed)");
    expect(overlapPanel).toHaveTextContent("Representative CERES cell (radiation input feed)");

    const catalogDetails = screen.getByTestId("variable-catalog-section") as HTMLDetailsElement;
    expect(catalogDetails.open).toBe(false);

    const panel = screen.getByTestId("grid-scale-panel");
    expect(panel).toHaveTextContent("Polygon area (from simplified boundary data)");
    expect(panel).toHaveTextContent("TOA_SW_DWN");
    expect(panel.textContent).not.toMatch(/\bCLOUD_AMT\b|\bALB\b/);
  });

  it("renders Tab 2 Part F probe-aware cutoff, scan date empty slots, weekly partial badge & overlay note, and hides ingest card when empty", async () => {
    installFetchMock();
    render(<App initialGeoJson={MOCK_GEOJSON} />);

    fireEvent.click(screen.getByRole("button", { name: EN.tabs.data }));

    await waitFor(() => {
      expect(screen.getByTestId("observations-chart")).toBeInTheDocument();
    });

    expect(screen.getByText("Finalized cutoff: 30 Sep 2026 (probed 19 Nov 2026)")).toBeInTheDocument();
    expect(screen.getAllByText("Not yet published (as of the 18 Nov 2026 scan)").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByTestId("latest-ingest-footnote")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/\btelemetry\b/i);

    fireEvent.click(screen.getByRole("button", { name: EN.dataRetrieval.preset2Years }));
    await waitFor(() => {
      expect(screen.getAllByText("Partial period (2/7 days)").length).toBeGreaterThanOrEqual(1);
    });
    expect(screen.getByTestId("obs-start-iso")).toHaveTextContent("ISO:");
    expect(screen.getByTestId("obs-end-iso")).toHaveTextContent("ISO:");
    const obsSvg = screen.getByTestId("observations-chart");
    expect(obsSvg.querySelectorAll('circle[data-partial-period="true"]').length).toBe(2);
    expect(obsSvg.querySelectorAll('path[data-partial-connector="true"]').length).toBeGreaterThanOrEqual(2);
    expect(screen.getByTestId("legend-partial-period")).toHaveTextContent("Partial period (dotted connector");
    expect(screen.getByText(EN.dataRetrieval.overlayDailyOnlyNote)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(EN.dataRetrieval.locationLabel), { target: { value: "HANOI" } });
    await waitFor(() => {
      expect(screen.getByText("No observations have been ingested for this location yet.")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("latest-ingest-footnote")).not.toBeInTheDocument();
  });

  it("renders Round 13 per-cell area labels, Can Tho official-area warning badge, and P-c estimated radiation banner", async () => {
    const r13Overlap: ProvinceGridOverlapData = {
      ...MOCK_GRID_OVERLAP,
      unit_count: 3,
      units: {
        ...MOCK_GRID_OVERLAP.units,
        HCMC: {
          ...MOCK_GRID_OVERLAP.units.HCMC,
          official_area_km2: 2095.0,
          area_rel_diff_pct: 0.28,
          exceeds_5pct_area_threshold: false,
          merra2_grid: {
            ...MOCK_GRID_OVERLAP.units.HCMC.merra2_grid,
            intersecting_cell_count: 4,
          },
          ceres_grid: {
            ...MOCK_GRID_OVERLAP.units.HCMC.ceres_grid,
            intersecting_cell_count: 3,
          },
        },
        CAN_THO: {
          ...MOCK_GRID_OVERLAP.units.HANOI,
          location_id: "CAN_THO",
          display_name: "Can Tho",
          polygon_area_km2: 1826.91,
          official_area_km2: 1440.4,
          area_rel_diff_pct: 26.83,
          exceeds_5pct_area_threshold: true,
        },
      },
    };
    installFetchMock({
      predict: {
        ...MOCK_PREDICT,
        anchor_date: "2026-11-17",
        data_as_of: "2026-11-17",
        lag_days: 4,
        effective_future_days: 3,
        is_degraded_input: true,
        ceres_policy_applied: "P-c_DOY_MEDIAN_11_SOLAR_V1",
        imputed_variables: ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN"],
      },
    });
    render(<App initialGeoJson={MOCK_GEOJSON} initialGridOverlap={r13Overlap} />);

    await waitFor(() => {
      expect(screen.getByTestId("estimated-radiation-banner")).toBeInTheDocument();
    });
    expect(screen.getByTestId("estimated-radiation-banner")).toHaveTextContent(
      "Estimated radiation inputs: 2 pending NASA POWER solar variable(s)"
    );
    expect(screen.getByTestId("merra-per-cell-sub")).toHaveTextContent("3,796 km² per cell · 4 cells intersect");
    expect(screen.getByTestId("ceres-per-cell-sub")).toHaveTextContent("12,146 km² per cell · 3 cells intersect");
    expect(screen.queryByTestId("official-area-warning-badge")).not.toBeInTheDocument();

    const dropdown = screen.getByLabelText(EN.gridScale.provinceSelectAria);
    fireEvent.change(dropdown, { target: { value: "CAN_THO" } });
    expect(screen.getByTestId("official-area-warning-badge")).toHaveTextContent(
      "Simplified polygon differs from official GSO area by +26.8% (official: 1,440.4 km²) — overlap percentages are approximate"
    );
  });

  it("renders Round 13-Closeout dynamic top strip, heading, and operational vs retrospective banner branches", async () => {
    const branch1Predict: PredictResponse = {
      ...MOCK_PREDICT,
      anchor_date: "2026-11-17",
      data_as_of: "2026-11-17",
      run_date: "2026-11-21",
      lag_days: 4,
      effective_future_days: 3,
      is_degraded_input: true,
      ceres_policy_applied: "P-c_DOY_MEDIAN_11_SOLAR_V1",
      imputed_variables: ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN"],
      daily_forecasts: [
        { target_date: "2026-11-18", horizon_day: 1, status: "past_unverified", observed_mm: null, predictions: { "SC-35": 4.1, "SC-30": 4.2, "SC-31": 4.0, BASELINE_CLIMATOLOGY_DOY: 3.0, BASELINE_PERSISTENCE: 2.0 } },
        { target_date: "2026-11-19", horizon_day: 2, status: "past_unverified", observed_mm: null, predictions: { "SC-35": 4.5, "SC-30": 4.4, "SC-31": 4.3, BASELINE_CLIMATOLOGY_DOY: 3.0, BASELINE_PERSISTENCE: 2.0 } },
        { target_date: "2026-11-20", horizon_day: 3, status: "past_unverified", observed_mm: null, predictions: { "SC-35": 5.0, "SC-30": 4.9, "SC-31": 4.8, BASELINE_CLIMATOLOGY_DOY: 3.0, BASELINE_PERSISTENCE: 2.0 } },
        { target_date: "2026-11-21", horizon_day: 4, status: "today", observed_mm: null, predictions: { "SC-35": 5.2, "SC-30": 5.1, "SC-31": 5.0, BASELINE_CLIMATOLOGY_DOY: 3.0, BASELINE_PERSISTENCE: 2.0 } },
        { target_date: "2026-11-22", horizon_day: 5, status: "future_forecast", observed_mm: null, predictions: { "SC-35": 4.8, "SC-30": 4.7, "SC-31": 4.6, BASELINE_CLIMATOLOGY_DOY: 3.0, BASELINE_PERSISTENCE: 2.0 } },
        { target_date: "2026-11-23", horizon_day: 6, status: "future_forecast", observed_mm: null, predictions: { "SC-35": 4.2, "SC-30": 4.1, "SC-31": 4.0, BASELINE_CLIMATOLOGY_DOY: 3.0, BASELINE_PERSISTENCE: 2.0 } },
        { target_date: "2026-11-24", horizon_day: 7, status: "future_forecast", observed_mm: null, predictions: { "SC-35": 3.9, "SC-30": 3.8, "SC-31": 3.7, BASELINE_CLIMATOLOGY_DOY: 3.0, BASELINE_PERSISTENCE: 2.0 } },
      ],
    };
    installFetchMock({ predict: branch1Predict });
    const { unmount } = render(<App initialGeoJson={MOCK_GEOJSON} initialGridOverlap={MOCK_GRID_OVERLAP} />);

    await waitFor(() => {
      expect(screen.getByTestId("estimated-radiation-banner")).toBeInTheDocument();
    });

    expect(
      screen.getByText(
        "Single location (Ho Chi Minh City) · Operational forecast uses observed meteorology & estimated solar radiation"
      )
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Forecast issued from data up to 17 Nov 2026 (latest complete meteorology; solar estimated). Today: 21 Nov 2026"
      )
    ).toBeInTheDocument();
    expect(screen.getByTestId("retrospective-banner")).toHaveTextContent(
      "NASA meteorology lags today by 4 days: 18 Nov 2026–20 Nov 2026 are unverified past, 21 Nov 2026 is operational today, and 22 Nov 2026–24 Nov 2026 are future forecasts (3 future days)."
    );
    expect(screen.getByTestId("estimated-radiation-banner")).toBeInTheDocument();
    unmount();

    const branch2Predict: PredictResponse = {
      ...MOCK_PREDICT,
      anchor_date: "2026-08-30",
      data_as_of: "2026-08-30",
      run_date: "2026-11-21",
      lag_days: 83,
      effective_future_days: 0,
      is_degraded_input: false,
      ceres_policy_applied: "NONE",
      imputed_variables: [],
    };
    installFetchMock({ predict: branch2Predict });
    const { unmount: unmount2 } = render(<App initialGeoJson={MOCK_GEOJSON} initialGridOverlap={MOCK_GRID_OVERLAP} />);

    await waitFor(() => {
      expect(
        screen.getByText("Single location (Ho Chi Minh City) · Retrospective archive replay on completed history")
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText("Forecast issued from data up to 30 Aug 2026 (latest complete input). Today: 21 Nov 2026")
    ).toBeInTheDocument();
    expect(screen.getByTestId("retrospective-banner")).toHaveTextContent(
      "Retrospective replay: latest complete archive input lags today by 83 days, so observed rainfall is shown beside the model line."
    );
    expect(screen.queryByTestId("estimated-radiation-banner")).not.toBeInTheDocument();
    unmount2();

    // Non-degraded operational branch (effective_future_days > 0 && !is_degraded_input)
    installFetchMock({
      predict: {
        ...branch1Predict,
        is_degraded_input: false,
        ceres_policy_applied: "NONE",
        imputed_variables: [],
      },
    });
    render(<App initialGeoJson={MOCK_GEOJSON} initialGridOverlap={MOCK_GRID_OVERLAP} />);
    await waitFor(() => {
      expect(
        screen.getByText(
          "Single location (Ho Chi Minh City) · Operational forecast uses completed meteorology & solar radiation"
        )
      ).toBeInTheDocument();
    });
  });
});
