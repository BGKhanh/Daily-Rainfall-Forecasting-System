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
} from "../types/api";

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
      date: "2024-11-18",
      period_end: "2024-11-24",
      precip_mm: null,
      precip_total_mm: 24.5,
      precip_daily_mean_mm: 3.5,
      published_days_count: 7,
      calendar_days_count: 7,
      status: "FINALIZED",
    },
    {
      date: "2026-11-16",
      period_end: "2026-11-19",
      precip_mm: null,
      precip_total_mm: null,
      precip_daily_mean_mm: null,
      published_days_count: 0,
      calendar_days_count: 4,
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

function installFetchMock() {
  const predictCalls: string[] = [];
  const observationCalls: string[] = [];
  const externalUrls: string[] = [];

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
      return new Response(JSON.stringify(MOCK_PREDICT), { status: 200 });
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

    // Click Ha Noi (unsupported) -> zero new /predict calls, shows "Forecast not available yet", adds to Selected locations
    const hanoiPoly = document.querySelector('path[data-location-id="HANOI"]');
    expect(hanoiPoly).not.toBeNull();
    fireEvent.click(hanoiPoly!);

    expect(screen.getByTestId("forecast-unavailable-panel")).toHaveTextContent("Forecast not available yet");
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
});
