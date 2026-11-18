import type { DayForecastStatus } from "../types/api";

export interface FriendlyErrorView {
  summary: string;
  code: string;
  message: string;
  details: Record<string, unknown>;
}

const SHORT_MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
] as const;

export function formatReadableDate(isoDate: string | null | undefined): string {
  if (!isoDate) return "N/A";
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(isoDate.trim());
  if (!match) return isoDate;
  const year = match[1];
  const monthIdx = Number(match[2]) - 1;
  const day = String(Number(match[3]));
  const monthLabel = SHORT_MONTHS[monthIdx];
  if (!monthLabel) return isoDate;
  return `${day} ${monthLabel} ${year}`;
}

export function formatUtcDateTime(isoUtc: string | null | undefined): string {
  if (!isoUtc) return "N/A";
  const datePart = isoUtc.slice(0, 10);
  const timeMatch = /T(\d{2}:\d{2})/.exec(isoUtc);
  const readable = formatReadableDate(datePart);
  return timeMatch ? `${readable} at ${timeMatch[1]} UTC` : readable;
}

export function normalizeAccentInsensitive(value: string): string {
  return value
    .replace(/Đ/g, "D")
    .replace(/đ/g, "d")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

export const EN = {
  app: {
    brandTitle: "Vietnam Daily Rainfall Console",
    brandSubtitle: "Point-regression telemetry & retrospective replay",
    openMenuLabel: "Open navigation menu",
    closeMenuLabel: "Close navigation menu",
    navAriaLabel: "Primary sections",
    coldStartNotice: "Serving container is warming up. Initial telemetry may take a few seconds.",
    summaryStripPrefix: "Single location (Ho Chi Minh City) · evaluated on completed data · real-time forecasting not supported",
    summaryStripLink: "Read governance & limits",
    footerDbLabel: "Active store",
    footerConnecting: "connecting",
    footerLinksAria: "Project links",
    linkGithub: "GitHub",
    linkCi: "CI Runs",
    linkMlflow: "MLflow",
    dbRoles: {
      bundled_demo_db: "Bundled Demo Snapshot",
      local_or_production_db: "Local SQLite Store",
      sample_csv: "Static 35-Day CSV",
      connecting: "Connecting",
    } as Record<string, string>,
  },
  tabs: {
    forecast: "Forecast by location",
    data: "Data retrieval & status",
    scores: "Comparison & scores",
    governance: "Governance & limitations",
  },
  errors: {
    technicalDetailsTitle: "Technical details",
    requestFailedFallback: "Request failed",
    byCode: {
      UNSUPPORTED_LOCATION:
        "Forecasting is only trained and validated for Ho Chi Minh City. Select an active location on the map.",
      INCOMPLETE_RADIATION_AT_ANCHOR:
        "The selected anchor date does not have all 35 meteorological and CERES radiation variables present.",
      ANCHOR_DATE_NOT_FOUND:
        "The requested anchor date was not found in the observation archive.",
      INVALID_CLIENT_HISTORY:
        "The submitted input history or query parameters violate continuity or physical bounds.",
      SERVER_DB_DATA_ERROR:
        "The observation database could not complete validation for this request. Try another anchor date or inspect Data retrieval & status.",
      PREDICTOR_NOT_READY:
        "Model artifacts are still loading into memory. Please retry shortly.",
      INVALID_QUERY_PARAMETER:
        "One or more query parameters do not match the API contract.",
      UNKNOWN:
        "Unable to load forecast results for this request. Please verify the selected anchor date and try again.",
    } as Record<string, string>,
  },
  map: {
    heading: "Location & pre-merger provincial map",
    subtitle: "Boundaries shown for spatial context and configured readiness.",
    searchLabel: "Find province or island",
    searchPlaceholder: "Search (e.g., Ho Chi Minh, Ha Noi, Hoang Sa)",
    listHeading: "Pre-merger units (65)",
    legendTitle: "Map legend",
    legendSupported: "Supported (forecast active)",
    legendUnsupported: "Not supported",
    legendGridCell: "Verified MERRA-2 0.5° × 0.625° grid cell",
    legendVersion: "Pre-merger 63 provinces (until 30 Jun 2025) + Hoang Sa Island & Truong Sa Island",
    attributionSummary:
      "Boundaries: geoBoundaries VNM-ADM1 (CC BY 4.0 / Public Domain); Hoang Sa Island & Truong Sa Island: Free-GIS-Data (Nguyen Duy Liem, 2025). See ATTRIBUTION.md.",
    svgAriaLabel: "Map of Vietnam 63 pre-merger provinces with Hoang Sa Island and Truong Sa Island",
    statusSupported: "supported",
    statusUnsupported: "not supported",
    markerLabel: "Ho Chi Minh City model point",
    markerNote:
      "Model location (10.7769°N, 106.7009°E). Data is a grid cell of about 0.5° × 0.625°, not a measuring station.",
    gridCellVerifiedDetail:
      "Verified MERRA-2 grid cell covers 10.75°N–11.25°N, 106.5625°E–107.1875°E (CERES SYN1deg: 10.0°N–11.0°N, 106.0°E–107.0°E).",
    unsupportedNotice: (provinceName: string) =>
      `Forecast not available for ${provinceName}. The model has only been trained and validated for Ho Chi Minh City.`,
    unsupportedActionHint: "Selecting another active location requires enabling its data and artifacts in the backend store.",
    loadingMap: "Loading local boundary vectors...",
    mapLoadError: "Unable to load local boundary vectors.",
    noSearchMatches: "No matching province or island found.",
    tooltipSupportedAction: "Click or press Enter to select location",
    tooltipUnsupportedAction: "Not supported for forecasting",
  },
  forecast: {
    anchorSectionTitle: "Anchor date selection",
    anchorLabel: "Anchor date",
    anchorTooltip: "The last day of observed input; the 7 days after it are shown",
    anchorHelpRange: (minDate: string, maxDate: string) =>
      `Available COMPLETE_ALL_35 range: ${minDate} to ${maxDate}`,
    runButton: "Run retrospective replay",
    loadingResult: "Running 7-day inference...",
    quickPresetsLabel: "Quick anchor presets",
    presetLatest: "Latest complete",
    presetWet: "Mid-monsoon",
    presetWetAnchor: "2026-07-15",
    presetDry: "Dry-to-wet onset",
    presetDryAnchor: "2026-05-01",
    defaultLocationName: "Ho Chi Minh City",
    horizonDayLabel: (h: number) => `+${h}d`,
    dockTitle: (formattedDate: string, isRetrospective: boolean) =>
      isRetrospective
        ? `7 days after anchor date ${formattedDate} (retrospective replay)`
        : `7 days after anchor date ${formattedDate}`,
    errorHeading: (formattedDate: string) => `Result for anchor date ${formattedDate}`,
    retrospectiveBanner: "This is a retrospective replay, not a future forecast.",
    selectedLocationLabel: "Selected location",
    effectiveFutureLabel: "Future days in window",
    splitRegimeLabel: "Split regime",
    chartHeading: "7-day rainfall comparison (mm)",
    chartYAxisLabel: "Rainfall (mm)",
    chartEmpty: "Select a valid anchor date to inspect 7-day model outputs.",
    chartHoverHint: "Hover or tap a day bar to inspect exact values.",
    seriesTogglesLabel: "Visible series",
    seriesLabels: {
      observed: "Observed",
      "SC-35": "SC-35 (XGBoost F3, @serving)",
      "SC-30": "SC-30 (LightGBM F2)",
      "SC-31": "SC-31 (LightGBM F3)",
      BASELINE_CLIMATOLOGY_DOY: "Climatology DOY",
      BASELINE_PERSISTENCE: "Persistence",
    } as Record<string, string>,
    statusLabels: {
      past_observed: "Observed",
      today: "Operational today",
      future_forecast: "Future forecast",
      past_unverified: "Unverified past",
    } as Record<DayForecastStatus, string>,
    tableHeaders: {
      horizon: "Day",
      readableDate: "Date",
      isoDate: "ISO date",
      status: "Status",
      observed: "Observed (mm)",
      sc35: "SC-35 (mm)",
      clim: "Climatology (mm)",
      sc30: "SC-30 (mm)",
      sc31: "SC-31 (mm)",
      pers: "Persistence (mm)",
    },
    nullValue: "—",
    millimeterUnit: "mm",
  },
  dataStatus: {
    heading: "Data retrieval & completeness status",
    subtitle: "Ingested NASA POWER feeds, scan window coverage, and lag transparency from /health.",
    scanWindowCardTitle: "Latest NASA POWER scan window",
    scanWindowRangeLabel: "Scan window",
    scanWindowDaysLabel: "Total scanned days",
    scanWindowSnapshotLabel: "Latest archived row",
    scanWindowCompleteLabel: "Latest COMPLETE_ALL_35 anchor",
    definitionsHeading: "Two freshness definitions side by side",
    latestPrecipTitle: "Latest observed day (by PRECTOTCORR)",
    latestPrecipDesc: "Most recent date with a non-null rainfall observation (other variables may still be pending).",
    latestGroupTitle: "Latest valid day per group (all variables present)",
    latestGroupDesc: "Most recent date where every variable in the data group is populated without gaps.",
    partialExplanationTitle: "Latest partial observation note",
    groupTableHeaders: {
      group: "Feed group",
      lastValid: "Latest valid day",
      lagDays: "Lag vs scan end",
      validDays: "Valid days in scan",
    },
    groupLabels: {
      METEO_GEOSIT: "METEO_GEOSIT (24 meteorological variables)",
      RAD_FLASHFLUX: "RAD_FLASHFLUX (3 surface/TOA radiation variables)",
      RAD_CERES_SYN1DEG: "RAD_CERES_SYN1DEG (8 CERES radiation variables)",
    } as Record<string, string>,
    daysUnit: "days",
    notAvailable: "Not available",
  },
  scores: {
    heading: "Post-test comparison & Moving Block Bootstrap scores",
    subtitle: "Evaluated on completed observations under canonical SQLite DDL V3.2 ledger rules.",
    loadingText: "Loading evaluation ledger metrics...",
    metaOrigins: "Anchor origins",
    metaForecasts: "Forecasts per model",
    metaWindow: "Origin window",
    metaMaseScale: "Frozen DevTrain MASE scale",
    modelsHeading: "Model performance summary (unranked)",
    modelsCaption: "Bootstrap confidence intervals use 7-day blocks",
    modelHeaders: {
      model: "Model",
      role: "Kind",
      mae: "MAE (mm)",
      maeCi: "MAE 95% CI",
      rmse: "RMSE (mm)",
      mase: "MASE",
      maseCi: "MASE 95% CI",
    },
    pairwiseHeading: "Pairwise MAE differences (95% Block Bootstrap CI)",
    pairwiseHeaders: {
      pair: "Comparison pair",
      delta: "Delta MAE (mm)",
      ci: "95% CI bounds",
      width: "CI width",
      decision: "Interpretation",
    },
    sensitivityHeading: "Block-length sensitivity (7, 14, and 21 days)",
    sensitivitySubtitle: "Shows how SC-30 vs SC-31 and model MAE intervals widen as block length grows.",
    sensitivityHeaders: {
      blockLength: "Block length",
      blocks: "Blocks (n_blocks)",
      sc35Mae: "SC-35 MAE & 95% CI",
      sc31Mae: "SC-31 MAE & 95% CI",
      sc30Mae: "SC-30 MAE & 95% CI",
      sc30vs31: "SC-30 − SC-31 ΔMAE (95% CI, width)",
    },
    notesHeading: "Mandatory methodological notes",
    noteTitles: {
      effective_sample_size: "Effective sample size",
      first_vs_latest_equality: "First-seen vs latest observations",
      seasonal_regime_note: "Seasonal regime & magnitude",
    } as Record<string, string>,
  },
  governance: {
    heading: "Model governance, training scope & operational limitations",
    subtitle: "Key methodological boundaries users must consider before interpreting any output.",
    cards: [
      {
        title: "Point-regression model (non-probabilistic)",
        body: "All ML models (SC-35, SC-30, SC-31) output single conditional point rainfall amounts in millimeters using Tweedie variance power 1.5. They do not emit rain probabilities, calibrated prediction intervals, or severe-weather alerts.",
      },
      {
        title: "Parameters frozen since April 2018",
        body: "Model weights and FeatureBuilder statistics were fit strictly on Dev Train (1 Jan 2001 to 30 Apr 2018). Neither post-2018 validation/test observations nor 2026 post-test records have been used to retrain parameters.",
      },
      {
        title: "Governance roles & Test-set exposure",
        body: "The Validation-selected champion (2018–2020) is SC-30 (LightGBM F2 Config A). The deployed @serving model is SC-35 (XGBoost F3), chosen by the project owner after inspecting the 6-year Test set (2020–2026). SC-31 (LightGBM F3 Config A) is tracked as @shadow_test_contender. The 3 ML models are separated by overlapping bootstrap CIs and are not strictly ranked.",
      },
      {
        title: "Evaluated on completed data · NRT & forward-test status",
        body: "All displayed metrics come from completed, quality-checked archives. CERES SYN1deg lags real time by roughly 75–85 days, NASA POWER Near-Real-Time (NRT) revision bias has not yet been quantified, and prospective forward operational testing has not yet been executed.",
      },
      {
        title: "Demo snapshot & spatial scope",
        body: "When running the bundled container image without an external production store, the database is a frozen demo snapshot (latest archived day 15 Nov 2026; latest complete anchor 30 Aug 2026). The 65-feature pre-merger map is provided strictly for spatial orientation and to display configured backend location support; grid-cell forecasts cover only Ho Chi Minh City (10.7769°N, 106.7009°E).",
      },
    ],
  },
} as const;

export function mapApiErrorToFriendly(err: {
  code?: string;
  message?: string;
  details?: Record<string, unknown>;
  status?: number;
}): FriendlyErrorView {
  const code = err.code || (err.status && err.status >= 500 ? "SERVER_DB_DATA_ERROR" : "UNKNOWN");
  const summary = EN.errors.byCode[code] ?? EN.errors.byCode.UNKNOWN;
  return {
    summary,
    code,
    message: err.message || EN.errors.requestFailedFallback,
    details: err.details ?? {},
  };
}
