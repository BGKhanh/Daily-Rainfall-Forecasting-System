export interface FriendlyErrorView {
  summary: string;
  code: string;
  message: string;
  details: Record<string, unknown>;
}

const MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function formatReadableDate(isoDate?: string | null): string {
  if (!isoDate) return "--";
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(isoDate.trim());
  if (!match) return isoDate;
  const year = match[1];
  const monthIdx = Number(match[2]) - 1;
  const dayNum = Number(match[3]);
  const mon = MONTH_ABBR[monthIdx] ?? match[2];
  return `${dayNum} ${mon} ${year}`;
}

export function formatUtcDateTime(isoTs?: string | null): string {
  if (!isoTs) return "--";
  const d = new Date(isoTs);
  if (Number.isNaN(d.getTime())) return isoTs;
  const day = d.getUTCDate();
  const mon = MONTH_ABBR[d.getUTCMonth()] ?? "";
  const year = d.getUTCFullYear();
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  return `${day} ${mon} ${year} ${hh}:${mm} UTC`;
}

export function stripDiacriticsForSearch(raw: string): string {
  return raw
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d")
    .replace(/Đ/g, "D")
    .toLowerCase()
    .trim();
}

export const EN = {
  app: {
    brandTitle: "Vietnam Daily Rainfall Console",
    brandSubtitle: "Single-model rainfall forecast & archive explorer",
    menuButtonLabel: "Open navigation menu",
    navAriaLabel: "Primary sections",
    summaryStripPrefix: "Single location (Ho Chi Minh City) · evaluated on completed data · real-time forecasting not supported",
    summaryStripLink: "Read limitations",
    coldStartNotice: "Backend container is initializing model artifacts; first response may take a few seconds.",
    footerDbLabel: "Active store",
    footerConnecting: "Connecting...",
    footerLinksAria: "Project links",
    linkGithub: "GitHub",
    linkCi: "CI Runs",
    linkMlflow: "MLflow",
    dbRoles: {
      bundled_demo_db: "Bundled Demo DB",
      local_or_production_db: "Local SQLite Store",
      sqlite: "SQLite Store",
      turso_replica: "Turso Replica",
      turso_remote: "Turso Cloud Store",
      sample_csv: "Static 35-Day CSV",
      connecting: "Connecting",
    } as Record<string, string>,
  },
  tabs: {
    forecast: "Forecast by location",
    data: "Data retrieval",
    governance: "System limitations",
  },
  errors: {
    technicalDetailsTitle: "Technical details",
    requestFailedFallback: "Request failed",
    byCode: {
      UNSUPPORTED_LOCATION:
        "This location is outside the 65 configured administrative units or not yet enabled for forecasting.",
      INCOMPLETE_RADIATION_AT_ANCHOR:
        "Complete satellite radiation variables are required at the anchor date and are not yet available.",
      ANCHOR_DATE_NOT_FOUND:
        "The requested anchor date was not found in the observation archive.",
      INVALID_CLIENT_HISTORY:
        "The submitted input sequence or parameters violate continuity or physical bounds.",
      SERVER_DB_DATA_ERROR:
        "The observation database could not complete validation for this request.",
      PREDICTOR_NOT_READY:
        "Model artifacts are still loading into memory. Please retry shortly.",
      INVALID_QUERY_PARAMETER:
        "One or more query parameters fall outside allowed ranges or point limits.",
      UNKNOWN:
        "Unable to load results for this request. Please verify parameters and try again.",
    } as Record<string, string>,
  },
  map: {
    heading: "Location & dual-layer spatial map",
    subtitle: "Toggle province boundaries and MERRA-2 grid cells independently, or select multiple units.",
    toggleBoundaries: "Province boundaries",
    toggleGridCells: "Grid cells",
    searchLabel: "Find province or island",
    searchPlaceholder: "Search (e.g., Ho Chi Minh, Ha Noi, Hoang Sa)",
    listHeading: "Pre-merger units (65)",
    selectedTableHeading: "Selected locations",
    selectedColLocation: "Location",
    selectedColStatus: "Forecast status",
    selectedColCells: "Intersecting grid cells",
    statusAvailable: "Available",
    statusNotAvailableYet: "Not available yet",
    legendTitle: "Map legend",
    legendSupported: "Supported (forecast active)",
    legendUnsupported: "Not available yet",
    legendHighlightedCell: "Intersecting / model MERRA-2 0.5° × 0.625° grid cell",
    legendGridLattice: "MERRA-2 0.5° × 0.625° grid lattice",
    legendVersion: "Pre-merger 63 provinces (until 30 Jun 2025) + Hoang Sa Island & Truong Sa Island",
    gridFormulaNote:
      "Grid lines are computed from the documented MERRA-2 0.5° × 0.625° grid structure and empirically verified via API probes at Ho Chi Minh City, Ha Noi, Da Nang, and Can Tho.",
    attributionSummary:
      "Boundaries: geoBoundaries VNM-ADM1 (CC BY 4.0 / Public Domain); Hoang Sa Island & Truong Sa Island: Free-GIS-Data (Nguyen Duy Liem, 2025). See ATTRIBUTION.md.",
    svgAriaLabel: "Dual-layer map of Vietnam 63 pre-merger provinces, islands, and MERRA-2 grid cells",
    statusSupported: "available",
    statusUnsupported: "not available yet",
    markerLabel: "Ho Chi Minh City model point",
    markerNote:
      "Model location (10.7769°N, 106.7009°E). Data is a grid cell of about 0.5° × 0.625°, not a measuring station.",
    gridCellVerifiedDetail:
      "Verified MERRA-2 cell [10.75°N, 11.25°N] × [106.5625°E, 107.1875°E]; radiation cell [10.0°N, 11.0°N] × [106.0°E, 107.0°E].",
    loadingMap: "Loading local boundary vectors...",
    mapLoadError: "Unable to load local boundary vectors.",
    noSearchMatches: "No matching province or island found.",
    resetSelectionBtn: "Reset",
    clearSelectionBtn: "Clear selection",
    selectedCountBadge: (count: number) => `${count} selected`,
    deselectBtnLabel: (name: string) => `Deselect ${name}`,
    deselectSymbol: "×",
    zoomInLabel: "+",
    zoomOutLabel: "−",
    zoomInAria: "Zoom in map",
    zoomOutAria: "Zoom out map",
    zoomResetLabel: "Reset view",
    zoomLevelText: (z: number) => `${z.toFixed(1)}×`,
    zoomHint: "Use +/− buttons, Ctrl/Cmd + scroll wheel, two-finger pinch, or arrow keys when map is focused.",
    hoverGridHint: "MERRA-2 grid cell bounds",
  },
  gridScale: {
    heading: "Grid-scale & spatial coverage alignment",
    subtitle:
      "Compare an administrative province polygon directly against the two empirical NASA POWER data grids: MERRA-2 / GEOS-IT (0.5° × 0.625°) and CERES SYN1deg / FLASHFlux (1.0° × 1.0°).",
    provinceSelectAria: "Select province for grid alignment",
    provinceSelectLabel: "Inspect province",
    showOverlapToggle: "Show cell-to-province overlap %",
    loadingOverlap: "Loading spatial overlap geometry...",
    svgAriaLabel: (name: string) => `True geographic overlay of ${name} boundary and intersecting NASA POWER grid cells`,
    legendProvince: "Province polygon",
    legendMerraAll: "Intersecting MERRA-2 / GEOS-IT (0.5° × 0.625°)",
    legendMerraRep: "Representative MERRA-2 cell (input feed)",
    legendCeresAll: "Intersecting CERES / FLASHFlux (1.0° × 1.0°)",
    legendCeresRep: "Representative CERES cell",
    legendRepPoint: "Configured coordinate anchor",
    metricLatitudeLabel: "Latitude φ / Polygon area",
    metricMerraLabel: "MERRA-2 / GEOS-IT cell (0.5° × 0.625°)",
    metricCeresLabel: "CERES / FLASHFlux cell (1.0° × 1.0°)",
    metricRatioLabel: "Intersecting cells (MERRA-2 / CERES)",
    metricAreaUnit: "km²",
    overlapTableHeading: "Cell-to-province overlap breakdown",
    overlapColGrid: "Grid",
    overlapColBounds: "Cell bounds (Lat × Lon)",
    overlapColInterKm2: "Overlap (km²)",
    overlapColPctProv: "% of province",
    overlapColPctCell: "% of cell",
    overlapColRep: "Rep. point",
    repYesBadge: "Anchor",
    timelineHeading: "Empirical source transition & documentation notes (as of 2026-11-19 probe)",
    timelineCeres: "Radiation transitions SYN1deg → FLASHFlux on 31 Aug 2026 (verified ~81-day lag as of 2026-11-19 probe).",
    timelineMerra: "Meteorology transitions MERRA-2 → GEOS-IT on 1 Oct 2026 (verified ~50-day lag as of 2026-11-19 probe).",
    timelineFaqNote:
      "Solar FAQ note: NASA POWER Solar FAQ mentions 1/2° × 1/2°, while Services Documentation and our empirical boundary probe across all 11 solar variables confirm 1.0° × 1.0°.",
    catalogHeading: "Canonical 35 serving input variables by verified spatial grid",
    catalogGroupMerra: "0.5° × 0.625° MERRA-2 / GEOS-IT meteorology & surface variables (24)",
    catalogGroupCeres: "1.0° × 1.0° CERES SYN1deg / FLASHFlux radiation & UV variables (11)",
    catalogHeaders: {
      code: "Code",
      name: "Variable",
      unit: "Unit",
      finalized: "Finalized source",
      provisional: "Provisional source",
    },
  },
  forecast: {
    mainHeading: (anchorDateStr: string, todayDateStr: string) =>
      `Forecast issued from data up to ${anchorDateStr} (latest complete input). Today: ${todayDateStr}`,
    retrospectiveBanner: (lagDays: number) =>
      `Retrospective replay: latest complete archive input lags today by ${lagDays} days, so observed rainfall is shown beside the model line.`,
    technicalBannerTooltip:
      "Technical note: Satellite CERES radiation (SYN1deg) lags real time by ~80 days. When all 7 forecasted days precede today, zero future days remain unobserved.",
    unavailableHeading: (locationName: string) => `Forecast for ${locationName}`,
    unavailableMessage: "Forecast not available yet",
    unavailableDetail:
      "The serving point-regression model has only been trained and validated for Ho Chi Minh City.",
    priorObservationsToggle: "Show 14 days of prior observations before anchor date",
    selectedLocationLabel: "Selected location",
    defaultLocationName: "Ho Chi Minh City",
    chartHeading: "7-day rainfall line comparison (mm)",
    chartYAxisLabel: "Rainfall (mm)",
    chartHoverHint: "Hover or tap a day point to inspect exact values.",
    legendForecast: "Forecast (@serving dashed)",
    legendObserved: "Observed (solid)",
    loadingResult: "Loading 7-day @serving forecast...",
    errorHeading: (formattedDate: string) => `Result for anchor date ${formattedDate}`,
    tableHeaders: {
      readableDate: "Date",
      isoDate: "ISO date",
      status: "Status",
      observed: "Observed (mm)",
      forecast: "Forecast (mm)",
    },
    statusLabels: {
      past_observed: "Observed",
      today: "Operational today",
      future_forecast: "Future forecast",
      past_unverified: "Unverified past",
    } as Record<string, string>,
    cardLabels: {
      observed: "Observed (mm)",
      forecast: "Forecast (mm)",
      status: "Status",
    },
    nullValue: "--",
    mmUnit: "mm",
    hoverReadoutPrefix: "Day detail",
  },
  dataRetrieval: {
    heading: "Data retrieval & precipitation archive explorer",
    subtitle:
      "Query real SQLite archived daily rainfall (PRECTOTCORR) with source finality cutoffs, aggregation, gaps, and CSV download.",
    locationLabel: "Location",
    startDateLabel: "Start date",
    endDateLabel: "End date",
    granularityLabel: "Granularity",
    granularityOptions: {
      daily: "Daily (max 366d)",
      weekly: "Weekly (ISO Mon–Sun)",
      monthly: "Monthly (YYYY-MM)",
    } as Record<string, string>,
    metricModeLabel: "Aggregate metric",
    metricTotalOption: "Period total (mm)",
    metricMeanOption: "Daily mean (mm/day)",
    preset90Days: "Last 90 days",
    preset2Years: "Last 2 years (Weekly)",
    presetFullHistory: "Full history (Monthly)",
    runQueryButton: "Query archive",
    autoSwitchedNotice:
      "Date span exceeds 366 days; granularity was automatically switched to preserve readable point density.",
    loadingQuery: "Querying observations archive...",
    emptyLocationTitle: "Archive unavailable for location",
    emptyLocationMessage: "No observations have been ingested for this location yet.",
    summaryTotalLabel: "Total rainfall (mm)",
    summaryRainDaysLabel: "Rain days (>= 1.0 mm)",
    summaryBreakdownLabel: "Finalized / Provisional / Not yet published",
    cutoffLabel: "Finalized cutoff",
    cutoffWithProbe: (cutoffDate: string, probeDate?: string | null) =>
      probeDate
        ? `Finalized cutoff: ${cutoffDate} (probed ${probeDate})`
        : `Finalized cutoff: ${cutoffDate}`,
    chartHeading: "Precipitation timeline & source finality",
    yAxisDaily: "Rainfall (mm/day)",
    yAxisPeriodTotal: "Period total (mm)",
    yAxisDailyMean: "Daily mean (mm/day)",
    legendFinalized: "Finalized (solid)",
    legendProvisional: "Provisional (solid)",
    legendForecast: "Forecast (@serving dashed)",
    legendNotPublished: "Not yet published by NASA POWER",
    overlayDailyOnlyNote:
      "The 7-day @serving forecast overlay line applies to Daily resolution only. Switch Granularity to Daily to view daily predicted vs observed rainfall.",
    partialPeriodBadge: (pub: number, cal: number) => `Partial period (${pub}/${cal} days)`,
    notPublishedAsOf: (scanDate?: string | null) =>
      scanDate ? `Not yet published (as of the ${scanDate} scan)` : "Not yet published",
    finalityTooltipLabel: "Finality inference disclosure",
    finalityTooltipText:
      "NASA POWER does not tag individual daily rows as finalized or provisional. Status is inferred from NASA POWER per-day source metadata (MERRA-2 vs GEOS-IT transition cutoff) or the configured finality threshold. Because each date in this store currently has a single ingested version (v0), actual NRT-to-Final value revisions have not yet been observed.",
    statusLabels: {
      FINALIZED: "Finalized",
      PROVISIONAL: "Provisional",
      NOT_PUBLISHED: "Not yet published",
    } as Record<string, string>,
    downloadCsv: "Download CSV",
    tableHeaders: {
      period: "Period",
      periodEnd: "Period end",
      status: "Status",
      value: "Rainfall",
      publishedDays: "Published / Calendar days",
    },
    paginationInfo: (page: number, totalPages: number, totalRows: number) =>
      `Page ${page} of ${totalPages} (${totalRows} rows)`,
    prevPage: "Previous",
    nextPage: "Next",
    scanWindowHeading: "Latest ingest & partial-row status",
    scanWindowRangeLabel: "Scan window",
    latestPrecipLabel: "Latest PRECTOTCORR day",
    latestGroupLabel: "Latest valid group day",
    daysUnit: "days",
  },
  governance: {
    heading: "System scope, spatial representativeness & operational limitations",
    subtitle: "Scientific boundaries, grid resolution limits, and data latency facts for interpreting outputs.",
    cards: [
      {
        title: "Point-regression model (non-probabilistic)",
        body: "The serving model outputs single conditional point rainfall amounts in millimeters using Tweedie variance power 1.5. It does not emit rain probabilities, calibrated prediction intervals, or severe-weather alerts.",
      },
      {
        title: "Parameters frozen since April 2018",
        body: "Model weights and FeatureBuilder statistics were fit strictly on Dev Train (1 Jan 2001 to 30 Apr 2018). Neither post-2018 archive data nor 2026 records have been used to retrain model parameters.",
      },
      {
        title: "Evaluated on completed data · Latency & NRT status",
        body: "Meteorology from MERRA-2 transitions to near-real-time GEOS-IT around a verified ~50-day cutoff (30 Sep 2026 in the Nov 2026 probe), while CERES SYN1deg radiation lags by ~80–81 days (~30 Aug 2026). NRT-to-Final revision bias has not yet been quantified in this v0 store, and prospective forward operational testing has not been executed.",
      },
      {
        title: "Spatial representativeness",
        body: "NASA POWER meteorology inputs come from a gridded MERRA-2 / GEOS-IT cell of 0.5° × 0.625° (~56 × 68 km), not a point rain gauge or provincial average; any coordinates inside the same cell receive the exact same series. For Ho Chi Minh City (10.7769°N, 106.7009°E), the active cell [10.75°N, 11.25°N] × [106.5625°E, 107.1875°E] is shifted north and excludes southern districts such as Nha Be and Can Gio. Solar radiation variables come from an even coarser 1° × 1° grid [10.0°N, 11.0°N] × [106.0°E, 107.0°E] (note: NASA POWER Solar FAQ mentions ½° × ½°, whereas POWER Services and Data Sources documentation & our empirical boundary probes confirm 1° × 1°).",
        links: [
          { label: "NASA POWER Services Documentation (accessed 2026-11-19)", url: "https://power.larc.nasa.gov/docs/services/" },
          { label: "NASA POWER Solar FAQs (accessed 2026-11-19)", url: "https://power.larc.nasa.gov/docs/faqs/solar/" },
          { label: "NASA POWER Methodology & Data Sources (accessed 2026-11-19)", url: "https://power.larc.nasa.gov/docs/methodology/data/sources/" },
        ],
      },
      {
        title: "Archive snapshot & map orientation scope",
        body: "When running the bundled container without an external database, the SQLite archive reflects a static snapshot with real NASA POWER source-transition boundaries. The 65-unit pre-merger map and MERRA-2 grid overlay orient users spatially and indicate configured backend readiness.",
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
