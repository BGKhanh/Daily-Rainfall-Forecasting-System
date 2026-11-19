import React, { useEffect, useRef, useState } from "react";
import {
  APP_CONFIG,
  COLD_START_THRESHOLD_MS,
  fetchHealth,
  fetchLocations,
  fetchObservations,
  fetchPredictByAnchor,
  fetchVietnamGeoJson,
} from "./api/client";
import { AboutLimitationsSection } from "./components/AboutLimitationsSection";
import { AnchorForecastSection } from "./components/AnchorForecastSection";
import { DataStatusSection } from "./components/DataStatusSection";
import { EN } from "./i18n/en";
import type {
  HealthResponse,
  LocationItem,
  NavTabId,
  ObservationSeriesPoint,
  PredictResponse,
  ProvinceGeoJson,
} from "./types/api";

interface AppProps {
  initialGeoJson?: ProvinceGeoJson;
}

const NAV_ORDER: NavTabId[] = ["forecast", "data", "governance"];

function isoMinusDays(iso: string, days: number): string {
  const dt = new Date(`${iso}T00:00:00Z`);
  if (Number.isNaN(dt.getTime())) return iso;
  return new Date(dt.getTime() - days * 86400000).toISOString().slice(0, 10);
}

export const App: React.FC<AppProps> = ({ initialGeoJson }) => {
  const [activeTab, setActiveTab] = useState<NavTabId>("forecast");
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [locations, setLocations] = useState<LocationItem[]>([]);
  const [selectedLocationIds, setSelectedLocationIds] = useState<string[]>(["HCMC"]);
  const [focusedLocationId, setFocusedLocationId] = useState<string>("HCMC");
  const [geoJson, setGeoJson] = useState<ProvinceGeoJson | null>(initialGeoJson ?? null);
  const [mapError, setMapError] = useState<string | null>(null);

  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [coldStartSlow, setColdStartSlow] = useState<boolean>(false);
  const [healthError, setHealthError] = useState<unknown>(null);
  const [predictError, setPredictError] = useState<unknown>(null);
  const [predictLoading, setPredictLoading] = useState<boolean>(true);

  const [showPrior14d, setShowPrior14d] = useState<boolean>(false);
  const [priorObservations, setPriorObservations] = useState<ObservationSeriesPoint[]>([]);

  const predictSeqRef = useRef<number>(0);

  const loadPrediction = async (locationId: string, anchorDate?: string) => {
    const seq = ++predictSeqRef.current;
    setPredictLoading(true);
    setPredictError(null);
    try {
      const predRes = await fetchPredictByAnchor(locationId, anchorDate);
      if (seq !== predictSeqRef.current) return;
      setPrediction(predRes);
    } catch (err) {
      if (seq !== predictSeqRef.current) return;
      setPrediction(null);
      setPredictError(err);
    } finally {
      if (seq === predictSeqRef.current) {
        setPredictLoading(false);
      }
    }
  };

  useEffect(() => {
    let active = true;
    const coldTimer = window.setTimeout(() => {
      if (active) setColdStartSlow(true);
    }, COLD_START_THRESHOLD_MS);

    if (!initialGeoJson) {
      fetchVietnamGeoJson()
        .then((gj) => {
          if (!active) return;
          setGeoJson(gj);
        })
        .catch(() => {
          if (!active) return;
          setMapError(EN.map.mapLoadError);
        });
    }

    async function bootstrap() {
      try {
        const [healthData, locsData] = await Promise.all([fetchHealth(), fetchLocations()]);
        if (!active) return;
        window.clearTimeout(coldTimer);
        setColdStartSlow(false);
        setHealth(healthData);
        setLocations(locsData.locations);

        const defaultLoc =
          locsData.locations.find((l) => l.is_active)?.location_id ?? "HCMC";
        setSelectedLocationIds([defaultLoc]);
        setFocusedLocationId(defaultLoc);

        const initialAnchor = healthData.default_complete_anchor_date ?? "";
        void loadPrediction(defaultLoc, initialAnchor || undefined);
      } catch (err) {
        if (!active) return;
        window.clearTimeout(coldTimer);
        setColdStartSlow(false);
        setHealthError(err);
        setPredictError(err);
        setPredictLoading(false);
      }
    }

    void bootstrap();
    return () => {
      active = false;
      predictSeqRef.current += 1;
      window.clearTimeout(coldTimer);
    };
  }, [initialGeoJson]);

  const handleToggleLocation = (locationId: string) => {
    setFocusedLocationId(locationId);
    setSelectedLocationIds((prev) =>
      prev.includes(locationId) ? prev : [...prev, locationId]
    );
    const activeLoc = locations.find((l) => l.location_id === locationId && l.is_active);
    if (activeLoc) {
      void loadPrediction(locationId, health?.default_complete_anchor_date ?? undefined);
    }
  };

  const handleResetSelection = () => {
    const defaultLoc = locations.find((l) => l.is_active)?.location_id ?? "HCMC";
    setSelectedLocationIds([defaultLoc]);
    setFocusedLocationId(defaultLoc);
    void loadPrediction(defaultLoc, health?.default_complete_anchor_date ?? undefined);
  };

  const handleTogglePrior14d = async (enabled: boolean) => {
    setShowPrior14d(enabled);
    if (!enabled) return;
    const anchorIso = prediction?.anchor_date ?? health?.default_complete_anchor_date;
    if (!anchorIso) return;
    try {
      const startIso = isoMinusDays(anchorIso, 13);
      const res = await fetchObservations({
        locationId: focusedLocationId,
        startDate: startIso,
        endDate: anchorIso,
        granularity: "daily",
      });
      setPriorObservations(res.series);
    } catch {
      setPriorObservations([]);
    }
  };

  const switchTab = (tabId: NavTabId) => {
    setActiveTab(tabId);
    setMobileMenuOpen(false);
  };

  const dbRoleKey = health?.db_role ?? "connecting";
  const dbLabel = EN.app.dbRoles[dbRoleKey] ?? dbRoleKey;

  return (
    <div className="shell-root">
      <div className="mobile-header">
        <button
          type="button"
          className="menu-toggle-btn"
          aria-label={EN.app.menuButtonLabel}
          aria-expanded={mobileMenuOpen}
          onClick={() => setMobileMenuOpen((v) => !v)}
        >
          <span className="menu-icon-bar" />
          <span className="menu-icon-bar" />
          <span className="menu-icon-bar" />
        </button>
        <div className="mobile-brand-text">{EN.app.brandTitle}</div>
      </div>

      {mobileMenuOpen && (
        <div
          className="drawer-backdrop"
          onClick={() => setMobileMenuOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={`sidebar ${mobileMenuOpen ? "sidebar--open" : ""}`}
        aria-label={EN.app.navAriaLabel}
      >
        <div className="sidebar-brand">
          <h1>{EN.app.brandTitle}</h1>
          <p>{EN.app.brandSubtitle}</p>
        </div>

        <nav className="sidebar-nav" aria-label={EN.app.navAriaLabel}>
          {NAV_ORDER.map((tabId) => (
            <button
              key={tabId}
              type="button"
              className={`nav-tab-btn ${activeTab === tabId ? "nav-tab-btn--active" : ""}`}
              aria-current={activeTab === tabId ? "page" : undefined}
              onClick={() => switchTab(tabId)}
            >
              {EN.tabs[tabId]}
            </button>
          ))}
        </nav>

        <footer className="sidebar-footer">
          <div className="db-badge" title={health?.db_role ?? EN.app.footerConnecting}>
            <span className="db-dot" />
            <span>{`${EN.app.footerDbLabel}: ${dbLabel}`}</span>
          </div>
          <div className="footer-links" aria-label={EN.app.footerLinksAria}>
            <a href={APP_CONFIG.repoUrl} target="_blank" rel="noopener noreferrer">
              {EN.app.linkGithub}
            </a>
            <a href={APP_CONFIG.ciUrl} target="_blank" rel="noopener noreferrer">
              {EN.app.linkCi}
            </a>
            <a href={APP_CONFIG.mlflowUrl} target="_blank" rel="noopener noreferrer">
              {EN.app.linkMlflow}
            </a>
          </div>
        </footer>
      </aside>

      <div className="workspace-col">
        <div className="persistent-summary-strip" role="region" aria-label={EN.tabs.governance}>
          <span>{EN.app.summaryStripPrefix}</span>
          <button
            type="button"
            className="strip-link-btn"
            onClick={() => switchTab("governance")}
          >
            {EN.app.summaryStripLink}
          </button>
        </div>

        <main className="main-content">
          {coldStartSlow && (
            <div className="info-banner" role="status">
              {EN.app.coldStartNotice}
            </div>
          )}

          {healthError !== null && activeTab !== "forecast" && (
            <div className="error-state-panel" role="alert">
              <p className="error-summary">{EN.errors.byCode.SERVER_DB_DATA_ERROR}</p>
            </div>
          )}

          {activeTab === "forecast" && (
            <AnchorForecastSection
              health={health}
              locations={locations}
              selectedLocationIds={selectedLocationIds}
              focusedLocationId={focusedLocationId}
              onToggleLocation={handleToggleLocation}
              onResetSelection={handleResetSelection}
              geoJson={geoJson}
              mapError={mapError}
              prediction={prediction}
              loading={predictLoading}
              errorValue={predictError}
              priorObservations={priorObservations}
              showPrior14d={showPrior14d}
              onTogglePrior14d={handleTogglePrior14d}
            />
          )}

          {activeTab === "data" && (
            <DataStatusSection health={health} locations={locations} geoJson={geoJson} />
          )}

          {activeTab === "governance" && <AboutLimitationsSection />}
        </main>
      </div>
    </div>
  );
};

export default App;
