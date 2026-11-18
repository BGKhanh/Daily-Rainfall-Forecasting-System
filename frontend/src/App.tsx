import React, { useEffect, useRef, useState } from "react";
import {
  APP_CONFIG,
  COLD_START_THRESHOLD_MS,
  fetchHealth,
  fetchLocations,
  fetchPredictByAnchor,
  fetchScores,
  fetchVietnamGeoJson,
} from "./api/client";
import { AboutLimitationsSection } from "./components/AboutLimitationsSection";
import { AnchorForecastSection } from "./components/AnchorForecastSection";
import { DataStatusSection } from "./components/DataStatusSection";
import { ScoresSection } from "./components/ScoresSection";
import { EN } from "./i18n/en";
import type {
  HealthResponse,
  LocationItem,
  NavTabId,
  PredictResponse,
  ProvinceGeoJson,
  ScoresResponse,
} from "./types/api";

interface AppProps {
  initialGeoJson?: ProvinceGeoJson;
}

const NAV_ORDER: NavTabId[] = ["forecast", "data", "scores", "governance"];

export const App: React.FC<AppProps> = ({ initialGeoJson }) => {
  const [activeTab, setActiveTab] = useState<NavTabId>("forecast");
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [locations, setLocations] = useState<LocationItem[]>([]);
  const [selectedLocationId, setSelectedLocationId] = useState<string>("HCMC");
  const [geoJson, setGeoJson] = useState<ProvinceGeoJson | null>(initialGeoJson ?? null);
  const [mapError, setMapError] = useState<string | null>(null);

  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [scores, setScores] = useState<ScoresResponse | null>(null);

  const [selectedAnchor, setSelectedAnchor] = useState<string>("");
  const [coldStartSlow, setColdStartSlow] = useState<boolean>(false);

  const [healthError, setHealthError] = useState<unknown>(null);
  const [predictError, setPredictError] = useState<unknown>(null);
  const [scoresError, setScoresError] = useState<unknown>(null);

  const [predictLoading, setPredictLoading] = useState<boolean>(true);
  const [scoresLoading, setScoresLoading] = useState<boolean>(true);
  const predictSeqRef = useRef<number>(0);

  const loadPrediction = async (locationId: string, anchorDate?: string) => {
    const seq = ++predictSeqRef.current;
    setPredictLoading(true);
    setPredictError(null);
    try {
      const predRes = await fetchPredictByAnchor(locationId, anchorDate);
      if (seq !== predictSeqRef.current) return;
      setPrediction(predRes);
      if (predRes.anchor_date) {
        setSelectedAnchor(predRes.anchor_date);
      }
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
        setSelectedLocationId(defaultLoc);

        const initialAnchor = healthData.default_complete_anchor_date ?? "";
        setSelectedAnchor(initialAnchor);

        void loadPrediction(defaultLoc, initialAnchor || undefined);
      } catch (err) {
        if (!active) return;
        window.clearTimeout(coldTimer);
        setColdStartSlow(false);
        setHealthError(err);
        setPredictError(err);
        setPredictLoading(false);
      }

      setScoresLoading(true);
      setScoresError(null);
      fetchScores("POST_TEST")
        .then((scoresData) => {
          if (!active) return;
          setScores(scoresData);
        })
        .catch((err) => {
          if (!active) return;
          setScoresError(err);
        })
        .finally(() => {
          if (active) setScoresLoading(false);
        });
    }

    void bootstrap();
    return () => {
      active = false;
      predictSeqRef.current += 1;
      window.clearTimeout(coldTimer);
    };
  }, [initialGeoJson]);

  const handleSelectAnchor = async (anchorDate: string) => {
    setSelectedAnchor(anchorDate);
    await loadPrediction(selectedLocationId, anchorDate);
  };

  const handleSelectLocation = async (locationId: string) => {
    setSelectedLocationId(locationId);
    await loadPrediction(locationId, selectedAnchor || undefined);
  };

  const switchTab = (tabId: NavTabId) => {
    setActiveTab(tabId);
    setMobileMenuOpen(false);
  };

  const dbLabel =
    EN.app.dbRoles[health?.db_role ?? "connecting"] ?? health?.db_role ?? EN.app.footerConnecting;

  return (
    <div className="shell-root">
      <div className="mobile-header">
        <button
          type="button"
          className="menu-toggle-btn"
          aria-label={mobileMenuOpen ? EN.app.closeMenuLabel : EN.app.openMenuLabel}
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
            <a href={APP_CONFIG.repoUrl} target="_blank" rel="noreferrer">
              {EN.app.linkGithub}
            </a>
            <a href={APP_CONFIG.ciUrl} target="_blank" rel="noreferrer">
              {EN.app.linkCi}
            </a>
            <a href={APP_CONFIG.mlflowUrl} target="_blank" rel="noreferrer">
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
              selectedLocationId={selectedLocationId}
              onSelectLocation={handleSelectLocation}
              geoJson={geoJson}
              mapError={mapError}
              prediction={prediction}
              loading={predictLoading}
              errorValue={predictError}
              selectedAnchor={selectedAnchor}
              onSelectAnchor={handleSelectAnchor}
            />
          )}

          {activeTab === "data" && <DataStatusSection health={health} />}

          {activeTab === "scores" && (
            <ScoresSection scores={scores} loading={scoresLoading} errorValue={scoresError} />
          )}

          {activeTab === "governance" && <AboutLimitationsSection />}
        </main>
      </div>
    </div>
  );
};

export default App;
