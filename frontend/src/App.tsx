import React, { useEffect, useState } from "react";
import {
  COLD_START_THRESHOLD_MS,
  fetchHealth,
  fetchPredictByAnchor,
  fetchScores,
  mapApiErrorToVietnamese,
} from "./api/client";
import { AboutLimitationsSection } from "./components/AboutLimitationsSection";
import { AnchorForecastSection } from "./components/AnchorForecastSection";
import { DataStatusSection } from "./components/DataStatusSection";
import { ScoresSection } from "./components/ScoresSection";
import type { HealthResponse, PredictResponse, ScoresResponse } from "./types/api";

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [scores, setScores] = useState<ScoresResponse | null>(null);

  const [selectedAnchor, setSelectedAnchor] = useState<string>("");
  const [coldStartSlow, setColdStartSlow] = useState<boolean>(false);

  const [healthError, setHealthError] = useState<string | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [scoresError, setScoresError] = useState<string | null>(null);

  const [predictLoading, setPredictLoading] = useState<boolean>(true);
  const [scoresLoading, setScoresLoading] = useState<boolean>(true);

  useEffect(() => {
    let active = true;
    const coldTimer = window.setTimeout(() => {
      if (active) setColdStartSlow(true);
    }, COLD_START_THRESHOLD_MS);

    async function bootstrap() {
      try {
        const healthData = await fetchHealth();
        if (!active) return;
        window.clearTimeout(coldTimer);
        setColdStartSlow(false);
        setHealth(healthData);
        const initialAnchor = healthData.default_complete_anchor_date ?? "";
        setSelectedAnchor(initialAnchor);

        setPredictLoading(true);
        setPredictError(null);
        fetchPredictByAnchor(initialAnchor || undefined)
          .then((predRes) => {
            if (!active) return;
            setPrediction(predRes);
            setSelectedAnchor(predRes.anchor_date);
          })
          .catch((err) => {
            if (!active) return;
            setPredictError(mapApiErrorToVietnamese(err));
          })
          .finally(() => {
            if (active) setPredictLoading(false);
          });
      } catch (err) {
        if (!active) return;
        window.clearTimeout(coldTimer);
        setColdStartSlow(false);
        setHealthError(mapApiErrorToVietnamese(err));
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
          setScoresError(mapApiErrorToVietnamese(err));
        })
        .finally(() => {
          if (active) setScoresLoading(false);
        });
    }

    void bootstrap();
    return () => {
      active = false;
      window.clearTimeout(coldTimer);
    };
  }, []);

  const handleSelectAnchor = async (anchorDate: string) => {
    setSelectedAnchor(anchorDate);
    setPredictLoading(true);
    setPredictError(null);
    try {
      const predRes = await fetchPredictByAnchor(anchorDate);
      setPrediction(predRes);
    } catch (err) {
      setPredictError(mapApiErrorToVietnamese(err));
    } finally {
      setPredictLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand-block">
            <h1>Hệ thống Lượng mưa Ngày TP. Hồ Chí Minh</h1>
            <p className="brand-meta">
              DS108 · Khóa trạm HCMC (10.7769°N, 106.7009°E) ·{" "}
              {health?.db_role ? `DB: ${health.db_role}` : "Đang kết nối máy chủ"} · Snapshot:{" "}
              {health?.data_snapshot_date ?? "—"}
            </p>
          </div>
          <nav className="top-nav" aria-label="Điều hướng khu vực">
            <a href="#theo-ngay-neo">Theo ngày neo</a>
            <a href="#bang-diem">Bảng điểm</a>
            <a href="#tinh-trang-du-lieu">Tình trạng dữ liệu</a>
            <a href="#gioi-thieu-va-gioi-han">Giới thiệu &amp; Giới hạn</a>
          </nav>
        </div>
      </header>

      <main className="main-content">
        {coldStartSlow && (
          <div className="alert alert-warn" role="status">
            Máy chủ đang khởi động, vui lòng chờ
          </div>
        )}

        {healthError && (
          <div className="alert alert-error" role="alert">
            {healthError}
          </div>
        )}

        <AnchorForecastSection
          health={health}
          prediction={prediction}
          loading={predictLoading}
          errorBanner={predictError}
          selectedAnchor={selectedAnchor}
          onSelectAnchor={handleSelectAnchor}
        />

        <ScoresSection scores={scores} loading={scoresLoading} errorBanner={scoresError} />

        <DataStatusSection health={health} />

        <AboutLimitationsSection health={health} />
      </main>
    </div>
  );
};

export default App;
