# Ho Chi Minh City Daily Rainfall Forecasting System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Backend CI](https://github.com/BGKhanh/Precipitation-EDA/actions/workflows/ci.yml/badge.svg)](https://github.com/BGKhanh/Precipitation-EDA/actions/workflows/ci.yml)
[![Models](https://img.shields.io/badge/models-XGBoost%20%7C%20LightGBM%20%7C%20Hurdle%20%7C%20Neural-green.svg)]()
[![Forecast Horizon](https://img.shields.io/badge/forecast%20horizon-1--7%20days-orange.svg)]()
[![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

An end-to-end machine learning application for multi-day daily rainfall forecasting ($H = 1 \dots 7$ days) in Ho Chi Minh City, Vietnam. The system processes meteorological data from NASA POWER (2001–2026) and provides daily rainfall amount predictions (mm/day) along with rain/no-rain event detection.

---

## 🌟 Highlights & Key Capabilities

- 🌧️ **Multi-Day Forecasting (1–7 Days Ahead)**: Delivers daily precipitation forecasts for the upcoming week from any given forecast date.
- 🎯 **High Prediction Accuracy**:
  - **82.5% accuracy** on rainy vs. dry day classification (WMO $\ge 1.0\text{ mm}$ standard).
  - **95.2% Probability of Detection (POD / Recall)** on actual rainy days.
  - **3.72–3.73 mm** Mean Absolute Error (MAE) on 7-day continuous rainfall amounts (`LightGBM Tweedie` Test #1 at `3.7246 mm`; deployed `@serving` `XGBoost Tweedie` at `3.7333 mm`).
- ⚡ **Fast & Lightweight Inference**: Less than **0.05 seconds** per 7-day forecast cycle, ideal for operational alerts and scheduled pipelines.
- 🛡️ **Zero-Inflation & Extreme Weather Handling**: Specifically tuned for tropical monsoon climates where 50–60% of days have zero rainfall, while rainy days can feature sudden torrential convective downpours (>50–100 mm/day).
- 🧩 **Modular & Extensible Architecture**: Reusable modules for automated data collection, leak-proof feature engineering, model training, and performance auditing.

---

## 📊 Performance at a Glance

The table below summarizes performance across different model architectures on the out-of-sample Test dataset (2020–2026, 2,184 forecast origins, 15,288 forecast instances):

| Model Family | Representative Model | Feature Set | MAE (mm) | RMSE (mm) | Rain Detection Accuracy | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Tweedie Regressors (Test #1)** | `LightGBM Tweedie` (`SC-31`, Config A) | $F_3$ (Lags + Rolling + Season) | **3.72** | 8.12 | 82.3% | ~0.03s |
| 🥈 **Tweedie Regressors (Deployed `@serving`)** | `XGBoost Tweedie` (`SC-35`) | $F_3$ (Lags + Rolling + Season) | **3.73** | **8.10** | **82.5%** | **~0.03s** |
| 🥉 **Direct ML Regressors** | `Random Forest Direct` | $F_3$ (Lags + Rolling + Season) | 3.94 | 8.11 | 80.1% | ~0.03s |
| **Two-Stage Hurdle (Mixture)** | `Logistic + Linear Regressor` | $F_3$ (Lags + Rolling + Season) | 3.97 | 8.02 | 81.2% | ~0.02s |
| **Deep Sequence Models** | `PatchTST (Transformer)` | Rainfall Target | 4.03 | 8.66 | 77.4% | ~1.76s |
| **Statistical Time Series** | `AutoARIMA` | Rainfall Target | 4.33 | 8.20 | 73.1% | ~49.8s |
| **Baseline Persistence** | `Naive (Yesterday's Rain)` | Rainfall Target | 5.14 | 10.76 | 68.2% | <0.01s |

### Forecast Error by Lead Day (Deployed `@serving`: `SC-35` XGBoost Tweedie on Test)

Rainfall predictability naturally decreases with longer lead times:

| Lead Day | Day 1 ($t+1$) | Day 2 ($t+2$) | Day 3 ($t+3$) | Day 4 ($t+4$) | Day 5 ($t+5$) | Day 6 ($t+6$) | Day 7 ($t+7$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MAE (mm)** | **3.50** | **3.77** | **3.76** | **3.78** | **3.76** | **3.80** | **3.78** |
| **RMSE (mm)** | 7.67 | 8.12 | 8.18 | 8.18 | 8.16 | 8.18 | 8.18 |

---

## 🛠️ Data & Feature Engineering

### Geographic Coordinates, Spatial Representativeness & Source Cutoffs
- **Canonical Location Lock (`HCMC`)**: The serving API (`/predict`) is **strictly locked to Ho Chi Minh City** (`location_id = "HCMC"`, urban center **10.7769°N, 106.7009°E** - District 1). While `src/config/constants.py::Config.province_coordinates` holds **68 keys / 65 unique coordinates** (63 pre-merger provinces + 2 archipelagos + 3 alias keys `'Huế'`, `'Quần đảo Hoàng Sa'`, `'Quần đảo Trường Sa'`), multi-province forecasting is disabled and non-HCMC requests to `/predict` return **HTTP 422**.
- **Spatial Representativeness (NASA POWER Grids)**:
  - Meteorology variables (`PRECTOTCORR`, `T2M`, etc.) originate from a gridded **$0.5^\circ \times 0.625^\circ$ (~56 × 68 km)** MERRA-2 / GEOS-IT cell, not a point rain gauge or provincial average; District 1 `(10.7769°N, 106.7009°E)` and Cu Chi `(10.99306°N, 106.63775°E)` receive identical series (`0.000000 mm` difference). The active HCMC cell `[10.75°N, 11.25°N] × [106.5625°E, 107.1875°E]` is shifted north relative to southern districts, excluding **Nha Be** and **Can Gio**.
  - Solar radiation variables originate from a coarser **$1.0^\circ \times 1.0^\circ$** grid (`[10.0°N, 11.0°N] × [106.0°E, 107.0°E]`), empirically verified across both `CERES SYN1deg` (`CLRSKY_SFC_SW_DWN`) and `FLASHFlux` (`ALLSKY_SFC_SW_DWN`) as well as MERRA-2/GEOS-IT (`2026-11-10..14`) and 3 additional provinces (`Ha Noi`, `Da Nang`, `Can Tho`) in `reports/storage_audit/grid_cell_verification.json`. Note that NASA POWER's Solar FAQ mentions $\frac{1}{2}^\circ \times \frac{1}{2}^\circ$, whereas Services/Data Sources docs and our API boundary probes confirm $1^\circ \times 1^\circ$.
  - **NASA POWER Citations (accessed `2026-11-19`)**: [POWER Services](https://power.larc.nasa.gov/docs/services/) · [POWER Solar FAQs](https://power.larc.nasa.gov/docs/faqs/solar/) · [POWER Methodology & Sources](https://power.larc.nasa.gov/docs/methodology/data/sources/).
- **Verified `header.sources` Finality Cutoffs (`v2.10.4` probe on `2026-11-19`)**:
  - Single-day queries confirm per-day source transitions: **`MERRA2 -> GEOSIT`** for precipitation/meteorology with **`2026-09-30`** as the last `MERRA2` date (**50-day lag**; `GEOSIT` starts `2026-10-01`), and **`SYN1DEG -> FLASHFLUX`** for solar radiation with **`2026-08-30`** as the last `SYN1DEG` date (**81-day lag**; `FLASHFLUX` starts `2026-08-31`). These cutoffs drive `FINALIZED` vs `PROVISIONAL` labeling in `GET /observations`.
- **Canonical Time Splits (Year 2000 Excluded)**:
  - **Dev Train**: `2001-01-01` to `2018-04-30` (**6,329 days**, **6,293** valid origins; year 2000 excluded due to 1,830 structural missing cells on 5 radiation/UV variables). FeatureBuilder statistics and baseline DOY climatology are frozen strictly as of **`2018-04-30`**.
  - **Validation**: `2018-05-01` to `2020-04-30` (**731 days**, **724** valid origins).
  - **Out-of-sample Test**: `2020-05-01` to `2026-04-30` (**2,191 days**, **2,184** valid origins).
  - **Post-Test Retrospective Window**: `>= 2026-05-01` (evaluated across **122** `COMPLETE_ALL_35` anchors `2026-05-01..2026-08-30` with `block_length = 7` Moving Block Bootstrap CIs).
- **Point Regression Models (`tweedie_variance_power = 1.5`)**:
  - `@serving` **`SC-35`**: `XGBRegressor` (`objective="reg:tweedie"`, `tweedie_variance_power=1.5`, `max_depth=5`, `n_estimators=100`, `learning_rate=0.03`).
  - `@shadow_val_champ` **`SC-30`**: `LGBMRegressor` (`objective="tweedie"`, `tweedie_variance_power=1.5`, Config A `max_depth=-1`, `num_leaves=31`, `n_estimators=100`, `learning_rate=0.03`).
  - `@shadow_test_contender` **`SC-31`**: `LGBMRegressor` (`objective="tweedie"`, `tweedie_variance_power=1.5`, Config A `max_depth=-1`, `num_leaves=31`, `n_estimators=100`, `learning_rate=0.03`).
- **Two-DB Convention & API Contract (`/`, `/health`, `/locations`, `/observations`, `/models`, `/predict`, `/scores`, `/benchmark/leaderboard`)**:
  - **Bundled Demo DB** (`artifacts/models/demo_rainfall_vintage.db`, tracked via Git LFS `.gitattributes` with automated rebuild fallback `scripts/mlops/ensure_demo_db.py`): Baked into `kgb0630/ds_108:1.4.0-serving` (`journal_mode=DELETE`) for reproducible CI, 2,184-origin Test parity, `/observations`, and UI showcase.
  - **Production / Live DB** (`data/vintage/rainfall_vintage.db` or `$VINTAGE_DB_PATH`): Kept outside the container cleanly with **observations only (`forecasts_ledger = 0`)**. See [`docs/RUNBOOK_STORAGE.md`](docs/RUNBOOK_STORAGE.md) and [`docs/openapi_snapshot.json`](docs/openapi_snapshot.json).
- **Web Dashboard UI (`frontend/`, Vite + React + TypeScript)**:
  - Multi-stage Docker builds the SPA static bundle (`frontend/dist`) and FastAPI mounts it at `/` (zero Node.js in Stage 2). Features a single-model (`@serving`) line-chart forecast view, an independent dual-layer `Province boundaries` + `Grid cells` Vietnam map (65 pre-merger units), a real `Data retrieval` archive explorer (`GET /observations`), and `System limitations` with full `Spatial representativeness` citations.
  - **Cloud Deployment Note**: Cloud hosting deployment (e.g., AWS) is deferred; no cloud hosting latency or platform sizing claims are published until benchmarked live.
- **Evaluation Scope & Live NRT Disclaimer**:
  - Evaluation metrics are computed strictly on completed NASA POWER MERRA-2 and CERES SYN1deg observations; autonomous real-time (live NRT) forecasting is not supported in this release due to ~78-day CERES latency and observed interior radiation dropouts such as `2026-11-06` in FLASHFlux.

### Dynamic Operational Lag & Forecast Horizons
- **Data Latency**: NASA POWER daily meteorological updates typically feature a **3 to 4-day publication latency** from real-time.
- **Effective Future Days ($7 - \text{lag}$)**:
  - When lag = 3 days: Horizons $h=1, 2$ represent past unverified hindcasts, $h=3$ represents the current date, and $h=4, 5, 6, 7$ provide **4 true future forecast days**.
  - When lag = 4 days: Horizons $h=1, 2, 3$ represent past unverified hindcasts, $h=4$ represents the current date, and $h=5, 6, 7$ provide **3 true future forecast days**.
- Every forecast response dynamically tags each lead day with its verified operational status (`past_unverified`, `today`, or `future_forecast`).

### Feature Hierarchy
- **Raw Weather Variables (36 features, $F_0$)**: Temperature (min/max/dew point), relative & specific humidity, surface pressure, solar insolation, wind speed ($U/V$ vectors at 2m, 10m, 50m), and multi-layer soil wetness.
- **Historical Lags ($F_1$, 53 features)**: 1, 2, 3, 7, 14, and 30-day historical values of rainfall, humidity, and dew point.
- **Rolling Statistics ($F_2$, 65 features)**: 3, 7, 14, and 30-day backward moving averages and standard deviations.
- **Calendar & Monsoon Signals ($F_3$, 70 features)**: Solar harmonic components ($\sin/\cos$ day of year), wet season indicators (May–November), and monsoon transition phases.

All feature transformations strictly use past observations ($t \le 0$) to ensure zero lookahead in operational forecasts.

- **Operational Limitations & Container Scope**: See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for the ~77-day CERES solar radiation publication lag, strict fail-fast input validation rules (no silent mean imputation on missing CERES/meteorological/target fields), and Docker serving container runtime scope (`StandaloneRainfallPredictor` only; `pyfunc.py` and `protocols.py` are dev/CI only).

---

## 💡 Quick Start & Usage

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/DS108_project.git
cd DS108_project

# Set up Conda environment (Python 3.12)
conda create -n ts_forecasting python=3.12 -y
conda activate ts_forecasting

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Verification Tests

```bash
# Run unit and integration tests
pytest tests/ -v
```

### 3. Making a 7-Day Forecast (Python Example)

```python
import pandas as pd
from src.data.loader import load_clean_data
from src.featurengineering.pipeline import build_feature_set
from src.models.tree_models import XGBoostTweedieModel

# 1. Load historical weather data
df = load_clean_data("data/raw/hcmc_weather_daily.csv")

# 2. Build F3 features (lags, rolling averages, season indicators)
X, y = build_feature_set(df, level="F3")

# 3. Initialize and train model (or load saved weights)
model = XGBoostTweedieModel(n_estimators=300, max_depth=6)
model.fit(X, y)

# 4. Generate 7-day multi-horizon forecast from latest date
latest_features = X.iloc[[-1]]
seven_day_forecast = model.predict(latest_features)

print("Predicted rainfall for next 7 days (mm):")
for day, rainfall in enumerate(seven_day_forecast[0], start=1):
    print(f"  Day +{day}: {rainfall:.2f} mm")
```

---

## 🏗️ Project Structure

```text
DS108_project/
├── data/
│   ├── raw/                           # Raw NASA POWER daily meteorological exports (.csv)
│   └── processed/                     # Quality-audited & canonical split datasets (.parquet)
├── artifacts/
│   ├── models/                        # Serialized weights (*.joblib, *.json) & metadata.json
│   │   ├── feature_builder_f2.joblib
│   │   ├── feature_builder_f3.joblib
│   │   ├── sc35_xgboost_tweedie_f3.joblib # Serving champion
│   │   ├── sc30_lightgbm_tweedie_f2.joblib # Shadow validation champion
│   │   ├── sc31_lightgbm_tweedie_f3.joblib # Shadow test contender
│   │   └── metadata.json              # Frozen hyperparameters & cryptographic SHA-256 hashes
│   └── feature_names.json             # Audited canonical feature lists (F0..F3, 70 features)
├── src/
│   ├── api/                           # FastAPI Serving Application (main.py)
│   ├── inference/                     # Standalone Serving Predictor & Pyfunc Wrapper
│   │   ├── predictor.py               # StandaloneRainfallPredictor (dynamic lag & horizons)
│   │   └── pyfunc.py                  # MLflow custom pyfunc wrapper (DagsHub Cloud ready)
│   ├── models/                        # Model definitions & champion factories
│   │   ├── tweedie.py                 # XGBoost & LightGBM Tweedie regressors
│   │   ├── direct.py                  # Multi-output direct regressors (RF, XGB, LGBM)
│   │   ├── hurdle.py                  # Two-Stage Hurdle forecasters
│   │   └── nixtla/                    # Adapters for StatsForecast & NeuralForecast
│   ├── feature_engineering/           # Feature engineering pipelines (FeatureBuilder F0..F3)
│   ├── data/                          # Data crawler, validation audits, canonical loader
│   ├── evaluation/                    # Metrics (MAE, RMSE, MASE) & benchmark protocols
│   ├── analysis/                      # Statistical diagnostics, ranking, Pareto frontiers
│   ├── config/                        # Constants, Vietnam coordinates & DagsHub MLOps config (mlops.py)
│   └── utils/                         # Reproducibility seeds and helpers
├── scripts/
│   ├── benchmarks/                    # 72 canonical scenario runners (Phase 1-5 & Test Stage 3)
│   │   ├── run_phase1_statistical.py
│   │   ├── run_phase2_ml_direct.py
│   │   ├── run_phase3_ml_tweedie.py
│   │   ├── run_phase4_neural.py
│   │   ├── run_phase5_ml_hurdle.py
│   │   └── run_stage3_test_benchmark.py
│   ├── experiments/                   # Statistical tests & ablation experiments
│   │   ├── compare_rainy_season_grid.py # 21-day rainy season grid equivalence test
│   │   ├── compare_sc30_configs.py     # Config A vs Config B audit
│   │   └── run_radiation_ablation.py   # Radiation latency impact study
│   ├── mlops/                         # Cloud MLflow sync, packaging & governance tools
│   │   ├── backfill_mlflow_tracking.py # Backfill 76 runs & register models to DagsHub Cloud
│   │   ├── verify_mlflow_registry.py   # End-to-end verification from DagsHub Model Registry
│   │   ├── promote_champion_model.py   # Cloud governance CLI (--status / --promote-scenario)
│   │   └── package_models.py           # Joblib export & SHA-256 generation
│   └── tools/                         # Automated notebook sync & reporting tools
├── notebooks/                         # 30+ reproducible research and diagnostic notebooks
│   ├── data_quality/                  # 00.1..00.4 data audit & continuity notebooks
│   ├── eda/                           # 01.1..01.7 exploratory data analysis notebooks
│   ├── feature_engineering/           # 02 feature engineering pipeline
│   ├── modeling/                      # 03.0..03.7 benchmark modeling notebooks
│   └── result_analysis/               # 00..09 comparative analysis & Pareto dashboards
├── reports/                           # Official leaderboard CSVs, metrics, and markdown briefs
├── tests/                             # Layered automated test suite (pytest)
│   ├── unit/                          # Isolated unit tests for features & metrics
│   ├── contracts/                     # Formal contract & leakage prevention tests
│   ├── integration/                   # Parity tests asserting bit-for-bit benchmark match
│   ├── serving/                       # FastAPI endpoints & standalone inference tests
│   └── analysis/                      # Diagnostic and ranking tests
├── Dockerfile                         # Production serving container definition
├── docker-compose.yml                 # Orchestration definition for serving engine
├── requirements-serving.txt           # Minimal runtime dependencies for pure inference
├── requirements-dev.txt               # Development, MLOps, and testing dependencies
└── README.md

---

## 🏃 Reproducing Model Training & Reports

To run the complete benchmark training and evaluation pipeline:

```bash
# 1. Run model family pipelines
python scripts/benchmarks/run_phase1_statistical.py   # Statistical baselines
python scripts/benchmarks/run_phase2_ml_direct.py     # Direct ML regressors
python scripts/benchmarks/run_phase3_ml_tweedie.py    # Tweedie gradient boosters
python scripts/benchmarks/run_phase4_neural.py        # Deep learning models
python scripts/benchmarks/run_phase5_ml_hurdle.py     # Two-Stage Hurdle forecasters

# 2. Run multi-horizon test evaluation across all 72 scenarios
python scripts/benchmarks/run_stage3_test_benchmark.py

# 3. MLOps: Sync 76 benchmark runs & register models to DagsHub Hosted MLflow Server
# (Ensure .env is configured from .env.example with your DagsHub access token)
python scripts/mlops/backfill_mlflow_tracking.py
python scripts/mlops/verify_mlflow_registry.py
python scripts/mlops/promote_champion_model.py --status
```

All summary tables and performance metrics will be exported to `reports/test_benchmark/`.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
