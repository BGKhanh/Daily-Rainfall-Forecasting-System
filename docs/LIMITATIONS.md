# Production Operational Limitations & Serving Contract

This document records verified operational constraints, canonical split boundaries, data-latency risks, and runtime boundaries for the HCMC 7-day rainfall forecasting service (`SC-35` serving `XGBRegressor`, `SC-30` shadow validation champion `LGBMRegressor`, `SC-31` shadow post-audit test contender `LGBMRegressor`; all using `tweedie_variance_power = 1.5`).

> **Scope Statement**: Evaluation metrics are computed strictly on completed NASA POWER series (MERRA-2 + CERES SYN1deg); autonomous live real-time (NRT) forecasting is not supported in this release.
>
> **Canonical Dev Train Boundary**: `2001-01-01` to `2018-04-30` (**6,329 days**, **6,293** valid origins; year 2000 excluded). Location is strictly locked to **HCMC** (`10.7769°N, 106.7009°E`); non-HCMC locations receive **HTTP 422**, malformed user `history` payloads receive **HTTP 400**, and server DB observation errors receive **HTTP 503**.

## 1. NASA POWER CERES Radiation Latency (~77-Day Availability Delay)

A live backward probe against the NASA/POWER Daily API (`v2.10.3`, executed `2026-11-15` covering `2026-04-25` to `2026-11-14`) established three distinct feed latencies:

| Variable Group | Variables (API Name / Vietnamese Column) | Last Valid Date (as of 2026-11-15) | Observed Lag vs Query Date |
| :--- | :--- | :--- | :--- |
| Meteorological reference (5) | `PRECTOTCORR`, `T2M`, `RH2M`, `WS2M`, `PS` | `2026-11-12` | 3 days |
| FlashFlux radiation (3) | `ALLSKY_SFC_SW_DWN` (`Bức xạ sóng ngắn bề mặt`), `TOA_SW_DWN` (`Bức xạ đỉnh khí quyển`), `ALLSKY_SFC_LW_DWN` (`Bức xạ sóng dài xuống`) | `2026-11-10` | 5 days |
| **CERES-only radiation (8)** | `CLRSKY_SFC_SW_DWN` (`Bức xạ sóng ngắn trời quang`), `ALLSKY_SFC_SW_DNI` (`Bức xạ trực tiếp pháp tuyến`), `ALLSKY_SFC_SW_DIFF` (`Bức xạ khuếch tán`), `ALLSKY_SFC_PAR_TOT` (`Bức xạ quang hợp tổng`), `CLRSKY_SFC_PAR_TOT` (`Bức xạ quang hợp trời quang`), `ALLSKY_SFC_UVA` (`Bức xạ UVA`), `ALLSKY_SFC_UVB` (`Bức xạ UVB`), `ALLSKY_SFC_UV_INDEX` (`Chỉ số UV`) | **`2026-08-30`** | **77 days** (76 missing days vs `2026-11-14` window end) |

All 8 CERES-only radiation variables contribute `5.382651%` normalized XGBoost gain importance in `SC-35` (compared to `2.449228%` for the 3 FlashFlux columns and `7.831879%` combined across all 11 radiation/UV inputs).

Because NASA POWER returns `-999.0` for unavailable CERES days in real time, **near-real-time (NRT) inference on recent calendar dates cannot obtain all 8 CERES columns directly from NASA POWER without an explicit fallback strategy**.

- **Historical Vintage Availability Citation**: The NASA POWER API (`https://power.larc.nasa.gov/docs/services/api/temporal/daily/`) only serves the latest values available at query time and provides no historical "as-of" or vintage query parameter; consequently, only records captured prospectively by the ingestion pipeline have true NRT revision history (*verified from NASA POWER Daily API v2.10.3 schema/behavior: no as-of vintage parameter exists*).

## 2. Near-Real-Time (NRT) Fallback Policy Status

Offline radiation-ablation experiments (`reports/storage_audit/mt_q1.json`) evaluated five fallback policies when replacing the 8 CERES-only variables (`NATIVE_NAN`, `CLIMATOLOGY_DOY`, `TRAIN_MEAN`, `STALE_LAG_90D`, `STALE_LAG_120D`):

- On Validation, `NATIVE_NAN` was selected by validation policy selection (`SC-35` `val_delta_MAE = -0.004222`, `SC-30` `-0.042340`, `SC-31` `-0.055936`), passing the `< 0.02 mm` gate.
- On Test, however, `NATIVE_NAN` degraded `SC-35` (`test_delta_MAE = +0.022221`, failing the `< 0.02 mm` gate) and increased Test RMSE on all three models (`8.298268` vs `8.098204` baseline on `SC-35`, `8.326911` vs `8.114345` on `SC-30`, `8.341221` vs `8.119362` on `SC-31`). By contrast, `CLIMATOLOGY_DOY` achieved `test_delta_MAE = -0.002190` (`SC-35`), `-0.003051` (`SC-30`), and `-0.002855` (`SC-31`), passing the `< 0.02 mm` gate offline.
- **Operational policy**: No NRT fallback/imputation policy (`D1.1`) has been finalized or deployed in the serving API. Current models are **not verified for autonomous live NRT operation** with missing/stale CERES feeds.
- **FLASHFlux Interior Single-Day Gap (`2026-11-06`)**: Live ingestion auditing recorded a synchronized 1-day NASA POWER dropout on `2026-11-06` across all 3 FLASHFlux radiation variables (`ALLSKY_SFC_SW_DWN`, `TOA_SW_DWN`, `ALLSKY_SFC_LW_DWN`, leaving `204/210` valid days when scanned on `2026-11-16`, `205/210` on `2026-11-17`). DDL V3.2 tags such rows as `FLASHFLUX_INCOMPLETE` and surfaces `per_variable_last_valid_date` plus `incomplete_dates_in_window` in `ingestion_runs.latest_valid_by_group_json`.
- **Two-DB Convention & Storage Runbook**: The container/CI ships `artifacts/models/demo_rainfall_vintage.db` (`bundled_demo_db`, `journal_mode=DELETE` via `VACUUM INTO`, containing retrospective `POST_TEST` evaluations `is_prospective=0`), whereas production forward-test evidence must run on a separate clean `data/vintage/rainfall_vintage.db` (`VINTAGE_DB_PATH`). See [`docs/RUNBOOK_STORAGE.md`](RUNBOOK_STORAGE.md).
- **Point Regression & Dev Train Freeze**: All models emit deterministic point predictions (mean under Tweedie $p=1.5$), not calibrated probabilistic predictive distributions; `FeatureBuilder` imputation statistics and seasonal DOY climatology remain strictly frozen on `Dev Train` (`2001-01-01` to `2018-04-30`, `6,329` days).

## 3. Strict Fail-Fast Serving Validation (Zero Silent Mean Imputation)

Even though `FeatureBuilder` (`artifacts/models/feature_builder_f3.joblib`) carries `dev_train`-fitted `fill_values` for all 35 raw schema keys for historical research preprocessing, `StandaloneRainfallPredictor.predict()` invokes `src/inference/validation.py::validate_serving_input()` **before** `FeatureBuilder.build_single_step()`.

- Every required meteorological, precipitation, and solar/UV radiation column (all 35 validated columns, including all 8 CERES-only columns and 3 FlashFlux columns) must be present, finite (non-`NaN`, non-`Inf`), strictly above the `-900` NASA sentinel threshold, and within `PHYSICAL_METEO_BOUNDS`.
- Missing columns, `NaN`/`Inf` values, `-999.0` sentinels, non-monotonic/gapped dates, or fewer than 35 history days raise `ServingValidationError` and return **HTTP 400** (`INVALID_OR_MISSING_METEO_INPUT` / `STALE_RADIATION_OR_LAG` / `HISTORY_DATE_GAP` / `INSUFFICIENT_HISTORY_WINDOW`) from FastAPI `POST /predict`.
- Silent mean imputation on missing or lagged CERES/meteorological feeds is strictly prohibited in production serving.

## 4. Container Runtime Scope (Pure CPU Inference)

- **The production container (`Dockerfile`, `requirements-serving.txt`)** uses a multi-stage build (Stage 1: `node:24-alpine` compiles `frontend/dist` SPA static assets; Stage 2: `python:3.12-slim` runtime with **no Node.js binary**), installing `xgboost-cpu==3.4.1`, `lightgbm==4.7.0`, `scikit-learn==1.9.1`, `numpy==2.5.2`, `pandas==2.3.3`, `joblib==1.6.0`, `fastapi==0.162.4`, `uvicorn==0.54.0`, `pydantic==2.13.5`, `requests==2.34.2`, and `tzdata==2026.4`.
- **Two-DB Snapshot, Git LFS & Memory Footprint (`VmRSS = 243.13 MB`)**: `artifacts/models/demo_rainfall_vintage.db` (`24.7 MB`) is managed via Git LFS (`.gitattributes` + `scripts/mlops/ensure_demo_db.py` CI/Docker fallback). Runtime Uvicorn memory increased from `210.73 MB` (Round 8) to `243.13 MB` (`+32.40 MB`, well below the `350 MB` limit) due to SQLite page cache warming across `4,270` `v_forecast_scores` rows and `2,000 × 122 × 7` moving-block bootstrap allocations across block lengths `7`, `14`, and `21`.
- **Block-Length Sensitivity Note (`122` Post-Test Anchors)**: Increasing block length to `14` and `21` days leaves only `9` and `6` blocks per bootstrap draw over `122` anchors; boundary effects shift the 95% CI upward and slightly narrow its width (`0.1285 → 0.1059 → 0.0960 mm` for `SC-30 − SC-31`).

## 5. NASA POWER `header.sources` Finality Cutoffs & Spatial Representativeness

- **Single-Day `header.sources` Experiment (`2026-11-19`, NASA POWER Daily API `v2.10.4`)**:
  - Querying 1-day windows every 10 days (`2026-04-25` to `2026-11-18`) + binary search proved that `header.sources` reflects exact per-day provenance transitions:
    - **Precipitation / Meteorology**: **`MERRA2 → GEOSIT`**, with **`2026-09-30`** as the last `MERRA2` date (**50-day lag** vs `2026-11-19`; `GEOSIT` begins `2026-10-01`). Latest published day at experiment time: `2026-11-16`.
    - **Solar Radiation**: **`SYN1DEG → FLASHFLUX`**, with **`2026-08-30`** as the last `SYN1DEG` date (**81-day lag**; `FLASHFLUX` begins `2026-08-31`).
  - These cutoffs are recorded via append-only `audit_probe` rows in `ingestion_runs.latest_valid_by_group_json["source_transition_cutoffs"]` and govern `FINALIZED` (`<= 2026-09-30`) vs `PROVISIONAL` (`>= 2026-10-01`) vs `NOT_PUBLISHED` in `GET /observations`.
- **Spatial Representativeness & Verified Multi-Province Grid Probes**:
  - **MERRA-2 / GEOS-IT `$0.5^\circ \times 0.625^\circ$` (~56 × 68 km)**: Meteorology is a cell mean, not a point station or provincial average; coordinates inside `[10.75°N, 11.25°N] × [106.5625°E, 107.1875°E]` receive identical values. Because the active HCMC cell sits slightly north, southern districts including **Nha Be** and **Can Gio** fall outside the cell.
  - **Solar Radiation `$1.0^\circ \times 1.0^\circ$`**: Verified empirically at `[10.0°N, 11.0°N] × [106.0°E, 107.0°E]` on both `CERES SYN1deg` (`CLRSKY_SFC_SW_DWN`) and `FLASHFlux` (`ALLSKY_SFC_SW_DWN`). *(Discrepancy note: NASA POWER's Solar FAQ cites $\frac{1}{2}^\circ \times \frac{1}{2}^\circ$, whereas Services/Data Sources docs and our API edge probes at `9.99°/10.01°/10.99°/11.01°` and `105.99°/106.01°/106.99°/107.01°` confirm $1^\circ \times 1^\circ$.)*
  - **Formula Confirmed Across 3 Additional Provinces & GEOS-IT**: Verified against recent `GEOS-IT` (`2026-11-10..14`) at HCMC, **Ha Noi** (`[20.75, 21.25] × [105.3125, 105.9375]`), **Da Nang** (`[15.75, 16.25] × [107.8125, 108.4375]`), and **Can Tho** (`[9.75, 10.25] × [105.3125, 105.9375]`), matching `Config.province_coordinates` (`68` keys / `65` unique points for 63 pre-merger provinces + 2 islands + 3 aliases).
  - **Citations (accessed `2026-11-19`)**: [POWER Services](https://power.larc.nasa.gov/docs/services/), [POWER Solar FAQs](https://power.larc.nasa.gov/docs/faqs/solar/), [POWER Methodology & Sources](https://power.larc.nasa.gov/docs/methodology/data/sources/).
- CUDA/NVIDIA packages and `mlflow` are excluded from the runtime container image. `src/api/main.py` and `src/inference/predictor.py` (`StandaloneRainfallPredictor`) run with zero MLflow dependency.
- `src/inference/pyfunc.py` and `src/evaluation/protocols.py` are used only in offline dev/CI parity and benchmark pipelines, not inside the runtime serving container.
