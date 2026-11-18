# Changelog

All notable canonical reconciliation, MLOps tracking, runtime optimization, and operational limitation updates are recorded in this file.

## [1.3.1-serving / Round 9b English Shell & Pre-Merger Map] - 2026-11-18

### Added & Changed
- **API & Schema (`src/api/schemas.py`, `src/api/main.py`, `src/data/vintage_store.py`, `docs/openapi_snapshot.json`)**:
  - Added `GET /locations` (`LocationsResponse`) backed by SQLite `locations` table (falling back to HCMC only in `sample_csv` mode).
  - Removed `run_date` from public `PredictionRequest` (server clock via `src.config.clock.today_hcmc()`), enforced `observed_mm = null` on `today` and `future_forecast` horizons, and converted all user-facing API strings (`future_forecast_notice`, notes, `error.message`, `decision_label`, `test_set_exposure_note`, `config_note`) to English.
  - Extended `/health` with `scan_window_start`, `scan_window_end`, `scan_window_total_days`, `latest_precip_obs_date`, and dynamic plain-English `latest_partial_obs_explanation`, and extended `/scores` with `decision_label` and `block_length_sensitivity` (`7/14/21` days).
- **English UI Shell & Multi-Province Map (`frontend/`)**:
  - Added single i18n resource file `frontend/src/i18n/en.ts`, sidebar/375px drawer, 4 tabs, persistent summary strip, interactive 7-day chart (`0/5/10/15` mm ticks + hover/tap readout), 375px vertical day cards, and error state hiding prior outputs.
  - Bundled offline pre-merger 63-province boundary GeoJSON + `Hoang Sa Island` & `Truong Sa Island` (`frontend/public/map/vietnam_63_provinces.geojson`, `polygon_to_location.json`, `ATTRIBUTION.md`) with `geoBoundaries` (`CC BY 4.0 / Public Domain`) & `Free-GIS-Data` attribution and empirically verified NASA POWER MERRA-2/CERES grid cell bounds (`reports/storage_audit/grid_cell_verification.json`).
  - Converted `README.md` and `docs/LIMITATIONS.md` to English and bumped Docker serving tag to `kgb0630/ds_108:1.3.1-serving`.

## [1.3.0-serving / Node 24 LTS Frontend Builder] - 2026-11-18

### Changed
- **Frontend Builder Toolchain Upgrade (Node 20 EOL → Node 24 LTS)**:
  - Upgraded multi-stage `Dockerfile` Stage 1 (`frontend-builder`) from `node:20-alpine` to `node:24-alpine` (`v24.22.3` Maintenance LTS, `npm 11.19.1`), updated `.github/workflows/ci.yml` (`node-version: "24"`), updated `docs/LIMITATIONS.md`, and added `.nvmrc` (`24.22.3`) while keeping Stage 2 (`python:3.12-slim`) free of Node.js.
  - Bumped serving container image tag to `kgb0630/ds_108:1.3.0-serving` (`.github/workflows/ci.yml`, `.agents/rules/project-structure.md`, `.agents/skills/serving-and-vintage-ops/SKILL.md`), keeping locked API schema/OpenAPI version at `1.2.0`.

## [1.2.2-round8-locked-contract-ui-v1] - 2026-11-18

### Locked API Contract & Added SPA UI v1
- **Hợp đồng API đã khóa (`src/api/schemas.py`, `src/api/main.py`, `docs/openapi_snapshot.json`)**:
  - Added explicit Pydantic v2 request/response models (`HealthResponse`, `ModelsResponse`, `PredictResponse`, `ScoresResponse`, `LeaderboardResponse`) and unified `ErrorResponse` (`{"error": {"code", "message", "details"}}`) covering all `4xx/5xx` responses (including Pydantic `RequestValidationError` 422) across `UNSUPPORTED_LOCATION`, `INCOMPLETE_RADIATION_AT_ANCHOR`, `ANCHOR_DATE_NOT_FOUND`, `INVALID_CLIENT_HISTORY`, `SERVER_DB_DATA_ERROR`, `PREDICTOR_NOT_READY`, and `INVALID_QUERY_PARAMETER`.
  - Extended `/predict` (`PredictResponse`) with `anchor_date`, `anchor_split_regime`, `run_date`, `run_timezone`, `effective_future_days`, nullable `future_forecast_notice`, `db_role`, and 7-day `daily_forecasts` covering all 5 models (`SC-35`, `SC-30`, `SC-31`, `BASELINE_CLIMATOLOGY_DOY`, `BASELINE_PERSISTENCE`).
  - Removed `/models/info` from OpenAPI schema (`include_in_schema=False`, preserving route as backwards-compatible alias for `/models`).
  - Extended `/scores` with all 3 ML pairwise comparisons (`SC-35 - SC-31`, `SC-35 - SC-30`, `SC-30 - SC-31`) plus `n_blocks` & `block_length`, and `/benchmark/leaderboard` with `test_set_exposure_note` and `config_note`.
  - Extended `/health` with `data_snapshot_date`, `anchor_range` (`min`/`max`), and `data_latency` (`METEO_GEOSIT`, `RAD_FLASHFLUX`, `RAD_CERES_SYN1DEG` from latest successful `ingestion_runs`).
  - Exported updated `docs/openapi_snapshot.json` verified against `app.openapi()` and validated all 5 live response schemas + `ErrorResponse`.
- **Canonical Clock (`src/config/clock.py`)**:
  - Unified operational clock around `today_hcmc()` (`ZoneInfo("Asia/Ho_Chi_Minh")`) and mockable `now_utc()` (`tzdata==2026.4` pinned in `requirements-serving.txt`), replacing `pd.Timestamp.today()` and naive `datetime.now()` across `validation.py`, `predictor.py`, `main.py`, `forecast_runner.py`, and `vintage_store.py`. Added 17:30 UTC / 16:59 UTC boundary tests and DB-mode `run_date` tests in `tests/serving/test_clock_and_rundate.py`.
- **CI 2,184-Origin Parity Enforcement & Two-DB Order Reversal**:
  - Whitelisted and tracked `data/processed/results/ml_tweedie_test_forecasts.npz` for CI Tier 1 parity (`CI_REQUIRE_PARITY=1` fails instead of skipping).
  - Rebuilt `data/vintage/rainfall_vintage.db` cleanly with observations only (`9,450` obs `2001-01-01..2026-11-15`, `forecasts_ledger = 0`), cloned `artifacts/models/demo_rainfall_vintage.db` via `VACUUM INTO` (snapshot `2026-11-15`, captured `2026-11-18`), and issued `4,270` `POST_TEST` rows strictly in Demo DB.
  - Added Moving Block Bootstrap sensitivity (`block_length ∈ {7, 14, 21}`) in `reports/storage_audit/post_test_retrospective_scores.json`.
- **Frontend SPA v1 (`frontend/`, Multi-Stage `Dockerfile`)**:
  - Implemented responsive Vietnamese React + Vite + TypeScript dashboard (`Theo ngày neo`, `Bảng điểm`, `Tình trạng dữ liệu`, `Giới thiệu & Giới hạn`) mounted at `/` by FastAPI (`StaticFiles`) inside multi-stage Docker build (zero Node.js in final Python runtime image), plus Vitest UI verification suite.

## [1.2.1-round7-guardrails-openapi-demo-db] - 2026-11-17

### Added & Changed — Entry Guards, UTC Normalization, Block Bootstrap CIs, Two-DB & OpenAPI Lock
- **Canonical Column Aligners & `FeatureBuilder` Entry Guard**:
  - Added shared `align_and_validate_canonical_raw_df()` (`CANONICAL_VI_COLUMN_ORDER`, 36 columns matching `DataLoader`/`Config.COLUMN_MAPPING`) in `src/data/vintage_store.py` (exported in `src/data/__init__.py`) and routed DB reads (`read_latest_observations`, `SQLiteVintageHistoryAdapter`), forecast runner (`_rows_to_vietnamese_df`), and `/predict` serving validation/prediction (`validate_serving_input`, `StandaloneRainfallPredictor.predict`) through it.
  - Added `_validate_canonical_input_columns()` to `src/featurengineering/builder.py` at entry of `fit()`, `transform()` (including `context_df`), and `build_single_step()`, raising `ValueError("[FEATURE_BUILDER_INPUT_SCHEMA_ERROR] ...")` on missing, renamed/misspelled, or swapped columns.
  - Added fixed CI Tier 1 tests in `tests/serving/test_feature_builder_column_guard.py`.
- **UTC ISO-8601 Timestamp Normalization & `julianday()` Scoring**:
  - Added `normalize_utc_iso()` (`YYYY-MM-DDTHH:MM:SS.ffffffZ`) across all DB timestamp writes and `query_observations_as_of()`, and switched `v_forecast_scores.is_prospective` and `query_observations_as_of()` predicates to SQLite `julianday(...)` comparisons.
  - Added opposing-format regression test `test_timestamp_normalization_and_opposing_formats_is_prospective` in `tests/storage/test_vintage_ddl_and_triggers.py`.
- **Post-Test Moving Block Bootstrap CI (`block_length = 7`) & Methodological Notes**:
  - Added `compute_block_bootstrap_ci(..., block_length=7, n_bootstrap=2000, seed=42)` and `METHODOLOGICAL_NOTES` (`effective_sample_size`, `first_vs_latest_equality`, `seasonal_regime_note`) to `src/inference/forecast_runner.py` and persisted full results to `reports/storage_audit/post_test_retrospective_scores.json`.
  - Confirmed statistically significant MAE reductions over same-window `BASELINE_CLIMATOLOGY_DOY` for `SC-31` (`-0.686 mm`, 95% CI `[-1.079, -0.315]`), `SC-30` (`-0.628 mm`, `[-1.030, -0.238]`), and `SC-35` (`-0.616 mm`, `[-1.013, -0.254]`), while `SC-35 - SC-31` (`+0.070 mm`, `[-0.019, +0.164]`) is non-significant.
- **Two-DB Convention & Bundled Demo DB (`artifacts/models/demo_rainfall_vintage.db`)**:
  - Exported `demo_rainfall_vintage.db` via `VACUUM INTO` (`journal_mode = DELETE`, `9,449` observations `2001-01-01..2026-11-14`, `4,270` `POST_TEST` forecasts, SHA-256 `5f7027bb86ea29ef1d15fa169e54b9f72ff33cb07417660c74e292ea5845470a`), recorded SHA-256 in `artifacts/models/metadata.json`, whitelisted in `.gitignore` & `.dockerignore`, and added `resolve_sqlite_db_path_and_role()` + `$VINTAGE_DB_PATH` in `src/data/provider.py`.
  - Added `docs/RUNBOOK_STORAGE.md` and updated `README.md` + `docs/LIMITATIONS.md` (point regression scope, `2018-04-30` Dev Train freeze, Two-DB convention, and `2026-11-06` FLASHFlux interior gap).
- **Locked API Contract (`/health`, `/models`, `/predict`, `/scores`) & OpenAPI Snapshot**:
  - Extended `GET /health` (`db_role`, `default_complete_anchor_date`, `journal_mode`, `baselines`), added `GET /models` (alias `/models/info` exposing 3 ML + 2 Baselines + `demo_rainfall_vintage.db` SHA-256), and added `GET /scores` (`eval_split_regime`, `is_prospective`, overall & `h1..h7` Block Bootstrap CIs, 3 methodological notes, HTTP 422 on invalid params).
  - Exported `docs/openapi_snapshot.json` and added `test_openapi_snapshot_matches_live_app`, `test_scores_endpoint`, `test_models_endpoint` (`78 passed` locally; Docker image `kgb0630/ds_108:1.2.0-serving` verified at `213.73 MB` `VmRSS`).

## [1.2.0-ddl-v3.2 / Step 4 Forecast Runner & Step 5 DB Serving] - 2026-11-16

### Changed — DDL V3.2 Freeze (`src/data/vintage_store.py`)
- Added `rows_unpublished_skipped` to `ingestion_runs` and skipped 35/35 missing days in `ingest_nasa_power_window()` while logging skipped dates in `revision_stats_json["unpublished_dates_skipped"]`.
- Extended `_compute_latest_valid_by_group()` with `per_variable_last_valid_date`, `per_variable_valid_days_in_window`, and `incomplete_dates_in_window` to expose interior single-day radiation gaps (e.g., `2026-11-06` in `RAD_FLASHFLUX`).
- Updated `eval_split_regime` to `('DEV_TRAIN', 'VALIDATION', 'TEST', 'POST_TEST')` with canonical `2026-04-30` Test calendar cutoff (`derive_eval_split_regime()`).
- Updated `v_forecast_scores` so `obs_first_*` uses the first non-null precipitation observation (`MIN(version_num) WHERE PRECTOTCORR IS NOT NULL`) and added `obs_first_ingested_at_utc` + dynamic `is_prospective`.
- Added `'past_observed'` to `temporal_status_at_issue` + helper `derive_temporal_status_at_issue()`.
- Added shared `get_default_complete_anchor_date()`, added `finalize_db_for_read_only()` (`PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode = DELETE;`), and removed unsafe `?mode=ro&immutable=1` fallback from `connect_sqlite(read_only=True)`.
- Enforced canonical DataLoader/FeatureBuilder column order (`CANONICAL_VI_COLUMN_ORDER` / `CANONICAL_EN_COLUMN_ORDER`) across SQLite observation readers.

### Added — Step 4 Forecast Runner & Step 5 DB-Backed API
- Added `src/inference/forecast_runner.py` and CLI `scripts/mlops/run_forecast_ledger.py`:
  - Verified `0.0 mm` max abs diff across all `2,184` canonical Test origins (`v_observations_latest` DB path vs `ml_tweedie_test_forecasts.npz`) for `SC-35`, `SC-30`, and `SC-31`.
  - Issued `4,270` Post-Test retrospective ledger forecasts (`2026-05-01..2026-08-30`, `122` anchors x `5` models x `7` horizons), verified `122/122` `input_window_hash` As-Of matches, and recorded scores in `reports/storage_audit/post_test_retrospective_scores.json`.
- Added `SQLiteVintageHistoryAdapter` & `IncompleteRadiationAtAnchorError` in `src/data/provider.py` and wired `/predict` in `src/api/main.py` with optional `anchor_date` rejection (`422 [INCOMPLETE_RADIATION_AT_ANCHOR]`), DB error handling (`503 [SERVER_DB_DATA_ERROR]`), `"past_observed"` horizon enrichment, and `"future_forecast_notice"` when `effective_future_days == 0`.
- Added DDL V3.2, WAL snapshot, Step 4 runner parity, and Step 5 DB API tests in `tests/storage/` and `tests/serving/` (`72 passed` locally, including fresh-process circular-import checks & `198.75 MB` Linux container RSS smoke test).
- Moved `PHYSICAL_METEO_BOUNDS` import inside `compute_physical_quality_flags()` in `src/data/vintage_store.py` to eliminate Uvicorn container startup circular import (`src.inference.validation` <-> `src.data.vintage_store`), and set `Asia/Ho_Chi_Minh` today as default `effective_run_date` for DB-backed `/predict` requests with omitted `run_date`.

## [1.1.0-serving / Config A Canonical Reconciliation] - 2026-11-16

### Changed — Canonical Config A Lock (`max_depth=-1`) for LightGBM Tweedie (`SC-28..31`)
- **Test benchmark regeneration**: Re-ran `scripts/benchmarks/run_stage3_test_benchmark.py --phase 3 --lightgbm-max-depth -1 --no-mlflow` (`dev_train`-fitted Config A matching Validation) and promoted Config A outputs to canonical paths in `data/processed/results/`.
- **Config B Test archive**: Archived prior Config B (`max_depth=5`) Test results under explicit `_configB` filenames (`ml_tweedie_test_results_configB.csv`, `ml_tweedie_test_horizon_breakdown_configB.csv`, `ml_tweedie_test_forecasts_configB.npz`).
- **Test ranking impact**: With canonical Config A Test metrics, `SC-31` (`lightgbm_tweedie_F3`, `MAE = 3.724641 mm`) moves from Test Rank 3 (under Config B `3.748180 mm`) to **Test Rank 1**, ahead of deployed `@serving` `SC-35` (`xgboost_tweedie_F3`, `MAE = 3.733345 mm`, now **Test Rank 2**) and `SC-30` (`lightgbm_tweedie_F2`, `MAE = 3.750072 mm`, now **Test Rank 3**). `SC-35` remains deployed as `@serving` (Version 1); `SC-31` remains `@shadow_test_contender` (`governance_role = "post_audit_test_contender"`, Version 3); `SC-30` remains `@shadow_val_champ` (Version 2).
- **Variable & CLI cleanup**: Renamed `train_full` -> `dev_train` across `scripts/benchmarks/run_stage3_test_benchmark.py` and `scripts/mlops/package_models.py`, and changed default `--lightgbm-max-depth` in `run_stage3_test_benchmark.py` from `5` to `-1`.
- **Rebuilt downstream reports & packaging metadata**: Regenerated `reports/test_benchmark/*` (`build_test_benchmark_leaderboard.py`, `build_test_provenance.py`), `reports/statistical_uncertainty_rerun.json` (`run_statistical_uncertainty.py`), `artifacts/models/metadata.json`, `README.md`, `tests/serving/test_api.py`, and `tests/integration/test_benchmark_parity.py`.

#### Metric Delta Summary (`SC-28..31` Test Set: Old Config B vs New Canonical Config A)
| Scenario | Old Config B Test MAE / RMSE | New Config A Test MAE / RMSE | Config A Val MAE / RMSE |
| :--- | :--- | :--- | :--- |
| `SC-28` (`F0`) | `3.879174` / `8.135974` | `3.849340` / `8.153258` | `3.237691` / `6.742826` |
| `SC-29` (`F1`) | `3.805444` / `8.072258` | `3.777388` / `8.090554` | `3.158415` / `6.668934` |
| `SC-30` (`F2`) | `3.771696` / `8.080049` | `3.750072` / `8.114345` | `3.148493` / `6.687314` |
| `SC-31` (`F3`) | `3.748180` / `8.092063` | `3.724641` / `8.119362` | `3.193921` / `6.713059` |

#### File & SHA-256 Inventory
| File | Old Size / SHA-256 | New Size / SHA-256 | Reason |
| :--- | :--- | :--- | :--- |
| `data/processed/results/ml_tweedie_test_results.csv` | `1473 B` / `92dea07eeee16bc85f20c68912a600e21e1f4267ec8df12ef0abc68a878e9c03` | `1477 B` / `fa27d6fc9f416e9ef924c6de06bedd7e6b61a60c2dce16715ecc5caab8b707f7` | Promoted Config A (`max_depth=-1`) |
| `data/processed/results/ml_tweedie_test_horizon_breakdown.csv` | `5165 B` / `bce36532c15603955272b31a46f0d63ebe7d5808d38b8faf008977f88bce7272` | `5175 B` / `b42f868aeb70a89091c0677ef61dc2bf5d7901498913a5fb742423240902494d` | Promoted Config A horizons `h1..h7` |
| `data/processed/results/ml_tweedie_test_forecasts.npz` | `710219 B` / `43bb2d34d9ffc363b021dda55af69a4b4ef7cf2f2a4efd32150bdb8cc01660f5` | `711362 B` / `ca588c6b1d22af6979a3c1a737710b0ef323fe88bd450ce4ecd867022d7f5416` | Promoted Config A 2,184x7 forecast matrix |
| `data/processed/results/ml_tweedie_test_results_configB.csv` | New archive | `1473 B` / `92dea07eeee16bc85f20c68912a600e21e1f4267ec8df12ef0abc68a878e9c03` | Archived Config B (`max_depth=5`) |
| `data/processed/results/ml_tweedie_test_horizon_breakdown_configB.csv` | New archive | `5165 B` / `bce36532c15603955272b31a46f0d63ebe7d5808d38b8faf008977f88bce7272` | Archived Config B horizons |
| `data/processed/results/ml_tweedie_test_forecasts_configB.npz` | New archive | `710219 B` / `43bb2d34d9ffc363b021dda55af69a4b4ef7cf2f2a4efd32150bdb8cc01660f5` | Archived Config B NPZ matrix |
| `reports/test_benchmark/unified_72_scenarios_test_leaderboard.csv` | Updated (`SC-31` Rank 1) | `13934 B` / `74547e06c29385933eaa9c613e87b544d410fe0014ead65c0cb2feeeb03d5ab7` | Canonical 72-scenario Test leaderboard |
| `reports/test_benchmark/unified_72_test_horizon_breakdown.csv` | Updated | `48579 B` / `372e980318b4cd44ea2e4802180e49f8b5d08ed034da18d8d85fa7fd6c276f0f` | Canonical Test horizon table |
| `reports/test_benchmark/executive_test_summary.md` | Updated | `10948 B` / `b417ce0c7669ada2400106afc5ac398bf226568780beac54b8fa66fa97097758` | Executive Test summary |
| `reports/test_benchmark/provenance_test.json` | Updated | `4428 B` / `25b7aaf7f0015fe5159e76bd3a77ef46b00520a09293e85c905c3aaa8036860b` | Canonical Test provenance checksums |
| `reports/test_benchmark/yearly_test_stability.csv` | Updated | `2596 B` / `d921c8bfd4973fe1d1a488b94c9af720453c5b22cdcbec67d434bc197f3f972f` | Yearly Test stability breakdown |
| `reports/statistical_uncertainty_rerun.json` | Updated | `11432 B` / `e025a9c9fabf39b721af8191b3ed873b8d67e82b168d5dfa0a3849113a18f4b8` | Bootstrap & HLN Diebold-Mariano stats |
| `artifacts/models/metadata.json` | Updated | `6464 B` / `e104673020e99a9767a8043c92632e1888ff1c52a978176c2a923432e56d5bd4` | Updated `SC-35` rank 2, `SC-30` rank 3, `SC-31` rank 1 & `post_audit_test_contender` |

### Fixed — DagsHub MLflow Tracking & Registry Remediation (`--remediate-audit-v2`)
- Removed hardcoded `CONFIG_A_NUMBERS` dict from `scripts/mlops/backfill_mlflow_tracking.py`; replaced with dynamic canonical loader `_load_canonical_ab_sources()` reading `lightgbm_tweedie_config_ab_canonical.json` and `_configB` CSVs.
- Patched all 9 affected DagsHub runs in-place without deleting any runs (`SC-28..31_ConfigB`, `SC-28..31_ConfigA`, `SC-35_xgboost_tweedie_F3`), adding `audit_remediation_v2="true"`, `audit_remediation_date="2026-11-16"`, `mlflow.note.content`, canonical Val/Test metrics + `h1..h7` steps, and `SC-35` `config_regime="DevTrain_Frozen_Canonical"`.
- Updated `scripts/audit/mt_q2_audit_76_runs.py` (`TOL_CSV_4DP = 5.5e-5` for 4-decimal Validation CSV rounding on `SC-32..35`; Test cross-check switched to canonical Config A + archived Config B). Verified `76/76` DagsHub runs clean (`runs_with_issues_count = 0`).

### Changed — CPU-Only Serving Container Optimization (`kgb0630/ds_108:1.1.0-serving`)
- Replaced `xgboost==3.4.1` with `xgboost-cpu==3.4.1` in `requirements-serving.txt`, added `"xgboost-cpu"` to `CORE_SERVING_PACKAGES` in `src/inference/pyfunc.py` (with matching test update in `tests/serving/test_mlops_pipeline.py`), and bumped image references in `scripts/audit/mt_docker_audit.py` and `.github/workflows/ci.yml` from `1.0.0-serving` to `1.1.0-serving`.
- Reduced Linux Docker image size from `810,913,996 bytes` (`773.35 MB`) to **`532,204,981 bytes`** (**`507.55 MB`**, `-265.80 MB` / `-34.4%`), eliminating `204.69 MB` of NVIDIA/CUDA/NCCL `site-packages` (`site-packages` dropped from `641.57 MB` to `375.77 MB`).
- Verified full Linux parity across Validation (`724` origins) and Test (`2,184` origins) inside `kgb0630/ds_108:1.1.0-serving`: `SC-35` (`2.384e-07` / `4.768e-07`), `SC-30` (`1.776e-15` / `8.882e-16`), `SC-31` (`8.882e-16` / `1.776e-15`) — all `< 1e-5 mm`.

### Added — Operational Limitations, DDL V3.1 Immutable Store, HCMC Lock & CI Tiering
- Implemented **DDL V3.1** in [`src/data/vintage_store.py`](../src/data/vintage_store.py): replaced `'MISSING_ALL_RAD'` with `'FLASHFLUX_INCOMPLETE'` (plus explicit SQL precedence comment), tightened `eval_split_regime` to `('DEV_TRAIN', 'VALIDATION', 'TEST', 'LIVE')` via deterministic `derive_eval_split_regime(data_as_of)`, added idempotent `insert_forecasts_idempotent(conn, forecast_rows)`, added `read_only=True` URI `?mode=ro` -> `?mode=ro&immutable=1` WAL fallback in `connect_sqlite()`, and added `ingest_nasa_power_window()` + CLI [`scripts/data/ingest_nasa_vintage.py`](../scripts/data/ingest_nasa_vintage.py).
- Added [`tests/storage/test_vintage_ddl_and_triggers.py`](../tests/storage/test_vintage_ddl_and_triggers.py) testing all 30 trigger combinations (`recursive_triggers=0/1`), FK deferred COMMIT & rollback audit, Vintage 0 CSV parity (`0.0` numeric diff across `9,251` rows), `FLASHFLUX_INCOMPLETE` / `derive_eval_split_regime`, idempotent forecast batches (`4` inserted / `3` skipped), `read_only` WAL container fallback, As-Of hash reconstruction, and mock NASA POWER revision ingestion.
- **Removed legacy test**: Permanently deleted obsolete `tests/unit/test_feature_complexity_infrastructure.py` (which imported intentionally purged `src.training` module) instead of skipping at collection time.
- Fixed clean-runner import resolution in [`tests/conftest.py`](../tests/conftest.py) (`sys.path`) & [`pytest.ini`](../pytest.ini) (`pythonpath = .`) so GitHub Actions `Backend Serving CI` runs green (`run_id=43840073114`), and enabled the verified Backend CI status badge in [`README.md`](../README.md).
- Locked [`src/api/main.py`](../src/api/main.py) `/predict` strictly to single-source `HCMC_ALIASES` imported from `src.data.vintage_store` (zero duplicate alias constants), added explicit OpenAPI `400`/`422`/`503` response documentation, and updated [`tests/serving/test_api.py`](../tests/serving/test_api.py).
- Added [`pytest.ini`](../pytest.ini), updated [`.gitignore`](../.gitignore) (`!tests/storage/**`, `!scripts/data/ingest_nasa_vintage.py`, ignore `data/vintage/**` & WAL files), and separated `.github/workflows/ci.yml` into Tier 1 (`tier1-ci`) and Tier 2 (`docker-smoke`).
- Corrected `Dev Train` period (`2001-01-01` to `2018-04-30`, `6,329` days / `6,293` origins), `tweedie_variance_power = 1.5`, `LGBMRegressor`/`XGBRegressor` labels, verified NASA POWER non-versioned API citation, and finalized-sequence / no-live-NRT disclaimer in [`README.md`](../README.md) and [`docs/LIMITATIONS.md`](LIMITATIONS.md).
- Added `tests/test_mlflow_registry_consistency.py` enforcing `backfill_mlflow_tracking.py`, canonical JSON/CSVs, horizon breakdowns, and `artifacts/models/metadata.json` consistency.
- Appended parameterized 8-column CERES strict validation & FastAPI HTTP 400 rejection tests to [`tests/serving/test_serving_hardening.py`](../tests/serving/test_serving_hardening.py).
