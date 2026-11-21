# Mini-Task Round 13 Summary — `P-c` Solar DOY Serving Integration, `DDL V3.5` Audited Ledger Migration, Turso Layer 3 Quotas & UI Polish

- **Docker Serving Image Tag**: `kgb0630/ds_108:1.7.0-serving`
- **Baseline Commit & CI (`8b2ddc4`)**:
  - `Backend Serving CI` Run ID: [`44893514827`](https://github.com/BGKhanh/Precipitation-EDA/actions/runs/44893514827) — `completed / success`
  - Scheduled Ingest (`3895d97`) Run ID: `44851190614` — `completed / success`

---

## Part A — `P-c` Solar DOY Experiment & Production Serving Integration

### A.1 Offline Pre-Declared 3-Clause Val Gate (`Dev Val n = 724` & `Dev Test n = 2,184`)
Evaluation benchmark source: `research/imputer_benchmark/` (`pc_integration_audit_2184.json` / `research/imputer_benchmark/README.md`).

- **Selected Policy**: `p_c_doy_median_all_11_solar` (`PC_CERES_POLICY_NAME = "P-c_DOY_MEDIAN_11_SOLAR_V1"`), `stop_condition_triggered = false`.
- **Pre-Declared 3-Clause Validation Gate (`n = 724`)**:
  - `max |ΔMAE_val| = 0.005294 mm <= 0.020 mm` (Gate passed).
  - `SC-35`: `ΔMAE = +0.002764`, `ΔRMSE = -0.004915`, `ΔP90 = -0.029554`
  - `SC-31`: `ΔMAE = -0.001976`, `ΔRMSE = -0.000003`, `ΔP90 = -0.011284`
  - `SC-30`: `ΔMAE = +0.005294`, `ΔRMSE = +0.000850`, `ΔP90 = -0.042642`
- **Held-Out Dev Test Parity (`n = 2,184`, `h = 1..7` / `h = 1..3` / `h = 4..7`)**:
  - `SC-35`: clean MAE `3.7333` vs `P-c` MAE `3.7314` (`ΔMAE = -0.0019`, RMSE `8.0940`, P90-Tail `15.4973`; `h1..3` `+0.0013`, `h4..7` `-0.0044`)
  - `SC-31`: clean MAE `3.7246` vs `P-c` MAE `3.7227` (`ΔMAE = -0.0020`, RMSE `8.1159`, P90-Tail `15.6210`; `h1..3` `+0.0015`, `h4..7` `-0.0046`)
  - `SC-30`: clean MAE `3.7501` vs `P-c` MAE `3.7502` (`ΔMAE = +0.0002`, RMSE `8.1059`, P90-Tail `15.5782`; `h1..3` `+0.0017`, `h4..7` `-0.0010`)
- **Prospective True Future Operational Coverage (`Nov 19 / 20 / 21`)**:
  - `P-0`: `0 / 0 / 0`
  - `P-a / P-b`: `0 / 0 / 0`
  - `P-c`: `5 / 4 / 3` (`is_prospective = True`)

### A.2 Production Serving Implementation & Startup Artifact Verification
- **Lookup Artifact**: `artifacts/models/solar_doy_climatology_dev_train.json`
  - Fit exclusively on `Dev Train` (`2001-01-01..2018-04-30`), circular window `r = 7`, explicit integer-string keys `"1".."366"` across all 11 solar/CERES variables.
  - Artifact SHA-256: `5f7aec6a446a7b097b8ac8823e4483e59dd642f983bc0255cc1b2e977e7d326b`.
- **Pure-Runtime Imputation Module**: `src/inference/solar_imputation.py`
  - Exports `load_solar_doy_climatology`, `apply_serving_solar_doy_fallback`, and `PC_CERES_POLICY_NAME = "P-c_DOY_MEDIAN_11_SOLAR_V1"`.
  - Rejects sentinels (`<= -900.0`), non-finite floats, and non-numeric strings; selectively fills **only** `NaN` solar cells in-memory while leaving observed FLASHFlux values untouched and never mutating `observations_vintage`.
- **Model Contract & FastAPI Surface**:
  - Layer 4 `ModelContractConfig` (`src/config/settings.py`) and `artifacts/models/metadata.json` declare `radiation_imputation_policy`, `leap_day_doy_policy`, `solar_doy_table_sha256`, and `solar_doy_climatology_dev_train.json` inside `sha256_checksums` with startup SHA-256 verification.
  - `src/inference/validation.py`, `src/inference/predictor.py`, and `src/data/provider.py` enforce `IncompleteMeteorologyAtAnchorError` (`HTTP 422 INCOMPLETE_METEOROLOGY_AT_ANCHOR`) for missing required meteorology columns.
  - `src/api/schemas.py`, `src/api/main.py` (including `redact_secrets` on `/predict` error handlers), and locked `docs/openapi_snapshot.json` expose the `P-c` response metadata on `/predict` and `/health`.
  - `src/inference/forecast_runner.py` (`issue_forecasts_for_anchors`) computes `input_window_hash` pre-imputation, applies `P-c` in-memory to ML models, records `imputed_values_json = '{}'` and `is_degraded_input = 0` on statistical baselines, and filters `_query_forecast_score_rows` with `obs_first_mm IS NOT NULL AND obs_latest_mm IS NOT NULL` so future unobserved target dates (`>= 2026-11-18`) do not fail `/scores`.
- **Contract Tests**: `tests/test_pc_solar_imputation_contract.py` (3 contract tests: hash reproducibility, 29-day solar prefix invariance, selective fill + leap year `2024-02-29` & `2024-12-31` + strict `METEO_24` fail-fast).

---

## Part B — `DDL V3.5` Audited Ledger Migration & Full Post-Test Issuance

- **Schema Upgrade (`DDL V3.5` & `apply_migrations_v3_5` in `src/data/vintage_store.py`)**:
  - Removed SQL `DEFAULT` on `issued_by` and enforced `CHECK` constraint restricting values to `'automated_pipeline'`, `'manual_retrospective_backfill'`, and `'manual_operational'`.
  - Added `imputed_values_json TEXT NOT NULL DEFAULT '{}'`, `doy_table_sha256 TEXT`, permitted `'P-c_DOY_MEDIAN_11_SOLAR_V1'` in `ceres_policy_applied`, and updated `v_forecast_scores`.
  - Tightened `insert_forecasts_idempotent` (`issued_by` validation), `result_config_sha256` hex validation, and `_baton` guard.
- **Demo Vintage DB Audited Migration (`scripts/mlops/migrate_ledger_v35_audited.py` -> `artifacts/models/demo_rainfall_vintage.db`)**:
  - Proof artifact: `reports/storage_audit/immutable_ledger_migration_proof_v35.json` (and checksum synced into `artifacts/models/metadata.json`).
  - Rows preserved: `4,270` -> `4,270` (`0` NULL `issued_by`).
  - Bit-identical column SHA-256 verification: `29/29` pre-existing columns match bit-for-bit.
  - Immutability & constraint probes: `5/5` `True` (`update_aborted`, `delete_aborted`, `duplicate_insert_aborted`, `null_issued_by_rejected`, `invalid_issued_by_rejected`).
  - Migrated `demo_rainfall_vintage.db` `db_file_sha256`: `603bfb2e3e6c310c9e7f13b9f7934c61aab0c38f1fd9ab6fef63f3251218d0f1`.
- **Production Vintage DB & Live Turso `ds108` Synchronization**:
  - Live Turso schema migrated to `V3.5` (`turso_v35: ('V3.5', '2026-11-21T12:04:31.920548Z')`).
  - Issued **`7,035` `automated_pipeline` rows** across all **`201` meteo-complete `POST_TEST` anchors (`2026-05-01..2026-11-17`)** (`0` manual backfill).
  - Updated `scripts/mlops/automated_pipeline.py`, `scripts/mlops/run_forecast_ledger.py`, and `scripts/mlops/verify_turso_parity_and_costs.py`.
  - 100% parity verified between local Production SQLite and Live Turso `ds108`:
    - `locations`: `1 / 1`
    - `ingestion_runs`: `4 / 4`
    - `observations_vintage`: `9454 / 9454`
    - `forecasts_ledger`: `7035 / 7035` (`forecasts_ledger` content SHA-256 = `b42ac242160b61d4247314310adc5a7b673e96ce5a5d61b797176227a1fadbba`).

---

## Part C — Turso Layer 3 Quota Externalization, Telemetry & Cost Tradeoff

- **Layer 3 Config Externalization (`config/runtime_config.toml` & `src/config/settings.py`)**:
  - Moved Turso `"Free"` plan ceilings (`500M` monthly row reads, `10M` monthly row writes, `3GB` sync bandwidth, `5GB` storage, `0.70` warn ratio) out of Python constants into `ApplicationConfig`.
  - Updated `src/data/turso_backend.py`: removed hardcoded quota constants, persisted `self._token = tok` in `TursoPlatformClient`, wired `turso_usage_source` through `TursoHranaClient.pipeline()` into `/health` (`HealthResponse`), and included `V3.5` columns in `compute_table_content_sha256`.
- **Operational Docs & Telemetry Benchmarks**:
  - Documented the `libsql 0.1.11` Python wheel tradeoff, `~5.9 KB/day` storage footprint for `imputed_values_json` (`json_extract()`), and 5-tuple homogeneous accuracy monitoring key `(eval_split_regime, is_prospective, is_degraded_input, ceres_policy_applied, issued_by)` in `docs/TURSO_COST_AND_OPERATIONS.md`, `docs/RUNBOOK_STORAGE.md`, and `reports/storage_audit/turso_python_client_wheel_logs.json`.

---

## Part D — UI Polish, Official Province-Area Enrichments & Playwright Screenshots

- **Geometry & Area Enrichments (`frontend/public/map/province_grid_overlap.json` & `reports/storage_audit/province_area_official_comparison_65.json`)**:
  - Added `official_area_km2`, `area_rel_diff_pct`, and `exceeds_5pct_area_threshold` across all 65 Vietnam units (`24/63` mainland provinces + `2` island units = `26/65` exceeding `5%`).
  - Documented the Mekong Delta geometric boundary tradeoff (`Can Tho` `+26.83%` at `1,826.91 km²` vs GSO official `1,440.4 km²`; adjacent `Hau Giang` `-21.43%` at `1,274.57 km²` vs `1,622.2 km²`; combined delta `+1.27%` at `3,101.48 km²` vs `3,062.6 km²`).
- **Frontend UI Updates (`frontend/src/`)**:
  - Updated `frontend/src/{types/api.ts,i18n/en.ts,styles.css,components/GridScalePanel.tsx,components/AnchorForecastSection.tsx,__tests__/App.test.tsx}`:
    - Per-cell area labels: `"3,796 km² per cell · 4 cells intersect"` and `"12,146 km² per cell · 3 cells intersect"`.
    - Can Tho amber warning badge: `"Simplified polygon differs from official GSO area by +26.8% (official: 1,440.4 km²) — overlap percentages are approximate"` + `boundaryCoverageCaveat`.
    - Anchor banner: `"Estimated radiation inputs:"` (`P-c` transparent radiation imputation indicator).
- **Playwright Visual Verification (Uncropped Screenshots in `reports/ui_screenshots/`)**:
  - `r13_map_zoom_and_selected_box_1280.png` (`339,455 B` — `6 selected`, `Scroll for 1 more`, anchor `2026-11-17`, `"Estimated radiation inputs"` banner)
  - `r13_map_zoom_and_selected_box_375.png` (`287,637 B` — `6 selected` mobile vertical cards)
  - `r13_grid_scale_hcmc_per_cell_1280.png` (`321,997 B` — HCMC per-cell overlap card)
  - `r13_grid_scale_cantho_area_warning_1280.png` (`326,233 B` — Can Tho `+26.8%` amber polygon warning badge)

---

## Part E — Verification Summary, Docs & GitHub Actions CI

- **Local Verification (Final Green Run After All Edits)**:
  - **Ruff Lint**: `ruff check ...` — `All checks passed!`
  - **Backend Pytest Suite**: `pytest tests/serving/ tests/storage/ tests/test_pc_solar_imputation_contract.py research/imputer_benchmark/test_exogenous_imputer.py -v -rs` — **`130 passed, 0 failed in 201.30s`**
  - **Frontend Tests & Build**: `cd frontend && npm test && npm run build` — **`10 passed (10)`**, Vite `dist/` built cleanly in `656ms`.
- **Documentation & CI Updates**:
  - Updated `README.md`, `docs/LIMITATIONS.md`, `docs/CHANGELOG.md`, `.github/workflows/ci.yml` (`kgb0630/ds_108:1.7.0-serving`, V3.5 migration proof validation, `tests/test_pc_solar_imputation_contract.py`, `2026-11-17` smoke checks), `.agents/skills/serving-and-vintage-ops/SKILL.md`, and `.gitignore`.
- **GitHub Actions CI Status**:
  - **Baseline (`8b2ddc4`)**: `Backend Serving CI` Run [`44893514827`](https://github.com/BGKhanh/Precipitation-EDA/actions/runs/44893514827) — `completed / success`.
  - **Round 13 (`1.7.0-serving`)**: Verified post-push on `origin/main` via GitHub Actions `Backend Serving CI` (`tier1-ci`, `docker-smoke`, `registry-verify`).
