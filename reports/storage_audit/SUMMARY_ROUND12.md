# Round 12 Executive Summary — CERES-8 Imputer Audit, Turso Resilience, DDL V3.4 & UI Polish (`1.6.0-serving`)

**Date:** `2026-11-21` · **Docker Image Tag:** `kgb0630/ds_108:1.6.0-serving` · **Schema Version:** `DDL V3.4`

---

## 1. Part A — Standalone Exogenous Imputer Research & CERES-8 vs. SYN1deg-11 Audit (`research/imputer_benchmark/`)

- **Isolation Boundary**: Relocated the experimental exogenous imputation module, 4-tier causal baselines, 48-anchor/72-config benchmark, and executed notebook (`imputation_based_inference.ipynb`) under git-tracked `research/imputer_benchmark/` (excluded from `.github/workflows/ci.yml` lint & serving test runs).
- **Full 724 Validation & 2,184 Test Audit (`run_ceres8_vs_11_audit.py` -> `ceres8_scientific_audit_2184.json`, `README.md`)**:
  - **Feature Scope Verification**: Verified `radiation_derived_lag_or_rolling_cols == []` across all 11 solar radiation variables; every radiation feature enters `SC-35` / `SC-31` / `SC-30` strictly at origin day (`F0`).
  - **Intrinsic `vr_scen` vs `vr_mean` (`L=80`, 24 Test anchors)**: `custom_mode2_k10` preserves scenario-level variance (`A-8`: `vr_scen = 1.0223`, `vr_mean = 0.3467` vs `doy_climatology` `0.2002`; `A-11`: `vr_scen = 1.0326`, `vr_mean = 0.4262` vs `doy_climatology` `0.2640`).
  - **Extrinsic Profile `A-8` (`8 CERES-only` outage, Test `n=2,184`, canonical `MAE` / `MSE` / `RMSE` / `P90-Tail MAE` with `assert_no_purged_metrics` enforced)**:
    - `SC-35`: `clean` `3.7333` (`RMSE 8.0982`, `P90-Tail 15.5139`), `raw_nan` `3.7565` (`+0.0232`, `RMSE 8.3075`), `doy_climatology` `3.7338` (`RMSE 8.0943`), `custom_mode2_k10` `3.7314` (`RMSE 8.1002`), `retrained_no_ceres_27vars` `3.7312` (`RMSE 8.0912`).
    - `SC-31`: `clean` `3.7246` (`RMSE 8.1194`), `raw_nan` `3.7379` (`RMSE 8.3499`), `doy_climatology` `3.7247` (`RMSE 8.1167`), `custom_mode2_k10` `3.7224` (`RMSE 8.1184`), `retrained_no_ceres_27vars` `3.7271`.
    - `SC-30`: `clean` `3.7501` (`RMSE 8.1143`), `raw_nan` `3.7405` (`RMSE 8.3365`), `doy_climatology` `3.7509` (`RMSE 8.1058`), `custom_mode2_k10` `3.7469` (`RMSE 8.1133`), `retrained_no_ceres_27vars` `3.7585`.
  - **Profile `A-11` (`11 SYN1deg` outage) `raw_nan` Degradation**: `SC-35` `+0.0625` mm, `SC-31` `+0.1768` mm, `SC-30` `+0.3485` mm (`h1..h7` breakdown saved in `ceres8_scientific_audit_2184.json`).
  - **`test_exogenous_imputer.py` + `test_imputation_benchmark.py`**: `10 passed in 160.96s`.

---

## 2. Part B — Turso Resilience, Atomic Replica Sync, Secret Redaction & Wheel Probes

- **Quota/BLOCKED Local Fallback (`src/data/turso_backend.py`, `src/api/main.py`, `scripts/mlops/automated_pipeline.py`)**:
  - `TursoQuotaBlockedError` on HTTP `402`/`403`/`429` or `BLOCKED`/quota responses triggers clean local SQLite fallback in `open_store_connection()`, surfaces `status="degraded"` + `degraded_reason="TURSO_REMOTE_BLOCKED_FALLBACK_LOCAL"` on `GET /health`, and emits `status="DEFERRED_QUOTA_BLOCKED"` (`exit 0`) in `automated_pipeline.py`. Automatic recovery via `clear_turso_blocked_state()` clears blocked flags upon subsequent successful Hrana pipeline calls or replica syncs while retaining `>= 70%` (`500M` monthly reads / `10M` writes / `3GB` syncs) `turso_quota_warning` telemetry.
- **Atomic Replica Staging & Health Telemetry (`TursoReplicaManager.sync_now()`, `src/api/schemas.py`, `docs/openapi_snapshot.json`)**:
  - Stages updates in `{replica_path.name}.staging.{pid}.db`, runs `PRAGMA integrity_check == "ok"`, checks table counts & `compute_table_content_sha256()`, calls `finalize_db_for_read_only()`, swaps atomically with `os.replace()`, closes `remote_client` sessions in `finally`, and flags `REPLICA_SYNC_CONSECUTIVE_FAILURES` after `>= 3` consecutive failures.
  - Exposes `replica_sync_count`, `replica_last_sync_at_utc`, `replica_last_sync_duration_ms`, `replica_sync_lag_sec`, `replica_hash_verified`, `replica_consecutive_failures`, and `replica_last_sync_error` on `HealthResponse`.
- **Secret Redaction & Cost S5 (`docs/TURSO_COST_AND_OPERATIONS.md`, `reports/storage_audit/turso_parity_and_cost_report.json`)**:
  - `redact_secrets()` scrubs `TURSO_*`, `DAGSHUB_*`, `MLFLOW_TRACKING_PASSWORD`, Bearer/JWT tokens, URL `user:pass@`, and query tokens (`authToken`, `token`, `api_key`, `key`, `secret`) across errors, logs, and API `503` details.
  - Clarified Turso `"Syncs"` bandwidth (`3 GB/mo`) vs replica sync billing in `rows_read`, added Scenario `S5` (peak NRT-to-Final revision wave with `5x` B-tree index multiplier), and persisted Windows/PyPI wheel probes + Docker daemon status log in `reports/storage_audit/turso_python_client_wheel_logs.json`.

---

## 3. Part C — Dual Config Hashes, DDL `V3.4`, Option A & Subset Completeness

- **Dual Config Hashes & `DDL V3.4` (`src/config/settings.py`, `src/data/vintage_store.py`)**:
  - Added `compute_result_config_sha256()` (Layer 4+5, stamped in `forecasts_ledger.result_config_sha256` alongside `issued_by`) and `compute_operational_config_sha256()` (Layer 2+3, stamped in `ingestion_runs.operational_config_sha256`).
  - Migrated `artifacts/models/demo_rainfall_vintage.db` (`4,270` rows tagged `issued_by='manual_retrospective_backfill'`, SHA-256 `adb85f50290ebe8d...` updated in `artifacts/models/metadata.json`), local `data/vintage/rainfall_vintage.db`, and live Turso `ds108` (`no_manual_retrospective_backfill_in_prod = True` while permitting `automated_pipeline` rows).
- **Subset `< 35` Contract & Layer 5 Parity (`tests/serving/test_config_contract.py`)**:
  - Updated `get_default_complete_anchor_date`, `get_complete_anchor_range`, and `list_complete_anchor_dates` to evaluate `required_vars` intersected with `missing_vars_json` (27-var subset advances cutoff to `2026-11-12` in `test_anchor_completeness_with_subset_model_contract_fewer_than_35_vars`), verified `training_eda_freeze_v1.toml` key-for-key against `metadata.json["frozen_eda_modeling_config"]`, and documented Layer 5 as a historical audit record (`docs/CONFIG_ARCHITECTURE.md`).

---

## 4. Part D — 65-Unit Area Audit & Priority 1–4 / Usability 5–11 UI Polish

- **65-Unit Spherical Area Audit (`scripts/data/audit_province_areas_65.py` -> `reports/storage_audit/province_area_official_comparison_65.json`)**:
  - `0.00 km²` max difference between recomputed spherical areas (`vietnam_63_provinces.geojson`) and cached `province_grid_overlap.json`.
  - Documented Can Tho (`1,826.91 km²` vs `1,440.40 km²` official GSO, `+26.83%`) alongside neighbor Hau Giang (`1,274.57 km²` vs `1,622.20 km²`, `-21.43%`): combined polygon area (`3,101.48 km²`) matches official combined (`3,062.60 km²`) within `+1.27%`, proving geoBoundaries simplified ADM1 generalized the 2004 split border southward.
- **UI & API Polish (`frontend/src/`, `src/data/vintage_store.py`)**:
  - **Priority 1**: Weekly (`calendar_days_count = 7`) and Monthly (`_days_in_month`) leading/trailing incomplete periods render as hollow circles (`fill="#ffffff"`, `strokeWidth={2.5}`) connected by `stroke-dasharray="2 4"` dotted connectors with legend entry & tooltips (`DataStatusSection.tsx`, `tests/serving/test_api.py`).
  - **Priority 2**: Multi-province selection keeps focus on `Ho Chi Minh City` by default, adds `forecast-location-tabs`, and merges duplicate unsupported headings into `"{DisplayName} — Forecast not available yet"`.
  - **Priority 3–4 & Usability 5–11**: Exact quantitative spatial representativeness (`29.11%` / `16.12%` / `~3.0 km` / `90.30%`), dynamic `/health` snapshot mode card, `"Polygon area (from simplified boundary data)"`, full-width overlap table below SVG with directional labels (`M-NW`, `C-SW`, etc.) & swatches without internal scrollbars, `km` scale bar & `N ↑` North arrow, representative CERES + MERRA-2 `"input feed"` highlights, `"1×"` / `"Fit to selection"` map buttons (removed outer `"Reset"`), scroll-more badge when `> 5` selected rows, `ISO: YYYY-MM-DD` date hints, collapsed 35-var `<details>`, and mobile chart horizontal scroll (`min-width: 540px`, fixed `.mobile-obs-cards` ordering in `styles.css`).

---

## 5. Verification Summary

- `ruff check` across all CI paths, `scripts/data/audit_province_areas_65.py`, and `research/imputer_benchmark/`: **All checks passed**.
- `CI_REQUIRE_PARITY=1 pytest tests/serving/ tests/storage/ -v -rs`: **120 passed, 0 skipped, 0 failed** (including live Turso parity check).
- `pytest research/imputer_benchmark/test_exogenous_imputer.py research/imputer_benchmark/test_imputation_benchmark.py -v`: **10 passed, 0 failed**.
- `frontend/`: `npm test` **9 passed**; `npm run build` (`tsc -b && vite build`) **0 errors**.
- Live Uvicorn + built `frontend/dist` headless Playwright inspection across `1280px` & `375px`: all 7 `reports/ui_screenshots/r12_*.png` captured & visually verified.
- *Local Docker note:* Local Windows `com.docker.service` daemon was stopped during execution (recorded in `turso_python_client_wheel_logs.json`), so Linux `kgb0630/ds_108:1.6.0-serving` container build & smoke verification runs on GitHub Actions CI upon push to `origin/main`.
