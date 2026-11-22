# Mini-Task Round 13-Closeout Summary — UI Copy & X-Axis Polish, 2-Tier NASA POWER Schedule (`14d`/`150d` + `--latest-only`), B-Tree Storage Audit (`58.22 -> 28.36 MB`) & Bootstrap CIs

- **Previous Repository Rename Push (`f2f23c8`)**: `Backend Serving CI` Run [`44913877678`](https://github.com/BGKhanh/Daily-Rainfall-Forecasting-System/actions/runs/44913877678) — `completed / success`
- **Live Production DB & Turso `ds108` Policy**: Kept **untouched** (`data/vintage/rainfall_vintage.db` local `41,775,104 B` / `41.78 MB` / `39.84 MiB`; `39.52 MB` reported on Turso `ds108`; `7,035` `automated_pipeline` rows across `2026-05-01..2026-11-17`). All prospective forward-testing queries must filter strictly by **`is_prospective = 1`**.

---

## Step 1 — Frontend UI Synchronization, Dynamic Scroll Counter & Chart X-Axis Spacing

- **Dynamic Top Summary Strip (`frontend/src/App.tsx`, `frontend/src/i18n/en.ts`)**:
  - Replaced the static summary string with `EN.app.summaryStripStatus(prediction)`:
    - `effective_future_days > 0 && is_degraded_input`: `"Single location (Ho Chi Minh City) · Operational forecast uses observed meteorology & estimated solar radiation"`
    - `effective_future_days > 0 && !is_degraded_input`: `"Single location (Ho Chi Minh City) · Operational forecast uses completed meteorology & solar radiation"`
    - `effective_future_days === 0` or loading fallback: `"Single location (Ho Chi Minh City) · Retrospective archive replay on completed history"`
- **Dynamic Forecast Heading & First Yellow Banner Branching (`frontend/src/components/AnchorForecastSection.tsx`, `en.ts`)**:
  - Updated `EN.forecast.mainHeading(..., isDegradedInput)` to emit `(latest complete meteorology; solar estimated)` when `is_degraded_input === true` and `(latest complete input)` otherwise.
  - Branched the lag banner on `prediction.effective_future_days === 0` (`retrospectiveBanner` + `technicalBannerTooltip`) vs `prediction.effective_future_days > 0` (`operationalLagBanner` summarizing `past_unverified`, `today`, and `future_forecast` buckets + `operationalBannerTooltip`).
- **Measured `Selected locations` Scroll Counter (`frontend/src/components/VietnamMap.tsx`, `frontend/src/styles.css`)**:
  - Attached `selectedScrollRef` to `.selected-table-scroll`, computing exact rendered visible vs. hidden `tbody tr` rows via `useLayoutEffect`, `onScroll`, and `ResizeObserver` (with JSDOM `clientHeight === 0` fallback).
  - Updated `.selected-table-scroll` (`max-height: 174px; max-width: 100%; overflow: auto; white-space: nowrap`) and flex/grid wrappers (`min-width: 0; max-width: 100%`) so desktop renders 4 complete rows with `"Scroll for 2 more (showing 4 of 6)"` when 6 units are selected, and mobile `375px` maintains zero horizontal overflow (`scrollWidth = 375 === innerWidth = 375`).
- **Chart X-Axis Inset Padding (`AnchorForecastSection.tsx`)**:
  - Expanded `svgW` from `700` to `760` with `plotInsetX = 38`, `usableW = plotW - 2 * plotInsetX`, and uniform `textAnchor="middle"` across all X-axis date labels (`18 Nov 2026` .. `24 Nov 2026`).
- **Frontend Tests & Screenshots (`frontend/src/__tests__/App.test.tsx`, `reports/ui_screenshots/`)**:
  - Added Vitest tests covering operational degraded, operational non-degraded, and retrospective replay branches (`11 passed (11)`, Vite `dist/` built cleanly in `995ms`).
  - Captured and visually inspected full-page uncropped Playwright screenshots:
    - `reports/ui_screenshots/r13_closeout_forecast_and_map_1280.png` (`364,190 B`)
    - `reports/ui_screenshots/r13_closeout_forecast_and_map_375.png` (`291,015 B`)

---

## Step 2 — 2-Tier NASA POWER Schedule (`14d`/`150d`), `--latest-only` Flag & Official Citations

- **Layer 2/3 Scan Config (`config/runtime_config.toml`, `src/config/settings.py`)**:
  - Set `short_scan_window_days = 14`, `deep_scan_window_days = 150`, and `scan_window_days = 150`, exposed on both `OperationalConfig` and `ApplicationConfig`.
- **Pipeline CLI Hardening (`scripts/mlops/automated_pipeline.py`)**:
  - `ingest-window` accepts `--days` / `--lookback-days` (default `short_scan_window_days = 14`, validated `1..1000` at CLI parse time).
  - `issue-forecasts` supports `--latest-only` (`latest_only: bool = False`): inspects strictly `complete_anchors[-1]`, issues `35` rows if unissued, or returns `SKIPPED` (`no_new_complete_anchor`) without historical backfill (`_skipped_issue_forecasts_result`).
- **2-Tier GitHub Actions Workflow (`.github/workflows/scheduled_ingest.yml`)**:
  - Monday–Saturday (`cron: "23 20 * * 1-6"`): `ingest-window --days 14` → `issue-forecasts --latest-only` → `refresh-cache`.
  - Sunday (`cron: "23 20 * * 0"`): `probe-headers` → `ingest-window --days 150` → `issue-forecasts --latest-only` → `refresh-cache`.
  - Added `scan_tier` (`auto`, `daily_14d`, `weekly_150d`) dispatch input, `permissions: contents: read`, env-bound workflow inputs, and shared `concurrency: ds108-vintage-ingest`.
- **Isolated Pipeline & Audit Tests (`tests/storage/test_automated_pipeline.py`)**:
  - Added tests for `--latest-only` single-anchor issuance & repeat `SKIPPED`, empty-window `SKIPPED`, `--days 14` alias, `--days 0` CLI rejection, and SQLite B-tree footprint / bounds guards.
- **Official `NASA POWER | Data Availability` Citations (`https://power.larc.nasa.gov/status/data/`, accessed `2026-11-21`, Temporal LST)**:
  - Synchronized across `README.md`, `docs/LIMITATIONS.md`, `docs/RUNBOOK_STORAGE.md`, `docs/TURSO_COST_AND_OPERATIONS.md`, and `frontend/src/i18n/en.ts`:
    - `GMAO GEOS-IT` (`0.5° × 0.625°`, `Oct 1, 2026 – Nov 18, 2026`): median `3 days`, current `3 days`.
    - `CERES FLASHFlux v4A` (**`1° × 1°` confirmed**, `Aug 31, 2026 – Nov 15, 2026`): median `5 days`, current `6 days` (resolving legacy `1/2° × 1/2°` Solar FAQ discrepancy).
    - `GMAO MERRA-2` (`0.5° × 0.625°`, `Jan 1, 1981 – Sep 30, 2026`): median `38 days`, current `52 days`.
    - `CERES SYN1deg Ed 4.1` (`1° × 1°`, `Jan 1, 2001 – Aug 30, 2026`): median `93 days`, current `83 days` (`GEWEX SRB R4-IP` ended `Dec 31, 2000`).

---

## Step 3 — B-Tree Storage Audit (`39.52 MB` / `58.22 -> 28.36 MB`) & `P-c` Bootstrap 95% CIs

- **SQLite B-Tree Page & Column Byte Audit (`scripts/audit/audit_db_storage_footprint.py` → `reports/storage_audit/db_storage_footprint_audit_r13_closeout.json`)**:
  - Deterministic binary SQLite B-tree + overflow page walker verified exact active page accounting:
    - **Production DB** (`data/vintage/rainfall_vintage.db`, read-only snapshot): `41,775,104 B` (`41.78 MB` / `39.84 MiB`, `39.52 MB` on Turso `ds108`), `page_count = 10,199`, `freelist_count = 0`; `forecasts_ledger` + 3 indexes = `34,553,856 B` (`8,217` table pages = `33,656,832 B`), with **`input_vintage_manifest_json`** consuming **`24,558,135 B` (`24.56 MB`, avg `3,490.85 B/row`, `76%` of table bytes due to 35× per-anchor duplication)** vs **`imputed_values_json`** at **`394,275 B` (`0.39 MB`, avg `56.04 B/row` overall / `231.18 B/row` across `1,659` degraded rows)**.
    - **Bundled Demo DB** (`artifacts/models/demo_rainfall_vintage.db`): before `VACUUM` = `58,224,640 B` (`58.22 MB`, `14,215` pages, `freelist_count = 7,282` = `29,827,072 B` freed during `DDL V3.5` table rebuild); after `VACUUM` + `finalize_db_for_read_only()` = **`28,356,608 B` (`28.36 MB` / `27.04 MiB`, `6,923` pages, `freelist_count = 0`, reclaimed `29,868,032 B` = `29.87 MB`)** with `29/29` pre-existing column SHA-256 digests bit-identical (`db_file_sha256 = 7466f39dcbbd71e1d70a13d5f3c81d7624094cce7edba637922c7341851bed08`), synced in `immutable_ledger_migration_proof_v35.json` & `artifacts/models/metadata.json`.
  - Documented the 63-province scaling roadmap (`input_window_manifests` deduplication table cutting row size by `~85%`) in `docs/TURSO_COST_AND_OPERATIONS.md`.
- **Moving Block Bootstrap 95% CIs (`block_length = 7`, `n_bootstrap = 2000`, `seed = 42`) in `pc_integration_audit_2184.json` & `research/imputer_benchmark/README.md`**:
  - Extracted shared `build_block_bootstrap_indices` and `summarize_bootstrap_samples` helpers in `src/inference/forecast_runner.py` and reused them in `research/imputer_benchmark/run_pc_integration_experiment.py`.
  - Held-out Dev Test (`n = 2,184`, `h=1..7`) `p_c_doy_median_all_11_solar`:
    - `SC-35`: `MAE 3.7314 [3.3259, 4.2174]`, `delta_MAE -0.0019 [-0.0069, +0.0023]`, `RMSE 8.0940 [6.6681, 9.7836]`, `Fidelity-MAE 0.1224`.
    - `SC-31`: `MAE 3.7227 [3.3155, 4.2071]`, `delta_MAE -0.0020 [-0.0082, +0.0037]`, `RMSE 8.1159 [6.6896, 9.8009]`, `Fidelity-MAE 0.1652`.
    - `SC-30`: `MAE 3.7502 [3.3402, 4.2285]`, `delta_MAE +0.0002 [-0.0068, +0.0064]`, `RMSE 8.1059 [6.6823, 9.7758]`, `Fidelity-MAE 0.1668`.

---

## Step 4 — Verification & Reviews

- **Independent Subagent Reviews**: Completed Standards (`e18bf807`), Spec Conformance (`8bfe8a36`), and Security/Differential (`0a11b5f8`) reviews; resolved all actionable findings (binary bounds guards, workflow `permissions` + env-bound expressions, shared bootstrap helper deduplication, stray `0` file cleanup, CLI `--days` validation, and extra unit tests).
- **Local Verification (Final Run After Last Edit)**:
  - **Ruff Lint**: `ruff check ...` — `All checks passed!`
  - **Backend Pytest** (`tests/serving/ tests/storage/ tests/test_pc_solar_imputation_contract.py research/imputer_benchmark/test_exogenous_imputer.py` with `SERVING_MODEL_SOURCE=local`, `WEATHER_DB_BACKEND=local`, `CI_REQUIRE_PARITY=1`): **`132 passed, 31 warnings in 263.10s`** (includes OpenAPI snapshot parity).
  - **Frontend Vitest + Vite Build**: **`11 passed (11)`**, Vite `dist/` compiled cleanly in `995ms`.
