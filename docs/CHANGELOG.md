# Changelog

All notable canonical reconciliation, MLOps tracking, runtime optimization, and operational limitation updates are recorded in this file.

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
