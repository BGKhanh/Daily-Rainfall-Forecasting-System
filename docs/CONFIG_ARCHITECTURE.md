# 5-Layer Configuration Architecture (`ds108-rainfall-forecast`)

All runtime, application, model contract, secret, and frozen research constants are organized into 5 strict layers loaded and validated via [`src/config/settings.py`](../src/config/settings.py) (Pydantic v2 + stdlib `tomllib`).

| Layer | Purpose & Owner | Storage Location | Mutable Without Retrain? | Included in `config_sha256`? |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1: Secrets & Credentials** | Database tokens, platform API keys, DagsHub MLflow tokens | `.env` (gitignored) & `.env.example` (template) | Yes | **No** (excluded so tokens never enter DB provenance) |
| **Layer 2: Operational Runtime** | Store backend (`local` / `turso_replica` / `turso_remote`), Turso sync interval/jitter, NASA API timeout/max params, scan windows, SQLite cache size, CORS | [`config/runtime_config.toml`](../config/runtime_config.toml) `[operational]` | Yes | Yes |
| **Layer 3: Application & UI Policy** | Default query window (`90d`), granularity range limits (`366 / 2555 / 10000`), rain threshold (`1.0 mm`), UI prior preview (`14d`), map zoom bounds (`1.0..6.0`), max visible selected rows (`5`) | [`config/runtime_config.toml`](../config/runtime_config.toml) `[application]` | Yes | Yes |
| **Layer 4: Serving Model Contract** | Immutable model boundary: `horizon_days=7`, `required_input_variables` (exact 35 NASA POWER vars verified via per-variable probe), `min_contiguous_history_days=30`, `recommended_fetch_window_days=35`, `feature_warmup_lookback_days=29`, `time_standard=LST`, `canonical_location_id=HCMC`, `completeness_policy=COMPLETE_MODEL_REQUIRED_VARS` | [`artifacts/models/metadata.json`](../artifacts/models/metadata.json) `["model_contract"]` + [`config/variable_catalog.json`](../config/variable_catalog.json) | **No** (startup guard aborts with `[MODEL_CONTRACT_MISMATCH]`) | Yes |
| **Layer 5: Frozen Research Constants** | Dev Train / Val / Test split cutoffs (`2018-04-30`, `2020-04-30`, `2026-04-30`), wet season months, rolling/lag windows, Tweedie power `1.5`, MASE scale `3.7015424361567635` | [`config/training_eda_freeze_v1.toml`](../config/training_eda_freeze_v1.toml) & `metadata.json["frozen_eda_modeling_config"]` | **No** (audit record of model training freeze) | Yes |

## Startup Contract Guard & Provenance Stamping
- `verify_model_contract_on_startup()` runs inside both `StandaloneRainfallPredictor.__init__()` and FastAPI `load_serving_models()`, raising `RuntimeError("[MODEL_CONTRACT_MISMATCH] ...")` if `horizon_days`, `time_standard`, `canonical_location_id`, `feature_warmup_lookback_days`, or any of the 35 input variables diverge from `metadata.json["model_contract"]` or `config/variable_catalog.json`.
- `compute_config_sha256()` hashes Layers 2–5 deterministically and stamps `config_sha256` into DDL V3.3 `ingestion_runs` and `forecasts_ledger` rows (`schema_migrations` version `V3.3`).
