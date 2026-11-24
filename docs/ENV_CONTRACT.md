# Environment Variables Contract (`ds108-rainfall-forecast`)

Every environment variable read across `src/`, `scripts/`, `frontend/`, `.github/workflows/`, and `deploy/` is inventoried below.
Layer 1 runtime/secret variables are centralized in [`src/config/settings.py`](../src/config/settings.py) (`RuntimeEnvConfig`, `SecretsEnvView`, `validate_runtime_env`) and are **strictly excluded** from `config_sha256` and `operational_config_sha256` so forecast provenance remains invariant across `development` and `production`.

Role-scoped templates:
- Local development: [`.env.example`](../.env.example)
- Backend Serving (AWS EC2 Docker + Turso Read-Only replica): [`.env.serving.example`](../.env.serving.example)
- Batch Ingest & MLOps CI: [`.env.ingest.example`](../.env.ingest.example)
- Frontend Static Build (Cloudflare Workers Assets): [`frontend/.env.example`](../frontend/.env.example)

---

## 1. Backend Serving Runtime & Operational Overrides

| Variable | Layer / Scope | Canonical Reader | Default (Dev) | Production / Turso Fail-Fast Contract (`validate_runtime_env`) | Sensitivity |
|---|---|---|---|---|---|
| `APP_ENV` | Layer 1 Serving | `src/config/settings.py::RuntimeEnvConfig` | `"development"` | Must be `"development"`, `"production"`, or `"test"`. When `"production"`, enforces Turso creds, HTTPS non-localhost CORS, and disables `/docs`, `/redoc`, `/openapi.json` by default. | Non-Secret |
| `LOG_LEVEL` | Layer 1 Serving | `src/config/settings.py::RuntimeEnvConfig` | `"INFO"` | Validated in `{"DEBUG","INFO","WARNING","ERROR","CRITICAL"}`. Applied to root & API logger. | Non-Secret |
| `ENABLE_API_DOCS` | Layer 1 Serving | `src/config/settings.py::RuntimeEnvConfig` | `true` (dev/test), `false` (prod) | Explicit `"true"`/`"1"` or `"false"`/`"0"`. Defaults to `false` when `APP_ENV=production`. | Non-Secret |
| `SERVE_STATIC_SPA` | Layer 1 Serving | `src/config/settings.py::RuntimeEnvConfig` | `false` | Decoupled architecture keeps SPA on Cloudflare Workers; only mounts `frontend/dist` when explicitly `"true"`. | Non-Secret |
| `STORE_BACKEND` | Layer 2 Override | `src/config/settings.py::load_settings`, `src/data/turso_backend.py::resolve_store_backend` | `"local"` (from TOML) | Validated in `{"local","turso_replica","turso_remote"}`. Serving production uses `"turso_replica"`. | Non-Secret |
| `TURSO_DB_URL` | Layer 1 Secret/Endpoint | `src/config/settings.py::SecretsEnvView` | `""` | **Required & fail-fast** when `APP_ENV=production` or `STORE_BACKEND in {"turso_replica","turso_remote"}`: non-placeholder `libsql://` or `https://`. Redacted from logs/errors. | Private Endpoint (Redacted) |
| `TURSO_DATABASE_URL` | Layer 1 Alias / Legacy | `src/config/settings.py::SecretsEnvView`, `src/data/provider.py` | `""` | Fallback alias if `TURSO_DB_URL` is unset. Redacted from logs/errors. | Private Endpoint (Redacted) |
| `TURSO_AUTH_TOKEN` | Layer 1 Secret | `src/config/settings.py::SecretsEnvView` | `""` | **Required & fail-fast** in `production` / Turso modes (unless `TURSO_READONLY_AUTH_TOKEN` or `TURSO_DB_TOKEN` is set). EC2 serving must receive a `--read-only` token. | **Secret** |
| `TURSO_DB_TOKEN` | Layer 1 Secret Alias | `src/config/settings.py::SecretsEnvView` | `""` | Alias / CI RW token slot; `TURSO_AUTH_TOKEN` works identically for both `read_only=True` and RW paths. | **Secret** |
| `TURSO_READONLY_AUTH_TOKEN` | Layer 1 Secret (Optional) | `src/config/settings.py::SecretsEnvView` | `""` | Highest-precedence token when `TursoHranaClient(read_only=True)` is opened (useful when one host holds both RO & RW tokens). | **Secret** |
| `TURSO_REPLICA_PATH` | Layer 1 Serving | `src/config/settings.py::resolve_turso_replica_path` | `"data/vintage/turso_replica.db"` | Relative paths resolve under `PROJECT_ROOT`; absolute paths (EC2 `/var/lib/rainfall/replica.db`) preserved verbatim. Redacted in error logs. | Non-Secret |
| `TURSO_SYNC_INTERVAL_SEC` | Layer 2 Override | `src/config/settings.py::load_settings` | `300` (from TOML) | Integer bounded in `[30, 86400]`. | Non-Secret |
| `CORS_ALLOWED_ORIGINS` | Layer 2 Override | `src/config/settings.py::load_settings` | Dev localhosts in TOML | Comma-separated, `.strip().rstrip("/")`, deduplicated. When `APP_ENV=production`, auto-strips `localhost`/`127.0.0.1` and requires >= 1 explicit `http(s)://` non-`*` non-placeholder origin. | Non-Secret |
| `VINTAGE_DB_PATH` | Layer 1 Storage | `src/config/settings.py::RuntimeEnvConfig`, `src/data/provider.py` | `""` | Optional local SQLite path override (`local_or_production_db` role). Redacted in errors. | Non-Secret |
| `SERVING_MODEL_SOURCE` | Layer 1 Serving | `src/config/settings.py::RuntimeEnvConfig`, `src/inference/predictor.py` | `"local"` | `"local"` or `"mlflow"` (serving container strictly `"local"`). | Non-Secret |
| `WEATHER_DB_BACKEND` | Layer 1 Legacy | `src/config/settings.py::RuntimeEnvConfig`, `src/data/provider.py` | `"local"` | `"local"`, `"sqlite"`, `"turso"`, `"supabase"`, `"sample_csv"`. | Non-Secret |
| `BACKEND_DOMAIN` | EC2 Compose / Caddy | `deploy/ec2/docker-compose.prod.yml`, `deploy/ec2/Caddyfile.example` | — | Hostname routed by Caddy TLS proxy (e.g. `api.your-domain.example`). | Non-Secret |
| `IMAGE_TAG` | EC2 Compose | `deploy/ec2/docker-compose.prod.yml` | `"1.8.0-serving"` | Docker image tag for `kgb0630/ds_108`. | Non-Secret |

---

## 2. Batch Ingest, MLOps & CI/CD Variables

| Variable | Scope | Reader | Default | Purpose / Sensitivity |
|---|---|---|---|---|
| `TURSO_API_TOKEN` | Batch / Audit Only | `src/config/settings.py::SecretsEnvView`, `TursoPlatformClient` | `""` | **Secret** — Turso Platform API token for usage/token ops (`scripts/mlops/` only; never on serving EC2). |
| `CACHE_REFRESH_WEBHOOK_URL` / `REFRESH_WEBHOOK_URL` | Batch / CI Webhook | `.github/workflows/scheduled_ingest.yml` -> `automated_pipeline.py refresh-cache` | `""` | Optional post-ingest HTTP refresh ping URL (**Secret** if tokenized). |
| `MLFLOW_TRACKING_URI` | MLOps / CI Verify | `src/config/mlops.py`, `.github/workflows/ci.yml` | DagsHub repo MLflow URI | Non-Secret tracking endpoint (dev/CI only; `mlflow` excluded from Docker serving image). |
| `MLFLOW_TRACKING_USERNAME` | MLOps / CI Verify | `src/config/mlops.py`, `.github/workflows/ci.yml` | `"BGKhanh"` | Non-Secret DagsHub username. |
| `MLFLOW_TRACKING_PASSWORD` | MLOps / CI Verify | `src/config/mlops.py`, `.github/workflows/ci.yml` | `""` | **Secret** (`DAGSHUB_USER_TOKEN` mapped into CI env). |
| `DAGSHUB_TOKEN` / `DAGSHUB_USER_TOKEN` | MLOps / CI Verify | `src/config/mlops.py`, `.github/workflows/ci.yml` | `""` | **Secret** DagsHub token (redacted by `redact_secrets`). |
| `MLFLOW_HTTP_REQUEST_*`, `MLFLOW_DISABLE_AGENT_HINT` | MLOps / CI Only | `src/config/mlops.py`, `scripts/mlops/check_ecosystem_compatibility.py` | `"45"`, `"3"`, `"2"`, `"1"` | Non-Secret MLflow client retry/hint defaults. |
| `CI_REQUIRE_PARITY` | Tier 1 CI | `tests/serving/test_api.py` | unset | `"1"` enforces 2,184-origin NPZ parity test in CI. |
| `GIT_SHA` | Batch Provenance | `src/data/vintage_store.py` | git HEAD subprocess fallback | Non-Secret SHA stamped in `ingestion_runs.git_sha`. |

---

## 3. Frontend Build-Time Variables (Cloudflare Workers Assets + Vite)

| Variable | Scope | Reader | Default | Build Contract & Notes |
|---|---|---|---|---|
| `VITE_API_BASE_URL` | Frontend Build (Public) | `frontend/src/api/client.ts`, `frontend/vite.config.ts` | `""` (Vite dev proxy) | `.trim().replace(/\/+$/, "")`. **Required in Cloudflare Workers production builds**; `vite.config.ts` warns on empty prod builds and fails fast when `CLOUDFLARE_WORKERS=1` or `REQUIRE_VITE_API_BASE_URL=1`. |
| `REQUIRE_VITE_API_BASE_URL` / `CLOUDFLARE_WORKERS` | Build Guard Flag | `frontend/vite.config.ts` | unset (`"1"` enables) | Fail-fast guard preventing accidental same-origin builds for decoupled Cloudflare SPA deployment. |
| `VITE_COLD_START_THRESHOLD_MS` | Frontend Build (Public) | `frontend/src/api/client.ts` | `"3000"` | Cold-start status banner threshold (ms). |
| `VITE_REPO_URL`, `VITE_CI_URL`, `VITE_MLFLOW_URL` | Frontend Build (Public) | `frontend/src/api/client.ts` | Repo / Actions / DagsHub URLs | External reference links rendered in UI header & About modal. |

---

## 4. Legacy / Optional Research & Provider Variables

| Variable | Reader | Default | Notes |
|---|---|---|---|
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `SUPABASE_ANON_KEY`, `SUPABASE_WEATHER_TABLE` | `src/data/provider.py` | `""` / `"daily_weather_history"` | Legacy optional Supabase adapter (`WEATHER_DB_BACKEND=supabase`); unused in Turso/SQLite serving. |
| `TURSO_WEATHER_TABLE` | `src/data/provider.py` | `"daily_weather_history"` | Legacy flat-table adapter (`WEATHER_DB_BACKEND=turso`). |
| `REPORT_MODE` | `src/data/dataquality.py` | `"notebook"` | Controls notebook vs console formatting in data-quality helper. |
