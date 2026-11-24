# Mini-Task Round 15 Summary — Pre-Production Deployment Hardening (`1.8.0-serving`, Decoupled Cloudflare + AWS EC2 + Turso Read-Only Replica)

- **Architecture Scope**: Decoupled static SPA on **Cloudflare Workers Assets** (`frontend/wrangler.jsonc`) + pure-Python CPU backend (`kgb0630/ds_108:1.8.0-serving`, Docker Hub digest `sha256:5c3788bef5113086004b4dd2b27211607ea20a36ed715e79f97106ed03a00f35`) behind Caddy TLS on AWS EC2 + embedded SQLite replica syncing exclusively over Turso Read-Only tokens.
- **Invariant Model & DB Lock Preserved**: Zero changes to ML models (`SC-35`, `SC-30`, `SC-31`), features (`70`/`65`), forecast horizon ($H=7$), `DDL V3.5` schema, `forecasts_ledger`, or default (`development`) `docs/openapi_snapshot.json` contract.

---

## DoD Gate 1 — Round 14 Doc Sync & Docker Version Boundary

- **Verified CI Runs for Round 14 Final Head (`fd614d86117560891534412b09f11cb9aa80ab0f`)**:
  - `Backend Serving CI`: Run ID [`45120243799`](https://github.com/BGKhanh/Daily-Rainfall-Forecasting-System/actions/runs/45120243799) — `completed / success`
  - `Scheduled NASA POWER Ingest & Forecast Issuance`: Run ID [`45329181377`](https://github.com/BGKhanh/Daily-Rainfall-Forecasting-System/actions/runs/45329181377) — `completed / success`
- **Updated [`reports/storage_audit/SUMMARY_ROUND14.md`](./SUMMARY_ROUND14.md)**:
  - Recorded `fd614d8` CI Run IDs under `Step 0`.
  - Added Docker release boundary clarification: `1.7.0-serving` (official Round 13-Closeout & Round 14 frozen bundled baseline), `1.7.1-serving` (intermediate experimental Docker Hub patch image prior to Round 15), `1.8.0-serving` (official Round 15 decoupled pure-Python serving release).
  - Verified exact parity against `research/horizon_extension/h11_feasibility_audit.json` (`Test P-c ΔP90-Tail h=2..7: +0.2401..+0.4301 mm`, `Val P-c ΔP90-Tail h=2..7: -0.0175..+0.3549 mm`, `Test P-c ΔRMSE h=1: -0.5741 [-0.7172, -0.4533] -> h=7: -0.0819 -> h=10: -0.0262 mm`).

---

## DoD Gate 2 — Zero-Hardcode Env Audit, Templates & Leak Prevention

- **Single Source of Truth ([`src/config/settings.py`](../../src/config/settings.py))**:
  - Added Pydantic `RuntimeEnvConfig` (Layer 1 non-secret runtime env: `app_env`, `log_level`, `enable_api_docs`, `serve_static_spa`, `turso_replica_path`, `vintage_db_path`, `serving_model_source`, `weather_db_backend`), strictly excluded from `config_sha256`, `result_config_sha256`, and `operational_config_sha256` so ledger hashes never diverge between `development` and `production`.
  - Centralized Layer 2 overrides (`STORE_BACKEND`, `TURSO_SYNC_INTERVAL_SEC` bounded in `[30, 86400]`, normalized `CORS_ALLOWED_ORIGINS` with `.strip().rstrip("/")`, deduplication, and automatic `localhost`/`127.0.0.1` removal when `app_env == "production"`).
  - Added fail-fast `validate_runtime_env(settings)` rejecting empty/placeholder `TURSO_DB_URL` (`your-db-your-org`, `<your_`, `changeme`, `replace_me`, `example.com`, `your-domain`), missing/placeholder Turso token, or missing/`*`/localhost/placeholder `CORS_ALLOWED_ORIGINS` in `APP_ENV=production` with explicit `ValueError`.
  - Eliminated scattered `os.environ.get(...)` reads in [`src/api/main.py`](../../src/api/main.py) and `get_replica_manager()` in [`src/data/turso_backend.py`](../../src/data/turso_backend.py).
  - Extended `redact_secrets()` in [`src/data/turso_backend.py`](../../src/data/turso_backend.py) to mask `TURSO_DB_URL` / `TURSO_DATABASE_URL`, `*.turso.io` URLs, `TURSO_REPLICA_PATH`, `VINTAGE_DB_PATH`, `PROJECT_ROOT`, and `/var/lib/rainfall*` paths alongside bearer/JWT/env tokens.
- **Templates, Inventory & Ignore Guards**:
  - Delivered role-scoped templates [`.env.example`](../../.env.example), [`.env.serving.example`](../../.env.serving.example), [`.env.ingest.example`](../../.env.ingest.example), [`frontend/.env.example`](../../frontend/.env.example), plus complete env matrix [`docs/ENV_CONTRACT.md`](../../docs/ENV_CONTRACT.md).
  - Tightened [`.gitignore`](../../.gitignore) (`.env.*` + explicit `!` allowlist for examples and `SUMMARY_ROUND15.md`, `*.pem`, `*.key`, `*.sqlite*`) and [`.dockerignore`](../../.dockerignore) (removed duplicate `scratch`, added `*.pem`, `*.key`, `*.sqlite*`, `*.db-wal`, `*.db-shm`, `research`, `frontend`, `deploy`, `docs`, `docker_reports`).
  - Anti-leak scan verified 0 live tokens, passwords, or static IPs tracked in Git.

---

## DoD Gate 3 — Turso Read-Only Token, Zero-Write Serving & Cross-Platform Replica Hashing

- **Zero-Write Verification**:
  - Confirmed `STORE_BACKEND=turso_replica` startup `sync_now()` issues strictly `SELECT` statements to Turso Primary (`0` DDL, `0` DML, `0` `PRAGMA` sent remotely; `init_schema(..., seed_hcmc_location=False)` executes exclusively on the local container staging SQLite file).
  - Fixed `lifespan()` ordering in [`src/api/main.py`](../../src/api/main.py) so `_cached_doy_climatology` reloads from `mgr.replica_path` immediately after startup `mgr.sync_now()`.
  - Confirmed `TURSO_AUTH_TOKEN` works directly for `TursoHranaClient(read_only=True)` while preserving precedence `TURSO_READONLY_AUTH_TOKEN` > `TURSO_DB_TOKEN` / `TURSO_AUTH_TOKEN`.
- **Cross-Platform Float Hashing Bug Caught & Fixed**:
  - Live Linux container testing against `ds108` caught a real `REPLICA_SHA256_MISMATCH` on `forecasts_ledger` caused by SQLite C-level `ROUND(pred_mm, 8)` differing between Linux SQLite `3.46.1` (`980f04f64ddf...`) and Windows/Turso Primary (`6849f2cc52cf...`). Replaced SQL `ROUND(pred_mm, 8)` in `compute_table_content_sha256` with deterministic Python formatting `_format_canonical_hash_cell()` (`f"{normalized:.8f}"`), yielding identical SHA-256 (`b2183b135619145038948e2b0afde6e1067795acaa0db6b07331880ccf99f1e5`) on Linux Docker and Turso Remote.
- **Live Docker + Real Turso Read-Only Token (`1d` Ephemeral `--read-only`) Smoke Proof**:
  - **Direct Primary write probe (`CREATE TABLE IF NOT EXISTS _ro_probe (id INT)`) with RO token**: Turso blocked with `code="BLOCKED"`, `"Operation was blocked: SQL write operations are forbidden (current session doesn't have write permission)"` (token & URL redacted).
  - **Container Boot #1 (cold-start initial seed with `APP_ENV=production`, `CORS_ALLOWED_ORIGINS=https://daily-rainfall-forecasting-system.22520630.workers.dev/`)**: `GET /health` -> `200 OK`, `status="healthy"`, `db_role="turso_replica"`, `default_anchor="2026-11-17"`, `replica_sync_count=1`, `replica_last_sync_at_utc="2026-11-24T09:25:33.092816Z"`, `replica_last_sync_duration_ms=7867.88`, `replica_hash_verified=true`, exact normalized header `access-control-allow-origin: https://daily-rainfall-forecasting-system.22520630.workers.dev` (zero URL/token/path in response).
  - **Container Boot #2 (restart preserving `/var/lib/rainfall` volume)**: Fast delta sync completed in **`934.6 ms`** (`~8.4x` faster than initial seed) with `replica_hash_verified=true`.

---

## DoD Gate 4 — Decoupled Pure-Python `Dockerfile` (`1.8.0-serving`) & FastAPI Factory

- Copied `frontend/public/map/polygon_to_location.json` -> [`config/polygon_to_location.json`](../../config/polygon_to_location.json) with bit-identical contract test (`test_config_polygon_to_location_matches_frontend_map`); updated `_load_map_location_catalog()` to prioritize `config/`.
- Stripped `node:24-alpine` Stage 1 & `frontend/dist` copy from [`Dockerfile`](../../Dockerfile); added `/var/lib/rainfall` owned by non-root `appuser` (`10001`) and `uvicorn ... --proxy-headers --forwarded-allow-ips *`.
- Added `create_app()` factory in [`src/api/main.py`](../../src/api/main.py): disables `/docs`, `/redoc`, `/openapi.json` (`404` verified in container) when `APP_ENV=production`, attaches `@app.exception_handler(Exception)` returning clean `ErrorResponse` `500` JSON without stacktrace/SQL leakage, and gates `StaticFiles` behind `SERVE_STATIC_SPA=true`.
- **Docker Image Comparison & Resource Verification**:
  - Build context reduced to **`28.59 MB`**.
  - `kgb0630/ds_108:1.8.0-serving` image disk size: **`561 MB`** vs prior `1.7.1-serving` **`592 MB`** (`-31 MB` smaller, `node` binary confirmed absent).
  - Runtime memory inside live container: `docker stats = 301.5 MiB (3.88%)`, **`VmRSS = 280.32 MB <= 350 MB`** (live Turso RO replica boot), and **`VmRSS = 212.55 MB`** (`APP_ENV=production` local smoke on final rebuilt image).
  - Pushed to Docker Hub: `kgb0630/ds_108:1.8.0-serving` (`digest: sha256:5c3788bef5113086004b4dd2b27211607ea20a36ed715e79f97106ed03a00f35`).

---

## DoD Gate 5 — Cloudflare Workers Assets Frontend & `npm audit`

- Added [`frontend/wrangler.jsonc`](../../frontend/wrangler.jsonc) (`workers_dev: true`, `not_found_handling: "single-page-application"`, `assets.directory: "./dist"`).
- Hardened [`frontend/vite.config.ts`](../../frontend/vite.config.ts) (`validateProdApiBasePlugin` warns if `VITE_API_BASE_URL` is empty on `vite build` and fails fast when `CLOUDFLARE_WORKERS=1` or `REQUIRE_VITE_API_BASE_URL=1` — verified failing when empty and succeeding in `583ms` when set) and [`frontend/src/api/client.ts`](../../frontend/src/api/client.ts) (`trim().replace(/\/+$/, "")` + runtime console warning in prod if unset).
- Retained [`frontend/public/map/ATTRIBUTION.md`](../../frontend/public/map/ATTRIBUTION.md) in place because `frontend/src/i18n/en.ts` references `See ATTRIBUTION.md.` for CC BY 4.0 compliance (`geoBoundaries` / `nguyenduy1133/Free-GIS-Data`).
- **`npm audit` Report (`frontend/`)**:
  - `npm audit --omit=dev` (runtime production dependencies `react`, `react-dom`): **`found 0 vulnerabilities`**.
  - Full audit (including `devDependencies`): **6 advisories** (`3 moderate` `@vitest/mocker`/`esbuild`, `1 high`/`2 critical` `tinypool <= 2.1.1`) confined strictly to test/dev tooling (`vitest@2.1.9`); `npm audit fix` (without `--force`) applied `0` breaking major-version jumps (`vitest 2.1 -> 5.1` excluded intentionally). All **11 Vitest tests passed** and `vite build` compiled cleanly (`239.99 kB` JS bundle).

---

## DoD Gate 6 — EC2 Production Compose, Caddy Proxy & Runbook

- Delivered [`deploy/ec2/docker-compose.prod.yml`](../../deploy/ec2/docker-compose.prod.yml) (`app` port `8000` exposed internally only, `512m` limit, `/var/lib/rainfall` volume, `10m x 3` JSON log rotation; `caddy:2-alpine` on `80/443`, `128m` limit, `depends_on: service_healthy`), [`deploy/ec2/Caddyfile.example`](../../deploy/ec2/Caddyfile.example) (`X-Content-Type-Options: nosniff`, `Referrer-Policy`, `X-Frame-Options: DENY`, `-Server`), and deployment runbook [`deploy/README.md`](../../deploy/README.md) (linked from [`README.md`](../../README.md)).

---

## DoD Gate 7 — Verification, Reviews & CI Pipeline Synchronization

- Updated [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml): Tier 2 builds `kgb0630/ds_108:1.8.0-serving`, removes `/` SPA HTML assertion, tests normalized CORS header with intentional trailing slash, checks `GET /observations?location_id=HANOI` (`has_data=false`), verifies `node` absent, and enforces `VmRSS <= 350 MB`.
- **Local Verification (Final Run After Last Code/Test Edit)**:
  - `ruff check` across serving, storage, config, and test targets: **`All checks passed!`**
  - `pytest tests/serving/ tests/storage/ tests/test_pc_solar_imputation_contract.py -q` (with `SERVING_MODEL_SOURCE=local`, `WEATHER_DB_BACKEND=local`, `CI_REQUIRE_PARITY=1`): **`143 passed, 24 warnings in 188.77s`** (includes new [`tests/serving/test_env_and_prod_hardening.py`](../../tests/serving/test_env_and_prod_hardening.py), [`test_config_contract.py`](../../tests/serving/test_config_contract.py) polygon catalog check, and full OpenAPI snapshot parity).
  - `cd frontend && npm test && npm run build`: **`11 passed`**, `vite build` clean (`583ms`).
- **Independent Subagent Reviews**: Executed Standards, Spec Conformance, and Security/Differential reviews across the full Round 15 diff before commit & push.
- **GitHub Actions CI Status on Final Code/Docker `main` Push (`a17f47801171dd8e27294b7b691fcfc224f974a8`)**:
  - `Backend Serving CI`: Run ID [`45370495578`](https://github.com/BGKhanh/Daily-Rainfall-Forecasting-System/actions/runs/45370495578) — **`completed / success`** across all 3 jobs (`Tier 1 CI — Ruff Lint, Serving Unit/Golden Parity & SQLite DDL V3 Trigger Tests`, `DagsHub MLflow Registry Checksum Audit (Optional)`, and `Tier 2 CI — Docker Container Build, Linux Parity & RSS Smoke Check`).
