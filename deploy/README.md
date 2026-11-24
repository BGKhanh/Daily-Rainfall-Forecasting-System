# Production Deployment Guide (Cloudflare Workers + AWS EC2 + Turso Read-Only Replica)

Architecture overview:
- **Frontend SPA**: Static Vite + React bundle hosted on **Cloudflare Workers Assets** (`frontend/wrangler.jsonc`), calling the backend via HTTPS CORS (`VITE_API_BASE_URL`).
- **Backend Serving API**: Pure-Python CPU Docker container (`kgb0630/ds_108:1.8.0-serving`) running behind **Caddy TLS reverse proxy** on AWS EC2 `t3.micro` (Ubuntu 24.04). Port `8000` stays internal to the Compose network (`expose: ["8000"]`).
- **Database**: **Turso Primary** (`libsql://...`) synced via Read-Only Hrana HTTP (`SELECT` only — 0 DDL, 0 DML) into a persistent Docker volume (`/var/lib/rainfall/replica.db`). Scheduled GitHub Actions ingest (`.github/workflows/scheduled_ingest.yml`) holds the sole Read-Write token.

---

## 1. Deploy Frontend to Cloudflare Workers Assets

In Cloudflare Dashboard (**Workers & Pages -> Create / Settings -> Builds**):

- **Root directory**: `frontend`
- **Build command**: `npm run build` *(Cloudflare automatically runs `npm clean-install` before build; avoid duplicating `npm ci` in the build command).*
- **Deploy command**: `npx wrangler deploy` (uses `frontend/wrangler.jsonc`, `not_found_handling: "single-page-application"`)
- **Build Watch Paths**: `frontend/*`
- **Build Environment Variables** (non-secret):
  - `VITE_API_BASE_URL=https://api.your-domain.example` *(exact HTTPS backend domain, no trailing slash)*
  - `REQUIRE_VITE_API_BASE_URL=1` *(fails the Vite build immediately if `VITE_API_BASE_URL` is blank)*
  - `VITE_COLD_START_THRESHOLD_MS=3000`

Local dry-run check:

```bash
cd frontend
REQUIRE_VITE_API_BASE_URL=1 VITE_API_BASE_URL=https://api.your-domain.example npm run build
```

---

## 2. Deploy Backend Serving on AWS EC2 `t3.micro` (Ubuntu 24.04 + 1 GB Swap)

### 2.1. One-Time Host Hardening & Swap

```bash
# Create 1 GB swap on t3.micro (1 GiB RAM safety buffer)
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Ensure UFW allows only SSH (22), HTTP (80), HTTPS (443) — NEVER expose 8000
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

Install Docker Engine & Docker Compose plugin, then prepare the deployment directory:

```bash
mkdir -p ~/ds108-serving && cd ~/ds108-serving
```

### 2.2. Create a Turso Read-Only Token (Principle of Least Privilege)

```bash
turso db tokens create <your-db-name> --read-only
```

Verify that your serving `.env.serving` uses **only** this `--read-only` token (never your Read-Write `TURSO_AUTH_TOKEN` from GitHub Actions).

### 2.3. Configure `.env.serving` and `Caddyfile`

Copy [`deploy/ec2/docker-compose.prod.yml`](./ec2/docker-compose.prod.yml), [`deploy/ec2/Caddyfile.example`](./ec2/Caddyfile.example) -> `Caddyfile`, and [`.env.serving.example`](../.env.serving.example) -> `.env.serving`:

```bash
chmod 600 .env.serving
# Required edits in .env.serving:
#   BACKEND_DOMAIN=api.your-domain.example
#   APP_ENV=production
#   STORE_BACKEND=turso_replica
#   TURSO_DB_URL=libsql://<your-db>.turso.io
#   TURSO_AUTH_TOKEN=<turso-read-only-token>
#   CORS_ALLOWED_ORIGINS=https://daily-rainfall-forecasting-system.22520630.workers.dev
```

Ensure `.env.serving` is in the same directory as `docker-compose.prod.yml`, and verify Compose variable interpolation (`BACKEND_DOMAIN`):

```bash
docker compose --env-file .env.serving -f docker-compose.prod.yml pull
docker compose --env-file .env.serving -f docker-compose.prod.yml up -d
```

### 2.4. Verify Production Deployment

```bash
# Container health & memory budget (expect status=healthy, db_role=turso_replica, VmRSS <= 350 MB)
curl -sf https://api.your-domain.example/health | python3 -m json.tool

# Confirm OpenAPI docs are hidden in APP_ENV=production (expect 404)
curl -i https://api.your-domain.example/docs

# Confirm CORS exact origin header (without trailing slash)
curl -i -H "Origin: https://daily-rainfall-forecasting-system.22520630.workers.dev" \
  https://api.your-domain.example/health
```
