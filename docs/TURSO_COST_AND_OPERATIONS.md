# Turso Storage Architecture, Parity & Cost Operations (`DS108`)

> **Generated:** `2026-11-21T09:05:16.185592Z` | **Active Plan:** `starter` | **Two-DB Boundary Enforced (Decision 1 Option A):** Production DB (`observations_vintage` + automated `POST_TEST` forecasts allowed via `issued_by='automated_pipeline'`; `no_manual_retrospective_backfill_in_prod=True`) in Turso; Demo DB (`demo_rainfall_vintage.db`, `4,270` `manual_retrospective_backfill` rows) strictly local.

## 1. Empirical Answers to the 4 Operational Questions

1. **Per-request `rows_read` before vs. after index-first rewrite (measured on 9,454 real production rows via Turso Hrana metadata):**
   - `GET /observations` (90-day slice): **`9,724` rows** (legacy full `v_observations_latest` view scan) -> **`192` rows** (indexed CTE range pushdown) (**98.03% reduction**).
   - `POST /predict` (35-day history + target check): **`47,377` rows** (legacy full view count + scan) -> **`107` rows** (bounded `LIMIT 35` + indexed range pushdown) (**99.77% reduction**).
   - `GET /health`, `GET /locations`, `GET /scores`: In-memory cache keyed by `(db_identifier, latest_run_id, max_forecast_rowid)` yields **0 extra DB row scans** across repeat requests until state changes.
   - `STORE_BACKEND="turso_replica"`: User HTTP requests hit the local SQLite replica with **0 remote Turso `rows_read` per request**, while the background thread syncs every `300s ± 30s`.

2. **Index maintenance impact on `rows_written` & Daily write load (`automated_pipeline.py`):**
   - Live Hrana 50-row insert comparison: unindexed table (`rows_written=100` = 1 table + 1 PK index per row) vs. 3 secondary indexes (`rows_written=250`, delta=`150` — empirically **+1 `rows_written` per secondary index per inserted row** in Turso/libSQL B-tree billing). Even counting index entries (~4–5 B-tree writes per logical row), daily write load (~1–7 `observations_vintage` + `1` `ingestion_runs` + ~`25` `forecasts_ledger` rows ≈ `33` logical rows/day ≈ `130–165` billed `rows_written/day`) stays well below Starter limits, with **0 `UPDATE`s** on `forecasts_ledger` and `SKIPPED` when no new complete anchor unlocks.

3. **Trigger body queries on `rows_read` & SQLite vs. Turso parity (verified on ephemeral live Turso DB `ds108-r11-probe-b7849b62`, then deleted):**
   - Live 50-row insert trigger isolation: `BEFORE INSERT WHEN EXISTS(...)` trigger probe added `rows_read` delta = `0` (unindexed=`0` vs triggered=`0` — indexed PK/composite lookup probes `O(1)` per row!).
   - `UPDATE` / `DELETE` immutability triggers blocked: `{'update_blocked': True, 'delete_blocked': True, 'delete_run_blocked': True}`
   - `PRAGMA foreign_keys = ON` enforced: `True`
   - Multi-statement atomic `execute_batch(..., atomic=True)` rollback on mid-batch error: `True` (0 leaked rows)
   - FastAPI `/health`, `/locations`, `/observations`, `/predict` numerical parity across `local`, `turso_remote`, `turso_replica`: `True`

4. **Monthly Capacity & Cost Envelope (with `2.0x` Safety Factor):**

| Traffic Scenario | `turso_remote` Reads/mo (2×) | `turso_replica` Reads/mo (2×) | Writes/mo (2×) | Sync/mo (Replica 2×) | Projected Storage | Fit vs. Plans |
|---|---:|---:|---:|---:|---:|---|
| S1: Idle Dev / Demo (5 req/day) | 60,600 (0.0121%) | 84,120 (0.016824%) | 2,180 | 6.152 MB | 8.39 MB | Starter ($0) |
| S2: Low Team / Class (100 req/day) | 927,000 (0.1854%) | 84,120 (0.016824%) | 2,180 | 6.152 MB | 8.39 MB | Starter ($0) |
| S3: Medium Public Console (2,000 req/day) | 18,255,000 (3.651%) | 84,120 (0.016824%) | 2,180 | 6.152 MB | 8.39 MB | Starter ($0) |
| S4: High Burst / Evaluation (20,000 req/day) | 182,415,000 (36.483%) | 84,120 (0.016824%) | 2,180 | 6.152 MB | 8.39 MB | Starter ($0) |
| S5: Peak NRT-to-Final Revision Wave (2,000 req/day, 5x) | 45,825,000 (9.165%) | 210,300 (0.04206%) | 30,200 | 92.285 MB | 15.54 MB | Starter ($0) |

### Official Turso Plan Quotas & Sync Bandwidth Clarification (accessed 2026-11-20)
- **Starter (Free — USD 0/mo):** 500M `rows_read`/mo, 10M `rows_written`/mo, 5 GB storage, **3 GB embedded `Syncs` bandwidth/mo** (counted separately from `rows_read`!), up to 100 databases ([Turso Pricing](https://turso.tech/pricing), [Turso Embedded Replicas Docs](https://docs.turso.tech/features/embedded-replicas/introduction)).
- **Developer (USD 4.99/mo billed yearly):** 2.5B `rows_read`/mo, 25M `rows_written`/mo, 9 GB storage, **10 GB `Syncs` bandwidth/mo**, unlimited databases ([Turso Pricing](https://turso.tech/pricing)).
- **Quota Warning & Peak Wave (`S5`):** Runtime triggers `turso_quota_warning = True` when monthly consumption reaches `>= 70%` (`350M` reads/mo on Starter). Even under **S5 (Peak NRT-to-Final Revision Wave, 5× multiplier)**, `turso_replica` keeps `rows_read` and `Syncs` bandwidth strictly within Starter free caps; HTTP `402`/`403`/`429` `BLOCKED` errors gracefully fall back to local SQLite (`status="degraded"`, `degraded_reason="TURSO_REMOTE_BLOCKED_FALLBACK_LOCAL"`).
- **Python Client Selection Proof:** See [`reports/storage_audit/turso_python_client_wheel_logs.json`](../reports/storage_audit/turso_python_client_wheel_logs.json) — retains zero-native-dependency Hrana HTTP v2 (`requests`) over `libsql-experimental` (no prebuilt Windows binary wheel).

## 2. Production Migration Verification (DDL V3.4, Decision 1 Option A)
- **Locations:** `1` rows (`parity_ok=True`, `sha256=427e78eba7cbea79...`)
- **Ingestion Runs:** `4` rows (`parity_ok=True`, `sha256=df65ea651dbda478...`)
- **Observations Vintage:** `9,454` rows (`parity_ok=True`, `sha256=c9e874b971c38fca...`)
- **Forecasts Ledger in Turso Production DB:** `0` `automated_pipeline` rows, `0` `manual_retrospective_backfill` rows (`no_manual_retrospective_backfill_in_prod=True`)
