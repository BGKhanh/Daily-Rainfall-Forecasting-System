# Mini-Task Round 14 Summary — Stage 1 H=11 Horizon Extension Feasibility Audit & Hard-Stop Gate Report

- **Execution Policy**: Strict Two-Stage Hard Gate (`Step 0 -> Stage 1 Benchmark & Gate -> Conditional Step 2`).
- **Final Stage 1 Gate Decision**: **`stop_and_report_triggered = True`** (`winning_H_star = None` across eligible targets $H^* \in \{11, 10\}$; diagnostics $H^* \in \{9, 8\}$ also failed `Clause B2` & `Clause B3`).
- **Step 2 Status**: **NOT EXECUTED (Hard Stop Triggered)**. Serving remains locked at **`H = 7`**, Docker tag **`1.7.0-serving`**, SQLite **`DDL V3.5`**, MLflow `v1` aliases, and zero edits were made to `artifacts/models/`, `src/`, `frontend/`, `docs/`, `scripts/`, or `forecasts_ledger`.

---

## Step 0 — Verified GitHub Actions CI for Round 13-Closeout (`31a042b`)

Queried via GitHub Actions REST API for `head_sha = 31a042b69ecda71ce7d1c91c6d466bd7d7580b44` on `BGKhanh/Daily-Rainfall-Forecasting-System`:

1. **`Backend Serving CI`** — Run ID [`45044546308`](https://github.com/BGKhanh/Daily-Rainfall-Forecasting-System/actions/runs/45044546308) — `status = completed`, `conclusion = success`
   - `Tier 1 CI — Ruff Lint, Serving Unit/Golden Parity & SQLite DDL V3 Trigger Tests`: `completed / success`
   - `DagsHub MLflow Registry Checksum Audit (Optional)`: `completed / success`
   - `Tier 2 CI — Docker Container Build, Linux Parity & RSS Smoke Check`: `completed / success`
2. **`Scheduled NASA POWER Ingest & Forecast Issuance`** — Run ID [`45091703773`](https://github.com/BGKhanh/Daily-Rainfall-Forecasting-System/actions/runs/45091703773) — `status = completed`, `conclusion = success`
   - `2-Tier NASA POWER Scan & Latest-Only Forecast Issuance`: `completed / success`

---

## Stage 1 — Strict `H=11` Feasibility Audit (`clean` & `P-c`)

- **Script & Tests**: [`research/horizon_extension/run_h11_feasibility_audit.py`](../../research/horizon_extension/run_h11_feasibility_audit.py) | [`research/horizon_extension/test_h11_feasibility_audit.py`](../../research/horizon_extension/test_h11_feasibility_audit.py)
- **Outputs**:
  - [`research/horizon_extension/h11_feasibility_audit.json`](../../research/horizon_extension/h11_feasibility_audit.json) (`28.529s` runtime)
  - [`research/horizon_extension/README.md`](../../research/horizon_extension/README.md) (full per-step `h=1..11` and cross-model `clean` vs `P-c` tables)
- **Strict 3-Split Purge (`horizon = 11`, continuous 9,251-day series verified via `check_continuity(..., raise_on_gap=True)`)**:
  - `Dev Train` (`2001-01-01..2018-04-30`, `6,329` days): **`6,289` origins** (`2001-01-30..2018-04-19`).
  - `Validation` (`2018-05-01..2020-04-30`, `731` days, `context_df=dev_train.tail(30)`): **`720` origins** (`2018-05-01..2020-04-19`).
  - `Test` (`2020-05-01..2026-04-30`, `2,191` days, `context_df=val.tail(30)`): **`2,180` origins** (`2020-05-01..2026-04-19`).
- **Evaluation Protocol**:
  - Canonical model factories (`_create_sc35_model`, `_create_lgbm_config_a_model` from `scripts/mlops/package_models.py`) fit strictly on `6,289` clean `Dev Train` origins: `SC-35` (XGBoost Tweedie `F3`), `SC-31` (LightGBM Tweedie `F3` Config A), `SC-30` (LightGBM Tweedie `F2` Config A), `BASELINE_CLIMATOLOGY_DOY` (DOY `1..366` `PRECTOTCORR` mean fit strictly on `Dev Train`), `BASELINE_PERSISTENCE` (`PRECTOTCORR[t]`).
  - Evaluated under `clean` (35 observed vars) and `P-c` (`solar_doy_climatology_dev_train.json`, SHA-256 `6d16431eeee0c44f55169c909d94bab3210e78dbe4a88a9435dc3c9dea6d9872`, transform parity diff `0.0 < 1e-12`).
  - Metrics: `MAE`, `RMSE`, `MSE`, `MASE` (`scale = 3.7015424361567635`), `P90-Tail MAE` (`>= 11.086 mm`), `Fidelity-MAE`, diffs vs `BASELINE_CLIMATOLOGY_DOY`, and **Moving Block Bootstrap 95% CIs** (`block_length = 11`, `n_bootstrap = 2000`, `seed = 42`). Purged-metrics firewall (`assert_no_purged_metrics`) passed across every flattened metric window.

---

### Pre-Declared Go/No-Go Decision Gate Results (`Val P-c n = 720`, `Test P-c n = 2,180`)

#### 1. Clause A — Non-Regression on `h=1..7` vs Packaged `H=7` Models (`|ΔMAE| ≤ 0.015 mm`): **PASSED**

| Model | `H=11` Val `P-c` `h1_to_7` MAE | Packaged `H=7` Val `P-c` `h1_to_7` MAE | Signed Diff (mm) | `|Diff|` (mm) | Passed (`≤ 0.015`) |
|---|---|---|---|---|---|
| `SC-35` | `3.174577` | `3.182903` | `-0.008326` | `0.008326` | **True** |
| `SC-31` | `3.203628` | `3.203757` | `-0.000129` | `0.000129` | **True** |
| `SC-30` | `3.157543` | `3.165047` | `-0.007504` | `0.007504` | **True** |

#### 2. Clauses B1, B2, B3 (`SC-35` vs `BASELINE_CLIMATOLOGY_DOY` under `P-c`)

| Candidate `H*` | Ext Slice | Role | Clause B1 (Val MAE & 95% CI) | Clause B2 (Val ΔRMSE, ΔP90-Tail) | Clause B3 (Test ΔMAE, ΔRMSE, ΔP90-Tail) | Overall |
|---|---|---|---|---|---|---|
| `H*=11` | `h8_to_11` | Eligible | **True** (`-0.4272 [-0.5769, -0.2675]`; all steps `< 0`) | **False** (`ΔRMSE=-0.2733`, **`ΔP90=+0.3357`**) | **False** (`ΔMAE=-0.2010 [-0.2680, -0.1118]`, `ΔRMSE=-0.0566`, **`ΔP90=+0.5673`**) | **FAIL** |
| `H*=10` | `h8_to_10` | Eligible | **True** (`-0.4104 [-0.5663, -0.2457]`; all steps `< 0`) | **False** (`ΔRMSE=-0.2668`, **`ΔP90=+0.3220`**) | **False** (`ΔMAE=-0.1984 [-0.2662, -0.1078]`, `ΔRMSE=-0.0576`, **`ΔP90=+0.5610`**) | **FAIL** |
| `H*=9` | `h8_to_9` | Diagnostic | **True** (`-0.4047 [-0.5610, -0.2375]`) | **False** (`ΔRMSE=-0.2754`, **`ΔP90=+0.2645`**) | **False** (`ΔMAE=-0.1992 [-0.2703, -0.1083]`, `ΔRMSE=-0.0733`, **`ΔP90=+0.4901`**) | **FAIL** |
| `H*=8` | `h8` | Diagnostic | **True** (`-0.4011 [-0.5865, -0.2164]`) | **False** (`ΔRMSE=-0.2753`, **`ΔP90=+0.2670`**) | **False** (`ΔMAE=-0.1865 [-0.2601, -0.0934]`, `ΔRMSE=-0.0875`, **`ΔP90=+0.4595`**) | **FAIL** |

---

### Root-Cause Analysis of `Clause B2` & `Clause B3` Failure

1. **Overall `MAE` and `RMSE` remain positive skill through `h=11`, but Heavy-Rain Tail (`>= 11.086 mm`) breaks cleanly past `h=1..7`**:
   - On `SC-35` under `P-c`, overall `MAE` beats `BASELINE_CLIMATOLOGY_DOY` at every single horizon `h=1..11` on both `Val` (`-0.4011` to `-0.7978 mm`) and `Test` (`-0.1865` to `-0.4986 mm`), with `95%` MBB CI upper bounds strictly `< 0.0` on every extended slice.
   - However, on extreme rain events (`PRECTOTCORR >= 11.086 mm`), `SC-35` beats `BASELINE_CLIMATOLOGY_DOY` only at `h=1` (`ΔP90 = -1.4009 mm` Val, `-1.4789 mm` Test), tracks near parity across `h1_to_7` on Val (`-0.0706 mm`, already `+0.1013 mm` on Test), and degrades across **`h=8..11`**:
     - **Val `P-c`**: `h8` `+0.2670`, `h9` `+0.2621`, `h10` `+0.4371`, `h11` `+0.3766 mm` (`h8_to_11 = +0.3357 mm`).
     - **Test `P-c`**: `h8` `+0.4595`, `h9` `+0.5207`, `h10` `+0.7029`, `h11` `+0.5862 mm` (`h8_to_11 = +0.5673 mm`).
2. **Intrinsic Predictability Limit, NOT a `P-c` Artifact**:
   - Identical `P90-Tail MAE` degradation occurs under **`clean` 35-variable** evaluation (`SC-35` `h8_to_11` `clean`: `ΔP90 = +0.3381 mm` on Val, `+0.5787 mm` on Test), and across both `SC-31` (`+0.5463` Val / `+0.6976` Test under `P-c`) and `SC-30` (`+0.5938` Val / `+0.6498` Test under `P-c`).
   - Direct Tweedie (`power = 1.5`) point regression regresses toward conservative conditional expectations once multi-day meteorological autocorrelation vanishes past ~7 days, under-predicting sudden heavy convective days (`>= 11.086 mm`) compared to the seasonal DOY climatological mean.

---

### Full `SC-35` Slice & Extended Cross-Model Tables (`P-c` & `clean`, 95% MBB CIs)

#### `SC-35` Validation (`n = 720`, `P-c`) — All Slices

| Slice | SC-35 MAE [95% CI] | Clim DOY MAE | ΔMAE vs Clim [95% CI] | ΔRMSE vs Clim [95% CI] | ΔP90-Tail vs Clim | Fidelity-MAE |
|---|---|---|---|---|---|---|
| `h1_to_7` | `3.1746 [2.5598, 3.8826]` | `3.7158` | `-0.5412 [-0.7132, -0.3840]` | `-0.3882 [-0.6152, -0.2006]` | `-0.0706` | `0.1125` |
| `h4_to_10` | `3.2670 [2.6306, 3.9597]` | `3.7172` | `-0.4502 [-0.6049, -0.3011]` | `-0.2932 [-0.5068, -0.1088]` | `+0.2676` | `0.1068` |
| `h5_to_11` | `3.2670 [2.6276, 3.9509]` | `3.7170` | `-0.4500 [-0.5966, -0.3017]` | `-0.2902 [-0.4958, -0.1045]` | `+0.2934` | `0.1018` |
| `h8_to_11` | `3.2912 [2.6009, 3.9886]` | `3.7184` | `-0.4272 [-0.5769, -0.2675]` | `-0.2733 [-0.4863, -0.0714]` | `+0.3357` | `0.1064` |
| `h8_to_10` | `3.3090 [2.6225, 4.0106]` | `3.7194` | `-0.4104 [-0.5663, -0.2457]` | `-0.2668 [-0.4817, -0.0663]` | `+0.3220` | `0.1080` |
| `h8_to_9` | `3.3152 [2.6268, 4.0201]` | `3.7199` | `-0.4047 [-0.5610, -0.2375]` | `-0.2754 [-0.4925, -0.0707]` | `+0.2645` | `0.1126` |
| `h1_to_11` | `3.2170 [2.6009, 3.8884]` | `3.7167` | `-0.4998 [-0.6460, -0.3528]` | `-0.3462 [-0.5451, -0.1700]` | `+0.0771` | `0.1103` |

#### `SC-35` Held-Out Test (`n = 2,180`, `P-c`) — All Slices

| Slice | SC-35 MAE [95% CI] | Clim DOY MAE | ΔMAE vs Clim [95% CI] | ΔRMSE vs Clim [95% CI] | ΔP90-Tail vs Clim | Fidelity-MAE |
|---|---|---|---|---|---|---|
| `h1_to_7` | `3.7337 [3.2625, 4.2633]` | `3.9845` | `-0.2508 [-0.3170, -0.1736]` | `-0.1594 [-0.2524, -0.0716]` | `+0.1013` | `0.1239` |
| `h4_to_10` | `3.7787 [3.2989, 4.3088]` | `3.9829` | `-0.2042 [-0.2673, -0.1216]` | `-0.0739 [-0.1684, +0.0209]` | `+0.4573` | `0.1148` |
| `h5_to_11` | `3.7777 [3.2914, 4.2969]` | `3.9818` | `-0.2041 [-0.2670, -0.1203]` | `-0.0696 [-0.1614, +0.0274]` | `+0.4828` | `0.1144` |
| `h8_to_11` | `3.7797 [3.2777, 4.3145]` | `3.9807` | `-0.2010 [-0.2680, -0.1118]` | `-0.0566 [-0.1630, +0.0522]` | `+0.5673` | `0.1225` |
| `h8_to_10` | `3.7833 [3.2753, 4.3239]` | `3.9817` | `-0.1984 [-0.2662, -0.1078]` | `-0.0576 [-0.1650, +0.0536]` | `+0.5610` | `0.1213` |
| `h8_to_9` | `3.7824 [3.2739, 4.3374]` | `3.9817` | `-0.1992 [-0.2703, -0.1083]` | `-0.0733 [-0.1833, +0.0389]` | `+0.4901` | `0.1282` |
| `h1_to_11` | `3.7504 [3.2836, 4.2496]` | `3.9831` | `-0.2327 [-0.2916, -0.1589]` | `-0.1219 [-0.2076, -0.0382]` | `+0.2708` | `0.1234` |

*(Full cross-model `clean` vs `P-c` tables and per-step `h=1..11` breakdowns are published in [`research/horizon_extension/README.md`](../../research/horizon_extension/README.md) and [`research/horizon_extension/h11_feasibility_audit.json`](../../research/horizon_extension/h11_feasibility_audit.json).)*

---

## Recommendations

1. **Keep Operational Serving Locked at `H = 7` (`1.7.0-serving`, `DDL V3.5`)**: Extending `H` to `10` or `11` with frozen Tweedie direct regressors degrades heavy-rain (`>= 11.086 mm`) tail accuracy across both Validation and Test, even though aggregate `MAE`/`RMSE` remain below Climatology DOY.
2. **Future Storage Optimization (`DDL V3.6` `input_window_manifests`)**: Can be pursued independently in a storage-only release without changing model horizon `H = 7`, since `input_vintage_manifest_json` accounts for `~76%` of `forecasts_ledger` bytes (`24.56 MB`).
3. **Future Medium-Range Research**: If days `+8..+11` are revisited later, evaluate hurdle/quantile or ensemble synoptic post-processing rather than pure Tweedie point regression, and retain the strict 4-clause Val+Test gate.

---

## Verification & Reviews (Stage 1 Only)

- **Ruff Check**: `ruff check research/horizon_extension/` — `All checks passed!`
- **Stage 1 Unit & Contract Tests**: `pytest research/horizon_extension/test_h11_feasibility_audit.py` — **`8 passed`** (`39.14s`)
- **Core Serving Regression Check**: Not modified (`src/`, `artifacts/models/`, `frontend/`, `tests/` untouched).
- **Subagent Reviews**: Executed independent Standards (`48c501e8`), Spec Conformance (`9814c66c`), and Security/Differential (`32ee2b40`) reviews on Stage 1 diff; resolved all actionable findings prior to atomic commits and push.
