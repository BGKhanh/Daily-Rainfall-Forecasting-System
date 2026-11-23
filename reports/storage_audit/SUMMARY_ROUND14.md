# Mini-Task Round 14 Summary — Stage 1 H=11 Feasibility Audit, Canonical `70`/`65` Contract, Closeout Diagnostics & Option A Freeze

- **Execution Policy**: Strict Two-Stage Hard Gate (`Step 0 -> Stage 1 Benchmark & Gate -> Conditional Step 2`).
- **Final Stage 1 Gate Decision**: **`stop_and_report_triggered = True`** (`winning_H_star = None` across eligible targets $H^* \in \{11, 10\}$; diagnostics $H^* \in \{9, 8\}$ also failed `Clause B2` & `Clause B3`).
- **Round 14 Closeout Decision (Option A Locked)**:
  - Keep production serving strictly locked at **`H = 7`**, Docker tag **`1.7.0-serving`**, SQLite **`DDL V3.5`**, and MLflow `v1` aliases.
  - **Stage 2 Release is permanently closed**; **`input_window_manifests` (`DDL V3.6`)** is deferred to multi-province scaling (`39.52 MB` current DB size, `< 1%` of Turso Free 5 GB quota).
  - Zero edits made to `src/`, `artifacts/models/`, `frontend/`, `docs/`, or `data/` (`forecasts_ledger`).

---

## Step 0 — Verified GitHub Actions CI (`31a042b`, `6e3e283` & Closeout `e95b6d1`)

1. **Round 13-Closeout Head (`31a042b69ecda71ce7d1c91c6d466bd7d7580b44`)**:
   - **`Backend Serving CI`** — Run ID [`45044546308`](https://github.com/BGKhanh/Daily-Rainfall-Forecasting-System/actions/runs/45044546308) — `completed / success` (`Tier 1 CI`, `DagsHub MLflow Registry Checksum Audit (Optional)`, `Tier 2 CI — Docker Build & RSS Smoke Check`).
   - **`Scheduled NASA POWER Ingest & Forecast Issuance`** — Run ID [`45091703773`](https://github.com/BGKhanh/Daily-Rainfall-Forecasting-System/actions/runs/45091703773) — `completed / success`.
2. **Round 14 Stage 1 Push (`6e3e28351af2e6b85be226c8b4dd2b20939976cb`)**:
   - **`Backend Serving CI`** — Run ID [`45114930563`](https://github.com/BGKhanh/Daily-Rainfall-Forecasting-System/actions/runs/45114930563) — `completed / success`.
3. **Round 14-Closeout Push (`71ec7a0` / `e95b6d1a830785df61341e9dc5258075500f5801`)**:
   - **`Backend Serving CI`** — Run ID [`45119684631`](https://github.com/BGKhanh/Daily-Rainfall-Forecasting-System/actions/runs/45119684631) — `completed / success` (`Tier 1 CI`, `DagsHub MLflow Registry Checksum Audit (Optional)`, `Tier 2 CI`).

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
  - Canonical model factories (`_create_sc35_model`, `_create_lgbm_config_a_model` from `scripts/mlops/package_models.py`) fit strictly on `6,289` clean `Dev Train` origins: `SC-35`, `SC-31`, `SC-30`, `BASELINE_CLIMATOLOGY_DOY` (DOY `1..366` `PRECTOTCORR` mean fit strictly on `Dev Train`), `BASELINE_PERSISTENCE` (`PRECTOTCORR[t]`).
  - Evaluated under `clean` (35 observed vars) and `P-c` (`solar_doy_climatology_dev_train.json`, SHA-256 `6d16431eeee0c44f55169c909d94bab3210e78dbe4a88a9435dc3c9dea6d9872`, transform parity diff `0.0 < 1e-12`).
  - Metrics: `MAE`, `RMSE`, `MSE`, `MASE` (`scale = 3.7015424361567635`), `P90-Tail MAE` (`>= 11.086 mm`), `Fidelity-MAE`, diffs vs `BASELINE_CLIMATOLOGY_DOY`, and **Moving Block Bootstrap 95% CIs** (`block_length = 11`, `n_bootstrap = 2000`, `seed = 42`). Purged-metrics firewall (`assert_no_purged_metrics`) passed across every flattened metric window.

---

### Canonical Feature Matrix Contract (`70` Columns `F3` / `65` Columns `F2`)

Verified directly in [`run_h11_feasibility_audit.py`](../../research/horizon_extension/run_h11_feasibility_audit.py) (`len(f3_cols) == 70`, `len(f2_cols) == 65`, `f3_cols[:65] == f2_cols`) and against [`artifacts/feature_names.json`](../../artifacts/feature_names.json) (`all_70_features`, exact ordered equality `fn == f3_cols`) plus packaged `FeatureBuilder` and `.joblib` estimators (`n_features_in_`):

| Model | Feature Set | `X_train.shape` | `X_val.shape` | `X_test.shape` | `len(feature_names)` |
|---|---|---|---|---|---|
| `SC-35` (XGBoost Tweedie) | `F3` (`f3_idx = list(range(70))`) | `(6289, 70)` | `(720, 70)` | `(2180, 70)` | `70` |
| `SC-31` (LightGBM Tweedie Config A) | `F3` (`f3_idx = list(range(70))`) | `(6289, 70)` | `(720, 70)` | `(2180, 70)` | `70` |
| `SC-30` (LightGBM Tweedie Config A) | `F2` (`f2_idx = list(range(65))`, first 65 of `F3`) | `(6289, 65)` | `(720, 65)` | `(2180, 65)` | `65` |

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

### Scientific Diagnostics & Roadmap v2.0 Conclusion (Round 14 Closeout — Option A Locked)

1. **Diagnosis 1 — Positive `ΔP90-Tail MAE` on Test emerges immediately at `h=2`, not only `h=8`**:
   - On **Held-Out Test (`n = 2,180`, `P-c`)**, `SC-35` outperforms `BASELINE_CLIMATOLOGY_DOY` on extreme rainfall (`PRECTOTCORR >= 11.086 mm`) strictly at **`h=1` (`-1.4789 mm`)**. Immediately from **`h=2` to `h=7`**, `ΔP90-Tail` turns continuously positive (`h=2: +0.2401`, `h=3: +0.4301`, `h=4: +0.4071`, `h=5: +0.3988`, `h=6: +0.3379`, `h=7: +0.3740 mm`), making `h1_to_7` on Test **`+0.1013 mm`** before expanding to `+0.4595..+0.7029 mm` across `h=8..11` (`h8_to_11 = +0.5673 mm`).
   - On **Validation (`n = 720`, `P-c`)**, `h1_to_7` stays marginally negative (`-0.0706 mm`) solely because the **`h=1` pull (`-1.4009 mm`)** offsets `h=2..7` (`h=2: +0.0170`, `h=3: -0.0175`, `h=4: +0.1962`, `h=5: +0.3549`, `h=6: +0.1268`, `h=7: +0.2291 mm`). Every horizon window excluding `h=1` (`h4_to_10`, `h5_to_11`, `h8_to_11`) yields positive `ΔP90-Tail` on both Validation (`+0.2676..+0.3357 mm`) and Test (`+0.4573..+0.5673 mm`). Identical degradation under **`clean` 35-var** inputs (`+0.3381 mm` Val, `+0.5787 mm` Test on `h8_to_11`) confirms an intrinsic medium-range predictability barrier rather than a `P-c` imputation artifact.
2. **Diagnosis 2 — Why `ΔMAE` flattens from `h=2` to `h=11` (`~ -0.20 mm`), while `ΔRMSE` captures convective skill decay**:
   - `BASELINE_CLIMATOLOGY_DOY` predicts calendar-day arithmetic means ($\mathbb{E}[Y \mid \text{DOY}]$). On right-skewed tropical rainfall with `47.56%–56.60%` sub-threshold dry days (`PRECTOTCORR < 1.0 mm`; exact zero on `22.77%–27.32%` of days), the arithmetic mean is $L_2$-optimal (`RMSE`) but suboptimal for $L_1$ (`MAE`), whose optimizer is the lower conditional median. Consequently, Tweedie (`p = 1.5`) point predictions shrunk toward conservative conditional expectations automatically retain a structural $L_1$ advantage on dry/light-rain days, flattening Test `ΔMAE` from **`-0.2086 mm` (`h=2`)** to **`-0.2087 mm` (`h=11`)**.
   - Conversely, **`RMSE` ($L_2$)** faithfully exposes convective skill decay: on Held-Out Test (`P-c`), `SC-35` `ΔRMSE vs Climatology DOY` falls from **`-0.5741 mm` at `h=1`** (`95% CI [-0.7172, -0.4533]`) to **`-0.0819 mm` at `h=7`** (`95% CI [-0.1755, +0.0233]`, crossing `0` from `h=3` onward) and **`-0.0262 mm` at `h=10`**, leaving **`h8_to_11` at `-0.0566 mm`** with **`95% MBB CI [-0.1630, +0.0522]` crossing `0`** (statistically non-significant).
3. **Roadmap v2.0 & Option A Closeout Decision**:
   - Overcoming the **3–4 true future days** (`effective_future_days`) limit imposed by GMAO GEOS-IT observation latency requires **NWP numerical weather prediction future covariates (`t+1..t+7`)** in v2.0 rather than stretching historical point-observation regression past day 7.
   - **Option A Locked**: Operational serving stays at **`H = 7`, Docker `1.7.0-serving`, SQLite `DDL V3.5`, MLflow `v1` aliases**; Stage 2 is closed; **`input_window_manifests` (`DDL V3.6`)** is deferred to multi-province expansion (current production DB `39.52 MB`, `< 1%` of Turso Free 5 GB quota).

---

### Full `SC-35` Slice Tables (`Val n = 720` & `Test n = 2,180`, `P-c`, 95% MBB CIs)

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

## Verification & Reviews

- **Ruff Check**: `ruff check research/horizon_extension/` — `All checks passed!`
- **Stage 1 & Closeout Contract Tests**: `pytest research/horizon_extension/test_h11_feasibility_audit.py` — **`9 passed`**
- **Core Serving Protection**: Zero changes in `src/`, `artifacts/models/`, `frontend/`, `docs/`, `data/`.
- **Round 14-Closeout Commits & CI Verification**:
  - `71ec7a0` (`docs(research): document 70/65 feature contract and Stage 1 closeout diagnostics`) + `e95b6d1` (`docs(reports): finalize SUMMARY_ROUND14 with 70/65 contract and Option A diagnostics`)
  - **`Backend Serving CI`** Run ID [`45119684631`](https://github.com/BGKhanh/Daily-Rainfall-Forecasting-System/actions/runs/45119684631) — `completed / success`.
