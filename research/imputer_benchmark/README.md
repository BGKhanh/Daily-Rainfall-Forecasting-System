# Standalone Exogenous Imputation Research & CERES-8 Scientific Audit

This directory houses the self-contained exogenous imputation module, deterministic 4-tier benchmark suite, interactive notebook, unit tests, and Round 12 full-horizon CERES-8 vs CERES-11 scientific audit.

> **Isolation Boundary:** Nothing in `research/imputer_benchmark/` alters the production `POST /predict` path or is executed in Tier 1/Tier 2 GitHub Actions CI.

## Inventory

- `exogenous_imputer.py` — Mode 0/1/2/3 causal exogenous imputer (`ClimatologyLookupTable`, `HistoricalAnalogStore`, `ExogenousImputerWrapper`).
- `imputation_baselines.py` — 4-tier Dev_Train-only baseline registry.
- `imputation_benchmark.py` / `imputation_reporting.py` / `run_imputation_benchmark.py` — 48-anchor, 72-config (position × L × profile) benchmark & Wilcoxon/Holm-Bonferroni gatechecks.
- `build_inference_notebook.py` / `imputation_based_inference.ipynb` — Executed exploratory & diagnostic benchmark notebook.
- `run_ceres8_vs_11_audit.py` -> `ceres8_scientific_audit_2184.json` — Round 12 full `724` Val & `2,184` Test audit comparing **Profile A-8** (8 CERES-only vars lagging ~80d) vs **Profile A-11** (all 11 SYN1deg vars).

## Key Findings (Full 724 Val & 2,184 Test Purged Origins, L=80 Trailing)

1. **Feature-Scope Verification (Question b):** `radiation_derived_lag_or_rolling_cols == []` — ZERO lag/rolling features exist on any of the 11 radiation variables; all 11 enter `SC-35` / `SC-31` / `SC-30` strictly as origin-day `F0` values.
2. **Intrinsic `vr_scen` vs `vr_mean` (Question b):** Look-up across 24 Test anchors (`L=80`) confirms `vr_scen` preserves realistic within-scenario synoptic variability whereas `vr_mean` collapses due to Double-Penalty averaging across `K=10` analog paths:
   - **A-8**: `custom_mode2_k10` `vr_scen = 1.0223`, `vr_mean = 0.3467` (vs `doy_climatology` `vr_mean = 0.2002`).
   - **A-11**: `custom_mode2_k10` `vr_scen = 1.0326`, `vr_mean = 0.4262` (vs `doy_climatology` `vr_mean = 0.2640`).

3. **Extrinsic Model Performance on Profile A-8 (`8 CERES-only` outage, Test n=2,184 origins):**

| Model | Method | Val MAE (724) | Val Delta MAE | Test MAE (2,184) | Test MSE | Test RMSE | Test P90-Tail MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| SC-35 | `clean` | 3.1683 | +0.0000 | 3.7333 | 65.5809 | 8.0982 | 15.5139 |
| SC-35 | `raw_nan` | 3.1605 | -0.0078 | 3.7565 | 69.0148 | 8.3075 | 16.5105 |
| SC-35 | `ffill_stale_80d` | 3.2208 | +0.0525 | 3.7422 | 66.0153 | 8.1250 | 15.5839 |
| SC-35 | `train_mean` | 3.1874 | +0.0190 | 3.7432 | 65.6640 | 8.1033 | 15.5437 |
| SC-35 | `doy_climatology` | 3.1711 | +0.0027 | 3.7338 | 65.5180 | 8.0943 | 15.4970 |
| SC-35 | `custom_mode1_ou` | 3.1711 | +0.0027 | 3.7338 | 65.5180 | 8.0943 | 15.4970 |
| SC-35 | `custom_hybrid_k10` | 3.1677 | -0.0006 | 3.7314 | 65.6126 | 8.1002 | 15.5430 |
| SC-35 | `custom_mode2_k10` | 3.1677 | -0.0006 | 3.7314 | 65.6126 | 8.1002 | 15.5430 |
| SC-35 | `retrained_no_ceres_27vars` | 3.1691 | +0.0008 | 3.7312 | 65.4680 | 8.0912 | 15.5223 |
| SC-31 | `clean` | 3.1939 | +0.0000 | 3.7246 | 65.9240 | 8.1194 | 15.6293 |
| SC-31 | `raw_nan` | 3.1312 | -0.0627 | 3.7379 | 69.7215 | 8.3499 | 16.6675 |
| SC-31 | `ffill_stale_80d` | 3.2484 | +0.0545 | 3.7415 | 66.2005 | 8.1364 | 15.6506 |
| SC-31 | `train_mean` | 3.2094 | +0.0155 | 3.7316 | 66.0505 | 8.1271 | 15.6773 |
| SC-31 | `doy_climatology` | 3.1921 | -0.0018 | 3.7247 | 65.8806 | 8.1167 | 15.6231 |
| SC-31 | `custom_mode1_ou` | 3.1921 | -0.0018 | 3.7247 | 65.8806 | 8.1167 | 15.6231 |
| SC-31 | `custom_hybrid_k10` | 3.1896 | -0.0043 | 3.7224 | 65.9091 | 8.1184 | 15.6495 |
| SC-31 | `custom_mode2_k10` | 3.1896 | -0.0043 | 3.7224 | 65.9091 | 8.1184 | 15.6495 |
| SC-31 | `retrained_no_ceres_27vars` | 3.1921 | -0.0019 | 3.7271 | 65.9788 | 8.1227 | 15.6577 |
| SC-30 | `clean` | 3.1485 | +0.0000 | 3.7501 | 65.8426 | 8.1143 | 15.6159 |
| SC-30 | `raw_nan` | 3.0992 | -0.0493 | 3.7405 | 69.4971 | 8.3365 | 16.6725 |
| SC-30 | `ffill_stale_80d` | 3.2048 | +0.0563 | 3.7722 | 66.2724 | 8.1408 | 15.6679 |
| SC-30 | `train_mean` | 3.1763 | +0.0278 | 3.7742 | 65.7411 | 8.1081 | 15.5974 |
| SC-30 | `doy_climatology` | 3.1543 | +0.0059 | 3.7509 | 65.7041 | 8.1058 | 15.5797 |
| SC-30 | `custom_mode1_ou` | 3.1543 | +0.0059 | 3.7509 | 65.7041 | 8.1058 | 15.5797 |
| SC-30 | `custom_hybrid_k10` | 3.1482 | -0.0002 | 3.7469 | 65.8261 | 8.1133 | 15.6383 |
| SC-30 | `custom_mode2_k10` | 3.1482 | -0.0002 | 3.7469 | 65.8261 | 8.1133 | 15.6383 |
| SC-30 | `retrained_no_ceres_27vars` | 3.1455 | -0.0030 | 3.7585 | 65.9188 | 8.1190 | 15.6244 |

4. **Profile A-11 vs A-8 `raw_nan` Degradation, Per-Horizon Breakdown & `mt_q1` Reconciliation:**

- **SC-35**: A-8 `raw_nan` delta_MAE = `+0.0232` mm (h1..h7 MAE: `[3.498538, 3.768265, 3.817633, 3.792492, 3.773522, 3.796472, 3.848623]`), vs A-11 `raw_nan` delta_MAE = `+0.0625` mm (h1..h7 MAE: `[3.528099, 3.801279, 3.856687, 3.840232, 3.829215, 3.847954, 3.867439]`).
- **SC-31**: A-8 `raw_nan` delta_MAE = `+0.0133` mm (h1..h7 MAE: `[3.457325, 3.768447, 3.8297, 3.807364, 3.751614, 3.757963, 3.792972]`), vs A-11 `raw_nan` delta_MAE = `+0.1768` mm (h1..h7 MAE: `[3.44524, 3.865992, 3.93134, 4.02324, 3.9648, 4.030123, 4.049645]`).
- **SC-30**: A-8 `raw_nan` delta_MAE = `-0.0096` mm (h1..h7 MAE: `[3.460992, 3.784966, 3.821635, 3.817093, 3.784548, 3.750779, 3.76323]`), vs A-11 `raw_nan` delta_MAE = `+0.3485` mm (h1..h7 MAE: `[3.486465, 4.002565, 4.272009, 4.264061, 4.303718, 4.213398, 4.148088]`).
- **Rule 2 (+/- 0.020 mm Occam's Razor Tie-Breaker)**: Prefers simpler causal candidates (`doy_climatology` -> `train_mean` -> `ffill_stale_80d`) when within `0.020 mm` of minimum Validation MAE, preventing over-complex fallbacks from winning on noise.
- **`mt_q1.json` Reconciliation**: Clean Test MAE reproduced exact parity against canonical `SC-35` / `SC-31` / `SC-30` evaluations in `reports/storage_audit/mt_q1.json`.

## Next-Phase Production Integration Plan

- **Round 12 Status:** Operational `POST /predict` and production serving artifacts remain untouched.
- **Recommended Next Step:** Run formal MLflow model governance evaluation to either (a) promote a FlashFlux-only 27-var (`retrained_no_ceres_27vars`) model contender via `@shadow_val_champ` -> `@shadow_test_contender` -> `@serving`, eliminating the ~80-day CERES SYN1deg lag dependency altogether, or (b) pair frozen `SC-31`/`SC-30` with an explicit DOY-climatology / Hybrid imputer fallback path behind an audited request flag.
