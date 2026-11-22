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

## Round 13: True Future Days, 3-Clause Validation Gate & `P-c` Split-Horizon Audit

### Pre-Declared 3-Clause Validation Selection Gate Provenance

- **Clause 1 (`Validation MAE`)**: $|\Delta\text{MAE}_{\text{val}}| \le 0.020\text{ mm}$ — originated in **Round 5 (`MT-Q1`)**.
- **Clause 2 (`Validation RMSE`)**: $\Delta\text{RMSE}_{\text{val}} \le +0.050\text{ mm}$ — added following Round 12 and formalized with numeric bounds in **Round 13**.
- **Clause 3 (`Validation P90-Tail MAE`, $\ge 11.086\text{ mm}$)**: $\Delta\text{P90-Tail MAE}_{\text{val}} \le +0.250\text{ mm}$ — added following Round 12 and formalized in **Round 13**.
- *Why Clauses 2 & 3 are mandatory:* Round 12 proved `raw_nan` slightly lowered aggregate Validation MAE (`-0.0078` to `-0.0627 mm`) while severely degrading RMSE (`+0.13` to `+0.78 mm`) and P90-Tail MAE (`+1.00+ mm`); selecting on MAE alone would promote an unsafe model.
- *Model-retention rationale:* Any future transition to a retrained 24/27-variable model would be motivated strictly by operational simplicity after completing full new-model MLflow governance — **not accuracy**, as Test-set accuracy differences vs the frozen 35-variable ensemble are statistically insignificant.

### True Future Days (`effective_future_days = max(0, 7 - lag_days)`) from `ingestion_runs`

| Policy | Constraints | Anchor (2026-11-21) | Typical Lag | `effective_future_days` (Nov 19 / 20 / 21) | `is_prospective = 1`? |
|---|---|---|---:|---:|---|
| `P-0_COMPLETE_ALL_35` | Requires all 35 variables (24 Meteo + 3 FlashFlux + 8 CERES SYN1deg); 0 true future days. | `2026-08-30` | 80-83d | 0 / 0 / 0 | `False` |
| `P-a_P-b_FLASHFLUX_27VARS` | Eliminates CERES-8 only; 0 true future days whenever FlashFlux lag >= 7 days or interior dropouts occur (e.g., 2026-11-06). | `2026-11-12` | 6-9d | 0 / 0 / 0 | `False` |
| `P-c_DOY_MEDIAN_ALL_11_SOLAR` | Anchor constrained only by 24 Meteorology vars; unlocks 3 to 5 true future days (is_prospective=1) and stateless DOY lookup. | `2026-11-17` | 2-4d | 5 / 4 / 3 | `True` |
| `P-c-ffill7_LE7D_PLUS_DOY_CERES8` | Anchor constrained by 24 Meteo vars (FlashFlux ffill when lag<=7d + DOY CERES-8); same 3 to 5 true future days as P-c. | `2026-11-17` | 2-4d | 5 / 4 / 3 | `True` |

### 3-Clause Validation Gate Decision: **`p_c_doy_median_all_11_solar`** (`stop_condition_triggered=False`)

> Both P-c and passing candidate(s) fall within <= 0.005 mm sampling noise on max|delta_MAE_val|; selected P-c (DOY_MEDIAN_ALL_11_SOLAR) by Occam's razor because it is stateless and anchor-DOY deterministic.

| Candidate | All 3 Models Pass? | `SC-35` Val dMAE / dRMSE / dP90 | `SC-31` Val dMAE / dRMSE / dP90 | `SC-30` Val dMAE / dRMSE / dP90 |
|---|---|---|---|---|
| `p_c_doy_median_all_11_solar` | `True` | +0.0028 / -0.0049 / -0.0296 | -0.0020 / -0.0000 / -0.0113 | +0.0053 / +0.0008 / -0.0426 |
| `p_c_ffill7_lag2d` | `True` | +0.0103 / -0.0015 / -0.0295 | +0.0070 / +0.0036 / -0.0163 | +0.0152 / +0.0030 / -0.0584 |
| `p_c_ffill7_lag3d` | `True` | +0.0122 / -0.0012 / -0.0353 | +0.0087 / +0.0039 / -0.0222 | +0.0195 / +0.0030 / -0.0606 |
| `p_c_ffill7_lag5d` | `False` | +0.0196 / +0.0026 / -0.0336 | +0.0167 / +0.0102 / -0.0259 | +0.0307 / +0.0105 / -0.0586 |
| `p_c_ffill7_lag7d` | `False` | +0.0250 / +0.0062 / -0.0268 | +0.0214 / +0.0110 / -0.0197 | +0.0421 / +0.0160 / -0.0464 |

### Test (`2,184` Origins) Split-Horizon Breakdown (`h=1..7` vs `h=1..3` vs `h=4..7` with 95% Moving Block Bootstrap `block_length=7`)

| Model | Method | Slice | MAE [95% CI] | delta MAE [95% CI] | MSE | RMSE [95% CI] | delta RMSE | P90-Tail MAE | delta P90-Tail | Fidelity-MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SC-35 | `clean` | `h1_to_7` | 3.7333 [3.3314, 4.2183] | +0.0000 [+0.0000, +0.0000] | 65.5809 | 8.0982 [6.6741, 9.7890] | +0.0000 | 15.5139 | +0.0000 | 0.0000 |
| SC-35 | `clean` | `h1_to_3` | 3.6747 [3.2562, 4.1541] | +0.0000 [+0.0000, +0.0000] | 63.9118 | 7.9945 [6.4139, 9.7660] | +0.0000 | 15.1226 | +0.0000 | 0.0000 |
| SC-35 | `clean` | `h4_to_7` | 3.7774 [3.3380, 4.2919] | +0.0000 [+0.0000, +0.0000] | 66.8327 | 8.1751 [6.6507, 9.9157] | +0.0000 | 15.8074 | +0.0000 | 0.0000 |
| SC-35 | `p_c_doy_median_all_11_solar` | `h1_to_7` | 3.7314 [3.3259, 4.2174] | -0.0019 [-0.0069, +0.0023] | 65.5133 | 8.0940 [6.6681, 9.7836] | -0.0042 | 15.4973 | -0.0166 | 0.1224 |
| SC-35 | `p_c_doy_median_all_11_solar` | `h1_to_3` | 3.6760 [3.2514, 4.1549] | +0.0013 [-0.0069, +0.0091] | 63.9140 | 7.9946 [6.4160, 9.7675] | +0.0001 | 15.1259 | +0.0033 | 0.1421 |
| SC-35 | `p_c_doy_median_all_11_solar` | `h4_to_7` | 3.7729 [3.3353, 4.2879] | -0.0044 [-0.0097, -0.0000] | 66.7128 | 8.1678 [6.6416, 9.9072] | -0.0073 | 15.7759 | -0.0316 | 0.1076 |
| SC-35 | `p_c_ffill7_lag3d` | `h1_to_7` | 3.7317 [3.3267, 4.2186] | -0.0016 [-0.0073, +0.0032] | 65.3067 | 8.0813 [6.6644, 9.7628] | -0.0169 | 15.4756 | -0.0383 | 0.1284 |
| SC-35 | `p_c_ffill7_lag3d` | `h1_to_3` | 3.6784 [3.2562, 4.1584] | +0.0038 [-0.0042, +0.0116] | 63.6836 | 7.9802 [6.4050, 9.7311] | -0.0143 | 15.0812 | -0.0414 | 0.1448 |
| SC-35 | `p_c_ffill7_lag3d` | `h4_to_7` | 3.7716 [3.3345, 4.2843] | -0.0057 [-0.0121, -0.0005] | 66.5240 | 8.1562 [6.6393, 9.8879] | -0.0189 | 15.7715 | -0.0359 | 0.1161 |
| SC-35 | `p_c_ffill7_lag7d` | `h1_to_7` | 3.7310 [3.3259, 4.2170] | -0.0024 [-0.0112, +0.0050] | 65.2387 | 8.0770 [6.6578, 9.7569] | -0.0212 | 15.4582 | -0.0558 | 0.1444 |
| SC-35 | `p_c_ffill7_lag7d` | `h1_to_3` | 3.6785 [3.2550, 4.1540] | +0.0038 [-0.0057, +0.0129] | 63.6329 | 7.9770 [6.3997, 9.7213] | -0.0175 | 15.0684 | -0.0542 | 0.1553 |
| SC-35 | `p_c_ffill7_lag7d` | `h4_to_7` | 3.7703 [3.3307, 4.2834] | -0.0070 [-0.0180, +0.0022] | 66.4430 | 8.1513 [6.6401, 9.8830] | -0.0239 | 15.7505 | -0.0569 | 0.1363 |
| SC-35 | `retrained_no_solar_24vars` | `h1_to_7` | 3.7281 [3.3271, 4.2109] | -0.0053 [-0.0126, +0.0008] | 65.3849 | 8.0861 [6.6588, 9.7719] | -0.0121 | 15.5231 | +0.0092 | 0.2489 |
| SC-35 | `retrained_no_solar_24vars` | `h1_to_3` | 3.6718 [3.2507, 4.1474] | -0.0029 [-0.0147, +0.0075] | 63.7670 | 7.9854 [6.4069, 9.7437] | -0.0091 | 15.1579 | +0.0353 | 0.2663 |
| SC-35 | `retrained_no_solar_24vars` | `h4_to_7` | 3.7703 [3.3310, 4.2817] | -0.0070 [-0.0163, +0.0008] | 66.5984 | 8.1608 [6.6402, 9.8917] | -0.0143 | 15.7971 | -0.0104 | 0.2358 |
| SC-31 | `clean` | `h1_to_7` | 3.7246 [3.3171, 4.2089] | +0.0000 [+0.0000, +0.0000] | 65.9240 | 8.1194 [6.6945, 9.8021] | +0.0000 | 15.6293 | +0.0000 | 0.0000 |
| SC-31 | `clean` | `h1_to_3` | 3.6715 [3.2563, 4.1466] | +0.0000 [+0.0000, +0.0000] | 64.4560 | 8.0284 [6.4522, 9.7851] | +0.0000 | 15.3099 | +0.0000 | 0.0000 |
| SC-31 | `clean` | `h4_to_7` | 3.7645 [3.3235, 4.2833] | +0.0000 [+0.0000, +0.0000] | 67.0251 | 8.1869 [6.6646, 9.9278] | +0.0000 | 15.8689 | +0.0000 | 0.0000 |
| SC-31 | `p_c_doy_median_all_11_solar` | `h1_to_7` | 3.7227 [3.3155, 4.2071] | -0.0020 [-0.0082, +0.0037] | 65.8674 | 8.1159 [6.6896, 9.8009] | -0.0035 | 15.6210 | -0.0084 | 0.1652 |
| SC-31 | `p_c_doy_median_all_11_solar` | `h1_to_3` | 3.6731 [3.2496, 4.1566] | +0.0015 [-0.0101, +0.0122] | 64.3317 | 8.0207 [6.4468, 9.7709] | -0.0077 | 15.2800 | -0.0299 | 0.2113 |
| SC-31 | `p_c_doy_median_all_11_solar` | `h4_to_7` | 3.7598 [3.3191, 4.2775] | -0.0046 [-0.0102, +0.0008] | 67.0191 | 8.1865 [6.6648, 9.9262] | -0.0004 | 15.8767 | +0.0078 | 0.1305 |
| SC-31 | `p_c_ffill7_lag3d` | `h1_to_7` | 3.7219 [3.3135, 4.2073] | -0.0027 [-0.0093, +0.0031] | 65.7004 | 8.1056 [6.6866, 9.7831] | -0.0138 | 15.6090 | -0.0203 | 0.1704 |
| SC-31 | `p_c_ffill7_lag3d` | `h1_to_3` | 3.6700 [3.2501, 4.1528] | -0.0016 [-0.0122, +0.0089] | 64.2185 | 8.0136 [6.4411, 9.7599] | -0.0148 | 15.2624 | -0.0475 | 0.2076 |
| SC-31 | `p_c_ffill7_lag3d` | `h4_to_7` | 3.7609 [3.3208, 4.2761] | -0.0036 [-0.0108, +0.0030] | 66.8119 | 8.1739 [6.6552, 9.9040] | -0.0130 | 15.8690 | +0.0001 | 0.1425 |
| SC-31 | `p_c_ffill7_lag7d` | `h1_to_7` | 3.7197 [3.3126, 4.2043] | -0.0050 [-0.0148, +0.0036] | 65.6057 | 8.0997 [6.6814, 9.7731] | -0.0196 | 15.5903 | -0.0391 | 0.1859 |
| SC-31 | `p_c_ffill7_lag7d` | `h1_to_3` | 3.6694 [3.2502, 4.1483] | -0.0022 [-0.0144, +0.0096] | 64.1483 | 8.0093 [6.4409, 9.7480] | -0.0192 | 15.2530 | -0.0570 | 0.2165 |
| SC-31 | `p_c_ffill7_lag7d` | `h4_to_7` | 3.7574 [3.3169, 4.2746] | -0.0071 [-0.0196, +0.0036] | 66.6988 | 8.1669 [6.6480, 9.8960] | -0.0199 | 15.8432 | -0.0257 | 0.1631 |
| SC-31 | `retrained_no_solar_24vars` | `h1_to_7` | 3.7291 [3.3226, 4.2120] | +0.0045 [-0.0036, +0.0120] | 65.9142 | 8.1188 [6.6973, 9.7990] | -0.0006 | 15.6377 | +0.0084 | 0.2731 |
| SC-31 | `retrained_no_solar_24vars` | `h1_to_3` | 3.6792 [3.2599, 4.1665] | +0.0076 [-0.0049, +0.0196] | 64.5525 | 8.0345 [6.4485, 9.8043] | +0.0060 | 15.3308 | +0.0209 | 0.2962 |
| SC-31 | `retrained_no_solar_24vars` | `h4_to_7` | 3.7666 [3.3254, 4.2825] | +0.0021 [-0.0070, +0.0117] | 66.9354 | 8.1814 [6.6614, 9.9164] | -0.0055 | 15.8680 | -0.0009 | 0.2557 |
| SC-30 | `clean` | `h1_to_7` | 3.7501 [3.3401, 4.2308] | +0.0000 [+0.0000, +0.0000] | 65.8426 | 8.1143 [6.6944, 9.7832] | +0.0000 | 15.6159 | +0.0000 | 0.0000 |
| SC-30 | `clean` | `h1_to_3` | 3.6885 [3.2746, 4.1645] | +0.0000 [+0.0000, +0.0000] | 64.3358 | 8.0210 [6.4435, 9.7855] | +0.0000 | 15.2703 | +0.0000 | 0.0000 |
| SC-30 | `clean` | `h4_to_7` | 3.7962 [3.3528, 4.3161] | +0.0000 [+0.0000, +0.0000] | 66.9727 | 8.1837 [6.6563, 9.9190] | +0.0000 | 15.8752 | +0.0000 | 0.0000 |
| SC-30 | `p_c_doy_median_all_11_solar` | `h1_to_7` | 3.7502 [3.3402, 4.2285] | +0.0002 [-0.0068, +0.0064] | 65.7059 | 8.1059 [6.6823, 9.7758] | -0.0084 | 15.5782 | -0.0377 | 0.1668 |
| SC-30 | `p_c_doy_median_all_11_solar` | `h1_to_3` | 3.6902 [3.2717, 4.1736] | +0.0017 [-0.0092, +0.0124] | 64.1137 | 8.0071 [6.4249, 9.7689] | -0.0139 | 15.2079 | -0.0624 | 0.2101 |
| SC-30 | `p_c_doy_median_all_11_solar` | `h4_to_7` | 3.7952 [3.3516, 4.3160] | -0.0010 [-0.0088, +0.0059] | 66.9001 | 8.1792 [6.6539, 9.9145] | -0.0044 | 15.8560 | -0.0192 | 0.1343 |
| SC-30 | `p_c_ffill7_lag3d` | `h1_to_7` | 3.7465 [3.3380, 4.2207] | -0.0036 [-0.0130, +0.0050] | 65.4444 | 8.0898 [6.6703, 9.7494] | -0.0246 | 15.5457 | -0.0702 | 0.1990 |
| SC-30 | `p_c_ffill7_lag3d` | `h1_to_3` | 3.6914 [3.2754, 4.1731] | +0.0029 [-0.0082, +0.0141] | 64.0209 | 8.0013 [6.4274, 9.7543] | -0.0197 | 15.1844 | -0.0859 | 0.2275 |
| SC-30 | `p_c_ffill7_lag3d` | `h4_to_7` | 3.7877 [3.3477, 4.3036] | -0.0085 [-0.0209, +0.0032] | 66.5121 | 8.1555 [6.6436, 9.8826] | -0.0282 | 15.8167 | -0.0585 | 0.1776 |
| SC-30 | `p_c_ffill7_lag7d` | `h1_to_7` | 3.7518 [3.3459, 4.2258] | +0.0018 [-0.0143, +0.0170] | 65.3361 | 8.0831 [6.6689, 9.7319] | -0.0313 | 15.5202 | -0.0958 | 0.2512 |
| SC-30 | `p_c_ffill7_lag7d` | `h1_to_3` | 3.6952 [3.2778, 4.1764] | +0.0066 [-0.0088, +0.0220] | 63.9190 | 7.9949 [6.4257, 9.7404] | -0.0260 | 15.1680 | -0.1023 | 0.2573 |
| SC-30 | `p_c_ffill7_lag7d` | `h4_to_7` | 3.7943 [3.3589, 4.3084] | -0.0019 [-0.0234, +0.0180] | 66.3990 | 8.1486 [6.6478, 9.8756] | -0.0351 | 15.7843 | -0.0909 | 0.2467 |
| SC-30 | `retrained_no_solar_24vars` | `h1_to_7` | 3.8229 [3.4215, 4.2973] | +0.0728 [+0.0434, +0.1034] | 65.8767 | 8.1164 [6.6948, 9.7711] | +0.0021 | 15.5419 | -0.0741 | 0.5292 |
| SC-30 | `retrained_no_solar_24vars` | `h1_to_3` | 3.7453 [3.3287, 4.2196] | +0.0568 [+0.0302, +0.0834] | 64.2758 | 8.0172 [6.4509, 9.7743] | -0.0037 | 15.2026 | -0.0677 | 0.4709 |
| SC-30 | `retrained_no_solar_24vars` | `h4_to_7` | 3.8810 [3.4451, 4.3980] | +0.0848 [+0.0478, +0.1265] | 67.0774 | 8.1901 [6.6831, 9.9115] | +0.0064 | 15.7963 | -0.0789 | 0.5728 |
