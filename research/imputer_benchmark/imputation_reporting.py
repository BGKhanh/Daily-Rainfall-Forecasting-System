"""
Imputation Benchmark Statistical Aggregation, Verification Gates & Reporting (DS108)
==================================================================================
Computes:
- Branch 2A (Prediction Fidelity vs y_full) & per-horizon h=1..7 breakdowns
- Branch 2B (Forecast Degradation vs y_true using canonical MAE/RMSE/MASE/WAPE & P90 tail)
- Spearman rank correlation between imputer uncertainty sigma_imp and |y_imp - y_full|
- Branch 3 paired Wilcoxon signed-rank tests with Holm-Bonferroni correction across L
- Hybrid threshold L* sensitivity sweep ({3, 5, 7, 14})
- 5 Automated Verification Gates & machine-readable Parquet/CSV/JSON + Markdown report
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

from src.analysis.integrity import assert_no_purged_metrics
from research.imputer_benchmark.imputation_baselines import METHOD_TIERS
from research.imputer_benchmark.imputation_benchmark import ImputationBenchmarkConfig
from src.evaluation.metrics import CANONICAL_DEV_TRAIN_MASE_SCALE, evaluate_point_forecast_metrics

DEFAULT_OUTPUT_DIR = Path("reports") / "imputation_benchmark"


def holm_bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    """Adjust raw p-values using the step-down Holm-Bonferroni family-wise correction."""
    arr = np.asarray(p_values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return []
    adjusted = np.full(n, np.nan, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(arr))
    m = len(finite_idx)
    if m == 0:
        return adjusted.tolist()

    order = finite_idx[np.argsort(arr[finite_idx], kind="mergesort")]
    running_max = 0.0
    for rank, orig_idx in enumerate(order):
        cand = min(1.0, float(arr[orig_idx]) * float(m - rank))
        running_max = max(running_max, cand)
        adjusted[orig_idx] = running_max
    return adjusted.tolist()


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    r, _ = pearsonr(a, b)
    return float(r)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < 3 or np.std(a[valid]) < 1e-12 or np.std(b[valid]) < 1e-12:
        return np.nan, np.nan
    res = spearmanr(a[valid], b[valid])
    return float(res.statistic), float(res.pvalue)


def _compute_group_extrinsic_metrics(
    grp: pd.DataFrame,
    *,
    p90_dev_train: float,
    rain_threshold_mm: float,
) -> Dict[str, Any]:
    """Compute canonical Branch 2A and Branch 2B metrics for one slice DataFrame."""
    status_vals = set(grp["status"].unique())
    if "NOT_APPLICABLE" in status_vals and len(status_vals) == 1:
        return {
            "status": "NOT_APPLICABLE",
            "fidelity_mae": np.nan,
            "fidelity_rmse": np.nan,
            "fidelity_pearson_r": np.nan,
            "rain_flip_rate_pct": np.nan,
            "mae_full": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "mase": np.nan,
            "wape_pct": np.nan,
            "delta_mae_pct": np.nan,
            "delta_rmse_pct": np.nan,
            "delta_mase_pct": np.nan,
            "tail_p90_mae": np.nan,
            "tail_p90_rmse": np.nan,
            "spearman_sigma_rho": np.nan,
            "spearman_sigma_p": np.nan,
        }

    ok = grp[grp["status"] == "OK"].copy()
    y_true = ok["y_true"].to_numpy(dtype=np.float64)
    y_full = ok["y_full"].to_numpy(dtype=np.float64)
    y_imp = ok["y_imp"].to_numpy(dtype=np.float64)
    sigma_imp = ok["sigma_imp"].to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(y_imp)):
        return {
            "status": "TRANSFORM_ERROR",
            "fidelity_mae": np.nan,
            "fidelity_rmse": np.nan,
            "fidelity_pearson_r": np.nan,
            "rain_flip_rate_pct": np.nan,
            "mae_full": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "mase": np.nan,
            "wape_pct": np.nan,
            "delta_mae_pct": np.nan,
            "delta_rmse_pct": np.nan,
            "delta_mase_pct": np.nan,
            "tail_p90_mae": np.nan,
            "tail_p90_rmse": np.nan,
            "spearman_sigma_rho": np.nan,
            "spearman_sigma_p": np.nan,
        }

    diff_fid = y_imp - y_full
    fid_mae = float(np.mean(np.abs(diff_fid)))
    fid_rmse = float(np.sqrt(np.mean(diff_fid ** 2)))
    fid_r = _safe_pearson(y_imp, y_full)

    rain_full = y_full >= rain_threshold_mm
    rain_imp = y_imp >= rain_threshold_mm
    flip_pct = 100.0 * float(np.mean(rain_full != rain_imp))

    full_pt = evaluate_point_forecast_metrics(y_true, y_full, scale=CANONICAL_DEV_TRAIN_MASE_SCALE)
    imp_pt = evaluate_point_forecast_metrics(y_true, y_imp, scale=CANONICAL_DEV_TRAIN_MASE_SCALE)

    denom_y = float(np.sum(np.abs(y_true)))
    wape_pct = (100.0 * float(np.sum(np.abs(y_true - y_imp))) / denom_y) if denom_y > 1e-12 else np.nan

    delta_mae_pct = 100.0 * (imp_pt["MAE"] - full_pt["MAE"]) / max(full_pt["MAE"], 1e-9)
    delta_rmse_pct = 100.0 * (imp_pt["RMSE"] - full_pt["RMSE"]) / max(full_pt["RMSE"], 1e-9)
    delta_mase_pct = 100.0 * (imp_pt["MASE"] - full_pt["MASE"]) / max(full_pt["MASE"], 1e-9)

    tail_mask = y_true > p90_dev_train
    if np.any(tail_mask):
        tail_err = y_true[tail_mask] - y_imp[tail_mask]
        tail_mae = float(np.mean(np.abs(tail_err)))
        tail_rmse = float(np.sqrt(np.mean(tail_err ** 2)))
    else:
        tail_mae, tail_rmse = np.nan, np.nan

    method_name = str(ok["method"].iloc[0])
    if method_name in ("custom_mode2", "custom_hybrid"):
        sp_rho, sp_p = _safe_spearman(sigma_imp, np.abs(diff_fid))
    else:
        sp_rho, sp_p = np.nan, np.nan

    return {
        "status": "OK",
        "fidelity_mae": fid_mae,
        "fidelity_rmse": fid_rmse,
        "fidelity_pearson_r": fid_r,
        "rain_flip_rate_pct": flip_pct,
        "mae_full": full_pt["MAE"],
        "mae": imp_pt["MAE"],
        "rmse": imp_pt["RMSE"],
        "mase": imp_pt["MASE"],
        "wape_pct": wape_pct,
        "delta_mae_pct": delta_mae_pct,
        "delta_rmse_pct": delta_rmse_pct,
        "delta_mase_pct": delta_mase_pct,
        "tail_p90_mae": tail_mae,
        "tail_p90_rmse": tail_rmse,
        "spearman_sigma_rho": sp_rho,
        "spearman_sigma_p": sp_p,
    }


def compute_wilcoxon_table(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    conf: ImputationBenchmarkConfig,
) -> pd.DataFrame:
    """Compute paired Wilcoxon signed-rank tests (Holm-Bonferroni across L) for MAE and Fidelity-MAE."""
    tier3_methods = ("doy_climatology", "sklearn_iterative", "sklearn_knn")
    anchor_err = (
        raw_df.assign(
            abs_err=np.abs(raw_df["y_true"] - raw_df["y_imp"]),
            abs_fid=np.abs(raw_df["y_imp"] - raw_df["y_full"]),
        )
        .groupby(["model_id", "position", "gap_length", "profile", "method", "anchor_date"], as_index=False)
        .agg(anchor_mae=("abs_err", "mean"), anchor_fid=("abs_fid", "mean"), status=("status", "first"))
    )

    wilcoxon_rows: List[Dict[str, Any]] = []
    for model_id in raw_df["model_id"].unique():
        for pos in conf.gap_positions:
            for prof in conf.profiles:
                raw_p_one_mae: List[float] = []
                raw_p_two_mae: List[float] = []
                raw_p_one_fid: List[float] = []
                raw_p_two_fid: List[float] = []
                temp_entries: List[Dict[str, Any]] = []

                for gap_len in [g for g in conf.gap_lengths if g > 0]:
                    sub_sum = summary_df[
                        (summary_df["model_id"] == model_id)
                        & (summary_df["position"] == pos)
                        & (summary_df["profile"] == prof)
                        & (summary_df["gap_length"] == gap_len)
                        & (summary_df["method"].isin(tier3_methods))
                        & (summary_df["status"] == "OK")
                    ]
                    best_t3 = str(sub_sum.sort_values("mae", ascending=True)["method"].iloc[0])
                    best_t3_fid = str(sub_sum.sort_values("fidelity_mae", ascending=True)["method"].iloc[0])

                    hyb_a = anchor_err[
                        (anchor_err["model_id"] == model_id)
                        & (anchor_err["position"] == pos)
                        & (anchor_err["profile"] == prof)
                        & (anchor_err["gap_length"] == gap_len)
                        & (anchor_err["method"] == "custom_hybrid")
                    ].sort_values("anchor_date")
                    t3_a = anchor_err[
                        (anchor_err["model_id"] == model_id)
                        & (anchor_err["position"] == pos)
                        & (anchor_err["profile"] == prof)
                        & (anchor_err["gap_length"] == gap_len)
                        & (anchor_err["method"] == best_t3)
                    ].sort_values("anchor_date")
                    t3_fid_a = anchor_err[
                        (anchor_err["model_id"] == model_id)
                        & (anchor_err["position"] == pos)
                        & (anchor_err["profile"] == prof)
                        & (anchor_err["gap_length"] == gap_len)
                        & (anchor_err["method"] == best_t3_fid)
                    ].sort_values("anchor_date")

                    diffs_mae = hyb_a["anchor_mae"].to_numpy() - t3_a["anchor_mae"].to_numpy()
                    if np.all(np.abs(diffs_mae) < 1e-12):
                        p1_mae, p2_mae = 1.0, 1.0
                    else:
                        p1_mae = float(wilcoxon(diffs_mae, alternative="less", zero_method="wilcox").pvalue)
                        p2_mae = float(wilcoxon(diffs_mae, alternative="two-sided", zero_method="wilcox").pvalue)

                    diffs_fid = hyb_a["anchor_fid"].to_numpy() - t3_fid_a["anchor_fid"].to_numpy()
                    if np.all(np.abs(diffs_fid) < 1e-12):
                        p1_fid, p2_fid = 1.0, 1.0
                    else:
                        p1_fid = float(wilcoxon(diffs_fid, alternative="less", zero_method="wilcox").pvalue)
                        p2_fid = float(wilcoxon(diffs_fid, alternative="two-sided", zero_method="wilcox").pvalue)

                    raw_p_one_mae.append(p1_mae)
                    raw_p_two_mae.append(p2_mae)
                    raw_p_one_fid.append(p1_fid)
                    raw_p_two_fid.append(p2_fid)
                    temp_entries.append(
                        {
                            "model_id": model_id,
                            "position": pos,
                            "profile": prof,
                            "gap_length": int(gap_len),
                            "best_tier3_method": best_t3,
                            "best_tier3_fidelity_method": best_t3_fid,
                            "hybrid_mean_anchor_mae": float(np.mean(hyb_a["anchor_mae"])),
                            "tier3_mean_anchor_mae": float(np.mean(t3_a["anchor_mae"])),
                            "mean_diff_mae": float(np.mean(diffs_mae)),
                            "hybrid_mean_fidelity_mae": float(np.mean(hyb_a["anchor_fid"])),
                            "tier3_mean_fidelity_mae": float(np.mean(t3_fid_a["anchor_fid"])),
                            "p_one_sided_raw": p1_mae,
                            "p_two_sided_raw": p2_mae,
                            "p_fid_one_sided_raw": p1_fid,
                            "p_fid_two_sided_raw": p2_fid,
                        }
                    )

                adj1_m = holm_bonferroni_adjust(raw_p_one_mae)
                adj2_m = holm_bonferroni_adjust(raw_p_two_mae)
                adj1_f = holm_bonferroni_adjust(raw_p_one_fid)
                adj2_f = holm_bonferroni_adjust(raw_p_two_fid)
                for row, p1m, p2m, p1f, p2f in zip(temp_entries, adj1_m, adj2_m, adj1_f, adj2_f):
                    row["p_one_sided_holm"] = p1m
                    row["p_two_sided_holm"] = p2m
                    row["significant_one_sided_005"] = bool(p1m < 0.05)
                    row["p_fid_one_sided_holm"] = p1f
                    row["p_fid_two_sided_holm"] = p2f
                    row["significant_fidelity_005"] = bool(p1f < 0.05)
                    wilcoxon_rows.append(row)

    return pd.DataFrame(wilcoxon_rows)


def build_summary_tables(benchmark_out: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Aggregate overall, by-horizon, sensitivity, Wilcoxon tests, and resource SLA tables."""
    conf: ImputationBenchmarkConfig = benchmark_out["config"]
    raw_df: pd.DataFrame = benchmark_out["raw_df"]
    intrinsic_df: pd.DataFrame = benchmark_out["intrinsic_df"]
    sens_raw_df: pd.DataFrame = benchmark_out["sensitivity_raw_df"]
    p90 = float(benchmark_out["p90_dev_train"])

    # Aggregate Branch 1 + timing over anchors
    intr_agg = (
        intrinsic_df.groupby(["position", "gap_length", "profile", "method"], as_index=False)
        .agg(
            nrmse_mean=("nrmse_mean", "mean"),
            nmae_mean=("nmae_mean", "mean"),
            nrmse_scen=("nrmse_scen", "mean"),
            nmae_scen=("nmae_scen", "mean"),
            vr_scen=("vr_scen", "mean"),
            vr_mean=("vr_mean", "mean"),
            fe_fidelity_nmae_scen=("fe_fidelity_nmae_scen", "mean"),
            fe_fidelity_nmae_mean=("fe_fidelity_nmae_mean", "mean"),
            cross_corr_err_scen=("cross_corr_err_scen", "mean"),
            cross_corr_err_mean=("cross_corr_err_mean", "mean"),
            domain_violation_pct=("domain_violation_pct", "mean"),
            impute_ms_p50=("impute_ms", "median"),
            impute_ms_p95=("impute_ms", lambda s: float(np.percentile(s, 95))),
            fe_predict_ms_p50=("fe_predict_ms", "median"),
            fe_predict_ms_p95=("fe_predict_ms", lambda s: float(np.percentile(s, 95))),
            total_ms_p50=("total_ms", "median"),
            total_ms_p95=("total_ms", lambda s: float(np.percentile(s, 95))),
        )
    )

    overall_rows: List[Dict[str, Any]] = []
    for (model_id, pos, gap_len, prof, method), grp in raw_df.groupby(
        ["model_id", "position", "gap_length", "profile", "method"], sort=False
    ):
        ext = _compute_group_extrinsic_metrics(grp, p90_dev_train=p90, rain_threshold_mm=conf.rain_threshold_mm)
        overall_rows.append(
            {
                "model_id": model_id,
                "position": pos,
                "gap_length": int(gap_len),
                "profile": prof,
                "method": method,
                "tier": METHOD_TIERS.get(method, "Unknown"),
                **ext,
            }
        )
    summary_df = pd.DataFrame(overall_rows).merge(
        intr_agg, on=["position", "gap_length", "profile", "method"], how="left"
    )
    assert_no_purged_metrics(summary_df)

    # Horizon breakdown h=1..7
    horizon_rows: List[Dict[str, Any]] = []
    for (model_id, pos, gap_len, prof, method, h), grp in raw_df.groupby(
        ["model_id", "position", "gap_length", "profile", "method", "horizon"], sort=False
    ):
        ext = _compute_group_extrinsic_metrics(grp, p90_dev_train=p90, rain_threshold_mm=conf.rain_threshold_mm)
        horizon_rows.append(
            {
                "model_id": model_id,
                "position": pos,
                "gap_length": int(gap_len),
                "profile": prof,
                "method": method,
                "horizon": int(h),
                **ext,
            }
        )
    horizon_df = pd.DataFrame(horizon_rows)
    assert_no_purged_metrics(horizon_df)

    # Sensitivity L* in {3, 5, 7, 14}
    sens_rows: List[Dict[str, Any]] = []
    for (model_id, l_star, pos, gap_len, prof), grp in sens_raw_df.groupby(
        ["model_id", "threshold_l_star", "position", "gap_length", "profile"], sort=False
    ):
        ext = _compute_group_extrinsic_metrics(grp, p90_dev_train=p90, rain_threshold_mm=conf.rain_threshold_mm)
        sens_rows.append(
            {
                "model_id": model_id,
                "threshold_l_star": int(l_star),
                "position": pos,
                "gap_length": int(gap_len),
                "profile": prof,
                "method": "custom_hybrid",
                **ext,
            }
        )
    sensitivity_df = pd.DataFrame(sens_rows)
    assert_no_purged_metrics(sensitivity_df)

    wilcoxon_df = compute_wilcoxon_table(raw_df, summary_df, conf)
    return {
        "summary_df": summary_df,
        "horizon_df": horizon_df,
        "sensitivity_df": sensitivity_df,
        "wilcoxon_df": wilcoxon_df,
    }


def evaluate_acceptance_gates(
    benchmark_out: Dict[str, Any],
    summary_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Run all 5 automated verification gates required by Section 7.2."""
    raw_df: pd.DataFrame = benchmark_out["raw_df"]
    l0_sub = raw_df[(raw_df["gap_length"] == 0) & (raw_df["method"] != "raw_nan") & (raw_df["status"] == "OK")]
    max_l0_diff = float(np.max(np.abs(l0_sub["y_imp"].to_numpy() - l0_sub["y_full"].to_numpy())))
    gate1_pass = max_l0_diff < 1e-9

    nan_failures: List[str] = benchmark_out["gate_nan_failures"]
    gate2_pass = len(nan_failures) == 0

    custom_sub = summary_df[summary_df["method"].isin(("custom_mode1", "custom_mode2", "custom_hybrid"))]
    max_dom_viol = float(np.nanmax(custom_sub["domain_violation_pct"].to_numpy()))
    gate3_pass = max_dom_viol == 0.0

    dev_start, dev_end, n_dev = benchmark_out["dev_train_range"]
    test_start, test_end, n_test = benchmark_out["test_range"]
    gate4_pass = pd.Timestamp(dev_end) < pd.Timestamp(test_start)

    isolated_c_rss = float(benchmark_out["heavy_single_anchor_rss_mb"])
    isolated_a_rss = float(benchmark_out.get("isolated_rss_profile_a_mode2_mb", isolated_c_rss))
    peak_runner_rss = float(benchmark_out["peak_rss_mb"])
    gate5_pass = isolated_c_rss <= 350.0 and isolated_a_rss <= 350.0 and peak_runner_rss <= 1536.0

    all_passed = bool(gate1_pass and gate2_pass and gate3_pass and gate4_pass and gate5_pass)
    return {
        "all_passed": all_passed,
        "gates": {
            "1_identity_l0_max_abs_diff": {"passed": gate1_pass, "max_abs_diff": max_l0_diff, "threshold": 1e-9},
            "2_zero_nan_input_gate": {"passed": gate2_pass, "failure_count": len(nan_failures), "samples": nan_failures[:5]},
            "3_domain_bounds_custom_0pct": {"passed": gate3_pass, "max_violation_pct": max_dom_viol},
            "4_leakage_audit_split_order": {
                "passed": gate4_pass,
                "dev_train": f"{dev_start}..{dev_end} ({n_dev} rows)",
                "test": f"{test_start}..{test_end} ({n_test} rows)",
            },
            "5_memory_and_runtime_sanity": {
                "passed": gate5_pass,
                "isolated_single_anchor_l90_profile_c_rss_mb": round(isolated_c_rss, 2),
                "isolated_single_anchor_l90_profile_a_mode2_rss_mb": round(isolated_a_rss, 2),
                "serving_rss_budget_mb": 350.0,
                "benchmark_runner_peak_rss_mb": round(peak_runner_rss, 2),
            },
        },
    }


def _df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                cells.append("N/A" if not np.isfinite(val) else format(float(val), floatfmt))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_benchmark_artifacts(
    benchmark_out: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    gates: Dict[str, Any],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Path]:
    """Persist Parquet/CSV raw results, summary CSVs, JSON run metadata, and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    conf: ImputationBenchmarkConfig = benchmark_out["config"]

    raw_parquet = output_dir / "benchmark_raw_predictions.parquet"
    raw_csv = output_dir / "benchmark_raw_predictions.csv"
    summary_csv = output_dir / "benchmark_summary_metrics.csv"
    horizon_csv = output_dir / "benchmark_horizon_breakdown.csv"
    sensitivity_csv = output_dir / "benchmark_hybrid_sensitivity.csv"
    wilcoxon_csv = output_dir / "benchmark_wilcoxon_tests.csv"
    shap_csv = output_dir / "shap_exogenous_ranking.csv"
    meta_json = output_dir / "run_metadata.json"
    report_md = output_dir / "benchmark_report.md"

    benchmark_out["raw_df"].to_parquet(raw_parquet, index=False)
    benchmark_out["raw_df"].to_csv(raw_csv, index=False)
    tables["summary_df"].to_csv(summary_csv, index=False)
    tables["horizon_df"].to_csv(horizon_csv, index=False)
    tables["sensitivity_df"].to_csv(sensitivity_csv, index=False)
    tables["wilcoxon_df"].to_csv(wilcoxon_csv, index=False)
    benchmark_out["shap_sc31"].to_csv(shap_csv, index=False)

    metadata = {
        "location_id": conf.location_id,
        "models_evaluated": ["SC-31 (lightgbm_tweedie_F3)", "SC-30 (lightgbm_tweedie_F2)"],
        "dev_train_range": benchmark_out["dev_train_range"],
        "test_range": benchmark_out["test_range"],
        "p90_dev_train_mm": benchmark_out["p90_dev_train"],
        "n_anchors": len(benchmark_out["anchors_df"]),
        "anchors": benchmark_out["anchors_df"].to_dict(orient="records"),
        "profile_a_ceres_columns": benchmark_out["profiles_map"]["A"],
        "profile_b_top_k_shap_columns": benchmark_out["profiles_map"]["B"],
        "profile_c_all_exogenous_count": len(benchmark_out["profiles_map"]["C"]),
        "imputer_params": {
            "hybrid_threshold_days": conf.hybrid_threshold_days,
            "sensitivity_thresholds": list(conf.sensitivity_thresholds),
            "analog_top_k": 10,
            "analog_match_window_days": 30,
            "seasonal_phase_radius_days": 5,
            "random_state": conf.random_state,
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "verification_gates": gates,
    }
    meta_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # Compose comprehensive Markdown technical report (§7.1.3)
    s_sc31 = tables["summary_df"][tables["summary_df"]["model_id"] == "SC-31"].copy()
    lines = [
        "# Benchmark Report: Single-Location Exogenous Imputer vs Tier 1-3 Baselines (HCMC Daily)",
        "",
        f"- **Dev_Train (Fit Only):** `{metadata['dev_train_range'][0]}` .. `{metadata['dev_train_range'][1]}` ({metadata['dev_train_range'][2]} rows)",
        f"- **Test (Evaluation):** `{metadata['test_range'][0]}` .. `{metadata['test_range'][1]}` ({metadata['test_range'][2]} rows; **{len(benchmark_out['anchors_df'])}** rolling-origin anchors)",
        "- **Frozen LightGBM Models:** `SC-31` (`F3`, 70 features — primary) & `SC-30` (`F2`, 65 features — comparison)",
        f"- **Top-{conf.profile_b_top_k} SHAP (Profile B, Dev_Train):** {', '.join(benchmark_out['profiles_map']['B'])}",
        (
            f"- **Verification Gates All Passed:** `{gates['all_passed']}` "
            f"(Isolated L=90 RSS: `{gates['gates']['5_memory_and_runtime_sanity']['isolated_single_anchor_l90_profile_c_rss_mb']} MB` "
            f"/ Mode2 `{gates['gates']['5_memory_and_runtime_sanity']['isolated_single_anchor_l90_profile_a_mode2_rss_mb']} MB` <= 350 MB budget)"
        ),
        "",
        "## 1. Automated Verification Gates (Section 7.2)",
        "```json",
        json.dumps(gates, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 2. Overall Ranking by Position × Profile × Gap Length (`SC-31`)",
    ]

    cols_show = [
        "gap_length",
        "method",
        "status",
        "mae",
        "delta_mae_pct",
        "fidelity_mae",
        "rain_flip_rate_pct",
        "nmae_mean",
        "vr_scen",
        "vr_mean",
        "domain_violation_pct",
    ]
    for pos in conf.gap_positions:
        for prof in conf.profiles:
            lines.append(f"\n### Position: `{pos}` | Profile `{prof}` (`SC-31`)\n")
            sub = (
                s_sc31[(s_sc31["position"] == pos) & (s_sc31["profile"] == prof) & (s_sc31["gap_length"] > 0)]
                .sort_values(["gap_length", "mae"], ascending=[True, True])
                .loc[:, cols_show]
            )
            lines.append(_df_to_markdown(sub, ".3f"))

    lines.append("\n## 3. Horizon Breakdown h=1..7 for Profile B (`SC-31`, L in {3, 14, 90})\n")
    h_sub = tables["horizon_df"][
        (tables["horizon_df"]["model_id"] == "SC-31")
        & (tables["horizon_df"]["profile"] == "B")
        & (tables["horizon_df"]["position"] == "trailing")
        & (tables["horizon_df"]["gap_length"].isin([3, 14, 90]))
        & (tables["horizon_df"]["method"].isin(["doy_climatology", "ffill", "custom_mode1", "custom_mode2", "custom_hybrid"]))
    ][["gap_length", "method", "horizon", "fidelity_mae", "mae", "rmse", "mase", "rain_flip_rate_pct"]]
    lines.append(_df_to_markdown(h_sub.sort_values(["gap_length", "horizon", "mae"]), ".3f"))

    lines.append("\n## 4. Hybrid Threshold Sensitivity `L*` in {3, 5, 7, 14} (`SC-31`, Trailing)\n")
    sens_sub = (
        tables["sensitivity_df"][
            (tables["sensitivity_df"]["model_id"] == "SC-31") & (tables["sensitivity_df"]["position"] == "trailing")
        ]
        .groupby(["threshold_l_star", "profile"], as_index=False)
        .agg(mean_mae=("mae", "mean"), mean_fidelity_mae=("fidelity_mae", "mean"), mean_delta_mae_pct=("delta_mae_pct", "mean"))
    )
    lines.append(_df_to_markdown(sens_sub, ".4f"))

    lines.append("\n## 5. Paired Wilcoxon Signed-Rank Tests (Holm-Bonferroni, `SC-31`)\n")
    w_sub = tables["wilcoxon_df"][tables["wilcoxon_df"]["model_id"] == "SC-31"][
        [
            "position",
            "profile",
            "gap_length",
            "best_tier3_method",
            "hybrid_mean_anchor_mae",
            "tier3_mean_anchor_mae",
            "p_one_sided_holm",
            "significant_one_sided_005",
            "best_tier3_fidelity_method",
            "hybrid_mean_fidelity_mae",
            "tier3_mean_fidelity_mae",
            "p_fid_one_sided_holm",
            "significant_fidelity_005",
        ]
    ]
    lines.append(_df_to_markdown(w_sub, ".4g"))

    lines.append("\n## 6. Resource & Latency Profile (P50 / P95 ms per anchor)\n")
    lat_sub = (
        s_sc31[s_sc31["gap_length"].isin([7, 30, 90]) & (s_sc31["position"] == "trailing") & (s_sc31["profile"] == "A")]
        [["gap_length", "method", "impute_ms_p50", "impute_ms_p95", "fe_predict_ms_p50", "total_ms_p50", "total_ms_p95"]]
        .sort_values(["gap_length", "total_ms_p50"])
    )
    lines.append(_df_to_markdown(lat_sub, ".2f"))

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "raw_parquet": raw_parquet,
        "raw_csv": raw_csv,
        "summary_csv": summary_csv,
        "horizon_csv": horizon_csv,
        "sensitivity_csv": sensitivity_csv,
        "wilcoxon_csv": wilcoxon_csv,
        "shap_csv": shap_csv,
        "meta_json": meta_json,
        "report_md": report_md,
    }
