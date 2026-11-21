"""Generator & executor helper for `research/imputer_benchmark/imputation_based_inference.ipynb` (DS108)."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK_PATH = Path("research/imputer_benchmark/imputation_based_inference.ipynb")


def _md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip())


def _code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(text.strip())


def build_notebook(notebook_path: Path = NOTEBOOK_PATH) -> Path:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        _md(
            """
# Inference-Time Exogenous Imputation & Marginalization Benchmark (`HCMC Daily`)

Notebook trình diễn thực nghiệm (**Presentation Layer** — toàn bộ logic chạy & metric được tách gọn trong [`exogenous_imputer.py`](./exogenous_imputer.py), [`imputation_baselines.py`](./imputation_baselines.py), [`imputation_benchmark.py`](./imputation_benchmark.py) và [`imputation_reporting.py`](./imputation_reporting.py) thuộc `research/imputer_benchmark/`).

## Tổng quan Kiến trúc & Guardrails
- **Cô lập Serving:** API `POST /predict` và automated pipeline giữ nguyên fail-fast (`COMPLETE_ALL_35`, zero real-time imputation).
- **Zero Data Leakage:** Mọi imputer baseline (`SimpleImputer`, `IterativeImputer`, `KNNImputer`, `doy_climatology`), TreeSHAP ranking và `ClimatologyLookupTable` / `HistoricalAnalogStore` đều fit/xây dựng **duy nhất trên `Dev_Train` (`2001-01-01 .. 2018-04-30`, 6,329 ngày)**.
- **Đánh giá sạch trên `Test`:** `48` rolling-origin anchors trải đều trên `Test` (`2020-05-01 .. 2026-04-23`, cách đều ~45 ngày), bao phủ đủ 3 giai đoạn khí hậu (`Dry`, `Transition`, `Wet`).
- **Mô hình đóng băng:** `SC-31` (`lightgbm_tweedie_F3`, 70 features — chính) & `SC-30` (`lightgbm_tweedie_F2`, 65 features — đối chiếu).
- **Ma trận 4 trục:** `Position (trailing, interior)` × `Length L ∈ {0, 3, 7, 14, 30, 60, 90}` × `Profile (A: 11 biến CERES SYN1deg, B: Top-5 SHAP, C: 34 biến ngoại sinh)` × `11 Methods (Tier 1–4)`.
            """
        ),
        _code(
            """
# Bootstrap root & load persisted benchmark outputs (or run fresh if absent)
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

ROOT_DIR = Path.cwd()
while not (ROOT_DIR / "src").exists() and ROOT_DIR != ROOT_DIR.parent:
    ROOT_DIR = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.imputer_benchmark.imputation_benchmark import ImputationBenchmarkConfig, run_benchmark_execution
from research.imputer_benchmark.imputation_reporting import (
    DEFAULT_OUTPUT_DIR,
    build_summary_tables,
    evaluate_acceptance_gates,
    write_benchmark_artifacts,
)

sns.set_theme(style="whitegrid", context="notebook")
OUT_DIR = ROOT_DIR / DEFAULT_OUTPUT_DIR
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

summary_csv = OUT_DIR / "benchmark_summary_metrics.csv"
if not summary_csv.exists():
    out = run_benchmark_execution(ImputationBenchmarkConfig())
    tables = build_summary_tables(out)
    gates = evaluate_acceptance_gates(out, tables["summary_df"])
    write_benchmark_artifacts(out, tables, gates, output_dir=OUT_DIR)

meta = json.loads((OUT_DIR / "run_metadata.json").read_text(encoding="utf-8"))
summary_df = pd.read_csv(OUT_DIR / "benchmark_summary_metrics.csv")
horizon_df = pd.read_csv(OUT_DIR / "benchmark_horizon_breakdown.csv")
sensitivity_df = pd.read_csv(OUT_DIR / "benchmark_hybrid_sensitivity.csv")
wilcoxon_df = pd.read_csv(OUT_DIR / "benchmark_wilcoxon_tests.csv")
shap_df = pd.read_csv(OUT_DIR / "shap_exogenous_ranking.csv")
raw_df = pd.read_parquet(OUT_DIR / "benchmark_raw_predictions.parquet")

print(f"Loaded benchmark artifacts: {len(raw_df):,} raw rows | {meta['n_anchors']} anchors")
print(f"Dev_Train: {meta['dev_train_range']} | Test: {meta['test_range']}")
            """
        ),
        _md(
            """
## 1. Automated Verification Gates (5 Kiểm định Tự động — Mục 7.2)

Xác nhận toàn bộ 5 gate tự động: Identity Check tại `L = 0` (`< 1e-9`), Zero-NaN Input Gate (trừ `raw_nan`), Domain Bounds (`0.0%` cho `custom_*`), Leakage Audit Gate, và Peak RSS Budget.
            """
        ),
        _code(
            """
gates_info = meta["verification_gates"]
print("ALL GATES PASSED:", gates_info["all_passed"])
gate_rows = [{"gate": k, **v} for k, v in gates_info["gates"].items()]
display(pd.DataFrame(gate_rows))
            """
        ),
        _md(
            """
## 2. Kiểm toán SHAP trên `Dev_Train` (Xác định Hồ sơ B — Top-K Biến Ngoại sinh Chủ lực)

TreeSHAP (`pred_contrib=True`) được tính hoàn toàn trên `Dev_Train` sạch (`6,293` origins) của `SC-31` nhằm bảo đảm không rò rỉ dữ liệu Test.
            """
        ),
        _code(
            """
top12_shap = shap_df.head(12).copy()
display(top12_shap)

fig, ax = plt.subplots(figsize=(9, 4.8))
plot_df = top12_shap.sort_values("mean_abs_shap", ascending=True)
colors = ["#d9534f" if r <= 5 else "#5b8db8" for r in plot_df["rank"]]
ax.barh(plot_df["variable_en"], plot_df["mean_abs_shap"], color=colors)
ax.set_title("Dev_Train Mean |TreeSHAP| by Raw Exogenous Group (SC-31, Red = Profile B Top-5)")
ax.set_xlabel("Sum of Mean |SHAP log-margin| across raw + lag + rolling + angular derivatives")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig01_shap_top_exogenous.png", dpi=150)
plt.show()
            """
        ),
        _md(
            r"""
## 3. Phân rã Đường cong Suy giảm Sai số theo $L$ (`Trailing` vs `Interior`, Profiles A/B/C)

So sánh mức suy giảm $\Delta\text{MAE}(\%)$ và `Fidelity-MAE` của 11 phương pháp (Tier 1–4) trên **`SC-31`**. Lưu ý: `linear_interp` không áp dụng cho `trailing` và `sklearn_iterative` được đánh dấu `NOT_APPLICABLE` tại Profile C.
            """
        ),
        _code(
            """
key_methods = [
    "raw_nan", "sklearn_mean", "sklearn_median", "ffill", "linear_interp",
    "doy_climatology", "sklearn_iterative", "sklearn_knn",
    "custom_mode1", "custom_mode2", "custom_hybrid",
]
sc31 = summary_df[(summary_df["model_id"] == "SC-31") & (summary_df["gap_length"] > 0)].copy()

for pos in ["trailing", "interior"]:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)
    for ax, prof, title in zip(axes, ["A", "B", "C"], [
        "Profile A (11 CERES Radiation)",
        "Profile B (Top-5 SHAP)",
        "Profile C (All 34 Exogenous)",
    ]):
        sub = sc31[(sc31["position"] == pos) & (sc31["profile"] == prof) & (sc31["status"] == "OK")]
        for m in ["raw_nan", "sklearn_median", "ffill", "doy_climatology", "sklearn_iterative", "sklearn_knn", "custom_mode1", "custom_mode2", "custom_hybrid"]:
            m_sub = sub[sub["method"] == m].sort_values("gap_length")
            if m_sub.empty:
                continue
            lw = 2.6 if m == "custom_hybrid" else (1.8 if m.startswith("custom") else 1.2)
            ls = "-" if m == "custom_hybrid" else ("--" if m.startswith("custom") else ":")
            ax.plot(m_sub["gap_length"], m_sub["delta_mae_pct"], marker="o", linewidth=lw, linestyle=ls, label=m)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.axvline(7, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(f"{pos.upper()} — {title}")
        ax.set_xlabel("Gap Length L (days)")
        ax.set_xticks([3, 7, 14, 30, 60, 90])
    axes[0].set_ylabel("Forecast Degradation ΔMAE (%) vs Clean Full-Data")
    axes[-1].legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"fig02_degradation_{pos}_sc31.png", dpi=150)
    plt.show()

# Prediction Fidelity-MAE (Branch 2A) curves on Trailing across Profiles A, B, C
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), sharey=False)
for ax, prof, title in zip(axes, ["A", "B", "C"], [
    "Profile A (11 CERES Radiation)",
    "Profile B (Top-5 SHAP)",
    "Profile C (All 34 Exogenous)",
]):
    sub = sc31[(sc31["position"] == "trailing") & (sc31["profile"] == prof) & (sc31["status"] == "OK") & (sc31["method"] != "raw_nan")]
    for m in ["sklearn_median", "sklearn_mean", "ffill", "doy_climatology", "sklearn_iterative", "sklearn_knn", "custom_mode1", "custom_mode2", "custom_hybrid"]:
        m_sub = sub[sub["method"] == m].sort_values("gap_length")
        if m_sub.empty:
            continue
        lw = 2.6 if m == "custom_hybrid" else (1.8 if m.startswith("custom") else 1.2)
        ax.plot(m_sub["gap_length"], m_sub["fidelity_mae"], marker="o", linewidth=lw, label=m)
    ax.axvline(7, color="gray", linestyle="--", alpha=0.4)
    ax.set_title(f"TRAILING Fidelity-MAE — {title}")
    ax.set_xlabel("Gap Length L (days)")
    ax.set_xticks([3, 7, 14, 30, 60, 90])
axes[0].set_ylabel("Fidelity-MAE vs Clean Forecast (mm/day)")
axes[-1].legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig02b_fidelity_mae_trailing_sc31.png", dpi=150)
plt.show()

for pos in ["trailing", "interior"]:
    for prof in ["A", "B", "C"]:
        print(f"=== RANKING TABLE (SC-31) | Position={pos} | Profile={prof} | L in {{3, 14, 90}} ===")
        tbl = (
            sc31[(sc31["position"] == pos) & (sc31["profile"] == prof) & (sc31["gap_length"].isin([3, 14, 90]))]
            [["gap_length", "method", "status", "mae", "rmse", "mase", "delta_mae_pct", "fidelity_mae", "rain_flip_rate_pct", "domain_violation_pct"]]
            .sort_values(["gap_length", "mae"])
        )
        display(tbl)
            """
        ),
        _md(
            """
## 4. Nhánh 1 (Intrinsic) & Tránh Bẫy Double-Penalty (`VR_scen` vs `VR_mean`)

Với Chế độ 2 (`custom_mode2`, `custom_hybrid`), các chỉ số cấu trúc (`Variance Ratio`, `FE Fidelity`, `Cross-Corr Error`) được tính trên **từng kịch bản trước khi lấy trung bình theo $w_k$** (`*_scen`), đặt cạnh giá trị tính trên chuỗi trung bình (`*_mean`) để kiểm chứng hiện tượng làm phẳng tín hiệu nếu cộng gộp sớm.
            """
        ),
        _code(
            """
dp_sub = sc31[(sc31["position"] == "trailing") & (sc31["profile"] == "A") & (sc31["gap_length"] >= 14) & (sc31["method"].isin(["doy_climatology", "ffill", "sklearn_knn", "custom_mode1", "custom_mode2", "custom_hybrid"]))][
    ["gap_length", "method", "nmae_mean", "nrmse_mean", "vr_scen", "vr_mean", "fe_fidelity_nmae_scen", "cross_corr_err_scen", "domain_violation_pct"]
].sort_values(["gap_length", "nmae_mean"])
display(dp_sub)

fig, ax = plt.subplots(figsize=(8.5, 4.4))
m2_dp = dp_sub[dp_sub["method"] == "custom_mode2"].sort_values("gap_length")
x = np.arange(len(m2_dp))
ax.bar(x - 0.18, m2_dp["vr_scen"], width=0.35, label="Scenario-wise Weighted VR (vr_scen)")
ax.bar(x + 0.18, m2_dp["vr_mean"], width=0.35, label="Weighted-Mean Series VR (vr_mean — Double Penalty)")
ax.axhline(1.0, color="green", linestyle="--", label="Ideal VR = 1.0")
ax.set_xticks(x)
ax.set_xticklabels([f"L={int(g)}" for g in m2_dp["gap_length"]])
ax.set_ylabel("Variance Ratio")
ax.set_title("Double-Penalty Proof on Mode 2 (Trailing CERES Outage, Profile A)")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "fig03_double_penalty_vr.png", dpi=150)
plt.show()
            """
        ),
        _md(
            r"""
## 5. Phân rã Chân trời $h = 1 \dots 7$ (Profile B Top-K SHAP) & Đối chiếu `SC-31` vs `SC-30`

Kiểm tra phân rã sai số từng ngày dự báo $h=1\dots 7$ tại `L ∈ {3, 14, 90}` cho cả `SC-31` (F3) và `SC-30` (F2).
            """
        ),
        _code(
            """
h_prof_b = horizon_df[
    (horizon_df["model_id"] == "SC-31")
    & (horizon_df["position"] == "trailing")
    & (horizon_df["profile"] == "B")
    & (horizon_df["gap_length"].isin([3, 14, 90]))
    & (horizon_df["method"].isin(["ffill", "doy_climatology", "custom_mode1", "custom_mode2", "custom_hybrid"]))
]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, L in zip(axes, [3, 14, 90]):
    sub = h_prof_b[h_prof_b["gap_length"] == L]
    for m in ["ffill", "doy_climatology", "custom_mode1", "custom_mode2", "custom_hybrid"]:
        ms = sub[sub["method"] == m].sort_values("horizon")
        ax.plot(ms["horizon"], ms["fidelity_mae"], marker="o", label=m)
    ax.set_title(f"Profile B Trailing L={L}d — Fidelity MAE by Horizon")
    ax.set_xlabel("Horizon h (days)")
    ax.set_xticks(range(1, 8))
axes[0].set_ylabel("Fidelity-MAE vs y_full (mm/day)")
axes[-1].legend(fontsize=8)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig04_horizon_breakdown_profile_b.png", dpi=150)
plt.show()

comp_models = (
    summary_df[(summary_df["gap_length"] > 0) & (summary_df["position"] == "trailing") & (summary_df["method"].isin(["doy_climatology", "custom_mode1", "custom_mode2", "custom_hybrid"]))]
    .groupby(["model_id", "profile", "method"], as_index=False)
    .agg(mean_mae=("mae", "mean"), mean_fidelity_mae=("fidelity_mae", "mean"), mean_delta_mae_pct=("delta_mae_pct", "mean"))
)
display(comp_models)
            """
        ),
        _md(
            r"""
## 6. Ablation `Mode 1` vs `Mode 2` vs `Hybrid`, Quét Ngưỡng $L^* \in \{3, 5, 7, 14\}$ & Kiểm định Tin cậy $\sigma_{\text{imp}}$
            """
        ),
        _code(
            """
sens_tbl = (
    sensitivity_df[(sensitivity_df["model_id"] == "SC-31") & (sensitivity_df["gap_length"] > 0)]
    .groupby(["position", "threshold_l_star"], as_index=False)
    .agg(
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_fidelity_mae=("fidelity_mae", "mean"),
        mean_delta_mae_pct=("delta_mae_pct", "mean"),
    )
)
display(sens_tbl)

sigma_rel = sc31[sc31["method"].isin(["custom_mode2", "custom_hybrid"]) & sc31["spearman_sigma_rho"].notna()][
    ["position", "profile", "gap_length", "method", "spearman_sigma_rho", "spearman_sigma_p"]
]
display(sigma_rel.sort_values(["position", "profile", "gap_length"]))
            """
        ),
        _md(
            """
## 7. Kiểm định Ý nghĩa Thống kê (Wilcoxon + Holm–Bonferroni) & Hồ sơ SLA Tài nguyên

Kiểm định Wilcoxon ghép cặp theo anchor giữa `custom_hybrid` và đối thủ mạnh nhất ở Tầng 3 (`doy_climatology` / `sklearn_iterative` / `sklearn_knn`) cùng độ trễ `P50/P95 (ms)` tách riêng bước điền khuyết vs `FE + Predict`.
            """
        ),
        _code(
            """
display(wilcoxon_df[wilcoxon_df["model_id"] == "SC-31"].sort_values(["position", "profile", "gap_length"]))

sla_tbl = (
    sc31[(sc31["position"] == "trailing") & (sc31["profile"] == "A") & (sc31["gap_length"].isin([7, 30, 90]))]
    [["gap_length", "method", "impute_ms_p50", "impute_ms_p95", "fe_predict_ms_p50", "total_ms_p50", "total_ms_p95"]]
    .sort_values(["gap_length", "total_ms_p50"])
)
display(sla_tbl)

print("Resource SLA summary:", json.dumps(meta["verification_gates"]["gates"]["5_memory_and_runtime_sanity"], indent=2))
            """
        ),
    ]

    nb = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbformat.write(nb, notebook_path)
    return notebook_path


def execute_notebook(notebook_path: Path = NOTEBOOK_PATH, timeout: int = 180) -> None:
    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
    client.execute()
    nbformat.write(nb, notebook_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build & execute imputation_based_inference.ipynb")
    parser.add_argument("--execute", action="store_true", help="Execute generated notebook via NotebookClient")
    args = parser.parse_args()
    path = build_notebook()
    print(f"Built notebook at {path}")
    if args.execute:
        execute_notebook(path)
        print(f"Executed notebook cleanly at {path}")
