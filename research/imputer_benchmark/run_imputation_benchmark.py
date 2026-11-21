"""Deterministic CLI Runner for Single-Location Exogenous Imputation Benchmark (DS108)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.imputer_benchmark.imputation_benchmark import (  # noqa: E402
    ImputationBenchmarkConfig,
    load_ceres_syn1deg_vi_columns,
    measure_isolated_single_anchor_rss,
    run_benchmark_execution,
)
from research.imputer_benchmark.imputation_reporting import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    build_summary_tables,
    compute_wilcoxon_table,
    evaluate_acceptance_gates,
    write_benchmark_artifacts,
)
from research.imputer_benchmark.exogenous_imputer import CANONICAL_EXOGENOUS_COLUMNS  # noqa: E402


def _refresh_from_saved(output_dir: Path, cfg: ImputationBenchmarkConfig) -> int:
    """Recompute Wilcoxon tests, isolated RSS Gate 5, metadata JSON & Markdown report from saved Parquet/CSVs."""
    meta = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    raw_df = pd.read_parquet(output_dir / "benchmark_raw_predictions.parquet")
    summary_df = pd.read_csv(output_dir / "benchmark_summary_metrics.csv")
    horizon_df = pd.read_csv(output_dir / "benchmark_horizon_breakdown.csv")
    sensitivity_df = pd.read_csv(output_dir / "benchmark_hybrid_sensitivity.csv")
    shap_df = pd.read_csv(output_dir / "shap_exogenous_ranking.csv")

    isolated_rss = measure_isolated_single_anchor_rss(cfg)
    old_gate5 = meta["verification_gates"]["gates"]["5_memory_and_runtime_sanity"]
    benchmark_out = {
        "config": cfg,
        "raw_df": raw_df,
        "sensitivity_raw_df": sensitivity_df,
        "shap_sc31": shap_df,
        "anchors_df": pd.DataFrame(meta["anchors"]),
        "profiles_map": {
            "A": load_ceres_syn1deg_vi_columns(cfg.catalog_path),
            "B": meta["profile_b_top_k_shap_columns"],
            "C": list(CANONICAL_EXOGENOUS_COLUMNS),
        },
        "p90_dev_train": meta["p90_dev_train_mm"],
        "dev_train_range": tuple(meta["dev_train_range"]),
        "test_range": tuple(meta["test_range"]),
        "gate_nan_failures": [],
        "peak_rss_mb": old_gate5.get("benchmark_runner_peak_rss_mb", old_gate5.get("peak_rss_mb", 0.0)),
        "heavy_single_anchor_rss_mb": isolated_rss["rss_l90_profile_c_mb"],
        "isolated_rss_profile_a_mode2_mb": isolated_rss["rss_l90_profile_a_mode2_mb"],
    }
    wilcoxon_df = compute_wilcoxon_table(raw_df, summary_df, cfg)
    tables = {
        "summary_df": summary_df,
        "horizon_df": horizon_df,
        "sensitivity_df": sensitivity_df,
        "wilcoxon_df": wilcoxon_df,
    }
    gates = evaluate_acceptance_gates(benchmark_out, summary_df)
    write_benchmark_artifacts(benchmark_out, tables, gates, output_dir=output_dir)

    print("[BENCHMARK] Refreshed Gates:", flush=True)
    for gname, gval in gates["gates"].items():
        print(f"  - {gname}: {gval}", flush=True)
    return 0 if gates["all_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Single-Location Exogenous Imputer Benchmark")
    parser.add_argument("--n-anchors", type=int, default=48, help="Number of evenly-spaced Test anchors")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv-path", type=Path, default=None, help="Optional raw/clean CSV override")
    parser.add_argument(
        "--refresh-from-saved",
        action="store_true",
        help="Re-run isolated RSS probe, Wilcoxon tests, gates & Markdown report from saved Parquet/CSVs",
    )
    args = parser.parse_args()

    cfg = ImputationBenchmarkConfig(
        n_anchors=args.n_anchors,
        csv_path=args.csv_path,
    )
    if args.refresh_from_saved:
        return _refresh_from_saved(args.output_dir, cfg)

    t0 = time.perf_counter()
    print(f"[BENCHMARK] Starting HCMC Daily Imputation Benchmark (n_anchors={cfg.n_anchors})...", flush=True)
    benchmark_out = run_benchmark_execution(cfg)
    print(
        f"[BENCHMARK] Matrix finished in {time.perf_counter() - t0:.1f}s "
        f"({len(benchmark_out['raw_df'])} raw rows; Isolated L=90 RSS={benchmark_out['heavy_single_anchor_rss_mb']:.1f} MB)",
        flush=True,
    )

    tables = build_summary_tables(benchmark_out)
    gates = evaluate_acceptance_gates(benchmark_out, tables["summary_df"])
    paths = write_benchmark_artifacts(benchmark_out, tables, gates, output_dir=args.output_dir)

    print("[BENCHMARK] Acceptance Gates:", flush=True)
    for gname, gval in gates["gates"].items():
        print(f"  - {gname}: {gval}", flush=True)
    print(f"[BENCHMARK] Artifacts written to: {args.output_dir}", flush=True)
    for k, p in paths.items():
        print(f"  - {k}: {p}", flush=True)

    if not gates["all_passed"]:
        print("[BENCHMARK] ERROR: One or more automated verification gates FAILED!", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
