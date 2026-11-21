"""Unit tests for exogenous imputation benchmark suite, baselines, Double-Penalty metrics & gates."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.vintage_store import CANONICAL_VI_COLUMN_ORDER  # noqa: E402
from research.imputer_benchmark.imputation_baselines import (  # noqa: E402
    ALL_BENCHMARK_METHODS,
    ImputationBaselineRegistry,
    ImputationExecutionResult,
    compute_domain_violation_rate,
    validate_dev_train_only,
)
from research.imputer_benchmark.imputation_benchmark import (  # noqa: E402
    ImputationBenchmarkConfig,
    apply_block_mask,
    compute_intrinsic_metrics,
    compute_shap_exogenous_ranking,
    run_benchmark_execution,
)
from research.imputer_benchmark.imputation_reporting import (  # noqa: E402
    build_summary_tables,
    evaluate_acceptance_gates,
    holm_bonferroni_adjust,
)
from src.inference.validation import PHYSICAL_METEO_BOUNDS  # noqa: E402


def _make_synthetic_df(start: str, n_days: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    doy = dates.dayofyear.to_numpy(dtype=np.float64)
    phase = 2.0 * np.pi * (doy - 1.0) / 365.25
    data: dict[str, object] = {"Ngày": dates}
    for col in CANONICAL_VI_COLUMN_ORDER[1:]:
        lo, hi = PHYSICAL_METEO_BOUNDS[col]
        if col == "Lượng mưa":
            data[col] = np.clip(np.maximum(0.0, 5.0 * (1.0 - np.cos(phase)) + rng.gamma(1.3, 2.0, n_days) - 2.0), lo, hi)
        elif col == "Nhiệt độ 2m":
            data[col] = np.clip(28.0 + 2.0 * np.sin(phase) + rng.normal(0.0, 0.7, n_days), lo, hi)
        elif col == "Điểm sương 2m":
            t2m = np.asarray(data["Nhiệt độ 2m"], dtype=np.float64)
            data[col] = np.clip(t2m - 4.0 + rng.normal(0.0, 0.5, n_days), lo, np.minimum(hi, t2m))
        else:
            mid = 0.5 * (lo + hi)
            amp = 0.12 * (hi - lo)
            data[col] = np.clip(mid + amp * np.sin(phase) + rng.normal(0.0, 0.04 * (hi - lo), n_days), lo, hi)
    return pd.DataFrame(data).loc[:, list(CANONICAL_VI_COLUMN_ORDER)]


def test_leakage_guard_not_applicable_rules_and_l0_identity():
    leaky = _make_synthetic_df("2018-04-01", 60)
    with pytest.raises(ValueError, match="DATA_LEAKAGE_GUARD"):
        validate_dev_train_only(leaky)

    m31 = joblib.load(PROJECT_ROOT / "artifacts" / "models" / "sc31_lightgbm_tweedie_f3.joblib")
    fb3 = joblib.load(PROJECT_ROOT / "artifacts" / "models" / "feature_builder_f3.joblib")
    train_df = _make_synthetic_df("2012-01-01", 1900, seed=10)
    registry = ImputationBaselineRegistry.fit_on_dev_train(train_df, model=m31, feature_builder=fb3)

    clean_win = _make_synthetic_df("2025-01-01", 120, seed=11)
    masked_trail, _ = apply_block_mask(
        clean_win, gap_position="trailing", gap_length=14, masked_cols=["Nhiệt độ 2m"]
    )
    res_lin = registry.impute_window(
        "linear_interp", masked_trail, gap_position="trailing", gap_length=14, profile_id="B"
    )
    assert res_lin.status == "NOT_APPLICABLE"

    masked_c, _ = apply_block_mask(
        clean_win, gap_position="interior", gap_length=14, masked_cols=list(registry.exo_cols)
    )
    res_iter = registry.impute_window(
        "sklearn_iterative", masked_c, gap_position="interior", gap_length=14, profile_id="C"
    )
    assert res_iter.status == "NOT_APPLICABLE"

    # Every valid method returns identical clean window at L=0
    for method in ALL_BENCHMARK_METHODS:
        res_l0 = registry.impute_window(
            method, clean_win, gap_position="trailing", gap_length=0, profile_id="A"
        )
        assert res_l0.status == "OK"
        assert np.allclose(
            res_l0.scenarios[0][list(registry.exo_cols)].to_numpy(),
            clean_win[list(registry.exo_cols)].to_numpy(),
            atol=1e-12,
        )


def test_double_penalty_branch1_vr_and_holm_bonferroni():
    clean_win = _make_synthetic_df("2025-01-01", 60, seed=19)
    cols = ["Nhiệt độ 2m", "Độ ẩm tương đối 2m"]
    masked, bool_mask = apply_block_mask(clean_win, gap_position="trailing", gap_length=14, masked_cols=cols)

    # Two opposing scenarios whose average flattens variance, demonstrating Double-Penalty VR contrast
    s1 = clean_win.copy()
    s2 = clean_win.copy()
    delta = np.sin(np.linspace(0, 4 * np.pi, 14)) * 3.0
    s1.loc[s1.index[-14:], "Nhiệt độ 2m"] = clean_win["Nhiệt độ 2m"].iloc[-14:].to_numpy() + delta
    s2.loc[s2.index[-14:], "Nhiệt độ 2m"] = clean_win["Nhiệt độ 2m"].iloc[-14:].to_numpy() - delta

    exec_res = ImputationExecutionResult(
        method="custom_mode2",
        status="OK",
        reason=None,
        scenarios=[s1, s2],
        weights=np.array([0.5, 0.5]),
        effective_k=2,
        imputation_mode="MODE_2_ANALOG_MARGINALIZATION",
        selected_analog_dates=["2014-01-01", "2015-01-01"],
    )
    stds = {c: 1.0 for c in cols}
    metrics = compute_intrinsic_metrics(
        clean_window=clean_win,
        exec_result=exec_res,
        masked_mask=bool_mask,
        masked_cols=cols,
        train_std_by_col=stds,
        gap_length=14,
    )
    assert metrics["vr_scen"] > metrics["vr_mean"]
    assert compute_domain_violation_rate([s1, s2], np.array([0.5, 0.5]), bool_mask, cols) == 0.0

    # Holm-Bonferroni adjustment check against exact textbook calculation
    adj = holm_bonferroni_adjust([0.005, 0.04, 0.02])
    assert np.allclose(adj, [0.015, 0.04, 0.04], atol=1e-12)


def test_mini_end_to_end_benchmark_and_all_five_gates():
    demo_db = PROJECT_ROOT / "artifacts" / "models" / "demo_rainfall_vintage.db"
    if not demo_db.exists():
        pytest.skip("demo_rainfall_vintage.db absent.")

    m31 = joblib.load(PROJECT_ROOT / "artifacts" / "models" / "sc31_lightgbm_tweedie_f3.joblib")
    fb3 = joblib.load(PROJECT_ROOT / "artifacts" / "models" / "feature_builder_f3.joblib")
    train_sample = _make_synthetic_df("2012-01-01", 90, seed=7)
    shap_df = compute_shap_exogenous_ranking(model=m31, feature_builder=fb3, dev_train_df=train_sample, model_id="SC-31")
    assert len(shap_df) == 34

    # Compact 3-anchor smoke run checking full pipeline & all 5 verification gates
    cfg = ImputationBenchmarkConfig(
        n_anchors=3,
        gap_lengths=(0, 3, 14),
        gap_positions=("trailing", "interior"),
        profiles=("A", "B", "C"),
    )
    out = run_benchmark_execution(cfg)
    tables = build_summary_tables(out)
    gates = evaluate_acceptance_gates(out, tables["summary_df"])
    assert gates["all_passed"] is True, f"Gates failed: {gates}"
    assert set(tables["summary_df"]["model_id"].unique()) == {"SC-31", "SC-30"}
