"""Unit and artifact-contract tests for Stage 1 H=11 Horizon Extension Feasibility Audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.horizon_extension.run_h11_feasibility_audit import (  # noqa: E402
    DEFAULT_JSON_OUT,
    DEFAULT_README_OUT,
    EXPECTED_COUNTS,
    SLICE_SPECS,
    _assert_no_purged_columns_in_report,
    _build_dev_train_rain_doy_table,
    _compute_window_metrics,
    _evaluate_gate,
    _predict_climatology_doy_matrix,
    _predict_persistence_matrix,
)
from src.inference.forecast_runner import build_block_bootstrap_indices  # noqa: E402


def _synthetic_window(
    mae: float,
    d_mae: float,
    d_ci_hi: float,
    d_rmse: float,
    d_p90: float,
) -> dict:
    return {
        "MAE": mae,
        "RMSE": 6.5,
        "MSE": 42.25,
        "MASE": mae / 3.7,
        "p90_tail_mae": 14.5,
        "fidelity_mae": 0.1,
        "delta_MAE_vs_clim": d_mae,
        "delta_MAE_vs_clim_ci95": {
            "point_estimate": d_mae,
            "ci_lower": d_mae - 0.15,
            "ci_upper": d_ci_hi,
            "ci_width": abs(d_ci_hi - (d_mae - 0.15)),
            "bootstrap_se": 0.05,
        },
        "delta_RMSE_vs_clim": d_rmse,
        "delta_RMSE_vs_clim_ci95": {
            "point_estimate": d_rmse,
            "ci_lower": d_rmse - 0.10,
            "ci_upper": d_rmse + 0.10,
            "ci_width": 0.20,
            "bootstrap_se": 0.05,
        },
        "delta_p90_tail_mae_vs_clim": d_p90,
    }


def _build_synthetic_split_pc(
    *,
    h1_to_7_mae: float = 3.17,
    step_d_mae: float = -0.30,
    slice_d_mae: float = -0.30,
    slice_ci_hi: float = -0.10,
    slice_d_rmse: float = -0.15,
    slice_d_p90: float = -0.05,
    h11_step_override: dict | None = None,
    h10_step_override: dict | None = None,
    h8_to_11_override: dict | None = None,
    h8_to_10_override: dict | None = None,
) -> dict:
    per_step = {
        f"h{h}": _synthetic_window(h1_to_7_mae, step_d_mae, slice_ci_hi, slice_d_rmse, slice_d_p90)
        for h in range(1, 12)
    }
    if h10_step_override is not None:
        per_step["h10"] = h10_step_override
    if h11_step_override is not None:
        per_step["h11"] = h11_step_override
    slices = {
        sk: _synthetic_window(h1_to_7_mae, slice_d_mae, slice_ci_hi, slice_d_rmse, slice_d_p90)
        for sk in SLICE_SPECS
    }
    if h8_to_10_override is not None:
        slices["h8_to_10"] = h8_to_10_override
    if h8_to_11_override is not None:
        slices["h8_to_11"] = h8_to_11_override
    return {
        "SC-35": {"per_step": per_step, "slices": slices},
        "SC-31": {"per_step": per_step, "slices": slices},
        "SC-30": {"per_step": per_step, "slices": slices},
    }


def test_evaluate_gate_selects_h11_when_all_clauses_pass() -> None:
    val_pc = _build_synthetic_split_pc()
    test_pc = _build_synthetic_split_pc()
    h7_mae = {"SC-35": 3.175, "SC-31": 3.175, "SC-30": 3.175}
    gate = _evaluate_gate(val_pc=val_pc, test_pc=test_pc, h7_val_pc_mae=h7_mae)
    assert gate["clause_A_h1_to_7_non_regression"]["passed"] is True
    assert gate["winning_H_star"] == 11
    assert gate["stop_and_report_triggered"] is False


def test_evaluate_gate_falls_back_to_h10_when_only_h11_fails_tail() -> None:
    val_pc = _build_synthetic_split_pc(
        h8_to_11_override=_synthetic_window(3.30, -0.30, -0.10, -0.10, +0.20),
    )
    test_pc = _build_synthetic_split_pc(
        h8_to_11_override=_synthetic_window(3.80, -0.20, -0.05, -0.05, +0.30),
    )
    h7_mae = {"SC-35": 3.175, "SC-31": 3.175, "SC-30": 3.175}
    gate = _evaluate_gate(val_pc=val_pc, test_pc=test_pc, h7_val_pc_mae=h7_mae)
    assert gate["candidates"]["H_star_11"]["overall_gate_passed"] is False
    assert gate["candidates"]["H_star_10"]["overall_gate_passed"] is True
    assert gate["winning_H_star"] == 10
    assert gate["stop_and_report_triggered"] is False


def test_evaluate_gate_clause_b1_path2_step_bound_pass_and_fail() -> None:
    h7_mae = {"SC-35": 3.175, "SC-31": 3.175, "SC-30": 3.175}
    # Step h11 loses by +0.015 <= +0.020 while slice 95% CI upper < 0 -> Path 2 passes
    val_pass = _build_synthetic_split_pc(
        h11_step_override=_synthetic_window(3.30, +0.015, +0.05, -0.10, -0.05),
    )
    gate_pass = _evaluate_gate(val_pc=val_pass, test_pc=_build_synthetic_split_pc(), h7_val_pc_mae=h7_mae)
    b1_pass = gate_pass["candidates"]["H_star_11"]["clause_B1"]
    assert b1_pass["path1_all_steps_beat_clim"] is False
    assert b1_pass["path2_slice_ci_upper_lt_0_and_step_le_0_020"] is True
    assert gate_pass["winning_H_star"] == 11

    # Step h11 loses by +0.025 > +0.020 while h10 stays negative -> H11 fails B1, falls back to H10
    val_fail = _build_synthetic_split_pc(
        h11_step_override=_synthetic_window(3.30, +0.025, +0.06, -0.10, -0.05),
    )
    gate_fail = _evaluate_gate(val_pc=val_fail, test_pc=_build_synthetic_split_pc(), h7_val_pc_mae=h7_mae)
    assert gate_fail["candidates"]["H_star_11"]["clause_B1"]["passed"] is False
    assert gate_fail["winning_H_star"] == 10


def test_evaluate_gate_fails_on_b2_rmse_or_b3_test_and_diagnostics_never_win() -> None:
    h7_mae = {"SC-35": 3.175, "SC-31": 3.175, "SC-30": 3.175}
    # H8 and H9 pass, while H10 & H11 fail B2 (delta_RMSE > 0) or B3 (test delta_MAE > 0)
    val_pc = _build_synthetic_split_pc(
        h8_to_10_override=_synthetic_window(3.30, -0.30, -0.10, +0.01, -0.05),
        h8_to_11_override=_synthetic_window(3.30, -0.30, -0.10, -0.10, -0.05),
    )
    test_pc = _build_synthetic_split_pc(
        h8_to_11_override=_synthetic_window(3.80, +0.05, +0.15, -0.05, -0.05),
    )
    gate = _evaluate_gate(val_pc=val_pc, test_pc=test_pc, h7_val_pc_mae=h7_mae)
    assert gate["candidates"]["H_star_10"]["clause_B2"]["passed"] is False
    assert gate["candidates"]["H_star_11"]["clause_B3"]["passed"] is False
    assert gate["candidates"]["H_star_9"]["overall_gate_passed"] is True
    assert gate["candidates"]["H_star_8"]["overall_gate_passed"] is True
    assert gate["winning_H_star"] is None
    assert gate["stop_and_report_triggered"] is True


def test_evaluate_gate_stops_on_clause_a_regression() -> None:
    val_pc = _build_synthetic_split_pc(h1_to_7_mae=3.20)
    test_pc = _build_synthetic_split_pc(h1_to_7_mae=3.20)
    h7_mae = {"SC-35": 3.170, "SC-31": 3.200, "SC-30": 3.200}
    gate = _evaluate_gate(val_pc=val_pc, test_pc=test_pc, h7_val_pc_mae=h7_mae)
    assert gate["clause_A_h1_to_7_non_regression"]["passed"] is False
    assert gate["winning_H_star"] is None
    assert gate["stop_and_report_triggered"] is True


def test_baseline_helpers_doy_persistence_leap_fallback_and_cutoff() -> None:
    # No day 366 in training sample -> should fall back 365 -> 366 cleanly
    df = pd.DataFrame(
        {
            "Ngày": pd.to_datetime(["2015-01-01", "2015-01-02", "2015-12-30", "2015-12-31"]),
            "Lượng mưa": [2.0, 4.0, 6.0, 8.0],
        }
    )
    doy_table = _build_dev_train_rain_doy_table(df)
    assert doy_table[1] == 2.0
    assert doy_table[365] == 8.0
    assert doy_table[366] == 8.0

    origins = pd.Series(pd.to_datetime(["2015-12-31"]))
    clim_mat = _predict_climatology_doy_matrix(doy_table, origins, horizon=2)
    assert np.allclose(clim_mat, [[2.0, 4.0]])

    pers_mat = _predict_persistence_matrix(np.array([5.5]), horizon=3)
    assert np.allclose(pers_mat, [[5.5, 5.5, 5.5]])

    # Leakage guard rejects post-Dev-Train date
    bad_df = pd.DataFrame({"Ngày": pd.to_datetime(["2018-05-01"]), "Lượng mưa": [1.0]})
    with pytest.raises(ValueError, match="Dev Train DOY cutoff violation"):
        _build_dev_train_rain_doy_table(bad_df)


def test_window_metrics_empty_p90_and_purged_metrics_rejection() -> None:
    y_true = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0], [0.5, 1.5]], dtype=np.float64)
    y_pred = np.array([[1.5, 2.5], [2.5, 3.5], [2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    y_clim = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    boot_idx = build_block_bootstrap_indices(len(y_true), block_length=2, n_bootstrap=50, seed=42)

    m = _compute_window_metrics(
        y_true,
        y_pred,
        y_pred,
        y_clim,
        h_slice=slice(0, 2),
        p90_thresh=11.086,
        boot_idx_matrix=boot_idx,
    )
    assert m["p90_tail_mae"] is None
    assert m["delta_p90_tail_mae_vs_clim"] is None
    assert "delta_RMSE_vs_clim_ci95" in m
    assert abs(m["delta_RMSE_vs_clim_ci95"]["point_estimate"] - m["delta_RMSE_vs_clim"]) < 1e-6

    bad_report = {"val": {"clean": {"SC-35": {"per_step": {"h1": {"CR-MAE@H": 1.2}}, "slices": {}}}}}
    with pytest.raises(Exception):
        _assert_no_purged_columns_in_report(bad_report)


def test_stage1_audit_artifact_contract() -> None:
    assert DEFAULT_JSON_OUT.exists(), f"Missing audit JSON: {DEFAULT_JSON_OUT}"
    assert DEFAULT_README_OUT.exists(), f"Missing audit README: {DEFAULT_README_OUT}"

    report = json.loads(DEFAULT_JSON_OUT.read_text(encoding="utf-8"))
    for split_name, (exp_n, exp_last) in EXPECTED_COUNTS.items():
        assert report["splits"][split_name]["n_origins"] == exp_n
        assert report["splits"][split_name]["last_origin"] == exp_last

    assert report["bootstrap"]["block_length"] == 11
    assert report["bootstrap"]["n_bootstrap"] == 2000
    assert report["bootstrap"]["seed"] == 42
    assert report["pc_transform_max_parity_diff"] < 1e-12
    assert "delta_RMSE_vs_clim_ci95" in (
        report["metrics"]["val"]["p_c_doy_median_all_11_solar"]["SC-35"]["slices"]["h8_to_11"]
    )

    _assert_no_purged_columns_in_report(report["metrics"])

    gate = report["go_no_go_gate"]
    assert gate["stop_and_report_triggered"] is True
    assert gate["winning_H_star"] is None
    assert gate["clause_A_h1_to_7_non_regression"]["passed"] is True
    assert gate["candidates"]["H_star_11"]["overall_gate_passed"] is False
    assert gate["candidates"]["H_star_10"]["overall_gate_passed"] is False


def test_canonical_70_and_65_feature_contract() -> None:
    artifact_dir = PROJECT_ROOT / "artifacts" / "models"
    feat_names = json.loads((PROJECT_ROOT / "artifacts" / "feature_names.json").read_text(encoding="utf-8"))
    all_70 = feat_names["all_70_features"]
    assert len(all_70) == 70

    m35 = joblib.load(artifact_dir / "sc35_xgboost_tweedie_f3.joblib")
    m31 = joblib.load(artifact_dir / "sc31_lightgbm_tweedie_f3.joblib")
    m30 = joblib.load(artifact_dir / "sc30_lightgbm_tweedie_f2.joblib")
    assert all(est.n_features_in_ == 70 for est in m35.estimators_)
    assert all(est.n_features_in_ == 70 for est in m31.estimators_)
    assert all(est.n_features_in_ == 65 for est in m30.estimators_)

    fb3 = joblib.load(artifact_dir / "feature_builder_f3.joblib")
    fb2 = joblib.load(artifact_dir / "feature_builder_f2.joblib")
    sample_df = pd.read_csv(artifact_dir / "sample_recent_history_35d.csv")
    from src.data.vintage_store import align_and_validate_canonical_raw_df  # noqa: E402

    raw_df = align_and_validate_canonical_raw_df(sample_df)
    f3_cols = [c for c in fb3.transform(raw_df, drop_warmup=True).columns if c not in ("Ngày", "Lượng mưa")]
    f2_cols = [c for c in fb2.transform(raw_df, drop_warmup=True).columns if c not in ("Ngày", "Lượng mưa")]
    assert len(f3_cols) == 70
    assert len(f2_cols) == 65
    assert f3_cols == all_70
    assert f3_cols[:65] == f2_cols

    readme_text = DEFAULT_README_OUT.read_text(encoding="utf-8")
    for expected in ("(6289, 70)", "(720, 70)", "(2180, 70)", "(6289, 65)", "(720, 65)", "(2180, 65)"):
        assert expected in readme_text, f"Missing canonical matrix shape {expected} in README.md"
    assert "Scientific Diagnostics & Roadmap v2.0 Conclusion" in readme_text
