"""Unit and verification tests for standalone inference-time exogenous imputer."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import time_series_split  # noqa: E402
from src.data.vintage_store import (  # noqa: E402
    CANONICAL_VI_COLUMN_ORDER,
    connect_sqlite,
    read_latest_observations,
)
from research.imputer_benchmark.exogenous_imputer import (  # noqa: E402
    MODE_0_CLEAN,
    MODE_1_OU_CLIMATOLOGY,
    MODE_2_ANALOG_MARGINALIZATION,
    MODE_3_UNCONDITIONAL_SEASONAL_FALLBACK,
    ClimatologyLookupTable,
    ExogenousImputationConfig,
    ExogenousImputerWrapper,
    GapInspector,
    HistoricalAnalogStore,
)
from src.inference.predictor import StandaloneRainfallPredictor  # noqa: E402
from src.inference.validation import PHYSICAL_METEO_BOUNDS  # noqa: E402

CERES_SYN1DEG_VI_COLS = [
    "Bức xạ sóng ngắn bề mặt",
    "Bức xạ đỉnh khí quyển",
    "Bức xạ sóng dài xuống",
    "Bức xạ sóng ngắn trời quang",
    "Bức xạ trực tiếp pháp tuyến",
    "Bức xạ khuếch tán",
    "Bức xạ quang hợp tổng",
    "Bức xạ quang hợp trời quang",
    "Bức xạ UVA",
    "Bức xạ UVB",
    "Chỉ số UV",
]


def _make_synthetic_canonical_df(
    start_date: str,
    n_days: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic canonical 36-column weather DataFrame with annual seasonality."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    doy = dates.dayofyear.to_numpy(dtype=np.float64)
    phase = 2.0 * np.pi * (doy - 1.0) / 365.25

    data: dict[str, object] = {"Ngày": dates}
    for col in CANONICAL_VI_COLUMN_ORDER[1:]:
        lo, hi = PHYSICAL_METEO_BOUNDS[col]
        mid = 0.5 * (lo + hi)
        amp = 0.15 * (hi - lo)
        if col == "Lượng mưa":
            rain = np.maximum(0.0, 6.0 * (1.0 - np.cos(phase)) + rng.gamma(1.5, 2.5, size=n_days) - 3.0)
            data[col] = np.clip(rain, lo, hi)
        elif col == "Độ ẩm tương đối 2m":
            data[col] = np.clip(75.0 - 10.0 * np.cos(phase) + rng.normal(0.0, 3.0, n_days), lo, hi)
        elif col == "Nhiệt độ 2m":
            data[col] = np.clip(28.0 + 2.5 * np.sin(phase) + rng.normal(0.0, 0.8, n_days), lo, hi)
        elif col == "Điểm sương 2m":
            t2m = np.asarray(data["Nhiệt độ 2m"], dtype=np.float64)
            data[col] = np.clip(t2m - 4.0 + rng.normal(0.0, 0.8, n_days), lo, np.minimum(hi, t2m))
        else:
            vals = mid + amp * np.sin(phase + (hash(col) % 7) * 0.4) + rng.normal(0.0, 0.05 * (hi - lo), n_days)
            data[col] = np.clip(vals, lo, hi)
    return pd.DataFrame(data).loc[:, list(CANONICAL_VI_COLUMN_ORDER)]


def _direct_build_single_step_predict(model: object, feature_builder: object, df: pd.DataFrame) -> np.ndarray:
    feat_df = feature_builder.build_single_step(df)
    feat_cols = [c for c in feat_df.columns if c not in (feature_builder.date_col, feature_builder.target_col)]
    raw_preds = model.predict(feat_df[feat_cols].values.astype(np.float64))[0]
    return np.clip(raw_preds, 0.0, None)


@pytest.fixture(scope="module")
def serving_predictor() -> StandaloneRainfallPredictor:
    return StandaloneRainfallPredictor(mode="local")


@pytest.fixture(scope="module")
def synthetic_dev_train_bundle():
    cfg = ExogenousImputationConfig()
    train_df = _make_synthetic_canonical_df("2011-01-01", n_days=2650, seed=41)
    clim = ClimatologyLookupTable.fit(train_df, config=cfg)
    store = HistoricalAnalogStore.from_dataframe(train_df, config=cfg)
    return train_df, clim, store, cfg


def test_climatology_fit_circular_smoothing_leap_day_and_leakage_guard(synthetic_dev_train_bundle):
    _, table, _, cfg = synthetic_dev_train_bundle

    assert len(table.exogenous_columns) == 34
    assert "Lượng mưa" not in table.exogenous_columns
    for col in table.exogenous_columns:
        clim = table.climatology[col]
        assert clim.shape == (365,)
        assert np.all(np.isfinite(clim))
        assert 0.05 <= table.rho1[col] <= 0.98
        assert table.tau[col] > 0.0
        assert table.std[col] > 0.0
        assert table.iqr[col] > 0.0

    # Feb 29 (leap day) resolves cleanly to a finite climatology value without error
    assert np.isfinite(table.climatology_value("Nhiệt độ 2m", pd.Timestamp("2024-02-29")))

    # Leakage prevention: Validation/Test dates (> 2018-04-30) must fail closed
    leaky_df = _make_synthetic_canonical_df("2018-04-20", n_days=60, seed=9)
    with pytest.raises(ValueError, match="2018-04-30"):
        ClimatologyLookupTable.fit(leaky_df, config=cfg)


def test_mode0_clean_zero_overhead_parity_and_rejection_guards(
    serving_predictor: StandaloneRainfallPredictor,
    synthetic_dev_train_bundle,
):
    _, clim, store, cfg = synthetic_dev_train_bundle
    clean_history = pd.read_csv(PROJECT_ROOT / "artifacts" / "models" / "sample_recent_history_35d.csv")

    expected_preds = _direct_build_single_step_predict(
        serving_predictor.model_serving,
        serving_predictor.fb_f3,
        clean_history,
    )
    wrapper = ExogenousImputerWrapper(
        model=serving_predictor.model_serving,
        feature_builder=serving_predictor.fb_f3,
        climatology=clim,
        analog_store=store,
        config=cfg,
    )
    res = wrapper.predict(clean_history)

    assert res["imputation_mode"] == MODE_0_CLEAN
    assert res["max_gap_length"] == 0
    assert res["missing_variables"] == []
    assert res["low_confidence_warning"] is False
    assert res["phase_radius_used"] is None
    assert float(np.max(np.abs(np.asarray(res["predictions_7d_mm"]) - expected_preds))) == 0.0

    # Target missing / negative target / calendar gap rejection
    missing_target = clean_history.copy()
    missing_target.loc[missing_target.index[-1], "Lượng mưa"] = np.nan
    with pytest.raises(ValueError, match="Lượng mưa"):
        wrapper.predict(missing_target)

    negative_target = clean_history.copy()
    negative_target.loc[negative_target.index[-1], "Lượng mưa"] = -1.5
    with pytest.raises(ValueError, match="negative"):
        wrapper.predict(negative_target)

    gapped_dates = clean_history.drop(index=10).reset_index(drop=True)
    with pytest.raises(ValueError):
        GapInspector.inspect(gapped_dates, config=cfg)

    # Mode routing checks (NASA sentinel -999 & gap lengths)
    mode1_df = clean_history.copy()
    mode1_df.loc[mode1_df.index[-3:], "Bức xạ sóng ngắn bề mặt"] = -999.0
    assert GapInspector.inspect(mode1_df, config=cfg).mode == MODE_1_OU_CLIMATOLOGY

    mode2_df = _make_synthetic_canonical_df("2025-01-01", n_days=120, seed=21)
    mode2_df.loc[mode2_df.index[-20:], "Bức xạ sóng ngắn bề mặt"] = np.nan
    assert GapInspector.inspect(mode2_df, config=cfg).mode == MODE_2_ANALOG_MARGINALIZATION

    mode3_df = _make_synthetic_canonical_df("2025-01-01", n_days=120, seed=22)
    mode3_df.loc[mode3_df.index[-95:], "Bức xạ sóng ngắn bề mặt"] = np.nan
    assert GapInspector.inspect(mode3_df, config=cfg).mode == MODE_3_UNCONDITIONAL_SEASONAL_FALLBACK


def test_mode1_ou_trailing_decay_interior_bridge_and_physical_bounds(
    serving_predictor: StandaloneRainfallPredictor,
    synthetic_dev_train_bundle,
):
    _, clim, store, cfg = synthetic_dev_train_bundle
    wrapper = ExogenousImputerWrapper(
        model=serving_predictor.model_serving,
        feature_builder=serving_predictor.fb_f3,
        climatology=clim,
        analog_store=store,
        config=cfg,
    )

    gap_df = _make_synthetic_canonical_df("2025-06-01", n_days=45, seed=32)
    idx_tail_anchor = len(gap_df) - 6
    gap_df.loc[gap_df.index[idx_tail_anchor], "Nhiệt độ 2m"] = (
        clim.climatology_value("Nhiệt độ 2m", gap_df.loc[idx_tail_anchor, "Ngày"]) + 6.0
    )
    gap_df.loc[gap_df.index[-5:], "Nhiệt độ 2m"] = np.nan

    gap_df.loc[gap_df.index[19], "Độ ẩm tương đối 2m"] = 99.5
    gap_df.loc[gap_df.index[24], "Độ ẩm tương đối 2m"] = 99.0
    gap_df.loc[gap_df.index[20:24], "Độ ẩm tương đối 2m"] = np.nan

    gap_df.loc[gap_df.index[-4:], "Điểm sương 2m"] = np.nan
    gap_df.loc[gap_df.index[-5], "Điểm sương 2m"] = gap_df.loc[gap_df.index[-5], "Nhiệt độ 2m"] - 0.1
    gap_df.loc[gap_df.index[-3:], "Bức xạ sóng ngắn bề mặt"] = np.nan
    gap_df.loc[gap_df.index[-4], "Bức xạ sóng ngắn bề mặt"] = 0.0

    filled_df, report = wrapper.transform_mode1_ou(gap_df)
    assert report.mode == MODE_1_OU_CLIMATOLOGY
    assert report.max_gap_length == 5

    anomalies_tail = np.array([
        filled_df.loc[i, "Nhiệt độ 2m"] - clim.climatology_value("Nhiệt độ 2m", filled_df.loc[i, "Ngày"])
        for i in range(idx_tail_anchor, len(filled_df))
    ])
    assert anomalies_tail[0] > anomalies_tail[1] > anomalies_tail[2] > anomalies_tail[5] > 0.0

    # Verify exact OU bridge analytical formula on interior gap
    t0, t1 = 19, 24
    l_gap = float(t1 - t0)
    tau_rh = clim.tau["Độ ẩm tương đối 2m"]
    a0 = float(gap_df.loc[t0, "Độ ẩm tương đối 2m"]) - clim.climatology_value("Độ ẩm tương đối 2m", gap_df.loc[t0, "Ngày"])
    a1 = float(gap_df.loc[t1, "Độ ẩm tương đối 2m"]) - clim.climatology_value("Độ ẩm tương đối 2m", gap_df.loc[t1, "Ngày"])
    for idx in range(20, 24):
        expected_a = (
            np.sinh(float(t1 - idx) / tau_rh) * a0 + np.sinh(float(idx - t0) / tau_rh) * a1
        ) / np.sinh(l_gap / tau_rh)
        expected_x = float(np.clip(clim.climatology_value("Độ ẩm tương đối 2m", filled_df.loc[idx, "Ngày"]) + expected_a, 0.0, 100.0))
        assert np.isclose(filled_df.loc[idx, "Độ ẩm tương đối 2m"], expected_x, atol=1e-9)

    assert (filled_df["Độ ẩm tương đối 2m"] >= 0.0).all() and (filled_df["Độ ẩm tương đối 2m"] <= 100.0).all()
    assert (filled_df["Bức xạ sóng ngắn bề mặt"] >= 0.0).all()
    assert (filled_df["Điểm sương 2m"] <= filled_df["Nhiệt độ 2m"] + 1e-12).all()

    res = wrapper.predict(gap_df)
    assert res["imputation_mode"] == MODE_1_OU_CLIMATOLOGY
    assert res["max_gap_length"] == 5
    assert len(res["predictions_7d_mm"]) == 7
    assert all(np.isfinite(v) and v >= 0.0 for v in res["predictions_7d_mm"])


@pytest.mark.parametrize(
    ("query_start", "gap_len"),
    [
        ("2025-10-15", 10),  # Boundary blending case (8 <= L <= 14)
        ("2025-11-10", 80),  # ~80-day CERES SYN1deg lag crossing Dec -> Jan into Q1
    ],
)
def test_mode2_joint_multivariate_analog_marginalization_causality_and_cross_year(
    serving_predictor: StandaloneRainfallPredictor,
    synthetic_dev_train_bundle,
    query_start: str,
    gap_len: int,
):
    train_df, clim, store, cfg = synthetic_dev_train_bundle
    wrapper = ExogenousImputerWrapper(
        model=serving_predictor.model_serving,
        feature_builder=serving_predictor.fb_f3,
        climatology=clim,
        analog_store=store,
        config=cfg,
    )

    query_df = _make_synthetic_canonical_df(query_start, n_days=120, seed=55)
    observed_sentinel = query_df.copy()
    observed_sentinel.loc[observed_sentinel.index[-gap_len:], CERES_SYN1DEG_VI_COLS] = np.nan
    # Co-occurring short 4-day gap on T2M when L=10 verifies boundary blend covers <=14d co-gaps
    if gap_len <= cfg.boundary_blend_max_gap_days:
        observed_sentinel.loc[observed_sentinel.index[-4:], "Nhiệt độ 2m"] = np.nan

    res = wrapper.predict(observed_sentinel)
    assert res["imputation_mode"] == MODE_2_ANALOG_MARGINALIZATION
    assert res["max_gap_length"] == gap_len
    assert res["effective_k"] == cfg.analog_top_k
    assert np.isfinite(res["analog_mean_distance"])

    for key in ("predictions_7d_mm", "weighted_mean_7d_mm", "weighted_median_7d_mm", "imputation_uncertainty_7d_mm"):
        arr = np.asarray(res[key], dtype=np.float64)
        assert arr.shape == (7,)
        assert np.all(np.isfinite(arr))
        assert np.all(arr >= 0.0)

    scenarios, weights, meta = wrapper.materialize_mode2_scenarios(observed_sentinel)
    assert len(scenarios) == cfg.analog_top_k
    assert np.isclose(np.sum(weights), 1.0, atol=1e-9)

    q_start_dt = pd.Timestamp(query_df["Ngày"].iloc[0]).normalize()
    for scen_df, cand_end_str in zip(scenarios, meta["selected_analog_dates"]):
        cand_end_dt = pd.Timestamp(cand_end_str).normalize()
        assert cand_end_dt < q_start_dt
        assert cand_end_dt <= pd.Timestamp("2018-04-30")

        for obs_col in ("Lượng mưa", "Độ ẩm tương đối 2m"):
            assert np.allclose(scen_df[obs_col].to_numpy(), observed_sentinel[obs_col].to_numpy(), atol=1e-12)

        if gap_len > cfg.boundary_blend_max_gap_days:
            cand_start_dt = (cand_end_dt - pd.DateOffset(days=gap_len - 1)).normalize()
            expected_slice = train_df[
                (train_df["Ngày"] >= cand_start_dt) & (train_df["Ngày"] <= cand_end_dt)
            ].reset_index(drop=True)
            assert len(expected_slice) == gap_len
            for rad_col in CERES_SYN1DEG_VI_COLS:
                assert np.allclose(scen_df[rad_col].iloc[-gap_len:].to_numpy(), expected_slice[rad_col].to_numpy(), atol=1e-9)
        else:
            cand_anchor_dt = (cand_end_dt - pd.DateOffset(days=gap_len)).normalize()
            expected_with_anchor = train_df[
                (train_df["Ngày"] >= cand_anchor_dt) & (train_df["Ngày"] <= cand_end_dt)
            ].reset_index(drop=True)
            assert len(expected_with_anchor) == gap_len + 1
            anchor_obs_row = observed_sentinel.iloc[-gap_len - 1]
            for rad_col in CERES_SYN1DEG_VI_COLS:
                lo, hi = PHYSICAL_METEO_BOUNDS[rad_col]
                tau_v = clim.tau[rad_col]
                delta0 = float(anchor_obs_row[rad_col]) - float(expected_with_anchor[rad_col].iloc[0])
                dt_steps = np.arange(1, gap_len + 1, dtype=np.float64)
                expected_blended = np.clip(
                    expected_with_anchor[rad_col].iloc[1:].to_numpy() + delta0 * np.exp(-dt_steps / tau_v),
                    lo,
                    hi,
                )
                assert np.allclose(scen_df[rad_col].iloc[-gap_len:].to_numpy(), expected_blended, atol=1e-9)


def test_mode2_backtest_causality_variable_weights_and_mode3_fallback(
    serving_predictor: StandaloneRainfallPredictor,
    synthetic_dev_train_bundle,
):
    train_df, clim, store, _ = synthetic_dev_train_bundle
    cfg_weighted = ExogenousImputationConfig(
        variable_weights={"Lượng mưa": 3.0, "T2M": 2.0},
    )
    wrapper = ExogenousImputerWrapper(
        model=serving_predictor.model_serving,
        feature_builder=serving_predictor.fb_f3,
        climatology=clim,
        analog_store=store,
        config=cfg_weighted,
    )

    # Backtest inside 2016 spanning leap day Feb 29; candidates must stay strictly < t_start
    query_slice = train_df.iloc[1840:1960].copy().reset_index(drop=True)
    t_start = pd.Timestamp(query_slice["Ngày"].iloc[0]).normalize()
    query_slice.loc[query_slice.index[-20:], CERES_SYN1DEG_VI_COLS] = np.nan

    res_mode2 = wrapper.predict(query_slice)
    assert res_mode2["imputation_mode"] == MODE_2_ANALOG_MARGINALIZATION
    assert all(pd.Timestamp(d) < t_start for d in res_mode2["selected_analog_dates"])

    # Mode 3 fallback when all 34 exogenous variables are missing over matching window
    all_exo_cols = [c for c in CANONICAL_VI_COLUMN_ORDER if c not in ("Ngày", "Lượng mưa")]
    total_blackout = query_slice.copy()
    total_blackout.loc[total_blackout.index[-30:], all_exo_cols] = np.nan
    res_mode3 = wrapper.predict(total_blackout)
    assert res_mode3["imputation_mode"] == MODE_3_UNCONDITIONAL_SEASONAL_FALLBACK
    assert res_mode3["low_confidence_warning"] is True
    assert res_mode3["effective_k"] == 0
    assert res_mode3["phase_radius_used"] is None


def test_mode2_phase_expansion_and_real_dev_train_models(
    serving_predictor: StandaloneRainfallPredictor,
):
    # Phase expansion (delta=3 -> delta=15) when fewer than K=10 clean candidates exist in +-3d
    cfg_expand = ExogenousImputationConfig(
        seasonal_phase_radius_days=3,
        expanded_phase_radius_days=15,
        analog_top_k=10,
    )
    sparse_train = _make_synthetic_canonical_df("2015-01-01", n_days=500, seed=91)
    clim_sparse = ClimatologyLookupTable.fit(sparse_train, config=cfg_expand)
    store_sparse = HistoricalAnalogStore.from_dataframe(sparse_train, config=cfg_expand)
    wrapper_sparse = ExogenousImputerWrapper(
        model=serving_predictor.model_serving,
        feature_builder=serving_predictor.fb_f3,
        climatology=clim_sparse,
        analog_store=store_sparse,
        config=cfg_expand,
    )
    q_sparse = _make_synthetic_canonical_df("2025-11-10", n_days=120, seed=92)
    q_sparse.loc[q_sparse.index[-15:], CERES_SYN1DEG_VI_COLS] = np.nan
    res_sparse = wrapper_sparse.predict(q_sparse)
    assert res_sparse["imputation_mode"] == MODE_2_ANALOG_MARGINALIZATION
    assert res_sparse["phase_radius_used"] == cfg_expand.expanded_phase_radius_days
    assert res_sparse["low_confidence_warning"] is True
    assert 1 <= res_sparse["effective_k"] <= cfg_expand.analog_top_k

    # Real canonical HCMC Dev_Train (6,329 days: 2001-01-01 .. 2018-04-30) via bundled demo DB + SC-35/SC-30/SC-31
    demo_db = PROJECT_ROOT / "artifacts" / "models" / "demo_rainfall_vintage.db"
    if not demo_db.exists():
        pytest.skip("Bundled demo_rainfall_vintage.db not present.")

    conn = connect_sqlite(demo_db, read_only=True)
    try:
        raw_df = read_latest_observations(conn, location_id="HCMC", as_vietnamese=True)
    finally:
        conn.close()

    dev_train_df, post_train_df = time_series_split(raw_df, date_col="Ngày", test_start_date="2018-05-01")
    test_q1_df = post_train_df[
        (post_train_df["Ngày"] >= pd.Timestamp("2025-11-15"))
        & (post_train_df["Ngày"] <= pd.Timestamp("2026-03-14"))
    ].copy().reset_index(drop=True)

    cfg = ExogenousImputationConfig()
    real_clim = ClimatologyLookupTable.fit(dev_train_df, config=cfg)
    assert real_clim.n_train_rows == 6329
    real_store = HistoricalAnalogStore.from_dataframe(dev_train_df, config=cfg)

    for model_obj, fb_obj in (
        (serving_predictor.model_serving, serving_predictor.fb_f3),
        (serving_predictor.model_shadow_val, serving_predictor.fb_f2),
        (serving_predictor.model_shadow_contender, serving_predictor.fb_f3),
    ):
        real_wrapper = ExogenousImputerWrapper(
            model=model_obj,
            feature_builder=fb_obj,
            climatology=real_clim,
            analog_store=real_store,
            config=cfg,
        )
        exp_clean = _direct_build_single_step_predict(model_obj, fb_obj, test_q1_df)
        res_clean = real_wrapper.predict(test_q1_df)
        assert float(np.max(np.abs(np.asarray(res_clean["predictions_7d_mm"]) - exp_clean))) == 0.0

        masked = test_q1_df.copy()
        masked.loc[masked.index[-80:], CERES_SYN1DEG_VI_COLS] = np.nan
        res_mode2 = real_wrapper.predict(masked)
        assert res_mode2["imputation_mode"] == MODE_2_ANALOG_MARGINALIZATION
        assert res_mode2["effective_k"] == 10
        assert all(pd.Timestamp(d) <= pd.Timestamp("2018-04-30") for d in res_mode2["selected_analog_dates"])
        assert np.all(np.isfinite(res_mode2["weighted_median_7d_mm"]))
        assert np.all(np.asarray(res_mode2["weighted_median_7d_mm"]) >= 0.0)
        assert np.all(np.isfinite(res_mode2["imputation_uncertainty_7d_mm"]))
