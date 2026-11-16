"""Cross-Ecosystem MLflow Compatibility & Cleanliness Gate (WP2).

Runs 4 fast, isolated end-to-end integration checks across all 4 modeling
ecosystems used in DS108:
  1. Sklearn / XGBoost / LightGBM MultiOutputRegressor
  2. Two-Stage DirectHurdleForecaster (Stage 1 Classifier + Stage 2 Regressor + Pipeline)
  3. Nixtla StatsForecastWrapper & FrozenIntermittentWrapper (_state_snapshot invariance)
  4. PyTorch Lightning / Nixtla NeuralForecast NHITS + InMemoryStepLossCallback

Asserts:
  - Hyperparameter & parameter-count extraction works per ecosystem.
  - Feature importance & Hurdle threshold grids serialize cleanly to JSON diagnostics.
  - Nixtla wrapper internal frozen state snapshots remain strictly unmutated.
  - NeuralForecast captures per-step training loss strictly in RAM via
    `InMemoryStepLossCallback` and produces zero local `lightning_logs/`.
  - No local `mlruns/`, `artifacts/mlflow.db`, or `lightning_logs/` directories are created.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from statsforecast.models import Naive
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE as NFMaeLoss
from neuralforecast.models import NHITS

from src.evaluation.metrics import evaluate_point_forecast_metrics
from src.evaluation.mlflow_tracker import (
    InMemoryStepLossCallback,
    MLflowTrainingTracker,
    detect_model_adapter,
    extract_feature_importance_payload,
    extract_hurdle_threshold_grid,
    extract_model_hyperparams,
    get_neural_trainer_kwargs,
)
from src.models.hurdle import DirectHurdleForecaster
from src.models.nixtla.wrappers import (
    FrozenIntermittentWrapper,
    StatsForecastWrapper,
    _assert_snapshots_equal,
    _deep_snapshot_state,
)


class _InMemoryCaptureClient:
    """Lightweight offline MLflowClient stand-in validating batch/dict payloads."""

    def __init__(self) -> None:
        self.tracking_uri = "memory://compatibility-check"
        self.runs: Dict[str, Dict[str, Any]] = {}
        self.batch_calls: int = 0
        self.logged_dicts: Dict[str, List[str]] = {}
        self.terminated: List[str] = []
        self._counter = 0

    def search_runs(self, experiment_ids: List[str], max_results: int = 1000) -> List[Any]:
        return []

    def create_run(self, experiment_id: str, run_name: str) -> Any:
        self._counter += 1
        rid = f"mock-run-{self._counter:03d}"
        self.runs[rid] = {
            "run_name": run_name,
            "metrics": [],
            "params": {},
            "tags": {"mlflow.runName": run_name},
        }
        return SimpleNamespace(
            info=SimpleNamespace(run_id=rid, status="RUNNING"),
            data=SimpleNamespace(params={}, tags={"mlflow.runName": run_name}, metrics={}),
        )

    def log_batch(self, run_id: str, metrics=(), params=(), tags=()) -> None:
        self.batch_calls += 1
        rec = self.runs[run_id]
        for m in metrics:
            rec["metrics"].append((m.key, m.value, m.step))
        for p in params:
            assert p.key not in rec["params"], f"Duplicate param attempted: {p.key}"
            rec["params"][p.key] = p.value
        for t in tags:
            rec["tags"][t.key] = t.value

    def log_dict(self, run_id: str, dictionary: Dict[str, Any], artifact_file: str) -> None:
        # Validate JSON serializability strictly
        json.dumps(dictionary)
        self.logged_dicts.setdefault(run_id, []).append(artifact_file)

    def set_terminated(self, run_id: str, status: str = "FINISHED") -> None:
        self.terminated.append(run_id)


def _assert_workspace_clean() -> None:
    forbidden = [
        PROJECT_ROOT / "mlruns",
        PROJECT_ROOT / "artifacts" / "mlflow.db",
        PROJECT_ROOT / "lightning_logs",
    ]
    existing = [str(p) for p in forbidden if p.exists()]
    if existing:
        raise AssertionError(f"Workspace pollution detected! Forbidden paths exist: {existing}")


def check_1_sklearn_multioutput(tracker: MLflowTrainingTracker, capture: _InMemoryCaptureClient | None) -> float:
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    X_tr = rng.normal(size=(120, 10))
    Y_tr = np.maximum(0.0, rng.gamma(shape=1.5, scale=2.5, size=(120, 7)))
    X_val = rng.normal(size=(40, 10))
    Y_val = np.maximum(0.0, rng.gamma(shape=1.5, scale=2.5, size=(40, 7)))
    feat_names = [f"feat_{i}" for i in range(10)]

    lgbm = MultiOutputRegressor(
        LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            learning_rate=0.05,
            num_leaves=15,
            n_estimators=15,
            verbosity=-1,
            random_state=42,
            n_jobs=1,
        )
    )
    lgbm.fit(X_tr, Y_tr)
    preds = np.clip(lgbm.predict(X_val), 0.0, None)
    detailed = evaluate_point_forecast_metrics(Y_val, preds, horizon=7)

    assert detect_model_adapter(lgbm) == "SklearnMultiOutputAdapter"
    hp = extract_model_hyperparams(lgbm)
    assert hp["n_outputs"] == 7
    assert hp["max_depth"] == -1
    fi = extract_feature_importance_payload(lgbm, feature_names=feat_names)
    assert fi is not None and fi["n_horizons"] == 7 and len(fi["top_features"]) == 10

    xgb = MultiOutputRegressor(
        XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=1.5,
            learning_rate=0.05,
            max_depth=3,
            n_estimators=15,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
    )
    xgb.fit(X_tr, Y_tr)
    assert extract_model_hyperparams(xgb)["max_depth"] == 3
    assert extract_feature_importance_payload(xgb, feature_names=feat_names) is not None

    rid = tracker.log_training_run(
        scenario_id="SC-30",
        model_name="lightgbm_tweedie_F2",
        model=lgbm,
        detailed_metrics=detailed,
        training_time=0.1,
        inference_time=0.01,
        feature_set="F2",
        feature_names=feat_names,
        feature_count=10,
        strategy="Compat Check",
    )
    assert rid is not None
    if capture is not None:
        assert "diagnostics/feature_importance.json" in capture.logged_dicts[rid]
        metric_keys = {k for k, _, _ in capture.runs[rid]["metrics"]}
        assert {"val_MAE", "val_MSE", "val_RMSE", "val_MASE", "val_horizon_MAE"}.issubset(metric_keys)
    return time.perf_counter() - t0


def check_2_hurdle_forecaster(tracker: MLflowTrainingTracker, capture: _InMemoryCaptureClient | None) -> float:
    t0 = time.perf_counter()
    rng = np.random.default_rng(7)
    X_tr = rng.normal(size=(160, 8))
    wet = rng.uniform(size=(160, 7)) < 0.45
    Y_tr = np.where(wet, rng.gamma(2.0, 4.0, size=(160, 7)) + 1.0, rng.uniform(0.0, 0.4, size=(160, 7)))
    X_val = rng.normal(size=(50, 8))
    wet_v = rng.uniform(size=(50, 7)) < 0.45
    Y_val = np.where(wet_v, rng.gamma(2.0, 4.0, size=(50, 7)) + 1.0, rng.uniform(0.0, 0.4, size=(50, 7)))
    feat_names = [f"hfeat_{i}" for i in range(8)]

    hurdle_lgbm = DirectHurdleForecaster(
        classifier_factory=lambda: LGBMClassifier(n_estimators=10, num_leaves=15, verbosity=-1, random_state=42, n_jobs=1),
        regressor_factory=lambda: LGBMRegressor(n_estimators=10, num_leaves=15, verbosity=-1, random_state=42, n_jobs=1),
        horizon=7,
        model_alias="Compat_LGBM_Hurdle",
    )
    hurdle_lgbm.fit(X_tr, Y_tr, y_dev_daily=Y_tr[:, 0])
    summary = hurdle_lgbm.select_optimal_strategy_on_validation(X_val, Y_val)
    preds = hurdle_lgbm.predict(X_val)
    detailed = evaluate_point_forecast_metrics(Y_val, preds, horizon=7)

    assert detect_model_adapter(hurdle_lgbm) == "HurdleModelAdapter"
    hp = extract_model_hyperparams(hurdle_lgbm)
    assert "fitted_dry_mean_mm" in hp and "selected_tau_clf" in hp
    fi = extract_feature_importance_payload(hurdle_lgbm, feature_names=feat_names)
    assert fi is not None and fi["stage1_classifier"] is not None and fi["stage2_regressor"] is not None
    assert extract_hurdle_threshold_grid(hurdle_lgbm) is not None

    # Also verify Pipeline unwrapping on Logistic+Linear Hurdle
    hurdle_pipe = DirectHurdleForecaster(
        classifier_factory=lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, random_state=42)),
        regressor_factory=lambda: make_pipeline(StandardScaler(), LinearRegression()),
        horizon=7,
        model_alias="Compat_LogLin_Hurdle",
    )
    hurdle_pipe.fit(X_tr, Y_tr, y_dev_daily=Y_tr[:, 0])
    fi_pipe = extract_feature_importance_payload(hurdle_pipe, feature_names=feat_names)
    assert fi_pipe is not None and fi_pipe["stage1_classifier"]["importance_type"] == "abs_coef_"

    rid = tracker.log_training_run(
        scenario_id="SC-55",
        model_name="lightgbm_stratB_F2",
        model=hurdle_lgbm,
        detailed_metrics=detailed,
        training_time=0.2,
        inference_time=0.02,
        feature_set="F2",
        feature_names=feat_names,
        feature_count=8,
        strategy="Compat Hurdle",
        selected_hurdle_threshold=hurdle_lgbm.selected_tau_clf_,
        threshold_grid=summary,
    )
    assert rid is not None
    if capture is not None:
        dicts = capture.logged_dicts[rid]
        assert "diagnostics/feature_importance.json" in dicts
        assert "diagnostics/hurdle_threshold_grid.json" in dicts
    return time.perf_counter() - t0


def check_3_nixtla_wrappers(tracker: MLflowTrainingTracker, capture: _InMemoryCaptureClient | None) -> float:
    t0 = time.perf_counter()
    rng = np.random.default_rng(123)
    y_train = np.where(rng.uniform(size=120) < 0.45, rng.gamma(2.0, 5.0, size=120) + 1.0, 0.0)
    y_eval = np.concatenate([y_train, np.where(rng.uniform(size=30) < 0.45, rng.gamma(2.0, 5.0, size=30) + 1.0, 0.0)])

    sf = StatsForecastWrapper(Naive(), alias="Naive")
    sf.fit(y_train)
    sf_snap_before = _deep_snapshot_state(sf.model_)
    assert detect_model_adapter(sf) == "NixtlaWrapperAdapter"
    sf_hp = extract_model_hyperparams(sf)
    assert sf_hp["base_model_type"] == "Naive" and sf_hp["n_train"] == 120
    assert extract_feature_importance_payload(sf) is None
    _assert_snapshots_equal(sf_snap_before, _deep_snapshot_state(sf.model_), context="CompatStatsForecast")
    sf_preds = sf.predict_rolling(y_eval, origin_indices=[120, 121, 122], h=7)
    assert sf_preds.shape == (3, 7)

    adida = FrozenIntermittentWrapper("ADIDA", aggregation_chunk_size=2, alias="ADIDA")
    adida.fit(y_train)
    iw_snap_before = _deep_snapshot_state(adida)
    assert detect_model_adapter(adida) == "NixtlaWrapperAdapter"
    iw_hp = extract_model_hyperparams(adida)
    assert iw_hp["domain_bucket_size"] == 2
    assert "alpha_d" in iw_hp and "wet_day_spacing_k0" in iw_hp
    assert extract_feature_importance_payload(adida) is None
    _assert_snapshots_equal(iw_snap_before, _deep_snapshot_state(adida), context="CompatADIDA")
    iw_preds = adida.predict_rolling(y_eval, origin_indices=[120, 121, 122], h=7)
    assert iw_preds.shape == (3, 7)

    y_true = np.array([y_eval[idx + 1 : idx + 8] for idx in (120, 121, 122)], dtype=np.float64)
    detailed = evaluate_point_forecast_metrics(y_true, iw_preds, horizon=7)
    rid = tracker.log_training_run(
        scenario_id="SC-10",
        model_name="ADIDA",
        model=adida,
        detailed_metrics=detailed,
        training_time=0.05,
        inference_time=0.01,
        feature_set="Target-Only",
        strategy="Intermittent Model",
        parameter_count=2,
    )
    assert rid is not None
    if capture is not None:
        assert capture.runs[rid]["params"].get("aggregation_chunk_size") == "2"
    return time.perf_counter() - t0


def check_4_neural_lightning(tracker: MLflowTrainingTracker, capture: _InMemoryCaptureClient | None) -> float:
    t0 = time.perf_counter()
    rng = np.random.default_rng(99)
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    y = np.where(rng.uniform(size=90) < 0.45, rng.gamma(1.8, 3.5, size=90), 0.0).astype(np.float32)
    df_mini = pd.DataFrame({"unique_id": "HCMC_MINI", "ds": dates, "y": y})

    loss_cb = InMemoryStepLossCallback(log_every_n_steps=1)
    trainer_kw = get_neural_trainer_kwargs(callbacks=[loss_cb])
    trainer_kw["accelerator"] = "cpu"
    trainer_kw["devices"] = 1

    model = NHITS(
        h=7,
        input_size=14,
        n_blocks=[1, 1, 1],
        mlp_units=[[16, 16], [16, 16], [16, 16]],
        n_pool_kernel_size=[2, 2, 1],
        loss=NFMaeLoss(),
        max_steps=8,
        batch_size=16,
        learning_rate=1e-3,
        random_seed=42,
        **trainer_kw,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=df_mini)

    fcst_df = nf.predict(df=df_mini.iloc[-14:].copy())
    preds = np.clip(fcst_df["NHITS"].values.reshape(1, 7), 0.0, None)
    assert preds.shape == (1, 7) and np.all(np.isfinite(preds))

    steps = loss_cb.get_step_metrics()
    assert len(steps) > 0, "InMemoryStepLossCallback captured 0 step losses!"
    assert detect_model_adapter(model) == "NeuralForecastAdapter"
    hp = extract_model_hyperparams(model)
    assert hp["parameter_count"] > 0
    assert hp["trainer_logger"] is False
    assert extract_feature_importance_payload(model) is None

    y_true = y[-7:].astype(np.float64).reshape(1, 7)
    detailed = evaluate_point_forecast_metrics(y_true, preds, horizon=7)
    rid = tracker.log_training_run(
        scenario_id="SC-36",
        model_name="NHITS",
        model=model,
        detailed_metrics=detailed,
        training_time=0.5,
        inference_time=0.05,
        feature_set="Target-Only",
        strategy="Neural Hierarchical Interpolation",
        loss_callback=loss_cb,
    )
    assert rid is not None
    if capture is not None:
        step_losses = [m for m in capture.runs[rid]["metrics"] if m[0] == "val_train_loss"]
        assert len(step_losses) == len(steps)
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify DS108 MLflow Ecosystem Compatibility (4 Frameworks)")
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Push telemetry to live DagsHub instead of in-memory mock client.",
    )
    args = parser.parse_args()

    t_all = time.perf_counter()
    _assert_workspace_clean()

    if args.cloud:
        capture = None
        tracker = MLflowTrainingTracker(
            experiment_name="Rainfall_Forecasting_HCMC_Compat_Test",
            partition="validation",
            source="compatibility_check",
            enabled=True,
            strict=True,
        )
    else:
        capture = _InMemoryCaptureClient()
        tracker = MLflowTrainingTracker(
            experiment_name="compat-test",
            partition="validation",
            source="compatibility_check",
            client=capture,
            experiment_id="compat-exp-1",
            auto_init=False,
            strict=True,
        )

    results = []
    for name, fn in [
        ("1. Sklearn/XGB/LGBM MultiOutput", check_1_sklearn_multioutput),
        ("2. Two-Stage DirectHurdleForecaster", check_2_hurdle_forecaster),
        ("3. Nixtla StatsForecast & Intermittent", check_3_nixtla_wrappers),
        ("4. PyTorch Lightning / NeuralForecast", check_4_neural_lightning),
    ]:
        dt = fn(tracker, capture)
        results.append((name, dt))
        print(f"[PASS] {name:<40} ({dt:.2f}s)", flush=True)

    _assert_workspace_clean()
    total_dt = time.perf_counter() - t_all
    print(f"\n[SUCCESS] 4/4 Ecosystem Compatibility Checks PASSED in {total_dt:.2f}s (workspace clean).")


if __name__ == "__main__":
    main()

