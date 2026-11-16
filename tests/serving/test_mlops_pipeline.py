"""
Automated Verification Suite for DagsHub Cloud MLflow MLOps Pipeline
===================================================================
Covers:
1. Configuration, .env validation & Dual-Resolution Governance fallback.
2. Custom Pyfunc wrapper signature, ModelSignature & pip requirements.
3. Workspace cleanliness (absence of local `artifacts/mlflow.db` and `mlruns/`).
4. Live DagsHub Cloud Tracking (76 runs, 4 canonical metrics, 7 horizon steps).
5. Live DagsHub Model Registry E2E inference parity (< 1e-5 mm vs local predictor).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

mlflow = pytest.importorskip("mlflow")
from mlflow.exceptions import MlflowException  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.abspath("."))

from src.config.mlops import (  # noqa: E402
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGISTERED_MODEL_NAME,
    ModelRegistryGovernance,
    init_dagshub_mlflow,
    load_dotenv_file,
    retry_with_backoff,
    validate_mlflow_env,
)
from src.data.loader import DataLoader
from src.evaluation.protocols import get_global_read_guard
from src.inference.predictor import StandaloneRainfallPredictor
from src.inference.pyfunc import (
    RainfallForecastingPyfuncWrapper,
    build_pyfunc_signature_and_example,
    get_serving_pip_requirements,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def unguard_for_mlops_tests():
    g = get_global_read_guard()
    was_active = g.is_active
    g.deactivate()
    yield
    if was_active:
        g.activate()


def _has_dagshub_credentials() -> bool:
    load_dotenv_file()
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    user = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    pwd = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
    return bool(uri and user and pwd and "<YOUR_DAGSHUB_ACCESS_TOKEN>" not in pwd)


def test_env_validation_missing_raises(tmp_path, monkeypatch):
    for var in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    empty_env = tmp_path / ".env"
    empty_env.write_text('MLFLOW_TRACKING_PASSWORD="<YOUR_DAGSHUB_ACCESS_TOKEN>"\n', encoding="utf-8")
    load_dotenv_file(empty_env, override=True)
    with pytest.raises(EnvironmentError, match="Missing required MLflow/DagsHub"):
        validate_mlflow_env()


def test_governance_dual_resolution_alias_and_tag_fallback():
    mock_client = MagicMock()
    mock_mv = SimpleNamespace(version="2", tags={"governance_alias": "shadow_val_champ"})

    # Case 1: Alias endpoint supported
    mock_client.get_model_version_by_alias.return_value = mock_mv
    uri_alias = ModelRegistryGovernance.resolve_model_uri(
        mock_client, "RainfallForecastHCMC", "shadow_val_champ"
    )
    assert uri_alias == "models:/RainfallForecastHCMC@shadow_val_champ"

    # Case 2: Alias endpoint unsupported -> fallback to governance_alias tag
    mock_client.get_model_version_by_alias.side_effect = MlflowException("Unsupported endpoint")
    mock_client.search_model_versions.return_value = [
        SimpleNamespace(version="1", tags={"governance_alias": "serving"}, current_stage="Production"),
        SimpleNamespace(version="2", tags={"governance_alias": "shadow_val_champ"}, current_stage="Staging"),
        SimpleNamespace(version="3", tags={"governance_alias": "shadow_val_champ"}, current_stage="Staging"),
    ]
    uri_tag = ModelRegistryGovernance.resolve_model_uri(
        mock_client, "RainfallForecastHCMC", "shadow_val_champ"
    )
    assert uri_tag == "models:/RainfallForecastHCMC/3"


def test_pyfunc_wrapper_signature_and_requirements():
    df_sample = DataLoader().load_data().iloc[-35:].copy().reset_index(drop=True)
    wrapper = RainfallForecastingPyfuncWrapper(
        model_path=str(PROJECT_ROOT / "artifacts" / "models" / "sc35_xgboost_tweedie_f3.joblib"),
        feature_builder_path=str(PROJECT_ROOT / "artifacts" / "models" / "feature_builder_f3.joblib"),
        feature_level=3,
        governance_role="serving",
    )
    signature, input_example = build_pyfunc_signature_and_example(wrapper, df_sample)
    assert len(input_example) == 35
    assert signature.outputs is not None

    # Verify JSON-stringified date column works (as in MLflow serving payloads)
    df_str_dates = input_example.copy()
    df_str_dates["Ngày"] = df_str_dates["Ngày"].dt.strftime("%Y-%m-%d")
    preds_str = wrapper.predict(None, df_str_dates, params={"mode": "test"})
    preds_dt = wrapper.predict(None, input_example)
    assert preds_str.shape == (1, 7)
    assert np.allclose(preds_str, preds_dt, atol=1e-12)

    reqs = get_serving_pip_requirements()
    pkg_names = {r.split("==")[0] for r in reqs}
    assert {"numpy", "pandas", "scikit-learn", "lightgbm", "joblib"}.issubset(pkg_names)
    assert bool({"xgboost", "xgboost-cpu"} & pkg_names)


def test_no_legacy_local_mlflow_files():
    assert not (PROJECT_ROOT / "artifacts" / "mlflow.db").exists()
    assert not (PROJECT_ROOT / "mlruns").exists()


@pytest.mark.skipif(not _has_dagshub_credentials(), reason="DagsHub MLflow credentials not configured in .env")
def test_dagshub_cloud_tracking_and_registry_parity():
    client = init_dagshub_mlflow()
    exp = retry_with_backoff(
        lambda: client.get_experiment_by_name(DEFAULT_EXPERIMENT_NAME),
        operation_name="get_experiment_by_name",
    )
    assert exp is not None

    runs = retry_with_backoff(
        lambda: client.search_runs(experiment_ids=[exp.experiment_id], max_results=1000),
        operation_name="search_runs",
    )
    assert len(runs) == 76

    required_metrics = {
        "val_MAE", "val_MSE", "val_RMSE", "val_MASE",
        "test_MAE", "test_MSE", "test_RMSE", "test_MASE",
    }
    naive_runs = []
    hurdle_thresh_runs = []
    target_sc35_run_id = None

    for r in runs:
        assert required_metrics.issubset(set(r.data.metrics.keys()))
        sc_id = r.data.tags.get("scenario_id")
        if sc_id in ("SC-01", "SC-02"):
            assert r.data.tags.get("is_naive_baseline") == "true"
            naive_runs.append(sc_id)
        if "selected_hurdle_threshold" in r.data.metrics:
            hurdle_thresh_runs.append(sc_id)
        if sc_id == "SC-35":
            target_sc35_run_id = r.info.run_id

    assert set(naive_runs) == {"SC-01", "SC-02"}
    assert len(hurdle_thresh_runs) > 0
    assert target_sc35_run_id is not None

    # Verify 7 horizon steps (1..7) on SC-35
    val_h_hist = retry_with_backoff(
        lambda: client.get_metric_history(target_sc35_run_id, "val_horizon_MAE"),
        operation_name="get_metric_history_val",
    )
    test_h_hist = retry_with_backoff(
        lambda: client.get_metric_history(target_sc35_run_id, "test_horizon_MAE"),
        operation_name="get_metric_history_test",
    )
    assert sorted({m.step for m in val_h_hist}) == [1, 2, 3, 4, 5, 6, 7]
    assert sorted({m.step for m in test_h_hist}) == [1, 2, 3, 4, 5, 6, 7]

    # Verify 3 Cloud Registry models match local StandaloneRainfallPredictor (< 1e-5 mm)
    df_sample_history = DataLoader().load_data().iloc[-35:].copy().reset_index(drop=True)
    local_predictor = StandaloneRainfallPredictor(artifact_dir=PROJECT_ROOT / "artifacts" / "models")

    feat_df3 = local_predictor.fb_f3.build_single_step(df_sample_history)
    cols3 = [c for c in feat_df3.columns if c not in [local_predictor.fb_f3.date_col, local_predictor.fb_f3.target_col]]
    X3 = feat_df3[cols3].values.astype(np.float64)

    feat_df2 = local_predictor.fb_f2.build_single_step(df_sample_history)
    cols2 = [c for c in feat_df2.columns if c not in [local_predictor.fb_f2.date_col, local_predictor.fb_f2.target_col]]
    X2 = feat_df2[cols2].values.astype(np.float64)

    expected_map = {
        "serving": np.clip(local_predictor.model_serving.predict(X3), 0.0, None),
        "shadow_val_champ": np.clip(local_predictor.model_shadow_val.predict(X2), 0.0, None),
        "shadow_test_contender": np.clip(local_predictor.model_shadow_contender.predict(X3), 0.0, None),
    }

    for alias, local_pred in expected_map.items():
        uri = ModelRegistryGovernance.resolve_model_uri(
            client, DEFAULT_REGISTERED_MODEL_NAME, alias
        )
        loaded_model = retry_with_backoff(
            lambda u=uri: mlflow.pyfunc.load_model(u),
            operation_name=f"load_model({uri})",
        )
        cloud_pred = np.asarray(loaded_model.predict(df_sample_history), dtype=np.float64)
        assert cloud_pred.shape == (1, 7)
        assert float(np.max(np.abs(cloud_pred - local_pred))) < 1e-5

    # Ensure no local MLflow artifacts materialized during cloud operations
    assert not (PROJECT_ROOT / "artifacts" / "mlflow.db").exists()
    assert not (PROJECT_ROOT / "mlruns").exists()



# ==============================================================================
# Ecosystem Compatibility & Deep MLflow Training Tracker Tests (WP6)
# ==============================================================================
from src.evaluation.mlflow_tracker import (
    HurdleModelAdapter,
    InMemoryStepLossCallback,
    MLflowTrainingTracker,
    NeuralForecastAdapter,
    NixtlaWrapperAdapter,
    SklearnMultiOutputAdapter,
    extract_feature_importance_payload,
    extract_model_hyperparams,
    get_neural_trainer_kwargs,
)
from src.models.hurdle import build_logistic_linear_hurdle
from src.models.nixtla.wrappers import (
    FrozenIntermittentWrapper,
    StatsForecastWrapper,
    _assert_snapshots_equal,
    _deep_snapshot_state,
)


def _make_mock_mlflow_client(run_id: str = "run-mock-001", existing_params: dict | None = None):
    client = MagicMock()
    exp = MagicMock()
    exp.experiment_id = "exp-mock-1"
    client.get_experiment_by_name.return_value = exp

    if existing_params is not None:
        existing_run = MagicMock()
        existing_run.info.run_id = run_id
        existing_run.data.tags = {"mlflow.runName": "SC-35_xgboost_tweedie_F3"}
        existing_run.data.params = dict(existing_params)
        client.search_runs.return_value = [existing_run]
    else:
        client.search_runs.return_value = []

    created_run = MagicMock()
    created_run.info.run_id = run_id
    client.create_run.return_value = created_run
    return client


def _sample_detailed_metrics() -> dict:
    metrics = {
        "MAE": 3.7333,
        "MSE": 124.5,
        "RMSE": 11.15,
        "MASE": 1.008,
        "mae_macro": 3.7333,
        "mse_macro": 124.5,
        "rmse_macro": 11.15,
        "mase_lag1_macro": 1.008,
    }
    for h in range(1, 8):
        metrics[f"mae_h{h}"] = 3.5 + 0.05 * h
        metrics[f"mse_h{h}"] = 110.0 + 3.0 * h
        metrics[f"rmse_h{h}"] = 10.5 + 0.1 * h
        metrics[f"mase_lag1_h{h}"] = 0.95 + 0.01 * h
    return metrics


def test_ecosystem_adapters_sklearn_and_hurdle():
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor

    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 6))
    y_daily = np.clip(rng.gamma(1.4, 3.0, size=100), 0.0, None)
    y_daily[rng.random(100) < 0.45] = 0.0
    Y = rng.gamma(1.4, 3.0, size=(90, 7))
    Y[rng.random((90, 7)) < 0.45] = 0.0
    feat_names = [f"f_{i}" for i in range(6)]

    xgb_mo = MultiOutputRegressor(
        XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=1.5,
            n_estimators=10,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
    ).fit(X, Y)
    assert SklearnMultiOutputAdapter.matches(xgb_mo)
    xgb_hp = extract_model_hyperparams(xgb_mo)
    assert xgb_hp["adapter"] == SklearnMultiOutputAdapter.name
    assert int(xgb_hp["max_depth"]) == 5
    xgb_fi = extract_feature_importance_payload(xgb_mo, feature_names=feat_names)
    assert xgb_fi is not None and len(xgb_fi["top_features"]) == 6

    lgbm_mo = MultiOutputRegressor(
        LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            n_estimators=10,
            max_depth=-1,
            num_leaves=15,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )
    ).fit(X, Y)
    lgbm_hp = extract_model_hyperparams(lgbm_mo)
    assert int(lgbm_hp["max_depth"]) == -1
    lgbm_fi = extract_feature_importance_payload(lgbm_mo, feature_names=feat_names)
    assert lgbm_fi is not None and len(lgbm_fi["top_features"]) == 6

    hurdle = build_logistic_linear_hurdle(horizon=2, random_state=42)
    hurdle.fit(X, Y[:, :2], y_dev_daily=y_daily)
    assert HurdleModelAdapter.matches(hurdle)
    h_hp = extract_model_hyperparams(hurdle)
    assert h_hp["adapter"] == HurdleModelAdapter.name
    assert int(h_hp["horizon"]) == 2
    h_fi = extract_feature_importance_payload(hurdle, feature_names=feat_names)
    assert h_fi is not None and len(h_fi["top_features"]) == 6


def test_ecosystem_adapter_nixtla_preserves_state_snapshot():
    pytest.importorskip("statsforecast")
    from statsforecast.models import Naive

    rng = np.random.default_rng(7)
    y_train = np.clip(rng.gamma(1.2, 4.0, size=120), 0.0, None)
    y_train[rng.random(120) < 0.5] = 0.0

    sf_wrap = StatsForecastWrapper(Naive(), alias="Naive").fit(y_train)
    croston_wrap = FrozenIntermittentWrapper("CrostonOptimized", alias="CrostonOptimized").fit(y_train)

    for wrapper in (sf_wrap, croston_wrap):
        assert NixtlaWrapperAdapter.matches(wrapper)
        target_obj = getattr(wrapper, "model_", wrapper) if hasattr(wrapper, "model_") else wrapper
        snap_before = _deep_snapshot_state(target_obj)
        hp = extract_model_hyperparams(wrapper)
        fi = extract_feature_importance_payload(wrapper)
        snap_after = _deep_snapshot_state(target_obj)
        assert hp["adapter"] == NixtlaWrapperAdapter.name
        assert fi is None
        _assert_snapshots_equal(snap_before, snap_after, context=type(wrapper).__name__)


def test_ecosystem_adapter_neuralforecast_step_loss_and_no_lightning_logs():
    pytest.importorskip("neuralforecast")
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE as NFMaeLoss
    from neuralforecast.models import NHITS

    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df_nf = pd.DataFrame(
        {
            "unique_id": "HCMC_Rainfall",
            "ds": dates,
            "y": np.linspace(0.0, 5.0, 100, dtype=np.float32),
        }
    )
    cb = InMemoryStepLossCallback(log_every_n_steps=1)
    model = NHITS(
        h=7,
        input_size=14,
        n_blocks=[1, 1, 1],
        mlp_units=[[16, 16], [16, 16], [16, 16]],
        loss=NFMaeLoss(),
        max_steps=6,
        batch_size=16,
        random_seed=42,
        **get_neural_trainer_kwargs(callbacks=[cb]),
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=df_nf)
    _ = nf.predict(df=df_nf.iloc[-14:])

    assert NeuralForecastAdapter.matches(nf)
    hp = extract_model_hyperparams(nf)
    assert hp["adapter"] == NeuralForecastAdapter.name
    assert int(hp["max_steps"]) == 6
    assert int(hp["parameter_count"]) > 0

    losses = cb.get_step_metrics()
    if not losses:
        losses = NeuralForecastAdapter.extract_loss_history(nf)
    assert len(losses) >= 1

    project_root = Path(__file__).resolve().parents[2]
    assert not (project_root / "lightning_logs").exists()
    assert not (project_root / "mlruns").exists()
    assert not (project_root / "artifacts" / "mlflow.db").exists()


def test_tracker_batched_chunking_purged_guardrail_and_param_idempotency():
    mock_client = _make_mock_mlflow_client(
        run_id="run-chunk-1",
        existing_params={"model_name": "xgboost_tweedie_F3", "feature_set": "F3"},
    )
    tracker = MLflowTrainingTracker(
        partition="test",
        source="pytest_unit",
        client=mock_client,
        strict=True,
    )

    loss_history = [(step, 1.0 / (step + 1)) for step in range(1, 451)]
    run_id = tracker.log_training_run(
        scenario_id="SC-35",
        model_name="xgboost_tweedie_F3",
        detailed_metrics=_sample_detailed_metrics(),
        training_time=1.25,
        inference_time=0.05,
        feature_set="F3",
        feature_count=70,
        parameter_count=22400,
        loss_history=loss_history,
        extra_params={"model_name": "xgboost_tweedie_F3", "new_param": "added_once"},
        extra_metrics={
            "CR-MAE@H7": 99.0,
            "cr_bias": 1.5,
            "RainDay-MAE": 12.0,
            "valid_parity_diff": 0.0,
        },
        extra_artifacts={"diagnostics/test_note.json": {"ok": True}},
    )
    assert run_id == "run-chunk-1"
    assert mock_client.log_batch.call_count >= 2

    all_logged_metric_keys = []
    all_logged_param_keys = []
    for call in mock_client.log_batch.call_args_list:
        kwargs = call.kwargs
        m_batch = kwargs.get("metrics", [])
        p_batch = kwargs.get("params", [])
        t_batch = kwargs.get("tags", [])
        assert len(m_batch) <= 400
        assert len(p_batch) <= 50
        assert len(t_batch) <= 50
        all_logged_metric_keys.extend(m.key for m in m_batch)
        all_logged_param_keys.extend(p.key for p in p_batch)

    assert "valid_parity_diff" in all_logged_metric_keys
    for banned in ("CR-MAE@H7", "cr_bias", "RainDay-MAE"):
        assert banned not in all_logged_metric_keys

    # Existing immutable params must NOT be re-logged; only new ones logged once
    assert "model_name" not in all_logged_param_keys
    assert "feature_set" not in all_logged_param_keys
    assert "new_param" in all_logged_param_keys

    # Calling a second time on same tracker should not re-log 'new_param' either
    mock_client.log_batch.reset_mock()
    tracker.log_training_run(
        scenario_id="SC-35",
        model_name="xgboost_tweedie_F3",
        detailed_metrics={"MAE": 3.7333},
        extra_params={"new_param": "changed_value"},
    )
    second_param_keys = [
        p.key
        for call in mock_client.log_batch.call_args_list
        for p in call.kwargs.get("params", [])
    ]
    assert "new_param" not in second_param_keys


def test_tracker_graceful_offline_fallback():
    broken_client = MagicMock()
    broken_client.get_experiment_by_name.side_effect = ConnectionError("simulated network outage")

    tracker = MLflowTrainingTracker(
        partition="validation",
        source="pytest_offline",
        client=broken_client,
        strict=False,
    )
    assert tracker.enabled is False
    assert tracker.degraded_reason is not None and "ConnectionError" in tracker.degraded_reason
    assert tracker.log_training_run(
        scenario_id="SC-01",
        model_name="Naive",
        detailed_metrics={"MAE": 4.2},
    ) is None

    with pytest.raises(ConnectionError):
        MLflowTrainingTracker(
            partition="validation",
            source="pytest_strict",
            client=broken_client,
            strict=True,
        )
