"""
Verification script for DagsHub Cloud MLflow Tracking & Model Registry
======================================================================
Verifies:
1. Absence of legacy local `artifacts/mlflow.db` and `mlruns/`.
2. 76 runs logged in DagsHub Experiment `Rainfall_Forecasting_HCMC_Benchmark`,
   each with 4 canonical metrics (MAE, MSE, RMSE, MASE) and 7 horizon steps (h=1..7).
3. All 3 governance models (`serving`, `shadow_val_champ`, `shadow_test_contender`)
   loaded directly from DagsHub Model Registry into temporary memory and matching
   local `StandaloneRainfallPredictor` within < 1e-5 mm absolute error.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.mlops import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGISTERED_MODEL_NAME,
    ModelRegistryGovernance,
    init_dagshub_mlflow,
    retry_with_backoff,
)
from src.data.loader import DataLoader
from src.inference.predictor import StandaloneRainfallPredictor


def compute_local_unrounded_predictions(
    predictor: StandaloneRainfallPredictor,
    df_sample_history: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    feat_df3 = predictor.fb_f3.build_single_step(df_sample_history)
    cols3 = [c for c in feat_df3.columns if c not in [predictor.fb_f3.date_col, predictor.fb_f3.target_col]]
    X3 = feat_df3[cols3].values.astype(np.float64)

    feat_df2 = predictor.fb_f2.build_single_step(df_sample_history)
    cols2 = [c for c in feat_df2.columns if c not in [predictor.fb_f2.date_col, predictor.fb_f2.target_col]]
    X2 = feat_df2[cols2].values.astype(np.float64)

    return {
        "serving": np.clip(predictor.model_serving.predict(X3), 0.0, None),
        "shadow_val_champ": np.clip(predictor.model_shadow_val.predict(X2), 0.0, None),
        "shadow_test_contender": np.clip(predictor.model_shadow_contender.predict(X3), 0.0, None),
    }


def main() -> None:
    # 0. Verify local MLflow artifacts are purged
    local_db = PROJECT_ROOT / "artifacts" / "mlflow.db"
    local_mlruns = PROJECT_ROOT / "mlruns"
    assert not local_db.exists(), f"Legacy local DB still exists: {local_db}"
    assert not local_mlruns.exists(), f"Legacy local mlruns directory still exists: {local_mlruns}"
    print("[PASS] Verified zero legacy local MLflow files (mlflow.db / mlruns).")

    client = init_dagshub_mlflow()
    print(f"Connected to DagsHub MLflow URI: {client.tracking_uri}")

    print("\n" + "=" * 75)
    print("1. EXPERIMENT & 76-RUN CANONICAL METRICS AUDIT")
    print("=" * 75)
    exp = retry_with_backoff(
        lambda: client.get_experiment_by_name(DEFAULT_EXPERIMENT_NAME),
        operation_name="get_experiment_by_name",
    )
    assert exp is not None, f"Experiment '{DEFAULT_EXPERIMENT_NAME}' not found on DagsHub!"

    runs = retry_with_backoff(
        lambda: client.search_runs(experiment_ids=[exp.experiment_id], max_results=1000),
        operation_name="search_runs",
    )
    print(f"Experiment ID: {exp.experiment_id}")
    print(f"Total runs logged: {len(runs)}")
    assert len(runs) == 76, f"Expected 76 runs on DagsHub, found {len(runs)}"

    required_val_metrics = {"val_MAE", "val_MSE", "val_RMSE", "val_MASE"}
    required_test_metrics = {"test_MAE", "test_MSE", "test_RMSE", "test_MASE"}

    df_rows = []
    for r in runs:
        metric_keys = set(r.data.metrics.keys())
        missing_val = required_val_metrics - metric_keys
        missing_test = required_test_metrics - metric_keys
        assert not missing_val, f"Run {r.info.run_id} missing val metrics: {missing_val}"
        assert not missing_test, f"Run {r.info.run_id} missing test metrics: {missing_test}"

        df_rows.append(
            {
                "run_id": r.info.run_id,
                "run_name": r.data.tags.get("mlflow.runName", ""),
                "scenario_id": r.data.tags.get("scenario_id", ""),
                "config_version": r.data.tags.get("config_version", "standard"),
                "governance_role": r.data.tags.get("governance_role", ""),
                "val_MAE": r.data.metrics.get("val_MAE"),
                "test_MAE": r.data.metrics.get("test_MAE"),
            }
        )

    df_summary = pd.DataFrame(df_rows)
    target_scenarios = ["SC-35", "SC-30", "SC-31"]
    df_targets = df_summary[df_summary["scenario_id"].isin(target_scenarios)]
    print("\nAudited Target Models in DagsHub Runs:")
    print(df_targets.to_string(index=False))

    # Audit 7 horizon steps (step=1..7) on key representative runs
    print("\nAuditing Horizon Breakdown (step=1..7) on representative runs...")
    for target_run_name in [
        "SC-35_xgboost_tweedie_F3",
        "SC-30_lightgbm_tweedie_F2_ConfigB",
        "SC-01_Naive",
    ]:
        match = df_summary[df_summary["run_name"] == target_run_name]
        assert not match.empty, f"Run '{target_run_name}' not found!"
        rid = match.iloc[0]["run_id"]
        val_hist = retry_with_backoff(
            lambda r=rid: client.get_metric_history(r, "val_horizon_MAE"),
            operation_name=f"get_metric_history val_horizon_MAE ({target_run_name})",
        )
        test_hist = retry_with_backoff(
            lambda r=rid: client.get_metric_history(r, "test_horizon_MAE"),
            operation_name=f"get_metric_history test_horizon_MAE ({target_run_name})",
        )
        val_steps = sorted({m.step for m in val_hist})
        test_steps = sorted({m.step for m in test_hist})
        assert val_steps == [1, 2, 3, 4, 5, 6, 7], f"{target_run_name} val steps={val_steps}"
        assert test_steps == [1, 2, 3, 4, 5, 6, 7], f"{target_run_name} test steps={test_steps}"
        print(f"  [PASS] {target_run_name}: val_horizon steps={val_steps}, test_horizon steps={test_steps}")

    print("\n" + "=" * 75)
    print("2. DAGSHUB MODEL REGISTRY AUDIT & ALIASES")
    print("=" * 75)
    reg_model = retry_with_backoff(
        lambda: client.get_registered_model(DEFAULT_REGISTERED_MODEL_NAME),
        operation_name="get_registered_model",
    )
    print(f"Registered Model Name: {reg_model.name}")
    print(f"Aliases defined: {getattr(reg_model, 'aliases', {})}")

    versions = retry_with_backoff(
        lambda: client.search_model_versions(f"name = '{DEFAULT_REGISTERED_MODEL_NAME}'"),
        operation_name="search_model_versions",
    )
    for mv in sorted(versions, key=lambda v: int(v.version)):
        print(f"  - Version {mv.version} (stage={mv.current_stage}, run_id={mv.run_id}):")
        print(f"      Tags: {mv.tags}")

    print("\n" + "=" * 75)
    print("3. END-TO-END CLOUD INFERENCE & LOCAL PARITY VERIFICATION (< 1e-5 mm)")
    print("=" * 75)
    df_weather = DataLoader().load_data()
    df_sample_history = df_weather.iloc[-35:].copy().reset_index(drop=True)

    local_predictor = StandaloneRainfallPredictor(artifact_dir=PROJECT_ROOT / "artifacts" / "models")
    local_unrounded = compute_local_unrounded_predictions(local_predictor, df_sample_history)
    sample_run_date = df_sample_history["Ngày"].iloc[-1].strftime("%Y-%m-%d")
    local_api_res = local_predictor.predict(df_sample_history, run_date=sample_run_date)

    for alias, api_extractor in [
        ("serving", lambda r: r["serving_model"]["predictions_7d_mm"]),
        ("shadow_val_champ", lambda r: r["shadow_models"]["shadow_val_champ"]["predictions_7d_mm"]),
        ("shadow_test_contender", lambda r: r["shadow_models"]["shadow_test_contender"]["predictions_7d_mm"]),
    ]:
        model_uri = ModelRegistryGovernance.resolve_model_uri(
            client, DEFAULT_REGISTERED_MODEL_NAME, alias
        )
        print(f"\n--- Loading '{alias}' from DagsHub URI: {model_uri} ---")
        loaded_model = retry_with_backoff(
            lambda uri=model_uri: mlflow.pyfunc.load_model(uri),
            operation_name=f"load_model({model_uri})",
        )
        cloud_pred = np.asarray(loaded_model.predict(df_sample_history), dtype=np.float64)
        local_pred = local_unrounded[alias]
        max_diff = float(np.max(np.abs(cloud_pred - local_pred)))

        print(f"  Cloud 7-day forecast : {np.round(cloud_pred[0], 4)}")
        print(f"  Local 7-day forecast : {np.round(local_pred[0], 4)}")
        print(f"  Max abs diff (mm)    : {max_diff:.2e}")
        assert cloud_pred.shape == (1, 7), f"Unexpected shape: {cloud_pred.shape}"
        assert max_diff < 1e-5, f"Parity violation for {alias}: max_diff={max_diff}"
        assert [round(float(x), 2) for x in cloud_pred[0]] == api_extractor(local_api_res)

    # Verify zero local mlruns/mlflow.db recreated after loading
    assert not local_db.exists(), "Local mlflow.db was unexpectedly created!"
    assert not local_mlruns.exists(), "Local mlruns/ was unexpectedly created!"

    print("\n" + "=" * 75)
    print("ALL DAGSHUB CLOUD VERIFICATIONS AND PARITY TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
