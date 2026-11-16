"""
Canonical Model Packaging, Parity Verification & MLflow Telemetry (DS108)
=======================================================================
Trains, parity-verifies, and serializes the three governance-frozen models:
- SC-35 (`serving`): XGBoost Tweedie on F3 (max_depth=5)
- SC-30 Config A (`shadow_val_champ`): LightGBM Tweedie on F2 (max_depth=-1)
- SC-31 Config A (`shadow_test_contender`): LightGBM Tweedie on F3 (max_depth=-1)

Computes/updates SHA-256 artifact checksums in `artifacts/models/metadata.json`
and logs full packaging telemetry + feature importance diagnostics + parity checks
to hosted DagsHub MLflow via `MLflowTrainingTracker`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.analysis.loaders import compute_sha256
from src.config.mlops import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGISTERED_MODEL_NAME,
    ModelRegistryGovernance,
)
from src.evaluation.metrics import evaluate_point_forecast_metrics
from src.evaluation.mlflow_tracker import MLflowTrainingTracker
from src.evaluation.protocols import (
    CANONICAL_DEV_TRAIN_MASE_SCALE,
    ExecutionStatus,
    build_purged_direct_samples,
    load_canonical_data,
)
from src.featurengineering.builder import FeatureBuilder, FeatureConfig
from src.inference.pyfunc import (
    RainfallForecastingPyfuncWrapper,
    build_pyfunc_signature_and_example,
    get_serving_pip_requirements,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PackageModels")

HORIZON = 7
NUM_TEST_ORIGINS = 2184
SC31_CONFIG_A_EXPECTED_TEST_MAE = 3.724641
DEFAULT_MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "results"


def _build_feature_matrices(
    dev_train: pd.DataFrame,
    val_set: pd.DataFrame,
    test_set: pd.DataFrame,
    feature_level: int,
) -> Dict[str, Any]:
    """Fit FeatureBuilder on Dev Train only and construct strictly causal matrices."""
    cfg = FeatureConfig.from_level(feature_level)
    fb = FeatureBuilder(cfg).fit(dev_train)

    df_feat_train = fb.transform(dev_train, drop_warmup=False)
    feat_cols = [c for c in df_feat_train.columns if c not in ["Ngày", "Lượng mưa"]]

    mask = np.ones(len(dev_train) - HORIZON, dtype=bool)
    mask[:29] = False

    X_train, Y_train, _ = build_purged_direct_samples(
        df_feat_train,
        feature_cols=feat_cols,
        target_col="Lượng mưa",
        horizon=HORIZON,
        origin_mask=mask,
    )
    assert len(X_train) == 6293, f"Expected 6,293 Dev Train origins, got {len(X_train)}"

    fset_val = fb.build_feature_set(val_set, context_df=dev_train)
    val_origins = len(val_set) - HORIZON
    X_val = fset_val.X.iloc[:val_origins].values.astype(np.float64)

    context_test = pd.concat([dev_train, val_set], ignore_index=True)
    fset_test = fb.build_feature_set(test_set, context_df=context_test)
    X_test = fset_test.X.iloc[:NUM_TEST_ORIGINS].values.astype(np.float64)

    return {
        "feature_builder": fb,
        "feature_cols": feat_cols,
        "feature_count": len(feat_cols),
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "X_test": X_test,
    }


def _create_sc35_model() -> MultiOutputRegressor:
    return MultiOutputRegressor(
        XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=1.5,
            learning_rate=0.03,
            max_depth=5,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    )


def _create_lgbm_config_a_model() -> MultiOutputRegressor:
    return MultiOutputRegressor(
        LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            learning_rate=0.03,
            num_leaves=31,
            n_estimators=100,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1,
            random_state=42,
            n_jobs=-1,
        )
    )


def _refresh_sha256_checksums(out_dir: Path, existing_checksums: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Recompute SHA-256 digests for all packaged artifacts and tracked model files."""
    checksums: Dict[str, str] = dict(existing_checksums or {})
    core_files = [
        "sc35_xgboost_tweedie_f3.joblib",
        "feature_builder_f3.joblib",
        "sc30_lightgbm_tweedie_f2.joblib",
        "feature_builder_f2.joblib",
        "sc31_lightgbm_tweedie_f3.joblib",
        "standalone_inference.py",
    ]
    for h in range(1, HORIZON + 1):
        core_files.append(f"sc35_xgb_h{h}.json")

    candidates = set(core_files) | set(checksums.keys())
    for fname in sorted(candidates):
        fpath = out_dir / fname
        if fpath.exists() and fpath.is_file():
            checksums[fname] = compute_sha256(fpath)
    return checksums


def _default_metadata_skeleton() -> Dict[str, Any]:
    return {
        "project": "DS108 HCMC Daily Rainfall Forecasting (2000-2026)",
        "version": "1.0.0",
        "date_packaged": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "governance_frozen_constants": {
            "serving_model": "SC-35",
            "shadow_models": ["SC-30", "SC-31"],
            "forward_test_window_days": 180,
            "multiple_comparisons_correction": "Holm-Bonferroni / Bonferroni (m=2 pairs vs serving)",
            "eval_scale_dev_train_mm": CANONICAL_DEV_TRAIN_MASE_SCALE,
        },
        "models": {},
        "sha256_checksums": {},
        "timezone": "Asia/Ho_Chi_Minh",
    }


def _update_model_entry(
    models_section: Dict[str, Any],
    role_key: str,
    updates: Dict[str, Any],
) -> None:
    existing = models_section.get(role_key, {})
    merged = dict(existing)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            sub = dict(merged[k])
            sub.update(v)
            merged[k] = sub
        else:
            merged[k] = v
    models_section[role_key] = merged


def _register_pyfunc_version(
    tracker: MLflowTrainingTracker,
    run_id: str,
    model_path: Path,
    fb_path: Path,
    feature_level_int: int,
    feature_level_str: str,
    role_key: str,
    governance_role: str,
    scenario_id: str,
    dev_train: pd.DataFrame,
    checksums: Dict[str, str],
) -> Optional[str]:
    """Package and register a custom Pyfunc version with dual-resolution aliases/tags."""
    if not tracker.enabled or tracker.client is None:
        return None

    ModelRegistryGovernance.ensure_registered_model(
        tracker.client, DEFAULT_REGISTERED_MODEL_NAME
    )
    wrapper = RainfallForecastingPyfuncWrapper(
        model_path=str(model_path),
        feature_builder_path=str(fb_path),
        feature_level=feature_level_int,
        governance_role=role_key,
    )
    sig, input_example = build_pyfunc_signature_and_example(wrapper, dev_train)
    pip_reqs = get_serving_pip_requirements()

    artifacts_map = {
        "model": str(model_path),
        "feature_builder": str(fb_path),
    }
    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.pyfunc.log_model(
            artifact_path=f"pyfunc_{role_key}",
            python_model=wrapper,
            artifacts=artifacts_map,
            signature=sig,
            input_example=input_example,
            pip_requirements=pip_reqs,
            registered_model_name=DEFAULT_REGISTERED_MODEL_NAME,
        )

    reg_ver = getattr(model_info, "registered_model_version", None)
    if reg_ver is None:
        versions = tracker.client.search_model_versions(
            f"name = '{DEFAULT_REGISTERED_MODEL_NAME}'"
        )
        matching = [v for v in versions if getattr(v, "run_id", None) == run_id]
        matching.sort(key=lambda m: int(m.version), reverse=True)
        if matching:
            reg_ver = matching[0].version

    if reg_ver is not None:
        ModelRegistryGovernance.register_model_version_governance(
            client=tracker.client,
            model_name=DEFAULT_REGISTERED_MODEL_NAME,
            version=str(reg_ver),
            alias=role_key,
            governance_role=governance_role,
            scenario_id=scenario_id,
            feature_level=feature_level_str,
            sha256_model=checksums.get(model_path.name, ""),
            sha256_feature_builder=checksums.get(fb_path.name, ""),
            description=f"{scenario_id} packaged Pyfunc model ({role_key})",
        )
    return str(reg_ver) if reg_ver is not None else None


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Package SC-35, SC-30 Config A, and SC-31 Config A with bit/score parity & MLflow tracking."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Destination directory for packaged models & metadata.json",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable hosted DagsHub MLflow telemetry logging",
    )
    parser.add_argument(
        "--register-models",
        action="store_true",
        help="Also register new Pyfunc versions and governance aliases in MLflow Model Registry",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"

    logger.info("1. Loading canonical splits & benchmark NPZ parity baselines...")
    dev_train, val_set, test_set = load_canonical_data(return_dev_train=True)
    val_forecasts = np.load(RESULTS_DIR / "ml_tweedie_val_forecasts.npz")
    test_forecasts = np.load(RESULTS_DIR / "ml_tweedie_test_forecasts.npz")

    y_val_true = np.asarray(val_forecasts["y_true"], dtype=np.float64)
    y_test_true = np.asarray(test_forecasts["y_true"], dtype=np.float64)

    logger.info("2. Constructing causal F2 (65 cols) and F3 (70 cols) feature matrices...")
    feat_f2 = _build_feature_matrices(dev_train, val_set, test_set, feature_level=2)
    feat_f3 = _build_feature_matrices(dev_train, val_set, test_set, feature_level=3)

    tracker = MLflowTrainingTracker(
        experiment_name=DEFAULT_EXPERIMENT_NAME,
        partition="packaging",
        source="model_packaging",
        enabled=not args.no_mlflow,
    )

    # ------------------------------------------------------------------
    # Model 1: SC-35 XGBoost Tweedie F3 (serving)
    # ------------------------------------------------------------------
    logger.info("3. Training & parity-verifying SC-35 (serving)...")
    t0_fit = time.time()
    model_sc35 = _create_sc35_model()
    model_sc35.fit(feat_f3["X_train"], feat_f3["Y_train"])
    t_fit_35 = time.time() - t0_fit

    t0_inf = time.time()
    preds_val_35 = np.clip(model_sc35.predict(feat_f3["X_val"]), 0.0, None)
    preds_test_35 = np.clip(model_sc35.predict(feat_f3["X_test"]), 0.0, None)
    t_inf_35 = time.time() - t0_inf

    diff_val_35 = float(np.max(np.abs(preds_val_35 - val_forecasts["SC-35_xgboost_tweedie_F3"])))
    diff_test_35 = float(np.max(np.abs(preds_test_35 - test_forecasts["SC-35_xgboost_tweedie_F3_preds"])))
    logger.info(f"   SC-35 Parity -> Val max diff: {diff_val_35:.2e} | Test max diff: {diff_test_35:.2e}")
    assert diff_val_35 < 1e-5 and diff_test_35 < 1e-5, (
        f"SC-35 parity check failed (val={diff_val_35:.2e}, test={diff_test_35:.2e})"
    )

    metrics_val_35 = evaluate_point_forecast_metrics(y_val_true, preds_val_35, scale=CANONICAL_DEV_TRAIN_MASE_SCALE, horizon=HORIZON)
    metrics_test_35 = evaluate_point_forecast_metrics(y_test_true, preds_test_35, scale=CANONICAL_DEV_TRAIN_MASE_SCALE, horizon=HORIZON)

    sc35_path = out_dir / "sc35_xgboost_tweedie_f3.joblib"
    fb3_path = out_dir / "feature_builder_f3.joblib"
    joblib.dump(model_sc35, sc35_path)
    joblib.dump(feat_f3["feature_builder"], fb3_path)
    for h_idx, estimator in enumerate(model_sc35.estimators_, start=1):
        estimator.get_booster().save_model(str(out_dir / f"sc35_xgb_h{h_idx}.json"))

    # ------------------------------------------------------------------
    # Model 2: SC-30 LightGBM Tweedie F2 Config A (shadow_val_champ)
    # ------------------------------------------------------------------
    logger.info("4. Training & parity-verifying SC-30 Config A (shadow_val_champ)...")
    t0_fit = time.time()
    model_sc30 = _create_lgbm_config_a_model()
    model_sc30.fit(feat_f2["X_train"], feat_f2["Y_train"])
    t_fit_30 = time.time() - t0_fit

    t0_inf = time.time()
    preds_val_30 = np.clip(model_sc30.predict(feat_f2["X_val"]), 0.0, None)
    preds_test_30 = np.clip(model_sc30.predict(feat_f2["X_test"]), 0.0, None)
    t_inf_30 = time.time() - t0_inf

    diff_val_30 = float(np.max(np.abs(preds_val_30 - val_forecasts["SC-30_lightgbm_tweedie_F2"])))
    diff_test_30 = float(np.max(np.abs(preds_test_30 - test_forecasts["SC-30_lightgbm_tweedie_F2_preds"])))
    logger.info(f"   SC-30 Config A Parity -> Val max diff: {diff_val_30:.2e} | Test max diff: {diff_test_30:.2e}")
    assert diff_val_30 < 1e-5 and diff_test_30 < 1e-5, (
        f"SC-30 parity check failed (val={diff_val_30:.2e}, test={diff_test_30:.2e})"
    )

    metrics_val_30 = evaluate_point_forecast_metrics(y_val_true, preds_val_30, scale=CANONICAL_DEV_TRAIN_MASE_SCALE, horizon=HORIZON)
    metrics_test_30 = evaluate_point_forecast_metrics(y_test_true, preds_test_30, scale=CANONICAL_DEV_TRAIN_MASE_SCALE, horizon=HORIZON)

    sc30_path = out_dir / "sc30_lightgbm_tweedie_f2.joblib"
    fb2_path = out_dir / "feature_builder_f2.joblib"
    joblib.dump(model_sc30, sc30_path)
    joblib.dump(feat_f2["feature_builder"], fb2_path)

    # ------------------------------------------------------------------
    # Model 3: SC-31 LightGBM Tweedie F3 Config A (shadow_test_contender)
    # ------------------------------------------------------------------
    logger.info("5. Training & parity-verifying SC-31 Config A (shadow_test_contender)...")
    t0_fit = time.time()
    model_sc31 = _create_lgbm_config_a_model()
    model_sc31.fit(feat_f3["X_train"], feat_f3["Y_train"])
    t_fit_31 = time.time() - t0_fit

    t0_inf = time.time()
    preds_val_31 = np.clip(model_sc31.predict(feat_f3["X_val"]), 0.0, None)
    preds_test_31 = np.clip(model_sc31.predict(feat_f3["X_test"]), 0.0, None)
    t_inf_31 = time.time() - t0_inf

    diff_val_31 = float(np.max(np.abs(preds_val_31 - val_forecasts["SC-31_lightgbm_tweedie_F3"])))
    diff_test_31 = float(np.max(np.abs(preds_test_31 - test_forecasts["SC-31_lightgbm_tweedie_F3_preds"])))
    metrics_val_31 = evaluate_point_forecast_metrics(y_val_true, preds_val_31, scale=CANONICAL_DEV_TRAIN_MASE_SCALE, horizon=HORIZON)
    metrics_test_31 = evaluate_point_forecast_metrics(y_test_true, preds_test_31, scale=CANONICAL_DEV_TRAIN_MASE_SCALE, horizon=HORIZON)
    mae_err_31 = abs(metrics_test_31["MAE"] - SC31_CONFIG_A_EXPECTED_TEST_MAE)
    logger.info(
        f"   SC-31 Config A Parity -> Val max diff: {diff_val_31:.2e} | "
        f"Test max diff: {diff_test_31:.2e} | Test MAE: {metrics_test_31['MAE']:.6f} mm (diff={mae_err_31:.2e})"
    )
    assert diff_val_31 < 1e-5 and diff_test_31 < 1e-5, (
        f"SC-31 parity check failed (val={diff_val_31:.2e}, test={diff_test_31:.2e})"
    )
    assert mae_err_31 < 1e-4, f"SC-31 Config A Test MAE parity failed ({metrics_test_31['MAE']:.6f})"

    sc31_path = out_dir / "sc31_lightgbm_tweedie_f3.joblib"
    joblib.dump(model_sc31, sc31_path)

    # ------------------------------------------------------------------
    # Metadata & SHA-256 Checksums
    # ------------------------------------------------------------------
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = _default_metadata_skeleton()

    metadata["date_packaged"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    metadata.setdefault("governance_frozen_constants", _default_metadata_skeleton()["governance_frozen_constants"])
    metadata.setdefault("models", {})

    checksums = _refresh_sha256_checksums(out_dir, metadata.get("sha256_checksums"))
    metadata["sha256_checksums"] = checksums

    _update_model_entry(
        metadata["models"],
        "serving",
        {
            "scenario_id": "SC-35",
            "model_name": "xgboost_tweedie_F3",
            "registry_alias": "serving",
            "governance_role": "test_ranking_winner_as_originally_benchmarked",
            "feature_level": "F3",
            "feature_count": feat_f3["feature_count"],
            "artifacts": {
                "model": sc35_path.name,
                "feature_builder": fb3_path.name,
            },
            "benchmark_metrics": {
                "val_MAE": round(float(metrics_val_35["MAE"]), 6),
                "val_RMSE": round(float(metrics_val_35["RMSE"]), 6),
                "val_MASE": round(float(metrics_val_35["MASE"]), 6),
                "test_MAE": round(float(metrics_test_35["MAE"]), 6),
                "test_RMSE": round(float(metrics_test_35["RMSE"]), 6),
                "test_MASE": round(float(metrics_test_35["MASE"]), 6),
                "test_rank": 2,
                "parity_val_max_diff": diff_val_35,
                "parity_test_max_diff": diff_test_35,
            },
        },
    )

    _update_model_entry(
        metadata["models"],
        "shadow_val_champ",
        {
            "scenario_id": "SC-30",
            "model_name": "lightgbm_tweedie_F2",
            "registry_alias": "shadow_val_champ",
            "governance_role": "validation_selected_champion",
            "feature_level": "F2",
            "feature_count": feat_f2["feature_count"],
            "packaged_configuration": "ConfigA",
            "artifacts": {
                "model": sc30_path.name,
                "feature_builder": fb2_path.name,
            },
            "benchmark_metrics": {
                "val_MAE": round(float(metrics_val_30["MAE"]), 6),
                "val_RMSE": round(float(metrics_val_30["RMSE"]), 6),
                "val_MASE": round(float(metrics_val_30["MASE"]), 6),
                "test_MAE": round(float(metrics_test_30["MAE"]), 6),
                "test_RMSE": round(float(metrics_test_30["RMSE"]), 6),
                "test_MASE": round(float(metrics_test_30["MASE"]), 6),
                "test_MAE_config_a": round(float(metrics_test_30["MAE"]), 6),
                "test_RMSE_config_a": round(float(metrics_test_30["RMSE"]), 6),
                "test_MASE_config_a": round(float(metrics_test_30["MASE"]), 6),
                "test_rank": 3,
                "parity_val_max_diff": diff_val_30,
                "parity_test_max_diff": diff_test_30,
            },
        },
    )

    _update_model_entry(
        metadata["models"],
        "shadow_test_contender",
        {
            "scenario_id": "SC-31",
            "model_name": "lightgbm_tweedie_F3",
            "registry_alias": "shadow_test_contender",
            "governance_role": "post_audit_test_contender",
            "feature_level": "F3",
            "feature_count": feat_f3["feature_count"],
            "packaged_configuration": "ConfigA",
            "artifacts": {
                "model": sc31_path.name,
                "feature_builder": fb3_path.name,
            },
            "benchmark_metrics": {
                "val_MAE": round(float(metrics_val_31["MAE"]), 6),
                "val_RMSE": round(float(metrics_val_31["RMSE"]), 6),
                "val_MASE": round(float(metrics_val_31["MASE"]), 6),
                "test_MAE": round(float(metrics_test_31["MAE"]), 6),
                "test_RMSE": round(float(metrics_test_31["RMSE"]), 6),
                "test_MASE": round(float(metrics_test_31["MASE"]), 6),
                "test_MAE_config_a": round(float(metrics_test_31["MAE"]), 6),
                "test_RMSE_config_a": round(float(metrics_test_31["RMSE"]), 6),
                "test_MASE_config_a": round(float(metrics_test_31["MASE"]), 6),
                "test_rank": 1,
                "parity_val_max_diff": diff_val_31,
                "parity_test_max_diff": diff_test_31,
            },
        },
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"6. Saved artifacts & updated {meta_path} ({len(checksums)} SHA-256 checksums).")

    # ------------------------------------------------------------------
    # MLflow Telemetry & Optional Model Registry Registration
    # ------------------------------------------------------------------
    packaging_specs = [
        {
            "role_key": "serving",
            "governance_role": "test_ranking_winner_as_originally_benchmarked",
            "scenario_id": "SC-35",
            "model_name": "xgboost_tweedie_F3",
            "model": model_sc35,
            "model_family": "xgboost_tweedie",
            "feature_level": "F3",
            "feature_level_int": 3,
            "feature_names": feat_f3["feature_cols"],
            "feature_count": feat_f3["feature_count"],
            "config_label": None,
            "parameter_count": 7 * 100 * 32,
            "training_time": t_fit_35,
            "inference_time": t_inf_35,
            "metrics_val": metrics_val_35,
            "metrics_test": metrics_test_35,
            "parity_val_diff": diff_val_35,
            "parity_test_diff": diff_test_35,
            "model_path": sc35_path,
            "fb_path": fb3_path,
        },
        {
            "role_key": "shadow_val_champ",
            "governance_role": "validation_selected_champion",
            "scenario_id": "SC-30",
            "model_name": "lightgbm_tweedie_F2",
            "model": model_sc30,
            "model_family": "lightgbm_tweedie",
            "feature_level": "F2",
            "feature_level_int": 2,
            "feature_names": feat_f2["feature_cols"],
            "feature_count": feat_f2["feature_count"],
            "config_label": "ConfigA",
            "parameter_count": 7 * 100 * 31,
            "training_time": t_fit_30,
            "inference_time": t_inf_30,
            "metrics_val": metrics_val_30,
            "metrics_test": metrics_test_30,
            "parity_val_diff": diff_val_30,
            "parity_test_diff": diff_test_30,
            "model_path": sc30_path,
            "fb_path": fb2_path,
        },
        {
            "role_key": "shadow_test_contender",
            "governance_role": "post_audit_test_contender",
            "scenario_id": "SC-31",
            "model_name": "lightgbm_tweedie_F3",
            "model": model_sc31,
            "model_family": "lightgbm_tweedie",
            "feature_level": "F3",
            "feature_level_int": 3,
            "feature_names": feat_f3["feature_cols"],
            "feature_count": feat_f3["feature_count"],
            "config_label": "ConfigA",
            "parameter_count": 7 * 100 * 31,
            "training_time": t_fit_31,
            "inference_time": t_inf_31,
            "metrics_val": metrics_val_31,
            "metrics_test": metrics_test_31,
            "parity_val_diff": diff_val_31,
            "parity_test_diff": diff_test_31,
            "model_path": sc31_path,
            "fb_path": fb3_path,
        },
    ]

    run_ids: Dict[str, Optional[str]] = {}
    for spec in packaging_specs:
        extra_metrics: Dict[str, float] = {
            "packaging_val_MAE": float(spec["metrics_val"]["MAE"]),
            "packaging_test_MAE": float(spec["metrics_test"]["MAE"]),
            "parity_val_max_diff": float(spec["parity_val_diff"]),
        }
        if spec["parity_test_diff"] is not None:
            extra_metrics["parity_test_max_diff"] = float(spec["parity_test_diff"])

        run_id = tracker.log_training_run(
            scenario_id=spec["scenario_id"],
            model_name=spec["model_name"],
            model=spec["model"],
            detailed_metrics=spec["metrics_test"],
            training_time=spec["training_time"],
            inference_time=spec["inference_time"],
            feature_set=spec["feature_level"],
            feature_names=spec["feature_names"],
            feature_count=spec["feature_count"],
            model_family=spec["model_family"],
            strategy="Tweedie Compound Poisson-Gamma",
            parameter_count=spec["parameter_count"],
            config_label=spec["config_label"],
            partition="packaging",
            governance_role=spec["governance_role"],
            extra_tags={
                "registry_alias": spec["role_key"],
                "sha256_model": checksums.get(spec["model_path"].name, ""),
                "sha256_feature_builder": checksums.get(spec["fb_path"].name, ""),
                "packaging_verified": "true",
            },
            extra_metrics=extra_metrics,
            extra_artifacts={"packaging/metadata_snapshot.json": metadata},
            status=ExecutionStatus.SUCCESS.value,
        )
        run_ids[spec["scenario_id"]] = run_id

        if args.register_models and run_id is not None:
            reg_ver = _register_pyfunc_version(
                tracker=tracker,
                run_id=run_id,
                model_path=spec["model_path"],
                fb_path=spec["fb_path"],
                feature_level_int=spec["feature_level_int"],
                feature_level_str=spec["feature_level"],
                role_key=spec["role_key"],
                governance_role=spec["governance_role"],
                scenario_id=spec["scenario_id"],
                dev_train=dev_train,
                checksums=checksums,
            )
            logger.info(
                f"   Registered MLflow Model Registry version v{reg_ver} for @{spec['role_key']}"
            )

    logger.info("Packaging & Parity Verification COMPLETED successfully.")
    return {
        "metadata_path": str(meta_path),
        "checksums": checksums,
        "run_ids": run_ids,
        "parity": {
            "SC-35_val_diff": diff_val_35,
            "SC-35_test_diff": diff_test_35,
            "SC-30_val_diff": diff_val_30,
            "SC-30_test_diff": diff_test_30,
            "SC-31_val_diff": diff_val_31,
            "SC-31_test_diff": diff_test_31,
            "SC-31_test_MAE": float(metrics_test_31["MAE"]),
        },
    }


if __name__ == "__main__":
    main()
