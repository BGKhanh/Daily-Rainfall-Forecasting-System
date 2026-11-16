"""Evaluation and benchmarking module for rainfall prediction.

Submodules:
- metrics: Cumulative rainfall metrics (CR-MAE@H, CR-Bias@H) and rain-day metrics.
- protocols: Canonical split, Horizon Purge, hardware detection, resource tracking, failure protocol, contracts.
- benchmark: UnifiedBenchmark engine, RollingBenchmarkCoordinator, and statistical selection/inference.
"""

from .metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_scaled_error,
    evaluate_point_forecast_metrics,
    CANONICAL_DEV_TRAIN_MASE_SCALE,
    rain_day_mae,
    rain_day_rmse,
)

from .protocols import (
    CANONICAL_PREDICTIONS_SCHEMA,
    UNIFIED_SCHEMA,
    ExecutionStatus,
    ModelTerminalStatus,
    SystemEvaluationStatus,
    InferenceModuleStatus,
    GatecheckStatus,
    FutureCovariatePolicy,
    GENUINELY_KNOWN_FUTURE_COLUMNS,
    get_hardware_metadata,
    measure_execution_resource,
    load_canonical_data,
    load_full_refit_data,
    build_purged_direct_samples,
    build_causal_direct_features,
    compute_benchmark_metrics,
    save_benchmark_artifacts,
    get_default_contract,
    save_modeling_contract,
    load_modeling_contract,
    evaluate_six_gatechecks,
    gatecheck_threshold_source_alignment,
    gatecheck_target_boundary_purge_h7,
    gatecheck_feature_causality_and_split_isolation,
    FORECAST_HORIZON,
    CONTEXT_LENGTH,
    SEASON_LENGTH,
    RAIN_THRESHOLD,
    RAIN_COMPARISON_OPERATOR,
    CANONICAL_RAIN_THRESHOLD,
    CANONICAL_RAIN_THRESHOLD_OPERATOR,
    LEGACY_EDA_RAIN_THRESHOLD,
    LEGACY_EDA_RAIN_THRESHOLD_OPERATOR,
    validate_and_normalize_canonical_threshold,
    DataLeakageSecurityError,
    PreReadFileReadGuard,
    get_global_read_guard,
    TARGET_COL,
    DATE_COL,
)

_LAZY_MLFLOW_TRACKER_EXPORTS = {
    "InMemoryStepLossCallback",
    "get_neural_trainer_kwargs",
    "SklearnMultiOutputAdapter",
    "HurdleModelAdapter",
    "NixtlaWrapperAdapter",
    "NeuralForecastAdapter",
    "detect_model_adapter",
    "extract_model_hyperparams",
    "extract_feature_importance_payload",
    "extract_hurdle_threshold_grid",
    "resolve_run_name",
    "MLflowTrainingTracker",
}


def __getattr__(name: str):
    if name in _LAZY_MLFLOW_TRACKER_EXPORTS:
        from . import mlflow_tracker

        return getattr(mlflow_tracker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Canonical Point Metrics & Diagnostics
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_scaled_error",
    "evaluate_point_forecast_metrics",
    "CANONICAL_DEV_TRAIN_MASE_SCALE",
    "rain_day_mae",
    "rain_day_rmse",
    # Protocols & Contracts
    "CANONICAL_PREDICTIONS_SCHEMA",
    "UNIFIED_SCHEMA",
    "ExecutionStatus",
    "ModelTerminalStatus",
    "SystemEvaluationStatus",
    "InferenceModuleStatus",
    "GatecheckStatus",
    "FutureCovariatePolicy",
    "GENUINELY_KNOWN_FUTURE_COLUMNS",
    "get_hardware_metadata",
    "measure_execution_resource",
    "load_canonical_data",
    "load_full_refit_data",
    "build_purged_direct_samples",
    "build_causal_direct_features",
    "compute_benchmark_metrics",
    "save_benchmark_artifacts",
    "get_default_contract",
    "save_modeling_contract",
    "load_modeling_contract",
    "evaluate_six_gatechecks",
    "gatecheck_threshold_source_alignment",
    "gatecheck_target_boundary_purge_h7",
    "gatecheck_feature_causality_and_split_isolation",
    "FORECAST_HORIZON",
    "CONTEXT_LENGTH",
    "SEASON_LENGTH",
    "RAIN_THRESHOLD",
    "RAIN_COMPARISON_OPERATOR",
    "TARGET_COL",
    "DATE_COL",
    # Deep MLflow Tracker & Ecosystem Adapters
    "InMemoryStepLossCallback",
    "get_neural_trainer_kwargs",
    "SklearnMultiOutputAdapter",
    "HurdleModelAdapter",
    "NixtlaWrapperAdapter",
    "NeuralForecastAdapter",
    "detect_model_adapter",
    "extract_model_hyperparams",
    "extract_feature_importance_payload",
    "extract_hurdle_threshold_grid",
    "resolve_run_name",
    "MLflowTrainingTracker",
]
