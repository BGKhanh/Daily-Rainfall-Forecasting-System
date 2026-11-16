"""
MLflow Custom Pyfunc Model Wrapper for DS108 Rainfall Forecasting
==================================================================
Wraps multi-horizon direct Tweedie forecasters (SC-35, SC-30, SC-31)
with their fitted FeatureBuilder for end-to-end inference from weather history.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import mlflow.pyfunc
from mlflow.models import ModelSignature, infer_signature
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_SERVING_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "xgboost-cpu",
    "lightgbm",
    "joblib",
)


class RainfallForecastingPyfuncWrapper(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_builder_path: Optional[str] = None,
        feature_level: int = 3,
        governance_role: str = "serving",
    ):
        self.model_path = model_path
        self.feature_builder_path = feature_builder_path
        self.feature_level = feature_level
        self.governance_role = governance_role
        self.model = None
        self.feature_builder = None

    def load_context(self, context) -> None:
        """Load serialized model and FeatureBuilder from MLflow artifacts or fallback paths."""
        if context is not None and hasattr(context, "artifacts") and context.artifacts:
            if "model" in context.artifacts:
                self.model = joblib.load(context.artifacts["model"])
            if "feature_builder" in context.artifacts:
                self.feature_builder = joblib.load(context.artifacts["feature_builder"])
        if self.model is None and self.model_path:
            self.model = joblib.load(self.model_path)
        if self.feature_builder is None and self.feature_builder_path:
            self.feature_builder = joblib.load(self.feature_builder_path)

    def predict(
        self,
        context,
        model_input: Union[pd.DataFrame, np.ndarray],
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Inference entrypoint conforming to MLflow 2.x/3.x Pyfunc signature.

        If model_input is a DataFrame of recent weather history (>= 30 days with 'Ngày'):
            extracts causal single-step features and outputs 7-day forecast array (N, 7).
        If model_input is already a 2D feature matrix X:
            predicts directly.
        """
        if self.model is None or self.feature_builder is None:
            self.load_context(context)

        if isinstance(model_input, pd.DataFrame):
            date_col = getattr(self.feature_builder, "date_col", "Ngày")
            target_col = getattr(self.feature_builder, "target_col", "Lượng mưa")
            if date_col in model_input.columns:
                from src.inference.validation import validate_serving_input

                run_date = params.get("run_date") if isinstance(params, dict) else None
                enforce_lag = (
                    bool(params.get("enforce_lag_limit", run_date is not None))
                    if isinstance(params, dict)
                    else False
                )
                ctx = validate_serving_input(
                    raw_history_df=model_input,
                    run_date=run_date,
                    date_col=date_col,
                    target_col=target_col,
                    feature_builder=self.feature_builder,
                    enforce_lag_limit=enforce_lag,
                )
                df_single = self.feature_builder.build_single_step(ctx.df)
                feat_cols = [
                    c for c in df_single.columns if c not in [date_col, target_col]
                ]
                X = df_single[feat_cols].values.astype(np.float64)
            else:
                X = model_input.values.astype(np.float64)
        else:
            X = np.asarray(model_input, dtype=np.float64)

        if not np.isfinite(X).all():
            from src.inference.validation import ServingValidationError

            raise ServingValidationError(
                "INVALID_OR_MISSING_METEO_INPUT",
                "Feature matrix passed to Pyfunc wrapper contains NaN or Inf values.",
            )

        preds = np.clip(self.model.predict(X), 0.0, None)
        return np.asarray(preds, dtype=np.float64)


def build_pyfunc_signature_and_example(
    pyfunc_model: RainfallForecastingPyfuncWrapper,
    sample_history_df: pd.DataFrame,
) -> Tuple[ModelSignature, pd.DataFrame]:
    """
    Generate an explicit MLflow ModelSignature and 35-day standard input_example
    from raw weather history.
    """
    if len(sample_history_df) < 35:
        raise ValueError(
            f"sample_history_df must have at least 35 rows (got {len(sample_history_df)})."
        )
    input_example = sample_history_df.iloc[-35:].copy().reset_index(drop=True)
    if pyfunc_model.model is None or pyfunc_model.feature_builder is None:
        pyfunc_model.load_context(None)

    sample_output = pyfunc_model.predict(None, input_example)
    signature = infer_signature(input_example, sample_output)
    return signature, input_example


def get_serving_pip_requirements(
    req_path: Optional[Path] = None,
) -> List[str]:
    """
    Extract pinned core inference library versions from requirements-serving.txt
    for reproducible MLflow Pyfunc packaging.
    """
    path = Path(req_path) if req_path is not None else PROJECT_ROOT / "requirements-serving.txt"
    if not path.exists():
        return list(CORE_SERVING_PACKAGES)

    selected: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].strip().lower()
            if pkg_name in CORE_SERVING_PACKAGES:
                selected.append(line)
    return selected
