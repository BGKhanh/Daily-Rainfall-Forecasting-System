"""
Standalone Serving Inference Class (DS108 Inference Module)
============================================================
Self-contained predictor for operational serving:
- Loads fitted models and feature builders from artifact directory.
- Requires only raw historical DataFrame (minimum 30 calendar days).
- Implements dynamic effective forecast calculation based on data_as_of vs run_date.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, Union, Optional

from src.inference.validation import validate_serving_input

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUPPORTED_MODEL_SOURCES = ("local", "registry")


class StandaloneRainfallPredictor:
    def __init__(
        self,
        artifact_dir: Optional[Union[str, Path]] = None,
        mode: Optional[str] = None,
    ):
        if artifact_dir is None:
            artifact_dir = PROJECT_ROOT / "artifacts" / "models"
        self.artifact_dir = Path(artifact_dir)
        self.mode = (mode or os.environ.get("SERVING_MODEL_SOURCE", "local")).strip().lower()
        if self.mode not in SUPPORTED_MODEL_SOURCES:
            raise ValueError(
                f"Unsupported serving mode '{self.mode}'. Expected one of {SUPPORTED_MODEL_SOURCES}."
            )

        # Load metadata (optional if booting purely from registry without local export dir)
        meta_path = self.artifact_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"model_source": self.mode}

        self._registry_pyfuncs: Dict[str, Any] = {}
        if self.mode == "local":
            self._load_local_artifacts()
        else:
            self._load_from_registry()

    def _load_local_artifacts(self) -> None:
        """Load serialized models and FeatureBuilders from local artifacts directory."""
        self.model_serving = joblib.load(self.artifact_dir / "sc35_xgboost_tweedie_f3.joblib")
        self.fb_f3 = joblib.load(self.artifact_dir / "feature_builder_f3.joblib")

        self.model_shadow_val = joblib.load(self.artifact_dir / "sc30_lightgbm_tweedie_f2.joblib")
        self.fb_f2 = joblib.load(self.artifact_dir / "feature_builder_f2.joblib")

        self.model_shadow_contender = joblib.load(self.artifact_dir / "sc31_lightgbm_tweedie_f3.joblib")

    def _load_from_registry(self) -> None:
        """Load models and FeatureBuilders directly from DagsHub MLflow Model Registry."""
        import mlflow.pyfunc
        from src.config.mlops import (
            DEFAULT_REGISTERED_MODEL_NAME,
            init_dagshub_mlflow,
            retry_with_backoff,
        )

        init_dagshub_mlflow()

        alias_map = {
            "serving": "serving",
            "shadow_val_champ": "shadow_val_champ",
            "shadow_test_contender": "shadow_test_contender",
        }
        loaded_pyfuncs: Dict[str, Any] = {}
        for key, alias in alias_map.items():
            uri = f"models:/{DEFAULT_REGISTERED_MODEL_NAME}@{alias}"
            loaded = retry_with_backoff(
                lambda u=uri: mlflow.pyfunc.load_model(u),
                operation_name=f"load_model({uri})",
            )
            loaded_pyfuncs[key] = loaded

        self._registry_pyfuncs = loaded_pyfuncs
        unwrapped_serving = (
            loaded_pyfuncs["serving"].unwrap_python_model()
            if hasattr(loaded_pyfuncs["serving"], "unwrap_python_model")
            else None
        )
        unwrapped_val = (
            loaded_pyfuncs["shadow_val_champ"].unwrap_python_model()
            if hasattr(loaded_pyfuncs["shadow_val_champ"], "unwrap_python_model")
            else None
        )
        unwrapped_contender = (
            loaded_pyfuncs["shadow_test_contender"].unwrap_python_model()
            if hasattr(loaded_pyfuncs["shadow_test_contender"], "unwrap_python_model")
            else None
        )

        if (
            unwrapped_serving is not None
            and unwrapped_val is not None
            and unwrapped_contender is not None
            and getattr(unwrapped_serving, "model", None) is not None
            and getattr(unwrapped_serving, "feature_builder", None) is not None
        ):
            self.model_serving = unwrapped_serving.model
            self.fb_f3 = unwrapped_serving.feature_builder
            self.model_shadow_val = unwrapped_val.model
            self.fb_f2 = unwrapped_val.feature_builder
            self.model_shadow_contender = unwrapped_contender.model
        else:
            self.model_serving = loaded_pyfuncs["serving"]
            self.model_shadow_val = loaded_pyfuncs["shadow_val_champ"]
            self.model_shadow_contender = loaded_pyfuncs["shadow_test_contender"]
            self.fb_f3 = joblib.load(self.artifact_dir / "feature_builder_f3.joblib")
            self.fb_f2 = joblib.load(self.artifact_dir / "feature_builder_f2.joblib")

    def predict(
        self,
        raw_history_df: pd.DataFrame,
        run_date: Optional[Union[str, date, datetime]] = None,
        date_col: str = "Ngày",
        target_col: str = "Lượng mưa",
        enforce_lag_limit: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute prediction from raw historical observations.

        Args:
            raw_history_df: DataFrame containing at least 30 continuous daily records.
            run_date: Date when prediction is issued (defaults to current system date).
            date_col: Name of the timestamp column.
            target_col: Name of rainfall target column.
            enforce_lag_limit: If True, reject histories older than 7 days before run_date.

        Returns:
            Structured JSON-ready dictionary conforming to Phase C API contract.
        """
        if run_date is None:
            run_date_str: Optional[str] = None
        elif isinstance(run_date, (datetime, date)):
            run_date_str = run_date.strftime("%Y-%m-%d")
        else:
            run_date_str = str(run_date)

        ctx = validate_serving_input(
            raw_history_df=raw_history_df,
            run_date=run_date_str,
            date_col=date_col,
            target_col=target_col,
            feature_builder=self.fb_f3,
            enforce_lag_limit=enforce_lag_limit,
        )
        from src.data.vintage_store import align_and_validate_canonical_raw_df

        df = align_and_validate_canonical_raw_df(ctx.df)
        data_as_of = ctx.data_as_of.date()
        run_d = ctx.effective_run_date.date()
        lag_days = ctx.lag_days
        effective_future_days = ctx.effective_future_days

        # 1. Feature Extraction via build_single_step
        feat_df3 = self.fb_f3.build_single_step(df)
        cols3 = [c for c in feat_df3.columns if c not in [self.fb_f3.date_col, self.fb_f3.target_col]]
        X3 = feat_df3[cols3].values.astype(np.float64)

        feat_df2 = self.fb_f2.build_single_step(df)
        cols2 = [c for c in feat_df2.columns if c not in [self.fb_f2.date_col, self.fb_f2.target_col]]
        X2 = feat_df2[cols2].values.astype(np.float64)

        # 2. Model Predictions (7 horizons)
        preds_serving = np.clip(self.model_serving.predict(X3)[0], 0.0, None).tolist()
        preds_shadow_val = np.clip(self.model_shadow_val.predict(X2)[0], 0.0, None).tolist()
        preds_shadow_contender = np.clip(self.model_shadow_contender.predict(X3)[0], 0.0, None).tolist()

        # 3. Dynamic Horizon Labeling
        hindcast_and_current = []
        forecast_future = []

        for h in range(1, 8):
            target_date = data_as_of + timedelta(days=h)
            pred_val = round(float(preds_serving[h - 1]), 2)

            if target_date < run_d:
                status = "past_unverified"
                item = {
                    "horizon": h,
                    "target_date": target_date.strftime("%Y-%m-%d"),
                    "pred_mm": pred_val,
                    "status": status,
                    "observed_mm": None,
                    "note": "Day has passed but observational data has not been released yet."
                }
                hindcast_and_current.append(item)
            elif target_date == run_d:
                status = "today"
                item = {
                    "horizon": h,
                    "target_date": target_date.strftime("%Y-%m-%d"),
                    "pred_mm": pred_val,
                    "status": status,
                    "observed_mm": None,
                    "note": "Current operational day (observation in progress)."
                }
                hindcast_and_current.append(item)
            else:
                status = "future_forecast"
                item = {
                    "horizon": h,
                    "target_date": target_date.strftime("%Y-%m-%d"),
                    "pred_mm": pred_val,
                    "status": status
                }
                forecast_future.append(item)

        return {
            "run_timestamp": datetime.now().isoformat() + "Z",
            "model_source": self.mode,
            "run_date": run_d.strftime("%Y-%m-%d"),
            "data_as_of": data_as_of.strftime("%Y-%m-%d"),
            "lag_days": lag_days,
            "effective_future_days": effective_future_days,
            "serving_model": {
                "scenario_id": "SC-35",
                "model_name": "xgboost_tweedie_F3",
                "governance_role": "test_ranking_winner_as_originally_benchmarked",
                "registry_alias": "serving",
                "feature_level": "F3",
                "predictions_7d_mm": [round(float(p), 2) for p in preds_serving],
                "hindcast_and_current": hindcast_and_current,
                "forecast_future": forecast_future
            },
            "shadow_models": {
                "shadow_val_champ": {
                    "scenario_id": "SC-30",
                    "model_name": "lightgbm_tweedie_F2",
                    "governance_role": "validation_selected_champion",
                    "registry_alias": "shadow_val_champ",
                    "feature_level": "F2",
                    "predictions_7d_mm": [round(float(p), 2) for p in preds_shadow_val]
                },
                "shadow_test_contender": {
                    "scenario_id": "SC-31",
                    "model_name": "lightgbm_tweedie_F3",
                    "governance_role": "post_audit_test_contender",
                    "registry_alias": "shadow_test_contender",
                    "feature_level": "F3",
                    "predictions_7d_mm": [round(float(p), 2) for p in preds_shadow_contender]
                }
            },
            # Backwards compatibility key
            "shadow_model": {
                "scenario_id": "SC-30",
                "model_name": "lightgbm_tweedie_F2",
                "governance_role": "validation_selected_champion",
                "predictions_7d_mm": [round(float(p), 2) for p in preds_shadow_val]
            }
        }

