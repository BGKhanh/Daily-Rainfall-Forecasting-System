"""Deep MLflow Training Tracker & Multi-Ecosystem Model Adapters.

Connects live training benchmarks (StatsForecast/Intermittent, Sklearn/XGB/LGBM Direct
& Tweedie, Two-Stage DirectHurdleForecaster, and PyTorch Lightning / NeuralForecast)
directly to the hosted DagsHub MLflow server without local tracking DBs or disk bloat.

Guarantees:
1. Calls `mlflow.autolog(disable=True)` to avoid DagsHub HTTP 429 storms.
2. Buffers PyTorch Lightning step losses strictly in RAM (`InMemoryStepLossCallback`)
   and flushes in chunks via `client.log_batch()` wrapped in `retry_with_backoff()`.
3. Sets `logger=False, enable_checkpointing=False, enable_progress_bar=False` via
   `get_neural_trainer_kwargs()` so `lightning_logs/` is never created.
4. Enforces read-only inspection for Nixtla `StatsForecastWrapper` and
   `FrozenIntermittentWrapper` to preserve `_state_snapshot` invariance.
5. Rejects purged metrics (CR-MAE, CR-Bias, RainDay-MAE) and supports graceful
   degradation when `--no-mlflow` is set or DagsHub is unreachable offline.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np

try:
    import mlflow
    from mlflow.entities import Metric, Param, RunTag
    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None
    Metric = Param = RunTag = None
    MLFLOW_AVAILABLE = False

try:
    from pytorch_lightning.callbacks import Callback as _PLCallback
    PL_CALLBACK_AVAILABLE = True
except ImportError:  # pragma: no cover
    class _PLCallback:  # type: ignore[no-redef]
        pass

    PL_CALLBACK_AVAILABLE = False

from src.analysis.loaders import assign_canonical_family
from src.config.mlops import (
    DEFAULT_EXPERIMENT_NAME,
    get_or_create_experiment,
    init_dagshub_mlflow,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)

# DagsHub/MLflow REST batch chunk caps (well below MLflow's 1000-entity max)
MLFLOW_BATCH_METRIC_LIMIT = 400
MLFLOW_BATCH_PARAM_LIMIT = 50
MLFLOW_BATCH_TAG_LIMIT = 50
MLFLOW_PARAM_VALUE_MAX_LEN = 500

# Forbidden purged metrics (Decision: Evaluation Standards Rule 3)
_PURGED_METRIC_SUBSTRINGS = (
    "cr-mae",
    "cr_mae",
    "cr-bias",
    "cr_bias",
    "rainday",
    "rain_day",
)


def _is_purged_metric_key(key: str) -> bool:
    low = str(key).strip().lower()
    return any(tok in low for tok in _PURGED_METRIC_SUBSTRINGS)


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num):
        return None
    return num


def _to_param_str(val: Any, max_len: int = MLFLOW_PARAM_VALUE_MAX_LEN) -> str:
    if isinstance(val, (np.floating, float)):
        s = f"{float(val):.8g}"
    elif isinstance(val, (np.integer, int)) and not isinstance(val, bool):
        s = str(int(val))
    elif isinstance(val, (np.ndarray,)):
        s = str(val.tolist())
    elif isinstance(val, (list, tuple, dict, set)):
        s = str(val)
    else:
        s = str(val)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _sanitize_param_dict(raw: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Convert nested or numpy-typed parameters into clean flat primitive values."""
    clean: Dict[str, Any] = {}
    for k, v in raw.items():
        if v is None:
            continue
        key_name = f"{prefix}{k}" if prefix else str(k)
        if callable(v) and not isinstance(v, (int, float, str, bool)):
            clean[key_name] = getattr(v, "__name__", type(v).__name__)
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if sub_v is not None:
                    clean[f"{key_name}_{sub_k}"] = _to_primitive_scalar(sub_v)
        else:
            clean[key_name] = _to_primitive_scalar(v)
    return clean


def _to_primitive_scalar(v: Any) -> Any:
    if isinstance(v, (bool, str)):
        return v
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        fv = float(v)
        return fv if np.isfinite(fv) else str(v)
    if isinstance(v, (tuple, list)):
        return [_to_primitive_scalar(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_to_primitive_scalar(x) for x in v.tolist()]
    return _to_param_str(v)


def _unwrap_sklearn_estimator(est: Any) -> Any:
    """Unwrap sklearn Pipeline to its final estimator without altering fitted state."""
    if hasattr(est, "steps") and getattr(est, "steps"):
        return est.steps[-1][1]
    return est


def _extract_single_importance_vector(est: Any) -> Tuple[Optional[np.ndarray], str]:
    core = _unwrap_sklearn_estimator(est)
    if hasattr(core, "feature_importances_"):
        arr = np.asarray(getattr(core, "feature_importances_"), dtype=np.float64).ravel()
        return arr, "feature_importances_"
    if hasattr(core, "coef_"):
        arr = np.abs(np.asarray(getattr(core, "coef_"), dtype=np.float64)).ravel()
        return arr, "abs_coef_"
    return None, ""


# ==============================================================================
# PyTorch Lightning In-Memory Loss Callback & Trainer Hygiene Helpers
# ==============================================================================
class InMemoryStepLossCallback(_PLCallback):
    """PyTorch Lightning Callback buffering per-step & per-epoch losses purely in RAM.

    Avoids TensorBoard/CSVLogger/checkpoint disk artifacts and DagsHub rate limits
    by keeping training telemetry in memory until batch logging completes.
    """

    def __init__(self, log_every_n_steps: int = 1) -> None:
        super().__init__()
        self.log_every_n_steps: int = max(1, int(log_every_n_steps))
        self.loss_history: List[Tuple[int, float]] = []
        self.epoch_losses: List[Tuple[int, float]] = []
        self._seen_steps: set[int] = set()
        self._parent_callback: Optional["InMemoryStepLossCallback"] = None

    def __deepcopy__(self, memo: Dict[int, Any]) -> "InMemoryStepLossCallback":
        clone = type(self)(log_every_n_steps=self.log_every_n_steps)
        memo[id(self)] = clone
        clone._parent_callback = self
        clone.loss_history = list(self.loss_history)
        clone.epoch_losses = list(self.epoch_losses)
        clone._seen_steps = set(self._seen_steps)
        return clone

    def clear(self) -> None:
        self.loss_history.clear()
        self.epoch_losses.clear()
        self._seen_steps.clear()

    def _record_step(self, step_idx: int, loss_val: float) -> None:
        if step_idx in self._seen_steps or step_idx % self.log_every_n_steps != 0:
            return
        self.loss_history.append((step_idx, loss_val))
        self._seen_steps.add(step_idx)
        if self._parent_callback is not None:
            self._parent_callback._record_step(step_idx, loss_val)

    def _record_epoch(self, epoch_idx: int, loss_val: float) -> None:
        self.epoch_losses.append((epoch_idx, loss_val))
        if self._parent_callback is not None:
            self._parent_callback._record_epoch(epoch_idx, loss_val)

    @staticmethod
    def _extract_loss_value(outputs: Any, trainer: Any) -> Optional[float]:
        raw = None
        if isinstance(outputs, dict) and "loss" in outputs:
            raw = outputs["loss"]
        elif outputs is not None and not isinstance(outputs, (list, tuple, dict)):
            raw = outputs

        if raw is None and trainer is not None:
            cb_metrics = getattr(trainer, "callback_metrics", {}) or {}
            raw = cb_metrics.get("train_loss", cb_metrics.get("train_loss_step"))

        if raw is None:
            return None
        try:
            if hasattr(raw, "detach"):
                raw = raw.detach()
            if hasattr(raw, "item"):
                raw = raw.item()
            val = float(raw)
            return val if np.isfinite(val) else None
        except (TypeError, ValueError, RuntimeError):
            return None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        global_step = int(getattr(trainer, "global_step", len(self.loss_history)))
        # Lightning executes optimizer step around batch_end; deduplicate step index
        step_idx = max(1, global_step) if global_step > 0 else len(self.loss_history) + 1
        if step_idx in self._seen_steps or step_idx % self.log_every_n_steps != 0:
            return
        loss_val = self._extract_loss_value(outputs, trainer)
        if loss_val is not None:
            self._record_step(step_idx, loss_val)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        cb_metrics = getattr(trainer, "callback_metrics", {}) or {}
        raw = cb_metrics.get("train_loss_epoch", cb_metrics.get("train_loss"))
        if raw is None:
            return
        try:
            if hasattr(raw, "detach"):
                raw = raw.detach()
            if hasattr(raw, "item"):
                raw = raw.item()
            val = float(raw)
            if np.isfinite(val):
                epoch_idx = int(getattr(trainer, "current_epoch", len(self.epoch_losses)))
                self._record_epoch(epoch_idx, val)
        except (TypeError, ValueError, RuntimeError):
            pass

    def on_train_end(self, trainer, pl_module) -> None:
        trajectories = getattr(pl_module, "train_trajectories", None)
        if not self.loss_history and trajectories:
            for idx, item in enumerate(trajectories, start=1):
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    s_raw, v_raw = item[0], item[1]
                    s_int = max(1, int(s_raw)) if int(s_raw) > 0 else idx
                    v_float = _safe_float(v_raw)
                    if v_float is not None:
                        self._record_step(s_int, v_float)

    def get_step_metrics(self) -> List[Tuple[int, float]]:
        return sorted(self.loss_history, key=lambda pair: pair[0])


def get_neural_trainer_kwargs(callbacks: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """Return NeuralForecast/PyTorch Lightning trainer kwargs preventing local logs."""
    kwargs: Dict[str, Any] = {
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
    }
    if callbacks:
        kwargs["callbacks"] = list(callbacks)
    return kwargs


# ==============================================================================
# 4 Ecosystem Model Adapters
# ==============================================================================
class SklearnMultiOutputAdapter:
    """Adapter for Scikit-Learn, XGBoost, LightGBM Direct & Tweedie models."""

    name = "SklearnMultiOutputAdapter"

    @staticmethod
    def matches(model: Any) -> bool:
        if model is None:
            return False
        if HurdleModelAdapter.matches(model) or NixtlaWrapperAdapter.matches(model) or NeuralForecastAdapter.matches(model):
            return False
        return hasattr(model, "estimators_") or hasattr(model, "get_params")

    @staticmethod
    def extract_hyperparams(model: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {"adapter": SklearnMultiOutputAdapter.name}
        if hasattr(model, "estimators_") and getattr(model, "estimators_"):
            estimators = list(getattr(model, "estimators_"))
            base = estimators[0]
            params["wrapper_type"] = type(model).__name__
            params["base_estimator_type"] = type(_unwrap_sklearn_estimator(base)).__name__
            params["n_outputs"] = len(estimators)
            if hasattr(base, "get_params"):
                params.update(_sanitize_param_dict(base.get_params(deep=False)))
        elif hasattr(model, "estimator") and hasattr(model.estimator, "get_params"):
            params["wrapper_type"] = type(model).__name__
            params["base_estimator_type"] = type(model.estimator).__name__
            params.update(_sanitize_param_dict(model.estimator.get_params(deep=False)))
        elif hasattr(model, "get_params"):
            params["estimator_type"] = type(model).__name__
            params.update(_sanitize_param_dict(model.get_params(deep=False)))
        return params

    @staticmethod
    def extract_feature_importance(
        model: Any,
        feature_names: Optional[Sequence[str]] = None,
        top_k: int = 30,
    ) -> Optional[Dict[str, Any]]:
        if not SklearnMultiOutputAdapter.matches(model):
            return None

        estimators: List[Any]
        if hasattr(model, "estimators_") and getattr(model, "estimators_"):
            estimators = list(getattr(model, "estimators_"))
        else:
            estimators = [model]

        by_horizon: Dict[str, Dict[str, float]] = {}
        vectors: List[np.ndarray] = []
        imp_type = ""
        n_features: Optional[int] = None
        names: Optional[List[str]] = None

        for h_idx, est in enumerate(estimators, start=1):
            vec, kind = _extract_single_importance_vector(est)
            if vec is None:
                continue
            if n_features is None:
                n_features = len(vec)
                names = (
                    list(feature_names)
                    if feature_names is not None and len(feature_names) == n_features
                    else [f"f_{i}" for i in range(n_features)]
                )
            elif len(vec) != n_features:
                continue

            imp_type = kind
            vectors.append(vec)
            by_horizon[f"h{h_idx}"] = {
                names[i]: round(float(vec[i]), 8) for i in range(n_features)
            }

        if not vectors or names is None or n_features is None:
            return None

        mean_imp = np.mean(np.vstack(vectors), axis=0)
        total = float(np.sum(mean_imp))
        order = np.argsort(-mean_imp)

        top_features = []
        for rank, idx in enumerate(order[:top_k], start=1):
            mean_val = float(mean_imp[idx])
            share = mean_val / total if total > 0.0 else 0.0
            top_features.append(
                {
                    "rank": rank,
                    "feature": names[idx],
                    "mean_importance": round(mean_val, 8),
                    "normalized_share": round(share, 8),
                }
            )

        base_name = type(_unwrap_sklearn_estimator(estimators[0])).__name__
        return {
            "adapter": SklearnMultiOutputAdapter.name,
            "model_type": base_name,
            "n_horizons": len(vectors),
            "n_features": n_features,
            "importance_type": imp_type,
            "top_features": top_features,
            "by_horizon": by_horizon,
        }


class HurdleModelAdapter:
    """Adapter for Two-Stage DirectHurdleForecaster (Stage 1 Clf + Stage 2 Reg)."""

    name = "HurdleModelAdapter"

    @staticmethod
    def matches(model: Any) -> bool:
        return (
            model is not None
            and hasattr(model, "classifiers_")
            and hasattr(model, "regressors_")
            and hasattr(model, "rain_threshold")
        )

    @staticmethod
    def extract_hyperparams(model: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "adapter": HurdleModelAdapter.name,
            "model_alias": getattr(model, "model_alias", type(model).__name__),
            "horizon": int(getattr(model, "horizon", 7)),
            "rain_threshold": float(getattr(model, "rain_threshold", 1.0)),
            "selected_strategy": str(getattr(model, "selected_strategy_", getattr(model, "strategy", ""))),
            "selected_tau_clf": float(getattr(model, "selected_tau_clf_", getattr(model, "tau_clf", 0.5))),
            "random_state": int(getattr(model, "random_state", 42)),
        }
        dry_mean = _safe_float(getattr(model, "fitted_dry_mean_", None))
        if dry_mean is not None:
            params["fitted_dry_mean_mm"] = round(dry_mean, 7)

        wet_counts = getattr(model, "horizon_wet_counts_", None)
        if wet_counts:
            params["horizon_wet_counts"] = _to_param_str(list(wet_counts))

        flags = getattr(model, "warning_flags_", None)
        if flags:
            params["warning_flags"] = ",".join(str(f) for f in flags)

        clfs = getattr(model, "classifiers_", []) or []
        regs = getattr(model, "regressors_", []) or []
        if clfs:
            core_clf = _unwrap_sklearn_estimator(clfs[0])
            params["classifier_type"] = type(core_clf).__name__
            if hasattr(core_clf, "get_params"):
                params.update(_sanitize_param_dict(core_clf.get_params(deep=False), prefix="clf_"))
        if regs:
            core_reg = _unwrap_sklearn_estimator(regs[0])
            params["regressor_type"] = type(core_reg).__name__
            if hasattr(core_reg, "get_params"):
                params.update(_sanitize_param_dict(core_reg.get_params(deep=False), prefix="reg_"))
        return params

    @staticmethod
    def _extract_stage_importance(
        estimators: Sequence[Any],
        feature_names: Optional[Sequence[str]],
        stage_label: str,
        top_k: int = 30,
    ) -> Optional[Dict[str, Any]]:
        if not estimators:
            return None
        by_horizon: Dict[str, Dict[str, float]] = {}
        vectors: List[np.ndarray] = []
        imp_type = ""
        n_features: Optional[int] = None
        names: Optional[List[str]] = None

        for h_idx, est in enumerate(estimators, start=1):
            vec, kind = _extract_single_importance_vector(est)
            if vec is None:
                continue
            if n_features is None:
                n_features = len(vec)
                names = (
                    list(feature_names)
                    if feature_names is not None and len(feature_names) == n_features
                    else [f"f_{i}" for i in range(n_features)]
                )
            elif len(vec) != n_features:
                continue
            imp_type = kind
            vectors.append(vec)
            by_horizon[f"h{h_idx}"] = {
                names[i]: round(float(vec[i]), 8) for i in range(n_features)
            }

        if not vectors or names is None or n_features is None:
            return None

        mean_imp = np.mean(np.vstack(vectors), axis=0)
        total = float(np.sum(mean_imp))
        order = np.argsort(-mean_imp)
        top_features = [
            {
                "rank": rank,
                "feature": names[idx],
                "mean_importance": round(float(mean_imp[idx]), 8),
                "normalized_share": round(float(mean_imp[idx]) / total if total > 0.0 else 0.0, 8),
            }
            for rank, idx in enumerate(order[:top_k], start=1)
        ]

        return {
            "stage": stage_label,
            "model_type": type(_unwrap_sklearn_estimator(estimators[0])).__name__,
            "n_horizons": len(vectors),
            "n_features": n_features,
            "importance_type": imp_type,
            "top_features": top_features,
            "by_horizon": by_horizon,
        }

    @staticmethod
    def extract_feature_importance(
        model: Any,
        feature_names: Optional[Sequence[str]] = None,
        top_k: int = 30,
    ) -> Optional[Dict[str, Any]]:
        if not HurdleModelAdapter.matches(model):
            return None

        stage1 = HurdleModelAdapter._extract_stage_importance(
            getattr(model, "classifiers_", []), feature_names, "stage1_classifier", top_k=top_k
        )
        stage2 = HurdleModelAdapter._extract_stage_importance(
            getattr(model, "regressors_", []), feature_names, "stage2_regressor", top_k=top_k
        )
        if stage1 is None and stage2 is None:
            return None

        primary = stage2 or stage1
        return {
            "adapter": HurdleModelAdapter.name,
            "model_alias": getattr(model, "model_alias", type(model).__name__),
            "selected_strategy": str(getattr(model, "selected_strategy_", getattr(model, "strategy", ""))),
            "selected_tau_clf": float(getattr(model, "selected_tau_clf_", getattr(model, "tau_clf", 0.5))),
            "fitted_dry_mean_mm": _safe_float(getattr(model, "fitted_dry_mean_", None)),
            "horizon_wet_counts": list(getattr(model, "horizon_wet_counts_", [])),
            "n_features": primary["n_features"],
            "top_features": primary["top_features"],
            "stage1_classifier": stage1,
            "stage2_regressor": stage2,
        }


class NixtlaWrapperAdapter:
    """Read-only Adapter for StatsForecastWrapper & FrozenIntermittentWrapper.

    Strictly avoids mutating any wrapper attribute or invoking mutating methods so
    `wrapper._state_snapshot == _deep_snapshot_state(...)` invariance remains intact.
    """

    name = "NixtlaWrapperAdapter"

    @staticmethod
    def matches(model: Any) -> bool:
        if model is None:
            return False
        cls_name = type(model).__name__
        if cls_name in ("StatsForecastWrapper", "FrozenIntermittentWrapper"):
            return True
        return hasattr(model, "_state_snapshot") and (
            hasattr(model, "base_model") or hasattr(model, "wet_day_spacing_k0_")
        )

    @staticmethod
    def extract_hyperparams(model: Any) -> Dict[str, Any]:
        cls_name = type(model).__name__
        params: Dict[str, Any] = {
            "adapter": NixtlaWrapperAdapter.name,
            "wrapper_class": cls_name,
            "alias": str(getattr(model, "alias", cls_name)),
            "fitted": bool(getattr(model, "fitted_", False)),
            "n_train": int(getattr(model, "n_train_", 0)),
        }

        if hasattr(model, "base_model"):
            base = getattr(model, "base_model")
            params["base_model_type"] = type(base).__name__
            if hasattr(base, "__dict__"):
                for k, v in base.__dict__.items():
                    if k.startswith("_") or k == "model_":
                        continue
                    if isinstance(v, (int, float, str, bool, tuple, list)):
                        params[f"sf_{k}"] = _to_primitive_scalar(v)

            fitted_sf = getattr(model, "model_", None)
            if fitted_sf is not None and hasattr(fitted_sf, "model_") and isinstance(getattr(fitted_sf, "model_", None), dict):
                sf_state = getattr(fitted_sf, "model_")
                for state_key in ("aic", "aicc", "bic", "sigma2", "loglik", "method"):
                    if state_key in sf_state:
                        val = _to_primitive_scalar(sf_state[state_key])
                        if val is not None:
                            params[f"fitted_{state_key}"] = val

        if hasattr(model, "wet_day_spacing_k0_"):
            for attr in (
                "model_name",
                "rain_threshold",
                "wet_day_spacing_k0_",
                "domain_bucket_size_",
                "alpha_d_",
                "alpha_p_",
                "sba_factor_",
                "aggregation_chunk_size_",
                "max_aggregation_level_",
                "aggregation_levels_",
                "level_alphas_",
            ):
                val = getattr(model, attr, None)
                if val is not None:
                    clean_key = attr.rstrip("_")
                    prim = _to_primitive_scalar(val)
                    if prim is not None and not (isinstance(prim, float) and not np.isfinite(prim)):
                        params[clean_key] = prim

        return params


class NeuralForecastAdapter:
    """Adapter for Nixtla NeuralForecast & PyTorch Lightning forecasting modules."""

    name = "NeuralForecastAdapter"

    @staticmethod
    def _unwrap_module(model: Any) -> Any:
        if hasattr(model, "models") and getattr(model, "models"):
            models_seq = list(getattr(model, "models"))
            if models_seq:
                return models_seq[0]
        return model

    @staticmethod
    def matches(model: Any) -> bool:
        if model is None:
            return False
        mod = NeuralForecastAdapter._unwrap_module(model)
        if hasattr(mod, "train_trajectories") and hasattr(mod, "input_size"):
            return True
        cls_mod = getattr(type(mod), "__module__", "")
        return ("neuralforecast" in cls_mod or "pytorch_lightning" in cls_mod) and hasattr(mod, "parameters")

    @staticmethod
    def count_trainable_parameters(model: Any) -> int:
        mod = NeuralForecastAdapter._unwrap_module(model)
        if not hasattr(mod, "parameters"):
            return 0
        try:
            return int(sum(p.numel() for p in mod.parameters() if getattr(p, "requires_grad", False)))
        except Exception:
            return 0

    @staticmethod
    def extract_hyperparams(model: Any) -> Dict[str, Any]:
        mod = NeuralForecastAdapter._unwrap_module(model)
        params: Dict[str, Any] = {
            "adapter": NeuralForecastAdapter.name,
            "architecture": type(mod).__name__,
            "parameter_count": NeuralForecastAdapter.count_trainable_parameters(mod),
        }
        common_attrs = (
            "h",
            "input_size",
            "max_steps",
            "batch_size",
            "learning_rate",
            "random_seed",
            "n_blocks",
            "mlp_units",
            "n_pool_kernel_size",
            "patch_len",
            "stride",
            "hidden_size",
            "n_heads",
            "n_head",
            "linear_hidden_size",
            "dropout",
            "encoder_hidden_size",
            "encoder_n_layers",
            "kernel_size",
            "dilations",
            "revin",
        )
        for attr in common_attrs:
            val = getattr(mod, attr, None)
            if val is not None:
                params[attr] = _to_primitive_scalar(val)

        loss_obj = getattr(mod, "loss", None)
        if loss_obj is not None:
            params["loss_type"] = type(loss_obj).__name__

        trainer_kw = getattr(mod, "trainer_kwargs", None)
        if isinstance(trainer_kw, dict):
            for tk in ("logger", "enable_checkpointing", "enable_progress_bar", "accelerator"):
                if tk in trainer_kw:
                    params[f"trainer_{tk}"] = _to_primitive_scalar(trainer_kw[tk])

        return params

    @staticmethod
    def extract_loss_history(model: Any) -> List[Tuple[int, float]]:
        mod = NeuralForecastAdapter._unwrap_module(model)
        trainer_kw = getattr(mod, "trainer_kwargs", None)
        if isinstance(trainer_kw, dict):
            for cb in trainer_kw.get("callbacks", []) or []:
                if isinstance(cb, InMemoryStepLossCallback):
                    cb_hist = cb.get_step_metrics()
                    if cb_hist:
                        return cb_hist
        trajectories = getattr(mod, "train_trajectories", None) or []
        history: List[Tuple[int, float]] = []
        seen: set[int] = set()
        for idx, entry in enumerate(trajectories, start=1):
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                step = max(1, int(entry[0])) if int(entry[0]) > 0 else idx
                val = _safe_float(entry[1])
                if val is not None and step not in seen:
                    history.append((step, val))
                    seen.add(step)
        return sorted(history, key=lambda x: x[0])


def detect_model_adapter(model: Any) -> str:
    """Identify matching Ecosystem Adapter name for a model object."""
    if HurdleModelAdapter.matches(model):
        return HurdleModelAdapter.name
    if NixtlaWrapperAdapter.matches(model):
        return NixtlaWrapperAdapter.name
    if NeuralForecastAdapter.matches(model):
        return NeuralForecastAdapter.name
    if SklearnMultiOutputAdapter.matches(model):
        return SklearnMultiOutputAdapter.name
    return "GenericAdapter"


def extract_model_hyperparams(model: Any) -> Dict[str, Any]:
    """Dispatch hyperparameter extraction across all 4 supported model ecosystems."""
    if model is None:
        return {}
    if HurdleModelAdapter.matches(model):
        return HurdleModelAdapter.extract_hyperparams(model)
    if NixtlaWrapperAdapter.matches(model):
        return NixtlaWrapperAdapter.extract_hyperparams(model)
    if NeuralForecastAdapter.matches(model):
        return NeuralForecastAdapter.extract_hyperparams(model)
    if SklearnMultiOutputAdapter.matches(model):
        return SklearnMultiOutputAdapter.extract_hyperparams(model)
    if hasattr(model, "get_params"):
        return {"adapter": "GenericAdapter", **_sanitize_param_dict(model.get_params(deep=False))}
    return {"adapter": "GenericAdapter", "model_type": type(model).__name__}


def extract_feature_importance_payload(
    model: Any,
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 30,
) -> Optional[Dict[str, Any]]:
    """Extract JSON-serializable feature importance diagnostics where supported."""
    if model is None:
        return None
    if HurdleModelAdapter.matches(model):
        return HurdleModelAdapter.extract_feature_importance(model, feature_names=feature_names, top_k=top_k)
    if SklearnMultiOutputAdapter.matches(model):
        return SklearnMultiOutputAdapter.extract_feature_importance(model, feature_names=feature_names, top_k=top_k)
    return None


def extract_hurdle_threshold_grid(
    model: Any,
    threshold_grid: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Extract Hurdle Strategy B Validation threshold grid payload if present."""
    if threshold_grid is not None:
        if isinstance(threshold_grid, dict):
            return threshold_grid
        return {"grid": threshold_grid}
    if HurdleModelAdapter.matches(model):
        summary = getattr(model, "validation_selection_summary_", None)
        if isinstance(summary, dict):
            return summary
    return None


def resolve_run_name(
    scenario_id: str,
    model_name: str,
    config_label: Optional[str] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve canonical MLflow runName matching DagsHub benchmark conventions."""
    clean_model = str(model_name).strip()
    if clean_model.startswith(f"{scenario_id}_"):
        base_name = clean_model
    else:
        base_name = f"{scenario_id}_{clean_model}"

    resolved_label = config_label
    if not resolved_label and scenario_id in ("SC-28", "SC-29", "SC-30", "SC-31") and "lightgbm" in clean_model.lower():
        if hyperparams and "max_depth" in hyperparams:
            md = hyperparams["max_depth"]
            resolved_label = "ConfigB" if str(md) == "5" else "ConfigA"

    if resolved_label:
        label_norm = str(resolved_label).strip().replace("_Validation_Canonical", "").replace("_", "")
        if not base_name.endswith(f"_{label_norm}"):
            return f"{base_name}_{label_norm}"
    return base_name


# ==============================================================================
# Unified Cloud MLflow Training Tracker
# ==============================================================================
class MLflowTrainingTracker:
    """Idempotent batched MLflow tracker targeting hosted DagsHub MLflow server.

    Designed for direct integration into benchmark runners (Phase 1..5 & Stage 3),
    packaging workflows, and ecosystem compatibility gates.
    """

    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        partition: str = "validation",
        source: str = "live_benchmark",
        enabled: bool = True,
        env_path: Optional[Path] = None,
        client: Optional[Any] = None,
        experiment_id: Optional[str] = None,
        auto_init: bool = True,
        strict: bool = False,
    ) -> None:
        self.experiment_name = experiment_name
        self.partition = self._normalize_partition(partition)
        self.source = str(source)
        self.enabled = bool(enabled) and MLFLOW_AVAILABLE
        self.strict = bool(strict)
        self.client: Optional[Any] = client
        self.experiment_id: Optional[str] = experiment_id
        self.degraded_reason: Optional[str] = None
        self._existing_runs_cache: Dict[str, Any] = {}
        self._logged_params_by_run: Dict[str, set[str]] = {}

        if MLFLOW_AVAILABLE and mlflow is not None:
            try:
                mlflow.autolog(disable=True)
            except Exception as exc:  # pragma: no cover
                logger.debug(f"mlflow.autolog(disable=True) note: {exc}")

        if self.enabled and self.client is None and auto_init:
            self._connect(env_path=env_path)
        elif self.enabled and self.client is not None and self.experiment_id is None:
            try:
                self.experiment_id = get_or_create_experiment(self.client, self.experiment_name)
                self._refresh_existing_runs()
            except Exception as exc:
                self._handle_connection_error(exc)
        elif self.enabled and self.client is not None and self.experiment_id is not None:
            self._refresh_existing_runs()

    @staticmethod
    def _normalize_partition(partition: str) -> str:
        part = str(partition).strip().lower()
        if part in ("val", "validation"):
            return "validation"
        if part in ("test", "stage3_test"):
            return "test"
        return part

    @staticmethod
    def _partition_prefix(partition: str) -> str:
        part = MLflowTrainingTracker._normalize_partition(partition)
        if part == "validation":
            return "val"
        return part

    def _handle_connection_error(self, exc: Exception) -> None:
        if self.strict:
            raise exc
        self.degraded_reason = f"{type(exc).__name__}: {exc}"
        self.enabled = False
        self.client = None
        logger.warning(
            f"[MLflowTrainingTracker] Graceful offline fallback activated ({self.degraded_reason}). "
            "Training will continue without cloud tracking."
        )

    def _connect(self, env_path: Optional[Path] = None) -> None:
        try:
            self.client = init_dagshub_mlflow(
                env_path=env_path,
                experiment_name=self.experiment_name,
            )
            self.experiment_id = get_or_create_experiment(self.client, self.experiment_name)
            self._refresh_existing_runs()
            logger.info(
                f"[MLflowTrainingTracker] Connected to DagsHub Experiment '{self.experiment_name}' "
                f"(ID={self.experiment_id}, partition={self.partition})."
            )
        except Exception as exc:
            self._handle_connection_error(exc)

    def _refresh_existing_runs(self) -> None:
        if self.client is None or self.experiment_id is None:
            return
        runs = retry_with_backoff(
            lambda: self.client.search_runs(experiment_ids=[self.experiment_id], max_results=1000),
            operation_name="tracker.search_runs",
        )
        cache: Dict[str, Any] = {}
        for r in runs or []:
            r_name = getattr(getattr(r, "data", None), "tags", {}).get("mlflow.runName")
            if r_name:
                cache[r_name] = r
        self._existing_runs_cache = cache

    def _get_or_create_run(self, run_name: str) -> Tuple[str, set[str]]:
        assert self.client is not None and self.experiment_id is not None
        existing = self._existing_runs_cache.get(run_name)
        if existing is not None:
            run_id = existing.info.run_id
            existing_params = set(getattr(getattr(existing, "data", None), "params", {}).keys())
            existing_params |= self._logged_params_by_run.get(run_id, set())
            return run_id, existing_params

        created = retry_with_backoff(
            lambda: self.client.create_run(experiment_id=self.experiment_id, run_name=run_name),
            operation_name=f"create_run({run_name})",
        )
        self._existing_runs_cache[run_name] = created
        run_id = created.info.run_id
        self._logged_params_by_run.setdefault(run_id, set())
        return run_id, set()

    def _log_batch_chunked(
        self,
        run_id: str,
        run_name: str,
        metrics: List[Metric],
        params: List[Param],
        tags: List[RunTag],
    ) -> None:
        """Flush metrics, params, and tags safely in bounded REST chunks."""
        assert self.client is not None
        m_chunks = [
            metrics[i : i + MLFLOW_BATCH_METRIC_LIMIT]
            for i in range(0, len(metrics), MLFLOW_BATCH_METRIC_LIMIT)
        ] or [[]]
        p_chunks = [
            params[i : i + MLFLOW_BATCH_PARAM_LIMIT]
            for i in range(0, len(params), MLFLOW_BATCH_PARAM_LIMIT)
        ] or [[]]
        t_chunks = [
            tags[i : i + MLFLOW_BATCH_TAG_LIMIT]
            for i in range(0, len(tags), MLFLOW_BATCH_TAG_LIMIT)
        ] or [[]]

        n_calls = max(len(m_chunks), len(p_chunks), len(t_chunks))
        for idx in range(n_calls):
            m_part = m_chunks[idx] if idx < len(m_chunks) else []
            p_part = p_chunks[idx] if idx < len(p_chunks) else []
            t_part = t_chunks[idx] if idx < len(t_chunks) else []
            if not m_part and not p_part and not t_part:
                continue
            retry_with_backoff(
                lambda mp=m_part, pp=p_part, tp=t_part: self.client.log_batch(
                    run_id=run_id,
                    metrics=mp,
                    params=pp,
                    tags=tp,
                ),
                operation_name=f"log_batch({run_name}#chunk{idx + 1}/{n_calls})",
            )

    @staticmethod
    def _extract_horizon_metric_value(detailed: Dict[str, float], key_short: str, h: int) -> Optional[float]:
        candidates = {
            "MAE": (f"mae_h{h}", f"MAE@h{h}"),
            "MSE": (f"mse_h{h}", f"MSE@h{h}"),
            "RMSE": (f"rmse_h{h}", f"RMSE@h{h}"),
            "MASE": (f"mase_lag1_h{h}", f"mase_h{h}", f"MASE@h{h}"),
        }[key_short]
        for ck in candidates:
            if ck in detailed:
                v = _safe_float(detailed[ck])
                if v is not None:
                    return v
        return None

    def log_training_run(
        self,
        scenario_id: str,
        model_name: str,
        model: Any = None,
        detailed_metrics: Optional[Dict[str, float]] = None,
        training_time: Optional[float] = None,
        inference_time: Optional[float] = None,
        feature_set: Optional[str] = None,
        feature_names: Optional[Sequence[str]] = None,
        feature_count: Optional[int] = None,
        model_family: Optional[str] = None,
        strategy: Optional[str] = None,
        parameter_count: Optional[int] = None,
        selected_hurdle_threshold: Optional[float] = None,
        threshold_grid: Optional[Any] = None,
        loss_callback: Optional[InMemoryStepLossCallback] = None,
        loss_history: Optional[Sequence[Tuple[int, float]]] = None,
        config_label: Optional[str] = None,
        partition: Optional[str] = None,
        extra_tags: Optional[Dict[str, str]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
        extra_artifacts: Optional[Dict[str, Any]] = None,
        governance_role: Optional[str] = None,
        status: str = "SUCCESS",
        include_horizon_breakdown: bool = True,
    ) -> Optional[str]:
        """Log complete scenario run telemetry (tags, params, metrics, JSON diagnostics)."""
        if not self.enabled or self.client is None or self.experiment_id is None:
            return None

        active_part = self._normalize_partition(partition or self.partition)
        prefix = self._partition_prefix(active_part)
        now_ms = int(time.time() * 1000)

        try:
            extracted_params = extract_model_hyperparams(model) if model is not None else {}
            run_name = resolve_run_name(
                scenario_id=scenario_id,
                model_name=model_name,
                config_label=config_label,
                hyperparams=extracted_params,
            )
            run_id, existing_param_keys = self._get_or_create_run(run_name)

            canonical_family = model_family or assign_canonical_family(scenario_id)
            f_set = feature_set or "Target-Only"
            adapter_name = extracted_params.get("adapter") or detect_model_adapter(model)

            # 1. Tags
            tags_dict: Dict[str, str] = {
                "mlflow.runName": run_name,
                "scenario_id": str(scenario_id),
                "model_family": str(canonical_family),
                "feature_set": str(f_set),
                "strategy": str(strategy or ""),
                "source": self.source,
                "partition": active_part,
                "execution_status": str(status),
                "ecosystem_adapter": str(adapter_name),
            }
            if scenario_id in ("SC-01", "SC-02"):
                tags_dict["is_naive_baseline"] = "true"
            if governance_role:
                tags_dict["governance_role"] = str(governance_role)

            if scenario_id in ("SC-28", "SC-29", "SC-30", "SC-31") and "lightgbm" in str(model_name).lower():
                if run_name.endswith("_ConfigB"):
                    tags_dict["config_version"] = "Config_B_Benchmark_Original"
                elif run_name.endswith("_ConfigA"):
                    tags_dict["config_version"] = "Config_A_Validation_Canonical"

            if scenario_id in ("SC-36", "SC-37", "SC-38", "SC-39", "SC-40") or adapter_name == NeuralForecastAdapter.name:
                tags_dict["val_test_configuration_note"] = (
                    "LECH, co chu dich (PROP-03): Val 1000 steps vs Test 10000 steps"
                )

            if extra_tags:
                for tk, tv in extra_tags.items():
                    if tv is not None:
                        tags_dict[str(tk)] = str(tv)

            # 2. Params (skip existing immutable keys on re-runs)
            merged_params: Dict[str, Any] = {
                "model_name": model_name,
                "family": canonical_family,
                "feature_set": f_set,
                "strategy": str(strategy or ""),
            }
            merged_params.update(extracted_params)
            if feature_count is not None:
                merged_params["feature_count"] = int(feature_count)
            elif feature_names is not None:
                merged_params["feature_count"] = len(feature_names)
            if parameter_count is not None:
                merged_params["parameter_count"] = int(parameter_count)
            elif "parameter_count" in extracted_params:
                parameter_count = int(extracted_params["parameter_count"])
            if extra_params:
                merged_params.update(_sanitize_param_dict(extra_params))

            params_list: List[Param] = [
                Param(key=str(k), value=_to_param_str(v))
                for k, v in merged_params.items()
                if str(k) not in existing_param_keys and v is not None
            ]
            tags_list: List[RunTag] = [
                RunTag(key=str(k), value=str(v)[:500]) for k, v in tags_dict.items()
            ]

            # 3. Metrics (strictly canonical + macro + horizon + training telemetry; zero purged metrics)
            metrics_list: List[Metric] = []

            def _add_m(m_key: str, m_val: Any, step: int = 0) -> None:
                if _is_purged_metric_key(m_key):
                    logger.warning(f"Blocked purged metric key '{m_key}' per Evaluation Standards Rule 3.")
                    return
                num = _safe_float(m_val)
                if num is not None:
                    metrics_list.append(Metric(key=m_key, value=num, timestamp=now_ms, step=step))

            if detailed_metrics:
                for m_name in ("MAE", "MSE", "RMSE", "MASE"):
                    val = detailed_metrics.get(m_name, detailed_metrics.get(f"{m_name.lower()}_pooled"))
                    _add_m(f"{prefix}_{m_name}", val)

                for macro_key in ("mae_macro", "mse_macro", "rmse_macro", "mase_lag1_macro"):
                    if macro_key in detailed_metrics:
                        _add_m(f"{prefix}_{macro_key}", detailed_metrics[macro_key])

                if include_horizon_breakdown:
                    for h in range(1, 8):
                        for m_short in ("MAE", "MSE", "RMSE", "MASE"):
                            h_val = self._extract_horizon_metric_value(detailed_metrics, m_short, h)
                            _add_m(f"{prefix}_horizon_{m_short}", h_val, step=h)

            if training_time is not None:
                _add_m(f"{prefix}_training_time", training_time)
                _add_m("training_time", training_time)
            if inference_time is not None:
                _add_m(f"{prefix}_inference_time", inference_time)
                _add_m("inference_time", inference_time)
            if parameter_count is not None:
                _add_m("parameter_count", parameter_count)
            if selected_hurdle_threshold is not None:
                _add_m("selected_hurdle_threshold", selected_hurdle_threshold)

            if extra_metrics:
                for ek, ev in extra_metrics.items():
                    _add_m(str(ek), ev)

            # Neural step loss trajectory
            steps_seq: List[Tuple[int, float]] = []
            if loss_callback is not None:
                steps_seq = loss_callback.get_step_metrics()
            if not steps_seq and loss_history:
                steps_seq = [(int(s), float(v)) for s, v in loss_history if _safe_float(v) is not None]
            if not steps_seq and model is not None and NeuralForecastAdapter.matches(model):
                steps_seq = NeuralForecastAdapter.extract_loss_history(model)

            for step_idx, loss_val in steps_seq:
                _add_m(f"{prefix}_train_loss", loss_val, step=int(step_idx))

            # Chunked flush to DagsHub MLflow server
            self._log_batch_chunked(
                run_id=run_id,
                run_name=run_name,
                metrics=metrics_list,
                params=params_list,
                tags=tags_list,
            )
            self._logged_params_by_run.setdefault(run_id, set()).update(
                p.key for p in params_list
            )

            # 4. JSON Diagnostic Artifacts (feature importance, hurdle threshold grid, extras)
            if model is not None:
                fi_payload = extract_feature_importance_payload(model, feature_names=feature_names)
                if fi_payload is not None:
                    fi_payload["scenario_id"] = scenario_id
                    fi_payload["model_name"] = model_name
                    fi_payload["partition"] = active_part
                    retry_with_backoff(
                        lambda: self.client.log_dict(
                            run_id, fi_payload, "diagnostics/feature_importance.json"
                        ),
                        operation_name=f"log_dict(feature_importance:{run_name})",
                    )

            grid_payload = extract_hurdle_threshold_grid(model, threshold_grid=threshold_grid)
            if grid_payload is not None:
                retry_with_backoff(
                    lambda: self.client.log_dict(
                        run_id, grid_payload, "diagnostics/hurdle_threshold_grid.json"
                    ),
                    operation_name=f"log_dict(hurdle_grid:{run_name})",
                )

            if extra_artifacts:
                for art_path, art_payload in extra_artifacts.items():
                    retry_with_backoff(
                        lambda p=art_path, d=art_payload: self.client.log_dict(run_id, d, p),
                        operation_name=f"log_dict({art_path}:{run_name})",
                    )

            retry_with_backoff(
                lambda: self.client.set_terminated(run_id, status="FINISHED"),
                operation_name=f"set_terminated({run_name})",
            )
            return run_id
        except Exception as exc:
            if self.strict:
                raise
            logger.warning(
                f"[MLflowTrainingTracker] Failed to log run {scenario_id}/{model_name} "
                f"(non-fatal): {type(exc).__name__}: {exc}"
            )
            return None


__all__ = [
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
