"""
DS108 Inference Module
======================
Exposes standalone predictor and lazy-loaded MLflow Pyfunc wrapper.
"""

from typing import TYPE_CHECKING, Any

from src.inference.predictor import StandaloneRainfallPredictor
from src.inference.validation import ServingValidationError, validate_serving_input

if TYPE_CHECKING:
    from src.inference.pyfunc import RainfallForecastingPyfuncWrapper

__all__ = [
    "StandaloneRainfallPredictor",
    "RainfallForecastingPyfuncWrapper",
    "ServingValidationError",
    "validate_serving_input",
]


def __getattr__(name: str) -> Any:
    if name == "RainfallForecastingPyfuncWrapper":
        from src.inference.pyfunc import RainfallForecastingPyfuncWrapper

        return RainfallForecastingPyfuncWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

