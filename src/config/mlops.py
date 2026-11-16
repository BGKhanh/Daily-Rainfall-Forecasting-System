"""
MLOps Configuration & DagsHub MLflow Governance Module (DS108)
===============================================================
Centralizes DagsHub Hosted MLflow configuration, environment loading,
retry-resilient client initialization, and Dual-Resolution Model Registry
governance (Aliases + Tags + Stages).
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, TypeVar, List

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_EXPERIMENT_NAME = "Rainfall_Forecasting_HCMC_Benchmark"
DEFAULT_REGISTERED_MODEL_NAME = "RainfallForecastHCMC"

ALIAS_TO_STAGE: Dict[str, str] = {
    "serving": "Production",
    "shadow_val_champ": "Staging",
    "shadow_test_contender": "Staging",
}

T = TypeVar("T")


def load_dotenv_file(env_path: Optional[Path] = None, override: bool = False) -> Dict[str, str]:
    """Load key=value pairs from a .env file into os.environ."""
    target_path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    loaded: Dict[str, str] = {}
    if not target_path.exists():
        return loaded

    try:
        from dotenv import dotenv_values  # type: ignore

        raw_values = dotenv_values(target_path)
        for key, val in raw_values.items():
            if key and val is not None:
                loaded[key] = val
                if override or key not in os.environ:
                    os.environ[key] = val
        return loaded
    except ImportError:
        pass

    with open(target_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]
            loaded[key] = val
            if override or key not in os.environ:
                os.environ[key] = val
    return loaded


def validate_mlflow_env() -> Dict[str, str]:
    """Ensure required DagsHub MLflow environment variables are configured."""
    required_vars = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    ]
    missing = [
        var
        for var in required_vars
        if not os.environ.get(var) or "<YOUR_DAGSHUB_ACCESS_TOKEN>" in os.environ.get(var, "")
    ]
    if missing:
        raise EnvironmentError(
            "Missing required MLflow/DagsHub environment variables: "
            f"{', '.join(missing)}. Please configure them in {DEFAULT_ENV_PATH} "
            "(see .env.example)."
        )

    # Apply resilient HTTP timeout and retry defaults
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "45")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "2")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

    return {var: os.environ[var] for var in required_vars}


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 5,
    base_delay: float = 1.5,
    operation_name: str = "MLflow API call",
) -> T:
    """Execute an MLflow API call with exponential backoff on transient/429 errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_transient = any(
                token in msg
                for token in ("429", "too many requests", "timeout", "timed out", "connection", "502", "503", "504")
            )
            if attempt >= max_retries - 1 or not is_transient:
                raise
            sleep_sec = base_delay * (2 ** attempt)
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                operation_name,
                attempt + 1,
                max_retries,
                exc,
                sleep_sec,
            )
            time.sleep(sleep_sec)
    assert last_exc is not None
    raise last_exc


def init_dagshub_mlflow(
    env_path: Optional[Path] = None,
    experiment_name: Optional[str] = None,
) -> MlflowClient:
    """Initialize MLflow tracking & registry URIs for DagsHub and return MlflowClient."""
    load_dotenv_file(env_path=env_path)
    creds = validate_mlflow_env()
    tracking_uri = creds["MLFLOW_TRACKING_URI"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def get_or_create_experiment(
    client: MlflowClient,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> str:
    """Retrieve an existing MLflow experiment or create it with canonical project tags."""
    exp = retry_with_backoff(
        lambda: client.get_experiment_by_name(experiment_name),
        operation_name=f"get_experiment_by_name({experiment_name})",
    )
    if exp is not None:
        return exp.experiment_id

    return retry_with_backoff(
        lambda: client.create_experiment(
            experiment_name,
            tags={
                "project": "DS108_Rainfall_Forecasting_HCMC",
                "canonical_cutoff": "2018-04-30",
                "dataset": "HCMC_Weather_Daily_2000_2026",
            },
        ),
        operation_name=f"create_experiment({experiment_name})",
    )


class ModelRegistryGovernance:
    """
    Dual-Resolution Governance Manager (Aliases + Tags + Stages)
    ensuring compatibility across MLflow 2.x/3.x and DagsHub Hosted MLflow.
    """

    @staticmethod
    def ensure_registered_model(client: MlflowClient, model_name: str) -> None:
        try:
            retry_with_backoff(
                lambda: client.get_registered_model(model_name),
                operation_name=f"get_registered_model({model_name})",
            )
        except MlflowException:
            retry_with_backoff(
                lambda: client.create_registered_model(
                    name=model_name,
                    description=(
                        "DS108 HCMC 7-Day Daily Rainfall Forecasting Champion & Shadow Models "
                        "(SC-35 Serving, SC-30 Shadow Val Champion, SC-31 Shadow Test Contender)."
                    ),
                ),
                operation_name=f"create_registered_model({model_name})",
            )

    @staticmethod
    def register_model_version_governance(
        client: MlflowClient,
        model_name: str,
        version: str,
        alias: str,
        governance_role: str,
        scenario_id: str,
        feature_level: str,
        sha256_model: str = "",
        sha256_feature_builder: str = "",
        description: Optional[str] = None,
    ) -> None:
        """Apply Dual-Resolution governance metadata (Tags, Stage, and Alias) to a model version."""
        version_str = str(version)
        tags = {
            "governance_alias": alias,
            "governance_role": governance_role,
            "scenario_id": scenario_id,
            "feature_level": feature_level,
        }
        if sha256_model:
            tags["sha256_model"] = sha256_model
        if sha256_feature_builder:
            tags["sha256_feature_builder"] = sha256_feature_builder

        for k, v in tags.items():
            retry_with_backoff(
                lambda key=k, val=v: client.set_model_version_tag(model_name, version_str, key, str(val)),
                operation_name=f"set_model_version_tag({model_name} v{version_str} {k})",
            )

        if description:
            retry_with_backoff(
                lambda: client.update_model_version(model_name, version_str, description=description),
                operation_name=f"update_model_version({model_name} v{version_str})",
            )

        stage = ALIAS_TO_STAGE.get(alias)
        if stage:
            try:
                retry_with_backoff(
                    lambda: client.transition_model_version_stage(
                        name=model_name,
                        version=version_str,
                        stage=stage,
                        archive_existing_versions=False,
                    ),
                    operation_name=f"transition_model_version_stage({model_name} v{version_str} -> {stage})",
                )
            except Exception as exc:
                logger.warning(
                    "Could not transition stage for %s v%s to %s: %s",
                    model_name,
                    version_str,
                    stage,
                    exc,
                )

        try:
            retry_with_backoff(
                lambda: client.set_registered_model_alias(model_name, alias, version_str),
                operation_name=f"set_registered_model_alias({model_name}@{alias} -> v{version_str})",
            )
        except MlflowException as exc:
            logger.warning(
                "Alias endpoint not supported or failed on server for %s@%s (falling back to tags/stages): %s",
                model_name,
                alias,
                exc,
            )

    @staticmethod
    def resolve_model_version(
        client: MlflowClient,
        model_name: str,
        alias: str,
    ):
        """Resolve a ModelVersion object via MLflow alias first, falling back to governance_alias tag."""
        try:
            mv = retry_with_backoff(
                lambda: client.get_model_version_by_alias(model_name, alias),
                operation_name=f"get_model_version_by_alias({model_name}@{alias})",
            )
            if mv is not None:
                return mv, True
        except Exception:
            pass

        versions = retry_with_backoff(
            lambda: client.search_model_versions(f"name = '{model_name}'"),
            operation_name=f"search_model_versions({model_name})",
        )
        matching = [
            mv for mv in versions
            if getattr(mv, "tags", {}).get("governance_alias") == alias
        ]
        if not matching:
            # Secondary fallback by stage if alias tag absent
            expected_stage = ALIAS_TO_STAGE.get(alias)
            if alias == "serving" and expected_stage:
                matching = [mv for mv in versions if getattr(mv, "current_stage", None) == expected_stage]

        if not matching:
            raise LookupError(
                f"Could not resolve model '{model_name}' for alias '{alias}' "
                "via either MLflow Aliases or 'governance_alias' tags."
            )

        matching.sort(key=lambda m: int(m.version), reverse=True)
        return matching[0], False

    @classmethod
    def resolve_model_uri(
        cls,
        client: MlflowClient,
        model_name: str,
        alias: str,
    ) -> str:
        """
        Return the canonical loadable MLflow URI for an alias.
        Prefers `models:/{model_name}@{alias}` if supported; otherwise falls back to
        `models:/{model_name}/{version}` resolved via `governance_alias` tag.
        """
        mv, used_alias_endpoint = cls.resolve_model_version(client, model_name, alias)
        if used_alias_endpoint:
            return f"models:/{model_name}@{alias}"
        return f"models:/{model_name}/{mv.version}"

