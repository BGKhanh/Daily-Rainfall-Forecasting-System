"""
5-Layer Configuration Loader, Validator & Startup Contract Guard (DS108)
=======================================================================
Loads and validates structured configuration across:
- Layer 1: Environment secrets (`.env` / OS env, never serialized into config hash)
- Layer 2: Operational config (`config/runtime_config.toml` `[operational]`)
- Layer 3: Application & business config (`config/runtime_config.toml` `[application]`)
- Layer 4: Model contract (`artifacts/models/metadata.json["model_contract"]` + `config/variable_catalog.json`)
- Layer 5: Frozen EDA & modeling config (`config/training_eda_freeze_v1.toml`)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tomllib
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_RUNTIME_TOML = PROJECT_ROOT / "config" / "runtime_config.toml"
DEFAULT_FREEZE_TOML = PROJECT_ROOT / "config" / "training_eda_freeze_v1.toml"
DEFAULT_VARIABLE_CATALOG_JSON = PROJECT_ROOT / "config" / "variable_catalog.json"
DEFAULT_METADATA_JSON = PROJECT_ROOT / "artifacts" / "models" / "metadata.json"

StoreBackend = Literal["local", "turso_replica", "turso_remote"]


def _load_dotenv_non_overriding(env_path: Path = DEFAULT_ENV_PATH) -> None:
    """Populate missing keys from `.env` without overriding explicit process environment variables."""
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val


class OperationalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    store_backend: StoreBackend = "local"
    turso_sync_interval_sec: int = Field(default=300, ge=30, le=86400)
    turso_sync_jitter_sec: int = Field(default=30, ge=0, le=300)
    nasa_api_base_url: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
    nasa_api_timeout_sec: int = Field(default=120, ge=5, le=600)
    nasa_api_max_params: int = Field(default=20, ge=1, le=20)
    scan_window_days: int = Field(default=150, ge=35, le=1000)
    short_scan_window_days: int = Field(default=14, ge=1, le=365)
    deep_scan_window_days: int = Field(default=150, ge=35, le=1000)
    provisional_window_days: int = Field(default=120, ge=1, le=365)
    sqlite_cache_size_kb: int = Field(default=16000, ge=1000, le=512000)
    cors_allowed_origins: List[str] = Field(default_factory=list)


class ApplicationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_query_days: int = Field(default=90, ge=1, le=366)
    obs_max_days_daily: int = Field(default=366, ge=1, le=1000)
    obs_max_days_weekly: int = Field(default=2555, ge=7, le=10000)
    obs_max_days_monthly: int = Field(default=10000, ge=30, le=50000)
    rain_day_threshold_mm: float = Field(default=1.0, ge=0.0, le=100.0)
    prior_observations_preview_days: int = Field(default=14, ge=0, le=90)
    map_max_selected_visible_rows: int = Field(default=5, ge=1, le=65)
    map_zoom_min: float = Field(default=1.0, ge=0.5, le=2.0)
    map_zoom_max: float = Field(default=6.0, ge=2.0, le=20.0)
    turso_plan_name: str = "Free"
    turso_monthly_read_quota: int = Field(default=500000000, ge=1)
    turso_monthly_write_quota: int = Field(default=10000000, ge=1)
    turso_monthly_sync_bytes_quota: int = Field(default=3221225472, ge=1)
    turso_storage_bytes_quota: int = Field(default=5368709120, ge=1)
    turso_quota_warn_ratio: float = Field(default=0.70, gt=0.0, le=1.0)
    short_scan_window_days: int = Field(default=14, ge=1, le=365)
    deep_scan_window_days: int = Field(default=150, ge=35, le=1000)


class ModelContractConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizon_days: int = Field(..., ge=1, le=30)
    required_input_variables: List[str]
    min_contiguous_history_days: int = Field(..., ge=1)
    recommended_fetch_window_days: int = Field(..., ge=1)
    feature_warmup_lookback_days: int = Field(..., ge=0)
    time_standard: str
    nasa_community: str
    canonical_location_id: str
    canonical_latitude: float
    canonical_longitude: float
    completeness_policy: str
    radiation_imputation_policy: str = "P-c_DOY_MEDIAN_11_SOLAR_V1"
    leap_day_doy_policy: str = "EXPLICIT_366_CIRCULAR_MEDIAN_R7"
    solar_doy_table_sha256: str = ""

    @field_validator("required_input_variables")
    @classmethod
    def _validate_vars_unique(cls, val: List[str]) -> List[str]:
        if len(val) != len(set(val)):
            raise ValueError("required_input_variables must not contain duplicates.")
        return val


class FrozenEdaModelingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dev_train_end: str
    validation_end: str
    test_end: str
    wet_season_months: List[int]
    rolling_windows: List[int]
    lag_days: List[int]
    rain_threshold_tau_mm: float
    tweedie_variance_power: float
    canonical_mase_scale_dev_train: float


class VariableCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    label_vi: str
    label_en: str
    unit: str
    group: str
    source_finalized: str
    source_provisional: str
    grid_id: str
    grid_resolution_deg: str
    required_by_serving_model: bool
    is_queryable_in_ui: bool
    verified_by_probe_date: str


class VariableCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    verified_by_probe_date: str
    probe_window: str
    total_variables: int
    variables: List[VariableCatalogEntry]


class RuntimeEnvConfig(BaseModel):
    """Layer 1 non-secret process environment settings (excluded from config_sha256)."""

    model_config = ConfigDict(extra="forbid")

    app_env: Literal["development", "production", "test"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    enable_api_docs: bool = True
    serve_static_spa: bool = False
    turso_replica_path: str = "data/vintage/turso_replica.db"
    vintage_db_path: str = ""
    serving_model_source: str = "local"
    weather_db_backend: str = "local"

    @field_validator("app_env", mode="before")
    @classmethod
    def _norm_app_env(cls, val: Any) -> str:
        s = str(val or "development").strip().lower()
        return s or "development"

    @field_validator("log_level", mode="before")
    @classmethod
    def _norm_log_level(cls, val: Any) -> str:
        s = str(val or "INFO").strip().upper()
        return s or "INFO"

    @classmethod
    def from_environ(cls, env: Optional[Mapping[str, str]] = None) -> "RuntimeEnvConfig":
        src = os.environ if env is None else env
        app_env = (src.get("APP_ENV") or "development").strip().lower() or "development"
        log_level = (src.get("LOG_LEVEL") or "INFO").strip().upper() or "INFO"
        raw_docs = (src.get("ENABLE_API_DOCS") or "").strip()
        enable_api_docs = (
            _parse_bool_env(raw_docs, "ENABLE_API_DOCS") if raw_docs else (app_env != "production")
        )
        raw_spa = (src.get("SERVE_STATIC_SPA") or "").strip()
        serve_static_spa = _parse_bool_env(raw_spa, "SERVE_STATIC_SPA") if raw_spa else False
        replica_path = (src.get("TURSO_REPLICA_PATH") or "").strip() or "data/vintage/turso_replica.db"
        vintage_db_path = (src.get("VINTAGE_DB_PATH") or "").strip()
        model_source = (src.get("SERVING_MODEL_SOURCE") or "local").strip().lower() or "local"
        weather_backend = (src.get("WEATHER_DB_BACKEND") or "local").strip().lower() or "local"
        return cls(
            app_env=app_env,  # type: ignore[arg-type]
            log_level=log_level,  # type: ignore[arg-type]
            enable_api_docs=enable_api_docs,
            serve_static_spa=serve_static_spa,
            turso_replica_path=replica_path,
            vintage_db_path=vintage_db_path,
            serving_model_source=model_source,
            weather_db_backend=weather_backend,
        )


def _parse_bool_env(raw: str, var_name: str) -> bool:
    low = raw.strip().lower()
    if low in ("true", "1", "yes"):
        return True
    if low in ("false", "0", "no"):
        return False
    raise ValueError(f"Invalid boolean value for {var_name}={raw!r}; expected true/false or 1/0.")


_PLACEHOLDER_MARKERS = (
    "your-db-your-org",
    "<your_",
    "changeme",
    "replace_me",
    "example.com",
    "your-domain",
)


def _is_placeholder_or_empty(val: Optional[str]) -> bool:
    s = (val or "").strip()
    if not s:
        return True
    low = s.lower()
    return any(marker in low for marker in _PLACEHOLDER_MARKERS)


def _normalize_cors_origins(origins: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in origins:
        cleaned = str(raw or "").strip().rstrip("/")
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _is_localhost_origin(origin: str) -> bool:
    low = origin.lower()
    parsed = urlparse(origin)
    host = (parsed.hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or "localhost" in low or "127.0.0.1" in low


def resolve_turso_replica_path(
    explicit_path: Optional[Union[str, Path]] = None,
    *,
    runtime_env: Optional[RuntimeEnvConfig] = None,
) -> Path:
    """Resolve `TURSO_REPLICA_PATH` keeping absolute paths intact and resolving relative paths under PROJECT_ROOT."""
    if explicit_path is not None and str(explicit_path).strip():
        cand = Path(explicit_path)
    else:
        cfg = runtime_env if runtime_env is not None else RuntimeEnvConfig.from_environ()
        cand = Path(cfg.turso_replica_path)
    return cand if cand.is_absolute() else (PROJECT_ROOT / cand).resolve()


class SecretsEnvView(BaseModel):
    """Non-serialized runtime env view (Layer 1 — excluded from config_sha256)."""

    model_config = ConfigDict(extra="forbid")

    turso_api_token: str = ""
    turso_db_url: str = ""
    turso_db_token: str = ""
    turso_readonly_auth_token: str = ""

    @classmethod
    def from_environ(cls) -> "SecretsEnvView":
        api_tok = os.environ.get("TURSO_API_TOKEN", "").strip()
        db_url = (os.environ.get("TURSO_DB_URL") or os.environ.get("TURSO_DATABASE_URL") or "").strip()
        db_tok = (os.environ.get("TURSO_DB_TOKEN") or os.environ.get("TURSO_AUTH_TOKEN") or "").strip()
        ro_tok = os.environ.get("TURSO_READONLY_AUTH_TOKEN", "").strip()
        return cls(
            turso_api_token=api_tok,
            turso_db_url=db_url,
            turso_db_token=db_tok,
            turso_readonly_auth_token=ro_tok,
        )


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operational: OperationalConfig
    application: ApplicationConfig
    model_contract: ModelContractConfig
    frozen_eda_modeling: FrozenEdaModelingConfig
    variable_catalog: VariableCatalog
    runtime_env: RuntimeEnvConfig = Field(default_factory=RuntimeEnvConfig)
    config_sha256: str = ""
    result_config_sha256: str = ""
    operational_config_sha256: str = ""

    def secrets(self) -> SecretsEnvView:
        return SecretsEnvView.from_environ()


def validate_runtime_env(settings: RuntimeSettings) -> None:
    """Fail-fast validation for production and Turso runtime env configuration."""
    app_env = settings.runtime_env.app_env
    store_backend = settings.operational.store_backend
    needs_turso = store_backend in ("turso_replica", "turso_remote") or app_env == "production"

    if needs_turso:
        sec = settings.secrets()
        db_url = sec.turso_db_url
        if _is_placeholder_or_empty(db_url) or not db_url.startswith(("libsql://", "https://")):
            raise ValueError(
                "Invalid or missing TURSO_DB_URL: expected a non-placeholder libsql:// or https:// URL."
            )
        if store_backend == "turso_remote":
            raw_tok = sec.turso_db_token
            tok_name = "TURSO_DB_TOKEN / TURSO_AUTH_TOKEN"
        else:
            raw_tok = sec.turso_readonly_auth_token or sec.turso_db_token
            tok_name = "TURSO_AUTH_TOKEN / TURSO_DB_TOKEN / TURSO_READONLY_AUTH_TOKEN"
        if _is_placeholder_or_empty(raw_tok):
            raise ValueError(
                f"Invalid or missing Turso auth token ({tok_name}): set a non-placeholder token before startup."
            )

    if app_env == "production":
        origins = settings.operational.cors_allowed_origins
        if not origins:
            raise ValueError(
                "APP_ENV=production requires at least one non-localhost origin in CORS_ALLOWED_ORIGINS."
            )
        for origin in origins:
            parsed = urlparse(origin)
            if (
                "*" in origin
                or _is_placeholder_or_empty(origin)
                or _is_localhost_origin(origin)
                or parsed.scheme not in ("http", "https")
                or not parsed.netloc
                or parsed.path not in ("", "/")
                or parsed.params
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    f"Invalid production CORS origin {origin!r}: must be an explicit non-localhost http(s) origin."
                )


def _canonical_json_sha256(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256_payload(
    operational: OperationalConfig,
    application: ApplicationConfig,
    model_contract: ModelContractConfig,
    frozen_eda_modeling: FrozenEdaModelingConfig,
    variable_catalog: VariableCatalog,
) -> str:
    return _canonical_json_sha256(
        {
            "operational": operational.model_dump(mode="json"),
            "application": application.model_dump(mode="json"),
            "model_contract": model_contract.model_dump(mode="json"),
            "frozen_eda_modeling": frozen_eda_modeling.model_dump(mode="json"),
            "variable_catalog": variable_catalog.model_dump(mode="json"),
        }
    )


def _canonical_result_sha256_payload(
    model_contract: ModelContractConfig,
    frozen_eda_modeling: FrozenEdaModelingConfig,
    variable_catalog: VariableCatalog,
) -> str:
    """Hash Layer 4 (model_contract + variable_catalog) + Layer 5 (frozen_eda_modeling)."""
    return _canonical_json_sha256(
        {
            "model_contract": model_contract.model_dump(mode="json"),
            "frozen_eda_modeling": frozen_eda_modeling.model_dump(mode="json"),
            "variable_catalog": variable_catalog.model_dump(mode="json"),
        }
    )


def _canonical_operational_sha256_payload(
    operational: OperationalConfig,
    application: ApplicationConfig,
) -> str:
    """Hash Layer 2 (operational) + Layer 3 (application)."""
    return _canonical_json_sha256(
        {
            "operational": operational.model_dump(mode="json"),
            "application": application.model_dump(mode="json"),
        }
    )


def load_settings(
    *,
    runtime_toml: Path = DEFAULT_RUNTIME_TOML,
    freeze_toml: Path = DEFAULT_FREEZE_TOML,
    catalog_json: Path = DEFAULT_VARIABLE_CATALOG_JSON,
    metadata_json: Path = DEFAULT_METADATA_JSON,
    env_path: Optional[Path] = DEFAULT_ENV_PATH,
) -> RuntimeSettings:
    """Load and validate all 5 configuration layers without embedding secrets in config_sha256."""
    if env_path is not None:
        _load_dotenv_non_overriding(env_path)

    with runtime_toml.open("rb") as fh:
        runtime_raw = tomllib.load(fh)
    op_dict = dict(runtime_raw.get("operational") or {})
    app_dict = dict(runtime_raw.get("application") or {})

    runtime_env = RuntimeEnvConfig.from_environ()

    env_backend = os.environ.get("STORE_BACKEND", "").strip().lower()
    if env_backend:
        op_dict["store_backend"] = env_backend

    raw_sync_interval = os.environ.get("TURSO_SYNC_INTERVAL_SEC", "").strip()
    if raw_sync_interval:
        try:
            op_dict["turso_sync_interval_sec"] = int(raw_sync_interval)
        except ValueError as exc:
            raise ValueError(
                f"Invalid integer for TURSO_SYNC_INTERVAL_SEC={raw_sync_interval!r}."
            ) from exc

    base_cors = _normalize_cors_origins(op_dict.get("cors_allowed_origins") or [])
    env_cors = _normalize_cors_origins(os.environ.get("CORS_ALLOWED_ORIGINS", "").split(","))
    combined_cors = _normalize_cors_origins(
        (env_cors + base_cors) if runtime_env.app_env == "production" else (base_cors + env_cors)
    )
    if runtime_env.app_env == "production":
        combined_cors = [o for o in combined_cors if not _is_localhost_origin(o)]
    op_dict["cors_allowed_origins"] = combined_cors

    operational = OperationalConfig.model_validate(op_dict)
    application = ApplicationConfig.model_validate(app_dict)

    with freeze_toml.open("rb") as fh:
        freeze_raw = tomllib.load(fh)
    frozen_eda_modeling = FrozenEdaModelingConfig.model_validate(freeze_raw.get("freeze") or {})

    catalog_data = json.loads(catalog_json.read_text(encoding="utf-8"))
    variable_catalog = VariableCatalog.model_validate(catalog_data)

    meta_data = json.loads(metadata_json.read_text(encoding="utf-8"))
    raw_contract = meta_data.get("model_contract")
    if not isinstance(raw_contract, dict):
        raise RuntimeError(f"[MODEL_CONTRACT_MISMATCH] Missing 'model_contract' block in {metadata_json}")
    model_contract = ModelContractConfig.model_validate(raw_contract)

    cfg_sha = _canonical_sha256_payload(
        operational, application, model_contract, frozen_eda_modeling, variable_catalog
    )
    res_sha = _canonical_result_sha256_payload(model_contract, frozen_eda_modeling, variable_catalog)
    op_sha = _canonical_operational_sha256_payload(operational, application)
    return RuntimeSettings(
        operational=operational,
        application=application,
        model_contract=model_contract,
        frozen_eda_modeling=frozen_eda_modeling,
        variable_catalog=variable_catalog,
        runtime_env=runtime_env,
        config_sha256=cfg_sha,
        result_config_sha256=res_sha,
        operational_config_sha256=op_sha,
    )


def compute_config_sha256(
    *,
    runtime_toml: Path = DEFAULT_RUNTIME_TOML,
    freeze_toml: Path = DEFAULT_FREEZE_TOML,
    catalog_json: Path = DEFAULT_VARIABLE_CATALOG_JSON,
    metadata_json: Path = DEFAULT_METADATA_JSON,
) -> str:
    """Compute deterministic SHA-256 across active TOML, catalog, and model_contract layers."""
    return load_settings(
        runtime_toml=runtime_toml,
        freeze_toml=freeze_toml,
        catalog_json=catalog_json,
        metadata_json=metadata_json,
        env_path=None,
    ).config_sha256


def compute_result_config_sha256(
    *,
    runtime_toml: Path = DEFAULT_RUNTIME_TOML,
    freeze_toml: Path = DEFAULT_FREEZE_TOML,
    catalog_json: Path = DEFAULT_VARIABLE_CATALOG_JSON,
    metadata_json: Path = DEFAULT_METADATA_JSON,
) -> str:
    """Compute deterministic SHA-256 strictly across Layer 4 (model_contract + catalog) + Layer 5 (freeze)."""
    return load_settings(
        runtime_toml=runtime_toml,
        freeze_toml=freeze_toml,
        catalog_json=catalog_json,
        metadata_json=metadata_json,
        env_path=None,
    ).result_config_sha256


def compute_operational_config_sha256(
    *,
    runtime_toml: Path = DEFAULT_RUNTIME_TOML,
    freeze_toml: Path = DEFAULT_FREEZE_TOML,
    catalog_json: Path = DEFAULT_VARIABLE_CATALOG_JSON,
    metadata_json: Path = DEFAULT_METADATA_JSON,
) -> str:
    """Compute deterministic SHA-256 strictly across Layer 2 (operational) + Layer 3 (application)."""
    return load_settings(
        runtime_toml=runtime_toml,
        freeze_toml=freeze_toml,
        catalog_json=catalog_json,
        metadata_json=metadata_json,
        env_path=None,
    ).operational_config_sha256


def _extract_builder_input_vars_en(feature_builder: Any) -> Optional[set[str]]:
    from src.data.vintage_store import VI_TO_EN_COLUMNS

    imputation_stats = getattr(feature_builder, "_imputation_stats", None) or {}
    fill_values = imputation_stats.get("fill_values")
    if not isinstance(fill_values, dict) or not fill_values:
        return None
    vi_cols = set(fill_values.keys()) | {str(getattr(feature_builder, "target_col", "Lượng mưa"))}
    return {VI_TO_EN_COLUMNS.get(c, c) for c in vi_cols}


def verify_model_contract_on_startup(
    *,
    feature_builders: Sequence[Any],
    settings: Optional[RuntimeSettings] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ModelContractConfig:
    """
    Fail-fast startup guard validating loaded FeatureBuilders and metadata model_contract.
    Raises `RuntimeError("[MODEL_CONTRACT_MISMATCH] ...")` on any discrepancy.
    """
    active_settings = settings or load_settings()
    if metadata is not None:
        raw_contract = metadata.get("model_contract")
        if not isinstance(raw_contract, dict):
            raise RuntimeError("[MODEL_CONTRACT_MISMATCH] Loaded metadata lacks 'model_contract' block.")
        try:
            contract = ModelContractConfig.model_validate(raw_contract)
        except Exception as exc:
            raise RuntimeError(f"[MODEL_CONTRACT_MISMATCH] Invalid model_contract schema: {exc}") from exc
    else:
        contract = active_settings.model_contract

    if contract.horizon_days != 7:
        raise RuntimeError(f"[MODEL_CONTRACT_MISMATCH] Expected horizon_days == 7, got {contract.horizon_days}.")
    if contract.time_standard != "LST":
        raise RuntimeError(f"[MODEL_CONTRACT_MISMATCH] Expected time_standard == 'LST', got {contract.time_standard!r}.")
    if contract.canonical_location_id != "HCMC":
        raise RuntimeError(
            f"[MODEL_CONTRACT_MISMATCH] Expected canonical_location_id == 'HCMC', got {contract.canonical_location_id!r}."
        )
    if contract.completeness_policy != "COMPLETE_MODEL_REQUIRED_VARS":
        raise RuntimeError(
            f"[MODEL_CONTRACT_MISMATCH] Unsupported completeness_policy={contract.completeness_policy!r}."
        )

    from src.data.vintage_store import ALL_35_VARS

    expected_vars = set(ALL_35_VARS)
    contract_vars = set(contract.required_input_variables)
    if len(contract.required_input_variables) != 35 or contract_vars != expected_vars:
        missing = sorted(expected_vars - contract_vars)
        extra = sorted(contract_vars - expected_vars)
        raise RuntimeError(
            f"[MODEL_CONTRACT_MISMATCH] required_input_variables mismatch (count={len(contract.required_input_variables)}, "
            f"missing={missing}, extra={extra})."
        )

    catalog_vars = {v.code for v in active_settings.variable_catalog.variables if v.required_by_serving_model}
    if catalog_vars != contract_vars:
        raise RuntimeError(
            f"[MODEL_CONTRACT_MISMATCH] variable_catalog required set diverges from model_contract: "
            f"diff={sorted(contract_vars ^ catalog_vars)}."
        )

    for idx, fb in enumerate(feature_builders):
        if fb is None:
            continue
        warmup = int(fb.warmup_rows() if callable(getattr(fb, "warmup_rows", None)) else fb.config.warmup_rows)
        if warmup != contract.feature_warmup_lookback_days:
            raise RuntimeError(
                f"[MODEL_CONTRACT_MISMATCH] FeatureBuilder[{idx}] warmup_rows={warmup} != "
                f"contract.feature_warmup_lookback_days={contract.feature_warmup_lookback_days}."
            )
        fb_vars = _extract_builder_input_vars_en(fb)
        if fb_vars is not None and fb_vars != contract_vars:
            raise RuntimeError(
                f"[MODEL_CONTRACT_MISMATCH] FeatureBuilder[{idx}] fitted input vars mismatch: "
                f"missing={sorted(contract_vars - fb_vars)}, extra={sorted(fb_vars - contract_vars)}."
            )

    if contract.solar_doy_table_sha256:
        from src.inference.solar_imputation import DEFAULT_SOLAR_DOY_CLIMATOLOGY_PATH, load_solar_doy_climatology

        if not DEFAULT_SOLAR_DOY_CLIMATOLOGY_PATH.exists():
            raise RuntimeError(
                f"[MODEL_CONTRACT_MISMATCH] Missing solar DOY climatology artifact at {DEFAULT_SOLAR_DOY_CLIMATOLOGY_PATH}."
            )
        _, actual_doy_sha = load_solar_doy_climatology(DEFAULT_SOLAR_DOY_CLIMATOLOGY_PATH)
        if actual_doy_sha != contract.solar_doy_table_sha256:
            raise RuntimeError(
                f"[MODEL_CONTRACT_MISMATCH] solar_doy_table_sha256 mismatch: expected {contract.solar_doy_table_sha256}, "
                f"got {actual_doy_sha}."
            )

    return contract
