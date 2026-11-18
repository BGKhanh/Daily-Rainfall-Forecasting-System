"""
FastAPI Serving Engine for HCMC Daily Rainfall Forecasting System
=================================================================
Serves operational multi-horizon rainfall forecasts using:
- Serving Champion: SC-35 XGBoost Tweedie F3 (governance_role: test_ranking_winner_as_originally_benchmarked)
- Shadow Model 1: SC-30 LightGBM Tweedie F2 Config A (governance_role: validation_selected_champion)
- Shadow Model 2: SC-31 LightGBM Tweedie F3 Config A (governance_role: post_audit_test_contender)

Features:
- Locked Pydantic v2 request/response schemas and unified ErrorResponse across 4xx/5xx.
- Canonical HCMC operational clock (`Asia/Ho_Chi_Minh`) and true UTC timestamps (`now_utc`).
- Dynamic horizon tagging (`past_observed`, `past_unverified`, `today`, `future_forecast`).
- Strict location lock to Ho Chi Minh City (`location_id = "HCMC"`, 10.7769N, 106.7009E).
- Serves built SPA static files at `/` when `frontend/dist` is present.
"""

from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import timedelta
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
from starlette.exceptions import HTTPException as StarletteHTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import (
    ErrorCode,
    ErrorResponse,
    HealthResponse,
    LeaderboardResponse,
    LocationsResponse,
    ModelsResponse,
    PredictionRequest,
    PredictResponse,
    ScoresResponse,
)
from src.config.clock import HCMC_TIMEZONE_NAME, today_hcmc
from src.data.provider import (
    DEFAULT_DEMO_VINTAGE_DB_PATH,
    IncompleteRadiationAtAnchorError,
    WeatherHistoryProvider,
    resolve_sqlite_db_path_and_role,
)
from src.data.vintage_store import (
    HCMC_ALIASES,
    HCMC_ENGLISH_DISPLAY_NAME,
    HCMC_LATITUDE,
    HCMC_LOCATION_ID,
    HCMC_LONGITUDE,
    connect_sqlite,
    derive_eval_split_regime,
    get_complete_anchor_range,
    get_data_snapshot_date,
    get_default_complete_anchor_date,
    get_latest_ingest_latency,
    get_latest_precip_and_partial_explanation,
    get_locations,
)
from src.inference.forecast_runner import (
    METHODOLOGICAL_NOTES,
    PERSISTENCE_SPEC_SHA256,
    classify_pairwise_decision_label,
    compute_block_length_sensitivity,
    compute_dev_train_doy_climatology,
    evaluate_post_test_retrospective_scores,
)
from src.inference.predictor import StandaloneRainfallPredictor
from src.inference.validation import ServingValidationError

CANONICAL_LOCATION_ID = HCMC_LOCATION_ID
CANONICAL_PROVINCE_NAME = HCMC_ENGLISH_DISPLAY_NAME
CANONICAL_LATITUDE = HCMC_LATITUDE
CANONICAL_LONGITUDE = HCMC_LONGITUDE
ALLOWED_HCMC_ALIASES = frozenset(HCMC_ALIASES)
VALID_EVAL_SPLIT_REGIMES = ("DEV_TRAIN", "VALIDATION", "TEST", "POST_TEST")

TEST_SET_EXPOSURE_NOTE = (
    "The Test set (2020-05-01 to 2026-04-30) has been exposed; the project owner selected @serving (SC-35) "
    "after inspecting Test metrics, whereas the validation-selected champion is SC-30 Config A."
)
CONFIG_NOTE = (
    "Config A is the official baseline configuration for LightGBM (num_leaves=31, max_depth=-1); "
    "SC-30 Config A and SC-31 Config A match the canonical NPZ benchmark within 0.000000 mm."
)

predictor: Optional[StandaloneRainfallPredictor] = None
history_provider: Optional[WeatherHistoryProvider] = None
_cached_doy_climatology: Optional[Dict[int, float]] = None


def _api_error(
    status_code: int,
    code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
            "details": details or {},
        },
    )


def _resolve_active_db_info() -> tuple[Optional[Path], str]:
    """Return active `(sqlite_db_path, db_role)` respecting active `history_provider` and Two-DB convention."""
    active_provider = history_provider
    if isinstance(active_provider, WeatherHistoryProvider):
        if active_provider.backend in ("local", "sqlite"):
            if active_provider.db_path is not None and Path(active_provider.db_path).exists():
                return Path(active_provider.db_path), str(active_provider.db_role)
            return None, str(active_provider.db_role)
        if active_provider.backend == "sample_csv":
            return None, "sample_csv"
        return None, str(active_provider.backend)
    return resolve_sqlite_db_path_and_role()


def _load_doy_climatology_table(db_path: Optional[Path]) -> Optional[Dict[int, float]]:
    if db_path is None or not db_path.exists():
        return None
    try:
        conn = connect_sqlite(db_path, read_only=True)
        try:
            doy_table, _ = compute_dev_train_doy_climatology(conn, location_id=CANONICAL_LOCATION_ID)
            return doy_table
        finally:
            conn.close()
    except Exception:
        return None


def _query_baseline_checksums_from_db(db_path: Optional[Path]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {
        "BASELINE_PERSISTENCE": PERSISTENCE_SPEC_SHA256,
        "BASELINE_CLIMATOLOGY_DOY": None,
    }
    if db_path is None or not db_path.exists():
        return out
    try:
        conn = connect_sqlite(db_path, read_only=True)
        try:
            for m_id in ("BASELINE_PERSISTENCE", "BASELINE_CLIMATOLOGY_DOY"):
                row = conn.execute(
                    "SELECT artifact_sha256 FROM forecasts_ledger WHERE model_id = ? LIMIT 1",
                    (m_id,),
                ).fetchone()
                if row is not None and row["artifact_sha256"]:
                    out[m_id] = str(row["artifact_sha256"])
        finally:
            conn.close()
        if out["BASELINE_CLIMATOLOGY_DOY"] is None and DEFAULT_DEMO_VINTAGE_DB_PATH.exists() and db_path.resolve() != DEFAULT_DEMO_VINTAGE_DB_PATH.resolve():
            demo_conn = connect_sqlite(DEFAULT_DEMO_VINTAGE_DB_PATH, read_only=True)
            try:
                row = demo_conn.execute(
                    "SELECT artifact_sha256 FROM forecasts_ledger WHERE model_id = 'BASELINE_CLIMATOLOGY_DOY' LIMIT 1"
                ).fetchone()
                if row is not None and row["artifact_sha256"]:
                    out["BASELINE_CLIMATOLOGY_DOY"] = str(row["artifact_sha256"])
            finally:
                demo_conn.close()
    except Exception:
        pass
    return out


def _normalize_location_token(value: str) -> str:
    return " ".join(value.strip().upper().split())


def resolve_canonical_hcmc_location(
    location_id: Optional[str],
    province: Optional[str],
) -> str:
    """Validate and normalize incoming location/province to canonical 'HCMC', else raise HTTP 422."""
    if location_id is not None:
        if _normalize_location_token(location_id) not in ALLOWED_HCMC_ALIASES:
            raise _api_error(
                422,
                "UNSUPPORTED_LOCATION",
                (
                    f"[UNSUPPORTED_LOCATION] location_id={location_id!r} is not supported. "
                    f"This endpoint is locked to Ho Chi Minh City ({CANONICAL_LOCATION_ID})."
                ),
                {"location_id": location_id},
            )
    if province is not None:
        if _normalize_location_token(province) not in ALLOWED_HCMC_ALIASES:
            raise _api_error(
                422,
                "UNSUPPORTED_LOCATION",
                (
                    f"[UNSUPPORTED_LOCATION] province={province!r} is not supported. "
                    f"This endpoint is locked to Ho Chi Minh City ({CANONICAL_LOCATION_ID})."
                ),
                {"province": province},
            )
    return CANONICAL_LOCATION_ID


def load_serving_models():
    global predictor, history_provider, _cached_doy_climatology
    artifact_path = PROJECT_ROOT / "artifacts" / "models"
    model_source = os.environ.get("SERVING_MODEL_SOURCE", "local")
    predictor = StandaloneRainfallPredictor(artifact_dir=artifact_path, mode=model_source)
    history_provider = WeatherHistoryProvider()
    db_path, _ = _resolve_active_db_info()
    if db_path is None:
        fallback_db, _ = resolve_sqlite_db_path_and_role()
        db_path = fallback_db
    _cached_doy_climatology = _load_doy_climatology_table(db_path)
    print(
        f"Serving models ({predictor.mode}) and history provider ({history_provider.backend}) "
        "loaded successfully into API memory."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_serving_models()
    yield


app = FastAPI(
    title="HCMC Daily Rainfall Forecasting Serving Engine",
    description=(
        "Operational API for 7-day multi-horizon daily rainfall forecasting locked to HCMC "
        "with locked OpenAPI schemas, Two-DB SQLite DDL V3.2 governance, and shadow model telemetry."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

_default_cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]
_extra_cors = [
    o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_cors_origins + _extra_cors,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


@app.exception_handler(StarletteHTTPException)
async def unified_http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    del request
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        payload = {
            "error": {
                "code": exc.detail["code"],
                "message": str(exc.detail.get("message", "")),
                "details": exc.detail.get("details") or {},
            }
        }
        return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(payload))

    fallback_code: ErrorCode = (
        "INVALID_QUERY_PARAMETER"
        if exc.status_code == 422
        else ("SERVER_DB_DATA_ERROR" if exc.status_code >= 500 else "INVALID_CLIENT_HISTORY")
    )
    payload = {
        "error": {
            "code": fallback_code,
            "message": str(exc.detail),
            "details": {"status_code": exc.status_code},
        }
    }
    return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(payload))


@app.exception_handler(RequestValidationError)
async def unified_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    del request
    payload = {
        "error": {
            "code": "INVALID_QUERY_PARAMETER",
            "message": "Request validation failed against API contract.",
            "details": {"errors": jsonable_encoder(exc.errors())},
        }
    }
    return JSONResponse(status_code=422, content=payload)


@app.get(
    "/locations",
    response_model=LocationsResponse,
    responses={503: {"model": ErrorResponse, "description": "Service unavailable"}},
    tags=["Monitoring"],
)
def get_configured_locations():
    """Return configured forecast locations from SQLite `locations` table (sample CSV fallback only in sample_csv mode)."""
    db_path, db_role = _resolve_active_db_info()
    if db_role == "sample_csv" or (db_path is None and history_provider is not None and history_provider.backend == "sample_csv"):
        return {
            "db_role": "sample_csv",
            "locations": [
                {
                    "location_id": CANONICAL_LOCATION_ID,
                    "display_name": HCMC_ENGLISH_DISPLAY_NAME,
                    "latitude": CANONICAL_LATITUDE,
                    "longitude": CANONICAL_LONGITUDE,
                    "meteo_grid_note": "MERRA-2 / GEOS-IT 0.5° x 0.625°",
                    "rad_grid_note": "CERES SYN1deg / FLASHFlux 1.0° x 1.0°",
                    "time_standard": "LST",
                    "is_active": True,
                }
            ],
        }

    if db_path is None or not db_path.exists():
        raise _api_error(
            503,
            "SERVER_DB_DATA_ERROR",
            "[SERVER_DB_DATA_ERROR] No SQLite vintage database available to serve /locations.",
        )

    try:
        conn = connect_sqlite(db_path, read_only=True)
        try:
            items = get_locations(conn, active_only=False)
        finally:
            conn.close()
    except Exception as exc:
        raise _api_error(
            503,
            "SERVER_DB_DATA_ERROR",
            f"[SERVER_DB_DATA_ERROR] Failed to read locations table: {exc}",
        ) from exc

    return {"db_role": db_role, "locations": items}


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse, "description": "Service unavailable"}},
    tags=["Monitoring"],
)
def get_health():
    """Health check endpoint reporting model readiness, active Two-DB role, anchor range, and group latency."""
    db_path, db_role = _resolve_active_db_info()
    default_anchor: Optional[str] = None
    data_snapshot_date: Optional[str] = None
    scan_window_start: Optional[str] = None
    scan_window_end: Optional[str] = None
    scan_window_total_days: Optional[int] = None
    latest_precip_obs_date: Optional[str] = None
    latest_partial_obs_explanation: Optional[str] = None
    anchor_range: Optional[Dict[str, str]] = None
    data_latency: Optional[Dict[str, Any]] = None
    journal_mode: Optional[str] = None
    if db_path is not None and db_path.exists():
        try:
            conn = connect_sqlite(db_path, read_only=True)
            try:
                default_anchor = get_default_complete_anchor_date(conn, location_id=CANONICAL_LOCATION_ID)
                data_snapshot_date = get_data_snapshot_date(conn, location_id=CANONICAL_LOCATION_ID)
                anchor_range = get_complete_anchor_range(conn, location_id=CANONICAL_LOCATION_ID, window_days=35)
                data_latency = get_latest_ingest_latency(conn, location_id=CANONICAL_LOCATION_ID)
                if isinstance(data_latency, dict):
                    scan_window_start = data_latency.get("scan_window_start")
                    scan_window_end = data_latency.get("scan_window_end")
                    scan_window_total_days = data_latency.get("scan_window_total_days")
                    meteo_last_valid = (data_latency.get("groups") or {}).get("METEO_GEOSIT", {}).get("last_valid_date")
                else:
                    meteo_last_valid = None
                latest_precip_obs_date, latest_partial_obs_explanation = get_latest_precip_and_partial_explanation(
                    conn,
                    location_id=CANONICAL_LOCATION_ID,
                    meteo_last_valid_date=meteo_last_valid,
                )
                jm_row = conn.execute("PRAGMA journal_mode;").fetchone()
                journal_mode = str(jm_row[0]).lower() if jm_row else None
            finally:
                conn.close()
        except Exception:
            pass

    return {
        "status": "healthy",
        "service": "HCMC Rainfall Forecasting Engine",
        "version": "1.2.0",
        "predictor_loaded": predictor is not None,
        "model_source": predictor.mode if predictor is not None else os.environ.get("SERVING_MODEL_SOURCE", "local"),
        "weather_backend": history_provider.backend if history_provider is not None else os.environ.get("WEATHER_DB_BACKEND", "local"),
        "db_role": db_role,
        "default_complete_anchor_date": default_anchor,
        "data_snapshot_date": data_snapshot_date,
        "scan_window_start": scan_window_start,
        "scan_window_end": scan_window_end,
        "scan_window_total_days": scan_window_total_days,
        "latest_precip_obs_date": latest_precip_obs_date,
        "latest_partial_obs_explanation": latest_partial_obs_explanation,
        "anchor_range": anchor_range,
        "data_latency": data_latency,
        "journal_mode": journal_mode,
        "serving_model": {
            "scenario_id": "SC-35",
            "model_name": "xgboost_tweedie_F3",
            "governance_role": "test_ranking_winner_as_originally_benchmarked",
            "alias": "serving",
            "status": "active",
        },
        "shadow_models": [
            {
                "scenario_id": "SC-30",
                "model_name": "lightgbm_tweedie_F2",
                "governance_role": "validation_selected_champion",
                "alias": "shadow_val_champ",
                "status": "active_shadow",
            },
            {
                "scenario_id": "SC-31",
                "model_name": "lightgbm_tweedie_F3",
                "governance_role": "post_audit_test_contender",
                "alias": "shadow_test_contender",
                "status": "active_shadow",
            },
        ],
        "baselines": [
            {
                "scenario_id": "BASELINE_PERSISTENCE",
                "model_kind": "BASELINE",
                "definition": "PRECTOTCORR[data_as_of] for h=1..7",
            },
            {
                "scenario_id": "BASELINE_CLIMATOLOGY_DOY",
                "model_kind": "BASELINE",
                "definition": "DevTrain (2001-01-01..2018-04-30) dayofyear mean",
            },
        ],
    }


def _build_models_payload() -> Dict[str, Any]:
    if predictor is None:
        raise _api_error(503, "PREDICTOR_NOT_READY", "Predictor not initialized")
    db_path, db_role = _resolve_active_db_info()
    baseline_shas = _query_baseline_checksums_from_db(db_path)

    payload = deepcopy(predictor.metadata)
    models_section = payload.setdefault("models", {})
    models_section.setdefault(
        "baseline_persistence",
        {
            "scenario_id": "BASELINE_PERSISTENCE",
            "model_name": "persistence_rain_anchor",
            "model_kind": "BASELINE",
            "governance_role": "mandatory_naive_baseline",
            "definition": "pred_mm(h) = PRECTOTCORR[data_as_of] for h in 1..7",
            "artifact_sha256": baseline_shas["BASELINE_PERSISTENCE"],
        },
    )
    models_section.setdefault(
        "baseline_climatology_doy",
        {
            "scenario_id": "BASELINE_CLIMATOLOGY_DOY",
            "model_name": "climatology_doy_dev_train",
            "model_kind": "BASELINE",
            "governance_role": "mandatory_seasonal_baseline",
            "definition": "Day-of-year (1..366) mean precipitation frozen on Dev Train (2001-01-01..2018-04-30)",
            "artifact_sha256": baseline_shas["BASELINE_CLIMATOLOGY_DOY"],
        },
    )
    payload["db_role"] = db_role
    return payload


@app.get(
    "/models",
    response_model=ModelsResponse,
    responses={503: {"model": ErrorResponse, "description": "Predictor not initialized"}},
    tags=["Governance"],
)
def get_model_metadata():
    """Returns governance metadata, hyperparameters, baselines, and cryptographic SHA-256 hashes."""
    return _build_models_payload()


@app.get(
    "/models/info",
    response_model=ModelsResponse,
    include_in_schema=False,
    tags=["Governance"],
)
def get_model_metadata_alias():
    """Backwards-compatible hidden alias for GET /models."""
    return _build_models_payload()


def _compute_baseline_predictions_for_anchor(
    df_history: pd.DataFrame,
    anchor_date_str: str,
) -> tuple[List[float], List[float]]:
    ordered = df_history.copy()
    ordered["Ngày"] = pd.to_datetime(ordered["Ngày"]).dt.normalize()
    ordered = ordered.sort_values("Ngày").reset_index(drop=True)
    anchor_rain = round(max(0.0, float(ordered["Lượng mưa"].iloc[-1])), 2)
    persistence_7d = [anchor_rain] * 7

    doy_table = _cached_doy_climatology
    if doy_table is None:
        fallback_db, _ = resolve_sqlite_db_path_and_role()
        doy_table = _load_doy_climatology_table(fallback_db)

    anchor_dt = pd.to_datetime(anchor_date_str).normalize()
    if doy_table:
        fallback_mean = float(np.mean(list(doy_table.values())))
        clim_7d = [
            round(max(0.0, float(doy_table.get((anchor_dt + timedelta(days=h)).dayofyear, fallback_mean))), 2)
            for h in range(1, 8)
        ]
    else:
        hist_mean = round(max(0.0, float(ordered["Lượng mưa"].mean())), 2)
        clim_7d = [hist_mean] * 7
    return persistence_7d, clim_7d


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Forecasting"],
    responses={
        400: {
            "model": ErrorResponse,
            "description": (
                "Bad Request — Client-supplied `request.history` payload violates business or physical rules "
                "(<30 continuous days, calendar date gap, NaN/-999.0 sentinel, physical bound violation, or stale radiation lag)."
            ),
        },
        422: {
            "model": ErrorResponse,
            "description": (
                "Unprocessable Entity — Invalid JSON/Pydantic schema types, unsupported `location_id`/`province` "
                "outside Ho Chi Minh City, OR `anchor_date` missing/violating `COMPLETE_ALL_35` radiation completeness."
            ),
        },
        503: {
            "model": ErrorResponse,
            "description": (
                "Service Unavailable — Predictor not initialized, OR server-side DB/history provider failed "
                "or returned incomplete/NaN meteorological records when `request.history` is omitted."
            ),
        },
    },
)
def predict_rainfall(request: PredictionRequest):
    """
    Generate 7-day multi-horizon rainfall predictions after `anchor_date` for Ho Chi Minh City (`HCMC`).
    """
    if predictor is None:
        raise _api_error(503, "PREDICTOR_NOT_READY", "Predictor not initialized")

    canonical_loc_id = resolve_canonical_hcmc_location(request.location_id, request.province)
    prov = CANONICAL_PROVINCE_NAME
    lat, lon = CANONICAL_LATITUDE, CANONICAL_LONGITUDE

    history_backend = "request_payload"
    effective_run_date = today_hcmc().strftime("%Y-%m-%d")
    observed_precip_by_date: Dict[str, float] = {}
    anchor_completeness_regime: Optional[str] = None
    enforce_lag_limit = True
    _, active_db_role = _resolve_active_db_info()
    response_db_role = "request_payload" if request.history is not None else active_db_role

    if request.history is not None:
        df_history = pd.DataFrame(request.history)
    else:
        try:
            active_provider = history_provider or WeatherHistoryProvider()
            fetch_kwargs: Dict[str, Any] = {
                "province": prov,
                "days": 35,
                "run_date": effective_run_date,
            }
            if request.anchor_date is not None:
                fetch_kwargs["anchor_date"] = request.anchor_date
            history_res = active_provider.fetch_recent_history(**fetch_kwargs)
            df_history = history_res.history_df
            history_backend = history_res.backend
            response_db_role = history_res.db_role
            observed_precip_by_date = dict(getattr(history_res, "observed_precip_by_date", None) or {})
            anchor_completeness_regime = getattr(history_res, "anchor_completeness_regime", None)
            if history_res.is_static_sample:
                effective_run_date = (history_res.data_as_of + timedelta(days=3)).strftime("%Y-%m-%d")
            enforce_lag_limit = bool(history_res.is_static_sample)
        except IncompleteRadiationAtAnchorError as e:
            if e.completeness_regime is None:
                raise _api_error(
                    422,
                    "ANCHOR_DATE_NOT_FOUND",
                    f"[ANCHOR_DATE_NOT_FOUND] {e}",
                    {"anchor_date": e.anchor_date},
                ) from e
            raise _api_error(
                422,
                "INCOMPLETE_RADIATION_AT_ANCHOR",
                str(e),
                {"anchor_date": e.anchor_date, "completeness_regime": e.completeness_regime},
            ) from e
        except Exception as e:
            detail_str = str(e)
            if "[SERVER_DB_DATA_ERROR]" not in detail_str:
                detail_str = f"[SERVER_DB_DATA_ERROR] Server history provider failed: {detail_str}"
            raise _api_error(503, "SERVER_DB_DATA_ERROR", detail_str) from e

    try:
        prediction_result = predictor.predict(
            raw_history_df=df_history,
            run_date=effective_run_date,
            enforce_lag_limit=enforce_lag_limit,
        )
    except ServingValidationError as e:
        if history_backend != "request_payload":
            raise _api_error(
                503,
                "SERVER_DB_DATA_ERROR",
                f"[SERVER_DB_DATA_ERROR] Server observation store failed validation: {e}",
                {"validation_code": e.code, **(e.details or {})},
            ) from e
        raise _api_error(
            400,
            "INVALID_CLIENT_HISTORY",
            str(e),
            {"validation_code": e.code, **(e.details or {})},
        ) from e
    except Exception as e:
        status_code = 503 if history_backend != "request_payload" else 400
        err_code: ErrorCode = "SERVER_DB_DATA_ERROR" if status_code == 503 else "INVALID_CLIENT_HISTORY"
        raise _api_error(status_code, err_code, str(e)) from e

    if observed_precip_by_date:
        for item in prediction_result.get("serving_model", {}).get("hindcast_and_current", []):
            tgt = item.get("target_date")
            if item.get("status") == "past_unverified" and tgt in observed_precip_by_date:
                obs_mm = float(observed_precip_by_date[tgt])
                item["status"] = "past_observed"
                item["observed_mm"] = round(obs_mm, 4)
                item["abs_error_mm"] = round(abs(float(item["pred_mm"]) - obs_mm), 4)
                item["note"] = "Day has passed and precipitation observation is recorded in SQLite vintage store."

    for section_key in ("hindcast_and_current", "forecast_future"):
        for item in prediction_result.get("serving_model", {}).get(section_key, []):
            if item.get("status") in ("today", "future_forecast"):
                item["observed_mm"] = None

    if prediction_result.get("effective_future_days") == 0:
        prediction_result["forecast_horizon_mode"] = "retrospective_hindcast_only"
        prediction_result["future_forecast_notice"] = (
            "This is a retrospective replay (effective_future_days = 0): the latest complete anchor date "
            "(COMPLETE_ALL_35) is more than 7 days before the current Ho Chi Minh City operational date."
        )
    else:
        prediction_result["forecast_horizon_mode"] = "operational_forecast"
        prediction_result["future_forecast_notice"] = None

    anchor_date_str = str(prediction_result["data_as_of"])
    persistence_7d, clim_7d = _compute_baseline_predictions_for_anchor(df_history, anchor_date_str)
    sc35_preds = prediction_result["serving_model"]["predictions_7d_mm"]
    sc30_preds = prediction_result["shadow_models"]["shadow_val_champ"]["predictions_7d_mm"]
    sc31_preds = prediction_result["shadow_models"]["shadow_test_contender"]["predictions_7d_mm"]

    status_by_horizon: Dict[int, tuple[str, Optional[float]]] = {}
    for item in prediction_result["serving_model"]["hindcast_and_current"]:
        status_by_horizon[int(item["horizon"])] = (
            str(item["status"]),
            None if item.get("status") in ("today", "future_forecast") else item.get("observed_mm"),
        )
    for item in prediction_result["serving_model"]["forecast_future"]:
        status_by_horizon[int(item["horizon"])] = (str(item["status"]), None)

    anchor_dt = pd.to_datetime(anchor_date_str).normalize()
    daily_forecasts: List[Dict[str, Any]] = []
    for h in range(1, 8):
        target_str = (anchor_dt + timedelta(days=h)).strftime("%Y-%m-%d")
        day_status, day_obs = status_by_horizon[h]
        daily_forecasts.append(
            {
                "target_date": target_str,
                "horizon_day": h,
                "status": day_status,
                "observed_mm": day_obs,
                "predictions": {
                    "SC-35": float(sc35_preds[h - 1]),
                    "SC-30": float(sc30_preds[h - 1]),
                    "SC-31": float(sc31_preds[h - 1]),
                    "BASELINE_CLIMATOLOGY_DOY": float(clim_7d[h - 1]),
                    "BASELINE_PERSISTENCE": float(persistence_7d[h - 1]),
                },
            }
        )

    prediction_result["anchor_date"] = anchor_date_str
    prediction_result["anchor_split_regime"] = derive_eval_split_regime(anchor_date_str)
    prediction_result["run_timezone"] = HCMC_TIMEZONE_NAME
    prediction_result["db_role"] = response_db_role
    prediction_result["daily_forecasts"] = daily_forecasts
    prediction_result["history_backend"] = history_backend
    if anchor_completeness_regime is not None:
        prediction_result["anchor_completeness_regime"] = anchor_completeness_regime
    prediction_result["location"] = {
        "location_id": canonical_loc_id,
        "province": prov,
        "latitude": lat,
        "longitude": lon,
        "grid_resolution_note": "MERRA-2 / GEOS-IT 0.5 deg x 0.625 deg; CERES SYN1deg / FLASHFlux 1.0 deg x 1.0 deg",
    }

    return prediction_result


def _evaluate_scores_with_sensitivity(
    conn: sqlite3.Connection,
    *,
    eval_split_regime: str,
    is_prospective: Optional[int],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Run primary `/scores` evaluation and matched 7/14/21-day block-length sensitivity."""
    eval_res = evaluate_post_test_retrospective_scores(
        conn,
        eval_split_regime=eval_split_regime,
        location_id=CANONICAL_LOCATION_ID,
        is_prospective_filter=is_prospective,
        block_length=7,
        n_bootstrap=2000,
        seed=42,
    )
    sensitivity_out: Dict[str, Any] = {}
    if int(eval_res.get("total_scored_rows", 0)) > 0:
        sens_res = compute_block_length_sensitivity(
            conn,
            block_lengths=(7, 14, 21),
            eval_split_regime=eval_split_regime,
            location_id=CANONICAL_LOCATION_ID,
            is_prospective_filter=is_prospective,
            n_bootstrap=2000,
            seed=42,
        )
        sensitivity_out = sens_res.get("by_block_length", {})
    return eval_res, sensitivity_out


def _empty_scores_payload(eval_split_regime: str, is_prospective: Optional[int]) -> Dict[str, Any]:
    return {
        "eval_split_regime": eval_split_regime,
        "is_prospective_filter": is_prospective,
        "mase_scale": 3.7015424361567635,
        "origin_min": None,
        "origin_max": None,
        "target_date_min": None,
        "target_date_max": None,
        "unique_origins": 0,
        "total_scored_rows": 0,
        "is_prospective_counts": {},
        "temporal_status_at_issue_counts": {},
        "obs_revision_delta_nonzero_count": 0,
        "leaderboard_vs_latest": [],
        "models": {},
        "block_bootstrap": {
            "method": "moving_block_bootstrap",
            "block_length": 7,
            "n_blocks": 0,
            "candidate_blocks": 0,
            "n_bootstrap": 2000,
            "seed": 42,
            "n_anchor_dates": 0,
            "models": {},
            "pairwise_delta_mae": {},
        },
        "methodological_notes": dict(METHODOLOGICAL_NOTES),
    }


@app.get(
    "/scores",
    response_model=ScoresResponse,
    tags=["Governance"],
    responses={
        422: {"model": ErrorResponse, "description": "Unprocessable Entity — Invalid `eval_split_regime` or `is_prospective` query parameter."},
        503: {"model": ErrorResponse, "description": "Service Unavailable — SQLite vintage database unavailable or corrupted."},
    },
)
def get_scores(
    eval_split_regime: str = Query(
        "POST_TEST",
        description="Canonical split regime filter ('DEV_TRAIN', 'VALIDATION', 'TEST', 'POST_TEST').",
    ),
    is_prospective: Optional[int] = Query(
        None,
        description="Optional filter: 1 for strictly prospective forecasts, 0 for retrospective, or omit for all.",
    ),
):
    """
    Query `v_forecast_scores` from active SQLite Vintage Store and return per-model overall & horizon
    scores (MAE, MSE, RMSE, MASE), Moving Block Bootstrap 95% CIs (`block_length = 7`), sample sizes,
    pairwise ML differences, and mandatory methodological notes.
    """
    if eval_split_regime not in VALID_EVAL_SPLIT_REGIMES:
        raise _api_error(
            422,
            "INVALID_QUERY_PARAMETER",
            (
                f"[INVALID_QUERY_PARAMETER] Unsupported eval_split_regime={eval_split_regime!r}. "
                f"Expected one of {VALID_EVAL_SPLIT_REGIMES}."
            ),
            {"eval_split_regime": eval_split_regime, "allowed": list(VALID_EVAL_SPLIT_REGIMES)},
        )
    if is_prospective is not None and is_prospective not in (0, 1):
        raise _api_error(
            422,
            "INVALID_QUERY_PARAMETER",
            f"[INVALID_QUERY_PARAMETER] is_prospective must be 0 or 1, got {is_prospective!r}.",
            {"is_prospective": is_prospective},
        )

    db_path, db_role = _resolve_active_db_info()
    if db_path is None or not db_path.exists():
        raise _api_error(
            503,
            "SERVER_DB_DATA_ERROR",
            "[SERVER_DB_DATA_ERROR] No SQLite vintage database available to serve /scores.",
        )

    sensitivity_out: Dict[str, Any] = {}
    try:
        conn = connect_sqlite(db_path, read_only=True)
        try:
            try:
                eval_res, sensitivity_out = _evaluate_scores_with_sensitivity(
                    conn,
                    eval_split_regime=eval_split_regime,
                    is_prospective=is_prospective,
                )
            except ValueError:
                eval_res = _empty_scores_payload(eval_split_regime, is_prospective)
        finally:
            conn.close()
        if (
            int(eval_res.get("total_scored_rows", 0)) == 0
            and db_role == "local_or_production_db"
            and is_prospective != 1
            and DEFAULT_DEMO_VINTAGE_DB_PATH.exists()
            and db_path.resolve() != DEFAULT_DEMO_VINTAGE_DB_PATH.resolve()
        ):
            # Production DB intentionally omits retrospective POST_TEST rows (Round 8 A4);
            # serve the bundled demo ledger truthfully under db_role="bundled_demo_db".
            demo_conn = connect_sqlite(DEFAULT_DEMO_VINTAGE_DB_PATH, read_only=True)
            try:
                try:
                    eval_res, sensitivity_out = _evaluate_scores_with_sensitivity(
                        demo_conn,
                        eval_split_regime=eval_split_regime,
                        is_prospective=is_prospective,
                    )
                    db_role = "bundled_demo_db"
                except ValueError:
                    pass
            finally:
                demo_conn.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise _api_error(
            503,
            "SERVER_DB_DATA_ERROR",
            f"[SERVER_DB_DATA_ERROR] Failed to evaluate /scores from SQLite store: {exc}",
        ) from exc

    boot = eval_res.get("block_bootstrap", {})
    boot_models = boot.get("models", {})
    b_len = int(boot.get("block_length", 7))
    b_n_blocks = int(boot.get("n_blocks", 0))

    pairwise_out: Dict[str, Any] = {}
    for pair_key, pair_summary in (boot.get("pairwise_delta_mae") or {}).items():
        enriched = dict(pair_summary)
        enriched.setdefault("decision_label", classify_pairwise_decision_label(pair_key, enriched))
        pairwise_out[pair_key] = enriched

    models_out: Dict[str, Any] = {}
    for m_id, m_entry in eval_res.get("models", {}).items():
        m_boot = boot_models.get(m_id, {})
        models_out[m_id] = {
            **m_entry,
            "confidence_intervals_95pct": {
                "block_length": b_len,
                "n_blocks": b_n_blocks,
                **m_boot,
            },
        }

    leaderboard_out: List[Dict[str, Any]] = []
    for row in eval_res.get("leaderboard_vs_latest", []):
        m_id = row["model_id"]
        ov_boot = boot_models.get(m_id, {}).get("overall", {})
        leaderboard_out.append(
            {
                **row,
                "MAE_ci95": ov_boot.get("MAE"),
                "RMSE_ci95": ov_boot.get("RMSE"),
                "MASE_ci95": ov_boot.get("MASE"),
            }
        )

    n_origins = int(eval_res.get("unique_origins", 0))
    n_forecasts_per_model = (
        int(next(iter(eval_res["models"].values()))["n_forecasts"]) if eval_res.get("models") else 0
    )

    return {
        "location_id": CANONICAL_LOCATION_ID,
        "db_role": db_role,
        "eval_split_regime": eval_split_regime,
        "is_prospective_filter": is_prospective,
        "mase_scale": eval_res.get("mase_scale"),
        "origin_min": eval_res.get("origin_min"),
        "origin_max": eval_res.get("origin_max"),
        "target_date_min": eval_res.get("target_date_min"),
        "target_date_max": eval_res.get("target_date_max"),
        "n_origins": n_origins,
        "n_forecasts_per_model": n_forecasts_per_model,
        "total_scored_rows": int(eval_res.get("total_scored_rows", 0)),
        "is_prospective_counts": eval_res.get("is_prospective_counts", {}),
        "temporal_status_at_issue_counts": eval_res.get("temporal_status_at_issue_counts", {}),
        "leaderboard": leaderboard_out,
        "models": models_out,
        "pairwise_delta_mae_ci": pairwise_out,
        "block_bootstrap": {
            "method": boot.get("method", "moving_block_bootstrap"),
            "block_length": b_len,
            "n_blocks": b_n_blocks,
            "candidate_blocks": boot.get("candidate_blocks"),
            "n_bootstrap": int(boot.get("n_bootstrap", 2000)),
            "seed": int(boot.get("seed", 42)),
            "n_anchor_dates": int(boot.get("n_anchor_dates", n_origins)),
        },
        "block_length_sensitivity": sensitivity_out,
        "methodological_notes": eval_res.get("methodological_notes", dict(METHODOLOGICAL_NOTES)),
    }


@app.get(
    "/benchmark/leaderboard",
    response_model=LeaderboardResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid limit parameter"},
        503: {"model": ErrorResponse, "description": "Benchmark leaderboard artifact unavailable"},
    },
    tags=["Governance"],
)
def get_leaderboard(limit: int = Query(10, ge=1, le=72)):
    """Returns top audited benchmark scenarios from official benchmark test leaderboards."""
    test_csv = PROJECT_ROOT / "reports" / "test_benchmark" / "unified_72_scenarios_test_leaderboard.csv"
    if not test_csv.exists():
        raise _api_error(503, "SERVER_DB_DATA_ERROR", "Test leaderboard file not found")

    df = pd.read_csv(test_csv)
    top_models = df.head(limit).replace({np.nan: None}).to_dict(orient="records")
    return {
        "benchmark_dataset": "Canonical Out-of-sample Test (2020-05-01 to 2026-04-20, 2,184 origins)",
        "evaluation_rule": "RULE: Model Evaluation Standards (MAE, MSE, RMSE, MASE)",
        "test_set_exposure_note": TEST_SET_EXPOSURE_NOTE,
        "config_note": CONFIG_NOTE,
        "top_scenarios": top_models,
    }


_FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
if (_FRONTEND_DIST_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST_DIR), html=True), name="frontend-spa")
