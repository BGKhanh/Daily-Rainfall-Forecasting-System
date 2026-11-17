"""
FastAPI Serving Engine for HCMC Daily Rainfall Forecasting System
==================================================================
Serves operational multi-horizon rainfall forecasts using:
- Serving Champion: SC-35 XGBoost Tweedie F3 (governance_role: test_ranking_winner_as_originally_benchmarked)
- Shadow Model 1: SC-30 LightGBM Tweedie F2 Config A (governance_role: validation_selected_champion)
- Shadow Model 2: SC-31 LightGBM Tweedie F3 Config A (governance_role: post_audit_test_contender)

Features:
- Dynamic horizon tagging (past_unverified, today, future_forecast) based on data_as_of vs run_date.
- Strict location lock to Ho Chi Minh City (`location_id = "HCMC"`, 10.7769N, 106.7009E).
- Strict HTTP status separation (422 for non-HCMC locations, 400 for invalid user history payload, 503 for server DB data failures).
"""

import os
import sys
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import StandaloneRainfallPredictor
from src.inference.validation import ServingValidationError
from src.data.provider import IncompleteRadiationAtAnchorError, WeatherHistoryProvider
from src.data.vintage_store import (
    HCMC_ALIASES,
    HCMC_DISPLAY_NAME,
    HCMC_LATITUDE,
    HCMC_LOCATION_ID,
    HCMC_LONGITUDE,
)

from contextlib import asynccontextmanager

CANONICAL_LOCATION_ID = HCMC_LOCATION_ID
CANONICAL_PROVINCE_NAME = HCMC_DISPLAY_NAME
CANONICAL_LATITUDE = HCMC_LATITUDE
CANONICAL_LONGITUDE = HCMC_LONGITUDE
ALLOWED_HCMC_ALIASES = frozenset(HCMC_ALIASES)


def _normalize_location_token(value: str) -> str:
    return " ".join(value.strip().upper().split())


def resolve_canonical_hcmc_location(
    location_id: Optional[str],
    province: Optional[str],
) -> str:
    """Validate and normalize incoming location/province to canonical 'HCMC', else raise HTTP 422."""
    if location_id is not None:
        if _normalize_location_token(location_id) not in ALLOWED_HCMC_ALIASES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"[UNSUPPORTED_LOCATION] location_id={location_id!r} is not supported. "
                    f"This endpoint is locked to Ho Chi Minh City ({CANONICAL_LOCATION_ID})."
                ),
            )
    if province is not None:
        if _normalize_location_token(province) not in ALLOWED_HCMC_ALIASES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"[UNSUPPORTED_LOCATION] province={province!r} is not supported. "
                    f"This endpoint is locked to Ho Chi Minh City ({CANONICAL_LOCATION_ID})."
                ),
            )
    return CANONICAL_LOCATION_ID


predictor: Optional[StandaloneRainfallPredictor] = None
history_provider: Optional[WeatherHistoryProvider] = None


def load_serving_models():
    global predictor, history_provider
    artifact_path = PROJECT_ROOT / "artifacts" / "models"
    model_source = os.environ.get("SERVING_MODEL_SOURCE", "local")
    predictor = StandaloneRainfallPredictor(artifact_dir=artifact_path, mode=model_source)
    history_provider = WeatherHistoryProvider()
    print(
        f"Serving models ({predictor.mode}) and history provider ({history_provider.backend}) "
        "loaded successfully into API memory."
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_serving_models()
    yield

# Initialize FastAPI App
app = FastAPI(
    title="HCMC Daily Rainfall Forecasting Serving Engine",
    description="Operational API for 1-7 days daily rainfall forecasting locked to HCMC with dynamic horizon calculation and shadow model governance.",
    version="1.0.0",
    lifespan=lifespan,
)

class DailyWeatherRecord(BaseModel):
    Ngày: str = Field(..., description="Date of observation in YYYY-MM-DD format")
    Lượng_mưa: float = Field(..., alias="Lượng mưa", description="Daily precipitation in mm")
    Nhiệt_độ_2m: float = Field(..., alias="Nhiệt độ 2m", description="Temperature at 2m (C)")
    Độ_ẩm_tương_đối_2m: float = Field(..., alias="Độ ẩm tương đối 2m", description="Relative humidity at 2m (%)")
    Tốc_độ_gió_2m: float = Field(..., alias="Tốc độ gió 2m", description="Wind speed at 2m (m/s)")
    Nhiệt_độ_điểm_sương_2m: Optional[float] = Field(None, alias="Nhiệt độ điểm sương 2m")
    Áp_suất_bề_mặt: Optional[float] = Field(None, alias="Áp suất bề mặt")
    Độ_ẩm_đặc_trưng_2m: Optional[float] = Field(None, alias="Độ ẩm đặc trưng 2m")
    Bức_xạ_mặt_trời: Optional[float] = Field(None, alias="Bức xạ mặt trời")

    model_config = ConfigDict(populate_by_name=True, extra="allow")

class PredictionRequest(BaseModel):
    history: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Historical continuous daily records (at least 30 days). If omitted, loads canonical recent data."
    )
    run_date: Optional[str] = Field(
        None,
        description="Operational run date (YYYY-MM-DD). Defaults to current system date."
    )
    anchor_date: Optional[str] = Field(
        None,
        description="Optional anchor date (YYYY-MM-DD) in DB. Must satisfy COMPLETE_ALL_35; defaults to latest complete anchor."
    )
    location_id: Optional[str] = Field(
        None,
        description="Canonical location identifier (locked strictly to HCMC / Ho Chi Minh City aliases)."
    )
    province: Optional[str] = Field(
        CANONICAL_PROVINCE_NAME,
        description="Backward-compatible location alias (locked strictly to TP. Hồ Chí Minh / HCMC aliases)."
    )

@app.get("/health", tags=["Monitoring"])
def get_health():
    """Health check endpoint indicating model readiness and governance status."""
    return {
        "status": "healthy",
        "service": "HCMC Rainfall Forecasting Engine",
        "version": "1.0.0",
        "predictor_loaded": predictor is not None,
        "model_source": predictor.mode if predictor is not None else os.environ.get("SERVING_MODEL_SOURCE", "local"),
        "weather_backend": history_provider.backend if history_provider is not None else os.environ.get("WEATHER_DB_BACKEND", "local"),
        "serving_model": {
            "scenario_id": "SC-35",
            "model_name": "xgboost_tweedie_F3",
            "governance_role": "test_ranking_winner_as_originally_benchmarked",
            "alias": "serving",
            "status": "active"
        },
        "shadow_models": [
            {
                "scenario_id": "SC-30",
                "model_name": "lightgbm_tweedie_F2",
                "governance_role": "validation_selected_champion",
                "alias": "shadow_val_champ",
                "status": "active_shadow"
            },
            {
                "scenario_id": "SC-31",
                "model_name": "lightgbm_tweedie_F3",
                "governance_role": "post_audit_test_contender",
                "alias": "shadow_test_contender",
                "status": "active_shadow"
            }
        ]
    }

@app.get("/models/info", tags=["Governance"])
def get_model_metadata():
    """Returns governance metadata, hyperparameters, and cryptographic SHA-256 hashes."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    return predictor.metadata

@app.post(
    "/predict",
    tags=["Forecasting"],
    responses={
        400: {
            "description": (
                "Bad Request — Client-supplied `request.history` payload violates business or physical rules "
                "(<35 continuous days, calendar date gap, NaN/-999.0 sentinel, physical bound violation, or >7d lag)."
            )
        },
        422: {
            "description": (
                "Unprocessable Entity — Invalid JSON/Pydantic schema types, unsupported `location_id`/`province` "
                "outside Ho Chi Minh City, OR requested `anchor_date` violates `COMPLETE_ALL_35` radiation completeness."
            )
        },
        503: {
            "description": (
                "Service Unavailable — Predictor not initialized, OR server-side DB/history provider failed "
                "or returned incomplete/NaN meteorological records when `request.history` is omitted."
            )
        },
    },
)
def predict_rainfall(request: PredictionRequest):
    """
    Generate 7-day multi-horizon rainfall forecasts for Ho Chi Minh City (`HCMC`).

    Error status contract:
    - `422 Unprocessable Entity`: Invalid Pydantic/JSON schema types or `location_id`/`province` outside HCMC.
    - `400 Bad Request`: Client-supplied `request.history` fails continuity/missingness/physical-bound checks
      (or caller-specified `run_date` exceeds 7-day lag limit).
    - `503 Service Unavailable`: Server-side DB/history provider failure or corrupted/incomplete server DB records
      when `request.history` is omitted (`None`).
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    # Strict HCMC location validation -> HTTP 422 for any other province/location
    canonical_loc_id = resolve_canonical_hcmc_location(request.location_id, request.province)
    prov = CANONICAL_PROVINCE_NAME
    lat, lon = CANONICAL_LATITUDE, CANONICAL_LONGITUDE

    history_backend = "request_payload"
    effective_run_date = request.run_date
    observed_precip_by_date: Dict[str, float] = {}
    anchor_completeness_regime: Optional[str] = None
    enforce_lag_limit = True

    if request.history is not None:
        df_history = pd.DataFrame(request.history)
    else:
        try:
            active_provider = history_provider or WeatherHistoryProvider()
            fetch_kwargs: Dict[str, Any] = {
                "province": prov,
                "days": 35,
                "run_date": request.run_date,
            }
            if request.anchor_date is not None:
                fetch_kwargs["anchor_date"] = request.anchor_date
            history_res = active_provider.fetch_recent_history(**fetch_kwargs)
            df_history = history_res.history_df
            history_backend = history_res.backend
            observed_precip_by_date = dict(getattr(history_res, "observed_precip_by_date", None) or {})
            anchor_completeness_regime = getattr(history_res, "anchor_completeness_regime", None)
            if effective_run_date is None and history_res.is_static_sample:
                effective_run_date = (history_res.data_as_of + timedelta(days=3)).strftime("%Y-%m-%d")
            # Operational DB-backed runs anchored at latest COMPLETE_ALL_35 date (~78d CERES lag) allow lag > 7d
            # and switch to past_observed + future_forecast_notice when effective_future_days == 0.
            enforce_lag_limit = bool(history_res.is_static_sample)
        except IncompleteRadiationAtAnchorError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:
            detail_str = str(e)
            if "[SERVER_DB_DATA_ERROR]" not in detail_str:
                detail_str = f"[SERVER_DB_DATA_ERROR] Server history provider failed: {detail_str}"
            raise HTTPException(status_code=503, detail=detail_str) from e

    try:
        prediction_result = predictor.predict(
            raw_history_df=df_history,
            run_date=effective_run_date,
            enforce_lag_limit=enforce_lag_limit,
        )
    except ServingValidationError as e:
        if history_backend != "request_payload":
            # Missing/NaN/gapped server DB data -> HTTP 503 Service Unavailable.
            # Keep HTTP 400 when caller explicitly supplied an out-of-range future run_date.
            if e.code == "STALE_RADIATION_OR_LAG" and request.run_date is not None:
                raise HTTPException(status_code=400, detail=str(e)) from e
            raise HTTPException(
                status_code=503,
                detail=f"[SERVER_DB_DATA_ERROR] Server observation store failed validation: {e}",
            ) from e
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        status_code = 503 if history_backend != "request_payload" else 400
        raise HTTPException(status_code=status_code, detail=str(e)) from e

    if observed_precip_by_date:
        for item in prediction_result.get("serving_model", {}).get("hindcast_and_current", []):
            tgt = item.get("target_date")
            if item.get("status") == "past_unverified" and tgt in observed_precip_by_date:
                obs_mm = float(observed_precip_by_date[tgt])
                item["status"] = "past_observed"
                item["observed_mm"] = round(obs_mm, 4)
                item["abs_error_mm"] = round(abs(float(item["pred_mm"]) - obs_mm), 4)
                item["note"] = "Day has passed and precipitation observation is recorded in SQLite vintage store."

    if prediction_result.get("effective_future_days") == 0:
        prediction_result["future_forecast_notice"] = (
            "Không có ngày tương lai (effective_future_days = 0): "
            "ngày neo dữ liệu đầy đủ (COMPLETE_ALL_35) cách ngày chạy hơn 7 ngày."
        )

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

@app.get("/benchmark/leaderboard", tags=["Governance"])
def get_leaderboard(limit: int = 10):
    """Returns top audited benchmark scenarios from official benchmark test leaderboards."""
    test_csv = PROJECT_ROOT / "reports" / "test_benchmark" / "unified_72_scenarios_test_leaderboard.csv"
    if not test_csv.exists():
        raise HTTPException(status_code=404, detail="Test leaderboard file not found")
    
    df = pd.read_csv(test_csv)
    top_models = df.head(limit).replace({np.nan: None}).to_dict(orient="records")
    return {
        "benchmark_dataset": "Canonical Out-of-sample Test (2020-05-01 to 2026-04-20, 2,184 origins)",
        "evaluation_rule": "RULE: Model Evaluation Standards (MAE, MSE, RMSE, MASE)",
        "top_scenarios": top_models
    }
