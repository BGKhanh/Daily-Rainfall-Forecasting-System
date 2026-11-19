"""
Locked Pydantic v2 API Schemas for HCMC Daily Rainfall Forecasting Engine
=========================================================================
Defines strict request/response models and unified `ErrorResponse` payloads
for all public endpoints (`/health`, `/models`, `/predict`, `/scores`, `/benchmark/leaderboard`).
"""

from __future__ import annotations

from datetime import date
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

ErrorCode = Literal[
    "UNSUPPORTED_LOCATION",
    "INCOMPLETE_RADIATION_AT_ANCHOR",
    "ANCHOR_DATE_NOT_FOUND",
    "INVALID_CLIENT_HISTORY",
    "SERVER_DB_DATA_ERROR",
    "PREDICTOR_NOT_READY",
    "INVALID_QUERY_PARAMETER",
]

SplitRegime = Literal["DEV_TRAIN", "VALIDATION", "TEST", "POST_TEST"]
DayForecastStatus = Literal["past_observed", "today", "future_forecast", "past_unverified"]
ModelKey5 = Literal[
    "SC-35",
    "SC-30",
    "SC-31",
    "BASELINE_CLIMATOLOGY_DOY",
    "BASELINE_PERSISTENCE",
]

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_optional_iso_date(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not _ISO_DATE_RE.match(raw):
        raise ValueError(f"{field_name} must use YYYY-MM-DD format, got {value!r}.")
    date.fromisoformat(raw)
    return raw


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthServingModel(BaseModel):
    scenario_id: str
    model_name: str
    governance_role: str
    alias: str
    status: str


class HealthShadowModel(BaseModel):
    scenario_id: str
    model_name: str
    governance_role: str
    alias: str
    status: str


class HealthBaselineModel(BaseModel):
    scenario_id: str
    model_kind: str
    definition: str


class AnchorRange(BaseModel):
    min: str
    max: str


class LocationItem(BaseModel):
    location_id: str
    display_name: str
    latitude: float
    longitude: float
    meteo_grid_note: str
    rad_grid_note: str
    time_standard: str
    is_active: bool


class LocationsResponse(BaseModel):
    db_role: str
    locations: List[LocationItem]


class DataGroupLatency(BaseModel):
    last_valid_date: Optional[str] = None
    lag_days_vs_scan_end: Optional[int] = None
    valid_days_in_window: Optional[int] = None


class DataLatencyInfo(BaseModel):
    last_ingest_run_id: Optional[str] = None
    last_ingest_at_utc: Optional[str] = None
    scan_window_start: Optional[str] = None
    scan_window_end: Optional[str] = None
    scan_window_total_days: Optional[int] = None
    groups: Dict[str, DataGroupLatency] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    predictor_loaded: bool
    model_source: str
    weather_backend: str
    db_role: str
    default_complete_anchor_date: Optional[str] = None
    data_snapshot_date: Optional[str] = None
    scan_window_start: Optional[str] = None
    scan_window_end: Optional[str] = None
    scan_window_total_days: Optional[int] = None
    latest_precip_obs_date: Optional[str] = None
    latest_partial_obs_explanation: Optional[str] = None
    anchor_range: Optional[AnchorRange] = None
    data_latency: Optional[DataLatencyInfo] = None
    journal_mode: Optional[str] = None
    serving_model: HealthServingModel
    shadow_models: List[HealthShadowModel]
    baselines: List[HealthBaselineModel]


class ModelsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    project: Optional[str] = None
    version: Optional[str] = None
    date_packaged: Optional[str] = None
    model_source: Optional[str] = None
    db_role: str
    timezone: Optional[str] = None
    governance_frozen_constants: Dict[str, Any] = Field(default_factory=dict)
    models: Dict[str, Dict[str, Any]]
    sha256_checksums: Dict[str, str] = Field(default_factory=dict)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Historical continuous daily records (at least 30 days). If omitted, loads canonical recent data.",
    )
    anchor_date: Optional[str] = Field(
        None,
        description="Optional anchor date (YYYY-MM-DD) in DB. Must satisfy COMPLETE_ALL_35; defaults to latest complete anchor.",
    )
    location_id: Optional[str] = Field(
        None,
        description="Canonical location identifier (locked strictly to HCMC / Ho Chi Minh City aliases).",
    )
    province: Optional[str] = Field(
        "Ho Chi Minh City",
        description="Backward-compatible location alias (locked strictly to Ho Chi Minh City / HCMC aliases).",
    )

    @field_validator("anchor_date")
    @classmethod
    def validate_anchor_date_format(cls, v: Optional[str]) -> Optional[str]:
        return _validate_optional_iso_date(v, "anchor_date")


class DailyForecastItem(BaseModel):
    target_date: str
    horizon_day: int = Field(..., ge=1, le=7)
    status: DayForecastStatus
    observed_mm: Optional[float] = None
    predictions: Dict[ModelKey5, float]


class ServingHorizonDetailItem(BaseModel):
    horizon: int = Field(..., ge=1, le=7)
    target_date: str
    pred_mm: float
    status: DayForecastStatus
    observed_mm: Optional[float] = None
    abs_error_mm: Optional[float] = None
    note: Optional[str] = None


class PredictServingModel(BaseModel):
    scenario_id: str
    model_name: str
    governance_role: str
    registry_alias: str
    feature_level: str
    predictions_7d_mm: List[float]
    hindcast_and_current: List[ServingHorizonDetailItem]
    forecast_future: List[ServingHorizonDetailItem]


class PredictShadowModel(BaseModel):
    scenario_id: str
    model_name: str
    governance_role: str
    registry_alias: Optional[str] = None
    feature_level: Optional[str] = None
    predictions_7d_mm: List[float]


class PredictLocationInfo(BaseModel):
    location_id: str
    province: str
    latitude: float
    longitude: float
    grid_resolution_note: str


class PredictResponse(BaseModel):
    anchor_date: str
    anchor_split_regime: SplitRegime
    run_date: str
    run_timezone: str
    effective_future_days: int = Field(..., ge=0, le=7)
    future_forecast_notice: Optional[str] = None
    db_role: str
    daily_forecasts: List[DailyForecastItem]

    # Backwards-compatible and observability metadata
    run_timestamp: str
    model_source: str
    history_backend: str
    data_as_of: str
    lag_days: int
    forecast_horizon_mode: Literal["operational_forecast", "retrospective_hindcast_only"]
    anchor_completeness_regime: Optional[str] = None
    serving_model: PredictServingModel
    shadow_models: Dict[str, PredictShadowModel]
    shadow_model: PredictShadowModel
    location: PredictLocationInfo


class BootstrapSummary(BaseModel):
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    bootstrap_se: float
    significant_at_95pct: Optional[bool] = None
    decision_label: Optional[str] = None


class FourMetricBlock(BaseModel):
    MAE: float
    MSE: float
    RMSE: float
    MASE: float


class HorizonBootstrapCI(BaseModel):
    MAE: BootstrapSummary
    RMSE: BootstrapSummary


class OverallBootstrapCI(BaseModel):
    MAE: BootstrapSummary
    RMSE: BootstrapSummary
    MASE: BootstrapSummary


class ModelConfidenceIntervals(BaseModel):
    block_length: int
    n_blocks: int
    overall: Optional[OverallBootstrapCI] = None
    by_horizon: Dict[str, HorizonBootstrapCI] = Field(default_factory=dict)


class ScoreLeaderboardEntry(BaseModel):
    model_id: str
    model_kind: str
    n_origins: int
    n_forecasts: int
    MAE: float
    MSE: float
    RMSE: float
    MASE: float
    MAE_vs_first: float
    rank: int
    MAE_ci95: Optional[BootstrapSummary] = None
    RMSE_ci95: Optional[BootstrapSummary] = None
    MASE_ci95: Optional[BootstrapSummary] = None


class ScoreModelDetail(BaseModel):
    model_id: str
    model_kind: str
    n_origins: int
    n_forecasts: int
    overall_vs_first: FourMetricBlock
    overall_vs_latest: FourMetricBlock
    by_horizon_vs_first: Dict[str, FourMetricBlock]
    by_horizon_vs_latest: Dict[str, FourMetricBlock]
    confidence_intervals_95pct: ModelConfidenceIntervals


class BlockBootstrapMeta(BaseModel):
    method: str
    block_length: int
    n_blocks: int
    candidate_blocks: Optional[int] = None
    n_bootstrap: int
    seed: int
    n_anchor_dates: int


class BlockSensitivityModelSummary(BaseModel):
    MAE: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    bootstrap_se: float


class BlockSensitivityEntry(BaseModel):
    block_length: int
    n_anchor_dates: int
    n_blocks: int
    candidate_blocks: int
    models: Dict[str, BlockSensitivityModelSummary] = Field(default_factory=dict)
    pairwise_delta_mae: Dict[str, BootstrapSummary] = Field(default_factory=dict)


class ScoresResponse(BaseModel):
    location_id: str
    db_role: str
    eval_split_regime: SplitRegime
    is_prospective_filter: Optional[int] = None
    mase_scale: float
    origin_min: Optional[str] = None
    origin_max: Optional[str] = None
    target_date_min: Optional[str] = None
    target_date_max: Optional[str] = None
    n_origins: int
    n_forecasts_per_model: int
    total_scored_rows: int
    is_prospective_counts: Dict[str, int]
    temporal_status_at_issue_counts: Dict[str, int]
    leaderboard: List[ScoreLeaderboardEntry]
    models: Dict[str, ScoreModelDetail]
    pairwise_delta_mae_ci: Dict[str, BootstrapSummary]
    block_bootstrap: BlockBootstrapMeta
    block_length_sensitivity: Dict[str, BlockSensitivityEntry] = Field(default_factory=dict)
    methodological_notes: Dict[str, str]


class LeaderboardResponse(BaseModel):
    benchmark_dataset: str
    evaluation_rule: str
    test_set_exposure_note: str
    config_note: str
    top_scenarios: List[Dict[str, Any]]


ObservationGranularity = Literal["daily", "weekly", "monthly"]
ObservationStatus = Literal["FINALIZED", "PROVISIONAL", "NOT_PUBLISHED"]


class ObservationsValidRange(BaseModel):
    min_date: str
    max_date: str


class ObservationSeriesPoint(BaseModel):
    date: str
    period_end: str
    precip_mm: Optional[float] = None
    precip_total_mm: Optional[float] = None
    precip_daily_mean_mm: Optional[float] = None
    published_days_count: int
    calendar_days_count: int
    status: ObservationStatus


class ForecastOverlayPoint(BaseModel):
    date: str
    horizon_day: int
    pred_mm: float


class ObservationsSummary(BaseModel):
    total_precip_mm: float
    rain_days_ge_1mm: int
    finalized_days: int
    provisional_days: int
    not_published_days: int
    published_days: int
    calendar_days: int


class ObservationsQueryResponse(BaseModel):
    location_id: str
    display_name: str
    db_role: str
    has_data: bool
    empty_reason: Optional[str] = None
    granularity: ObservationGranularity
    start_date: str
    end_date: str
    valid_range: ObservationsValidRange
    finality_mode: str
    finality_cutoff_date: Optional[str] = None
    finality_rule: str
    series: List[ObservationSeriesPoint]
    forecast_overlay: List[ForecastOverlayPoint] = Field(default_factory=list)
    forecast_overlay_anchor_date: Optional[str] = None
    summary: ObservationsSummary
