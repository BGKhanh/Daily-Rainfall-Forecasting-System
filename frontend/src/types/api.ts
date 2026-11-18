export type ErrorCode =
  | "UNSUPPORTED_LOCATION"
  | "INCOMPLETE_RADIATION_AT_ANCHOR"
  | "ANCHOR_DATE_NOT_FOUND"
  | "INVALID_CLIENT_HISTORY"
  | "SERVER_DB_DATA_ERROR"
  | "PREDICTOR_NOT_READY"
  | "INVALID_QUERY_PARAMETER";

export type SplitRegime = "DEV_TRAIN" | "VALIDATION" | "TEST" | "POST_TEST";

export type DayForecastStatus =
  | "past_observed"
  | "today"
  | "future_forecast"
  | "past_unverified";

export type ModelKey5 =
  | "SC-35"
  | "SC-30"
  | "SC-31"
  | "BASELINE_CLIMATOLOGY_DOY"
  | "BASELINE_PERSISTENCE";

export interface ErrorPayload {
  error: {
    code: ErrorCode | string;
    message: string;
    details?: Record<string, unknown>;
  };
}

export interface DataGroupLatency {
  last_valid_date: string | null;
  lag_days_vs_scan_end: number | null;
  valid_days_in_window: number | null;
}

export interface DataLatencyInfo {
  last_ingest_run_id: string | null;
  last_ingest_at_utc: string | null;
  scan_window_end: string | null;
  groups: Record<string, DataGroupLatency>;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  predictor_loaded: boolean;
  model_source: string;
  weather_backend: string;
  db_role: string;
  default_complete_anchor_date: string | null;
  data_snapshot_date: string | null;
  anchor_range: { min: string; max: string } | null;
  data_latency: DataLatencyInfo | null;
  journal_mode: string | null;
}

export interface DailyForecastItem {
  target_date: string;
  horizon_day: number;
  status: DayForecastStatus;
  observed_mm: number | null;
  predictions: Record<ModelKey5, number>;
}

export interface PredictResponse {
  anchor_date: string;
  anchor_split_regime: SplitRegime;
  run_date: string;
  run_timezone: string;
  effective_future_days: number;
  future_forecast_notice: string | null;
  db_role: string;
  daily_forecasts: DailyForecastItem[];
  run_timestamp: string;
  model_source: string;
  history_backend: string;
  data_as_of: string;
  lag_days: number;
  forecast_horizon_mode: "operational_forecast" | "retrospective_hindcast_only";
  anchor_completeness_regime: string | null;
}

export interface BootstrapSummary {
  point_estimate: number;
  ci_lower: number;
  ci_upper: number;
  ci_width: number;
  bootstrap_se: number;
  significant_at_95pct: boolean | null;
}

export interface FourMetricBlock {
  MAE: number;
  MSE: number;
  RMSE: number;
  MASE: number;
}

export interface ScoreModelDetail {
  model_id: string;
  model_kind: string;
  n_origins: number;
  n_forecasts: number;
  overall_vs_first: FourMetricBlock;
  overall_vs_latest: FourMetricBlock;
  by_horizon_vs_first: Record<string, FourMetricBlock>;
  by_horizon_vs_latest: Record<string, FourMetricBlock>;
  confidence_intervals_95pct: {
    block_length: number;
    n_blocks: number;
    overall?: {
      MAE: BootstrapSummary;
      RMSE: BootstrapSummary;
      MASE: BootstrapSummary;
    };
    by_horizon: Record<
      string,
      {
        MAE: BootstrapSummary;
        RMSE: BootstrapSummary;
      }
    >;
  };
}

export interface ScoresResponse {
  location_id: string;
  db_role: string;
  eval_split_regime: SplitRegime;
  is_prospective_filter: number | null;
  mase_scale: number;
  origin_min: string | null;
  origin_max: string | null;
  target_date_min: string | null;
  target_date_max: string | null;
  n_origins: number;
  n_forecasts_per_model: number;
  total_scored_rows: number;
  models: Record<string, ScoreModelDetail>;
  pairwise_delta_mae_ci: Record<string, BootstrapSummary>;
  block_bootstrap: {
    method: string;
    block_length: number;
    n_blocks: number;
    candidate_blocks?: number | null;
    n_bootstrap: number;
    seed: number;
    n_anchor_dates: number;
  };
  methodological_notes: Record<string, string>;
}
