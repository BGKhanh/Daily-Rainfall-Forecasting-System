import type { ErrorPayload, HealthResponse, PredictResponse, ScoresResponse } from "../types/api";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");

export const COLD_START_THRESHOLD_MS = Number(
  import.meta.env.VITE_COLD_START_THRESHOLD_MS ?? "3000"
);

export const APP_CONFIG = {
  repoUrl: import.meta.env.VITE_REPO_URL || "https://github.com/BGKhanh/Precipitation-EDA",
  ciUrl:
    import.meta.env.VITE_CI_URL ||
    "https://github.com/BGKhanh/Precipitation-EDA/actions",
  mlflowUrl:
    import.meta.env.VITE_MLFLOW_URL ||
    "https://dagshub.com/BGKhanh/Precipitation-EDA.mlflow",
};

export class ApiClientError extends Error {
  status: number;
  code: string;
  details: Record<string, unknown>;

  constructor(status: number, code: string, message: string, details: Record<string, unknown> = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export function mapApiErrorToVietnamese(err: unknown): string {
  if (err instanceof ApiClientError) {
    switch (err.code) {
      case "INCOMPLETE_RADIATION_AT_ANCHOR":
      case "ANCHOR_DATE_NOT_FOUND":
        return `Ngày neo không hợp lệ: ${err.message}`;
      case "UNSUPPORTED_LOCATION":
        return `Địa điểm không được hỗ trợ (hệ thống khóa cứng TP.HCM): ${err.message}`;
      case "INVALID_CLIENT_HISTORY":
      case "INVALID_QUERY_PARAMETER":
        return `Tham số hoặc chuỗi đầu vào không hợp lệ: ${err.message}`;
      case "SERVER_DB_DATA_ERROR":
      case "PREDICTOR_NOT_READY":
        return "Lỗi máy chủ dữ liệu, thử lại sau";
      default:
        if (err.status >= 500) {
          return "Lỗi máy chủ dữ liệu, thử lại sau";
        }
        return `Ngày neo không hợp lệ: ${err.message}`;
    }
  }
  return "Lỗi máy chủ dữ liệu, thử lại sau";
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...(init?.headers ?? {}),
    },
  });

  const bodyText = await resp.text();
  let parsed: unknown = null;
  if (bodyText) {
    try {
      parsed = JSON.parse(bodyText);
    } catch {
      parsed = null;
    }
  }

  if (!resp.ok) {
    const errObj = (parsed as ErrorPayload | null)?.error;
    if (errObj && errObj.code) {
      throw new ApiClientError(
        resp.status,
        errObj.code,
        errObj.message || resp.statusText,
        errObj.details ?? {}
      );
    }
    throw new ApiClientError(
      resp.status,
      resp.status >= 500 ? "SERVER_DB_DATA_ERROR" : "INVALID_QUERY_PARAMETER",
      resp.statusText || "Yêu cầu thất bại"
    );
  }

  return parsed as T;
}

export function fetchHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health");
}

export function fetchPredictByAnchor(anchorDate?: string): Promise<PredictResponse> {
  const payload: Record<string, string> = {
    location_id: "HCMC",
  };
  if (anchorDate) {
    payload.anchor_date = anchorDate;
  }
  return requestJson<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function fetchScores(evalSplitRegime = "POST_TEST"): Promise<ScoresResponse> {
  const q = encodeURIComponent(evalSplitRegime);
  return requestJson<ScoresResponse>(`/scores?eval_split_regime=${q}`);
}
