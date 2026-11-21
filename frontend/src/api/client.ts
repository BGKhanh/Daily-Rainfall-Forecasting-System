import { EN } from "../i18n/en";
import type {
  HealthResponse,
  LocationsResponse,
  ObservationGranularity,
  ObservationsQueryResponse,
  PredictResponse,
  ProvinceGeoJson,
  ProvinceGridOverlapData,
} from "../types/api";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");

export const COLD_START_THRESHOLD_MS = Number(
  import.meta.env.VITE_COLD_START_THRESHOLD_MS ?? "3000"
);

export const APP_CONFIG = {
  repoUrl: import.meta.env.VITE_REPO_URL || "https://github.com/BGKhanh/Daily-Rainfall-Forecasting-System",
  ciUrl:
    import.meta.env.VITE_CI_URL ||
    "https://github.com/BGKhanh/Daily-Rainfall-Forecasting-System/actions",
  mlflowUrl:
    import.meta.env.VITE_MLFLOW_URL ||
    "https://dagshub.com/BGKhanh/Daily-Rainfall-Forecasting-System.mlflow",
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
    const errObj = (parsed as { error?: { code?: string; message?: string; details?: Record<string, unknown> } } | null)
      ?.error;
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
      resp.statusText || EN.errors.requestFailedFallback
    );
  }

  return parsed as T;
}

export function fetchHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health");
}

export function fetchLocations(): Promise<LocationsResponse> {
  return requestJson<LocationsResponse>("/locations");
}

export function fetchPredictByAnchor(locationId: string, anchorDate?: string): Promise<PredictResponse> {
  const payload: Record<string, string> = {
    location_id: locationId,
  };
  if (anchorDate) {
    payload.anchor_date = anchorDate;
  }
  return requestJson<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function fetchObservations(params: {
  locationId: string;
  startDate?: string;
  endDate?: string;
  granularity: ObservationGranularity;
}): Promise<ObservationsQueryResponse> {
  const search = new URLSearchParams();
  search.set("location_id", params.locationId);
  search.set("granularity", params.granularity);
  if (params.startDate) search.set("start_date", params.startDate);
  if (params.endDate) search.set("end_date", params.endDate);
  return requestJson<ObservationsQueryResponse>(`/observations?${search.toString()}`);
}

export async function fetchSameOriginJson<T>(relativePath: string): Promise<T> {
  const resp = await fetch(relativePath, {
    headers: { Accept: "application/json" },
  });
  if (!resp.ok) {
    throw new ApiClientError(resp.status, "SERVER_DB_DATA_ERROR", `${EN.errors.requestFailedFallback}: ${relativePath}`);
  }
  return (await resp.json()) as T;
}

export function fetchVietnamGeoJson(): Promise<ProvinceGeoJson> {
  return fetchSameOriginJson<ProvinceGeoJson>("/map/vietnam_63_provinces.geojson");
}

export function fetchProvinceGridOverlap(): Promise<ProvinceGridOverlapData> {
  return fetchSameOriginJson<ProvinceGridOverlapData>("/map/province_grid_overlap.json");
}
