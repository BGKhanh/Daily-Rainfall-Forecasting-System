"""
Unit and Integration Tests for FastAPI Serving Engine
=====================================================
Tests:
- GET /health
- GET /models/info
- GET /benchmark/leaderboard
- POST /predict with default canonical fallback
- POST /predict with explicit custom history payload and dynamic lag
"""

import sys
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.protocols import get_global_read_guard
from src.api.main import app, load_serving_models

@pytest.fixture(autouse=True)
def init_api_and_unguard():
    g = get_global_read_guard()
    was_active = g.is_active
    g.deactivate()
    load_serving_models()
    yield
    if was_active:
        g.activate()

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["predictor_loaded"] is True
    assert data["serving_model"]["scenario_id"] == "SC-35"
    assert len(data["shadow_models"]) == 2
    aliases = [m["alias"] for m in data["shadow_models"]]
    assert "shadow_val_champ" in aliases
    assert "shadow_test_contender" in aliases

def test_models_info_endpoint(client):
    resp = client.get("/models/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "serving" in data["models"]
    assert "shadow_val_champ" in data["models"]
    assert "shadow_test_contender" in data["models"]
    assert "sha256_checksums" in data
    assert "sc31_lightgbm_tweedie_f3.joblib" in data["sha256_checksums"]

def test_benchmark_leaderboard_endpoint(client):
    resp = client.get("/benchmark/leaderboard?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert "top_scenarios" in data
    assert len(data["top_scenarios"]) <= 5
    # Under canonical Config A (max_depth=-1), SC-31 ranks #1 (3.7246 mm) and deployed SC-35 ranks #2 (3.7333 mm).
    assert data["top_scenarios"][0]["scenario_id"] == "SC-31"
    assert data["top_scenarios"][1]["scenario_id"] == "SC-35"

def test_predict_with_default_data(client):
    resp = client.post("/predict", json={
        "province": "TP. Hồ Chí Minh"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "serving_model" in data
    assert "shadow_models" in data
    assert len(data["serving_model"]["predictions_7d_mm"]) == 7
    assert data["location"]["province"] == "TP. Hồ Chí Minh"
    assert "effective_future_days" in data
    assert "lag_days" in data

def test_predict_dynamic_horizon_lag4(client):
    # First get data_as_of from base prediction
    resp_base = client.post("/predict", json={})
    assert resp_base.status_code == 200
    data_as_of_str = resp_base.json()["data_as_of"]
    d_as_of = datetime.strptime(data_as_of_str, "%Y-%m-%d").date()

    # Set run_date to exactly 4 days after data_as_of (lag = 4)
    run_date_lag4 = (d_as_of + timedelta(days=4)).strftime("%Y-%m-%d")
    resp = client.post("/predict", json={
        "run_date": run_date_lag4
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["lag_days"] == 4
    assert data["effective_future_days"] == 3
    assert len(data["serving_model"]["hindcast_and_current"]) == 4
    assert len(data["serving_model"]["forecast_future"]) == 3
    assert data["serving_model"]["hindcast_and_current"][-1]["status"] == "today"

@pytest.mark.parametrize(
    "payload",
    [
        {"province": "Hoàng Sa"},
        {"province": "Trường Sa"},
        {"province": "Hà Nội"},
        {"province": "UNKNOWN"},
        {"province": ""},
        {"province": "   "},
        {"location_id": "Hà Nội"},
        {"location_id": "Hoàng Sa"},
        {"location_id": ""},
        {"location_id": "HCMC", "province": "Hà Nội"},
    ],
)
def test_predict_rejects_non_hcmc_locations_with_422(client, payload):
    """Mục 4 & 6: API is locked to HCMC; any other location returns HTTP 422."""
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    assert "UNSUPPORTED_LOCATION" in str(resp.json().get("detail", ""))


@pytest.mark.parametrize(
    "alias",
    [
        "HCMC",
        "hcmc",
        "HO_CHI_MINH",
        "HOCHIMINH",
        "Saigon",
        "TPHCM",
        "TP.HCM",
        "TP. Hồ Chí Minh",
        "  tp   ho  chi  minh  ",
    ],
)
def test_predict_accepts_hcmc_aliases_and_normalizes(client, alias):
    """Mục 6: All HCMC aliases resolve to canonical location_id='HCMC' and (10.7769, 106.7009)."""
    resp = client.post("/predict", json={"location_id": alias})
    assert resp.status_code == 200
    loc = resp.json()["location"]
    assert loc["location_id"] == "HCMC"
    assert loc["province"] == "TP. Hồ Chí Minh"
    assert loc["latitude"] == pytest.approx(10.7769)
    assert loc["longitude"] == pytest.approx(106.7009)


def test_predict_server_db_data_error_returns_503(client, monkeypatch):
    """Mục 4.6: Missing/NaN meteo data from server history_provider returns HTTP 503 (not 400)."""
    import pandas as pd
    from unittest.mock import MagicMock
    import src.api.main as api_main
    from src.data.provider import WeatherHistoryResult

    sample_csv = PROJECT_ROOT / "artifacts" / "models" / "sample_recent_history_35d.csv"
    corrupt_df = pd.read_csv(sample_csv)
    corrupt_df.loc[corrupt_df.index[-1], "Lượng mưa"] = float("nan")

    fake_provider = MagicMock()
    fake_provider.backend = "turso"
    fake_provider.fetch_recent_history.return_value = WeatherHistoryResult(
        history_df=corrupt_df,
        backend="turso",
        is_static_sample=False,
        data_as_of=pd.Timestamp("2026-04-30"),
        source_detail="turso:observations_vintage",
    )
    monkeypatch.setattr(api_main, "history_provider", fake_provider)

    # Server-sourced corrupt data (request.history is None) -> HTTP 503
    resp_server = client.post("/predict", json={"location_id": "HCMC", "run_date": "2026-05-03"})
    assert resp_server.status_code == 503
    assert "SERVER_DB_DATA_ERROR" in str(resp_server.json().get("detail", ""))

    # Client-supplied corrupt history (request.history is not None) -> HTTP 400
    corrupt_payload = corrupt_df.copy()
    corrupt_payload["Ngày"] = pd.to_datetime(corrupt_payload["Ngày"]).dt.strftime("%Y-%m-%d")
    corrupt_records = corrupt_payload.to_dict(orient="records")
    corrupt_records[-1]["Lượng mưa"] = None
    resp_client = client.post(
        "/predict",
        json={
            "location_id": "HCMC",
            "run_date": "2026-05-03",
            "history": corrupt_records,
        },
    )
    assert resp_client.status_code == 400
    assert "INVALID_OR_MISSING_METEO_INPUT" in str(resp_client.json().get("detail", ""))


def _seed_round6_api_sqlite_db(db_path: Path, *, include_aug31_to_nov13: bool = True, corrupt_nan: bool = False, truncate_rows: Optional[int] = None) -> None:
    """Create a deterministic DDL V3.2 SQLite DB mirroring HCMC data (COMPLETE up to 2026-08-30, MISSING_CERES afterwards)."""
    from src.data.vintage_store import (
        ALL_35_VARS,
        CERES_ONLY_8_VARS,
        connect_sqlite,
        init_schema,
        ingest_nasa_power_window,
        seed_historical_csv_vintage_v0,
    )

    sample_csv = PROJECT_ROOT / "artifacts" / "models" / "sample_recent_history_35d.csv"
    df_sample = pd.read_csv(sample_csv)
    base_row = df_sample.iloc[-1]
    from src.config.constants import Config

    vi_map = {k: Config.COLUMN_MAPPING[k] for k in ALL_35_VARS}

    dates = pd.date_range("2026-07-15", "2026-08-30", freq="D")
    if truncate_rows is not None:
        dates = dates[:truncate_rows]

    conn = connect_sqlite(db_path, read_only=False, recursive_triggers=True)
    try:
        init_schema(conn)
        seed_rows = []
        for idx, d in enumerate(dates):
            row = {"DATE": d.strftime("%Y-%m-%d")}
            for var in ALL_35_VARS:
                val = float(base_row[vi_map[var]])
                if var == "PRECTOTCORR":
                    val = float(round((idx % 6) * 2.5, 2))
                elif var in ("ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_LW_DWN"):
                    val = val + (idx % 5) * 0.25
                row[var] = val
            seed_rows.append(row)
        df_seed = pd.DataFrame(seed_rows)
        if corrupt_nan:
            # Place NaN at -32 (outside 30d anchor completeness window, inside 35d API history fetch)
            df_seed.loc[df_seed.index[-32], "T2M"] = float("nan")
        seed_historical_csv_vintage_v0(conn, df=df_seed, ingested_at_utc="2026-11-16T00:00:00Z")

        if include_aug31_to_nov13 and not corrupt_nan and truncate_rows is None:
            ext_dates = pd.date_range("2026-08-31", "2026-11-13", freq="D")

            def _mock_fetcher(**kwargs):
                param_map = {v: {} for v in ALL_35_VARS}
                for idx, d in enumerate(ext_dates):
                    ymd = d.strftime("%Y%m%d")
                    for var in ALL_35_VARS:
                        if var in CERES_ONLY_8_VARS:
                            param_map[var][ymd] = -999.0
                        elif var == "PRECTOTCORR":
                            param_map[var][ymd] = float(round(1.0 + (idx % 7) * 3.0, 2))
                        else:
                            param_map[var][ymd] = float(base_row[vi_map[var]]) + (idx % 4) * 0.1
                return {
                    "header": {"api": {"version": "v2.10.3"}, "sources": ["GEOSIT", "FLASHFLUX"]},
                    "properties": {"parameter": param_map},
                }

            ingest_nasa_power_window(
                conn,
                start_date="2026-08-31",
                end_date="2026-11-13",
                fetcher=_mock_fetcher,
                ingested_at_utc="2026-11-16T01:00:00Z",
            )
    finally:
        conn.close()


def test_predict_reads_from_sqlite_db_effective_zero_past_observed_and_notice(client, tmp_path: Path, monkeypatch):
    """Step 5: /predict reading SQLite DB defaults to 2026-08-30, sets past_observed & future_forecast_notice."""
    import src.api.main as api_main
    from src.data.provider import WeatherHistoryProvider

    db_path = tmp_path / "api_sqlite_fixture.db"
    _seed_round6_api_sqlite_db(db_path)

    provider = WeatherHistoryProvider(backend="sqlite", db_path=db_path)
    monkeypatch.setattr(api_main, "history_provider", provider)

    resp = client.post("/predict", json={"location_id": "HCMC", "run_date": "2026-11-16"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["history_backend"] == "sqlite"
    assert data["data_as_of"] == "2026-08-30"
    assert data["anchor_completeness_regime"] == "COMPLETE_ALL_35"
    assert data["lag_days"] == 78
    assert data["effective_future_days"] == 0
    assert "future_forecast_notice" in data
    assert "effective_future_days = 0" in data["future_forecast_notice"]

    hc = data["serving_model"]["hindcast_and_current"]
    assert len(hc) == 7
    assert len(data["serving_model"]["forecast_future"]) == 0
    assert all(item["status"] == "past_observed" for item in hc)
    assert all(item["observed_mm"] is not None for item in hc)


def test_predict_rejects_incomplete_ceres_anchor_date(client, tmp_path: Path, monkeypatch):
    """Step 5 / Invariant 5: Explicit anchor_date='2026-10-15' (MISSING_CERES_ONLY) is rejected cleanly."""
    import src.api.main as api_main
    from src.data.provider import WeatherHistoryProvider

    db_path = tmp_path / "api_anchor_reject.db"
    _seed_round6_api_sqlite_db(db_path)

    provider = WeatherHistoryProvider(backend="sqlite", db_path=db_path)
    monkeypatch.setattr(api_main, "history_provider", provider)

    resp = client.post(
        "/predict",
        json={"location_id": "HCMC", "anchor_date": "2026-10-15", "run_date": "2026-11-16"},
    )
    assert resp.status_code == 422
    detail = str(resp.json().get("detail", ""))
    assert "INCOMPLETE_RADIATION_AT_ANCHOR" in detail
    assert "MISSING_CERES_ONLY" in detail


@pytest.mark.parametrize("mode", ["insufficient_rows", "nan_meteo"])
def test_predict_sqlite_db_errors_return_503(client, tmp_path: Path, monkeypatch, mode: str):
    """Step 5: Corrupted or <35-row SQLite DB returns HTTP 503 SERVER_DB_DATA_ERROR."""
    import src.api.main as api_main
    from src.data.provider import WeatherHistoryProvider

    db_path = tmp_path / f"api_err_{mode}.db"
    if mode == "insufficient_rows":
        _seed_round6_api_sqlite_db(db_path, include_aug31_to_nov13=False, truncate_rows=20)
    else:
        _seed_round6_api_sqlite_db(db_path, include_aug31_to_nov13=False, corrupt_nan=True)

    provider = WeatherHistoryProvider(backend="sqlite", db_path=db_path)
    monkeypatch.setattr(api_main, "history_provider", provider)

    resp = client.post("/predict", json={"location_id": "HCMC"})
    assert resp.status_code == 503
    assert "SERVER_DB_DATA_ERROR" in str(resp.json().get("detail", ""))


def test_predict_sample_csv_backend_still_works_for_static_tests(client, monkeypatch):
    """Step 5: WEATHER_DB_BACKEND=sample_csv continues to serve sample_recent_history_35d.csv cleanly."""
    import src.api.main as api_main
    from src.data.provider import WeatherHistoryProvider

    provider = WeatherHistoryProvider(backend="sample_csv")
    monkeypatch.setattr(api_main, "history_provider", provider)

    resp = client.post("/predict", json={"province": "TP. Hồ Chí Minh"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["history_backend"] == "local"
    assert data["lag_days"] == 3
    assert data["effective_future_days"] == 4
    assert "future_forecast_notice" not in data


def test_predict_live_vintage_db_if_present(client):
    """When live data/vintage/rainfall_vintage.db exists locally, verify end-to-end /predict behavior."""
    live_db = PROJECT_ROOT / "data" / "vintage" / "rainfall_vintage.db"
    if not live_db.exists():
        pytest.skip("Live gitignored data/vintage/rainfall_vintage.db not present in CI workspace.")

    resp = client.post("/predict", json={"location_id": "HCMC"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["history_backend"] == "sqlite"
    assert data["data_as_of"] == "2026-08-30"
    assert data["effective_future_days"] == 0
    assert "future_forecast_notice" in data
    assert all(item["status"] == "past_observed" for item in data["serving_model"]["hindcast_and_current"])

    resp_bad_anchor = client.post("/predict", json={"location_id": "HCMC", "anchor_date": "2026-10-15"})
    assert resp_bad_anchor.status_code == 422
    assert "INCOMPLETE_RADIATION_AT_ANCHOR" in str(resp_bad_anchor.json().get("detail", ""))
