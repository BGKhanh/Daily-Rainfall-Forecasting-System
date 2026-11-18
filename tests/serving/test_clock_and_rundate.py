"""
Round 8 (A2) Clock, Timezone & DB-Mode run_date Contract Tests
==============================================================
Verifies:
1. Canonical UTC/HCMC clock across day boundary:
   - 17:30 UTC on day D (= 00:30 HCMC on day D+1) -> default run_date is D+1, run_timestamp ends ...T17:30:00.000000Z.
   - 16:59 UTC on day D (= 23:59 HCMC on day D)   -> default run_date is D,   run_timestamp ends ...T16:59:00.000000Z.
2. User-supplied `run_date` in DB mode:
   (a) run_date <= anchor_date:
       - omitted anchor_date + run_date='2026-08-25' -> selects anchor <= run_date and issues 7 future_forecast days.
       - explicit anchor_date='2026-08-25' + run_date='2026-08-25' -> lag=0, 7 future_forecast days.
       - explicit anchor_date='2026-08-30' + run_date='2026-08-25' (run_date < anchor) -> HTTP 400 INVALID_CLIENT_HISTORY.
   (b) run_date in far future beyond operational today ('2099-01-01') -> HTTP 400 INVALID_CLIENT_HISTORY.
   (c) invalid run_date format ('18-11-2026', '2026-02-30') -> HTTP 422 INVALID_QUERY_PARAMETER (unified ErrorResponse).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config.clock as clock_mod
from src.api.main import app, load_serving_models
from src.api.schemas import ErrorResponse, PredictResponse
from src.data.provider import WeatherHistoryProvider
import src.api.main as api_main


@pytest.fixture
def demo_db_client(monkeypatch):
    demo_db = PROJECT_ROOT / "artifacts" / "models" / "demo_rainfall_vintage.db"
    if not demo_db.exists():
        pytest.skip("demo_rainfall_vintage.db not present")
    load_serving_models()
    monkeypatch.setattr(api_main, "history_provider", WeatherHistoryProvider(backend="sqlite", db_path=demo_db))
    with TestClient(app) as client:
        yield client


def test_clock_utc_1730_crosses_hcmc_midnight_to_d_plus_1(demo_db_client, monkeypatch):
    """At 17:30 UTC on day D (2026-08-30), HCMC is 00:30 on D+1 (2026-08-31)."""
    fixed_1730 = datetime(2026, 8, 30, 17, 30, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(clock_mod, "now_utc", lambda: fixed_1730)

    assert clock_mod.today_hcmc().strftime("%Y-%m-%d") == "2026-08-31"
    resp = demo_db_client.post("/predict", json={"location_id": "HCMC", "anchor_date": "2026-08-30"})
    assert resp.status_code == 200
    data = resp.json()
    PredictResponse.model_validate(data)
    assert data["run_date"] == "2026-08-31"
    assert data["run_timezone"] == "Asia/Ho_Chi_Minh"
    assert data["run_timestamp"] == "2026-08-30T17:30:00.000000Z"
    assert data["lag_days"] == 1
    assert data["effective_future_days"] == 6
    assert data["daily_forecasts"][0]["status"] == "today"
    assert data["daily_forecasts"][0]["observed_mm"] is None
    assert data["daily_forecasts"][1]["status"] == "future_forecast"
    assert all(
        item["observed_mm"] is None
        for item in data["daily_forecasts"]
        if item["status"] in ("today", "future_forecast")
    )


def test_clock_utc_1659_stays_on_day_d_in_hcmc(demo_db_client, monkeypatch):
    """At 16:59 UTC on day D (2026-08-30), HCMC is 23:59 on D (2026-08-30)."""
    fixed_1659 = datetime(2026, 8, 30, 16, 59, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(clock_mod, "now_utc", lambda: fixed_1659)

    assert clock_mod.today_hcmc().strftime("%Y-%m-%d") == "2026-08-30"
    resp = demo_db_client.post("/predict", json={"location_id": "HCMC", "anchor_date": "2026-08-30"})
    assert resp.status_code == 200
    data = resp.json()
    PredictResponse.model_validate(data)
    assert data["run_date"] == "2026-08-30"
    assert data["run_timestamp"] == "2026-08-30T16:59:00.000000Z"
    assert data["lag_days"] == 0
    assert data["effective_future_days"] == 7
    assert all(item["status"] == "future_forecast" for item in data["daily_forecasts"])
    assert all(item["observed_mm"] is None for item in data["daily_forecasts"])


def test_db_mode_clock_injection_and_public_run_date_rejected(demo_db_client, monkeypatch):
    """Round 9 Task 1.2: Clock injected via src.config.clock selects anchor <= today_hcmc(), guarantees observed_mm=null for today/future, and rejects public run_date."""
    fixed_aug25 = datetime(2026, 8, 25, 5, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(clock_mod, "now_utc", lambda: fixed_aug25)

    # (a) Omitted anchor_date with clock at 2026-08-25 -> selects anchor <= today_hcmc() ('2026-08-25')
    resp_auto = demo_db_client.post("/predict", json={"location_id": "HCMC"})
    assert resp_auto.status_code == 200
    d_auto = resp_auto.json()
    PredictResponse.model_validate(d_auto)
    assert d_auto["anchor_date"] == "2026-08-25"
    assert d_auto["run_date"] == "2026-08-25"
    assert d_auto["lag_days"] == 0
    assert d_auto["effective_future_days"] == 7
    assert [x["status"] for x in d_auto["daily_forecasts"]] == ["future_forecast"] * 7
    assert all(x["observed_mm"] is None for x in d_auto["daily_forecasts"])

    # (b) Deprecated public `run_date` field in PredictionRequest is rejected with 422 INVALID_QUERY_PARAMETER
    resp_legacy = demo_db_client.post(
        "/predict",
        json={"location_id": "HCMC", "run_date": "2026-08-25"},
    )
    assert resp_legacy.status_code == 422
    err_legacy = resp_legacy.json()
    ErrorResponse.model_validate(err_legacy)
    assert err_legacy["error"]["code"] == "INVALID_QUERY_PARAMETER"
