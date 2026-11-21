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
import src.config.clock as clock_mod
from src.api.main import app, load_serving_models
from src.api.schemas import (
    HealthResponse,
    LeaderboardResponse,
    LocationsResponse,
    ModelsResponse,
    PredictResponse,
    ScoresResponse,
)

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
    HealthResponse.model_validate(data)
    assert data["status"] == "healthy"
    assert data["predictor_loaded"] is True
    assert data["db_role"] in ("local_or_production_db", "bundled_demo_db", "sample_csv")
    assert "default_complete_anchor_date" in data
    assert "data_snapshot_date" in data
    assert "anchor_range" in data
    assert "data_latency" in data
    assert "journal_mode" in data
    if data["db_role"] in ("bundled_demo_db", "local_or_production_db"):
        assert data["default_complete_anchor_date"] == "2026-08-30"
        assert data["data_snapshot_date"] is not None
        assert data["scan_window_start"] is not None
        assert data["scan_window_end"] is not None
        assert isinstance(data["scan_window_total_days"], int) and data["scan_window_total_days"] > 0
        assert data["latest_precip_obs_date"] is not None
        assert data["latest_partial_obs_explanation"] is not None
        assert data["anchor_range"] == {"min": "2001-02-04", "max": "2026-08-30"}
        assert set(data["data_latency"]["groups"].keys()) == {
            "METEO_GEOSIT",
            "RAD_FLASHFLUX",
            "RAD_CERES_SYN1DEG",
        }
        assert data["journal_mode"] == "delete"
    assert data["serving_model"]["scenario_id"] == "SC-35"
    assert len(data["shadow_models"]) == 2
    aliases = [m["alias"] for m in data["shadow_models"]]
    assert "shadow_val_champ" in aliases
    assert "shadow_test_contender" in aliases
    assert len(data["baselines"]) == 2


def test_locations_endpoint(client, tmp_path: Path, monkeypatch):
    """Round 9 Task 1.1: GET /locations reads from SQLite `locations` table (sample_csv fallback only in sample_csv mode)."""
    import src.api.main as api_main
    from src.data.provider import WeatherHistoryProvider

    resp = client.get("/locations")
    assert resp.status_code == 200
    data = resp.json()
    LocationsResponse.model_validate(data)
    assert len(data["locations"]) == 1
    hcmc = data["locations"][0]
    assert hcmc["location_id"] == "HCMC"
    assert hcmc["display_name"] == "Ho Chi Minh City"
    assert hcmc["is_active"] is True

    # In sample_csv mode, falls back to canonical HCMC record
    monkeypatch.setattr(api_main, "history_provider", WeatherHistoryProvider(backend="sample_csv"))
    resp_csv = client.get("/locations")
    assert resp_csv.status_code == 200
    data_csv = resp_csv.json()
    LocationsResponse.model_validate(data_csv)
    assert data_csv["db_role"] == "sample_csv"
    assert data_csv["locations"][0]["location_id"] == "HCMC"

    # In SQLite mode, returns strictly configured locations rows
    from src.data.vintage_store import connect_sqlite, finalize_db_for_read_only, init_schema

    db_path = tmp_path / "locations_test.db"
    conn = connect_sqlite(db_path, read_only=False)
    try:
        init_schema(conn)
        conn.execute(
            """
            INSERT INTO locations (
                location_id, display_name, aliases_json, latitude, longitude,
                meteo_grid_note, rad_grid_note, time_standard, provisional_window_days,
                is_active, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "HANOI",
                "Ha Noi",
                '["HANOI"]',
                21.0285,
                105.8542,
                "MERRA-2 / GEOS-IT 0.5° x 0.625°",
                "CERES SYN1deg / FLASHFlux 1.0° x 1.0°",
                "LST",
                120,
                0,
                "2026-11-18T00:00:00.000000Z",
            ),
        )
        finalize_db_for_read_only(conn)
    finally:
        conn.close()

    monkeypatch.setattr(api_main, "history_provider", WeatherHistoryProvider(backend="sqlite", db_path=db_path))
    resp_sqlite = client.get("/locations")
    assert resp_sqlite.status_code == 200
    locs = {item["location_id"]: item for item in resp_sqlite.json()["locations"]}
    assert set(locs.keys()) == {"HCMC", "HANOI"}
    assert locs["HANOI"]["is_active"] is False


def test_models_endpoint(client):
    """Task 5: /models returns 3 ML models + 2 baselines + sha256_checksums including demo_rainfall_vintage.db."""
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    ModelsResponse.model_validate(data)
    for key in (
        "serving",
        "shadow_val_champ",
        "shadow_test_contender",
        "baseline_persistence",
        "baseline_climatology_doy",
    ):
        assert key in data["models"], f"Missing {key} in /models response"
    assert "demo_rainfall_vintage.db" in data["sha256_checksums"]
    assert data == client.get("/models/info").json()


def test_models_info_endpoint(client):
    resp = client.get("/models/info")
    assert resp.status_code == 200
    data = resp.json()
    ModelsResponse.model_validate(data)
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
    LeaderboardResponse.model_validate(data)
    assert "test_set_exposure_note" in data
    assert "config_note" in data
    assert "Test set" in data["test_set_exposure_note"]
    assert "Config A" in data["config_note"]
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
    PredictResponse.model_validate(data)
    assert data["run_timezone"] == "Asia/Ho_Chi_Minh"
    assert len(data["daily_forecasts"]) == 7
    for day_item in data["daily_forecasts"]:
        assert set(day_item["predictions"].keys()) == {
            "SC-35",
            "SC-30",
            "SC-31",
            "BASELINE_CLIMATOLOGY_DOY",
            "BASELINE_PERSISTENCE",
        }
    assert "serving_model" in data
    assert "shadow_models" in data
    assert len(data["serving_model"]["predictions_7d_mm"]) == 7
    assert data["location"]["province"] == "Ho Chi Minh City"
    assert "effective_future_days" in data
    assert "lag_days" in data


def test_predict_dynamic_horizon_lag4(client, monkeypatch):
    # Inject clock 4 days after latest complete anchor (2026-08-30 -> 2026-09-03) via src.config.clock
    from datetime import timezone

    resp_base = client.post("/predict", json={})
    assert resp_base.status_code == 200
    data_as_of_str = resp_base.json()["data_as_of"]
    d_as_of = datetime.strptime(data_as_of_str, "%Y-%m-%d").date()

    run_date_lag4 = d_as_of + timedelta(days=4)
    fixed_dt = datetime(run_date_lag4.year, run_date_lag4.month, run_date_lag4.day, 5, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(clock_mod, "now_utc", lambda: fixed_dt)

    if resp_base.json()["db_role"] == "sample_csv":
        sample_csv = PROJECT_ROOT / "artifacts" / "models" / "sample_recent_history_35d.csv"
        df_s = pd.read_csv(sample_csv)
        df_s["Ngày"] = pd.to_datetime(df_s["Ngày"]).dt.strftime("%Y-%m-%d")
        payload = {"history": df_s.to_dict(orient="records")}
    else:
        payload = {"anchor_date": data_as_of_str}

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["lag_days"] == 4
    assert data["effective_future_days"] == 3
    assert len(data["serving_model"]["hindcast_and_current"]) == 4
    assert len(data["serving_model"]["forecast_future"]) == 3
    assert data["serving_model"]["hindcast_and_current"][-1]["status"] == "today"
    for item in data["daily_forecasts"]:
        if item["status"] in ("today", "future_forecast"):
            assert item["observed_mm"] is None

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
        {"location_id": "HANOI"},
        {"location_id": "HCMC", "province": "Hà Nội"},
    ],
)
def test_predict_rejects_non_hcmc_locations_with_422(client, payload):
    """Mục 4 & 6 + Round 8 A7: API is locked to HCMC; any other location returns unified 422 ErrorResponse."""
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "UNSUPPORTED_LOCATION"


@pytest.mark.parametrize(
    "alias",
    [
        "HCMC",
        "hcmc",
        "HO_CHI_MINH",
        "HOCHIMINH",
        "Ho Chi Minh City",
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
    assert loc["province"] == "Ho Chi Minh City"
    assert loc["latitude"] == pytest.approx(10.7769)
    assert loc["longitude"] == pytest.approx(106.7009)


def test_predict_server_db_data_error_returns_503(client, monkeypatch):
    """Mục 4.6: Missing/NaN meteo data from server history_provider returns HTTP 503 (not 400)."""
    import pandas as pd
    from datetime import timezone
    from unittest.mock import MagicMock
    import src.api.main as api_main
    from src.data.provider import WeatherHistoryResult

    monkeypatch.setattr(clock_mod, "now_utc", lambda: datetime(2026, 5, 3, 5, 0, 0, tzinfo=timezone.utc))

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

    # Server-sourced corrupt data (request.history is None) -> HTTP 503 unified ErrorResponse
    resp_server = client.post("/predict", json={"location_id": "HCMC"})
    assert resp_server.status_code == 503
    assert resp_server.json()["error"]["code"] == "SERVER_DB_DATA_ERROR"

    # Client-supplied corrupt history (request.history is not None) -> HTTP 400 unified ErrorResponse
    corrupt_payload = corrupt_df.copy()
    corrupt_payload["Ngày"] = pd.to_datetime(corrupt_payload["Ngày"]).dt.strftime("%Y-%m-%d")
    corrupt_records = corrupt_payload.to_dict(orient="records")
    corrupt_records[-1]["Lượng mưa"] = None
    resp_client = client.post(
        "/predict",
        json={
            "location_id": "HCMC",
            "history": corrupt_records,
        },
    )
    assert resp_client.status_code == 400
    err_client = resp_client.json()["error"]
    assert err_client["code"] == "INVALID_CLIENT_HISTORY"
    assert err_client["details"].get("validation_code") == "INVALID_OR_MISSING_METEO_INPUT"


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
    from datetime import timezone
    import src.api.main as api_main
    from src.data.provider import WeatherHistoryProvider

    monkeypatch.setattr(clock_mod, "now_utc", lambda: datetime(2026, 11, 16, 5, 0, 0, tzinfo=timezone.utc))

    db_path = tmp_path / "api_sqlite_fixture.db"
    _seed_round6_api_sqlite_db(db_path)

    provider = WeatherHistoryProvider(backend="sqlite", db_path=db_path)
    monkeypatch.setattr(api_main, "history_provider", provider)

    resp = client.post("/predict", json={"location_id": "HCMC"})
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
    from datetime import timezone
    import src.api.main as api_main
    from src.data.provider import WeatherHistoryProvider

    monkeypatch.setattr(clock_mod, "now_utc", lambda: datetime(2026, 11, 16, 5, 0, 0, tzinfo=timezone.utc))

    db_path = tmp_path / "api_anchor_reject.db"
    _seed_round6_api_sqlite_db(db_path)

    provider = WeatherHistoryProvider(backend="sqlite", db_path=db_path)
    monkeypatch.setattr(api_main, "history_provider", provider)

    resp = client.post(
        "/predict",
        json={"location_id": "HCMC", "anchor_date": "2026-10-15"},
    )
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "INCOMPLETE_RADIATION_AT_ANCHOR"
    assert err["details"].get("completeness_regime") == "MISSING_CERES_ONLY"

    resp_missing = client.post(
        "/predict",
        json={"location_id": "HCMC", "anchor_date": "1999-01-01"},
    )
    assert resp_missing.status_code == 422
    assert resp_missing.json()["error"]["code"] == "ANCHOR_DATE_NOT_FOUND"


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
    assert resp.json()["error"]["code"] == "SERVER_DB_DATA_ERROR"


def test_predict_sample_csv_backend_still_works_for_static_tests(client, monkeypatch):
    """Step 5: WEATHER_DB_BACKEND=sample_csv continues to serve sample_recent_history_35d.csv cleanly."""
    import src.api.main as api_main
    from src.data.provider import WeatherHistoryProvider

    provider = WeatherHistoryProvider(backend="sample_csv")
    monkeypatch.setattr(api_main, "history_provider", provider)

    resp = client.post("/predict", json={"province": "TP. Hồ Chí Minh"})
    assert resp.status_code == 200
    data = resp.json()
    PredictResponse.model_validate(data)
    assert data["history_backend"] == "local"
    assert data["db_role"] == "sample_csv"
    assert data["lag_days"] == 3
    assert data["effective_future_days"] == 4
    assert data["future_forecast_notice"] is None


def test_predict_live_vintage_db_if_present(client):
    """Verify end-to-end /predict behavior when live or bundled demo SQLite DB exists."""
    live_db = PROJECT_ROOT / "data" / "vintage" / "rainfall_vintage.db"
    demo_db = PROJECT_ROOT / "artifacts" / "models" / "demo_rainfall_vintage.db"
    if not (live_db.exists() or demo_db.exists()):
        pytest.skip("Neither rainfall_vintage.db nor demo_rainfall_vintage.db present.")

    resp = client.post("/predict", json={"location_id": "HCMC"})
    assert resp.status_code == 200
    data = resp.json()
    PredictResponse.model_validate(data)
    assert data["history_backend"] == "sqlite"
    assert data["anchor_date"] == "2026-08-30"
    assert data["anchor_split_regime"] == "POST_TEST"
    assert data["data_as_of"] == "2026-08-30"
    assert data["effective_future_days"] == 0
    assert data["forecast_horizon_mode"] == "retrospective_hindcast_only"
    assert data["future_forecast_notice"] is not None
    assert all(item["status"] == "past_observed" for item in data["serving_model"]["hindcast_and_current"])

    resp_bad_anchor = client.post("/predict", json={"location_id": "HCMC", "anchor_date": "2026-10-15"})
    assert resp_bad_anchor.status_code == 422
    assert resp_bad_anchor.json()["error"]["code"] == "INCOMPLETE_RADIATION_AT_ANCHOR"


def test_scores_endpoint(client):
    """Round 8 A1.5: /scores serves DB-backed POST_TEST scores, 95% Block Bootstrap CIs, and all 3 ML pairs."""
    resp = client.get("/scores")
    assert resp.status_code == 200
    data = resp.json()
    ScoresResponse.model_validate(data)
    assert data["eval_split_regime"] == "POST_TEST"
    assert data["db_role"] in ("local_or_production_db", "bundled_demo_db")
    assert data["n_origins"] == 122
    assert data["n_forecasts_per_model"] == 854
    assert data["total_scored_rows"] == 4270
    assert data["origin_min"] == "2026-05-01"
    assert data["origin_max"] == "2026-08-30"
    assert data["block_bootstrap"]["block_length"] == 7
    assert data["block_bootstrap"]["n_blocks"] == 18
    assert data["block_bootstrap"]["n_bootstrap"] == 2000

    for m_id in ("SC-31", "SC-30", "SC-35", "BASELINE_CLIMATOLOGY_DOY", "BASELINE_PERSISTENCE"):
        assert m_id in data["models"]
        m_entry = data["models"][m_id]
        assert set(m_entry["by_horizon_vs_latest"].keys()) == {f"h{h}" for h in range(1, 8)}
        ci_95 = m_entry["confidence_intervals_95pct"]
        assert ci_95["block_length"] == 7
        assert ci_95["n_blocks"] == 18
        for metric in ("MAE", "RMSE", "MASE"):
            assert ci_95["overall"][metric]["ci_lower"] <= ci_95["overall"][metric]["ci_upper"]
        for h in range(1, 8):
            assert f"h{h}" in ci_95["by_horizon"]

    for pair_key in (
        "SC-35 - BASELINE_CLIMATOLOGY_DOY",
        "SC-31 - BASELINE_CLIMATOLOGY_DOY",
        "SC-30 - BASELINE_CLIMATOLOGY_DOY",
        "SC-35 - SC-31",
        "SC-35 - SC-30",
        "SC-30 - SC-31",
    ):
        assert pair_key in data["pairwise_delta_mae_ci"], f"Missing pairwise comparison {pair_key}"
        assert data["pairwise_delta_mae_ci"][pair_key]["decision_label"]
    assert data["pairwise_delta_mae_ci"]["SC-35 - BASELINE_CLIMATOLOGY_DOY"]["significant_at_95pct"] is True
    assert data["pairwise_delta_mae_ci"]["SC-30 - SC-31"]["decision_label"] == "Borderline, not adjusted for multiple comparisons"
    assert data["pairwise_delta_mae_ci"]["SC-35 - SC-31"]["decision_label"] == "Not statistically significant (95% CI includes 0)"
    assert data["pairwise_delta_mae_ci"]["SC-35 - BASELINE_CLIMATOLOGY_DOY"]["decision_label"] == "Statistically significant (95% CI excludes 0)"
    assert set(data["block_length_sensitivity"].keys()) == {"7", "14", "21"}
    for blk_key in ("7", "14", "21"):
        assert "SC-30 - SC-31" in data["block_length_sensitivity"][blk_key]["pairwise_delta_mae"]
    assert set(data["methodological_notes"].keys()) == {
        "effective_sample_size",
        "first_vs_latest_equality",
        "seasonal_regime_note",
    }

    # Explicit retrospective filter matches 122 origins; invalid params return unified 422 ErrorResponse
    resp_retro = client.get("/scores?eval_split_regime=POST_TEST&is_prospective=0")
    assert resp_retro.status_code == 200
    assert resp_retro.json()["n_origins"] == 122

    bad_split = client.get("/scores?eval_split_regime=INVALID_SPLIT")
    assert bad_split.status_code == 422
    assert bad_split.json()["error"]["code"] == "INVALID_QUERY_PARAMETER"

    bad_prosp = client.get("/scores?is_prospective=9")
    assert bad_prosp.status_code == 422
    assert bad_prosp.json()["error"]["code"] == "INVALID_QUERY_PARAMETER"


def test_english_only_api_strings(client):
    """Round 9 Task 1.3: Every user-facing API string is English (no Vietnamese script characters)."""
    import re

    vi_re = re.compile(
        r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
    )

    h = client.get("/health").json()
    assert not vi_re.search(h.get("latest_partial_obs_explanation") or "")

    locs = client.get("/locations").json()
    for item in locs["locations"]:
        assert not vi_re.search(item["display_name"])

    pred = client.post("/predict", json={"location_id": "HCMC"}).json()
    assert not vi_re.search(pred.get("future_forecast_notice") or "")
    assert not vi_re.search(pred["location"]["province"])

    sc = client.get("/scores").json()
    for note in sc["methodological_notes"].values():
        assert not vi_re.search(note)
    for pair_val in sc["pairwise_delta_mae_ci"].values():
        assert not vi_re.search(pair_val.get("decision_label") or "")

    lb = client.get("/benchmark/leaderboard").json()
    assert not vi_re.search(lb["test_set_exposure_note"])
    assert not vi_re.search(lb["config_note"])

    err = client.post("/predict", json={"location_id": "HANOI"}).json()["error"]
    assert not vi_re.search(err["message"])


def test_openapi_snapshot_matches_live_app():
    """Round 8/9: Locked OpenAPI snapshot matches live FastAPI schema, hides /models/info, covers all 7 schemas."""
    import json

    snapshot_path = PROJECT_ROOT / "docs" / "openapi_snapshot.json"
    assert snapshot_path.exists(), f"Missing {snapshot_path}"
    expected_spec = json.loads(snapshot_path.read_text(encoding="utf-8"))
    live_spec = app.openapi()
    assert live_spec == expected_spec
    assert {"/locations", "/health", "/models", "/predict", "/scores", "/benchmark/leaderboard"}.issubset(
        live_spec["paths"].keys()
    )
    assert "/models/info" not in live_spec["paths"]

    schemas = live_spec.get("components", {}).get("schemas", {})
    for schema_name in (
        "LocationsResponse",
        "HealthResponse",
        "ModelsResponse",
        "PredictResponse",
        "ScoresResponse",
        "LeaderboardResponse",
        "ErrorResponse",
    ):
        assert schema_name in schemas, f"Missing {schema_name} in OpenAPI components.schemas"
    assert "run_date" not in schemas["PredictionRequest"].get("properties", {})


@pytest.mark.parametrize(
    "import_stmt",
    [
        "import src.api.main",
        "import src.inference.validation",
        "import src.data.provider",
        "import src.data.vintage_store",
    ],
)
def test_fresh_process_imports_have_no_circular_dependency(import_stmt: str):
    """Ensure every serving entrypoint imports cleanly in a fresh Python interpreter without circular imports."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c", import_stmt],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"{import_stmt} failed:\n{proc.stderr}"


def test_observations_leading_and_trailing_partial_periods_weekly_and_monthly(monkeypatch):
    """Priority 1: verify leading and trailing incomplete periods preserve true calendar_days_count."""
    from datetime import datetime as _dt, timezone as _tz
    import src.config.clock as clock_mod

    monkeypatch.setattr(clock_mod, "now_utc", lambda: _dt(2026, 11, 19, 8, 0, tzinfo=_tz.utc))
    with TestClient(app) as client:
        # Start Wednesday 2026-08-26 -> leading week has 5 published days out of 7 calendar days;
        # end Thursday 2026-11-19 -> trailing week has 0 < pub < 7 (or unpublished after 11-15).
        w_res = client.get("/observations?granularity=weekly&start_date=2026-08-26&end_date=2026-11-12")
        assert w_res.status_code == 200
        w_series = w_res.json()["series"]
        lead_w = w_series[0]
        trail_w = w_series[-1]
        assert lead_w["calendar_days_count"] == 7
        assert 0 < lead_w["published_days_count"] < lead_w["calendar_days_count"]
        assert lead_w["published_days_count"] == 5
        assert trail_w["calendar_days_count"] == 7
        assert 0 < trail_w["published_days_count"] < trail_w["calendar_days_count"]
        assert trail_w["published_days_count"] == 4

        # Monthly: 2026-08-26..2026-11-12 -> leading August has 6/31 days, trailing November has 12/30 days.
        m_res = client.get("/observations?granularity=monthly&start_date=2026-08-26&end_date=2026-11-12")
        assert m_res.status_code == 200
        m_series = m_res.json()["series"]
        lead_m = m_series[0]
        trail_m = m_series[-1]
        assert lead_m["calendar_days_count"] == 31
        assert lead_m["published_days_count"] == 6
        assert trail_m["calendar_days_count"] == 30
        assert trail_m["published_days_count"] == 12
