"""
Serving Hardening & Operational Gate Tests (Step 1 + Step 3 CI)
=================================================================
Tests validation gates (2e), WeatherHistoryProvider backends,
local vs mocked-registry mode switching, and small-sample parity
without requiring large benchmark .npz files or live DagsHub tokens.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))

from src.api.main import app
from src.data.provider import (
    SupabaseWeatherHistoryAdapter,
    TursoWeatherHistoryAdapter,
    WeatherHistoryProvider,
)
from src.inference.predictor import StandaloneRainfallPredictor
from src.inference.validation import ServingValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "models"
SAMPLE_CSV = ARTIFACT_DIR / "sample_recent_history_35d.csv"


@pytest.fixture(scope="module")
def sample_35d_df() -> pd.DataFrame:
    assert SAMPLE_CSV.exists(), f"Missing bundled sample CSV at {SAMPLE_CSV}"
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture(scope="module")
def local_predictor() -> StandaloneRainfallPredictor:
    return StandaloneRainfallPredictor(artifact_dir=ARTIFACT_DIR, mode="local")


def _to_api_records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def test_small_sample_parity_without_npz(sample_35d_df: pd.DataFrame, local_predictor: StandaloneRainfallPredictor):
    """Verify StandaloneRainfallPredictor, direct FeatureBuilder+joblib, and API /predict parity (< 1e-5)."""
    df_sample = sample_35d_df.copy()
    run_date = pd.to_datetime(df_sample["Ngày"]).max().strftime("%Y-%m-%d")

    pred_out = local_predictor.predict(df_sample, run_date=run_date)

    feat3 = local_predictor.fb_f3.build_single_step(df_sample)
    cols3 = [c for c in feat3.columns if c not in [local_predictor.fb_f3.date_col, local_predictor.fb_f3.target_col]]
    X3 = feat3[cols3].values.astype(np.float64)

    feat2 = local_predictor.fb_f2.build_single_step(df_sample)
    cols2 = [c for c in feat2.columns if c not in [local_predictor.fb_f2.date_col, local_predictor.fb_f2.target_col]]
    X2 = feat2[cols2].values.astype(np.float64)

    raw_s35 = np.round(np.clip(local_predictor.model_serving.predict(X3)[0], 0.0, None), 2)
    raw_s30 = np.round(np.clip(local_predictor.model_shadow_val.predict(X2)[0], 0.0, None), 2)
    raw_s31 = np.round(np.clip(local_predictor.model_shadow_contender.predict(X3)[0], 0.0, None), 2)

    assert np.max(np.abs(raw_s35 - np.array(pred_out["serving_model"]["predictions_7d_mm"]))) < 1e-5
    assert np.max(np.abs(raw_s30 - np.array(pred_out["shadow_models"]["shadow_val_champ"]["predictions_7d_mm"]))) < 1e-5
    assert np.max(np.abs(raw_s31 - np.array(pred_out["shadow_models"]["shadow_test_contender"]["predictions_7d_mm"]))) < 1e-5
    assert pred_out["lag_days"] == 0
    assert pred_out["effective_future_days"] == 7

    with TestClient(app) as client:
        resp = client.post(
            "/predict",
            json={"history": _to_api_records(df_sample), "run_date": run_date},
        )
        assert resp.status_code == 200
        api_json = resp.json()
        assert api_json["history_backend"] == "request_payload"
        assert api_json["model_source"] == "local"
        assert api_json["serving_model"]["predictions_7d_mm"] == pred_out["serving_model"]["predictions_7d_mm"]


def test_optional_pyfunc_small_sample_parity(sample_35d_df: pd.DataFrame, local_predictor: StandaloneRainfallPredictor):
    """When MLflow is installed, verify RainfallForecastingPyfuncWrapper raw DF parity (< 1e-5)."""
    pytest.importorskip("mlflow")
    from src.inference.pyfunc import RainfallForecastingPyfuncWrapper

    run_date = pd.to_datetime(sample_35d_df["Ngày"]).max().strftime("%Y-%m-%d")
    wrapper = RainfallForecastingPyfuncWrapper(
        model_path=str(ARTIFACT_DIR / "sc35_xgboost_tweedie_f3.joblib"),
        feature_builder_path=str(ARTIFACT_DIR / "feature_builder_f3.joblib"),
        feature_level=3,
        governance_role="serving",
    )
    wrapper.load_context(None)
    pyfunc_preds = wrapper.predict(None, sample_35d_df, params={"run_date": run_date})[0]
    api_out = local_predictor.predict(sample_35d_df, run_date=run_date)
    diff = float(np.max(np.abs(np.round(pyfunc_preds, 2) - np.array(api_out["serving_model"]["predictions_7d_mm"]))))
    assert diff < 1e-5


def test_validation_rejection_matrix_python_and_api(
    sample_35d_df: pd.DataFrame,
    local_predictor: StandaloneRainfallPredictor,
    monkeypatch,
):
    """Check (2e): Every invalid/stale payload is rejected in Python and returns HTTP 400 on /predict."""
    valid_run_date = pd.to_datetime(sample_35d_df["Ngày"]).max().strftime("%Y-%m-%d")
    stale_run_date = (pd.to_datetime(valid_run_date) + pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    earlier_run_date = (pd.to_datetime(valid_run_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df_gap = sample_35d_df.drop(index=sample_35d_df.index[12]).reset_index(drop=True)
    df_dup = sample_35d_df.copy()
    df_dup.loc[df_dup.index[8], "Ngày"] = df_dup.loc[df_dup.index[7], "Ngày"]

    df_nan = sample_35d_df.copy()
    df_nan.loc[df_nan.index[-1], "Nhiệt độ 2m"] = np.nan

    df_sentinel = sample_35d_df.copy()
    df_sentinel.loc[df_sentinel.index[-2], "Bức xạ sóng ngắn bề mặt"] = -999.0

    df_neg_rain = sample_35d_df.copy()
    df_neg_rain.loc[df_neg_rain.index[-1], "Lượng mưa"] = -3.5

    df_bad_rh = sample_35d_df.copy()
    df_bad_rh.loc[df_bad_rh.index[-1], "Độ ẩm tương đối 2m"] = 145.0

    df_frozen_rad = sample_35d_df.copy()
    df_frozen_rad.iloc[-5:, df_frozen_rad.columns.get_loc("Bức xạ sóng ngắn bề mặt")] = 4.321

    cases = [
        (sample_35d_df.iloc[-29:].copy(), valid_run_date, "INSUFFICIENT_HISTORY"),
        (sample_35d_df.iloc[0:0].copy(), valid_run_date, "INSUFFICIENT_HISTORY"),
        (df_gap, valid_run_date, "DATE_CONTINUITY_VIOLATION"),
        (df_dup, valid_run_date, "DATE_CONTINUITY_VIOLATION"),
        (df_nan, valid_run_date, "INVALID_OR_MISSING_METEO_INPUT"),
        (df_sentinel, valid_run_date, "INVALID_OR_MISSING_METEO_INPUT"),
        (df_neg_rain, valid_run_date, "INVALID_OR_MISSING_METEO_INPUT"),
        (df_bad_rh, valid_run_date, "INVALID_OR_MISSING_METEO_INPUT"),
        (df_frozen_rad, valid_run_date, "STALE_RADIATION_OR_LAG"),
        (sample_35d_df.copy(), stale_run_date, "STALE_RADIATION_OR_LAG"),
        (sample_35d_df.copy(), earlier_run_date, "DATE_CONTINUITY_VIOLATION"),
    ]

    with TestClient(app) as client:
        for bad_df, r_date, expected_code in cases:
            with pytest.raises(ServingValidationError) as exc_info:
                local_predictor.predict(bad_df, run_date=r_date)
            assert exc_info.value.code == expected_code

            records = [] if bad_df.empty else _to_api_records(bad_df)
            resp = client.post("/predict", json={"history": records, "run_date": r_date})
            assert resp.status_code == 400
            err = resp.json()["error"]
            assert err["code"] == "INVALID_CLIENT_HISTORY"
            assert expected_code in err["message"] or err["details"].get("validation_code") == expected_code

        # Omitted history + explicit stale run_date against static sample CSV -> HTTP 400
        monkeypatch.setattr(
            "src.api.main.history_provider",
            WeatherHistoryProvider(backend="sample_csv", sample_csv_path=SAMPLE_CSV),
        )
        resp_stale_omitted = client.post("/predict", json={"run_date": stale_run_date})
        assert resp_stale_omitted.status_code == 400
        err_stale = resp_stale_omitted.json()["error"]
        assert err_stale["code"] == "INVALID_CLIENT_HISTORY"
        assert "STALE_RADIATION_OR_LAG" in err_stale["message"]


def test_weather_history_provider_backends(sample_35d_df: pd.DataFrame):
    """Verify local sample CSV loading, error behavior, and mocked Supabase/Turso REST parsing."""
    local_provider = WeatherHistoryProvider(backend="local", sample_csv_path=SAMPLE_CSV)
    res = local_provider.fetch_recent_history(days=35)
    assert res.backend == "local"
    assert res.is_static_sample is True
    assert len(res.history_df) == 35

    with pytest.raises(ValueError, match="Unsupported WEATHER_DB_BACKEND"):
        WeatherHistoryProvider(backend="unknown_db")

    # Missing config raises explicitly when fallback_to_local=False
    with patch.dict(os.environ, {}, clear=True):
        supa_strict = WeatherHistoryProvider(backend="supabase", fallback_to_local=False)
        with pytest.raises(RuntimeError, match="SUPABASE_URL"):
            supa_strict.fetch_recent_history()

        turso_fallback = WeatherHistoryProvider(
            backend="turso",
            sample_csv_path=SAMPLE_CSV,
            fallback_to_local=True,
        )
        fb_res = turso_fallback.fetch_recent_history(days=30)
        assert fb_res.backend == "local"
        assert len(fb_res.history_df) == 30

    # Mocked Supabase HTTP GET
    mock_rows = _to_api_records(sample_35d_df.iloc[::-1])
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = json.dumps(mock_rows).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        supa = SupabaseWeatherHistoryAdapter(url="https://example.supabase.co", api_key="test-key")
        supa_res = supa.fetch_recent_history(province="TP.HCM", days=35)
        assert supa_res.backend == "supabase"
        assert supa_res.is_static_sample is False
        assert len(supa_res.history_df) == 35
        assert supa_res.history_df["Ngày"].is_monotonic_increasing

    # Mocked Turso pipeline HTTP POST
    cols = list(sample_35d_df.columns)
    turso_payload = {
        "results": [
            {
                "response": {
                    "result": {
                        "cols": [{"name": c} for c in cols],
                        "rows": [
                            [{"value": v} for v in row]
                            for row in sample_35d_df.iloc[::-1].itertuples(index=False, name=None)
                        ],
                    }
                }
            }
        ]
    }
    turso_resp = MagicMock()
    turso_resp.__enter__.return_value = turso_resp
    turso_resp.read.return_value = json.dumps(turso_payload).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=turso_resp):
        turso = TursoWeatherHistoryAdapter(database_url="libsql://demo.turso.io", auth_token="tok")
        assert turso.database_url == "https://demo.turso.io"
        turso_res = turso.fetch_recent_history(province="TP.HCM", days=35)
        assert turso_res.backend == "turso"
        assert turso_res.is_static_sample is False
        assert len(turso_res.history_df) == 35
        assert turso_res.history_df["Ngày"].is_monotonic_increasing


def test_predictor_registry_mode_mocked(sample_35d_df: pd.DataFrame, local_predictor: StandaloneRainfallPredictor):
    """Verify StandaloneRainfallPredictor(mode='registry') fetches alias URIs and preserves parity."""
    with pytest.raises(ValueError, match="Unsupported serving mode"):
        StandaloneRainfallPredictor(artifact_dir=ARTIFACT_DIR, mode="invalid_mode")

    def _make_stub(model, fb):
        stub = MagicMock()
        wrapper = MagicMock()
        wrapper.model = model
        wrapper.feature_builder = fb
        stub.unwrap_python_model.return_value = wrapper
        return stub

    uri_map = {
        "models:/RainfallForecastHCMC@serving": _make_stub(
            local_predictor.model_serving, local_predictor.fb_f3
        ),
        "models:/RainfallForecastHCMC@shadow_val_champ": _make_stub(
            local_predictor.model_shadow_val, local_predictor.fb_f2
        ),
        "models:/RainfallForecastHCMC@shadow_test_contender": _make_stub(
            local_predictor.model_shadow_contender, local_predictor.fb_f3
        ),
    }

    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow_pyfunc = types.ModuleType("mlflow.pyfunc")
    fake_mlflow_pyfunc.load_model = lambda uri: uri_map[uri]
    fake_mlflow.pyfunc = fake_mlflow_pyfunc

    fake_mlflow_exceptions = types.ModuleType("mlflow.exceptions")
    fake_mlflow_exceptions.MlflowException = RuntimeError
    fake_mlflow_tracking = types.ModuleType("mlflow.tracking")
    fake_mlflow_tracking.MlflowClient = MagicMock()
    fake_config_mlops = types.ModuleType("src.config.mlops")
    fake_config_mlops.DEFAULT_REGISTERED_MODEL_NAME = "RainfallForecastHCMC"
    fake_config_mlops.init_dagshub_mlflow = lambda: MagicMock()
    fake_config_mlops.retry_with_backoff = lambda fn, operation_name="": fn()

    with patch.dict(
        sys.modules,
        {
            "mlflow": fake_mlflow,
            "mlflow.pyfunc": fake_mlflow_pyfunc,
            "mlflow.exceptions": fake_mlflow_exceptions,
            "mlflow.tracking": fake_mlflow_tracking,
            "src.config.mlops": fake_config_mlops,
        },
    ):
        reg_pred = StandaloneRainfallPredictor(artifact_dir=ARTIFACT_DIR, mode="registry")
        assert reg_pred.mode == "registry"
        assert set(reg_pred._registry_pyfuncs.keys()) == {
            "serving",
            "shadow_val_champ",
            "shadow_test_contender",
        }
        run_date = pd.to_datetime(sample_35d_df["Ngày"]).max().strftime("%Y-%m-%d")
        assert reg_pred.predict(sample_35d_df, run_date=run_date)["serving_model"]["predictions_7d_mm"] == (
            local_predictor.predict(sample_35d_df, run_date=run_date)["serving_model"]["predictions_7d_mm"]
        )



@pytest.mark.parametrize(
    "ceres_col",
    [
        "Bức xạ sóng ngắn trời quang",
        "Bức xạ trực tiếp pháp tuyến",
        "Bức xạ khuếch tán",
        "Bức xạ quang hợp tổng",
        "Bức xạ quang hợp trời quang",
        "Bức xạ UVA",
        "Bức xạ UVB",
        "Chỉ số UV",
    ],
)
def test_strict_validation_rejects_missing_or_nan_ceres_columns(ceres_col: str):
    """T7: Zero silent mean imputation when any of the 8 CERES radiation columns is missing or NaN."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.inference.predictor import StandaloneRainfallPredictor
    from src.inference.validation import ServingValidationError

    sample_path = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "sample_recent_history_35d.csv"
    df_base = pd.read_csv(sample_path)
    predictor = StandaloneRainfallPredictor(artifact_dir=sample_path.parent, mode="local")

    # 1) NaN in latest row -> must raise ServingValidationError (zero silent fill)
    df_nan = df_base.copy()
    df_nan.loc[df_nan.index[-1], ceres_col] = float("nan")
    with pytest.raises(ServingValidationError) as exc_nan:
        predictor.predict(df_nan)
    assert exc_nan.value.code == "INVALID_OR_MISSING_METEO_INPUT"

    # 2) Dropped CERES column -> must raise ServingValidationError
    df_drop = df_base.drop(columns=[ceres_col])
    with pytest.raises(ServingValidationError) as exc_drop:
        predictor.predict(df_drop)
    assert exc_drop.value.code == "INVALID_OR_MISSING_METEO_INPUT"

    # 3) FastAPI /predict rejects NASA -999 sentinel or NaN with HTTP 400
    with TestClient(app) as api_client:
        df_sentinel = df_base.copy()
        df_sentinel.loc[df_sentinel.index[-1], ceres_col] = -999.0
        resp = api_client.post(
            "/predict",
            json={"history": df_sentinel.to_dict(orient="records"), "province": "TP. Hồ Chí Minh"},
        )
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert err["code"] == "INVALID_CLIENT_HISTORY"
        assert "INVALID_OR_MISSING_METEO_INPUT" in err["message"]
