import sys
import os
import pytest
from datetime import timedelta

sys.path.insert(0, os.path.abspath("."))
from src.evaluation.protocols import get_global_read_guard
from src.data.loader import load_canonical_data
from src.inference.predictor import StandaloneRainfallPredictor

@pytest.fixture(autouse=True)
def unguard_for_standalone_inference():
    g = get_global_read_guard()
    was_active = g.is_active
    g.deactivate()
    yield
    if was_active:
        g.activate()

def test_standalone_predictor_smoke_and_dynamic_lag():
    train_full, val_set, test_set = load_canonical_data()
    predictor = StandaloneRainfallPredictor(artifact_dir="artifacts/models")

    # Slice 35 continuous days ending at 2026-04-20
    sample_history = test_set.iloc[-35:].copy()
    data_as_of_expected = sample_history["Ngày"].iloc[-1].strftime("%Y-%m-%d")

    # Case 1: Run date is 4 days after data_as_of (lag = 4)
    run_date_lag4 = (sample_history["Ngày"].iloc[-1].to_pydatetime().date() + timedelta(days=4)).strftime("%Y-%m-%d")
    res_lag4 = predictor.predict(sample_history, run_date=run_date_lag4)

    assert res_lag4["data_as_of"] == data_as_of_expected
    assert res_lag4["lag_days"] == 4
    assert res_lag4["effective_future_days"] == 3  # 7 - 4 = 3 days!
    assert len(res_lag4["serving_model"]["hindcast_and_current"]) == 4  # h=1,2,3 past_unverified + h=4 today
    assert len(res_lag4["serving_model"]["forecast_future"]) == 3        # h=5,6,7 future_forecast
    assert res_lag4["serving_model"]["hindcast_and_current"][-1]["status"] == "today"

    # Case 2: Run date is 3 days after data_as_of (lag = 3)
    run_date_lag3 = (sample_history["Ngày"].iloc[-1].to_pydatetime().date() + timedelta(days=3)).strftime("%Y-%m-%d")
    res_lag3 = predictor.predict(sample_history, run_date=run_date_lag3)

    assert res_lag3["lag_days"] == 3
    assert res_lag3["effective_future_days"] == 4  # 7 - 3 = 4 days!
    assert len(res_lag3["serving_model"]["hindcast_and_current"]) == 3  # h=1,2 past_unverified + h=3 today
    assert len(res_lag3["serving_model"]["forecast_future"]) == 4        # h=4,5,6,7 future_forecast

    # Verify shadow models output
    assert "shadow_models" in res_lag3
    assert "shadow_val_champ" in res_lag3["shadow_models"]
    assert "shadow_test_contender" in res_lag3["shadow_models"]

    val_champ = res_lag3["shadow_models"]["shadow_val_champ"]
    assert val_champ["scenario_id"] == "SC-30"
    assert len(val_champ["predictions_7d_mm"]) == 7
    assert all(p >= 0.0 for p in val_champ["predictions_7d_mm"])

    test_contender = res_lag3["shadow_models"]["shadow_test_contender"]
    assert test_contender["scenario_id"] == "SC-31"
    assert len(test_contender["predictions_7d_mm"]) == 7
    assert all(p >= 0.0 for p in test_contender["predictions_7d_mm"])

    # Backwards compatibility check
    assert len(res_lag3["shadow_model"]["predictions_7d_mm"]) == 7
    assert all(p >= 0.0 for p in res_lag3["shadow_model"]["predictions_7d_mm"])

    print("ALL STANDALONE PREDICTOR & DYNAMIC LAG TESTS PASSED!")

if __name__ == "__main__":
    test_standalone_predictor_smoke_and_dynamic_lag()

