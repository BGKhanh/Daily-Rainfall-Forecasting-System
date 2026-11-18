"""
Operational Backend Verification Suite (Step 1.3)
=================================================
Runs 5 empirical operational verification checks and saves results to
reports/mlops/operational_verification_report.json:

  (a) Array parity (< 1e-5) vs canonical Val (724) and Test (2,184) .npz
      forecast matrices (read-only), with explicit Config A / Config B notes.
  (b) Raw 35-day DataFrame parity (< 1e-5) between MLflow Pyfunc wrappers
      and StandaloneRainfallPredictor across 10 rolling origins.
  (c) 35-day sample operational check across lag=0, lag=3, lag=4 horizons.
  (d) Latency breakdown (mean/P50/P95 ms for validation, features, predict, total)
      and process RSS (MB) via psutil.
  (e) Empirical rejection suite verifying Python ValueError codes and FastAPI
      HTTP 400 Bad Request responses across all invalid/stale operational inputs.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.loaders import load_all_forecast_matrices
from src.api.main import app as fastapi_app
from src.evaluation.metrics import evaluate_point_forecast_metrics
from src.evaluation.protocols import build_purged_direct_samples, load_canonical_data
from src.inference.predictor import StandaloneRainfallPredictor
from src.inference.pyfunc import RainfallForecastingPyfuncWrapper
from src.inference.validation import ServingValidationError, validate_serving_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("operational_checks")

REPORT_PATH = PROJECT_ROOT / "reports" / "mlops" / "operational_verification_report.json"
PARITY_TOL = 1e-5


def _build_wrapper(
    model_file: str,
    fb_file: str,
    feature_level: int,
    governance_role: str,
) -> RainfallForecastingPyfuncWrapper:
    artifact_dir = PROJECT_ROOT / "artifacts" / "models"
    wrapper = RainfallForecastingPyfuncWrapper(
        model_path=str(artifact_dir / model_file),
        feature_builder_path=str(artifact_dir / fb_file),
        feature_level=feature_level,
        governance_role=governance_role,
    )
    wrapper.load_context(None)
    return wrapper


def run_check_a_array_parity(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Check (a): Validate Pyfunc array parity vs canonical Val (724) and Test (2,184) NPZ."""
    dev_train, val_set, test_set = load_canonical_data(return_dev_train=True)
    train_full = pd.concat([dev_train, val_set], ignore_index=True)

    forecasts_val, y_val_npz, _ = load_all_forecast_matrices("data/processed/results", partition="validation")
    forecasts_test, y_test_npz, _ = load_all_forecast_matrices("data/processed/results", partition="test")

    wrappers = {
        "SC-35": _build_wrapper("sc35_xgboost_tweedie_f3.joblib", "feature_builder_f3.joblib", 3, "serving"),
        "SC-30": _build_wrapper("sc30_lightgbm_tweedie_f2.joblib", "feature_builder_f2.joblib", 2, "shadow_val_champ"),
        "SC-31": _build_wrapper("sc31_lightgbm_tweedie_f3.joblib", "feature_builder_f3.joblib", 3, "shadow_test_contender"),
    }

    fb_f2 = wrappers["SC-30"].feature_builder
    fb_f3 = wrappers["SC-35"].feature_builder

    # Build purged validation & test feature matrices strictly using causal context
    def _extract_matrix(fb, target_df: pd.DataFrame, context_df: pd.DataFrame) -> np.ndarray:
        feat_df = fb.transform(target_df, context_df=context_df, drop_warmup=False)
        feat_cols = [c for c in feat_df.columns if c not in [fb.date_col, fb.target_col]]
        X_mat, _, _ = build_purged_direct_samples(feat_df, feat_cols, horizon=7)
        return X_mat

    X_val_by_level = {
        2: _extract_matrix(fb_f2, val_set, dev_train),
        3: _extract_matrix(fb_f3, val_set, dev_train),
    }
    X_test_by_level = {
        2: _extract_matrix(fb_f2, test_set, train_full),
        3: _extract_matrix(fb_f3, test_set, train_full),
    }

    assert X_val_by_level[3].shape[0] == 724
    assert X_test_by_level[3].shape[0] == 2184

    meta_models = {
        entry["scenario_id"]: entry
        for entry in metadata.get("models", {}).values()
        if isinstance(entry, dict) and entry.get("scenario_id")
    }

    results: Dict[str, Any] = {}
    all_passed = True

    for sc_id, wrapper in wrappers.items():
        level_int = int(wrapper.feature_level)
        level = f"F{level_int}"
        pred_val = wrapper.predict(None, X_val_by_level[level_int])
        pred_test = wrapper.predict(None, X_test_by_level[level_int])

        val_diff = float(np.max(np.abs(pred_val - forecasts_val[sc_id])))
        test_diff_npz = float(np.max(np.abs(pred_test - forecasts_test[sc_id])))

        val_metrics = evaluate_point_forecast_metrics(y_val_npz, pred_val)
        test_metrics = evaluate_point_forecast_metrics(y_test_npz, pred_test)

        # SC-35 is trained under Config B (matches both Val & Test .npz < 1e-5).
        # SC-30 and SC-31 are Dev-Train Config A artifacts (match Val .npz < 1e-5; Test
        # is evaluated against metadata Config A Test MAE while annotating Config B .npz).
        val_parity_pass = val_diff < PARITY_TOL
        if sc_id == "SC-35":
            test_check_pass = test_diff_npz < PARITY_TOL
            test_regime_note = "Config_B_FullTrain_Refit (matches canonical Test NPZ < 1e-5)"
        else:
            bench = meta_models[sc_id]["benchmark_metrics"]
            expected_test_mae = float(bench.get("test_MAE_config_a", bench["test_MAE"]))
            mae_gap = abs(float(test_metrics["MAE"]) - expected_test_mae)
            test_check_pass = mae_gap < 1e-4
            test_regime_note = (
                f"Config_A_DevTrain_Frozen (test_MAE={test_metrics['MAE']:.6f} matches "
                f"metadata Config A={expected_test_mae:.6f}; canonical Test NPZ holds Config B)"
            )

        if not (val_parity_pass and test_check_pass):
            all_passed = False

        results[sc_id] = {
            "feature_level": level,
            "val_origins": int(pred_val.shape[0]),
            "val_max_abs_diff_vs_npz": val_diff,
            "val_parity_pass": val_parity_pass,
            "val_metrics": {k: round(float(val_metrics[k]), 6) for k in ("MAE", "MSE", "RMSE", "MASE")},
            "test_origins": int(pred_test.shape[0]),
            "test_max_abs_diff_vs_npz": test_diff_npz,
            "test_regime_note": test_regime_note,
            "test_check_pass": test_check_pass,
            "test_metrics": {k: round(float(test_metrics[k]), 6) for k in ("MAE", "MSE", "RMSE", "MASE")},
        }

    return {"passed": all_passed, "tolerance": PARITY_TOL, "models": results}


def run_check_b_raw_df_parity(predictor: StandaloneRainfallPredictor) -> Dict[str, Any]:
    """Check (b): 10-origin raw 35-day DataFrame parity between Pyfunc and StandaloneRainfallPredictor."""
    _, _, test_set = load_canonical_data(return_dev_train=True)
    test_df = test_set.copy().reset_index(drop=True)

    wrappers = {
        "serving": _build_wrapper("sc35_xgboost_tweedie_f3.joblib", "feature_builder_f3.joblib", 3, "serving"),
        "shadow_val_champ": _build_wrapper("sc30_lightgbm_tweedie_f2.joblib", "feature_builder_f2.joblib", 2, "shadow_val_champ"),
        "shadow_test_contender": _build_wrapper("sc31_lightgbm_tweedie_f3.joblib", "feature_builder_f3.joblib", 3, "shadow_test_contender"),
    }

    # Pick 10 evenly spaced end indices with at least 35 prior rows in test_df
    indices = np.linspace(40, len(test_df) - 1, 10, dtype=int)
    max_rounded_diff = 0.0
    max_unrounded_diff = 0.0
    origin_dates_checked: List[str] = []

    for idx in indices:
        window_df = test_df.iloc[idx - 34 : idx + 1].copy().reset_index(drop=True)
        origin_str = pd.Timestamp(window_df["Ngày"].iloc[-1]).strftime("%Y-%m-%d")
        origin_dates_checked.append(origin_str)

        api_out = predictor.predict(window_df, run_date=origin_str)

        api_preds = {
            "serving": np.array(api_out["serving_model"]["predictions_7d_mm"], dtype=np.float64),
            "shadow_val_champ": np.array(api_out["shadow_models"]["shadow_val_champ"]["predictions_7d_mm"], dtype=np.float64),
            "shadow_test_contender": np.array(api_out["shadow_models"]["shadow_test_contender"]["predictions_7d_mm"], dtype=np.float64),
        }

        # Direct feature-matrix check matching Standalone unrounded output vs Pyfunc raw DF predict
        for key, wrapper in wrappers.items():
            pyfunc_raw = wrapper.predict(None, window_df, params={"run_date": origin_str})[0]
            pyfunc_rounded = np.round(pyfunc_raw, 2)
            rounded_diff = float(np.max(np.abs(pyfunc_rounded - api_preds[key])))
            max_rounded_diff = max(max_rounded_diff, rounded_diff)

            # Compare unrounded Pyfunc vs Standalone underlying estimator directly
            feat_df = wrapper.feature_builder.build_single_step(window_df)
            feat_cols = [c for c in feat_df.columns if c not in [wrapper.feature_builder.date_col, wrapper.feature_builder.target_col]]
            X = feat_df[feat_cols].values.astype(np.float64)
            est_model = {
                "serving": predictor.model_serving,
                "shadow_val_champ": predictor.model_shadow_val,
                "shadow_test_contender": predictor.model_shadow_contender,
            }[key]
            est_raw = np.clip(est_model.predict(X)[0], 0.0, None)
            max_unrounded_diff = max(max_unrounded_diff, float(np.max(np.abs(pyfunc_raw - est_raw))))

    passed = (max_rounded_diff < PARITY_TOL) and (max_unrounded_diff < PARITY_TOL)
    return {
        "passed": passed,
        "origins_count": len(indices),
        "origin_dates": origin_dates_checked,
        "max_unrounded_abs_diff": max_unrounded_diff,
        "max_rounded_abs_diff": max_rounded_diff,
        "tolerance": PARITY_TOL,
    }


def run_check_c_sample_operational(predictor: StandaloneRainfallPredictor) -> Dict[str, Any]:
    """Check (c): Validate 35-day sample operational prediction under lag=0, lag=3, lag=4."""
    sample_path = PROJECT_ROOT / "artifacts" / "models" / "sample_recent_history_35d.csv"
    df_sample = pd.read_csv(sample_path)
    data_as_of = pd.to_datetime(df_sample["Ngày"]).max().normalize()

    scenarios = []
    all_passed = True
    for expected_lag in (0, 3, 4):
        run_d = (data_as_of + pd.Timedelta(days=expected_lag)).strftime("%Y-%m-%d")
        res = predictor.predict(df_sample, run_date=run_d)
        hindcast = res["serving_model"]["hindcast_and_current"]
        future = res["serving_model"]["forecast_future"]
        statuses = [item["status"] for item in hindcast + future]

        ok = (
            res["data_as_of"] == data_as_of.strftime("%Y-%m-%d")
            and res["lag_days"] == expected_lag
            and res["effective_future_days"] == max(0, 7 - expected_lag)
            and len(hindcast) + len(future) == 7
            and len(future) == max(0, 7 - expected_lag)
        )
        if not ok:
            all_passed = False

        scenarios.append(
            {
                "run_date": run_d,
                "data_as_of": res["data_as_of"],
                "lag_days": res["lag_days"],
                "effective_future_days": res["effective_future_days"],
                "horizon_statuses": statuses,
                "passed": ok,
            }
        )

    return {"passed": all_passed, "sample_csv": str(sample_path), "scenarios": scenarios}


def run_check_d_latency_and_memory(predictor: StandaloneRainfallPredictor, iterations: int = 30) -> Dict[str, Any]:
    """Check (d): Measure latency (validation/feature/predict/total) and process RSS in MB."""
    df_sample = pd.read_csv(PROJECT_ROOT / "artifacts" / "models" / "sample_recent_history_35d.csv")
    run_d = (pd.to_datetime(df_sample["Ngày"]).max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    process = psutil.Process(os.getpid())
    rss_before_mb = process.memory_info().rss / (1024 * 1024)

    # Warmup
    predictor.predict(df_sample, run_date=run_d)

    val_times: List[float] = []
    feat_times: List[float] = []
    pred_times: List[float] = []
    total_times: List[float] = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        ctx = validate_serving_input(df_sample, run_date=run_d, feature_builder=predictor.fb_f3, enforce_lag_limit=True)
        t1 = time.perf_counter()

        feat_df3 = predictor.fb_f3.build_single_step(ctx.df)
        cols3 = [c for c in feat_df3.columns if c not in [predictor.fb_f3.date_col, predictor.fb_f3.target_col]]
        X3 = feat_df3[cols3].values.astype(np.float64)
        feat_df2 = predictor.fb_f2.build_single_step(ctx.df)
        cols2 = [c for c in feat_df2.columns if c not in [predictor.fb_f2.date_col, predictor.fb_f2.target_col]]
        X2 = feat_df2[cols2].values.astype(np.float64)
        t2 = time.perf_counter()

        _ = predictor.model_serving.predict(X3)
        _ = predictor.model_shadow_val.predict(X2)
        _ = predictor.model_shadow_contender.predict(X3)
        t3 = time.perf_counter()

        _ = predictor.predict(df_sample, run_date=run_d)
        t4 = time.perf_counter()

        val_times.append((t1 - t0) * 1000.0)
        feat_times.append((t2 - t1) * 1000.0)
        pred_times.append((t3 - t2) * 1000.0)
        total_times.append((t4 - t3) * 1000.0)

    rss_after_mb = process.memory_info().rss / (1024 * 1024)

    # Also measure a clean standalone FastAPI/Predictor process RSS without torch/npz loaded
    import subprocess

    probe_code = "\n".join(
        [
            "import os, sys, psutil",
            f"sys.path.insert(0, {repr(str(PROJECT_ROOT))})",
            "from fastapi.testclient import TestClient",
            "from src.api.main import app",
            "with TestClient(app) as client:",
            "    for _ in range(10):",
            "        client.post('/predict', json={'province': 'TP. Hồ Chí Minh'})",
            "rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)",
            "print(f'STANDALONE_RSS_MB={rss_mb:.2f}')",
        ]
    )
    standalone_rss_mb = None
    try:
        proc_out = subprocess.check_output(
            [sys.executable, "-c", probe_code],
            text=True,
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.STDOUT,
        )
        for line in proc_out.splitlines():
            if line.startswith("STANDALONE_RSS_MB="):
                standalone_rss_mb = round(float(line.split("=", 1)[1]), 2)
    except Exception as exc:
        logger.warning("Failed standalone RSS subprocess probe: %s", exc)

    def _stats(vals: List[float]) -> Dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean_ms": round(float(np.mean(arr)), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "max_ms": round(float(np.max(arr)), 3),
        }

    return {
        "iterations": iterations,
        "standalone_serving_rss_mb": standalone_rss_mb,
        "full_verification_process_rss_before_warmup_mb": round(rss_before_mb, 2),
        "full_verification_process_rss_after_runs_mb": round(rss_after_mb, 2),
        "validation_ms": _stats(val_times),
        "feature_engineering_ms": _stats(feat_times),
        "model_predict_ms": _stats(pred_times),
        "end_to_end_total_ms": _stats(total_times),
        "passed": True,
    }


def run_check_e_rejection_suite(predictor: StandaloneRainfallPredictor) -> Dict[str, Any]:
    """Check (e): Verify Python & FastAPI 400 rejection across all invalid/stale failure modes."""
    df_base = pd.read_csv(PROJECT_ROOT / "artifacts" / "models" / "sample_recent_history_35d.csv")
    valid_run_date = pd.to_datetime(df_base["Ngày"]).max().strftime("%Y-%m-%d")

    cases: List[Dict[str, Any]] = []

    # Build negative test fixtures
    df_short = df_base.iloc[-29:].copy()

    df_gap = df_base.copy()
    df_gap = df_gap.drop(index=df_gap.index[15]).reset_index(drop=True)

    df_dup = df_base.copy()
    df_dup.loc[df_dup.index[10], "Ngày"] = df_dup.loc[df_dup.index[9], "Ngày"]

    df_nan = df_base.copy()
    df_nan.loc[df_nan.index[-1], "Nhiệt độ 2m"] = np.nan

    df_sentinel = df_base.copy()
    df_sentinel.loc[df_sentinel.index[-2], "Bức xạ sóng ngắn bề mặt"] = -999.0

    df_bounds = df_base.copy()
    df_bounds.loc[df_bounds.index[-1], "Độ ẩm tương đối 2m"] = 145.0

    df_frozen_rad = df_base.copy()
    df_frozen_rad.iloc[-5:, df_frozen_rad.columns.get_loc("Bức xạ sóng ngắn bề mặt")] = 5.1234

    stale_run_date = (pd.to_datetime(valid_run_date) + pd.Timedelta(days=8)).strftime("%Y-%m-%d")

    python_cases = [
        ("short_29_rows", df_short, valid_run_date, "INSUFFICIENT_HISTORY"),
        ("empty_df", df_base.iloc[0:0].copy(), valid_run_date, "INSUFFICIENT_HISTORY"),
        ("date_gap", df_gap, valid_run_date, "DATE_CONTINUITY_VIOLATION"),
        ("duplicate_date", df_dup, valid_run_date, "DATE_CONTINUITY_VIOLATION"),
        ("nan_meteo", df_nan, valid_run_date, "INVALID_OR_MISSING_METEO_INPUT"),
        ("sentinel_minus_999", df_sentinel, valid_run_date, "INVALID_OR_MISSING_METEO_INPUT"),
        ("out_of_physical_bounds", df_bounds, valid_run_date, "INVALID_OR_MISSING_METEO_INPUT"),
        ("frozen_radiation_5d", df_frozen_rad, valid_run_date, "STALE_RADIATION_OR_LAG"),
        ("stale_lag_8d", df_base.copy(), stale_run_date, "STALE_RADIATION_OR_LAG"),
    ]

    all_passed = True
    with TestClient(fastapi_app) as client:
        for name, df_case, r_date, expected_code in python_cases:
            py_rejected = False
            py_code = None
            py_msg = ""
            try:
                predictor.predict(df_case, run_date=r_date)
            except ServingValidationError as exc:
                py_rejected = exc.code == expected_code
                py_code = exc.code
                py_msg = str(exc)
            except Exception as exc:
                py_msg = f"Unexpected exception: {exc}"

            # Also verify HTTP /predict rejects with 400 under clock-injected HCMC date
            if name == "empty_df":
                payload = {"history": []}
            else:
                # Replace NaN with None and serialize dates as ISO YYYY-MM-DD strings
                records = json.loads(df_case.to_json(orient="records", date_format="iso"))
                payload = {"history": records}

            # 05:00 UTC == 12:00 Asia/Ho_Chi_Minh on r_date
            injected_utc = datetime.fromisoformat(f"{r_date}T05:00:00+00:00")
            with patch("src.config.clock.now_utc", return_value=injected_utc):
                resp = client.post("/predict", json=payload)
            body = resp.json()
            err_blob = body.get("error") or body.get("detail") or body
            detail_str = json.dumps(err_blob, ensure_ascii=False)
            api_rejected = resp.status_code == 400 and expected_code in detail_str

            ok = py_rejected and api_rejected
            if not ok:
                all_passed = False

            cases.append(
                {
                    "case": name,
                    "expected_code": expected_code,
                    "python_rejected": py_rejected,
                    "python_code": py_code,
                    "api_status_code": resp.status_code,
                    "api_rejected_with_code": api_rejected,
                    "detail": detail_str[:200] if detail_str else py_msg[:200],
                    "passed": ok,
                }
            )

        # Verify legacy public `run_date` parameter is rejected with HTTP 422 INVALID_QUERY_PARAMETER
        resp_legacy_rundate = client.post("/predict", json={"run_date": stale_run_date})
        detail_legacy = json.dumps(resp_legacy_rundate.json(), ensure_ascii=False)
        legacy_rundate_ok = (
            resp_legacy_rundate.status_code == 422 and "INVALID_QUERY_PARAMETER" in detail_legacy
        )
        if not legacy_rundate_ok:
            all_passed = False
        cases.append(
            {
                "case": "legacy_public_run_date_rejected",
                "expected_code": "INVALID_QUERY_PARAMETER",
                "python_rejected": True,
                "python_code": "N/A (API schema forbids run_date)",
                "api_status_code": resp_legacy_rundate.status_code,
                "api_rejected_with_code": legacy_rundate_ok,
                "detail": detail_legacy[:200],
                "passed": legacy_rundate_ok,
            }
        )

    return {"passed": all_passed, "cases": cases}


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictor = StandaloneRainfallPredictor(artifact_dir=PROJECT_ROOT / "artifacts" / "models", mode="local")

    logger.info("Running Check (a): Canonical NPZ array parity & benchmark alignment...")
    check_a = run_check_a_array_parity(predictor.metadata)

    logger.info("Running Check (b): 10-origin raw DataFrame Pyfunc vs Standalone parity...")
    check_b = run_check_b_raw_df_parity(predictor)

    logger.info("Running Check (c): 35-day raw sample operational lag check...")
    check_c = run_check_c_sample_operational(predictor)

    logger.info("Running Check (d): Latency breakdown & RSS memory measurement...")
    check_d = run_check_d_latency_and_memory(predictor, iterations=30)

    logger.info("Running Check (e): Empirical input rejection matrix (Python + FastAPI)...")
    check_e = run_check_e_rejection_suite(predictor)

    overall_pass = all(c["passed"] for c in (check_a, check_b, check_c, check_d, check_e))
    report = {
        "overall_passed": overall_pass,
        "check_a_array_parity": check_a,
        "check_b_raw_df_parity_10_origins": check_b,
        "check_c_sample_operational_35d": check_c,
        "check_d_latency_and_rss": check_d,
        "check_e_rejection_matrix": check_e,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Saved operational verification report to %s (overall_passed=%s)", REPORT_PATH, overall_pass)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
