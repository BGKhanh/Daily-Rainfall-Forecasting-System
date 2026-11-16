"""
Script: backfill_mlflow_tracking.py
===================================
Backfills all 76 canonical benchmark runs (72 scenarios + Config A/B audit split
for SC-28..31) into DagsHub Hosted MLflow Tracking Server using idempotent upsert
and single-batch `client.log_batch()` calls with retry backoff.

Registers 3 Pyfunc Model Versions (`RainfallForecastHCMC`):
  - SC-35 (XGBoost Tweedie F3) -> @serving (Production)
  - SC-30 Config A (LightGBM Tweedie F2) -> @shadow_val_champ (Staging)
  - SC-31 Config A (LightGBM Tweedie F3) -> @shadow_test_contender (Staging)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.entities import Metric, Param, RunTag

from src.config.mlops import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGISTERED_MODEL_NAME,
    ModelRegistryGovernance,
    get_or_create_experiment,
    init_dagshub_mlflow,
    retry_with_backoff,
)
from src.data.loader import DataLoader
from src.inference.pyfunc import (
    RainfallForecastingPyfuncWrapper,
    build_pyfunc_signature_and_example,
    get_serving_pip_requirements,
)

CANONICAL_AB_JSON = PROJECT_ROOT / "reports" / "storage_audit" / "lightgbm_tweedie_config_ab_canonical.json"
TEST_CONFIG_B_CSV = PROJECT_ROOT / "data" / "processed" / "results" / "ml_tweedie_test_results_configB.csv"
TEST_CONFIG_B_H_CSV = PROJECT_ROOT / "data" / "processed" / "results" / "ml_tweedie_test_horizon_breakdown_configB.csv"
LGBM_SCENARIO_IDS = ("SC-28", "SC-29", "SC-30", "SC-31")
AUDIT_REMEDIATION_DATE = "2026-11-16"


def _load_canonical_ab_sources() -> Dict[str, Any]:
    """Load canonical Config A & Config B LightGBM Tweedie metrics from generated JSON/CSVs."""
    if not CANONICAL_AB_JSON.exists():
        raise FileNotFoundError(
            f"Missing canonical LightGBM Config A/B JSON at {CANONICAL_AB_JSON}. "
            "Run scripts/audit/mt_q2_audit_76_runs.py to regenerate it."
        )
    with open(CANONICAL_AB_JSON, "r", encoding="utf-8") as f:
        canonical_ab = json.load(f)

    scenarios = canonical_ab.get("scenarios", {})
    missing = [sc for sc in LGBM_SCENARIO_IDS if sc not in scenarios]
    if missing:
        raise ValueError(f"Canonical Config A/B JSON missing scenario(s): {missing}")

    df_test_b = pd.read_csv(TEST_CONFIG_B_CSV).set_index("scenario_id") if TEST_CONFIG_B_CSV.exists() else None
    df_test_b_h = pd.read_csv(TEST_CONFIG_B_H_CSV).set_index("scenario_id") if TEST_CONFIG_B_H_CSV.exists() else None
    return {
        "canonical_ab": canonical_ab,
        "df_test_b": df_test_b,
        "df_test_b_h": df_test_b_h,
    }


def _build_canonical_horizon_metrics_from_dict(
    horizons_dict: Dict[str, Dict[str, float]],
    prefix: str,
    timestamp: int,
) -> List[Metric]:
    metrics: List[Metric] = []
    for h in range(1, 8):
        h_vals = horizons_dict.get(f"h{h}", {})
        for m in ("MAE", "MSE", "RMSE", "MASE"):
            _append_metric(metrics, f"{prefix}_{m}", h_vals.get(m), timestamp, step=h)
    return metrics


def _safe_float(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        return None
    return float(val)


def _append_metric(metrics: List[Metric], key: str, value: Any, timestamp: int, step: int = 0) -> None:
    num = _safe_float(value)
    if num is not None:
        metrics.append(Metric(key=key, value=num, timestamp=timestamp, step=step))


def _build_horizon_metrics(
    vh_row: Optional[pd.Series],
    th_row: Optional[pd.Series],
    timestamp: int,
    include_val: bool = True,
    include_test: bool = True,
) -> List[Metric]:
    metrics: List[Metric] = []
    for h in range(1, 8):
        if include_val and vh_row is not None:
            for m in ("MAE", "MSE", "RMSE", "MASE"):
                _append_metric(metrics, f"val_horizon_{m}", vh_row.get(f"{m}@h{h}"), timestamp, step=h)
        if include_test and th_row is not None:
            for m in ("MAE", "MSE", "RMSE", "MASE"):
                _append_metric(metrics, f"test_horizon_{m}", th_row.get(f"{m}@h{h}"), timestamp, step=h)
    return metrics


def _upsert_run_batch(
    client,
    exp_id: str,
    run_name: str,
    existing_runs_by_name: Dict[str, Any],
    tags_dict: Dict[str, str],
    params_dict: Dict[str, Any],
    metrics_list: List[Metric],
    force_update: bool = False,
) -> str:
    existing_run = existing_runs_by_name.get(run_name)
    if existing_run is not None:
        run_id = existing_run.info.run_id
        expected_metric_keys = {m.key for m in metrics_list}
        has_all_metrics = expected_metric_keys.issubset(set(existing_run.data.metrics.keys()))
        if not force_update and existing_run.info.status == "FINISHED" and has_all_metrics:
            return run_id
        existing_param_keys = set(existing_run.data.params.keys())
    else:
        created = retry_with_backoff(
            lambda: client.create_run(experiment_id=exp_id, run_name=run_name),
            operation_name=f"create_run({run_name})",
        )
        run_id = created.info.run_id
        existing_param_keys = set()

    tags = [RunTag(key=k, value=str(v)) for k, v in tags_dict.items()]
    params = [
        Param(key=k, value=str(v))
        for k, v in params_dict.items()
        if k not in existing_param_keys
    ]

    retry_with_backoff(
        lambda: client.log_batch(run_id=run_id, metrics=metrics_list, params=params, tags=tags),
        operation_name=f"log_batch({run_name})",
    )
    retry_with_backoff(
        lambda: client.set_terminated(run_id, status="FINISHED"),
        operation_name=f"set_terminated({run_name})",
    )
    time.sleep(0.25)
    return run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill 76 DS108 runs & registry to DagsHub MLflow")
    parser.add_argument(
        "--clean-existing",
        action="store_true",
        help="Delete existing runs in the DagsHub experiment before backfilling.",
    )
    parser.add_argument(
        "--force-update-runs",
        action="store_true",
        help="Re-upload metrics and tags even for runs already marked FINISHED.",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Skip run metric/param backfill and only update Model Registry.",
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="Skip Pyfunc Model Registry upload.",
    )
    parser.add_argument(
        "--force-reregister",
        action="store_true",
        help="Force logging new Pyfunc model versions even if matching SHA-256 versions exist.",
    )
    parser.add_argument(
        "--remediate-audit-v2",
        action="store_true",
        help="Patch only the 9 flagged DagsHub runs (SC-28..31 ConfigA/B + SC-35 config_regime) preserving run history.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = init_dagshub_mlflow(experiment_name=DEFAULT_EXPERIMENT_NAME)
    print(f"Connected to DagsHub MLflow Tracking URI: {client.tracking_uri}")

    exp_id = get_or_create_experiment(client, DEFAULT_EXPERIMENT_NAME)
    print(f"Experiment '{DEFAULT_EXPERIMENT_NAME}' ready (ID: {exp_id})")

    existing_runs = retry_with_backoff(
        lambda: client.search_runs(experiment_ids=[exp_id], max_results=1000),
        operation_name="search_runs",
    )
    if args.clean_existing and existing_runs:
        print(f"Cleaning {len(existing_runs)} existing runs (--clean-existing)...")
        for r in existing_runs:
            retry_with_backoff(
                lambda rid=r.info.run_id: client.delete_run(rid),
                operation_name=f"delete_run({r.info.run_id})",
            )
        existing_runs = []

    existing_by_name: Dict[str, Any] = {}
    for r in existing_runs:
        r_name = r.data.tags.get("mlflow.runName")
        if r_name:
            existing_by_name[r_name] = r
    print(f"Found {len(existing_by_name)} existing named runs on DagsHub.")

    # Load 4 canonical benchmark CSVs
    val_csv = PROJECT_ROOT / "reports" / "validation_benchmark" / "unified_72_scenarios_leaderboard.csv"
    test_csv = PROJECT_ROOT / "reports" / "test_benchmark" / "unified_72_scenarios_test_leaderboard.csv"
    val_h_csv = PROJECT_ROOT / "reports" / "validation_benchmark" / "unified_72_horizon_breakdown.csv"
    test_h_csv = PROJECT_ROOT / "reports" / "test_benchmark" / "unified_72_test_horizon_breakdown.csv"

    for csv_path in (val_csv, test_csv, val_h_csv, test_h_csv):
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required canonical benchmark file: {csv_path}")

    df_val = pd.read_csv(val_csv).set_index("scenario_id")
    df_test = pd.read_csv(test_csv).set_index("scenario_id")
    df_val_h = pd.read_csv(val_h_csv).set_index("scenario_id")
    df_test_h = pd.read_csv(test_h_csv).set_index("scenario_id")
    ab_sources = _load_canonical_ab_sources()
    canonical_ab = ab_sources["canonical_ab"]
    df_test_b = ab_sources["df_test_b"]
    df_test_b_h = ab_sources["df_test_b_h"]
    print(f"Loaded {len(df_val)} Validation and {len(df_test)} Test scenarios (+ Config A/B canonical JSON).")

    run_id_map: Dict[str, str] = {}
    now_ms = int(time.time() * 1000)

    if args.remediate_audit_v2:
        print("\n[AUDIT REMEDIATION V2] Patching 9 flagged DagsHub runs in-place (preserving history)...")
        remediated = []
        for sc_id in LGBM_SCENARIO_IDS:
            model_name = str(df_val.loc[sc_id, "model"])
            sc_canon = canonical_ab["scenarios"][sc_id]

            # 1. Patch SC-28..31_ConfigB: keep Test metrics untouched, replace val_{MAE,MSE,RMSE,MASE} & 28 val_horizon_*
            run_name_b = f"{sc_id}_{model_name}_ConfigB"
            run_b = existing_by_name.get(run_name_b)
            if run_b is None:
                raise KeyError(f"Expected existing DagsHub run '{run_name_b}' not found!")
            old_b_metrics = dict(run_b.data.metrics)
            val_b_canon = sc_canon["Config_B"]["validation"]
            val_b_overall = val_b_canon["overall"]

            metrics_b_patch: List[Metric] = []
            for m in ("MAE", "MSE", "RMSE", "MASE"):
                _append_metric(metrics_b_patch, f"val_{m}", val_b_overall[m], now_ms)
            metrics_b_patch.extend(
                _build_canonical_horizon_metrics_from_dict(val_b_canon["horizons"], "val_horizon", now_ms)
            )
            note_b = (
                f"Audit Remediation v2 ({AUDIT_REMEDIATION_DATE}): Replaced Config B validation metrics & 28 val_horizon_* "
                f"(previously copied from Config A val leaderboard) with recomputed Config B (max_depth=5) validation values; "
                f"Test metrics preserved untouched.\n"
                f"Old val_MAE={old_b_metrics.get('val_MAE')}, val_MSE={old_b_metrics.get('val_MSE')}, "
                f"val_RMSE={old_b_metrics.get('val_RMSE')}, val_MASE={old_b_metrics.get('val_MASE')}\n"
                f"New val_MAE={val_b_overall['MAE']:.6f}, val_MSE={val_b_overall['MSE']:.6f}, "
                f"val_RMSE={val_b_overall['RMSE']:.6f}, val_MASE={val_b_overall['MASE']:.6f}"
            )
            tags_b_patch = [
                RunTag(key="audit_remediation_v2", value="true"),
                RunTag(key="audit_remediation_date", value=AUDIT_REMEDIATION_DATE),
                RunTag(key="mlflow.note.content", value=note_b),
            ]
            retry_with_backoff(
                lambda rid=run_b.info.run_id, m=metrics_b_patch, t=tags_b_patch: client.log_batch(
                    run_id=rid, metrics=m, params=[], tags=t
                ),
                operation_name=f"remediate({run_name_b})",
            )
            remediated.append(run_name_b)
            print(f"  [PATCHED] {run_name_b}: val_MAE {old_b_metrics.get('val_MAE')} -> {val_b_overall['MAE']:.6f}")
            time.sleep(0.25)

            # 2. Patch SC-28..31_ConfigA: fix val & test overall (notably test_MSE, test_RMSE), add val/test horizon steps 1..7, fix SC-31 tag
            run_name_a = f"{sc_id}_{model_name}_ConfigA"
            run_a = existing_by_name.get(run_name_a)
            if run_a is None:
                raise KeyError(f"Expected existing DagsHub run '{run_name_a}' not found!")
            old_a_metrics = dict(run_a.data.metrics)
            old_a_role = run_a.data.tags.get("governance_role", "")
            val_a_canon = sc_canon["Config_A"]["validation"]
            test_a_canon = sc_canon["Config_A"]["test"]

            metrics_a_patch: List[Metric] = []
            for m in ("MAE", "MSE", "RMSE", "MASE"):
                _append_metric(metrics_a_patch, f"val_{m}", val_a_canon["overall"][m], now_ms)
                _append_metric(metrics_a_patch, f"test_{m}", test_a_canon["overall"][m], now_ms)
            if sc_id in df_test.index:
                _append_metric(metrics_a_patch, "test_rank", df_test.loc[sc_id, "rank"], now_ms)
            metrics_a_patch.extend(
                _build_canonical_horizon_metrics_from_dict(val_a_canon["horizons"], "val_horizon", now_ms)
            )
            metrics_a_patch.extend(
                _build_canonical_horizon_metrics_from_dict(test_a_canon["horizons"], "test_horizon", now_ms)
            )
            role_a = "validation_selected_champion" if sc_id == "SC-30" else (
                "post_audit_test_contender" if sc_id == "SC-31" else "unassigned"
            )
            note_a = (
                f"Audit Remediation v2 ({AUDIT_REMEDIATION_DATE}): Replaced hardcoded Config A test_MSE/test_RMSE with "
                f"canonical recomputed Config A (max_depth=-1) values, synced val metrics & 28 val_horizon_* + added 28 "
                f"test_horizon_* (steps 1..7) from canonical benchmark, and verified governance_role='{role_a}'.\n"
                f"Old test_MSE={old_a_metrics.get('test_MSE')}, test_RMSE={old_a_metrics.get('test_RMSE')}, "
                f"governance_role='{old_a_role}'\n"
                f"New test_MSE={test_a_canon['overall']['MSE']:.6f}, test_RMSE={test_a_canon['overall']['RMSE']:.6f}, "
                f"governance_role='{role_a}'"
            )
            tags_a_patch = [
                RunTag(key="governance_role", value=role_a),
                RunTag(key="audit_remediation_v2", value="true"),
                RunTag(key="audit_remediation_date", value=AUDIT_REMEDIATION_DATE),
                RunTag(key="mlflow.note.content", value=note_a),
            ]
            retry_with_backoff(
                lambda rid=run_a.info.run_id, m=metrics_a_patch, t=tags_a_patch: client.log_batch(
                    run_id=rid, metrics=m, params=[], tags=t
                ),
                operation_name=f"remediate({run_name_a})",
            )
            remediated.append(run_name_a)
            print(
                f"  [PATCHED] {run_name_a}: test_RMSE {old_a_metrics.get('test_RMSE')} -> "
                f"{test_a_canon['overall']['RMSE']:.6f} | +28 test_horizon_* | role={role_a}"
            )
            time.sleep(0.25)

        # 3. Patch SC-35: only tag config_regime="DevTrain_Frozen_Canonical" + audit tags (NO metric changes)
        run_35_name = "SC-35_xgboost_tweedie_F3"
        run_35 = existing_by_name.get(run_35_name)
        if run_35 is None:
            raise KeyError(f"Expected existing DagsHub run '{run_35_name}' not found!")
        old_35_regime = run_35.data.tags.get("config_regime", "<unset>")
        note_35 = (
            f"Audit Remediation v2 ({AUDIT_REMEDIATION_DATE}): Set config_regime='DevTrain_Frozen_Canonical' "
            f"(previously '{old_35_regime}') since Config A/B applies strictly to LightGBM Tweedie SC-28..31. "
            f"Val metrics left untouched (MT-Q2 CSV delta <= 4.4e-5 is a 4-decimal CSV rounding artifact)."
        )
        tags_35_patch = [
            RunTag(key="config_regime", value="DevTrain_Frozen_Canonical"),
            RunTag(key="audit_remediation_v2", value="true"),
            RunTag(key="audit_remediation_date", value=AUDIT_REMEDIATION_DATE),
            RunTag(key="mlflow.note.content", value=note_35),
        ]
        retry_with_backoff(
            lambda rid=run_35.info.run_id, t=tags_35_patch: client.log_batch(
                run_id=rid, metrics=[], params=[], tags=t
            ),
            operation_name=f"remediate({run_35_name})",
        )
        remediated.append(run_35_name)
        print(f"  [PATCHED] {run_35_name}: config_regime -> DevTrain_Frozen_Canonical")
        print(f"\nAudit Remediation v2 complete: {len(remediated)} runs updated in-place.")
        return

    if args.skip_runs:
        for sc_id in df_val.index:
            v_row = df_val.loc[sc_id]
            model_name = v_row["model"]
            if sc_id in LGBM_SCENARIO_IDS:
                r_b = existing_by_name.get(f"{sc_id}_{model_name}_ConfigB")
                r_a = existing_by_name.get(f"{sc_id}_{model_name}_ConfigA")
                if r_b:
                    run_id_map[f"{sc_id}_ConfigB"] = r_b.info.run_id
                if r_a:
                    run_id_map[f"{sc_id}_ConfigA"] = r_a.info.run_id
            else:
                r_std = existing_by_name.get(f"{sc_id}_{model_name}")
                if r_std:
                    run_id_map[sc_id] = r_std.info.run_id
    else:
        print("\nSyncing 76 runs to DagsHub via log_batch...")
        for idx, sc_id in enumerate(df_val.index, start=1):
            v_row = df_val.loc[sc_id]
            t_row = df_test.loc[sc_id] if sc_id in df_test.index else None
            vh_row = df_val_h.loc[sc_id] if sc_id in df_val_h.index else None
            th_row = df_test_h.loc[sc_id] if sc_id in df_test_h.index else None

            model_name = str(v_row["model"])
            family = str(v_row["family"])
            fset = str(v_row["feature_set"])
            strategy = str(v_row["strategy"])
            is_naive = sc_id in ("SC-01", "SC-02")

            if sc_id in LGBM_SCENARIO_IDS:
                sc_canon = canonical_ab["scenarios"][sc_id]
                val_b_canon = sc_canon["Config_B"]["validation"]
                test_b_canon = sc_canon["Config_B"]["test"]
                tb_row = df_test_b.loc[sc_id] if (df_test_b is not None and sc_id in df_test_b.index) else None
                tbh_row = df_test_b_h.loc[sc_id] if (df_test_b_h is not None and sc_id in df_test_b_h.index) else None

                # 1. Config B (Legacy max_depth=5 archive)
                run_name_b = f"{sc_id}_{model_name}_ConfigB"
                tags_b = {
                    "mlflow.runName": run_name_b,
                    "scenario_id": sc_id,
                    "model_family": family,
                    "feature_set": fset,
                    "strategy": strategy,
                    "source": "backfill",
                    "config_version": "Config_B_Benchmark_Original",
                    "hyperparameters_note": "max_depth=5, num_leaves=31, n_estimators=100",
                }
                params_b = {
                    "model_name": model_name,
                    "family": family,
                    "feature_set": fset,
                    "strategy": strategy,
                    "objective": "tweedie",
                    "tweedie_variance_power": 1.5,
                    "learning_rate": 0.03,
                    "max_depth": 5,
                    "num_leaves": 31,
                    "n_estimators": 100,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                }
                metrics_b: List[Metric] = []
                for m in ("MAE", "MSE", "RMSE", "MASE"):
                    _append_metric(metrics_b, f"val_{m}", val_b_canon["overall"][m], now_ms)
                    test_val_b = tb_row.get(m) if tb_row is not None else test_b_canon["overall"][m]
                    _append_metric(metrics_b, f"test_{m}", test_val_b, now_ms)
                if tb_row is not None:
                    _append_metric(metrics_b, "test_training_time", tb_row.get("training_time"), now_ms)
                    _append_metric(metrics_b, "test_inference_time", tb_row.get("inference_time"), now_ms)
                    _append_metric(metrics_b, "training_time", tb_row.get("training_time"), now_ms)
                    _append_metric(metrics_b, "inference_time", tb_row.get("inference_time"), now_ms)
                metrics_b.extend(
                    _build_canonical_horizon_metrics_from_dict(val_b_canon["horizons"], "val_horizon", now_ms)
                )
                if tbh_row is not None:
                    metrics_b.extend(
                        _build_horizon_metrics(None, tbh_row, now_ms, include_val=False, include_test=True)
                    )
                else:
                    metrics_b.extend(
                        _build_canonical_horizon_metrics_from_dict(test_b_canon["horizons"], "test_horizon", now_ms)
                    )

                run_id_map[f"{sc_id}_ConfigB"] = _upsert_run_batch(
                    client, exp_id, run_name_b, existing_by_name, tags_b, params_b, metrics_b,
                    force_update=args.force_update_runs,
                )

                # 2. Config A (Canonical Config: max_depth=-1 for both Validation & Test)
                run_name_a = f"{sc_id}_{model_name}_ConfigA"
                val_a_canon = sc_canon["Config_A"]["validation"]
                test_a_canon = sc_canon["Config_A"]["test"]
                role_tag = "unassigned"
                if sc_id == "SC-30":
                    role_tag = "validation_selected_champion"
                elif sc_id == "SC-31":
                    role_tag = "post_audit_test_contender"

                tags_a = {
                    "mlflow.runName": run_name_a,
                    "scenario_id": sc_id,
                    "model_family": family,
                    "feature_set": fset,
                    "strategy": strategy,
                    "source": "backfill",
                    "config_version": "Config_A_Validation_Canonical",
                    "governance_role": role_tag,
                    "hyperparameters_note": "max_depth=-1 (unconstrained), num_leaves=31, n_estimators=100",
                }
                params_a = {
                    "model_name": model_name,
                    "family": family,
                    "feature_set": fset,
                    "strategy": strategy,
                    "objective": "tweedie",
                    "tweedie_variance_power": 1.5,
                    "learning_rate": 0.03,
                    "max_depth": -1,
                    "num_leaves": 31,
                    "n_estimators": 100,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                }
                metrics_a: List[Metric] = []
                for m in ("MAE", "MSE", "RMSE", "MASE"):
                    _append_metric(metrics_a, f"val_{m}", val_a_canon["overall"][m], now_ms)
                    _append_metric(metrics_a, f"test_{m}", test_a_canon["overall"][m], now_ms)
                _append_metric(metrics_a, "val_rank", v_row.get("rank"), now_ms)
                _append_metric(metrics_a, "val_training_time", v_row.get("training_time"), now_ms)
                _append_metric(metrics_a, "val_inference_time", v_row.get("inference_time"), now_ms)
                if t_row is not None:
                    _append_metric(metrics_a, "test_rank", t_row.get("rank"), now_ms)
                    _append_metric(metrics_a, "test_training_time", t_row.get("training_time"), now_ms)
                    _append_metric(metrics_a, "test_inference_time", t_row.get("inference_time"), now_ms)
                    _append_metric(metrics_a, "training_time", t_row.get("training_time"), now_ms)
                    _append_metric(metrics_a, "inference_time", t_row.get("inference_time"), now_ms)
                else:
                    _append_metric(metrics_a, "training_time", v_row.get("training_time"), now_ms)
                    _append_metric(metrics_a, "inference_time", v_row.get("inference_time"), now_ms)
                metrics_a.extend(
                    _build_canonical_horizon_metrics_from_dict(val_a_canon["horizons"], "val_horizon", now_ms)
                )
                metrics_a.extend(
                    _build_canonical_horizon_metrics_from_dict(test_a_canon["horizons"], "test_horizon", now_ms)
                )

                run_id_map[f"{sc_id}_ConfigA"] = _upsert_run_batch(
                    client, exp_id, run_name_a, existing_by_name, tags_a, params_a, metrics_a,
                    force_update=args.force_update_runs,
                )
            else:
                run_name = f"{sc_id}_{model_name}"
                tags = {
                    "mlflow.runName": run_name,
                    "scenario_id": sc_id,
                    "model_family": family,
                    "feature_set": fset,
                    "strategy": strategy,
                    "source": "backfill",
                }
                if is_naive:
                    tags["is_naive_baseline"] = "true"
                if sc_id == "SC-35":
                    tags["config_regime"] = "DevTrain_Frozen_Canonical"
                    tags["governance_role"] = "test_ranking_winner_as_originally_benchmarked"
                    tags["governance_note"] = (
                        "Deployed Serving champion (Test MAE=3.7333 mm, Rank 2 canonical Config A / Rank 1 Config B)."
                    )
                if family in ["NHITS", "NBEATS", "TiDE", "PatchTST", "DLinear"] or "Neural" in str(
                    v_row.get("group", "")
                ):
                    tags["val_test_configuration_note"] = (
                        "LECH, co chu dich (PROP-03): Val 1000 steps vs Test 10000 steps"
                    )

                params = {
                    "model_name": model_name,
                    "family": family,
                    "feature_set": fset,
                    "strategy": strategy,
                }

                metrics: List[Metric] = []
                for m in ("MAE", "MSE", "RMSE", "MASE"):
                    _append_metric(metrics, f"val_{m}", v_row.get(m), now_ms)
                    if t_row is not None:
                        _append_metric(metrics, f"test_{m}", t_row.get(m), now_ms)
                _append_metric(metrics, "val_rank", v_row.get("rank"), now_ms)
                _append_metric(metrics, "val_training_time", v_row.get("training_time"), now_ms)
                _append_metric(metrics, "val_inference_time", v_row.get("inference_time"), now_ms)
                if t_row is not None:
                    _append_metric(metrics, "test_rank", t_row.get("rank"), now_ms)
                    _append_metric(metrics, "test_training_time", t_row.get("training_time"), now_ms)
                    _append_metric(metrics, "test_inference_time", t_row.get("inference_time"), now_ms)
                    _append_metric(metrics, "training_time", t_row.get("training_time"), now_ms)
                    _append_metric(metrics, "inference_time", t_row.get("inference_time"), now_ms)
                else:
                    _append_metric(metrics, "training_time", v_row.get("training_time"), now_ms)
                    _append_metric(metrics, "inference_time", v_row.get("inference_time"), now_ms)

                # Rule 6: Hurdle classification threshold selected on validation
                thresh = _safe_float(v_row.get("selected_hurdle_threshold"))
                if thresh is None and t_row is not None:
                    thresh = _safe_float(t_row.get("selected_hurdle_threshold"))
                if thresh is not None:
                    _append_metric(metrics, "selected_hurdle_threshold", thresh, now_ms)

                metrics.extend(
                    _build_horizon_metrics(vh_row, th_row, now_ms, include_val=True, include_test=True)
                )

                run_id_map[sc_id] = _upsert_run_batch(
                    client, exp_id, run_name, existing_by_name, tags, params, metrics,
                    force_update=args.force_update_runs,
                )

            if idx % 6 == 0 or idx == len(df_val):
                print(f"  Synced {idx}/{len(df_val)} benchmark scenarios ({len(run_id_map)} runs total)...", flush=True)

        print(f"Successfully synced {len(run_id_map)} runs to DagsHub Experiment '{DEFAULT_EXPERIMENT_NAME}'!")

    if args.skip_registry:
        print("Skipping Model Registry step (--skip-registry).")
        return

    # 3. Model Registry Registration with Pyfunc Wrapper, Signature & code_paths
    print("\nRegistering Serving & Shadow Models on DagsHub Model Registry...")
    meta_path = PROJECT_ROOT / "artifacts" / "models" / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    checksums: Dict[str, str] = metadata.get("sha256_checksums", {})

    ModelRegistryGovernance.ensure_registered_model(client, DEFAULT_REGISTERED_MODEL_NAME)
    existing_versions = retry_with_backoff(
        lambda: client.search_model_versions(f"name = '{DEFAULT_REGISTERED_MODEL_NAME}'"),
        operation_name="search_model_versions",
    )

    sample_history_df = DataLoader().load_data().iloc[-35:].copy()
    pip_reqs = get_serving_pip_requirements()

    models_to_register = [
        {
            "run_key": "SC-35",
            "scenario_id": "SC-35",
            "artifact_model_file": "artifacts/models/sc35_xgboost_tweedie_f3.joblib",
            "feature_builder_file": "artifacts/models/feature_builder_f3.joblib",
            "feature_level": 3,
            "alias": "serving",
            "governance_role": "test_ranking_winner_as_originally_benchmarked",
            "description": "SC-35 XGBoost Tweedie F3: Test Ranking Winner as originally benchmarked (Test MAE = 3.7333 mm).",
        },
        {
            "run_key": "SC-30_ConfigA",
            "scenario_id": "SC-30",
            "artifact_model_file": "artifacts/models/sc30_lightgbm_tweedie_f2.joblib",
            "feature_builder_file": "artifacts/models/feature_builder_f2.joblib",
            "feature_level": 2,
            "alias": "shadow_val_champ",
            "governance_role": "validation_selected_champion",
            "description": "SC-30 LightGBM Tweedie F2 Config A: Validation-Selected Champion (Val MAE = 3.1485 mm, Test MAE = 3.7501 mm).",
        },
        {
            "run_key": "SC-31_ConfigA",
            "scenario_id": "SC-31",
            "artifact_model_file": "artifacts/models/sc31_lightgbm_tweedie_f3.joblib",
            "feature_builder_file": "artifacts/models/feature_builder_f3.joblib",
            "feature_level": 3,
            "alias": "shadow_test_contender",
            "governance_role": "post_audit_test_contender",
            "description": "SC-31 LightGBM Tweedie F3 Config A: Post-Audit Test Contender (Test MAE = 3.7246 mm).",
        },
    ]

    for item in models_to_register:
        run_id = run_id_map[item["run_key"]]
        model_fname = Path(item["artifact_model_file"]).name
        fb_fname = Path(item["feature_builder_file"]).name
        sha_model = checksums.get(model_fname, "")
        sha_fb = checksums.get(fb_fname, "")

        # Idempotent skip if exact run_id + sha256_model + alias already exists
        matching_existing = [
            mv
            for mv in existing_versions
            if mv.run_id == run_id
            and getattr(mv, "tags", {}).get("sha256_model") == sha_model
            and getattr(mv, "tags", {}).get("governance_alias") == item["alias"]
        ]
        if matching_existing and not args.force_reregister and not args.clean_existing:
            mv_version = matching_existing[0].version
            print(
                f"  [Idempotent Skip] {item['run_key']} already registered as v{mv_version} "
                f"(alias '{item['alias']}', sha256={sha_model[:12]}...)"
            )
            ModelRegistryGovernance.register_model_version_governance(
                client=client,
                model_name=DEFAULT_REGISTERED_MODEL_NAME,
                version=mv_version,
                alias=item["alias"],
                governance_role=item["governance_role"],
                scenario_id=item["scenario_id"],
                feature_level=f"F{item['feature_level']}",
                sha256_model=sha_model,
                sha256_feature_builder=sha_fb,
                description=item["description"],
            )
            continue

        print(f"Logging Pyfunc model for {item['run_key']} (run_id: {run_id})...")
        pyfunc_model = RainfallForecastingPyfuncWrapper(
            model_path=str(PROJECT_ROOT / item["artifact_model_file"]),
            feature_builder_path=str(PROJECT_ROOT / item["feature_builder_file"]),
            feature_level=item["feature_level"],
            governance_role=item["governance_role"],
        )
        signature, input_example = build_pyfunc_signature_and_example(pyfunc_model, sample_history_df)

        with mlflow.start_run(run_id=run_id):
            model_info = retry_with_backoff(
                lambda: mlflow.pyfunc.log_model(
                    name="rainfall_model",
                    python_model=pyfunc_model,
                    artifacts={
                        "model": str(PROJECT_ROOT / item["artifact_model_file"]),
                        "feature_builder": str(PROJECT_ROOT / item["feature_builder_file"]),
                    },
                    code_paths=[str(PROJECT_ROOT / "src")],
                    signature=signature,
                    input_example=input_example,
                    pip_requirements=pip_reqs,
                    registered_model_name=DEFAULT_REGISTERED_MODEL_NAME,
                ),
                operation_name=f"pyfunc.log_model({item['run_key']})",
            )

        mv_version = getattr(model_info, "registered_model_version", None)
        if mv_version is None:
            refreshed = retry_with_backoff(
                lambda: client.search_model_versions(f"name = '{DEFAULT_REGISTERED_MODEL_NAME}'"),
                operation_name="search_model_versions_refresh",
            )
            matching = [v for v in refreshed if v.run_id == run_id]
            mv_version = (
                str(max(int(v.version) for v in matching))
                if matching
                else str(max(int(v.version) for v in refreshed))
            )

        ModelRegistryGovernance.register_model_version_governance(
            client=client,
            model_name=DEFAULT_REGISTERED_MODEL_NAME,
            version=str(mv_version),
            alias=item["alias"],
            governance_role=item["governance_role"],
            scenario_id=item["scenario_id"],
            feature_level=f"F{item['feature_level']}",
            sha256_model=sha_model,
            sha256_feature_builder=sha_fb,
            description=item["description"],
        )
        print(
            f"  -> Registered v{mv_version}: alias='{item['alias']}', "
            f"role='{item['governance_role']}', sha256={sha_model[:12]}..."
        )

    print("\nALL DAGSHUB MLFLOW TRACKING & REGISTRY STEPS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
