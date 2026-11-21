"""
Round 13 Step 1.3 Contract Tests: P-c Serving-Time Solar DOY Imputation
========================================================================
Verifies:
1. `test_input_window_hash_as_of_reproducibility_with_degraded_solar`:
   `input_window_hash` hashes raw pre-imputation `observations_vintage` rows,
   never mutates `observations_vintage`, and reconstructs identically via As-Of queries.
2. `test_feature_builder_29_day_solar_prefix_invariance`:
   Replacing any subset or all of the preceding 29 days' solar cells before FeatureBuilder
   yields max feature diff == 0.0 (proving P-c single-anchor solar invariance).
3. `test_selective_per_variable_fill_and_leap_day_2024_02_29`:
   Present FLASHFlux values at anchor T (including 2024-02-29 & 2024-12-31 leap-day `366`)
   are preserved untouched while only NULL solar columns are filled via circular r=7 DOY median.
"""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import sqlite3
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.vintage_store import (
    ALL_35_VARS,
    CERES_ONLY_8_VARS,
    FLASHFLUX_3_VARS,
    METEO_24_VARS,
    compute_input_window_hash,
    compute_row_value_hash,
    init_schema,
    insert_forecasts_idempotent,
    query_observations_as_of,
)
from src.inference.forecast_runner import issue_forecasts_for_anchors, list_complete_anchor_dates
from src.inference.predictor import StandaloneRainfallPredictor
from src.inference.solar_imputation import (
    PC_CERES_POLICY_NAME,
    SOLAR_11_EN_VARS,
    SOLAR_EN_TO_VI,
    apply_serving_solar_doy_fallback,
    load_solar_doy_climatology,
)
from src.inference.validation import ServingValidationError

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "models"
SAMPLE_CSV = ARTIFACT_DIR / "sample_recent_history_35d.csv"


def _seed_degraded_solar_db(db_path: Path) -> sqlite3.Connection:
    base_df = pd.read_csv(SAMPLE_CSV)
    base_df["Ngày"] = pd.to_datetime(base_df["Ngày"]).dt.normalize()
    base_row = base_df.iloc[-1]
    from src.data.vintage_store import EN_TO_VI_CANONICAL

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    conn.execute(
        """
        INSERT INTO ingestion_runs (
            run_id, run_type, started_at_utc, finished_at_utc, status, location_id,
            scan_window_start, scan_window_end, rows_scanned, rows_inserted_v0,
            git_sha, config_sha256, operational_config_sha256
        ) VALUES (
            'seed_pc_contract', 'manual_scan', '2026-11-18T00:00:00Z', '2026-11-18T00:01:00Z',
            'SUCCESS', 'HCMC', '2026-10-15', '2026-11-17', 34, 34, 'sha_pc', 'cfg', 'opcfg'
        )
        """
    )

    col_list = ", ".join(ALL_35_VARS)
    ph = ", ".join(["?"] * len(ALL_35_VARS))
    # Seed one Dev Train row (<= 2018-04-30) so BASELINE_CLIMATOLOGY_DOY can also be issued
    dev_vals = {var: (3.5 if var == "PRECTOTCORR" else float(base_row[EN_TO_VI_CANONICAL[var]])) for var in ALL_35_VARS}
    conn.execute(
        f"""
        INSERT INTO observations_vintage (
            location_id, obs_date, version_num, ingested_at_utc, run_id,
            age_days_at_ingest, ingest_origin, missing_count, missing_meteo_count,
            missing_flashflux_count, missing_ceres_count, completeness_regime,
            missing_vars_json, change_type_summary, late_fill_count, late_fill_vars_json,
            value_change_count, value_change_details_json, anomaly_count, quality_flags_json,
            row_value_hash, {col_list}
        ) VALUES (
            'HCMC', '2018-04-30', 0, '2026-11-18T00:00:30Z', 'seed_pc_contract',
            1000, 'HISTORICAL_CSV_SEED', 0, 0, 0, 0, 'COMPLETE_ALL_35',
            '[]', 'INITIAL_SEED', 0, '[]', 0, '{{}}', 0, '{{}}', ?, {ph}
        )
        """,
        (compute_row_value_hash(dev_vals), *(dev_vals[c] for c in ALL_35_VARS)),
    )
    d0 = date(2026, 10, 15)
    for i in range(34):  # 2026-10-15 .. 2026-11-17
        d_iso = (d0 + timedelta(days=i)).isoformat()
        vals: dict[str, float | None] = {}
        for var in ALL_35_VARS:
            v = float(base_row[EN_TO_VI_CANONICAL[var]])
            if var == "PRECTOTCORR":
                v = float(round((i % 5) * 1.8, 2))
            else:
                v = round(v + (i % 4) * 0.05, 4)
            vals[var] = v

        # Last 2 days (Nov 16-17): all 11 solar missing (FLASHFLUX_INCOMPLETE); Oct 25..Nov 15: 8 CERES missing
        if d_iso >= "2026-11-16":
            for s in SOLAR_11_EN_VARS:
                vals[s] = None
            m_ff, m_cer, regime = 3, 8, "FLASHFLUX_INCOMPLETE"
        elif d_iso >= "2026-10-25":
            for s in CERES_ONLY_8_VARS:
                vals[s] = None
            m_ff, m_cer, regime = 0, 8, "MISSING_CERES_ONLY"
        else:
            m_ff, m_cer, regime = 0, 0, "COMPLETE_ALL_35"

        rhash = compute_row_value_hash(vals)
        conn.execute(
            f"""
            INSERT INTO observations_vintage (
                location_id, obs_date, version_num, ingested_at_utc, run_id,
                age_days_at_ingest, ingest_origin, missing_count, missing_meteo_count,
                missing_flashflux_count, missing_ceres_count, completeness_regime,
                missing_vars_json, change_type_summary, late_fill_count, late_fill_vars_json,
                value_change_count, value_change_details_json, anomaly_count, quality_flags_json,
                row_value_hash, {col_list}
            ) VALUES (
                'HCMC', ?, 0, '2026-11-18T00:00:30Z', 'seed_pc_contract',
                1, 'NASA_POWER_API', ?, 0, ?, ?, ?,
                '[]', 'INITIAL_NRT', 0, '[]', 0, '{{}}', 0, '{{}}', ?, {ph}
            )
            """,
            (d_iso, m_ff + m_cer, m_ff, m_cer, regime, rhash, *(vals[c] for c in ALL_35_VARS)),
        )
    conn.commit()
    return conn


def test_input_window_hash_as_of_reproducibility_with_degraded_solar(tmp_path: Path):
    """
    Pre-imputation `input_window_hash` matches As-Of reconstruction from `observations_vintage`,
    `observations_vintage` stays unmodified (NULL solar preserved), and V3.5 stores P-c metadata cleanly.
    """
    db_path = tmp_path / "pc_contract_hash.db"
    conn = _seed_degraded_solar_db(db_path)
    try:
        anchors = list_complete_anchor_dates(
            conn,
            start_date="2026-11-15",
            end_date="2026-11-17",
            location_id="HCMC",
            required_vars=METEO_24_VARS,
        )
        assert anchors == ["2026-11-15", "2026-11-16", "2026-11-17"]

        issue_ts = "2026-11-18T02:00:00Z"
        before_nulls = conn.execute(
            "SELECT ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN FROM observations_vintage WHERE obs_date = '2026-11-17'"
        ).fetchone()
        assert before_nulls["ALLSKY_SFC_SW_DWN"] is None
        assert before_nulls["CLRSKY_SFC_SW_DWN"] is None

        res = issue_forecasts_for_anchors(
            conn,
            anchor_dates=["2026-11-17"],
            location_id="HCMC",
            artifact_dir=ARTIFACT_DIR,
            issued_at_utc=issue_ts,
            issued_run_date="2026-11-18",
            issued_by="automated_pipeline",
        )
        assert res["rows_inserted"] == 35

        # observations_vintage MUST NOT be mutated by serving-time P-c imputation
        after_nulls = conn.execute(
            "SELECT ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN FROM observations_vintage WHERE obs_date = '2026-11-17'"
        ).fetchone()
        assert after_nulls["ALLSKY_SFC_SW_DWN"] is None
        assert after_nulls["CLRSKY_SFC_SW_DWN"] is None

        # As-Of query over raw pre-imputation vintages reproduces ledger `input_window_hash`
        as_of_rows = query_observations_as_of(
            conn,
            location_id="HCMC",
            input_window_start="2026-10-19",
            input_window_end="2026-11-17",
            issued_at_utc=issue_ts,
        )
        expected_hash = compute_input_window_hash(as_of_rows)

        sc35 = dict(
            conn.execute(
                "SELECT * FROM forecasts_ledger WHERE data_as_of = '2026-11-17' AND model_id = 'SC-35' AND horizon_day = 1"
            ).fetchone()
        )
        assert sc35["input_window_hash"] == expected_hash
        assert sc35["is_degraded_input"] == 1
        assert sc35["ceres_policy_applied"] == PC_CERES_POLICY_NAME
        assert sc35["issued_by"] == "automated_pipeline"
        imp_map = json.loads(sc35["imputed_values_json"])
        assert sorted(imp_map.keys()) == sorted(SOLAR_11_EN_VARS)
        _, table_sha = load_solar_doy_climatology()
        assert sc35["doy_table_sha256"] == table_sha

        # Baselines NEVER consume solar inputs -> imputed_values_json == '{}', is_degraded_input == 0
        pers = dict(
            conn.execute(
                "SELECT * FROM forecasts_ledger WHERE data_as_of = '2026-11-17' AND model_id = 'BASELINE_PERSISTENCE' AND horizon_day = 1"
            ).fetchone()
        )
        assert pers["is_degraded_input"] == 0
        assert pers["ceres_policy_applied"] == "NOT_APPLICABLE_BASELINE"
        assert pers["imputed_values_json"] == "{}"
        assert pers["doy_table_sha256"] is None

        # Strict DDL V3.5: missing or invalid issued_by must fail
        with pytest.raises(ValueError, match="issued_by"):
            insert_forecasts_idempotent(
                conn,
                [{**sc35, "forecast_id": "bad_1", "data_as_of": "2026-11-16", "issued_by": None}],
            )
        with pytest.raises(ValueError, match="issued_by"):
            insert_forecasts_idempotent(
                conn,
                [{**sc35, "forecast_id": "bad_2", "data_as_of": "2026-11-16", "issued_by": "rogue_actor"}],
            )
    finally:
        conn.close()


def test_feature_builder_29_day_solar_prefix_invariance():
    """
    Replacing any values in the preceding 29 days' solar cells before FeatureBuilder
    leaves F3 and F2 features at anchor T completely invariant (max abs diff == 0.0).
    """
    fb_f3 = joblib.load(ARTIFACT_DIR / "feature_builder_f3.joblib")
    fb_f2 = joblib.load(ARTIFACT_DIR / "feature_builder_f2.joblib")
    df_clean = pd.read_csv(SAMPLE_CSV).tail(35).copy().reset_index(drop=True)
    df_perturbed = df_clean.copy()
    for code in SOLAR_11_EN_VARS:
        vi_col = SOLAR_EN_TO_VI[code]
        df_perturbed.iloc[:-1, df_perturbed.columns.get_loc(vi_col)] = 1.23456

    f3_clean = fb_f3.build_single_step(df_clean)
    f3_pert = fb_f3.build_single_step(df_perturbed)
    f3_num = f3_clean.select_dtypes(include=[np.number]).to_numpy() - f3_pert.select_dtypes(include=[np.number]).to_numpy()
    assert float(np.max(np.abs(f3_num))) == 0.0

    f2_clean = fb_f2.build_single_step(df_clean)
    f2_pert = fb_f2.build_single_step(df_perturbed)
    f2_num = f2_clean.select_dtypes(include=[np.number]).to_numpy() - f2_pert.select_dtypes(include=[np.number]).to_numpy()
    assert float(np.max(np.abs(f2_num))) == 0.0


def test_selective_per_variable_fill_and_leap_day_2024_02_29():
    """
    Present FLASHFlux values at anchor T remain untouched while only missing CERES columns are
    imputed; explicit leap-year Feb 29 (tm_yday=60) and Dec 31 (tm_yday=366) resolve cleanly;
    and missing METEO_24 columns fail fast instead of silently imputing.
    """
    doy_table, table_sha = load_solar_doy_climatology()
    df_base = pd.read_csv(SAMPLE_CSV).tail(35).copy().reset_index(drop=True)

    # Remap dates to end on leap-year Dec 31 (2024-12-31, tm_yday == 366) and separately check 2024-02-29
    for end_str, expected_tm_yday in [("2024-02-29", "60"), ("2024-12-31", "366")]:
        end_dt = pd.to_datetime(end_str)
        df_case = df_base.copy()
        df_case["Ngày"] = [(end_dt - timedelta(days=34 - idx)).strftime("%Y-%m-%d") for idx in range(35)]

        # Keep FLASHFlux_3 present at anchor T, drop only CERES_ONLY_8 at anchor T
        anchor_idx = df_case.index[-1]
        for code in CERES_ONLY_8_VARS:
            df_case.loc[anchor_idx, SOLAR_EN_TO_VI[code]] = np.nan

        orig_ff = {code: float(df_case.loc[anchor_idx, SOLAR_EN_TO_VI[code]]) for code in FLASHFLUX_3_VARS}

        outcome = apply_serving_solar_doy_fallback(df_case, doy_table=doy_table, doy_table_sha256=table_sha)
        assert outcome.is_degraded_input is True
        assert outcome.ceres_policy_applied == PC_CERES_POLICY_NAME
        assert outcome.imputed_variables == sorted(CERES_ONLY_8_VARS)
        assert outcome.doy_table_sha256 == table_sha

        filled_anchor = outcome.filled_df.iloc[-1]
        for code in FLASHFLUX_3_VARS:
            assert float(filled_anchor[SOLAR_EN_TO_VI[code]]) == pytest.approx(orig_ff[code], abs=1e-9)
        for code in CERES_ONLY_8_VARS:
            expected_med = round(float(doy_table["doy_medians"][code][expected_tm_yday]), 6)
            assert float(filled_anchor[SOLAR_EN_TO_VI[code]]) == pytest.approx(expected_med, abs=1e-9)
            assert outcome.imputed_values[code] == pytest.approx(expected_med, abs=1e-9)

    # Verify StandaloneRainfallPredictor fails fast when any METEO_24 variable is NaN
    predictor = StandaloneRainfallPredictor(artifact_dir=ARTIFACT_DIR, mode="local")
    df_bad_meteo = df_base.copy()
    df_bad_meteo.loc[df_bad_meteo.index[-1], "Nhiệt độ 2m"] = np.nan
    with pytest.raises(ServingValidationError, match="INVALID_OR_MISSING_METEO_INPUT"):
        predictor.predict(df_bad_meteo, run_date=pd.to_datetime(df_bad_meteo["Ngày"]).max().strftime("%Y-%m-%d"))
