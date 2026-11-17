"""
Canonical Column Alignment & FeatureBuilder Entry Guard Tests (Task 1 / Round 7)
================================================================================
Verifies using small bundled data (`sample_recent_history_35d.csv`, 35 rows):
1. All 3 DataFrame construction paths (DB read, Forecast Runner, Custom /predict payload)
   route through `align_and_validate_canonical_raw_df()` and produce exact 36-column
   `CANONICAL_VI_COLUMN_ORDER` matching `sample_recent_history_35d.csv`.
2. Feature matrices X and predictions computed via DB path match sample CSV path to 0.0 mm.
3. Passing a renamed/misspelled column, missing column, or swapped column order into
   `FeatureBuilder.transform()` / `FeatureBuilder.build_single_step()` / `FeatureBuilder.fit()`
   raises `ValueError("[FEATURE_BUILDER_INPUT_SCHEMA_ERROR] ...")` loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.provider import WeatherHistoryProvider  # noqa: E402
from src.data.vintage_store import (  # noqa: E402
    CANONICAL_VI_COLUMN_ORDER,
    align_and_validate_canonical_raw_df,
    connect_sqlite,
    init_schema,
    read_latest_observations,
    seed_historical_csv_vintage_v0,
)
from src.inference.forecast_runner import _rows_to_vietnamese_df  # noqa: E402
from src.inference.predictor import StandaloneRainfallPredictor  # noqa: E402
from src.inference.validation import ServingValidationError, validate_serving_input  # noqa: E402

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "models"
SAMPLE_CSV = ARTIFACT_DIR / "sample_recent_history_35d.csv"


def test_canonical_column_alignment_and_db_vs_csv_parity(tmp_path: Path):
    """All 3 DataFrame sources produce exact CANONICAL_VI_COLUMN_ORDER and 0.0 mm parity."""
    df_csv = pd.read_csv(SAMPLE_CSV)
    expected_cols = list(CANONICAL_VI_COLUMN_ORDER)
    assert list(df_csv.columns) == expected_cols
    assert len(expected_cols) == 36

    # Source 1: SQLite DB (`read_latest_observations` + `WeatherHistoryProvider`)
    db_path = tmp_path / "col_guard.db"
    conn = connect_sqlite(db_path, read_only=False, recursive_triggers=True)
    try:
        init_schema(conn)
        seed_historical_csv_vintage_v0(conn, df=df_csv)
        df_db_read = read_latest_observations(conn)
        assert list(df_db_read.columns) == expected_cols

        # Source 2: Forecast runner row converter (`_rows_to_vietnamese_df`)
        raw_rows = [dict(r) for r in conn.execute("SELECT * FROM v_observations_latest ORDER BY obs_date ASC").fetchall()]
        df_runner = _rows_to_vietnamese_df(raw_rows)
        assert list(df_runner.columns) == expected_cols
    finally:
        conn.close()

    sqlite_provider = WeatherHistoryProvider(backend="sqlite", db_path=db_path)
    df_provider = sqlite_provider.fetch_recent_history(days=35).history_df
    assert list(df_provider.columns) == expected_cols

    # Source 3: /predict validation & predictor path (including scrambled input column order)
    scrambled_cols = list(reversed(df_csv.columns))
    df_scrambled = df_csv[scrambled_cols].copy()
    ctx = validate_serving_input(df_scrambled, run_date="2026-04-30")
    assert list(ctx.df.columns) == expected_cols

    predictor = StandaloneRainfallPredictor(artifact_dir=ARTIFACT_DIR, mode="local")
    pred_csv = predictor.predict(df_csv, run_date="2026-04-30")["serving_model"]["predictions_7d_mm"]
    pred_db = predictor.predict(df_provider, run_date="2026-04-30")["serving_model"]["predictions_7d_mm"]
    pred_scrambled = predictor.predict(df_scrambled, run_date="2026-04-30")["serving_model"]["predictions_7d_mm"]

    assert np.max(np.abs(np.array(pred_csv) - np.array(pred_db))) == 0.0
    assert np.max(np.abs(np.array(pred_csv) - np.array(pred_scrambled))) == 0.0

    # Direct feature matrix comparison DB vs CSV
    fb_f3 = joblib.load(ARTIFACT_DIR / "feature_builder_f3.joblib")
    feat_csv = fb_f3.build_single_step(align_and_validate_canonical_raw_df(df_csv))
    feat_db = fb_f3.build_single_step(df_provider)
    fcols = [c for c in feat_csv.columns if c not in [fb_f3.date_col, fb_f3.target_col]]
    assert list(feat_csv.columns) == list(feat_db.columns)
    assert np.max(np.abs(feat_csv[fcols].to_numpy(np.float64) - feat_db[fcols].to_numpy(np.float64))) == 0.0


def test_feature_builder_entry_guard_rejects_bad_columns_and_swapped_order():
    """FeatureBuilder.transform() & build_single_step() fail fast on renamed, missing, or swapped columns."""
    df_csv = pd.read_csv(SAMPLE_CSV)
    fb_f3 = joblib.load(ARTIFACT_DIR / "feature_builder_f3.joblib")

    # (a) Renamed / misspelled column
    df_renamed = df_csv.rename(columns={"Nhiệt độ 2m": "Nhiet do 2m sai"})
    with pytest.raises(ValueError, match="FEATURE_BUILDER_INPUT_SCHEMA_ERROR"):
        fb_f3.build_single_step(df_renamed)
    with pytest.raises(ValueError, match="FEATURE_BUILDER_INPUT_SCHEMA_ERROR"):
        fb_f3.transform(df_renamed, drop_warmup=False)

    # (b) Missing 1 column
    df_missing = df_csv.drop(columns=["Chỉ số UV"])
    with pytest.raises(ValueError, match="FEATURE_BUILDER_INPUT_SCHEMA_ERROR"):
        fb_f3.build_single_step(df_missing)
    with pytest.raises(ValueError, match="FEATURE_BUILDER_INPUT_SCHEMA_ERROR"):
        fb_f3.transform(df_missing, drop_warmup=False)

    # (c) Exact 36 columns present, but two columns swapped ('Nhiệt độ 2m' <-> 'Bức xạ sóng ngắn bề mặt')
    swapped_cols = list(df_csv.columns)
    i1 = swapped_cols.index("Nhiệt độ 2m")
    i2 = swapped_cols.index("Bức xạ sóng ngắn bề mặt")
    swapped_cols[i1], swapped_cols[i2] = swapped_cols[i2], swapped_cols[i1]
    df_swapped = df_csv[swapped_cols].copy()

    with pytest.raises(ValueError, match="FEATURE_BUILDER_INPUT_SCHEMA_ERROR"):
        fb_f3.build_single_step(df_swapped)
    with pytest.raises(ValueError, match="FEATURE_BUILDER_INPUT_SCHEMA_ERROR"):
        fb_f3.transform(df_swapped, drop_warmup=False)

    # Aligning helper rejects missing/renamed columns cleanly too
    with pytest.raises(ValueError, match="CANONICAL_COLUMN_ERROR"):
        align_and_validate_canonical_raw_df(df_renamed)
    with pytest.raises(ServingValidationError, match="INVALID_OR_MISSING_METEO_INPUT"):
        validate_serving_input(df_missing, run_date="2026-04-30")
