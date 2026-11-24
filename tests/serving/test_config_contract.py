"""Round 11 Step 1 TDD Seam: 5-Layer Config, Variable Catalog & Startup Model Contract Guard."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import joblib
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import (  # noqa: E402
    DEFAULT_FREEZE_TOML,
    DEFAULT_RUNTIME_TOML,
    compute_config_sha256,
    compute_operational_config_sha256,
    compute_result_config_sha256,
    load_settings,
    verify_model_contract_on_startup,
)
from src.data.vintage_store import (  # noqa: E402
    ALL_35_VARS,
    FLASHFLUX_3_VARS,
    METEO_24_VARS,
    compute_row_value_hash,
    connect_sqlite,
    get_complete_anchor_range,
    get_default_complete_anchor_date,
    init_schema,
)
from src.inference.forecast_runner import list_complete_anchor_dates  # noqa: E402


def test_load_settings_and_variable_catalog_contract():
    settings = load_settings(env_path=None)
    assert settings.operational.store_backend in ("local", "turso_replica", "turso_remote")
    assert settings.application.obs_max_days_daily == 366
    assert settings.model_contract.horizon_days == 7
    assert settings.model_contract.time_standard == "LST"
    assert settings.model_contract.canonical_location_id == "HCMC"
    assert len(settings.model_contract.required_input_variables) == 35
    assert "TOA_SW_DWN" in settings.model_contract.required_input_variables

    codes = {v.code for v in settings.variable_catalog.variables}
    assert len(codes) == 35
    assert "TOA_SW_DWN" in codes
    assert "CLOUD_AMT" not in codes
    assert "ALB" not in codes

    sha1 = compute_config_sha256()
    sha2 = settings.config_sha256
    assert len(sha1) == 64
    assert sha1 == sha2

    frontend_catalog = PROJECT_ROOT / "frontend" / "src" / "data" / "variableCatalog.json"
    backend_catalog = PROJECT_ROOT / "config" / "variable_catalog.json"
    assert json.loads(frontend_catalog.read_text(encoding="utf-8")) == json.loads(
        backend_catalog.read_text(encoding="utf-8")
    )


def test_startup_contract_guard_accepts_packaged_artifacts():
    fb3 = joblib.load(PROJECT_ROOT / "artifacts" / "models" / "feature_builder_f3.joblib")
    fb2 = joblib.load(PROJECT_ROOT / "artifacts" / "models" / "feature_builder_f2.joblib")
    meta = json.loads((PROJECT_ROOT / "artifacts" / "models" / "metadata.json").read_text(encoding="utf-8"))

    contract = verify_model_contract_on_startup(feature_builders=[fb3, fb2], metadata=meta)
    assert contract.horizon_days == 7
    assert contract.canonical_location_id == "HCMC"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda m: m["model_contract"].__setitem__("horizon_days", 5),
        lambda m: m["model_contract"].__setitem__("time_standard", "UTC"),
        lambda m: m["model_contract"].__setitem__("canonical_location_id", "HANOI"),
        lambda m: m["model_contract"]["required_input_variables"].pop(),
        lambda m: m.pop("model_contract"),
    ],
)
def test_startup_contract_guard_rejects_tampered_contract(mutator):
    fb3 = joblib.load(PROJECT_ROOT / "artifacts" / "models" / "feature_builder_f3.joblib")
    meta = copy.deepcopy(
        json.loads((PROJECT_ROOT / "artifacts" / "models" / "metadata.json").read_text(encoding="utf-8"))
    )
    mutator(meta)

    with pytest.raises(RuntimeError, match=r"\[MODEL_CONTRACT_MISMATCH\]"):
        verify_model_contract_on_startup(feature_builders=[fb3], metadata=meta)


def test_migration_v3_3_adds_schema_migrations_and_config_sha256(tmp_path: Path):
    db_path = tmp_path / "test_v33.db"
    conn = connect_sqlite(db_path, read_only=False)
    try:
        init_schema(conn)
        mig_row = conn.execute("SELECT version FROM schema_migrations WHERE version = 'V3.3'").fetchone()
        assert mig_row is not None
        mig_v34 = conn.execute("SELECT version FROM schema_migrations WHERE version = 'V3.4'").fetchone()
        assert mig_v34 is not None

        ir_cols = {r[1] for r in conn.execute("PRAGMA table_info(ingestion_runs)").fetchall()}
        fl_cols = {r[1] for r in conn.execute("PRAGMA table_info(forecasts_ledger)").fetchall()}
        assert "config_sha256" in ir_cols
        assert "operational_config_sha256" in ir_cols
        assert "config_sha256" in fl_cols
        assert "issued_by" in fl_cols
        assert "result_config_sha256" in fl_cols

        idx_names = {
            r[1] for r in conn.execute("PRAGMA index_list(observations_vintage)").fetchall()
        }
        assert "idx_obs_vintage_loc_date_ingested" in idx_names
    finally:
        conn.close()


def test_dual_config_hashes_isolate_operational_vs_result_layers(tmp_path: Path):
    base_res = compute_result_config_sha256()
    base_op = compute_operational_config_sha256()
    assert len(base_res) == 64 and len(base_op) == 64
    assert base_res != base_op

    # Mutate Layer 2/3 runtime TOML -> operational_config_sha256 MUST change, result_config_sha256 MUST stay identical
    rt_copy = tmp_path / "runtime_mod.toml"
    rt_text = DEFAULT_RUNTIME_TOML.read_text(encoding="utf-8")
    rt_copy.write_text(
        rt_text.replace("sqlite_cache_size_kb = 16000", "sqlite_cache_size_kb = 32000"),
        encoding="utf-8",
    )
    assert compute_operational_config_sha256(runtime_toml=rt_copy) != base_op
    assert compute_result_config_sha256(runtime_toml=rt_copy) == base_res

    # Mutate Layer 5 freeze TOML -> result_config_sha256 MUST change, operational_config_sha256 MUST stay identical
    fz_copy = tmp_path / "freeze_mod.toml"
    fz_text = DEFAULT_FREEZE_TOML.read_text(encoding="utf-8")
    fz_copy.write_text(
        fz_text.replace("tweedie_variance_power = 1.5", "tweedie_variance_power = 1.6"),
        encoding="utf-8",
    )
    assert compute_result_config_sha256(freeze_toml=fz_copy) != base_res
    assert compute_operational_config_sha256(freeze_toml=fz_copy) == base_op


def test_anchor_completeness_with_subset_model_contract_fewer_than_35_vars(tmp_path: Path):
    from datetime import date, timedelta

    db_path = tmp_path / "subset_27.db"
    conn = connect_sqlite(db_path, read_only=False)
    try:
        init_schema(conn)
        conn.execute(
            """
            INSERT INTO ingestion_runs (
                run_id, run_type, started_at_utc, finished_at_utc, status, location_id,
                scan_window_start, scan_window_end, rows_scanned, rows_inserted_v0,
                git_sha, config_sha256, operational_config_sha256
            ) VALUES (
                'subset_seed', 'manual_scan', '2026-11-15T00:00:00Z', '2026-11-15T00:01:00Z',
                'SUCCESS', 'HCMC', '2026-10-05', '2026-11-13', 40, 40, 'sha', 'cfg', 'opcfg'
            )
            """
        )
        vars_27 = list(FLASHFLUX_3_VARS + METEO_24_VARS)
        col_list = ", ".join(ALL_35_VARS)
        ph = ", ".join(["?"] * len(ALL_35_VARS))
        d0 = date(2026, 10, 5)
        for i in range(40):
            d_iso = (d0 + timedelta(days=i)).isoformat()
            if i <= 26:  # 2026-10-05..2026-10-31: COMPLETE_ALL_35
                vals_map = {c: 1.5 for c in ALL_35_VARS}
                m_cnt, m_cer, regime = 0, 0, "COMPLETE_ALL_35"
            elif i <= 38:  # 2026-11-01..2026-11-12: MISSING_CERES_ONLY (27 FlashFlux+MERRA2 intact)
                vals_map = {c: (1.5 if c in vars_27 else None) for c in ALL_35_VARS}
                m_cnt, m_cer, regime = 8, 8, "MISSING_CERES_ONLY"
            else:  # 2026-11-13: INCOMPLETE_METEO (PRECTOTCORR=None)
                vals_map = {c: (None if c == "PRECTOTCORR" else (1.5 if c in vars_27 else None)) for c in ALL_35_VARS}
                m_cnt, m_cer, regime = 9, 8, "INCOMPLETE_METEO"
            rhash = compute_row_value_hash(vals_map)
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
                    'HCMC', ?, 0, '2026-11-15T00:00:30Z', 'subset_seed',
                    2, 'NASA_POWER_API', ?, ?, 0, ?, ?,
                    '[]', 'INITIAL_NRT', 0, '[]', 0, '{{}}', 0, '{{}}', ?, {ph}
                )
                """,
                (
                    d_iso,
                    m_cnt,
                    1 if regime == "INCOMPLETE_METEO" else 0,
                    m_cer,
                    regime,
                    rhash,
                    *(vals_map[c] for c in ALL_35_VARS),
                ),
            )
        conn.commit()

        # Full 35-var contract stops at 2026-10-31 (0 COMPLETE_ALL_35 with 30-day lookback -> None since only 27 complete days)
        assert get_default_complete_anchor_date(conn, location_id="HCMC") is None
        # 27-var subset advances latest complete anchor right into 2026-11-12!
        assert get_default_complete_anchor_date(conn, location_id="HCMC", required_vars=vars_27) == "2026-11-12"
        rng_27 = get_complete_anchor_range(conn, location_id="HCMC", window_days=30, required_vars=vars_27)
        assert rng_27 == {"min": "2026-11-03", "max": "2026-11-12"}
        anchors_27 = list_complete_anchor_dates(
            conn,
            start_date="2026-11-01",
            end_date="2026-11-13",
            location_id="HCMC",
            required_vars=vars_27,
        )
        assert anchors_27[-1] == "2026-11-12"
        assert "2026-11-13" not in anchors_27
    finally:
        conn.close()


def test_layer5_freeze_toml_matches_metadata_key_for_key():
    """Verify Layer 5 frozen research TOML matches metadata.json['frozen_eda_modeling_config'] key-for-key."""
    import tomllib

    freeze_cfg = tomllib.loads(DEFAULT_FREEZE_TOML.read_text(encoding="utf-8"))["freeze"]
    meta_path = PROJECT_ROOT / "artifacts" / "models" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frozen_meta = meta["frozen_eda_modeling_config"]

    assert set(freeze_cfg.keys()) == set(frozen_meta.keys())
    for key, val in frozen_meta.items():
        if isinstance(val, float):
            assert freeze_cfg[key] == pytest.approx(val)
        else:
            assert freeze_cfg[key] == val


def test_config_polygon_to_location_matches_frontend_map():
    """Verify decoupled backend config/polygon_to_location.json matches frontend map copy bit-for-bit."""
    cfg_map = PROJECT_ROOT / "config" / "polygon_to_location.json"
    fe_map = PROJECT_ROOT / "frontend" / "public" / "map" / "polygon_to_location.json"
    assert cfg_map.exists(), f"Missing decoupled backend catalog: {cfg_map}"
    assert json.loads(cfg_map.read_text(encoding="utf-8")) == json.loads(
        fe_map.read_text(encoding="utf-8")
    )
