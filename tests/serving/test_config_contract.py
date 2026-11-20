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
    compute_config_sha256,
    load_settings,
    verify_model_contract_on_startup,
)
from src.data.vintage_store import connect_sqlite, init_schema  # noqa: E402


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

        ir_cols = {r[1] for r in conn.execute("PRAGMA table_info(ingestion_runs)").fetchall()}
        fl_cols = {r[1] for r in conn.execute("PRAGMA table_info(forecasts_ledger)").fetchall()}
        assert "config_sha256" in ir_cols
        assert "config_sha256" in fl_cols

        idx_names = {
            r[1] for r in conn.execute("PRAGMA index_list(observations_vintage)").fetchall()
        }
        assert "idx_obs_vintage_loc_date_ingested" in idx_names
    finally:
        conn.close()
