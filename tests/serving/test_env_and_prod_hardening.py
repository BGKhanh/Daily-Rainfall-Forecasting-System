"""Round 15 Pre-Production Deployment Hardening Tests (Tier 1 Offline)."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

from fastapi.testclient import TestClient
import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.api.main as api_main  # noqa: E402
from src.api.schemas import ErrorResponse  # noqa: E402
from src.config.settings import (  # noqa: E402
    compute_config_sha256,
    compute_operational_config_sha256,
    compute_result_config_sha256,
    load_settings,
    resolve_turso_replica_path,
    validate_runtime_env,
)
from src.data.turso_backend import (  # noqa: E402
    TursoHranaClient,
    _clear_replica_registry_for_tests,
    _format_canonical_hash_cell,
    get_replica_manager,
    redact_secrets,
)
from src.data.vintage_store import clear_db_query_cache  # noqa: E402
from tests.storage.test_turso_parity import _FakeHranaServer, _seed_minimal_store  # noqa: E402


def _set_valid_prod_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    base = {
        "APP_ENV": "production",
        "STORE_BACKEND": "turso_replica",
        "TURSO_DB_URL": "libsql://prod-unit-test.turso.io",
        "TURSO_AUTH_TOKEN": "ro_token_example_value_12345",
        "CORS_ALLOWED_ORIGINS": "https://daily-rainfall-forecasting-system.22520630.workers.dev/",
    }
    base.update(overrides)
    for k, v in base.items():
        if v == "__UNSET__":
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)


def test_runtime_env_excluded_from_hashes_and_cors_normalized(monkeypatch: pytest.MonkeyPatch):
    base_cfg = compute_config_sha256()
    base_res = compute_result_config_sha256()
    base_op = compute_operational_config_sha256()

    # Pure Layer-1 RuntimeEnvConfig overrides must leave ALL three config hashes untouched.
    for key, val in (
        ("APP_ENV", "development"),
        ("LOG_LEVEL", "debug"),
        ("ENABLE_API_DOCS", "false"),
        ("SERVE_STATIC_SPA", "true"),
        ("TURSO_REPLICA_PATH", "/tmp/custom_replica.db"),
        ("VINTAGE_DB_PATH", "/tmp/custom_vintage.db"),
    ):
        monkeypatch.setenv(key, val)
    layer1_only = load_settings(env_path=None)
    assert layer1_only.config_sha256 == base_cfg
    assert layer1_only.operational_config_sha256 == base_op
    assert layer1_only.result_config_sha256 == base_res

    _set_valid_prod_env(
        monkeypatch,
        ENABLE_API_DOCS="__UNSET__",
        SERVE_STATIC_SPA="__UNSET__",
        LOG_LEVEL="warning",
        TURSO_REPLICA_PATH="/var/lib/rainfall/replica.db",
        CORS_ALLOWED_ORIGINS=" https://a.workers.dev/ , http://localhost:5173/ , https://a.workers.dev , http://127.0.0.1:4173 ",
    )
    prod_settings = load_settings(env_path=None)
    assert prod_settings.runtime_env.app_env == "production"
    assert prod_settings.runtime_env.log_level == "WARNING"
    assert prod_settings.runtime_env.enable_api_docs is False
    assert prod_settings.runtime_env.serve_static_spa is False
    assert prod_settings.operational.cors_allowed_origins == ["https://a.workers.dev"]
    assert prod_settings.result_config_sha256 == base_res

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173/, https://b.workers.dev/ ")
    dev_settings = load_settings(env_path=None)
    assert dev_settings.runtime_env.enable_api_docs is True
    assert dev_settings.operational.cors_allowed_origins == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "https://b.workers.dev",
    ]
    assert dev_settings.result_config_sha256 == base_res
    assert base_cfg != base_op


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"TURSO_DB_URL": "__UNSET__"}, "TURSO_DB_URL"),
        ({"TURSO_DB_URL": "libsql://your-db-your-org.turso.io"}, "TURSO_DB_URL"),
        ({"TURSO_AUTH_TOKEN": "__UNSET__"}, "token"),
        ({"TURSO_AUTH_TOKEN": "changeme"}, "token"),
        ({"CORS_ALLOWED_ORIGINS": "__UNSET__"}, "CORS"),
        ({"CORS_ALLOWED_ORIGINS": "http://localhost:5173,http://127.0.0.1:4173"}, "CORS"),
        ({"CORS_ALLOWED_ORIGINS": "*"}, "CORS"),
        ({"CORS_ALLOWED_ORIGINS": "https://*.workers.dev"}, "CORS"),
        ({"CORS_ALLOWED_ORIGINS": "https://app.workers.dev/subpath"}, "CORS"),
    ],
)
def test_validate_runtime_env_fail_fast_in_production(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, str], match: str
):
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
    monkeypatch.delenv("TURSO_DB_TOKEN", raising=False)
    monkeypatch.delenv("TURSO_READONLY_AUTH_TOKEN", raising=False)
    _set_valid_prod_env(monkeypatch, **overrides)
    settings = load_settings(env_path=None)
    with pytest.raises(ValueError, match=match):
        validate_runtime_env(settings)


def test_turso_sync_interval_override_and_replica_path_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TURSO_SYNC_INTERVAL_SEC", "600")
    s = load_settings(env_path=None)
    assert s.operational.turso_sync_interval_sec == 600

    assert str(resolve_turso_replica_path("/var/lib/rainfall/replica.db")).replace("\\", "/").endswith(
        "/var/lib/rainfall/replica.db"
    )
    rel = resolve_turso_replica_path("data/vintage/turso_replica.db")
    assert rel == (PROJECT_ROOT / "data" / "vintage" / "turso_replica.db").resolve()


def test_redact_secrets_masks_tokens_turso_url_and_paths(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TURSO_DB_URL", "libsql://secret-org-db.turso.io")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "ro_secret_token_999999")
    monkeypatch.setenv("TURSO_REPLICA_PATH", "/var/lib/rainfall/replica.db")

    msg = redact_secrets(
        "Failed https://secret-org-db.turso.io/v2/pipeline Bearer ro_secret_token_999999 at /var/lib/rainfall/replica.db"
    )
    assert "secret-org-db" not in msg
    assert "ro_secret_token_999999" not in msg
    assert "/var/lib/rainfall/replica.db" not in msg
    assert "[REDACTED_TURSO_URL]" in msg
    assert "[REDACTED]" in msg
    assert "[REDACTED_PATH]" in msg


def test_turso_hrana_client_supports_turso_auth_token_and_readonly_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TURSO_DB_URL", "libsql://ro-precedence.turso.io")
    monkeypatch.delenv("TURSO_DB_TOKEN", raising=False)
    monkeypatch.delenv("TURSO_READONLY_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "single_ro_token_12345")

    c1 = TursoHranaClient(read_only=True)
    assert c1._resolved_token == "single_ro_token_12345"
    c1.close()

    monkeypatch.setenv("TURSO_READONLY_AUTH_TOKEN", "dedicated_ro_token_67890")
    c2 = TursoHranaClient(read_only=True)
    assert c2._resolved_token == "dedicated_ro_token_67890"
    c2.close()


def test_production_app_hides_docs_normalizes_cors_and_sanitizes_500(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_valid_prod_env(
        monkeypatch,
        STORE_BACKEND="local",
        VINTAGE_DB_PATH=str(PROJECT_ROOT / "artifacts" / "models" / "demo_rainfall_vintage.db"),
        CORS_ALLOWED_ORIGINS="https://daily-rainfall-forecasting-system.22520630.workers.dev/",
    )
    prod_settings = load_settings(env_path=None)
    prod_app = api_main.create_app(prod_settings)

    @prod_app.get("/__test_unhandled_boom")
    def _boom():
        raise sqlite3.OperationalError(
            "SELECT * FROM secrets_leak at libsql://prod-unit-test.turso.io token=ro_token_example_value_12345"
        )

    with TestClient(prod_app, raise_server_exceptions=False) as client:
        for hidden_path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(hidden_path).status_code == 404

        r_health = client.get(
            "/health",
            headers={"Origin": "https://daily-rainfall-forecasting-system.22520630.workers.dev"},
        )
        assert r_health.status_code == 200
        assert (
            r_health.headers.get("access-control-allow-origin")
            == "https://daily-rainfall-forecasting-system.22520630.workers.dev"
        )
        health_txt = json.dumps(r_health.json())
        assert "prod-unit-test.turso.io" not in health_txt
        assert "ro_token_example_value_12345" not in health_txt

        r_err = client.get("/__test_unhandled_boom")
        assert r_err.status_code == 500
        err_body = r_err.json()
        ErrorResponse.model_validate(err_body)
        assert err_body == {
            "error": {
                "code": "SERVER_DB_DATA_ERROR",
                "message": "Internal server error.",
                "details": {},
            }
        }
        assert "SELECT" not in json.dumps(err_body)
        assert "prod-unit-test" not in json.dumps(err_body)


def test_turso_replica_startup_zero_write_sql_and_all_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _clear_replica_registry_for_tests()
    clear_db_query_cache()

    backing = tmp_path / "remote_primary.db"
    replica_db = tmp_path / "replica_local.db"
    _seed_minimal_store(backing)
    sim = _FakeHranaServer(backing)

    sent_sql: list[str] = []

    def _spy_post(sess, url, json=None, timeout=None):
        for req in (json or {}).get("requests", []):
            if req.get("type") == "execute":
                sql_txt = str((req.get("stmt") or {}).get("sql") or "").strip()
                if sql_txt:
                    sent_sql.append(sql_txt)
        return sim.handle_post(url, dict(sess.headers), json or {})

    monkeypatch.setattr(requests.Session, "post", _spy_post)
    monkeypatch.delenv("TURSO_DB_TOKEN", raising=False)
    monkeypatch.delenv("TURSO_READONLY_AUTH_TOKEN", raising=False)
    _set_valid_prod_env(
        monkeypatch,
        APP_ENV="development",
        STORE_BACKEND="turso_replica",
        TURSO_AUTH_TOKEN="ro_token_only_123456",
        TURSO_REPLICA_PATH=str(replica_db),
    )

    orig_provider = api_main.history_provider
    try:
        api_main.history_provider = None
        mgr_pre = get_replica_manager()
        assert mgr_pre.replica_path == replica_db.resolve()

        test_app = api_main.create_app()
        with TestClient(test_app) as client:
            assert sent_sql, "Expected SELECT sync queries sent to Turso Primary"
            for sql in sent_sql:
                assert sql.upper().startswith("SELECT "), f"Non-SELECT SQL leaked to Turso Primary: {sql}"

            r_h = client.get("/health")
            r_l = client.get("/locations")
            r_o = client.get("/observations?location_id=HCMC&start_date=2026-08-30&end_date=2026-08-31")
            r_s = client.get("/scores?eval_split_regime=POST_TEST")
            assert r_h.status_code == 200 and r_h.json()["db_role"] == "turso_replica"
            assert r_h.json()["replica_sync_count"] >= 1
            assert r_h.json()["replica_last_sync_at_utc"] is not None
            assert r_l.status_code == 200
            assert r_o.status_code == 200
            assert r_s.status_code == 200

            from datetime import datetime, timezone
            import pandas as pd
            import src.config.clock as clock_mod

            monkeypatch.setattr(
                clock_mod,
                "now_utc",
                lambda: datetime(2026, 4, 30, 5, 0, 0, tzinfo=timezone.utc),
            )
            df_s = pd.read_csv(PROJECT_ROOT / "artifacts" / "models" / "sample_recent_history_35d.csv")
            r_p = client.post("/predict", json={"history": json.loads(df_s.to_json(orient="records", date_format="iso"))})
            assert r_p.status_code == 200, r_p.text
    finally:
        monkeypatch.delenv("STORE_BACKEND", raising=False)
        monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("TURSO_REPLICA_PATH", raising=False)
        api_main.history_provider = orig_provider
        _clear_replica_registry_for_tests()
        clear_db_query_cache()
        api_main.load_serving_models()


def test_format_canonical_hash_cell_cross_platform_floats():
    assert _format_canonical_hash_cell(None) == ""
    assert _format_canonical_hash_cell(-0.0) == "0.00000000"
    assert _format_canonical_hash_cell(0.6843764053375142) == "0.68437641"
    assert _format_canonical_hash_cell(12) == "12"


def test_readonly_turso_connection_blocks_immediate_begin_and_skips_fk_pragma(
    monkeypatch: pytest.MonkeyPatch,
):
    from src.data.turso_backend import TursoRemoteConnection

    sent_batches = []

    class _DummyClient:
        def pipeline(self, reqs, baton=None, base_url=None):
            sent_batches.append(reqs)
            return {"baton": "b1", "base_url": "https://dummy", "results": [{"type": "ok", "response": {"type": "execute", "result": {"cols": [], "rows": []}}} for _ in reqs]}

        def close(self):
            return None

    monkeypatch.setattr("src.data.turso_backend.PROJECT_ROOT", "/app")
    assert "/application/state" in redact_secrets("open /application/state and /app/replica.db")
    assert "[REDACTED_PATH]/replica.db" in redact_secrets("open /application/state and /app/replica.db")

    conn = TursoRemoteConnection(_DummyClient(), read_only=True)
    conn.execute("PRAGMA foreign_keys = ON;")
    assert not sent_batches
    conn.execute("BEGIN")
    assert len(sent_batches) == 1
    assert [r["stmt"]["sql"] for r in sent_batches[0]] == ["BEGIN"]
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("BEGIN IMMEDIATE")
    conn.close()
