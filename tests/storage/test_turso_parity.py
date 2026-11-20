"""Round 11 Step 3 (Part D) Tests: Local / Turso Replica / Turso Remote Parity, ACID Atomic Transactions & Cache Invalidation."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.turso_backend import (  # noqa: E402
    TursoHranaClient,
    TursoRemoteConnection,
    TursoReplicaManager,
    _clear_replica_registry_for_tests,
    open_store_connection,
    resolve_store_backend,
)
from src.data.vintage_store import (  # noqa: E402
    ALL_35_VARS,
    cached_db_query,
    clear_db_query_cache,
    connect_sqlite,
    get_db_state_cache_key,
    init_schema,
)


class _FakeHranaServer:
    """In-memory SQLite-backed simulator implementing Turso Hrana HTTP v2 /v2/pipeline semantics."""

    def __init__(self, backing_db_path: Path) -> None:
        self.backing_db_path = backing_db_path
        self._batons: Dict[str, sqlite3.Connection] = {}
        self.request_count = 0
        self.transient_failures_remaining = 0
        self.last_auth_headers: List[str] = []

    def _decode_arg(self, arg: Dict[str, Any]) -> Any:
        t = arg.get("type")
        if t == "null":
            return None
        if t == "integer":
            return int(arg["value"])
        if t == "float":
            return float(arg["value"])
        if t == "blob":
            return base64.b64decode(arg.get("base64", ""))
        return arg.get("value")

    def _encode_cell(self, val: Any) -> Dict[str, Any]:
        if val is None:
            return {"type": "null"}
        if isinstance(val, bool):
            return {"type": "integer", "value": "1" if val else "0"}
        if isinstance(val, int):
            return {"type": "integer", "value": str(val)}
        if isinstance(val, float):
            return {"type": "float", "value": val}
        if isinstance(val, (bytes, bytearray)):
            return {"type": "blob", "base64": base64.b64encode(bytes(val)).decode("ascii")}
        return {"type": "text", "value": str(val)}

    def handle_post(self, url: str, headers: Dict[str, str], json_payload: Dict[str, Any]):
        self.request_count += 1
        auth = headers.get("Authorization", "")
        self.last_auth_headers.append(auth)

        class _Resp:
            def __init__(self, code: int, body: Any):
                self.status_code = code
                self._body = body
                self.text = json.dumps(body)

            def json(self):
                return self._body

        if self.transient_failures_remaining > 0:
            self.transient_failures_remaining -= 1
            return _Resp(503, {"error": "transient"})

        baton = json_payload.get("baton")
        if baton and baton in self._batons:
            conn = self._batons[baton]
            keep_open = True
        else:
            conn = sqlite3.connect(str(self.backing_db_path), isolation_level=None)
            conn.row_factory = sqlite3.Row
            keep_open = False

        results = []
        closed = False
        for req in json_payload.get("requests", []):
            rtype = req.get("type")
            if rtype == "close":
                closed = True
                results.append({"type": "ok", "response": {"type": "close"}})
                continue
            stmt = req.get("stmt") or {}
            sql = stmt["sql"]
            args = [self._decode_arg(a) for a in stmt.get("args", [])]
            try:
                cur = conn.execute(sql, args)
                cols = [{"name": d[0]} for d in (cur.description or [])]
                fetched = cur.fetchall() if cur.description else []
                rows_json = [[self._encode_cell(cell) for cell in row] for row in fetched]
                rows_read_est = max(len(fetched), 1 if cur.description else 0)
                rows_written_est = max(0, int(cur.rowcount or 0)) if not cur.description else 0
                results.append(
                    {
                        "type": "ok",
                        "response": {
                            "type": "execute",
                            "result": {
                                "cols": cols,
                                "rows": rows_json,
                                "affected_row_count": cur.rowcount if cur.rowcount >= 0 else 0,
                                "last_insert_rowid": cur.lastrowid,
                                "rows_read": rows_read_est,
                                "rows_written": rows_written_est,
                                "query_duration_ms": 0.4,
                            },
                        },
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "type": "error",
                        "error": {"message": str(exc)},
                    }
                )

        if closed:
            if baton and baton in self._batons:
                self._batons.pop(baton, None)
            conn.close()
            out_baton = None
        else:
            if not keep_open:
                baton = f"baton_{len(self._batons) + 1}"
                self._batons[baton] = conn
            out_baton = baton

        return _Resp(200, {"baton": out_baton, "results": results})


def _seed_minimal_store(db_path: Path) -> None:
    conn = connect_sqlite(db_path, read_only=False)
    try:
        init_schema(conn)
        conn.execute(
            """
            INSERT INTO ingestion_runs (
                run_id, run_type, started_at_utc, finished_at_utc, status, location_id,
                scan_window_start, scan_window_end, rows_scanned, rows_inserted_v0,
                rows_new_vintage, rows_late_fill_only, rows_value_change, rows_unchanged,
                rows_unpublished_skipped, git_sha, config_sha256, error_message
            ) VALUES (
                'run_seed_1', 'manual_scan', '2026-11-18T00:00:00Z', '2026-11-18T00:01:00Z',
                'SUCCESS', 'HCMC', '2026-08-30', '2026-08-30', 1, 1, 0, 0, 0, 0, 0,
                'testsha', 'abc123', NULL
            )
            """
        )
        col_list = ", ".join(ALL_35_VARS)
        vals = ", ".join(["1.5"] * len(ALL_35_VARS))
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
                'HCMC', '2026-08-30', 0, '2026-11-18T00:00:30Z', 'run_seed_1',
                80, 'NASA_POWER_API', 0, 0, 0, 0, 'COMPLETE_ALL_35',
                '[]', 'INITIAL_NRT', 0, '[]', 0, '{{}}', 0, '{{}}',
                'seedhash01', {vals}
            )
            """
        )
    finally:
        conn.close()


def test_store_backend_resolution_and_validation(monkeypatch):
    monkeypatch.delenv("STORE_BACKEND", raising=False)
    assert resolve_store_backend() == "local"
    for mode in ("local", "turso_replica", "turso_remote"):
        monkeypatch.setenv("STORE_BACKEND", mode)
        assert resolve_store_backend() == mode
    monkeypatch.setenv("STORE_BACKEND", "invalid_mode")
    with pytest.raises(ValueError, match="Unsupported STORE_BACKEND"):
        resolve_store_backend()


def test_hrana_client_readonly_token_and_retry_accounting(monkeypatch):
    monkeypatch.setenv("TURSO_DB_URL", "libsql://demo-test.turso.io")
    monkeypatch.setenv("TURSO_DB_TOKEN", "rw_secret_tok")
    monkeypatch.setenv("TURSO_READONLY_AUTH_TOKEN", "ro_secret_tok")

    tmp_calls: List[str] = []

    class _FakeResp:
        def __init__(self, status_code: int, payload: Dict[str, Any]):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    attempts = {"n": 0}

    def _fake_post(self, url, json=None, timeout=None):
        tmp_calls.append(str(self.headers.get("Authorization")))
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _FakeResp(503, {"error": "temporary"})
        return _FakeResp(
            200,
            {
                "results": [
                    {
                        "type": "ok",
                        "response": {
                            "type": "execute",
                            "result": {
                                "cols": [{"name": "c"}],
                                "rows": [[{"type": "integer", "value": "1"}]],
                                "affected_row_count": 0,
                                "rows_read": 7,
                                "rows_written": 2,
                                "query_duration_ms": 1.2,
                            },
                        },
                    }
                ]
            },
        )

    import requests

    monkeypatch.setattr(requests.Session, "post", _fake_post)

    ro_client = TursoHranaClient(read_only=True, backoff_base_sec=0.01)
    rw_client = TursoHranaClient(read_only=False, backoff_base_sec=0.01)
    ro_conn = TursoRemoteConnection(client=ro_client, read_only=True)
    row = ro_conn.execute("SELECT 1 AS c").fetchone()
    assert row is not None and row["c"] == 1
    assert ro_client.usage_snapshot().total_rows_read == 7
    assert ro_client.usage_snapshot().total_rows_written == 2
    assert tmp_calls[-1] == "Bearer ro_secret_tok"
    assert rw_client._auth_header == "Bearer rw_secret_tok"

    with pytest.raises(sqlite3.OperationalError, match="readonly database"):
        ro_conn.execute("DELETE FROM observations_vintage")
    ro_conn.close()


def test_turso_remote_atomic_transactions_triggers_and_fk_parity(tmp_path: Path, monkeypatch):
    backing = tmp_path / "backing.db"
    _seed_minimal_store(backing)
    sim = _FakeHranaServer(backing)

    import requests

    monkeypatch.setattr(
        requests.Session,
        "post",
        lambda sess, url, json=None, timeout=None: sim.handle_post(url, dict(sess.headers), json or {}),
    )
    monkeypatch.setenv("TURSO_DB_URL", "libsql://unit-test.turso.io")
    monkeypatch.setenv("TURSO_DB_TOKEN", "rw_tok")

    conn = TursoRemoteConnection(read_only=False)
    try:
        # 1. Immutability triggers enforced identically to SQLite
        with pytest.raises(sqlite3.IntegrityError, match="IMMUTABLE_VIOLATION"):
            conn.execute("UPDATE observations_vintage SET PRECTOTCORR = 9.9 WHERE obs_date = '2026-08-30'")
        with pytest.raises(sqlite3.IntegrityError, match="IMMUTABLE_VIOLATION"):
            conn.execute("DELETE FROM ingestion_runs WHERE run_id = 'run_seed_1'")

        # 2. Atomic batch rollback: valid stmt #1 followed by invalid FK/duplicate in stmt #2 rolls back #1
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute_batch(
                [
                    (
                        """
                        INSERT INTO ingestion_runs (
                            run_id, run_type, started_at_utc, finished_at_utc, status, location_id,
                            scan_window_start, scan_window_end, git_sha, config_sha256
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "run_atomic_abort",
                            "manual_scan",
                            "2026-11-19T00:00:00Z",
                            "2026-11-19T00:00:01Z",
                            "SUCCESS",
                            "HCMC",
                            "2026-09-01",
                            "2026-09-01",
                            "sha",
                            "cfg",
                        ),
                    ),
                    (
                        "UPDATE observations_vintage SET PRECTOTCORR = 1.0",
                        (),
                    ),
                ],
                atomic=True,
            )
        aborted = conn.execute(
            "SELECT COUNT(*) FROM ingestion_runs WHERE run_id = 'run_atomic_abort'"
        ).fetchone()[0]
        assert aborted == 0

        # 4. Nested outer BEGIN IMMEDIATE + executemany() + COMMIT (matches ingest_nasa_power_window path)
        from src.data.vintage_store import ingest_nasa_power_window

        def _mock_win(*, start_date: str, end_date: str, latitude: float, longitude: float, **kw):
            del latitude, longitude, kw
            return {
                "header": {"api": {"version": "v2.10.3"}, "sources": ["MERRA2"]},
                "properties": {
                    "parameter": {v: {start_date.replace("-", ""): 3.25} for v in ALL_35_VARS}
                },
            }

        ing = ingest_nasa_power_window(
            conn,
            start_date="2026-09-01",
            end_date="2026-09-01",
            location_id="HCMC",
            run_type="manual_scan",
            fetcher=_mock_win,
        )
        assert ing["rows_inserted_v0"] == 1
        assert conn.execute(
            "SELECT PRECTOTCORR FROM observations_vintage WHERE obs_date = '2026-09-01'"
        ).fetchone()[0] == pytest.approx(3.25)

        # 5. PRAGMA index_list + V3.3 schema_migrations parity
        mig = conn.execute("SELECT version FROM schema_migrations WHERE version = 'V3.3'").fetchone()
        assert mig is not None
        idx_names = {r[1] for r in conn.execute("PRAGMA index_list(observations_vintage)").fetchall()}
        assert "idx_obs_vintage_loc_date_ingested" in idx_names
    finally:
        conn.close()


def test_turso_replica_sync_cache_invalidation_and_background_timer(tmp_path: Path, monkeypatch):
    _clear_replica_registry_for_tests()
    clear_db_query_cache()
    backing = tmp_path / "remote_primary.db"
    replica_db = tmp_path / "local_replica.db"
    _seed_minimal_store(backing)
    sim = _FakeHranaServer(backing)

    import requests

    monkeypatch.setattr(
        requests.Session,
        "post",
        lambda sess, url, json=None, timeout=None: sim.handle_post(url, dict(sess.headers), json or {}),
    )
    monkeypatch.setenv("TURSO_DB_URL", "libsql://unit-test.turso.io")
    monkeypatch.setenv("TURSO_DB_TOKEN", "rw_tok")
    monkeypatch.setenv("TURSO_REPLICA_PATH", str(replica_db))
    monkeypatch.setenv("STORE_BACKEND", "turso_replica")

    mgr = TursoReplicaManager(
        replica_path=replica_db,
        on_synced_callback=clear_db_query_cache,
    )
    sync_stats = mgr.sync_now()
    assert sync_stats["sync_count"] == 1
    assert replica_db.exists()

    # Check read-only replica query + cache invalidation via state_key
    ro_conn, backend_name = open_store_connection(replica_db, read_only=True, backend="turso_replica")
    try:
        assert backend_name == "turso_replica"
        key1 = get_db_state_cache_key(ro_conn, str(replica_db))
        calls = {"n": 0}

        def _compute():
            calls["n"] += 1
            return ro_conn.execute("SELECT COUNT(*) FROM observations_vintage").fetchone()[0]

        assert cached_db_query((key1, "cnt"), _compute) == 1
        assert cached_db_query((key1, "cnt"), _compute) == 1
        assert calls["n"] == 1  # Cache hit skipped recomputation
    finally:
        ro_conn.close()

    # Write a new run directly to remote primary -> re-sync invalidates cache
    b_conn = connect_sqlite(backing, read_only=False)
    try:
        b_conn.execute(
            """
            INSERT INTO ingestion_runs (
                run_id, run_type, started_at_utc, finished_at_utc, status, location_id,
                scan_window_start, scan_window_end, git_sha, config_sha256
            ) VALUES (
                'run_seed_2', 'manual_scan', '2026-11-19T00:00:00Z', '2026-11-19T00:01:00Z',
                'SUCCESS', 'HCMC', '2026-08-31', '2026-08-31', 'testsha', 'abc123'
            )
            """
        )
    finally:
        b_conn.close()

    mgr.sync_now()
    ro_conn2 = connect_sqlite(replica_db, read_only=True)
    try:
        key2 = get_db_state_cache_key(ro_conn2, str(replica_db))
        assert key2 != key1
    finally:
        ro_conn2.close()

    # Background timer cleanly starts and stops
    mgr.start_background_sync(interval_sec=300.0, jitter_sec=30.0)
    time.sleep(0.05)
    mgr.stop_background_sync()
    _clear_replica_registry_for_tests()


def test_turso_api_integration_across_modes(tmp_path: Path, monkeypatch):
    """Verify FastAPI /health, /locations, /observations work cleanly under turso_replica & turso_remote."""
    from fastapi.testclient import TestClient
    import requests
    import src.api.main as api_main
    from src.api.main import app, load_serving_models

    _clear_replica_registry_for_tests()
    clear_db_query_cache()
    backing = tmp_path / "remote_api.db"
    replica_db = tmp_path / "replica_api.db"
    _seed_minimal_store(backing)
    sim = _FakeHranaServer(backing)

    monkeypatch.setattr(
        requests.Session,
        "post",
        lambda sess, url, json=None, timeout=None: sim.handle_post(url, dict(sess.headers), json or {}),
    )
    monkeypatch.setenv("TURSO_DB_URL", "libsql://api-parity.turso.io")
    monkeypatch.setenv("TURSO_DB_TOKEN", "rw_tok")
    monkeypatch.setenv("TURSO_REPLICA_PATH", str(replica_db))

    orig_provider = api_main.history_provider
    try:
        for mode in ("turso_replica", "turso_remote"):
            clear_db_query_cache()
            monkeypatch.setenv("STORE_BACKEND", mode)
            api_main.history_provider = None
            load_serving_models()
            with TestClient(app) as client:
                r_h = client.get("/health")
                r_l = client.get("/locations")
                r_o = client.get("/observations?location_id=HCMC&start_date=2026-08-30&end_date=2026-08-31")
                assert r_h.status_code == 200, (mode, r_h.text)
                assert r_l.status_code == 200, (mode, r_l.text)
                assert r_o.status_code == 200, (mode, r_o.text)
                assert r_h.json()["db_role"] == mode
                assert r_l.json()["db_role"] == mode
                assert r_o.json()["summary"]["published_days"] == 1
    finally:
        monkeypatch.delenv("STORE_BACKEND", raising=False)
        api_main.history_provider = orig_provider
        _clear_replica_registry_for_tests()
        clear_db_query_cache()
        load_serving_models()
