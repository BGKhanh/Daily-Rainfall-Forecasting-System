"""TDD tests for Round 15.1: remote-safe finalize_db_for_read_only and post-commit failure isolation."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.turso_backend import TursoRemoteConnection  # noqa: E402
from src.data.vintage_store import (  # noqa: E402
    ALL_35_VARS,
    connect_sqlite,
    finalize_db_for_read_only,
    ingest_nasa_power_window,
    init_schema,
)


def _valid_nasa_payload(obs_date_compact: str = "20261120") -> Dict[str, Any]:
    params: Dict[str, Dict[str, float]] = {}
    for idx, var in enumerate(ALL_35_VARS):
        if var == "PRECTOTCORR":
            params[var] = {obs_date_compact: 1.5}
        elif var in ("ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN"):
            params[var] = {obs_date_compact: 18.0}
        else:
            params[var] = {obs_date_compact: float(10 + idx)}
    return {
        "header": {"api": {"version": "2.6.0"}, "sources": ["MERRA2", "CERES"]},
        "properties": {"parameter": params},
    }


def _hrana_ok_result(cols: List[str], rows: List[tuple]) -> Dict[str, Any]:
    def _encode_cell(v: Any) -> Dict[str, Any]:
        if v is None:
            return {"type": "null"}
        if isinstance(v, int):
            return {"type": "integer", "value": str(v)}
        if isinstance(v, float):
            return {"type": "float", "value": v}
        return {"type": "text", "value": str(v)}

    return {
        "type": "ok",
        "response": {
            "type": "execute",
            "result": {
                "cols": [{"name": c, "decltype": None} for c in cols],
                "rows": [[_encode_cell(c) for c in r] for r in rows],
            },
        },
    }


class _SimulatedRemoteHranaClient:
    """In-process Hrana pipeline backed by a local SQLite DB that rejects Turso-disallowed PRAGMAs."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
        self.sent_sql: List[str] = []
        self.last_rows_read = 0
        self.last_rows_written = 0

    def pipeline(
        self,
        requests_list: List[Dict[str, Any]],
        baton: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for req in requests_list:
            if req.get("type") == "close":
                results.append({"type": "ok", "response": {"type": "close"}})
                continue
            stmt = req["stmt"]
            sql = stmt["sql"]
            self.sent_sql.append(sql)
            if re.search(r"PRAGMA\s+(?:wal_checkpoint|journal_mode)\b", sql, re.IGNORECASE):
                raise sqlite3.OperationalError("SQL not allowed statement")

            args = []
            for arg in stmt.get("args", []):
                t = arg.get("type")
                if t == "null":
                    args.append(None)
                elif t == "integer":
                    args.append(int(arg["value"]))
                elif t == "float":
                    args.append(float(arg["value"]))
                else:
                    args.append(arg.get("value"))

            cur = self._conn.execute(sql, args)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall() if cur.description else []
            results.append(_hrana_ok_result(cols, rows))

        return {"baton": "sim-baton", "base_url": "https://sim.turso.local", "results": results}

    def close(self) -> None:
        self._conn.close()


def test_finalize_is_noop_for_non_sqlite_connection(tmp_path: Path) -> None:
    mock_conn = MagicMock()
    finalize_db_for_read_only(mock_conn)
    mock_conn.execute.assert_not_called()

    sim_client = _SimulatedRemoteHranaClient(tmp_path / "sim_noop.db")
    try:
        remote_conn = TursoRemoteConnection(sim_client, read_only=False)
        finalize_db_for_read_only(remote_conn)
        assert sim_client.sent_sql == []
        remote_conn.close()
    finally:
        sim_client.close()


def test_ingest_success_not_followed_by_failed_row_when_finalize_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "vintage_finalize_err.db"
    conn = connect_sqlite(db_path, read_only=False)
    try:
        init_schema(conn, seed_hcmc_location=True)

        def _boom_finalize(_conn: Any) -> None:
            raise sqlite3.OperationalError("SQL not allowed statement")

        monkeypatch.setattr("src.data.vintage_store.finalize_db_for_read_only", _boom_finalize)

        res = ingest_nasa_power_window(
            conn,
            start_date="2026-11-20",
            end_date="2026-11-20",
            fetcher=lambda **_: _valid_nasa_payload("20261120"),
            ingested_at_utc="2026-11-25T03:00:00Z",
        )
        assert res["status"] == "SUCCESS"

        runs = conn.execute(
            "SELECT run_id, status FROM ingestion_runs WHERE run_id = ?",
            (res["run_id"],),
        ).fetchall()
        assert len(runs) == 1
        assert runs[0]["status"] == "SUCCESS"

        obs = conn.execute(
            "SELECT obs_date, PRECTOTCORR FROM observations_vintage WHERE obs_date = '2026-11-20'"
        ).fetchall()
        assert len(obs) == 1
        assert float(obs[0]["PRECTOTCORR"]) == pytest.approx(1.5)
    finally:
        conn.close()


def test_ingest_failure_writes_single_failed_row_and_reraises_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "vintage_precommit_err.db"
    conn = connect_sqlite(db_path, read_only=False)
    try:
        init_schema(conn, seed_hcmc_location=True)
        monkeypatch.setenv("TURSO_DB_TOKEN", "secret_rw_token_should_not_leak")

        def _leaky_fetcher(**_kwargs: Any) -> Dict[str, Any]:
            raise RuntimeError(
                "NASA fetch failed via libsql://secret-db.turso.io token=secret_rw_token_should_not_leak"
            )

        with pytest.raises(RuntimeError, match="NASA fetch failed"):
            ingest_nasa_power_window(
                conn,
                start_date="2026-11-20",
                end_date="2026-11-20",
                fetcher=_leaky_fetcher,
                ingested_at_utc="2026-11-25T03:00:00Z",
            )

        runs = conn.execute("SELECT run_id, status, error_message FROM ingestion_runs").fetchall()
        assert len(runs) == 1
        assert runs[0]["status"] == "FAILED"
        err_msg = str(runs[0]["error_message"])
        assert "NASA fetch failed" in err_msg
        assert "secret_rw_token_should_not_leak" not in err_msg
        assert "secret-db.turso.io" not in err_msg
        assert "IMMUTABLE_VIOLATION" not in err_msg
    finally:
        conn.close()


def test_ingest_failure_preserves_original_error_when_failed_audit_insert_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "vintage_audit_fail.db"
    conn = connect_sqlite(db_path, read_only=False)
    try:
        init_schema(conn, seed_hcmc_location=True)

        class _BrokenAuditConn:
            def __init__(self, inner: sqlite3.Connection) -> None:
                self._inner = inner

            def execute(self, sql: str, params: Any = ()) -> Any:
                if "VALUES (?, ?, ?, ?, 'FAILED'" in sql:
                    raise sqlite3.IntegrityError("IMMUTABLE_VIOLATION: simulated audit insert failure")
                return self._inner.execute(sql, params)

            def executemany(self, sql: str, seq: Any) -> Any:
                return self._inner.executemany(sql, seq)

        wrapped = _BrokenAuditConn(conn)
        with pytest.raises(ValueError, match="No valid observation dates returned by NASA POWER"):
            ingest_nasa_power_window(
                wrapped,  # type: ignore[arg-type]
                start_date="2026-11-20",
                end_date="2026-11-20",
                fetcher=lambda **_: {"header": {}, "properties": {"parameter": {"PRECTOTCORR": {}}}},
                ingested_at_utc="2026-11-25T03:00:00Z",
            )
    finally:
        conn.close()


def test_ingest_remote_simulation_blocks_wal_and_journal_pragmas(tmp_path: Path) -> None:
    db_path = tmp_path / "sim_remote_ingest.db"
    local_init = connect_sqlite(db_path, read_only=False)
    init_schema(local_init, seed_hcmc_location=True)
    local_init.close()

    sim_client = _SimulatedRemoteHranaClient(db_path)
    remote_conn = TursoRemoteConnection(sim_client, read_only=False)
    try:
        res = ingest_nasa_power_window(
            remote_conn,
            start_date="2026-11-20",
            end_date="2026-11-20",
            fetcher=lambda **_: _valid_nasa_payload("20261120"),
            ingested_at_utc="2026-11-25T03:00:00Z",
        )
        assert res["status"] == "SUCCESS"
        for sql in sim_client.sent_sql:
            assert not re.search(
                r"PRAGMA\s+(?:wal_checkpoint|journal_mode)\b", sql, re.IGNORECASE
            ), f"Forbidden Turso PRAGMA sent: {sql}"

        runs = remote_conn.execute(
            "SELECT run_id, status, rows_inserted_v0 FROM ingestion_runs WHERE run_id = ?",
            (res["run_id"],),
        ).fetchall()
        assert len(runs) == 1
        assert runs[0]["status"] == "SUCCESS"
        assert int(runs[0]["rows_inserted_v0"]) == 1
        assert json.loads(json.dumps(res["nasa_sources_header"])) == ["MERRA2", "CERES"]
    finally:
        remote_conn.close()
        sim_client.close()
