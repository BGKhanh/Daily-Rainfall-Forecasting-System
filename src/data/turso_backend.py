"""
Unified Local / Turso Embedded-Replica / Turso Remote Store Backend (DS108 DDL V3.3)
====================================================================================
Supports `STORE_BACKEND` in {"local", "turso_replica", "turso_remote"} with `"local"`
as zero-token default for CI and offline Docker serving containers.

Uses Turso Hrana HTTP v2 (`POST {https_url}/v2/pipeline`) over `requests.Session`
so no native Rust/C libSQL binary dependencies are required in slim serving images.
Features:
- SQLite `sqlite3.Connection` / `sqlite3.Row` compatible `TursoRemoteConnection`
- Atomic multi-statement transactions via interactive Hrana stream `baton`
- Server-side `rows_read` & `rows_written` accounting captured from Hrana metadata
- Local SQLite replica sync (`300s ± 30s` jitter background timer) in `turso_replica`
- Separate `TURSO_READONLY_AUTH_TOKEN` support for read-only connections
- Exponential backoff retry on transient `429`/`5xx`/network errors
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import requests

from src.config.clock import now_utc_iso
from src.config.settings import PROJECT_ROOT, SecretsEnvView, load_settings

logger = logging.getLogger(__name__)

VALID_STORE_BACKENDS = ("local", "turso_replica", "turso_remote")
DEFAULT_TURSO_REPLICA_PATH = PROJECT_ROOT / "data" / "vintage" / "turso_replica.db"
TURSO_PLATFORM_API_BASE = "https://api.turso.tech/v1"

_INTEGRITY_ERROR_MARKERS = (
    "UNIQUE constraint failed",
    "FOREIGN KEY constraint failed",
    "CHECK constraint failed",
    "NOT NULL constraint failed",
    "IMMUTABLE_VIOLATION",
    "IMMUTABLE_VINTAGE_ERROR",
    "ABORT",
)

_BEGIN_RE = re.compile(r"^\s*BEGIN(?:\s+(?:DEFERRED|IMMEDIATE|EXCLUSIVE))?\b", re.IGNORECASE)
_COMMIT_RE = re.compile(r"^\s*(?:COMMIT|END)(?:\s+TRANSACTION)?\b", re.IGNORECASE)
_ROLLBACK_RE = re.compile(r"^\s*ROLLBACK(?:\s+TRANSACTION)?\b", re.IGNORECASE)
_READ_ONLY_SQL_RE = re.compile(r"^\s*(?:SELECT|WITH|EXPLAIN|PRAGMA\s+(?:table_info|index_list|foreign_key_check|database_list|query_only|foreign_keys)\b)", re.IGNORECASE)

def _load_turso_quota_config() -> Dict[str, Any]:
    """Load Turso plan & monthly quota thresholds dynamically from Layer 3 runtime_config.toml."""
    app_cfg = load_settings(env_path=None).application
    return {
        "plan_name": str(app_cfg.turso_plan_name),
        "rows_read_limit": int(app_cfg.turso_monthly_read_quota),
        "rows_written_limit": int(app_cfg.turso_monthly_write_quota),
        "sync_bytes_limit": int(app_cfg.turso_monthly_sync_bytes_quota),
        "storage_bytes_limit": int(app_cfg.turso_storage_bytes_quota),
        "warn_ratio": float(app_cfg.turso_quota_warn_ratio),
    }


_BEARER_TOKEN_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-+/=]+", re.IGNORECASE)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}\b")
_URL_USERINFO_RE = re.compile(r"(https?://)([^:@/\s]+):([^@/\s]+)@", re.IGNORECASE)
_QUERY_SECRET_RE = re.compile(r"([?&](?:authToken|token|api_key|key|secret)=)([^&\s'\"]+)", re.IGNORECASE)
_KV_SECRET_RE = re.compile(
    r"((?:TURSO_(?:DB|READONLY_AUTH|AUTH|API)_TOKEN|DAGSHUB(?:_USER)?_TOKEN|MLFLOW_TRACKING_PASSWORD|Authorization)\s*[:=]\s*)([^\s,;'\"]+)",
    re.IGNORECASE,
)

_TURSO_RUNTIME_LOCK = threading.Lock()
_TURSO_RUNTIME_STATE: Dict[str, Any] = {
    "blocked": False,
    "degraded_reason": None,
    "last_blocked_at_utc": None,
    "last_error": None,
    "quota_warning": False,
    "quota_rows_read_pct": 0.0,
    "usage_source": "in_process_hrana_counter_only",
}


class TursoQuotaBlockedError(sqlite3.OperationalError):
    """Raised when Turso returns HTTP 402/403/429 or a BLOCKED/quota exhaustion error."""


def redact_secrets(text: Optional[str], extra_secrets: Sequence[str] = ()) -> str:
    """Redact bearer tokens, JWTs, and credential env values before logging or returning diagnostics."""
    if not text:
        return ""
    out = str(text)
    for env_key in (
        "TURSO_DB_TOKEN",
        "TURSO_READONLY_AUTH_TOKEN",
        "TURSO_AUTH_TOKEN",
        "TURSO_API_TOKEN",
        "DAGSHUB_TOKEN",
        "DAGSHUB_USER_TOKEN",
        "MLFLOW_TRACKING_PASSWORD",
    ):
        val = os.environ.get(env_key, "").strip()
        if len(val) >= 6 and val in out:
            out = out.replace(val, "[REDACTED]")
    for sec in extra_secrets:
        s = (sec or "").strip()
        if len(s) >= 6 and s in out:
            out = out.replace(s, "[REDACTED]")
    out = _BEARER_TOKEN_RE.sub(r"\1[REDACTED]", out)
    out = _URL_USERINFO_RE.sub(r"\1[REDACTED]:[REDACTED]@", out)
    out = _QUERY_SECRET_RE.sub(r"\1[REDACTED]", out)
    out = _KV_SECRET_RE.sub(r"\1[REDACTED]", out)
    out = _JWT_RE.sub("[REDACTED_JWT]", out)
    return out


def _is_quota_or_blocked_message(message: str) -> bool:
    low = (message or "").lower()
    return any(
        marker in low
        for marker in ("blocked", "quota", "reads_blocked", "writes_blocked", "payment required", "rate limit", "too many requests")
    )


def get_turso_runtime_status() -> Dict[str, Any]:
    """Return a thread-safe snapshot of global Turso blocked/quota status."""
    with _TURSO_RUNTIME_LOCK:
        return dict(_TURSO_RUNTIME_STATE)


def record_turso_quota_blocked(reason: str, error_msg: str = "") -> None:
    """Mark global Turso runtime state as degraded/blocked using secret-redacted error details."""
    redacted = redact_secrets(error_msg)
    with _TURSO_RUNTIME_LOCK:
        _TURSO_RUNTIME_STATE["blocked"] = True
        _TURSO_RUNTIME_STATE["degraded_reason"] = reason
        _TURSO_RUNTIME_STATE["last_blocked_at_utc"] = now_utc_iso()
        _TURSO_RUNTIME_STATE["last_error"] = redacted[:300] if redacted else None


def evaluate_turso_quota_warning(
    rows_read_used: int,
    *,
    rows_written_used: int = 0,
    sync_bytes_used: int = 0,
    rows_read_limit: Optional[int] = None,
    rows_written_limit: Optional[int] = None,
    sync_bytes_limit: Optional[int] = None,
    warn_ratio: Optional[float] = None,
    usage_source: str = "in_process_hrana_counter_only",
) -> Dict[str, Any]:
    """Evaluate monthly Turso quota against Layer 3 `runtime_config.toml` plan thresholds."""
    cfg = _load_turso_quota_config()
    eff_read_limit = int(rows_read_limit if rows_read_limit is not None else cfg["rows_read_limit"])
    eff_write_limit = int(rows_written_limit if rows_written_limit is not None else cfg["rows_written_limit"])
    eff_sync_limit = int(sync_bytes_limit if sync_bytes_limit is not None else cfg["sync_bytes_limit"])
    eff_warn_ratio = float(warn_ratio if warn_ratio is not None else cfg["warn_ratio"])

    read_ratio = float(rows_read_used) / float(max(1, eff_read_limit))
    write_ratio = float(rows_written_used) / float(max(1, eff_write_limit))
    sync_ratio = float(sync_bytes_used) / float(max(1, eff_sync_limit))
    max_ratio = max(read_ratio, write_ratio, sync_ratio)
    warning = max_ratio >= eff_warn_ratio
    read_pct = round(read_ratio * 100.0, 4)
    with _TURSO_RUNTIME_LOCK:
        _TURSO_RUNTIME_STATE["quota_rows_read_pct"] = read_pct
        _TURSO_RUNTIME_STATE["usage_source"] = usage_source
        if warning:
            _TURSO_RUNTIME_STATE["quota_warning"] = True
    if warning:
        logger.warning(
            "Turso monthly quota warning triggered (>= %.0f%% on %s plan): rows_read=%.2f%% (%d/%d).",
            eff_warn_ratio * 100.0,
            cfg["plan_name"],
            read_pct,
            rows_read_used,
            eff_read_limit,
        )
    return {
        "plan_name": cfg["plan_name"],
        "turso_quota_warning": warning,
        "rows_read_pct": read_pct,
        "rows_written_pct": round(write_ratio * 100.0, 4),
        "sync_bytes_pct": round(sync_ratio * 100.0, 4),
        "threshold_pct": eff_warn_ratio * 100.0,
        "usage_source": usage_source,
    }


def clear_turso_blocked_state() -> None:
    """Clear degraded/blocked flags after a successful remote or replica sync, preserving quota warning."""
    with _TURSO_RUNTIME_LOCK:
        _TURSO_RUNTIME_STATE["blocked"] = False
        _TURSO_RUNTIME_STATE["degraded_reason"] = None
        _TURSO_RUNTIME_STATE["last_error"] = None


def clear_turso_runtime_status() -> None:
    """Reset runtime blocked/quota state (used on successful remote probe or in tests)."""
    with _TURSO_RUNTIME_LOCK:
        _TURSO_RUNTIME_STATE.update(
            {
                "blocked": False,
                "degraded_reason": None,
                "last_blocked_at_utc": None,
                "last_error": None,
                "quota_warning": False,
                "quota_rows_read_pct": 0.0,
                "usage_source": "in_process_hrana_counter_only",
            }
        )


def resolve_store_backend(explicit_backend: Optional[str] = None) -> str:
    """Resolve `STORE_BACKEND` from explicit arg, env var, or runtime TOML (default: 'local')."""
    if explicit_backend is not None and explicit_backend.strip():
        candidate = explicit_backend.strip().lower()
    else:
        env_val = os.environ.get("STORE_BACKEND", "").strip().lower()
        if env_val:
            candidate = env_val
        else:
            try:
                candidate = load_settings(env_path=None).operational.store_backend
            except Exception:
                candidate = "local"
    if candidate not in VALID_STORE_BACKENDS:
        raise ValueError(
            f"Unsupported STORE_BACKEND={candidate!r}. Expected one of {VALID_STORE_BACKENDS}."
        )
    return candidate


def normalize_turso_http_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if url.startswith("libsql://"):
        url = "https://" + url[len("libsql://") :]
    elif url.startswith("wss://"):
        url = "https://" + url[len("wss://") :]
    elif url.startswith("ws://"):
        url = "http://" + url[len("ws://") :]
    return url.rstrip("/")


class HranaRow:
    """sqlite3.Row-compatible container supporting both positional and column-name access."""

    __slots__ = ("_cols", "_vals", "_index")

    def __init__(self, cols: Tuple[str, ...], vals: Tuple[Any, ...]) -> None:
        self._cols = cols
        self._vals = vals
        self._index = {name: idx for idx, name in enumerate(cols)}

    def keys(self) -> Tuple[str, ...]:
        return self._cols

    def __len__(self) -> int:
        return len(self._vals)

    def __iter__(self):
        return iter(self._vals)

    def __getitem__(self, key: Union[int, slice, str]) -> Any:
        if isinstance(key, str):
            idx = self._index.get(key)
            if idx is None:
                raise IndexError(f"No item with that key: {key!r}")
            return self._vals[idx]
        return self._vals[key]

    def __contains__(self, item: object) -> bool:
        return item in self._vals

    def __repr__(self) -> str:
        return f"HranaRow({dict(zip(self._cols, self._vals))!r})"


class HranaCursor:
    """Minimal sqlite3.Cursor-compatible cursor returned by TursoRemoteConnection.execute()."""

    def __init__(
        self,
        cols: Tuple[str, ...],
        rows: List[HranaRow],
        *,
        rowcount: int = -1,
        lastrowid: Optional[int] = None,
        rows_read: int = 0,
        rows_written: int = 0,
    ) -> None:
        self._cols = cols
        self._rows = rows
        self._pos = 0
        self.rowcount = rowcount
        self.lastrowid = lastrowid
        self.rows_read = rows_read
        self.rows_written = rows_written
        self.description = tuple((c, None, None, None, None, None, None) for c in cols) if cols else None

    def fetchone(self) -> Optional[HranaRow]:
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def fetchall(self) -> List[HranaRow]:
        rem = self._rows[self._pos :]
        self._pos = len(self._rows)
        return rem

    def fetchmany(self, size: int = 1) -> List[HranaRow]:
        end = min(len(self._rows), self._pos + max(0, int(size)))
        chunk = self._rows[self._pos : end]
        self._pos = end
        return chunk

    def __iter__(self):
        return iter(self.fetchall())

    def close(self) -> None:
        pass


def _encode_hrana_arg(val: Any) -> Dict[str, Any]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return {"type": "null"}
    if isinstance(val, bool):
        return {"type": "integer", "value": "1" if val else "0"}
    if isinstance(val, (int, np.integer)):
        return {"type": "integer", "value": str(int(val))}
    if isinstance(val, (float, np.floating)):
        return {"type": "float", "value": float(val)}
    if isinstance(val, (bytes, bytearray, memoryview)):
        return {"type": "blob", "base64": base64.b64encode(bytes(val)).decode("ascii")}
    return {"type": "text", "value": str(val)}


def _decode_hrana_cell(cell: Any) -> Any:
    if not isinstance(cell, dict):
        return cell
    ctype = cell.get("type")
    if ctype == "null":
        return None
    if ctype == "integer":
        return int(cell["value"])
    if ctype == "float":
        return float(cell["value"])
    if ctype == "blob":
        b64 = cell.get("base64", "")
        return base64.b64decode(b64) if b64 else b""
    return cell.get("value")


def _build_stmt_payload(sql: str, params: Any = None) -> Dict[str, Any]:
    stmt: Dict[str, Any] = {"sql": sql}
    if params is None:
        return stmt
    if isinstance(params, Mapping):
        stmt["named_args"] = [
            {"name": str(k).lstrip(":@$"), "value": _encode_hrana_arg(v)}
            for k, v in params.items()
        ]
    elif isinstance(params, (list, tuple)):
        if len(params) > 0:
            stmt["args"] = [_encode_hrana_arg(v) for v in params]
    else:
        stmt["args"] = [_encode_hrana_arg(params)]
    return stmt


def _map_turso_error(message: str) -> Exception:
    for marker in _INTEGRITY_ERROR_MARKERS:
        if marker in message:
            return sqlite3.IntegrityError(message)
    return sqlite3.OperationalError(message)


@dataclass
class TursoUsageSnapshot:
    total_requests: int
    total_rows_read: int
    total_rows_written: int
    last_rows_read: int
    last_rows_written: int
    last_query_duration_ms: float


class TursoHranaClient:
    """Pooled HTTP v2 pipeline client for Turso / libSQL with retry and row-usage tracking."""

    def __init__(
        self,
        *,
        db_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        read_only: bool = False,
        timeout_sec: float = 20.0,
        max_retries: int = 3,
        backoff_base_sec: float = 0.4,
    ) -> None:
        secrets = SecretsEnvView.from_environ()
        raw_url = (db_url or secrets.turso_db_url).strip()
        if not raw_url:
            raise RuntimeError(
                "Turso backend requires TURSO_DB_URL (or TURSO_DATABASE_URL) in environment."
            )
        if auth_token is not None:
            resolved_token = auth_token.strip()
        elif read_only and secrets.turso_readonly_auth_token:
            resolved_token = secrets.turso_readonly_auth_token
        else:
            resolved_token = secrets.turso_db_token
        if not resolved_token:
            raise RuntimeError(
                "Turso backend requires TURSO_DB_TOKEN (or TURSO_READONLY_AUTH_TOKEN for read-only)."
            )

        self.http_url = normalize_turso_http_url(raw_url)
        self.read_only = bool(read_only)
        self.timeout_sec = float(timeout_sec)
        self.max_retries = int(max_retries)
        self.backoff_base_sec = float(backoff_base_sec)
        self._resolved_token = resolved_token
        self._auth_header = f"Bearer {resolved_token}"
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": self._auth_header,
                "Content-Type": "application/json",
            }
        )
        self._lock = threading.Lock()
        self.total_requests = 0
        self.total_rows_read = 0
        self.total_rows_written = 0
        self.last_rows_read = 0
        self.last_rows_written = 0
        self.last_query_duration_ms = 0.0

    def usage_snapshot(self) -> TursoUsageSnapshot:
        with self._lock:
            return TursoUsageSnapshot(
                total_requests=self.total_requests,
                total_rows_read=self.total_rows_read,
                total_rows_written=self.total_rows_written,
                last_rows_read=self.last_rows_read,
                last_rows_written=self.last_rows_written,
                last_query_duration_ms=self.last_query_duration_ms,
            )

    def reset_counters(self) -> None:
        with self._lock:
            self.total_requests = 0
            self.total_rows_read = 0
            self.total_rows_written = 0
            self.last_rows_read = 0
            self.last_rows_written = 0
            self.last_query_duration_ms = 0.0

    def pipeline(
        self,
        requests_list: List[Dict[str, Any]],
        *,
        baton: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        target_base = normalize_turso_http_url(base_url) if base_url else self.http_url
        endpoint = f"{target_base}/v2/pipeline"
        payload: Dict[str, Any] = {"requests": requests_list}
        if baton:
            payload["baton"] = baton

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(endpoint, json=payload, timeout=self.timeout_sec)
                if resp.status_code in (402, 403):
                    msg = redact_secrets(
                        f"Turso Hrana HTTP {resp.status_code} BLOCKED/quota: {resp.text[:300]}",
                        extra_secrets=[self._resolved_token],
                    )
                    record_turso_quota_blocked("TURSO_REMOTE_BLOCKED_FALLBACK_LOCAL", msg)
                    raise TursoQuotaBlockedError(msg)
                if resp.status_code == 429:
                    if attempt < self.max_retries:
                        time.sleep(self.backoff_base_sec * (2**attempt))
                        continue
                    msg = redact_secrets(
                        f"Turso Hrana HTTP 429 rate-limit/quota exhausted: {resp.text[:300]}",
                        extra_secrets=[self._resolved_token],
                    )
                    record_turso_quota_blocked("TURSO_REMOTE_BLOCKED_FALLBACK_LOCAL", msg)
                    raise TursoQuotaBlockedError(msg)
                if resp.status_code in (502, 503, 504) and attempt < self.max_retries:
                    time.sleep(self.backoff_base_sec * (2**attempt))
                    continue
                if resp.status_code >= 400:
                    safe_txt = redact_secrets(resp.text[:300], extra_secrets=[self._resolved_token])
                    if _is_quota_or_blocked_message(safe_txt):
                        record_turso_quota_blocked("TURSO_REMOTE_BLOCKED_FALLBACK_LOCAL", safe_txt)
                        raise TursoQuotaBlockedError(f"Turso Hrana HTTP {resp.status_code}: {safe_txt}")
                    raise sqlite3.OperationalError(
                        f"Turso Hrana HTTP {resp.status_code}: {safe_txt}"
                    )
                body = resp.json()
                req_read = 0
                req_written = 0
                req_dur = 0.0
                for item in body.get("results", []):
                    if item.get("type") == "ok":
                        res = (item.get("response") or {}).get("result") or {}
                        req_read += int(res.get("rows_read") or 0)
                        req_written += int(res.get("rows_written") or 0)
                        req_dur += float(res.get("query_duration_ms") or 0.0)
                with self._lock:
                    self.total_requests += 1
                    self.last_rows_read = req_read
                    self.last_rows_written = req_written
                    self.total_rows_read += req_read
                    self.total_rows_written += req_written
                    self.last_query_duration_ms = req_dur
                clear_turso_blocked_state()
                with _TURSO_RUNTIME_LOCK:
                    cur_src = str(_TURSO_RUNTIME_STATE.get("usage_source") or "in_process_hrana_counter_only")
                    hrana_r = self.total_rows_read
                    hrana_w = self.total_rows_written
                if cur_src == "in_process_hrana_counter_only":
                    evaluate_turso_quota_warning(
                        hrana_r,
                        rows_written_used=hrana_w,
                        usage_source="in_process_hrana_counter_only",
                    )
                return body
            except TursoQuotaBlockedError:
                raise
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base_sec * (2**attempt))
                    continue
                safe_exc = redact_secrets(str(exc), extra_secrets=[self._resolved_token])
                raise sqlite3.OperationalError(f"Turso network failure: {safe_exc}") from exc
        raise sqlite3.OperationalError(
            f"Turso request failed: {redact_secrets(str(last_exc), extra_secrets=[self._resolved_token])}"
        )

    def close(self) -> None:
        self._session.close()


class TursoRemoteConnection:
    """
    sqlite3.Connection-compatible remote connection backed by Turso Hrana HTTP v2.
    Supports interactive transactions via Hrana stream `baton`, atomic `execute_batch()`,
    and `PRAGMA foreign_keys = ON` + `PRAGMA query_only` semantics.
    """

    def __init__(
        self,
        client: Optional[TursoHranaClient] = None,
        *,
        read_only: bool = False,
        db_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        on_commit_callback: Optional[Any] = None,
    ) -> None:
        self.read_only = bool(read_only)
        self.client = client or TursoHranaClient(
            db_url=db_url, auth_token=auth_token, read_only=read_only
        )
        self._owns_client = client is None
        self._on_commit_callback = on_commit_callback
        self._baton: Optional[str] = None
        self._baton_base_url: Optional[str] = None
        self._foreign_keys = True
        self._query_only = bool(read_only)
        self._closed = False
        self.row_factory: Any = HranaRow

    @staticmethod
    def _strip_leading_comments(sql: str) -> str:
        s = sql.lstrip()
        while s.startswith("--") or s.startswith("/*"):
            if s.startswith("--"):
                nl = s.find("\n")
                s = "" if nl == -1 else s[nl + 1 :].lstrip()
            elif s.startswith("/*"):
                end = s.find("*/")
                s = "" if end == -1 else s[end + 2 :].lstrip()
        return s

    def _check_not_closed(self) -> None:
        if self._closed:
            raise sqlite3.ProgrammingError("Cannot operate on a closed TursoRemoteConnection.")

    def _enforce_write_policy(self, sql: str) -> None:
        cleaned = self._strip_leading_comments(sql)
        if not cleaned:
            return
        if (self.read_only or self._query_only) and not _READ_ONLY_SQL_RE.match(cleaned):
            if _BEGIN_RE.match(cleaned) or _ROLLBACK_RE.match(cleaned):
                return
            raise sqlite3.OperationalError("attempt to write a readonly database")

    def execute(self, sql: str, parameters: Any = ()) -> HranaCursor:
        self._check_not_closed()
        cleaned = self._strip_leading_comments(sql)
        upper = cleaned.upper()

        # Local tracking for connection-scoped PRAGMAs
        if upper.startswith("PRAGMA"):
            norm = re.sub(r"\s+", " ", upper).rstrip(";")
            if norm.startswith("PRAGMA QUERY_ONLY"):
                if "=" in norm:
                    val = norm.split("=", 1)[1].strip()
                    self._query_only = val in ("1", "ON", "TRUE")
                    return HranaCursor((), [], rowcount=0)
                return HranaCursor(("query_only",), [HranaRow(("query_only",), (1 if self._query_only else 0,))])
            if norm.startswith("PRAGMA FOREIGN_KEYS"):
                if "=" in norm:
                    val = norm.split("=", 1)[1].strip()
                    self._foreign_keys = val in ("1", "ON", "TRUE")
                elif self._baton is None:
                    return HranaCursor(("foreign_keys",), [HranaRow(("foreign_keys",), (1 if self._foreign_keys else 0,))])
            if norm.startswith("PRAGMA RECURSIVE_TRIGGERS") or norm.startswith("PRAGMA JOURNAL_MODE") or norm.startswith("PRAGMA SYNCHRONOUS") or norm.startswith("PRAGMA CACHE_SIZE") or norm.startswith("PRAGMA BUSY_TIMEOUT"):
                # Handled natively/safely by Turso server or ignored for managed WAL pragmas
                if "=" in norm:
                    return HranaCursor((), [], rowcount=0)

        self._enforce_write_policy(sql)

        is_begin = bool(_BEGIN_RE.match(cleaned))
        is_commit = bool(_COMMIT_RE.match(cleaned))
        is_rollback = bool(_ROLLBACK_RE.match(cleaned))

        if is_begin:
            reqs: List[Dict[str, Any]] = []
            if self._foreign_keys:
                reqs.append({"type": "execute", "stmt": {"sql": "PRAGMA foreign_keys = ON;"}})
            reqs.append({"type": "execute", "stmt": {"sql": cleaned}})
            body = self.client.pipeline(reqs, baton=None, base_url=None)
            self._raise_if_any_error(body)
            self._baton = body.get("baton")
            self._baton_base_url = body.get("base_url")
            return HranaCursor((), [], rowcount=0)

        if is_commit or is_rollback:
            if self._baton:
                reqs = [
                    {"type": "execute", "stmt": {"sql": "COMMIT" if is_commit else "ROLLBACK"}},
                    {"type": "close"},
                ]
                try:
                    body = self.client.pipeline(
                        reqs, baton=self._baton, base_url=self._baton_base_url
                    )
                    self._raise_if_any_error(body)
                finally:
                    self._baton = None
                    self._baton_base_url = None
            if is_commit and self._on_commit_callback is not None:
                self._on_commit_callback()
            return HranaCursor((), [], rowcount=0)

        stmt_req = {"type": "execute", "stmt": _build_stmt_payload(sql, parameters)}
        if self._baton:
            body = self.client.pipeline(
                [stmt_req], baton=self._baton, base_url=self._baton_base_url
            )
            if body.get("baton"):
                self._baton = body["baton"]
            self._raise_if_any_error(body)
            res = (body["results"][0].get("response") or {}).get("result") or {}
        else:
            reqs = []
            need_fk = self._foreign_keys and not _READ_ONLY_SQL_RE.match(cleaned)
            if need_fk:
                reqs.append({"type": "execute", "stmt": {"sql": "PRAGMA foreign_keys = ON;"}})
            reqs.append(stmt_req)
            reqs.append({"type": "close"})
            body = self.client.pipeline(reqs)
            self._raise_if_any_error(body)
            stmt_idx = 1 if need_fk else 0
            res = (body["results"][stmt_idx].get("response") or {}).get("result") or {}

        cols = tuple(c["name"] for c in res.get("cols", []))
        rows = [
            HranaRow(cols, tuple(_decode_hrana_cell(cell) for cell in raw_row))
            for raw_row in res.get("rows", [])
        ]
        last_rowid_raw = res.get("last_insert_rowid")
        last_rowid = int(last_rowid_raw) if last_rowid_raw is not None else None
        cursor = HranaCursor(
            cols,
            rows,
            rowcount=int(res.get("affected_row_count") or len(rows)),
            lastrowid=last_rowid,
            rows_read=int(res.get("rows_read") or 0),
            rows_written=int(res.get("rows_written") or 0),
        )
        if not self._baton and not _READ_ONLY_SQL_RE.match(cleaned) and self._on_commit_callback is not None:
            self._on_commit_callback()
        return cursor

    def _raise_if_any_error(self, body: Dict[str, Any]) -> None:
        for item in body.get("results", []):
            if item.get("type") == "error":
                err = item.get("error") or {}
                msg = redact_secrets(
                    str(err.get("message") or item),
                    extra_secrets=[self.client._resolved_token],
                )
                if _is_quota_or_blocked_message(msg):
                    record_turso_quota_blocked("TURSO_REMOTE_BLOCKED_FALLBACK_LOCAL", msg)
                    raise TursoQuotaBlockedError(msg)
                raise _map_turso_error(msg)

    def execute_batch(self, statements: Sequence[Tuple[str, Any]], *, atomic: bool = True) -> Tuple[int, int]:
        """
        Execute many parameterized SQL statements over minimal HTTP round trips.
        When `atomic=True`, uses a single Turso baton transaction that strictly rolls back
        EVERY prior statement in the batch if ANY statement fails.
        Returns `(total_rows_read, total_rows_written)`.
        """
        self._check_not_closed()
        if self.read_only or self._query_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        if not statements:
            return 0, 0

        if self._baton:
            # Participate cleanly inside the already-open outer interactive baton transaction.
            rows_read_sum = 0
            rows_written_sum = 0
            chunk_size = 200
            for start in range(0, len(statements), chunk_size):
                chunk = statements[start : start + chunk_size]
                reqs = [
                    {"type": "execute", "stmt": _build_stmt_payload(sql, params)}
                    for sql, params in chunk
                ]
                body = self.client.pipeline(reqs, baton=self._baton, base_url=self._baton_base_url)
                if body.get("baton"):
                    self._baton = body["baton"]
                rows_read_sum += self.client.last_rows_read
                rows_written_sum += self.client.last_rows_written
                self._raise_if_any_error(body)
            return rows_read_sum, rows_written_sum

        if not atomic:
            reqs = [{"type": "execute", "stmt": {"sql": "PRAGMA foreign_keys = ON;"}}] if self._foreign_keys else []
            for sql, params in statements:
                reqs.append({"type": "execute", "stmt": _build_stmt_payload(sql, params)})
            reqs.append({"type": "close"})
            body = self.client.pipeline(reqs)
            self._raise_if_any_error(body)
            snap = self.client.usage_snapshot()
            if self._on_commit_callback is not None:
                self._on_commit_callback()
            return snap.last_rows_read, snap.last_rows_written

        # Interactive atomic transaction: BEGIN IMMEDIATE + chunked statements -> error => ROLLBACK
        begin_reqs = []
        if self._foreign_keys:
            begin_reqs.append({"type": "execute", "stmt": {"sql": "PRAGMA foreign_keys = ON;"}})
        begin_reqs.append({"type": "execute", "stmt": {"sql": "BEGIN IMMEDIATE"}})
        begin_body = self.client.pipeline(begin_reqs)
        self._raise_if_any_error(begin_body)
        baton = begin_body.get("baton")
        base_url = begin_body.get("base_url")
        rows_read_sum = self.client.last_rows_read
        rows_written_sum = self.client.last_rows_written

        chunk_size = 200
        try:
            for start in range(0, len(statements), chunk_size):
                chunk = statements[start : start + chunk_size]
                reqs = [{"type": "execute", "stmt": _build_stmt_payload(sql, params)} for sql, params in chunk]
                body = self.client.pipeline(reqs, baton=baton, base_url=base_url)
                if body.get("baton"):
                    baton = body["baton"]
                rows_read_sum += self.client.last_rows_read
                rows_written_sum += self.client.last_rows_written
                self._raise_if_any_error(body)

            commit_body = self.client.pipeline(
                [{"type": "execute", "stmt": {"sql": "COMMIT"}}, {"type": "close"}],
                baton=baton,
                base_url=base_url,
            )
            self._raise_if_any_error(commit_body)
            baton = None
            if self._on_commit_callback is not None:
                self._on_commit_callback()
            return rows_read_sum, rows_written_sum
        except Exception:
            if baton:
                try:
                    self.client.pipeline(
                        [{"type": "execute", "stmt": {"sql": "ROLLBACK"}}, {"type": "close"}],
                        baton=baton,
                        base_url=base_url,
                    )
                except Exception:
                    pass
            raise

    def executemany(self, sql: str, seq_of_parameters: Iterable[Any]) -> HranaCursor:
        stmts = [(sql, params) for params in seq_of_parameters]
        rr, rw = self.execute_batch(stmts, atomic=True)
        return HranaCursor((), [], rowcount=len(stmts), rows_read=rr, rows_written=rw)

    def executescript(self, sql_script: str) -> None:
        self._check_not_closed()
        if self.read_only or self._query_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        parts = [s.strip() for s in sql_script.split(";") if s.strip()]
        if parts:
            self.execute_batch([(p, ()) for p in parts], atomic=True)

    def commit(self) -> None:
        if self._baton:
            self.execute("COMMIT")

    def rollback(self) -> None:
        if self._baton:
            self.execute("ROLLBACK")

    def __enter__(self) -> "TursoRemoteConnection":
        self.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()

    def close(self) -> None:
        if not self._closed:
            if self._baton:
                try:
                    self.rollback()
                except Exception:
                    self._baton = None
            self._closed = True
            if self._owns_client:
                self.client.close()


_LOCATIONS_COLS = (
    "location_id",
    "display_name",
    "aliases_json",
    "latitude",
    "longitude",
    "meteo_grid_note",
    "rad_grid_note",
    "time_standard",
    "provisional_window_days",
    "is_active",
    "created_at_utc",
)

_INGESTION_RUNS_COLS = (
    "run_id",
    "run_type",
    "started_at_utc",
    "finished_at_utc",
    "status",
    "location_id",
    "scan_window_start",
    "scan_window_end",
    "nasa_api_version",
    "nasa_sources_header_json",
    "rows_scanned",
    "rows_inserted_v0",
    "rows_new_vintage",
    "rows_late_fill_only",
    "rows_value_change",
    "rows_unchanged",
    "rows_unpublished_skipped",
    "latest_valid_by_group_json",
    "revision_stats_json",
    "git_sha",
    "config_sha256",
    "operational_config_sha256",
    "error_message",
)

_FORECASTS_LEDGER_META_COLS = (
    "forecast_id",
    "issued_at_utc",
    "issued_run_date",
    "run_timezone",
    "location_id",
    "model_id",
    "model_kind",
    "registry_alias",
    "registry_model_version",
    "artifact_sha256",
    "feature_builder_sha256",
    "data_as_of",
    "eval_split_regime",
    "horizon_day",
    "target_date",
    "temporal_status_at_issue",
    "pred_mm",
    "input_window_start",
    "input_window_end",
    "input_window_hash",
    "input_vintage_manifest_json",
    "input_provisional_days",
    "meteo_missing_count",
    "flashflux_max_ffill_days",
    "ceres_policy_applied",
    "ceres_max_staleness_days",
    "is_degraded_input",
    "degraded_reason",
    "git_sha",
    "config_sha256",
    "issued_by",
    "result_config_sha256",
    "imputed_values_json",
    "doy_table_sha256",
)


def _obs_all_cols(all_35_vars: Sequence[str]) -> Tuple[str, ...]:
    return (
        "location_id",
        "obs_date",
        "version_num",
        "ingested_at_utc",
        "run_id",
        "age_days_at_ingest",
        "ingest_origin",
        "missing_count",
        "missing_meteo_count",
        "missing_flashflux_count",
        "missing_ceres_count",
        "completeness_regime",
        "missing_vars_json",
        "change_type_summary",
        "late_fill_count",
        "late_fill_vars_json",
        "value_change_count",
        "value_change_details_json",
        "anomaly_count",
        "quality_flags_json",
        "row_value_hash",
        *all_35_vars,
    )


class TursoReplicaManager:
    """
    Maintains a local SQLite replica DB synced from remote Turso on startup and
    via a periodic background timer (`300s ± 30s` jitter) with clean cache invalidation.
    """

    def __init__(
        self,
        *,
        replica_path: Path = DEFAULT_TURSO_REPLICA_PATH,
        db_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        on_synced_callback: Optional[Any] = None,
    ) -> None:
        self.replica_path = Path(replica_path)
        self.db_url = db_url
        self.auth_token = auth_token
        self.on_synced_callback = on_synced_callback
        self._sync_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.sync_count = 0
        self.last_sync_at_utc: Optional[str] = None
        self.last_sync_duration_ms: float = 0.0
        self.last_sync_rows_read: int = 0
        self.last_sync_error: Optional[str] = None
        self.last_hash_verified: Optional[bool] = None
        self.consecutive_failures: int = 0
        self.is_degraded: bool = False
        self.degraded_reason: Optional[str] = None

    @staticmethod
    def _cleanup_sqlite_files(path: Path) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            cand = path.with_name(path.name + suffix)
            if cand.exists():
                try:
                    cand.unlink()
                except OSError:
                    pass

    def sync_now(self, *, verify_hashes: Optional[bool] = None) -> Dict[str, Any]:
        """Atomically stage, integrity-check, verify parity, finalize, and swap local SQLite replica."""
        import json as _json

        from src.data.vintage_store import (
            ALL_35_VARS,
            connect_sqlite,
            finalize_db_for_read_only,
            init_schema,
        )

        t0 = time.perf_counter()
        watermark_path = self.replica_path.with_suffix(".sync_state.json")
        staging_path = self.replica_path.with_name(f"{self.replica_path.name}.staging.{os.getpid()}.db")
        staging_wm_path = self.replica_path.with_name(f"{watermark_path.name}.tmp.{os.getpid()}")

        with self._sync_lock:
            self.replica_path.parent.mkdir(parents=True, exist_ok=True)
            self._cleanup_sqlite_files(staging_path)
            if self.replica_path.exists():
                shutil.copy2(self.replica_path, staging_path)

            remote_client: Optional[TursoHranaClient] = None
            remote_conn: Optional[TursoRemoteConnection] = None
            local_conn: Optional[sqlite3.Connection] = None
            try:
                remote_client = TursoHranaClient(
                    db_url=self.db_url, auth_token=self.auth_token, read_only=True
                )
                remote_conn = TursoRemoteConnection(client=remote_client, read_only=True)
                local_conn = connect_sqlite(staging_path, read_only=False)
                init_schema(local_conn, seed_hcmc_location=False)
                wm = {"obs_rowid": 0, "fc_rowid": 0}
                if watermark_path.exists():
                    try:
                        wm.update(_json.loads(watermark_path.read_text(encoding="utf-8")))
                    except Exception:
                        pass
                local_obs_count = int(
                    local_conn.execute("SELECT COUNT(*) FROM observations_vintage").fetchone()[0]
                )
                is_initial_seed = local_obs_count == 0
                if is_initial_seed:
                    wm["obs_rowid"] = 0
                    wm["fc_rowid"] = 0

                # 1. schema_migrations + locations (small reference tables)
                mig_rows = remote_conn.execute(
                    "SELECT version, description, applied_at_utc FROM schema_migrations ORDER BY applied_at_utc"
                ).fetchall()
                for mr in mig_rows:
                    local_conn.execute(
                        "INSERT OR IGNORE INTO schema_migrations (version, description, applied_at_utc) VALUES (?, ?, ?)",
                        (mr["version"], mr["description"], mr["applied_at_utc"]),
                    )

                existing_locs = {
                    r[0] for r in local_conn.execute("SELECT location_id FROM locations").fetchall()
                }
                loc_cols_str = ", ".join(_LOCATIONS_COLS)
                loc_ph = ", ".join(["?"] * len(_LOCATIONS_COLS))
                loc_rows = remote_conn.execute(f"SELECT {loc_cols_str} FROM locations").fetchall()
                for lr in loc_rows:
                    if lr["location_id"] not in existing_locs:
                        local_conn.execute(
                            f"INSERT INTO locations ({loc_cols_str}) VALUES ({loc_ph})",
                            tuple(lr),
                        )

                # 2. ingestion_runs delta by run_id
                existing_runs = {
                    r[0] for r in local_conn.execute("SELECT run_id FROM ingestion_runs").fetchall()
                }
                ir_cols_str = ", ".join(_INGESTION_RUNS_COLS)
                ir_ph = ", ".join(["?"] * len(_INGESTION_RUNS_COLS))
                ir_rows = remote_conn.execute(
                    f"SELECT {ir_cols_str} FROM ingestion_runs ORDER BY started_at_utc"
                ).fetchall()
                for ir in ir_rows:
                    if ir["run_id"] not in existing_runs:
                        local_conn.execute(
                            f"INSERT INTO ingestion_runs ({ir_cols_str}) VALUES ({ir_ph})",
                            tuple(ir),
                        )

                # 3. observations_vintage incremental watermark by remote rowid
                last_obs_rid = int(wm.get("obs_rowid", 0))
                obs_cols = _obs_all_cols(ALL_35_VARS)
                obs_col_str = ", ".join(obs_cols)
                obs_ph = ", ".join(["?"] * len(obs_cols))
                existing_obs_keys = {
                    (r[0], r[1], int(r[2]))
                    for r in local_conn.execute(
                        "SELECT location_id, obs_date, version_num FROM observations_vintage"
                    ).fetchall()
                }
                while True:
                    batch = remote_conn.execute(
                        f"SELECT rowid AS _rid, {obs_col_str} FROM observations_vintage WHERE rowid > ? ORDER BY rowid LIMIT 500",
                        (last_obs_rid,),
                    ).fetchall()
                    if not batch:
                        break
                    local_conn.execute("BEGIN IMMEDIATE")
                    try:
                        for row in batch:
                            last_obs_rid = max(last_obs_rid, int(row[0]))
                            vals = tuple(row)[1:]
                            key = (vals[0], vals[1], int(vals[2]))
                            if key not in existing_obs_keys:
                                local_conn.execute(
                                    f"INSERT INTO observations_vintage ({obs_col_str}) VALUES ({obs_ph})",
                                    vals,
                                )
                                existing_obs_keys.add(key)
                        local_conn.execute("COMMIT")
                    except Exception:
                        local_conn.execute("ROLLBACK")
                        raise
                    if len(batch) < 500:
                        break

                # 4. forecasts_ledger incremental watermark by remote rowid
                last_fc_rid = int(wm.get("fc_rowid", 0))
                fc_col_str = ", ".join(_FORECASTS_LEDGER_META_COLS)
                fc_ph = ", ".join(["?"] * len(_FORECASTS_LEDGER_META_COLS))
                existing_fc_ids = {
                    r[0] for r in local_conn.execute("SELECT forecast_id FROM forecasts_ledger").fetchall()
                }
                while True:
                    batch = remote_conn.execute(
                        f"SELECT rowid AS _rid, {fc_col_str} FROM forecasts_ledger WHERE rowid > ? ORDER BY rowid LIMIT 500",
                        (last_fc_rid,),
                    ).fetchall()
                    if not batch:
                        break
                    local_conn.execute("BEGIN IMMEDIATE")
                    try:
                        for row in batch:
                            last_fc_rid = max(last_fc_rid, int(row[0]))
                            vals = tuple(row)[1:]
                            if vals[0] not in existing_fc_ids:
                                local_conn.execute(
                                    f"INSERT INTO forecasts_ledger ({fc_col_str}) VALUES ({fc_ph})",
                                    vals,
                                )
                                existing_fc_ids.add(vals[0])
                        local_conn.execute("COMMIT")
                    except Exception:
                        local_conn.execute("ROLLBACK")
                        raise
                    if len(batch) < 500:
                        break

                # 5. Staging verification: SQLite integrity_check + O(1) row counts + optional full table SHA-256
                integ_row = local_conn.execute("PRAGMA integrity_check;").fetchone()
                integ_status = str(integ_row[0]).lower() if integ_row else ""
                if integ_status != "ok":
                    raise RuntimeError(f"[REPLICA_INTEGRITY_ERROR] PRAGMA integrity_check returned {integ_status!r}.")

                check_hashes = is_initial_seed if verify_hashes is None else bool(verify_hashes)
                verified_counts: Dict[str, int] = {}
                verified_hashes: Dict[str, str] = {}
                for tbl in ("locations", "ingestion_runs", "observations_vintage", "forecasts_ledger"):
                    loc_cnt = int(local_conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0])
                    rem_cnt = int(remote_conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0])
                    if loc_cnt != rem_cnt:
                        raise RuntimeError(
                            f"[REPLICA_PARITY_MISMATCH] Table {tbl} count mismatch: local={loc_cnt} != remote={rem_cnt}."
                        )
                    verified_counts[tbl] = loc_cnt
                    _, loc_sha = compute_table_content_sha256(local_conn, tbl)
                    verified_hashes[tbl] = loc_sha
                    if check_hashes:
                        _, rem_sha = compute_table_content_sha256(remote_conn, tbl)
                        if loc_sha != rem_sha:
                            raise RuntimeError(
                                f"[REPLICA_SHA256_MISMATCH] Table {tbl}: local={loc_sha[:12]} != remote={rem_sha[:12]}."
                            )

                finalize_db_for_read_only(local_conn)
                local_conn.close()
                local_conn = None
                remote_conn.close()
                remote_conn = None

                # Cleanly flush any WAL/SHM on target before atomic os.replace
                for suffix in ("-wal", "-shm", "-journal"):
                    cand = self.replica_path.with_name(self.replica_path.name + suffix)
                    if cand.exists():
                        try:
                            cand.unlink()
                        except OSError:
                            pass
                os.replace(staging_path, self.replica_path)
                staging_wm_path.write_text(
                    _json.dumps({"obs_rowid": last_obs_rid, "fc_rowid": last_fc_rid}),
                    encoding="utf-8",
                )
                os.replace(staging_wm_path, watermark_path)

                snap = remote_client.usage_snapshot()
                self.sync_count += 1
                self.last_sync_at_utc = now_utc_iso()
                self.last_sync_duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                self.last_sync_rows_read = snap.total_rows_read
                self.last_sync_error = None
                self.last_hash_verified = bool(verify_hashes) if verify_hashes is not None else bool(verified_hashes)
                self.consecutive_failures = 0
                self.is_degraded = False
                self.degraded_reason = None
                clear_turso_blocked_state()
                if self.on_synced_callback is not None:
                    self.on_synced_callback()
                return {
                    "sync_count": self.sync_count,
                    "last_sync_at_utc": self.last_sync_at_utc,
                    "duration_ms": self.last_sync_duration_ms,
                    "rows_read": self.last_sync_rows_read,
                    "hash_verified": self.last_hash_verified,
                    "replica_path": str(self.replica_path),
                    "table_counts": verified_counts,
                    "table_sha256": verified_hashes,
                }
            except Exception as exc:
                self.consecutive_failures += 1
                extra = [self.auth_token] if self.auth_token else []
                safe_err = redact_secrets(str(exc), extra_secrets=extra)
                self.last_sync_error = safe_err[:300]
                if isinstance(exc, TursoQuotaBlockedError):
                    self.is_degraded = True
                    self.degraded_reason = "TURSO_REMOTE_BLOCKED_FALLBACK_LOCAL"
                elif self.consecutive_failures >= 3:
                    self.is_degraded = True
                    self.degraded_reason = "REPLICA_SYNC_CONSECUTIVE_FAILURES"
                raise
            finally:
                if local_conn is not None:
                    try:
                        local_conn.close()
                    except Exception:
                        pass
                if remote_conn is not None:
                    try:
                        remote_conn.close()
                    except Exception:
                        pass
                if remote_client is not None:
                    try:
                        remote_client.close()
                    except Exception:
                        pass
                self._cleanup_sqlite_files(staging_path)
                if staging_wm_path.exists():
                    try:
                        staging_wm_path.unlink()
                    except OSError:
                        pass

    def start_background_sync(self, *, interval_sec: float = 300.0, jitter_sec: float = 30.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.is_set():
                jitter = random.uniform(-jitter_sec, jitter_sec) if jitter_sec > 0 else 0.0
                wait_s = max(5.0, float(interval_sec) + jitter)
                if self._stop_event.wait(wait_s):
                    break
                try:
                    self.sync_now()
                except Exception as exc:
                    extra = [self.auth_token] if self.auth_token else []
                    logger.warning(
                        "Background Turso replica sync failed: %s",
                        redact_secrets(str(exc), extra_secrets=extra),
                    )

        self._thread = threading.Thread(
            target=_loop, name="turso-replica-sync", daemon=True
        )
        self._thread.start()

    def stop_background_sync(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None


_REPLICA_MANAGERS_LOCK = threading.Lock()
_REPLICA_MANAGERS: Dict[str, TursoReplicaManager] = {}


def get_replica_manager(
    replica_path: Optional[Path] = None,
    *,
    db_url: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> TursoReplicaManager:
    from src.data.vintage_store import clear_db_query_cache

    resolved = Path(
        replica_path or os.environ.get("TURSO_REPLICA_PATH", "") or DEFAULT_TURSO_REPLICA_PATH
    ).resolve()
    key = str(resolved)
    with _REPLICA_MANAGERS_LOCK:
        mgr = _REPLICA_MANAGERS.get(key)
        if mgr is None:
            mgr = TursoReplicaManager(
                replica_path=resolved,
                db_url=db_url,
                auth_token=auth_token,
                on_synced_callback=clear_db_query_cache,
            )
            _REPLICA_MANAGERS[key] = mgr
        return mgr


def _clear_replica_registry_for_tests() -> None:
    with _REPLICA_MANAGERS_LOCK:
        for mgr in _REPLICA_MANAGERS.values():
            mgr.stop_background_sync()
        _REPLICA_MANAGERS.clear()
    clear_turso_runtime_status()


def _resolve_local_fallback_sqlite(explicit_db_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    from src.data.provider import resolve_sqlite_db_path_and_role

    p_arg = Path(explicit_db_path) if explicit_db_path else None
    fallback_path, _ = resolve_sqlite_db_path_and_role(p_arg)
    if fallback_path is not None and fallback_path.exists():
        return fallback_path
    return None


class _QuotaFallbackReadConnection:
    """
    Read-only wrapper over `TursoRemoteConnection` that seamlessly falls back to a local
    SQLite vintage DB if Turso returns HTTP 402/403/429 BLOCKED/quota exhaustion.
    """

    def __init__(self, remote_conn: TursoRemoteConnection, fallback_sqlite_path: Path) -> None:
        self._remote = remote_conn
        self._fallback_path = fallback_sqlite_path
        self._local: Optional[sqlite3.Connection] = None

    def _activate_fallback(self, reason: str, error_msg: str) -> sqlite3.Connection:
        from src.data.vintage_store import connect_sqlite

        record_turso_quota_blocked(reason, error_msg)
        if self._local is None:
            self._local = connect_sqlite(self._fallback_path, read_only=True)
        return self._local

    def execute(self, sql: str, parameters: Any = ()) -> Any:
        if self._local is not None:
            return self._local.execute(sql, parameters)
        try:
            return self._remote.execute(sql, parameters)
        except TursoQuotaBlockedError as exc:
            local_conn = self._activate_fallback("TURSO_REMOTE_BLOCKED_FALLBACK_LOCAL", str(exc))
            return local_conn.execute(sql, parameters)

    def commit(self) -> None:
        if self._local is not None:
            self._local.commit()
        else:
            self._remote.commit()

    def rollback(self) -> None:
        if self._local is not None:
            self._local.rollback()
        else:
            self._remote.rollback()

    def close(self) -> None:
        if self._local is not None:
            try:
                self._local.close()
            finally:
                self._local = None
        self._remote.close()


def open_store_connection(
    db_path: Optional[Union[str, Path]] = None,
    *,
    read_only: bool = True,
    backend: Optional[str] = None,
) -> Tuple[Any, str]:
    """
    Open a store connection according to `STORE_BACKEND` (`local`, `turso_replica`, `turso_remote`).
    Returns `(connection, resolved_backend)`.
    """
    from src.data.vintage_store import clear_db_query_cache, connect_sqlite

    resolved = resolve_store_backend(backend)
    if resolved == "local":
        if db_path is None:
            from src.data.provider import resolve_sqlite_db_path_and_role

            auto_path, _ = resolve_sqlite_db_path_and_role()
            if auto_path is None:
                raise FileNotFoundError("No local SQLite vintage database found.")
            target_path = auto_path
        else:
            target_path = Path(db_path)
        return connect_sqlite(target_path, read_only=read_only), resolved

    if resolved == "turso_replica":
        mgr = get_replica_manager(replica_path=Path(db_path) if db_path else None)
        if read_only:
            if not mgr.replica_path.exists():
                try:
                    mgr.sync_now()
                except TursoQuotaBlockedError as exc:
                    fallback_path = _resolve_local_fallback_sqlite(None)
                    if fallback_path is None:
                        raise
                    record_turso_quota_blocked("TURSO_REMOTE_BLOCKED_FALLBACK_LOCAL", str(exc))
                    return connect_sqlite(fallback_path, read_only=True), "local_fallback"
            return connect_sqlite(mgr.replica_path, read_only=True), resolved

        def _after_write() -> None:
            clear_db_query_cache()
            try:
                mgr.sync_now()
            except Exception as exc:
                logger.warning("Post-write replica sync skipped: %s", redact_secrets(str(exc)))

        return TursoRemoteConnection(read_only=False, on_commit_callback=_after_write), resolved

    remote_conn = TursoRemoteConnection(
        read_only=read_only,
        on_commit_callback=None if read_only else clear_db_query_cache,
    )
    if read_only:
        fallback_path = _resolve_local_fallback_sqlite(db_path)
        if fallback_path is not None:
            return _QuotaFallbackReadConnection(remote_conn, fallback_path), resolved
    return remote_conn, resolved


class TursoPlatformClient:
    """Thin wrapper for Turso Platform REST API v1 (database lifecycle & org usage telemetry)."""

    def __init__(self, api_token: Optional[str] = None, timeout_sec: float = 20.0) -> None:
        secrets = SecretsEnvView.from_environ()
        tok = (api_token or secrets.turso_api_token).strip()
        if not tok:
            raise RuntimeError("TURSO_API_TOKEN is required for Turso Platform API calls.")
        self._token = tok
        self.timeout_sec = timeout_sec
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {tok}"})

    def _request_json(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        r = self._session.request(method, f"{TURSO_PLATFORM_API_BASE}{path}", timeout=self.timeout_sec, **kwargs)
        try:
            r.raise_for_status()
        except requests.HTTPError as exc:
            safe_err = redact_secrets(str(exc), extra_secrets=[self._token])
            raise requests.HTTPError(safe_err, response=r) from None
        return r.json() if r.content else {}

    def list_organizations(self) -> List[Dict[str, Any]]:
        data = self._request_json("GET", "/organizations")
        return data if isinstance(data, list) else data.get("organizations", [data])

    def list_databases(self, org_slug: str) -> List[Dict[str, Any]]:
        return self._request_json("GET", f"/organizations/{org_slug}/databases").get("databases", [])

    def create_database(
        self, org_slug: str, name: str, *, group: str = "default", size_limit: Optional[str] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"name": name, "group": group}
        if size_limit:
            payload["size_limit"] = size_limit
        return self._request_json("POST", f"/organizations/{org_slug}/databases", json=payload)

    def create_db_token(
        self,
        org_slug: str,
        db_name: str,
        *,
        expiration: str = "1d",
        authorization: str = "full-access",
    ) -> str:
        data = self._request_json(
            "POST",
            f"/organizations/{org_slug}/databases/{db_name}/auth/tokens",
            params={"expiration": expiration, "authorization": authorization},
        )
        return str(data["jwt"])

    def delete_database(self, org_slug: str, db_name: str) -> None:
        self._request_json("DELETE", f"/organizations/{org_slug}/databases/{db_name}")

    def get_org_usage(self, org_slug: str) -> Dict[str, Any]:
        data = self._request_json("GET", f"/organizations/{org_slug}/usage")
        total = data.get("total") or {}
        if total:
            data["quota_status"] = evaluate_turso_quota_warning(
                int(total.get("rows_read") or 0),
                rows_written_used=int(total.get("rows_written") or 0),
                sync_bytes_used=int(total.get("bytes_synced") or 0),
                usage_source="turso_platform_api_snapshot_plus_in_process_hrana_counter",
            )
        return data


def compute_table_content_sha256(conn: Any, table_name: str) -> Tuple[int, str]:
    """Compute canonical `(row_count, sha256)` over ordered rows of a given table."""
    from src.data.vintage_store import ALL_35_VARS

    if table_name == "locations":
        order = "location_id"
        cols = ", ".join(_LOCATIONS_COLS)
    elif table_name == "ingestion_runs":
        order = "run_id"
        cols = ", ".join(_INGESTION_RUNS_COLS)
    elif table_name == "observations_vintage":
        order = "location_id, obs_date, version_num"
        cols = "location_id, obs_date, version_num, run_id, ingest_origin, missing_count, completeness_regime, row_value_hash, PRECTOTCORR, " + ", ".join(ALL_35_VARS[:3])
    elif table_name == "forecasts_ledger":
        order = "location_id, model_id, data_as_of, horizon_day, target_date"
        cols = "location_id, model_id, eval_split_regime, data_as_of, horizon_day, target_date, ROUND(pred_mm, 8), input_window_hash, config_sha256, issued_by, result_config_sha256, ceres_policy_applied, is_degraded_input, imputed_values_json, doy_table_sha256"
    else:
        raise ValueError(f"Unsupported verification table: {table_name!r}")

    rows = conn.execute(f"SELECT {cols} FROM {table_name} ORDER BY {order}").fetchall()
    h = hashlib.sha256()
    for row in rows:
        vals = "|".join("" if v is None else str(v) for v in row)
        h.update(vals.encode("utf-8"))
        h.update(b"\n")
    return len(rows), h.hexdigest()


def migrate_sqlite_to_turso(
    source_sqlite_path: Path,
    target_conn: TursoRemoteConnection,
    *,
    include_forecasts: bool = False,
    batch_size: int = 200,
) -> Dict[str, Any]:
    """
    Migrate local SQLite Vintage DB (DDL V3.3) to remote Turso cleanly and verify
    100% row-count & content SHA-256 parity. Defaults `include_forecasts=False`
    (Production DB observations-only policy; Demo DB stays strictly local).
    """
    from src.data.vintage_store import (
        ALL_35_VARS,
        DDL_V3_STATEMENTS,
        apply_migrations_v3_3,
        apply_migrations_v3_4,
        apply_migrations_v3_5,
        connect_sqlite,
    )

    src_conn = connect_sqlite(source_sqlite_path, read_only=True)
    try:
        # 1. Ensure schema V3.2..V3.5 + triggers + views exist on remote Turso
        for stmt in DDL_V3_STATEMENTS:
            target_conn.execute(stmt)
        apply_migrations_v3_3(target_conn)
        apply_migrations_v3_4(target_conn)
        apply_migrations_v3_5(target_conn)

        stats: Dict[str, Any] = {
            "source_db": str(source_sqlite_path),
            "include_forecasts": include_forecasts,
            "tables": {},
        }

        # 2. locations
        existing_locs = {
            r[0] for r in target_conn.execute("SELECT location_id FROM locations").fetchall()
        }
        loc_col_str = ", ".join(_LOCATIONS_COLS)
        loc_ph = ", ".join(["?"] * len(_LOCATIONS_COLS))
        loc_rows = src_conn.execute(f"SELECT {loc_col_str} FROM locations ORDER BY location_id").fetchall()
        loc_stmts = [
            (
                f"INSERT INTO locations ({loc_col_str}) VALUES ({loc_ph})",
                tuple(r),
            )
            for r in loc_rows
            if r["location_id"] not in existing_locs
        ]
        if loc_stmts:
            target_conn.execute_batch(loc_stmts, atomic=True)

        # 3. ingestion_runs
        existing_runs = {
            r[0] for r in target_conn.execute("SELECT run_id FROM ingestion_runs").fetchall()
        }
        ir_col_str = ", ".join(_INGESTION_RUNS_COLS)
        ir_ph = ", ".join(["?"] * len(_INGESTION_RUNS_COLS))
        ir_rows = src_conn.execute(
            f"SELECT {ir_col_str} FROM ingestion_runs ORDER BY started_at_utc"
        ).fetchall()
        ir_stmts = [
            (
                f"INSERT INTO ingestion_runs ({ir_col_str}) VALUES ({ir_ph})",
                tuple(r),
            )
            for r in ir_rows
            if r["run_id"] not in existing_runs
        ]
        if ir_stmts:
            target_conn.execute_batch(ir_stmts, atomic=True)

        # 4. observations_vintage (chronological + version_num order for triggers/vintages)
        existing_obs = {
            (r[0], r[1], int(r[2]))
            for r in target_conn.execute(
                "SELECT location_id, obs_date, version_num FROM observations_vintage"
            ).fetchall()
        }
        obs_cols = _obs_all_cols(ALL_35_VARS)
        obs_col_str = ", ".join(obs_cols)
        obs_ph = ", ".join(["?"] * len(obs_cols))
        obs_sql = f"INSERT INTO observations_vintage ({obs_col_str}) VALUES ({obs_ph})"
        obs_rows = src_conn.execute(
            f"SELECT {obs_col_str} FROM observations_vintage ORDER BY location_id, obs_date, version_num"
        ).fetchall()
        pending_obs = [
            (obs_sql, tuple(r))
            for r in obs_rows
            if (r["location_id"], r["obs_date"], int(r["version_num"])) not in existing_obs
        ]
        for start in range(0, len(pending_obs), batch_size):
            target_conn.execute_batch(pending_obs[start : start + batch_size], atomic=True)

        # 5. Optional forecasts_ledger (strictly skipped when include_forecasts=False)
        if include_forecasts:
            existing_fc = {
                (r[0], r[1], r[2], int(r[3]))
                for r in target_conn.execute(
                    "SELECT location_id, model_id, data_as_of, horizon_day FROM forecasts_ledger"
                ).fetchall()
            }
            fc_col_str = ", ".join(_FORECASTS_LEDGER_META_COLS)
            fc_ph = ", ".join(["?"] * len(_FORECASTS_LEDGER_META_COLS))
            fc_sql = f"INSERT INTO forecasts_ledger ({fc_col_str}) VALUES ({fc_ph})"
            fc_rows = src_conn.execute(
                f"SELECT {fc_col_str} FROM forecasts_ledger ORDER BY forecast_id"
            ).fetchall()
            pending_fc = [
                (fc_sql, tuple(r))
                for r in fc_rows
                if (r["location_id"], r["model_id"], r["data_as_of"], int(r["horizon_day"])) not in existing_fc
            ]
            for start in range(0, len(pending_fc), batch_size):
                target_conn.execute_batch(pending_fc[start : start + batch_size], atomic=True)

        # 6. Verify parity per table
        tables_to_verify = ["locations", "ingestion_runs", "observations_vintage"]
        if include_forecasts:
            tables_to_verify.append("forecasts_ledger")
        for tbl in tables_to_verify:
            src_cnt, src_sha = compute_table_content_sha256(src_conn, tbl)
            tgt_cnt, tgt_sha = compute_table_content_sha256(target_conn, tbl)
            stats["tables"][tbl] = {
                "source_count": src_cnt,
                "target_count": tgt_cnt,
                "source_sha256": src_sha,
                "target_sha256": tgt_sha,
                "parity_ok": (src_cnt == tgt_cnt and src_sha == tgt_sha),
            }
            if not stats["tables"][tbl]["parity_ok"]:
                raise RuntimeError(
                    f"[TURSO_MIGRATION_PARITY_MISMATCH] Table {tbl}: "
                    f"src=({src_cnt}, {src_sha[:12]}) != tgt=({tgt_cnt}, {tgt_sha[:12]})."
                )
        if not include_forecasts:
            tgt_fc_cnt = int(target_conn.execute("SELECT COUNT(*) FROM forecasts_ledger").fetchone()[0])
            stats["tables"]["forecasts_ledger"] = {
                "source_count": int(src_conn.execute("SELECT COUNT(*) FROM forecasts_ledger").fetchone()[0]),
                "target_count": tgt_fc_cnt,
                "observations_only_enforced": tgt_fc_cnt == 0,
            }
        return stats
    finally:
        src_conn.close()
