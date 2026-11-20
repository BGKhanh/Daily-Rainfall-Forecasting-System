"""
Weather History Provider Abstraction (DS108)
============================================
Decouples FastAPI operational serving from raw archival CSVs by supporting
selectable history backends via WEATHER_DB_BACKEND:
- "local" (default): uses SQLite Vintage Store (`data/vintage/rainfall_vintage.db`) when present,
  falling back cleanly to the bundled 35-day canonical sample CSV (`sample_recent_history_35d.csv`)
  in stateless CI/container environments without a mounted DB.
- "sqlite": explicit SQLite Vintage Store (`v_observations_latest`) adapter using DDL V3.2.
- "sample_csv": explicit static 35-day canonical sample CSV adapter.
- "supabase": PostgreSQL / PostgREST HTTPS adapter for Supabase deployments.
- "turso": LibSQL HTTP pipeline adapter for Turso serverless SQLite.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config.constants import Config
from src.data.turso_backend import open_store_connection, resolve_store_backend
from src.data.vintage_store import (
    ALL_35_VARS,
    HCMC_LOCATION_ID,
    align_and_validate_canonical_raw_df,
    connect_sqlite,
    get_default_complete_anchor_date,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_HISTORY_PATH = PROJECT_ROOT / "artifacts" / "models" / "sample_recent_history_35d.csv"
DEFAULT_VINTAGE_DB_PATH = PROJECT_ROOT / "data" / "vintage" / "rainfall_vintage.db"
DEFAULT_DEMO_VINTAGE_DB_PATH = PROJECT_ROOT / "artifacts" / "models" / "demo_rainfall_vintage.db"
SUPPORTED_BACKENDS = ("local", "sqlite", "sample_csv", "supabase", "turso")


def resolve_sqlite_db_path_and_role(explicit_db_path: Optional[Path] = None) -> tuple[Optional[Path], str]:
    """
    Resolve active SQLite database path and explicit DB role according to Two-DB convention:
    1. Explicit `db_path` argument (e.g. test fixtures) -> `(path, "local_or_production_db")`
    2. `VINTAGE_DB_PATH` env var -> `(path, "local_or_production_db")`
    3. `data/vintage/rainfall_vintage.db` if exists -> `(path, "local_or_production_db")`
    4. `artifacts/models/demo_rainfall_vintage.db` if exists -> `(path, "bundled_demo_db")`
    5. Otherwise -> `(None, "sample_csv")`
    """
    if explicit_db_path is not None:
        return Path(explicit_db_path), "local_or_production_db"
    env_db = os.environ.get("VINTAGE_DB_PATH", "").strip()
    if env_db:
        return Path(env_db), "local_or_production_db"
    if DEFAULT_VINTAGE_DB_PATH.exists():
        return DEFAULT_VINTAGE_DB_PATH, "local_or_production_db"
    if DEFAULT_DEMO_VINTAGE_DB_PATH.exists():
        return DEFAULT_DEMO_VINTAGE_DB_PATH, "bundled_demo_db"
    return None, "sample_csv"


class IncompleteRadiationAtAnchorError(ValueError):
    """Raised when an anchor date lacks COMPLETE_ALL_35 radiation/meteo completeness (Invariant #5)."""

    def __init__(self, message: str, *, anchor_date: str, completeness_regime: Optional[str] = None) -> None:
        self.code = "INCOMPLETE_RADIATION_AT_ANCHOR"
        self.anchor_date = anchor_date
        self.completeness_regime = completeness_regime
        super().__init__(f"[{self.code}] {message}")


@dataclass(frozen=True)
class WeatherHistoryResult:
    """Container returned by WeatherHistoryProvider adapters."""

    history_df: pd.DataFrame
    backend: str
    is_static_sample: bool
    data_as_of: pd.Timestamp
    source_detail: str
    observed_precip_by_date: Optional[Dict[str, float]] = None
    anchor_completeness_regime: Optional[str] = None
    db_role: str = "sample_csv"


class BaseWeatherHistoryAdapter(ABC):
    """Abstract adapter interface for fetching recent daily weather history."""

    backend_name: str

    @abstractmethod
    def fetch_recent_history(
        self,
        province: str = "TP.HCM",
        days: int = 35,
        run_date: Optional[str] = None,
        anchor_date: Optional[str] = None,
    ) -> WeatherHistoryResult:
        """Fetch at least `days` consecutive historical daily rows ending on or before run_date."""


class SQLiteVintageHistoryAdapter(BaseWeatherHistoryAdapter):
    """Reads canonical daily observations using indexed vintage queries (local SQLite or Turso)."""

    backend_name = "sqlite"

    def __init__(
        self,
        db_path: Optional[Path] = None,
        location_id: str = HCMC_LOCATION_ID,
        db_role: Optional[str] = None,
    ) -> None:
        self._explicit_db_path = db_path is not None
        resolved_path, inferred_role = resolve_sqlite_db_path_and_role(db_path)
        self.db_path = resolved_path if resolved_path is not None else DEFAULT_VINTAGE_DB_PATH
        self.db_role = db_role or inferred_role
        self.location_id = location_id

    def fetch_recent_history(
        self,
        province: str = "TP.HCM",
        days: int = 35,
        run_date: Optional[str] = None,
        anchor_date: Optional[str] = None,
    ) -> WeatherHistoryResult:
        del province  # Location is strictly locked to canonical HCMC.
        store_backend = resolve_store_backend()
        use_turso = not self._explicit_db_path and store_backend in ("turso_replica", "turso_remote")
        if use_turso:
            conn, resolved_backend = open_store_connection(read_only=True, backend=store_backend)
            active_db_role = resolved_backend
            source_detail = f"{resolved_backend}:turso"
        else:
            if not self.db_path.exists():
                raise FileNotFoundError(f"SQLite vintage DB not found: {self.db_path}")
            conn = connect_sqlite(self.db_path, read_only=True)
            active_db_role = self.db_role
            source_detail = f"sqlite:{self.db_path.name}"

        try:
            avail_rows = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT obs_date
                        FROM observations_vintage
                        WHERE location_id = ?
                        GROUP BY obs_date
                        LIMIT ?
                    )
                    """,
                    (self.location_id, int(days)),
                ).fetchone()[0]
            )
            if avail_rows < days:
                raise RuntimeError(
                    f"[SERVER_DB_DATA_ERROR] Vintage store has only {avail_rows} day(s), need >= {days} continuous days."
                )

            if anchor_date is not None:
                parsed_anchor = pd.to_datetime(anchor_date, errors="coerce")
                if pd.isna(parsed_anchor):
                    raise IncompleteRadiationAtAnchorError(
                        f"Invalid anchor_date format {anchor_date!r}; expected YYYY-MM-DD.",
                        anchor_date=str(anchor_date),
                    )
                chosen_anchor = pd.Timestamp(parsed_anchor).strftime("%Y-%m-%d")
                anchor_row = conn.execute(
                    """
                    SELECT obs_date, completeness_regime, missing_count,
                           missing_meteo_count, missing_flashflux_count, missing_ceres_count
                    FROM observations_vintage
                    WHERE location_id = ? AND obs_date = ?
                    ORDER BY version_num DESC
                    LIMIT 1
                    """,
                    (self.location_id, chosen_anchor),
                ).fetchone()
                if anchor_row is None:
                    raise IncompleteRadiationAtAnchorError(
                        f"Requested anchor_date={chosen_anchor} has no published observations in SQLite vintage store.",
                        anchor_date=chosen_anchor,
                        completeness_regime=None,
                    )
                if str(anchor_row["completeness_regime"]) != "COMPLETE_ALL_35":
                    raise IncompleteRadiationAtAnchorError(
                        (
                            f"Requested anchor_date={chosen_anchor} has completeness_regime="
                            f"{anchor_row['completeness_regime']!r} (missing_meteo={anchor_row['missing_meteo_count']}, "
                            f"missing_flashflux={anchor_row['missing_flashflux_count']}, "
                            f"missing_ceres={anchor_row['missing_ceres_count']}); "
                            "operational ML inference requires COMPLETE_ALL_35."
                        ),
                        anchor_date=chosen_anchor,
                        completeness_regime=str(anchor_row["completeness_regime"]),
                    )
                anchor_regime = str(anchor_row["completeness_regime"])
            else:
                chosen_anchor = get_default_complete_anchor_date(
                    conn,
                    location_id=self.location_id,
                    max_date=run_date,
                )
                if chosen_anchor is None:
                    raise RuntimeError(
                        f"[SERVER_DB_DATA_ERROR] No valid COMPLETE_ALL_35 anchor date with continuous 30-day window "
                        f"found in {source_detail} (max_date={run_date})."
                    )
                anchor_regime = "COMPLETE_ALL_35"

            anchor_dt = pd.to_datetime(chosen_anchor).normalize()
            win_start = (anchor_dt - timedelta(days=days - 1)).strftime("%Y-%m-%d")

            rows = conn.execute(
                f"""
                SELECT o.obs_date, o.completeness_regime, {', '.join(f'o.{c}' for c in ALL_35_VARS)}
                FROM observations_vintage o
                WHERE o.location_id = ?
                  AND o.obs_date BETWEEN ? AND ?
                  AND o.version_num = (
                      SELECT MAX(v2.version_num)
                      FROM observations_vintage v2
                      WHERE v2.location_id = o.location_id
                        AND v2.obs_date = o.obs_date
                  )
                ORDER BY o.obs_date ASC
                """,
                (self.location_id, win_start, chosen_anchor),
            ).fetchall()

            if len(rows) < days:
                raise RuntimeError(
                    f"[SERVER_DB_DATA_ERROR] Expected {days} continuous daily rows in [{win_start}, {chosen_anchor}], "
                    f"got {len(rows)}."
                )

            incomplete_in_win = [
                (str(r["obs_date"]), str(r["completeness_regime"]))
                for r in rows
                if str(r["completeness_regime"]) != "COMPLETE_ALL_35"
            ]
            if anchor_date is not None and incomplete_in_win:
                bad_date, bad_reg = incomplete_in_win[-1]
                raise IncompleteRadiationAtAnchorError(
                    (
                        f"35-day window ending {chosen_anchor} contains incomplete observation on {bad_date} "
                        f"(completeness_regime={bad_reg!r})."
                    ),
                    anchor_date=chosen_anchor,
                    completeness_regime=bad_reg,
                )

            target_start = (anchor_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            target_end = (anchor_dt + timedelta(days=7)).strftime("%Y-%m-%d")
            target_rows = conn.execute(
                """
                SELECT o.obs_date, o.PRECTOTCORR
                FROM observations_vintage o
                WHERE o.location_id = ?
                  AND o.obs_date BETWEEN ? AND ?
                  AND o.PRECTOTCORR IS NOT NULL
                  AND o.version_num = (
                      SELECT MAX(v2.version_num)
                      FROM observations_vintage v2
                      WHERE v2.location_id = o.location_id
                        AND v2.obs_date = o.obs_date
                  )
                ORDER BY o.obs_date ASC
                """,
                (self.location_id, target_start, target_end),
            ).fetchall()
            observed_precip = {str(r["obs_date"]): float(r["PRECTOTCORR"]) for r in target_rows}
        finally:
            conn.close()

        row_cols = ["obs_date", "completeness_regime", *ALL_35_VARS]
        df = pd.DataFrame([dict(zip(row_cols, r)) for r in rows]).drop(columns=["completeness_regime"])
        df = df.rename(columns={"obs_date": "DATE"})
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.normalize()
        en_to_vi = {k: v for k, v in Config.COLUMN_MAPPING.items() if k == "DATE" or k in ALL_35_VARS}
        df = df.rename(columns=en_to_vi)
        df = align_and_validate_canonical_raw_df(df).sort_values("Ngày").reset_index(drop=True)

        return WeatherHistoryResult(
            history_df=df,
            backend=self.backend_name,
            is_static_sample=False,
            data_as_of=anchor_dt,
            source_detail=source_detail,
            observed_precip_by_date=observed_precip,
            anchor_completeness_regime=anchor_regime,
            db_role=active_db_role,
        )


class LocalWeatherHistoryAdapter(BaseWeatherHistoryAdapter):
    """Reads the lightweight 35-day exported sample CSV, with DataLoader fallback for dev setups."""

    backend_name = "local"

    def __init__(self, sample_csv_path: Optional[Path] = None) -> None:
        self.sample_csv_path = Path(sample_csv_path) if sample_csv_path else DEFAULT_SAMPLE_HISTORY_PATH

    def fetch_recent_history(
        self,
        province: str = "TP.HCM",
        days: int = 35,
        run_date: Optional[str] = None,
        anchor_date: Optional[str] = None,
    ) -> WeatherHistoryResult:
        del province, anchor_date  # Static sample fallback has fixed HCMC 35d window.
        if self.sample_csv_path.exists():
            df = pd.read_csv(self.sample_csv_path)
            source_detail = str(self.sample_csv_path)
        else:
            from src.data.loader import DataLoader

            df = DataLoader().load_data()
            source_detail = "DataLoader.load_data()"

        df["Ngày"] = pd.to_datetime(df["Ngày"]).dt.normalize()
        df = df.sort_values("Ngày").reset_index(drop=True)

        if run_date is not None and not self.sample_csv_path.exists():
            cutoff = pd.to_datetime(run_date).normalize()
            eligible = df[df["Ngày"] <= cutoff]
            if len(eligible) >= days:
                df = eligible

        recent_df = df.tail(days).copy().reset_index(drop=True)
        data_as_of = pd.Timestamp(recent_df["Ngày"].max()).normalize()
        return WeatherHistoryResult(
            history_df=recent_df,
            backend=self.backend_name,
            is_static_sample=True,
            data_as_of=data_as_of,
            source_detail=source_detail,
        )


class SupabaseWeatherHistoryAdapter(BaseWeatherHistoryAdapter):
    """Fetches daily weather history rows from a Supabase PostgREST endpoint."""

    backend_name = "supabase"

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        table: Optional[str] = None,
        timeout_sec: float = 15.0,
    ) -> None:
        self.url = (url or os.environ.get("SUPABASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY", "")
        self.table = table or os.environ.get("SUPABASE_WEATHER_TABLE", "daily_weather_history")
        self.timeout_sec = timeout_sec

    def _validate_config(self) -> None:
        if not self.url or not self.api_key:
            raise RuntimeError(
                "Supabase backend selected (WEATHER_DB_BACKEND=supabase) but "
                "SUPABASE_URL and SUPABASE_SERVICE_KEY/SUPABASE_ANON_KEY are not configured."
            )

    def fetch_recent_history(
        self,
        province: str = "TP.HCM",
        days: int = 35,
        run_date: Optional[str] = None,
        anchor_date: Optional[str] = None,
    ) -> WeatherHistoryResult:
        del anchor_date
        self._validate_config()
        params: Dict[str, Any] = {
            "select": "*",
            "province": f"eq.{province}",
            "order": "Ngày.desc",
            "limit": str(int(days)),
        }
        if run_date is not None:
            params["Ngày"] = f"lte.{pd.to_datetime(run_date).strftime('%Y-%m-%d')}"

        query_string = urllib.parse.urlencode(params, safe=".,()")
        endpoint = f"{self.url}/rest/v1/{urllib.parse.quote(self.table)}?{query_string}"
        req = urllib.request.Request(
            endpoint,
            headers={
                "apikey": self.api_key,
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                rows = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to query Supabase weather history: {exc}") from exc

        df = pd.DataFrame(rows)
        if df.empty or "Ngày" not in df.columns:
            raise RuntimeError(f"Supabase returned empty or schema-invalid history for province '{province}'.")

        df["Ngày"] = pd.to_datetime(df["Ngày"]).dt.normalize()
        df = df.sort_values("Ngày").reset_index(drop=True)
        return WeatherHistoryResult(
            history_df=df,
            backend=self.backend_name,
            is_static_sample=False,
            data_as_of=pd.Timestamp(df["Ngày"].max()).normalize(),
            source_detail=f"supabase:{self.table}",
        )


class TursoWeatherHistoryAdapter(BaseWeatherHistoryAdapter):
    """Fetches daily weather history rows from a Turso / libSQL HTTP pipeline endpoint."""

    backend_name = "turso"

    def __init__(
        self,
        database_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        table: Optional[str] = None,
        timeout_sec: float = 15.0,
    ) -> None:
        raw_url = (database_url or os.environ.get("TURSO_DATABASE_URL", "")).strip()
        if raw_url.startswith("libsql://"):
            raw_url = "https://" + raw_url[len("libsql://"):]
        self.database_url = raw_url.rstrip("/")
        self.auth_token = auth_token or os.environ.get("TURSO_AUTH_TOKEN", "")
        self.table = table or os.environ.get("TURSO_WEATHER_TABLE", "daily_weather_history")
        self.timeout_sec = timeout_sec

    def _validate_config(self) -> None:
        if not self.database_url or not self.auth_token:
            raise RuntimeError(
                "Turso backend selected (WEATHER_DB_BACKEND=turso) but "
                "TURSO_DATABASE_URL and TURSO_AUTH_TOKEN are not configured."
            )

    def fetch_recent_history(
        self,
        province: str = "TP.HCM",
        days: int = 35,
        run_date: Optional[str] = None,
        anchor_date: Optional[str] = None,
    ) -> WeatherHistoryResult:
        del anchor_date
        self._validate_config()
        if not self.table.replace("_", "").isalnum():
            raise ValueError(f"Unsafe Turso table identifier: '{self.table}'")

        args = [{"type": "text", "value": province}]
        where_clause = "WHERE province = ?"
        if run_date is not None:
            where_clause += ' AND "Ngày" <= ?'
            args.append({"type": "text", "value": pd.to_datetime(run_date).strftime("%Y-%m-%d")})
        args.append({"type": "integer", "value": str(int(days))})

        sql = f'SELECT * FROM "{self.table}" {where_clause} ORDER BY "Ngày" DESC LIMIT ?'
        payload = json.dumps(
            {
                "requests": [
                    {"type": "execute", "stmt": {"sql": sql, "args": args}},
                    {"type": "close"},
                ]
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{self.database_url}/v2/pipeline",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to query Turso weather history: {exc}") from exc

        result_block = body.get("results", [{}])[0].get("response", {}).get("result", {})
        cols = [c["name"] for c in result_block.get("cols", [])]
        raw_rows = result_block.get("rows", [])
        parsed_rows = [
            {col: (cell.get("value") if isinstance(cell, dict) else cell) for col, cell in zip(cols, row)}
            for row in raw_rows
        ]
        df = pd.DataFrame(parsed_rows)
        if df.empty or "Ngày" not in df.columns:
            raise RuntimeError(f"Turso returned empty or schema-invalid history for province '{province}'.")

        df["Ngày"] = pd.to_datetime(df["Ngày"]).dt.normalize()
        df = df.sort_values("Ngày").reset_index(drop=True)
        return WeatherHistoryResult(
            history_df=df,
            backend=self.backend_name,
            is_static_sample=False,
            data_as_of=pd.Timestamp(df["Ngày"].max()).normalize(),
            source_detail=f"turso:{self.table}",
        )


class WeatherHistoryProvider:
    """Factory and router selecting SQLite vintage DB, local sample CSV, or cloud DB history adapters."""

    def __init__(
        self,
        backend: Optional[str] = None,
        sample_csv_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        fallback_to_local: bool = False,
    ) -> None:
        resolved = (backend or os.environ.get("WEATHER_DB_BACKEND", "local")).strip().lower()
        if resolved not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported WEATHER_DB_BACKEND='{resolved}'. Expected one of {SUPPORTED_BACKENDS}."
            )
        self.backend = resolved
        self.sample_csv_path = sample_csv_path
        resolved_db, resolved_role = resolve_sqlite_db_path_and_role(db_path)
        self.explicit_db_path = db_path
        self.db_path = resolved_db
        self.db_role = "sample_csv" if resolved == "sample_csv" else resolved_role
        self.fallback_to_local = fallback_to_local
        self._adapter = self._build_adapter(resolved)

    def _build_adapter(self, backend: str) -> BaseWeatherHistoryAdapter:
        if backend == "sqlite":
            return SQLiteVintageHistoryAdapter(
                db_path=self.db_path or DEFAULT_VINTAGE_DB_PATH,
                db_role=self.db_role if self.db_path is not None else "local_or_production_db",
            )
        if backend == "sample_csv":
            self.db_role = "sample_csv"
            return LocalWeatherHistoryAdapter(sample_csv_path=self.sample_csv_path)
        if backend == "local":
            if self.explicit_db_path is None and resolve_store_backend() in ("turso_replica", "turso_remote"):
                sb = resolve_store_backend()
                self.db_role = sb
                return SQLiteVintageHistoryAdapter(db_path=None, db_role=sb)
            if self.sample_csv_path is None and self.db_path is not None and self.db_path.exists():
                return SQLiteVintageHistoryAdapter(db_path=self.db_path, db_role=self.db_role)
            self.db_role = "sample_csv"
            return LocalWeatherHistoryAdapter(sample_csv_path=self.sample_csv_path)
        if backend == "supabase":
            self.db_role = "supabase"
            return SupabaseWeatherHistoryAdapter()
        if backend == "turso":
            self.db_role = "turso"
            return TursoWeatherHistoryAdapter()
        raise ValueError(f"Unsupported WEATHER_DB_BACKEND='{backend}'.")

    def fetch_recent_history(
        self,
        province: str = "TP.HCM",
        days: int = 35,
        run_date: Optional[str] = None,
        anchor_date: Optional[str] = None,
    ) -> WeatherHistoryResult:
        try:
            kwargs: Dict[str, Any] = {"province": province, "days": days, "run_date": run_date}
            if anchor_date is not None:
                kwargs["anchor_date"] = anchor_date
            return self._adapter.fetch_recent_history(**kwargs)
        except Exception:
            if self.fallback_to_local and self.backend not in ("local", "sample_csv"):
                return LocalWeatherHistoryAdapter(sample_csv_path=self.sample_csv_path).fetch_recent_history(
                    province=province,
                    days=days,
                    run_date=run_date,
                )
            raise

