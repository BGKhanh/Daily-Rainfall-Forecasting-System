"""
Weather History Provider Abstraction (DS108)
============================================
Decouples FastAPI operational serving from the raw local CSV archive by
supporting selectable history backends via WEATHER_DB_BACKEND:
- "local" (default): lightweight bundled 35-day canonical sample CSV
  (artifacts/models/sample_recent_history_35d.csv) with fallback to DataLoader.
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
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_HISTORY_PATH = PROJECT_ROOT / "artifacts" / "models" / "sample_recent_history_35d.csv"
SUPPORTED_BACKENDS = ("local", "supabase", "turso")


@dataclass(frozen=True)
class WeatherHistoryResult:
    """Container returned by WeatherHistoryProvider adapters."""

    history_df: pd.DataFrame
    backend: str
    is_static_sample: bool
    data_as_of: pd.Timestamp
    source_detail: str


class BaseWeatherHistoryAdapter(ABC):
    """Abstract adapter interface for fetching recent daily weather history."""

    backend_name: str

    @abstractmethod
    def fetch_recent_history(
        self,
        province: str = "TP.HCM",
        days: int = 35,
        run_date: Optional[str] = None,
    ) -> WeatherHistoryResult:
        """Fetch at least `days` consecutive historical daily rows ending on or before run_date."""


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
    ) -> WeatherHistoryResult:
        del province  # Canonical static sample represents HCMC station; future multi-station DB keys by province.
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
    ) -> WeatherHistoryResult:
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
    ) -> WeatherHistoryResult:
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
    """Factory and router selecting local sample vs cloud DB history adapters."""

    def __init__(
        self,
        backend: Optional[str] = None,
        sample_csv_path: Optional[Path] = None,
        fallback_to_local: bool = False,
    ) -> None:
        resolved = (backend or os.environ.get("WEATHER_DB_BACKEND", "local")).strip().lower()
        if resolved not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported WEATHER_DB_BACKEND='{resolved}'. Expected one of {SUPPORTED_BACKENDS}."
            )
        self.backend = resolved
        self.sample_csv_path = sample_csv_path
        self.fallback_to_local = fallback_to_local
        self._adapter = self._build_adapter(resolved)

    def _build_adapter(self, backend: str) -> BaseWeatherHistoryAdapter:
        if backend == "local":
            return LocalWeatherHistoryAdapter(sample_csv_path=self.sample_csv_path)
        if backend == "supabase":
            return SupabaseWeatherHistoryAdapter()
        if backend == "turso":
            return TursoWeatherHistoryAdapter()
        raise ValueError(f"Unsupported WEATHER_DB_BACKEND='{backend}'.")

    def fetch_recent_history(
        self,
        province: str = "TP.HCM",
        days: int = 35,
        run_date: Optional[str] = None,
    ) -> WeatherHistoryResult:
        try:
            return self._adapter.fetch_recent_history(province=province, days=days, run_date=run_date)
        except Exception:
            if self.fallback_to_local and self.backend != "local":
                return LocalWeatherHistoryAdapter(sample_csv_path=self.sample_csv_path).fetch_recent_history(
                    province=province,
                    days=days,
                    run_date=run_date,
                )
            raise

