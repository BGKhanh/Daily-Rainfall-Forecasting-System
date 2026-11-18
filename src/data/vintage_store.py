"""
Immutable Vintage Time-Series & Forecast Ledger Store (DDL V3.2)
================================================================
Implements the frozen V3.2 SQLite (3.35+ / Turso libSQL compatible) schema for:
- `locations` (single source of truth for `HCMC` coordinates & `provisional_window_days`)
- `ingestion_runs` (strict append-only audit log with `SUCCESS` / `FAILED` states only & `rows_unpublished_skipped`)
- `observations_vintage` (append-only meteorological vintages + deterministic completeness checks)
- `forecasts_ledger` (immutable issued forecasts + `is_degraded_input` contract + deterministic `eval_split_regime` / `temporal_status_at_issue`)
- 9 immutability triggers (`BEFORE INSERT` no-replace, `BEFORE UPDATE`, `BEFORE DELETE`)
- 3 derived views (`v_observations_latest`, `v_observations_first_seen`, `v_forecast_scores` with `is_prospective`)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from src.config.clock import now_utc_iso
from src.config.constants import Config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HCMC_LOCATION_ID = "HCMC"
HCMC_DISPLAY_NAME = "TP. Hồ Chí Minh"
HCMC_LATITUDE = 10.7769
HCMC_LONGITUDE = 106.7009
DEFAULT_PROVISIONAL_WINDOW_DAYS = 120
HCMC_ALIASES: Tuple[str, ...] = (
    "HCMC",
    "HO_CHI_MINH",
    "HOCHIMINH",
    "SAIGON",
    "TPHCM",
    "TP.HCM",
    "TP. HỒ CHÍ MINH",
    "TP HO CHI MINH",
)

FLASHFLUX_3_VARS: Tuple[str, ...] = (
    "ALLSKY_SFC_SW_DWN",
    "TOA_SW_DWN",
    "ALLSKY_SFC_LW_DWN",
)

CERES_ONLY_8_VARS: Tuple[str, ...] = (
    "CLRSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
    "ALLSKY_SFC_PAR_TOT",
    "CLRSKY_SFC_PAR_TOT",
    "ALLSKY_SFC_UVA",
    "ALLSKY_SFC_UVB",
    "ALLSKY_SFC_UV_INDEX",
)

METEO_24_VARS: Tuple[str, ...] = (
    "T2M",
    "T2MDEW",
    "T2MWET",
    "TS",
    "T2M_RANGE",
    "T2M_MAX",
    "T2M_MIN",
    "QV2M",
    "RH2M",
    "PRECTOTCORR",
    "PS",
    "WS2M",
    "WS2M_MAX",
    "WS2M_MIN",
    "WS2M_RANGE",
    "WD2M",
    "WS10M",
    "WS10M_MAX",
    "WS10M_MIN",
    "WS10M_RANGE",
    "WD10M",
    "GWETTOP",
    "GWETROOT",
    "GWETPROF",
)

ALL_35_VARS: Tuple[str, ...] = FLASHFLUX_3_VARS + CERES_ONLY_8_VARS + METEO_24_VARS

# Canonical DataLoader / training column order (matches FeatureBuilder positional ndarray contract)
CANONICAL_EN_COLUMN_ORDER: Tuple[str, ...] = ("DATE",) + tuple(
    k for k in Config.COLUMN_MAPPING if k in ALL_35_VARS
)
CANONICAL_VI_COLUMN_ORDER: Tuple[str, ...] = tuple(
    Config.COLUMN_MAPPING[k] for k in CANONICAL_EN_COLUMN_ORDER
)

# Reverse mapping: Vietnamese column names -> English NASA parameter names
VI_TO_EN_COLUMNS: Dict[str, str] = {vi: en for en, vi in Config.COLUMN_MAPPING.items() if en in ALL_35_VARS or en == "DATE"}
EN_TO_VI_CANONICAL: Dict[str, str] = {en: Config.COLUMN_MAPPING[en] for en in CANONICAL_EN_COLUMN_ORDER}


def align_and_validate_canonical_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize a raw meteorological DataFrame to exact Vietnamese names & column order
    (`CANONICAL_VI_COLUMN_ORDER`, 36 columns matching `DataLoader` / `Config.COLUMN_MAPPING`).

    Accepts either Vietnamese or NASA POWER English column names; raises
    `ValueError("[CANONICAL_COLUMN_ERROR] ...")` on missing or unexpected columns.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("[CANONICAL_COLUMN_ERROR] Expected a pandas DataFrame.")

    out = df.copy()
    rename_map = {en: vi for en, vi in EN_TO_VI_CANONICAL.items() if en in out.columns and vi not in out.columns}
    if rename_map:
        out = out.rename(columns=rename_map)

    expected = list(CANONICAL_VI_COLUMN_ORDER)
    expected_set = set(expected)
    actual_cols = list(out.columns)
    actual_set = set(actual_cols)

    missing = [col for col in expected if col not in actual_set]
    extra = [col for col in actual_cols if col not in expected_set]
    if missing or extra:
        raise ValueError(
            f"[CANONICAL_COLUMN_ERROR] Missing required columns: {missing}; unexpected columns: {extra}"
        )

    return out.loc[:, expected].copy()

DDL_V3_STATEMENTS: Tuple[str, ...] = (
    # 1. locations
    """
    CREATE TABLE IF NOT EXISTS locations (
        location_id              TEXT PRIMARY KEY,
        display_name             TEXT NOT NULL,
        aliases_json             TEXT NOT NULL DEFAULT '["HCMC","HO_CHI_MINH","HOCHIMINH","SAIGON","TPHCM","TP.HCM","TP. HỒ CHÍ MINH","TP HO CHI MINH"]',
        latitude                 REAL NOT NULL,
        longitude                REAL NOT NULL,
        meteo_grid_note          TEXT NOT NULL DEFAULT 'MERRA-2 / GEOS-IT 0.5° x 0.625°',
        rad_grid_note            TEXT NOT NULL DEFAULT 'CERES SYN1deg / FLASHFlux 1.0° x 1.0°',
        time_standard            TEXT NOT NULL DEFAULT 'LST',
        provisional_window_days  INTEGER NOT NULL DEFAULT 120 CHECK (provisional_window_days > 0),
        is_active                INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
        created_at_utc           TEXT NOT NULL
    );
    """,
    # 2. ingestion_runs
    """
    CREATE TABLE IF NOT EXISTS ingestion_runs (
        run_id                     TEXT PRIMARY KEY,
        run_type                   TEXT NOT NULL CHECK (run_type IN ('historical_seed', 'manual_scan', 'audit_probe')),
        started_at_utc             TEXT NOT NULL,
        finished_at_utc            TEXT NOT NULL,
        status                     TEXT NOT NULL CHECK (status IN ('SUCCESS', 'FAILED')),
        location_id                TEXT NOT NULL REFERENCES locations(location_id),
        scan_window_start          TEXT NOT NULL,
        scan_window_end            TEXT NOT NULL,
        nasa_api_version           TEXT,
        nasa_sources_header_json   TEXT,
        rows_scanned               INTEGER NOT NULL DEFAULT 0 CHECK (rows_scanned >= 0),
        rows_inserted_v0           INTEGER NOT NULL DEFAULT 0 CHECK (rows_inserted_v0 >= 0),
        rows_new_vintage           INTEGER NOT NULL DEFAULT 0 CHECK (rows_new_vintage >= 0),
        rows_late_fill_only        INTEGER NOT NULL DEFAULT 0 CHECK (rows_late_fill_only >= 0),
        rows_value_change          INTEGER NOT NULL DEFAULT 0 CHECK (rows_value_change >= 0),
        rows_unchanged             INTEGER NOT NULL DEFAULT 0 CHECK (rows_unchanged >= 0),
        rows_unpublished_skipped   INTEGER NOT NULL DEFAULT 0 CHECK (rows_unpublished_skipped >= 0),
        latest_valid_by_group_json TEXT,
        revision_stats_json        TEXT,
        git_sha                    TEXT NOT NULL,
        error_message              TEXT,
        CHECK (status = 'SUCCESS' OR (rows_inserted_v0 = 0 AND rows_new_vintage = 0 AND rows_unpublished_skipped = 0))
    );
    """,
    # 3. observations_vintage
    """
    CREATE TABLE IF NOT EXISTS observations_vintage (
        location_id               TEXT NOT NULL REFERENCES locations(location_id),
        obs_date                  TEXT NOT NULL,
        version_num               INTEGER NOT NULL CHECK (version_num >= 0),
        ingested_at_utc           TEXT NOT NULL,
        run_id                    TEXT NOT NULL REFERENCES ingestion_runs(run_id) DEFERRABLE INITIALLY DEFERRED,
        age_days_at_ingest        INTEGER NOT NULL,
        ingest_origin             TEXT NOT NULL CHECK (ingest_origin IN ('HISTORICAL_CSV_SEED', 'NASA_POWER_API')),
        missing_count             INTEGER NOT NULL DEFAULT 0 CHECK (missing_count BETWEEN 0 AND 35),
        missing_meteo_count       INTEGER NOT NULL DEFAULT 0 CHECK (missing_meteo_count BETWEEN 0 AND 24),
        missing_flashflux_count   INTEGER NOT NULL DEFAULT 0 CHECK (missing_flashflux_count BETWEEN 0 AND 3),
        missing_ceres_count       INTEGER NOT NULL DEFAULT 0 CHECK (missing_ceres_count BETWEEN 0 AND 8),
        -- Precedence: (1) Khí tượng (missing_meteo_count > 0 -> 'INCOMPLETE_METEO'), (2) FLASHFlux (missing_flashflux_count > 0 -> 'FLASHFLUX_INCOMPLETE'), (3) CERES (missing_ceres_count > 0 -> 'MISSING_CERES_ONLY'), (4) Đủ 35 biến ('COMPLETE_ALL_35')
        completeness_regime       TEXT NOT NULL CHECK (completeness_regime IN (
                                      'COMPLETE_ALL_35',
                                      'MISSING_CERES_ONLY',
                                      'FLASHFLUX_INCOMPLETE',
                                      'INCOMPLETE_METEO'
                                  )),
        missing_vars_json         TEXT NOT NULL DEFAULT '[]',
        change_type_summary       TEXT NOT NULL CHECK (change_type_summary IN (
                                      'INITIAL_SEED',
                                      'INITIAL_NRT',
                                      'LATE_FILL_ONLY',
                                      'VALUE_CHANGE_ONLY',
                                      'MIXED_FILL_AND_CHANGE'
                                  )),
        late_fill_count           INTEGER NOT NULL DEFAULT 0 CHECK (late_fill_count BETWEEN 0 AND 35),
        late_fill_vars_json       TEXT NOT NULL DEFAULT '[]',
        value_change_count        INTEGER NOT NULL DEFAULT 0 CHECK (value_change_count BETWEEN 0 AND 35),
        value_change_details_json TEXT NOT NULL DEFAULT '{}',
        anomaly_count             INTEGER NOT NULL DEFAULT 0 CHECK (anomaly_count >= 0),
        quality_flags_json        TEXT NOT NULL DEFAULT '{}',
        row_value_hash            TEXT NOT NULL,
        ALLSKY_SFC_SW_DWN         REAL,
        TOA_SW_DWN                REAL,
        ALLSKY_SFC_LW_DWN         REAL,
        CLRSKY_SFC_SW_DWN         REAL,
        ALLSKY_SFC_SW_DNI         REAL,
        ALLSKY_SFC_SW_DIFF        REAL,
        ALLSKY_SFC_PAR_TOT        REAL,
        CLRSKY_SFC_PAR_TOT        REAL,
        ALLSKY_SFC_UVA            REAL,
        ALLSKY_SFC_UVB            REAL,
        ALLSKY_SFC_UV_INDEX       REAL,
        T2M                       REAL,
        T2MDEW                    REAL,
        T2MWET                    REAL,
        TS                        REAL,
        T2M_RANGE                 REAL,
        T2M_MAX                   REAL,
        T2M_MIN                   REAL,
        QV2M                      REAL,
        RH2M                      REAL,
        PRECTOTCORR               REAL,
        PS                        REAL,
        WS2M                      REAL,
        WS2M_MAX                  REAL,
        WS2M_MIN                  REAL,
        WS2M_RANGE                REAL,
        WD2M                      REAL,
        WS10M                     REAL,
        WS10M_MAX                 REAL,
        WS10M_MIN                 REAL,
        WS10M_RANGE               REAL,
        WD10M                     REAL,
        GWETTOP                   REAL,
        GWETROOT                  REAL,
        GWETPROF                  REAL,
        PRIMARY KEY (location_id, obs_date, version_num),
        CHECK (missing_count = missing_meteo_count + missing_flashflux_count + missing_ceres_count),
        -- Precedence: (1) Khí tượng (missing_meteo_count > 0 -> 'INCOMPLETE_METEO'), (2) FLASHFlux (missing_flashflux_count > 0 -> 'FLASHFLUX_INCOMPLETE'), (3) CERES (missing_ceres_count > 0 -> 'MISSING_CERES_ONLY'), (4) Đủ 35 biến ('COMPLETE_ALL_35')
        CHECK (
            completeness_regime = CASE
                WHEN missing_meteo_count > 0 THEN 'INCOMPLETE_METEO'
                WHEN missing_flashflux_count > 0 THEN 'FLASHFLUX_INCOMPLETE'
                WHEN missing_ceres_count > 0 THEN 'MISSING_CERES_ONLY'
                ELSE 'COMPLETE_ALL_35'
            END
        ),
        CHECK (
            (version_num = 0 AND change_type_summary IN ('INITIAL_SEED', 'INITIAL_NRT') AND late_fill_count = 0 AND value_change_count = 0)
            OR
            (version_num > 0 AND change_type_summary IN ('LATE_FILL_ONLY', 'VALUE_CHANGE_ONLY', 'MIXED_FILL_AND_CHANGE') AND (late_fill_count + value_change_count) > 0)
        )
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_obs_vintage_loc_date_ver ON observations_vintage (location_id, obs_date, version_num DESC);",
    "CREATE INDEX IF NOT EXISTS idx_obs_vintage_as_of_time ON observations_vintage (location_id, ingested_at_utc, obs_date, version_num DESC);",
    # 4. forecasts_ledger
    """
    CREATE TABLE IF NOT EXISTS forecasts_ledger (
        forecast_id                 TEXT PRIMARY KEY,
        issued_at_utc               TEXT NOT NULL,
        issued_run_date             TEXT NOT NULL,
        run_timezone                TEXT NOT NULL DEFAULT 'Asia/Ho_Chi_Minh',
        location_id                 TEXT NOT NULL REFERENCES locations(location_id),
        model_id                    TEXT NOT NULL,
        model_kind                  TEXT NOT NULL CHECK (model_kind IN ('ML_SERVING', 'ML_SHADOW', 'BASELINE')),
        registry_alias              TEXT,
        registry_model_version      INTEGER,
        artifact_sha256             TEXT NOT NULL,
        feature_builder_sha256      TEXT,
        data_as_of                  TEXT NOT NULL,
        eval_split_regime           TEXT NOT NULL CHECK (eval_split_regime IN ('DEV_TRAIN', 'VALIDATION', 'TEST', 'POST_TEST')),
        horizon_day                 INTEGER NOT NULL CHECK (horizon_day BETWEEN 1 AND 7),
        target_date                 TEXT NOT NULL,
        temporal_status_at_issue    TEXT NOT NULL CHECK (temporal_status_at_issue IN ('past_observed', 'past_unverified', 'today', 'future_forecast')),
        pred_mm                     REAL NOT NULL CHECK (pred_mm >= 0.0),
        input_window_start          TEXT NOT NULL,
        input_window_end            TEXT NOT NULL,
        input_window_hash           TEXT NOT NULL,
        input_vintage_manifest_json TEXT NOT NULL,
        input_provisional_days      INTEGER NOT NULL CHECK (input_provisional_days BETWEEN 0 AND 30),
        meteo_missing_count         INTEGER NOT NULL DEFAULT 0 CHECK (meteo_missing_count = 0),
        flashflux_max_ffill_days    INTEGER NOT NULL DEFAULT 0 CHECK (flashflux_max_ffill_days BETWEEN 0 AND 7),
        ceres_policy_applied        TEXT NOT NULL CHECK (ceres_policy_applied IN (
                                        'NONE_COMPLETE',
                                        'POLICY_A_CLIMATOLOGY_DOY',
                                        'POLICY_B_LAG_90',
                                        'POLICY_B_LAG_120',
                                        'POLICY_C_NATIVE_NAN',
                                        'NOT_APPLICABLE_BASELINE'
                                    )),
        ceres_max_staleness_days    INTEGER NOT NULL DEFAULT 0 CHECK (ceres_max_staleness_days >= 0),
        is_degraded_input           INTEGER NOT NULL DEFAULT 0 CHECK (is_degraded_input IN (0, 1)),
        degraded_reason             TEXT,
        git_sha                     TEXT NOT NULL,
        UNIQUE (location_id, model_id, data_as_of, horizon_day),
        CHECK (
            is_degraded_input = CASE
                WHEN flashflux_max_ffill_days > 0
                  OR ceres_policy_applied NOT IN ('NONE_COMPLETE', 'NOT_APPLICABLE_BASELINE')
                THEN 1
                ELSE 0
            END
        )
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_forecasts_ledger_target ON forecasts_ledger (location_id, target_date, model_id);",
    # 5. Nine immutability triggers
    """
    CREATE TRIGGER IF NOT EXISTS trg_obs_vintage_no_replace
    BEFORE INSERT ON observations_vintage
    WHEN EXISTS (
        SELECT 1 FROM observations_vintage
        WHERE location_id = NEW.location_id
          AND obs_date = NEW.obs_date
          AND version_num = NEW.version_num
    )
    BEGIN
        SELECT RAISE(ABORT, 'IMMUTABLE_VIOLATION: observations_vintage row already exists; INSERT OR REPLACE is prohibited.');
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_obs_vintage_no_update
    BEFORE UPDATE ON observations_vintage
    BEGIN
        SELECT RAISE(ABORT, 'IMMUTABLE_VIOLATION: observations_vintage is append-only; UPDATE is prohibited.');
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_obs_vintage_no_delete
    BEFORE DELETE ON observations_vintage
    BEGIN
        SELECT RAISE(ABORT, 'IMMUTABLE_VIOLATION: observations_vintage is append-only; DELETE is prohibited.');
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_forecasts_ledger_no_replace
    BEFORE INSERT ON forecasts_ledger
    WHEN EXISTS (
        SELECT 1 FROM forecasts_ledger
        WHERE forecast_id = NEW.forecast_id
           OR (location_id = NEW.location_id
               AND model_id = NEW.model_id
               AND data_as_of = NEW.data_as_of
               AND horizon_day = NEW.horizon_day)
    )
    BEGIN
        SELECT RAISE(ABORT, 'IMMUTABLE_VIOLATION: forecast already issued for (location_id, model_id, data_as_of, horizon_day); overwrite/REPLACE is prohibited.');
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_forecasts_ledger_no_update
    BEFORE UPDATE ON forecasts_ledger
    BEGIN
        SELECT RAISE(ABORT, 'IMMUTABLE_VIOLATION: forecasts_ledger is immutable; UPDATE is prohibited.');
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_forecasts_ledger_no_delete
    BEFORE DELETE ON forecasts_ledger
    BEGIN
        SELECT RAISE(ABORT, 'IMMUTABLE_VIOLATION: forecasts_ledger is immutable; DELETE is prohibited.');
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_ingestion_runs_no_replace
    BEFORE INSERT ON ingestion_runs
    WHEN EXISTS (
        SELECT 1 FROM ingestion_runs WHERE run_id = NEW.run_id
    )
    BEGIN
        SELECT RAISE(ABORT, 'IMMUTABLE_VIOLATION: ingestion_runs row already exists; INSERT OR REPLACE is prohibited.');
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_ingestion_runs_no_update
    BEFORE UPDATE ON ingestion_runs
    BEGIN
        SELECT RAISE(ABORT, 'IMMUTABLE_VIOLATION: ingestion_runs is append-only; UPDATE is prohibited.');
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_ingestion_runs_no_delete
    BEFORE DELETE ON ingestion_runs
    BEGIN
        SELECT RAISE(ABORT, 'IMMUTABLE_VIOLATION: ingestion_runs is append-only; DELETE is prohibited.');
    END;
    """,
    # 6. Three derived views
    """
    CREATE VIEW IF NOT EXISTS v_observations_latest AS
    SELECT
        o.*,
        CASE
            WHEN o.missing_count > 0 THEN 1
            WHEN (julianday('now') - julianday(o.obs_date)) <= l.provisional_window_days THEN 1
            ELSE 0
        END AS is_provisional
    FROM observations_vintage o
    INNER JOIN (
        SELECT location_id, obs_date, MAX(version_num) AS max_ver
        FROM observations_vintage
        GROUP BY location_id, obs_date
    ) m ON o.location_id = m.location_id
       AND o.obs_date = m.obs_date
       AND o.version_num = m.max_ver
    INNER JOIN locations l ON o.location_id = l.location_id;
    """,
    """
    CREATE VIEW IF NOT EXISTS v_observations_first_seen AS
    SELECT
        o.*,
        CASE
            WHEN o.missing_count > 0 THEN 1
            WHEN o.age_days_at_ingest <= l.provisional_window_days THEN 1
            ELSE 0
        END AS was_provisional_at_first_ingest
    FROM observations_vintage o
    INNER JOIN (
        SELECT location_id, obs_date, MIN(version_num) AS min_ver
        FROM observations_vintage
        GROUP BY location_id, obs_date
    ) m ON o.location_id = m.location_id
       AND o.obs_date = m.obs_date
       AND o.version_num = m.min_ver
    INNER JOIN locations l ON o.location_id = l.location_id;
    """,
    """
    CREATE VIEW IF NOT EXISTS v_forecast_scores AS
    SELECT
        f.forecast_id,
        f.location_id,
        f.model_id,
        f.model_kind,
        f.registry_alias,
        f.registry_model_version,
        f.issued_at_utc,
        f.issued_run_date,
        f.data_as_of,
        f.eval_split_regime,
        f.horizon_day,
        f.target_date,
        f.temporal_status_at_issue,
        f.is_degraded_input,
        f.ceres_policy_applied,
        f.pred_mm,
        fst.version_num                           AS obs_first_version_num,
        fst.ingest_origin                         AS obs_first_ingest_origin,
        fst.ingested_at_utc                       AS obs_first_ingested_at_utc,
        fst.age_days_at_ingest                    AS obs_first_age_days_at_ingest,
        fst.completeness_regime                   AS obs_first_completeness,
        fst.PRECTOTCORR                           AS obs_first_mm,
        CASE
            WHEN fst.ingested_at_utc IS NULL THEN NULL
            WHEN julianday(fst.ingested_at_utc) > julianday(f.issued_at_utc) THEN 1
            ELSE 0
        END                                       AS is_prospective,
        ABS(f.pred_mm - fst.PRECTOTCORR)          AS abs_error_vs_first_mm,
        (f.pred_mm - fst.PRECTOTCORR)             AS signed_error_vs_first_mm,
        lat.version_num                           AS obs_latest_version_num,
        lat.ingest_origin                         AS obs_latest_ingest_origin,
        lat.ingested_at_utc                       AS obs_latest_ingested_at_utc,
        lat.age_days_at_ingest                    AS obs_latest_age_days_at_ingest,
        lat.completeness_regime                   AS obs_latest_completeness,
        lat.is_provisional                        AS obs_latest_is_provisional,
        lat.PRECTOTCORR                           AS obs_latest_mm,
        ABS(f.pred_mm - lat.PRECTOTCORR)          AS abs_error_vs_latest_mm,
        (f.pred_mm - lat.PRECTOTCORR)             AS signed_error_vs_latest_mm,
        (lat.PRECTOTCORR - fst.PRECTOTCORR)       AS obs_revision_delta_mm
    FROM forecasts_ledger f
    LEFT JOIN (
        SELECT o.*
        FROM observations_vintage o
        INNER JOIN (
            SELECT location_id, obs_date, MIN(version_num) AS first_precip_ver
            FROM observations_vintage
            WHERE PRECTOTCORR IS NOT NULL
            GROUP BY location_id, obs_date
        ) fp ON o.location_id = fp.location_id
            AND o.obs_date = fp.obs_date
            AND o.version_num = fp.first_precip_ver
    ) fst ON f.location_id = fst.location_id AND f.target_date = fst.obs_date
    LEFT JOIN v_observations_latest lat
           ON f.location_id = lat.location_id AND f.target_date = lat.obs_date;
    """,
)


def normalize_utc_iso(ts: Optional[str] = None) -> str:
    """
    Parse any ISO-8601 timestamp string (space or 'T', 'Z', '+00:00', '+07:00', with/without microseconds)
    or `None` (current UTC time) and return canonical `YYYY-MM-DDTHH:MM:SS.ffffffZ`.
    """
    if ts is None:
        return now_utc_iso()

    raw = str(ts).strip()
    if not raw:
        raise ValueError("Timestamp string cannot be empty.")
    dt = pd.to_datetime(raw, utc=True, errors="raise")
    return dt.tz_convert(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utc_now_iso() -> str:
    return normalize_utc_iso(None)


def resolve_git_sha() -> str:
    env_sha = os.environ.get("GIT_SHA", "").strip()
    if env_sha:
        return env_sha
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def connect_sqlite(
    db_path: str | Path,
    *,
    read_only: bool = False,
    recursive_triggers: bool = True,
) -> sqlite3.Connection:
    """
    Open SQLite connection with required V3.2 pragmas.

    When `read_only=True`, opens strictly via SQLite URI `file:{path}?mode=ro`,
    enforces `PRAGMA foreign_keys = ON; PRAGMA query_only = ON;`, and NEVER falls
    back to `immutable=1` (preventing silent reads of stale uncheckpointed snapshots).
    """
    if read_only:
        if str(db_path) == ":memory:":
            raise ValueError("Cannot open ':memory:' database with read_only=True.")
        resolved = Path(db_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"SQLite database not found for read_only mode: {resolved}")

        ro_uri = f"{resolved.as_uri()}?mode=ro"
        conn = sqlite3.connect(ro_uri, uri=True, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA query_only = ON;")
            conn.execute("SELECT name FROM sqlite_master LIMIT 1;").fetchall()
            return conn
        except Exception:
            conn.close()
            raise

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(f"PRAGMA recursive_triggers = {'ON' if recursive_triggers else 'OFF'};")
    if str(db_path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def finalize_db_for_read_only(conn: sqlite3.Connection) -> None:
    """Checkpoint WAL frames cleanly and switch to DELETE mode so single-file ?mode=ro reads see 100% of commits."""
    try:
        db_list = conn.execute("PRAGMA database_list;").fetchall()
        main_file = next((row[2] for row in db_list if row[1] == "main"), "")
        if not main_file:
            return
        if conn.execute("PRAGMA query_only;").fetchone()[0]:
            return
    except sqlite3.Error:
        return
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.execute("PRAGMA journal_mode = DELETE;")


def init_schema(conn: sqlite3.Connection, *, seed_hcmc_location: bool = True) -> None:
    """Apply DDL V3.2 tables, indexes, 9 triggers, and 3 views cleanly, then finalize WAL."""
    for stmt in DDL_V3_STATEMENTS:
        conn.execute(stmt)
    if seed_hcmc_location:
        conn.execute(
            """
            INSERT OR IGNORE INTO locations (
                location_id, display_name, aliases_json, latitude, longitude,
                provisional_window_days, is_active, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                HCMC_LOCATION_ID,
                HCMC_DISPLAY_NAME,
                json.dumps(list(HCMC_ALIASES), ensure_ascii=False),
                HCMC_LATITUDE,
                HCMC_LONGITUDE,
                DEFAULT_PROVISIONAL_WINDOW_DAYS,
                utc_now_iso(),
            ),
        )
    finalize_db_for_read_only(conn)


def _normalize_float_token(val: Any) -> Optional[float]:
    if val is None or pd.isna(val):
        return None
    f = float(val)
    if not np.isfinite(f) or f <= -900.0:
        return None
    return 0.0 if f == 0.0 else f


def compute_row_value_hash(values: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 across the 35 observation columns."""
    parts: List[str] = []
    for var in ALL_35_VARS:
        norm = _normalize_float_token(values.get(var))
        parts.append(f"{var}=NULL" if norm is None else f"{var}={repr(norm)}")
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classify_completeness(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Classify missingness counts and deterministic completeness_regime precedence."""
    missing_meteo: List[str] = []
    missing_flash: List[str] = []
    missing_ceres: List[str] = []

    for col in METEO_24_VARS:
        if _normalize_float_token(values.get(col)) is None:
            missing_meteo.append(col)
    for col in FLASHFLUX_3_VARS:
        if _normalize_float_token(values.get(col)) is None:
            missing_flash.append(col)
    for col in CERES_ONLY_8_VARS:
        if _normalize_float_token(values.get(col)) is None:
            missing_ceres.append(col)

    m_meteo = len(missing_meteo)
    m_flash = len(missing_flash)
    m_ceres = len(missing_ceres)
    total = m_meteo + m_flash + m_ceres

    if m_meteo > 0:
        regime = "INCOMPLETE_METEO"
    elif m_flash > 0:
        regime = "FLASHFLUX_INCOMPLETE"
    elif m_ceres > 0:
        regime = "MISSING_CERES_ONLY"
    else:
        regime = "COMPLETE_ALL_35"

    return {
        "missing_count": total,
        "missing_meteo_count": m_meteo,
        "missing_flashflux_count": m_flash,
        "missing_ceres_count": m_ceres,
        "completeness_regime": regime,
        "missing_vars_json": json.dumps(missing_meteo + missing_flash + missing_ceres),
    }


def compute_input_window_hash(rows_30d: Sequence[Mapping[str, Any]]) -> str:
    """Compute SHA-256 over sorted (obs_date, version_num, row_value_hash) tuples."""
    ordered = sorted(rows_30d, key=lambda r: str(r["obs_date"]))
    canonical_lines = [f"{r['obs_date']}:{int(r['version_num'])}:{r['row_value_hash']}" for r in ordered]
    return hashlib.sha256("\n".join(canonical_lines).encode("utf-8")).hexdigest()


def _normalize_input_df_to_english(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {c: VI_TO_EN_COLUMNS[c] for c in out.columns if c in VI_TO_EN_COLUMNS}
    if rename_map:
        out = out.rename(columns=rename_map)
    if "DATE" not in out.columns:
        raise ValueError("Input DataFrame must contain 'DATE' or 'Ngày' column.")
    out["DATE"] = pd.to_datetime(out["DATE"]).dt.strftime("%Y-%m-%d")
    missing_cols = [c for c in ALL_35_VARS if c not in out.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing canonical variables: {missing_cols}")
    return out.sort_values("DATE").reset_index(drop=True)


def seed_historical_csv_vintage_v0(
    conn: sqlite3.Connection,
    *,
    df: Optional[pd.DataFrame] = None,
    location_id: str = HCMC_LOCATION_ID,
    ingested_at_utc: Optional[str] = None,
    git_sha: Optional[str] = None,
    fail_after_rows: Optional[int] = None,
) -> str:
    """
    Seed historical CSV observations as version_num = 0 inside a single atomic transaction.
    On any error, rolls back all observations and writes a single FAILED `ingestion_runs` audit row.
    """
    if df is None:
        from src.data.loader import DataLoader

        loaded = DataLoader().load_data(rename_columns=False)
        if loaded is None:
            raise RuntimeError("DataLoader().load_data() returned None.")
        df = loaded

    norm_df = _normalize_input_df_to_english(df)
    if norm_df.empty:
        raise ValueError("Cannot seed empty historical DataFrame.")

    run_id = f"seed_v0_{uuid.uuid4().hex[:16]}"
    started_at = utc_now_iso()
    ingest_ts = normalize_utc_iso(ingested_at_utc) if ingested_at_utc is not None else started_at
    ingest_dt = pd.to_datetime(ingest_ts[:10])
    sha = git_sha or resolve_git_sha()
    win_start = str(norm_df["DATE"].iloc[0])
    win_end = str(norm_df["DATE"].iloc[-1])
    total_rows = int(len(norm_df))

    obs_col_list = [
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
        *ALL_35_VARS,
    ]
    placeholders = ", ".join(["?"] * len(obs_col_list))
    insert_obs_sql = f"INSERT INTO observations_vintage ({', '.join(obs_col_list)}) VALUES ({placeholders})"

    records: List[Tuple[Any, ...]] = []
    for row in norm_df.to_dict(orient="records"):
        obs_date_str = str(row["DATE"])
        age_days = max(0, int((ingest_dt - pd.to_datetime(obs_date_str)).days))
        comp = classify_completeness(row)
        r_hash = compute_row_value_hash(row)
        vals = [_normalize_float_token(row.get(v)) for v in ALL_35_VARS]
        records.append(
            (
                location_id,
                obs_date_str,
                0,
                ingest_ts,
                run_id,
                age_days,
                "HISTORICAL_CSV_SEED",
                comp["missing_count"],
                comp["missing_meteo_count"],
                comp["missing_flashflux_count"],
                comp["missing_ceres_count"],
                comp["completeness_regime"],
                comp["missing_vars_json"],
                "INITIAL_SEED",
                0,
                "[]",
                0,
                "{}",
                0,
                "{}",
                r_hash,
                *vals,
            )
        )

    finished_at = utc_now_iso()
    try:
        conn.execute("BEGIN IMMEDIATE;")
        conn.execute(
            """
            INSERT INTO ingestion_runs (
                run_id, run_type, started_at_utc, finished_at_utc, status,
                location_id, scan_window_start, scan_window_end,
                rows_scanned, rows_inserted_v0, rows_new_vintage,
                rows_late_fill_only, rows_value_change, rows_unchanged, git_sha
            ) VALUES (?, 'historical_seed', ?, ?, 'SUCCESS', ?, ?, ?, ?, ?, 0, 0, 0, 0, ?)
            """,
            (run_id, started_at, finished_at, location_id, win_start, win_end, total_rows, total_rows, sha),
        )
        if fail_after_rows is not None and 0 <= fail_after_rows <= len(records):
            if fail_after_rows > 0:
                conn.executemany(insert_obs_sql, records[:fail_after_rows])
            raise RuntimeError(f"Simulated mid-batch failure after {fail_after_rows} rows.")
        conn.executemany(insert_obs_sql, records)
        conn.execute("COMMIT;")
        finalize_db_for_read_only(conn)
        return run_id
    except Exception as exc:
        try:
            conn.execute("ROLLBACK;")
        except sqlite3.Error:
            pass
        # Separate audit transaction recording clean FAILED state (0 v0 / new rows)
        conn.execute("BEGIN IMMEDIATE;")
        conn.execute(
            """
            INSERT INTO ingestion_runs (
                run_id, run_type, started_at_utc, finished_at_utc, status,
                location_id, scan_window_start, scan_window_end,
                rows_scanned, rows_inserted_v0, rows_new_vintage,
                rows_late_fill_only, rows_value_change, rows_unchanged,
                rows_unpublished_skipped, git_sha, error_message
            ) VALUES (?, 'historical_seed', ?, ?, 'FAILED', ?, ?, ?, ?, 0, 0, 0, 0, 0, 0, ?, ?)
            """,
            (run_id, started_at, utc_now_iso(), location_id, win_start, win_end, total_rows, sha, str(exc)),
        )
        conn.execute("COMMIT;")
        finalize_db_for_read_only(conn)
        raise


def read_latest_observations(
    conn: sqlite3.Connection,
    *,
    location_id: str = HCMC_LOCATION_ID,
    as_vietnamese: bool = True,
    target_column_order: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Read `v_observations_latest` as a clean pandas DataFrame matching DataLoader format."""
    rows = conn.execute(
        f"""
        SELECT obs_date, {', '.join(ALL_35_VARS)}
        FROM v_observations_latest
        WHERE location_id = ?
        ORDER BY obs_date ASC
        """,
        (location_id,),
    ).fetchall()

    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df

    df = df.rename(columns={"obs_date": "DATE"})
    df["DATE"] = pd.to_datetime(df["DATE"])

    if as_vietnamese:
        en_to_vi = {k: v for k, v in Config.COLUMN_MAPPING.items() if k == "DATE" or k in ALL_35_VARS}
        df = df.rename(columns=en_to_vi)
        if target_column_order is None:
            df = align_and_validate_canonical_raw_df(df)
        else:
            df = df[list(target_column_order)]
    else:
        default_order = list(CANONICAL_EN_COLUMN_ORDER)
        if target_column_order is not None:
            df = df[list(target_column_order)]
        else:
            df = df[default_order]
    return df.reset_index(drop=True)


def get_default_complete_anchor_date(
    conn: sqlite3.Connection,
    location_id: str = HCMC_LOCATION_ID,
    max_date: Optional[str] = None,
) -> Optional[str]:
    """
    Return the latest `obs_date` in `v_observations_latest` with `completeness_regime = 'COMPLETE_ALL_35'`
    (optionally `<= max_date`) whose trailing 30-day window `[anchor - 29d .. anchor]` is continuous and
    contains no missing meteorological/precipitation observations.
    """
    where_clauses = ["location_id = ?", "completeness_regime = 'COMPLETE_ALL_35'"]
    params: List[Any] = [location_id]
    if max_date is not None:
        max_date_str = pd.to_datetime(max_date).strftime("%Y-%m-%d")
        where_clauses.append("obs_date <= ?")
        params.append(max_date_str)

    candidates = conn.execute(
        f"""
        SELECT obs_date
        FROM v_observations_latest
        WHERE {' AND '.join(where_clauses)}
        ORDER BY obs_date DESC
        """,
        tuple(params),
    ).fetchall()

    for cand in candidates:
        anchor_str = str(cand["obs_date"])
        anchor_dt = pd.to_datetime(anchor_str)
        win_start = (anchor_dt - timedelta(days=29)).strftime("%Y-%m-%d")
        stats = conn.execute(
            """
            SELECT
                COUNT(*) AS cnt,
                SUM(CASE WHEN missing_count = 0 THEN 1 ELSE 0 END) AS complete_cnt
            FROM v_observations_latest
            WHERE location_id = ?
              AND obs_date BETWEEN ? AND ?
            """,
            (location_id, win_start, anchor_str),
        ).fetchone()
        if stats is not None and int(stats["cnt"] or 0) == 30 and int(stats["complete_cnt"] or 0) == 30:
            return anchor_str
    return None


def get_complete_anchor_range(
    conn: sqlite3.Connection,
    *,
    location_id: str = HCMC_LOCATION_ID,
    window_days: int = 35,
) -> Optional[Dict[str, str]]:
    """
    Return `{"min": earliest_anchor, "max": latest_anchor}` where each anchor satisfies
    `COMPLETE_ALL_35` and has `window_days` continuous preceding days with `missing_count = 0`.
    """
    rows = conn.execute(
        """
        SELECT obs_date, missing_count, completeness_regime
        FROM v_observations_latest
        WHERE location_id = ?
        ORDER BY obs_date ASC
        """,
        (location_id,),
    ).fetchall()
    if not rows:
        return None

    streak = 0
    prev_dt: Optional[datetime] = None
    min_anchor: Optional[str] = None
    max_anchor: Optional[str] = None

    for row in rows:
        date_str = str(row["obs_date"])
        curr_dt = datetime.strptime(date_str, "%Y-%m-%d")
        is_complete = (
            str(row["completeness_regime"]) == "COMPLETE_ALL_35" and int(row["missing_count"] or 0) == 0
        )
        if not is_complete:
            streak = 0
            prev_dt = curr_dt
            continue

        if prev_dt is not None and (curr_dt - prev_dt).days == 1:
            streak += 1
        else:
            streak = 1
        prev_dt = curr_dt

        if streak >= window_days:
            if min_anchor is None:
                min_anchor = date_str
            max_anchor = date_str

    if min_anchor is None or max_anchor is None:
        return None
    return {"min": min_anchor, "max": max_anchor}


def get_data_snapshot_date(
    conn: sqlite3.Connection,
    *,
    location_id: str = HCMC_LOCATION_ID,
) -> Optional[str]:
    """Return latest `obs_date` recorded in `v_observations_latest` for `location_id`."""
    row = conn.execute(
        "SELECT MAX(obs_date) AS max_obs_date FROM v_observations_latest WHERE location_id = ?",
        (location_id,),
    ).fetchone()
    if row is None or row["max_obs_date"] is None:
        return None
    return str(row["max_obs_date"])


def get_latest_ingest_latency(
    conn: sqlite3.Connection,
    *,
    location_id: str = HCMC_LOCATION_ID,
) -> Optional[Dict[str, Any]]:
    """
    Return latency metadata from the latest successful `ingestion_runs` entry
    containing `latest_valid_by_group_json` (falling back to latest `SUCCESS` run).
    """
    row = conn.execute(
        """
        SELECT run_id, finished_at_utc, scan_window_end, latest_valid_by_group_json
        FROM ingestion_runs
        WHERE location_id = ?
          AND status = 'SUCCESS'
          AND latest_valid_by_group_json IS NOT NULL
        ORDER BY julianday(finished_at_utc) DESC, rowid DESC
        LIMIT 1
        """,
        (location_id,),
    ).fetchone()
    if row is None:
        return None

    raw_groups = json.loads(str(row["latest_valid_by_group_json"]))
    groups: Dict[str, Dict[str, Any]] = {}
    for g_name in ("METEO_GEOSIT", "RAD_FLASHFLUX", "RAD_CERES_SYN1DEG"):
        g_info = raw_groups.get(g_name)
        if not isinstance(g_info, dict):
            continue
        groups[g_name] = {
            "last_valid_date": g_info.get("last_valid_date"),
            "lag_days_vs_scan_end": g_info.get("lag_days_vs_scan_end"),
            "valid_days_in_window": g_info.get("valid_days_in_window"),
        }

    return {
        "last_ingest_run_id": str(row["run_id"]),
        "last_ingest_at_utc": str(row["finished_at_utc"]),
        "scan_window_end": str(row["scan_window_end"]),
        "groups": groups,
    }


def derive_eval_split_regime(data_as_of: str) -> str:
    """Deterministically assign split regime from canonical project calendar boundaries."""
    date_str = pd.to_datetime(data_as_of).strftime("%Y-%m-%d")
    if date_str <= "2018-04-30":
        return "DEV_TRAIN"
    if date_str <= "2020-04-30":
        return "VALIDATION"
    if date_str <= "2026-04-30":
        return "TEST"
    return "POST_TEST"


def derive_temporal_status_at_issue(
    target_date: str,
    issued_run_date: str,
    target_has_precip_at_issue: bool = False,
) -> str:
    """Deterministically assign temporal status at forecast issue time."""
    tgt = pd.to_datetime(target_date).strftime("%Y-%m-%d")
    run_d = pd.to_datetime(issued_run_date).strftime("%Y-%m-%d")
    if tgt < run_d:
        return "past_observed" if target_has_precip_at_issue else "past_unverified"
    if tgt == run_d:
        return "today"
    return "future_forecast"


def insert_forecasts_idempotent(
    conn: sqlite3.Connection,
    forecast_rows: Sequence[Mapping[str, Any]],
) -> Tuple[int, int]:
    """
    Insert forecasts into `forecasts_ledger` idempotently.

    - Always derives `eval_split_regime = derive_eval_split_regime(row['data_as_of'])`.
    - Pre-checks `(location_id, model_id, data_as_of, horizon_day)` and `forecast_id`;
      existing duplicates are skipped and logged via `logger.info` so one existing key
      never aborts a batch insert.
    - Returns `(inserted_count, skipped_count)`.
    """
    if not forecast_rows:
        return (0, 0)

    to_insert: List[Tuple[Any, ...]] = []
    skipped_count = 0
    seen_batch_keys: set[Tuple[str, str, str, int]] = set()
    seen_batch_ids: set[str] = set()

    for row in forecast_rows:
        location_id = str(row["location_id"])
        model_id = str(row["model_id"])
        data_as_of = pd.to_datetime(row["data_as_of"]).strftime("%Y-%m-%d")
        horizon_day = int(row["horizon_day"])
        forecast_id = str(
            row.get("forecast_id") or f"{location_id}_{model_id}_{data_as_of}_h{horizon_day}"
        )
        nat_key = (location_id, model_id, data_as_of, horizon_day)

        exists = conn.execute(
            """
            SELECT 1 FROM forecasts_ledger
            WHERE forecast_id = ?
               OR (location_id = ? AND model_id = ? AND data_as_of = ? AND horizon_day = ?)
            LIMIT 1
            """,
            (forecast_id, location_id, model_id, data_as_of, horizon_day),
        ).fetchone()

        if exists is not None or nat_key in seen_batch_keys or forecast_id in seen_batch_ids:
            skipped_count += 1
            logger.info(
                "Skipping existing forecast key location_id=%s model_id=%s data_as_of=%s horizon_day=%s forecast_id=%s",
                location_id, model_id, data_as_of, horizon_day, forecast_id,
            )
            continue

        seen_batch_keys.add(nat_key)
        seen_batch_ids.add(forecast_id)
        eval_regime = derive_eval_split_regime(data_as_of)
        issued_at_str = normalize_utc_iso(str(row["issued_at_utc"]))
        issued_run_str = pd.to_datetime(row["issued_run_date"]).strftime("%Y-%m-%d")
        target_date_str = pd.to_datetime(row["target_date"]).strftime("%Y-%m-%d")
        temp_status = row.get("temporal_status_at_issue")
        if temp_status is None:
            has_precip_row = conn.execute(
                """
                SELECT 1
                FROM observations_vintage
                WHERE location_id = ?
                  AND obs_date = ?
                  AND julianday(ingested_at_utc) <= julianday(?)
                  AND PRECTOTCORR IS NOT NULL
                LIMIT 1
                """,
                (location_id, target_date_str, issued_at_str),
            ).fetchone()
            temp_status = derive_temporal_status_at_issue(
                target_date_str,
                issued_run_str,
                target_has_precip_at_issue=(has_precip_row is not None),
            )
        else:
            temp_status = str(temp_status)

        to_insert.append(
            (
                forecast_id,
                issued_at_str,
                issued_run_str,
                str(row.get("run_timezone", "Asia/Ho_Chi_Minh")),
                location_id,
                model_id,
                str(row["model_kind"]),
                row.get("registry_alias"),
                row.get("registry_model_version"),
                str(row["artifact_sha256"]),
                row.get("feature_builder_sha256"),
                data_as_of,
                eval_regime,
                horizon_day,
                target_date_str,
                temp_status,
                float(row["pred_mm"]),
                pd.to_datetime(row["input_window_start"]).strftime("%Y-%m-%d"),
                pd.to_datetime(row["input_window_end"]).strftime("%Y-%m-%d"),
                str(row["input_window_hash"]),
                str(row["input_vintage_manifest_json"]),
                int(row["input_provisional_days"]),
                int(row.get("meteo_missing_count", 0)),
                int(row.get("flashflux_max_ffill_days", 0)),
                str(row["ceres_policy_applied"]),
                int(row.get("ceres_max_staleness_days", 0)),
                int(row.get("is_degraded_input", 0)),
                row.get("degraded_reason"),
                str(row["git_sha"]),
            )
        )

    if to_insert:
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.executemany(
                """
                INSERT INTO forecasts_ledger (
                    forecast_id, issued_at_utc, issued_run_date, run_timezone, location_id,
                    model_id, model_kind, registry_alias, registry_model_version,
                    artifact_sha256, feature_builder_sha256, data_as_of, eval_split_regime,
                    horizon_day, target_date, temporal_status_at_issue, pred_mm,
                    input_window_start, input_window_end, input_window_hash,
                    input_vintage_manifest_json, input_provisional_days,
                    meteo_missing_count, flashflux_max_ffill_days, ceres_policy_applied,
                    ceres_max_staleness_days, is_degraded_input, degraded_reason, git_sha
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                to_insert,
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finalize_db_for_read_only(conn)

    return (len(to_insert), skipped_count)


def compute_physical_quality_flags(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Check non-null observation values against PHYSICAL_METEO_BOUNDS without blocking INSERT."""
    from src.inference.validation import PHYSICAL_METEO_BOUNDS

    flags: Dict[str, Dict[str, Any]] = {}
    for en_var in ALL_35_VARS:
        val = _normalize_float_token(values.get(en_var))
        if val is None:
            continue
        vi_name = Config.COLUMN_MAPPING.get(en_var)
        bounds = PHYSICAL_METEO_BOUNDS.get(vi_name) if vi_name else None
        if bounds is not None:
            lo, hi = bounds
            if val < lo or val > hi:
                flags[en_var] = {"value": val, "bounds": [lo, hi]}
    return {"anomaly_count": len(flags), "quality_flags_json": json.dumps(flags)}


def fetch_nasa_power_window_json(
    *,
    start_date: str,
    end_date: str,
    latitude: float = HCMC_LATITUDE,
    longitude: float = HCMC_LONGITUDE,
    timeout_sec: int = Config.NASA_API_TIMEOUT,
) -> Dict[str, Any]:
    """Fetch all 35 canonical daily NASA POWER variables across <=20-param batches as unified JSON."""
    start_ymd = pd.to_datetime(start_date).strftime("%Y%m%d")
    end_ymd = pd.to_datetime(end_date).strftime("%Y%m%d")
    max_per_batch = Config.NASA_API_MAX_PARAMS
    batches = [ALL_35_VARS[i:i + max_per_batch] for i in range(0, len(ALL_35_VARS), max_per_batch)]

    merged_params: Dict[str, Dict[str, Any]] = {}
    merged_header: Dict[str, Any] = {}

    for batch in batches:
        params = {
            "parameters": ",".join(batch),
            "community": "RE",
            "longitude": longitude,
            "latitude": latitude,
            "start": start_ymd,
            "end": end_ymd,
            "time-standard": "LST",
            "format": "JSON",
        }
        resp = requests.get(Config.NASA_API_BASE_URL, params=params, timeout=timeout_sec)
        resp.raise_for_status()
        payload = resp.json()
        api_messages = payload.get("messages")
        if api_messages:
            raise RuntimeError(f"NASA POWER API error: {api_messages}")
        if not merged_header and isinstance(payload.get("header"), dict):
            merged_header = payload["header"]
        param_map = payload.get("properties", {}).get("parameter", {})
        if not isinstance(param_map, dict) or not param_map:
            raise RuntimeError("NASA POWER response missing properties.parameter dictionary.")
        for var, date_dict in param_map.items():
            merged_params[var] = date_dict

    return {
        "header": merged_header,
        "properties": {"parameter": merged_params},
    }


def _compute_latest_valid_by_group(
    parsed_rows: Sequence[Tuple[str, Dict[str, Any]]],
    scan_end_str: str,
) -> Dict[str, Any]:
    end_dt = pd.to_datetime(scan_end_str)
    groups = {
        "METEO_GEOSIT": METEO_24_VARS,
        "RAD_FLASHFLUX": FLASHFLUX_3_VARS,
        "RAD_CERES_SYN1DEG": CERES_ONLY_8_VARS,
    }
    summary: Dict[str, Any] = {}
    for group_name, cols in groups.items():
        valid_dates: List[str] = []
        incomplete_dates: List[str] = []
        for obs_date, row_vals in parsed_rows:
            if all(_normalize_float_token(row_vals.get(c)) is not None for c in cols):
                valid_dates.append(obs_date)
            else:
                incomplete_dates.append(obs_date)

        per_var_last_valid: Dict[str, Optional[str]] = {}
        per_var_valid_days: Dict[str, int] = {}
        for var in cols:
            var_dates = [
                obs_date
                for obs_date, row_vals in parsed_rows
                if _normalize_float_token(row_vals.get(var)) is not None
            ]
            per_var_last_valid[var] = max(var_dates) if var_dates else None
            per_var_valid_days[var] = len(var_dates)

        if valid_dates:
            latest_dt = max(valid_dates)
            lag_days = max(0, int((end_dt - pd.to_datetime(latest_dt)).days))
            summary[group_name] = {
                "last_valid_date": latest_dt,
                "lag_days_vs_scan_end": lag_days,
                "valid_days_in_window": len(valid_dates),
                "incomplete_dates_in_window": incomplete_dates,
                "per_variable_last_valid_date": per_var_last_valid,
                "per_variable_valid_days_in_window": per_var_valid_days,
            }
        else:
            summary[group_name] = {
                "last_valid_date": None,
                "lag_days_vs_scan_end": None,
                "valid_days_in_window": 0,
                "incomplete_dates_in_window": incomplete_dates,
                "per_variable_last_valid_date": per_var_last_valid,
                "per_variable_valid_days_in_window": per_var_valid_days,
            }
    return summary


def ingest_nasa_power_window(
    conn: sqlite3.Connection,
    *,
    lookback_days: int = 180,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    location_id: str = HCMC_LOCATION_ID,
    run_type: str = "manual_scan",
    fetcher: Optional[Callable[..., Dict[str, Any]]] = None,
    ingested_at_utc: Optional[str] = None,
    git_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pull a daily window from NASA POWER API and append only new observations or changed vintages.

    Executes atomically in one `BEGIN IMMEDIATE` transaction; on failure, rolls back all
    observation inserts and records one `FAILED` `ingestion_runs` row in a separate transaction.
    """
    loc_row = conn.execute(
        "SELECT location_id, latitude, longitude FROM locations WHERE location_id = ?",
        (location_id,),
    ).fetchone()
    if loc_row is None:
        raise ValueError(f"Unknown location_id='{location_id}' in locations table.")

    started_at = utc_now_iso()
    ingest_ts = normalize_utc_iso(ingested_at_utc) if ingested_at_utc is not None else started_at
    ingest_dt = pd.to_datetime(ingest_ts[:10])
    sha = git_sha or resolve_git_sha()
    run_id = f"nasa_{run_type}_{uuid.uuid4().hex[:14]}"

    if end_date is not None:
        end_dt = pd.to_datetime(end_date).normalize()
    else:
        end_dt = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").tz_localize(None).normalize()

    if start_date is not None:
        start_dt = pd.to_datetime(start_date).normalize()
    else:
        start_dt = end_dt - timedelta(days=max(1, int(lookback_days)) - 1)

    win_start = start_dt.strftime("%Y-%m-%d")
    win_end = end_dt.strftime("%Y-%m-%d")
    active_fetcher = fetcher or fetch_nasa_power_window_json

    try:
        payload = active_fetcher(
            start_date=win_start,
            end_date=win_end,
            latitude=float(loc_row["latitude"]),
            longitude=float(loc_row["longitude"]),
        )
        header = payload.get("header") if isinstance(payload.get("header"), dict) else {}
        api_version = (
            header.get("api", {}).get("version")
            if isinstance(header.get("api"), dict)
            else None
        )
        sources_list = header.get("sources") if isinstance(header.get("sources"), list) else []
        sources_json = json.dumps(sources_list)

        param_map = payload.get("properties", {}).get("parameter", {})
        if not isinstance(param_map, dict) or not param_map:
            raise ValueError("NASA POWER payload missing properties.parameter data.")

        all_raw_dates: set[str] = set()
        for var in ALL_35_VARS:
            var_dict = param_map.get(var, {})
            if isinstance(var_dict, dict):
                all_raw_dates.update(str(k) for k in var_dict.keys())

        parsed_rows: List[Tuple[str, Dict[str, Any]]] = []
        for raw_d in sorted(all_raw_dates):
            dt_parsed = pd.to_datetime(raw_d, format="%Y%m%d", errors="coerce")
            if pd.isna(dt_parsed):
                dt_parsed = pd.to_datetime(raw_d, errors="coerce")
            if pd.isna(dt_parsed):
                continue
            obs_date_str = dt_parsed.strftime("%Y-%m-%d")
            if obs_date_str < win_start or obs_date_str > win_end:
                continue
            row_vals: Dict[str, Any] = {}
            for var in ALL_35_VARS:
                raw_v = param_map.get(var, {}).get(raw_d) if isinstance(param_map.get(var), dict) else None
                row_vals[var] = _normalize_float_token(raw_v)
            parsed_rows.append((obs_date_str, row_vals))

        parsed_rows.sort(key=lambda item: item[0])
        if not parsed_rows:
            raise ValueError(f"No valid observation dates returned by NASA POWER in [{win_start}, {win_end}].")

        # Load latest existing vintage for every date in scanned window
        existing_latest: Dict[str, Dict[str, Any]] = {}
        db_latest_rows = conn.execute(
            """
            SELECT o.*
            FROM observations_vintage o
            INNER JOIN (
                SELECT obs_date, MAX(version_num) AS max_ver
                FROM observations_vintage
                WHERE location_id = ? AND obs_date BETWEEN ? AND ?
                GROUP BY obs_date
            ) m ON o.obs_date = m.obs_date AND o.version_num = m.max_ver
            WHERE o.location_id = ?
            """,
            (location_id, win_start, win_end, location_id),
        ).fetchall()
        for er in db_latest_rows:
            existing_latest[str(er["obs_date"])] = dict(er)

        rows_scanned = len(parsed_rows)
        rows_inserted_v0 = 0
        rows_new_vintage = 0
        rows_late_fill_only = 0
        rows_value_change = 0
        rows_unchanged = 0
        rows_unpublished_skipped = 0
        unpublished_dates_skipped: List[str] = []
        revision_records: List[Dict[str, Any]] = []
        obs_inserts: List[Tuple[Any, ...]] = []

        for obs_date_str, row_vals in parsed_rows:
            age_days = max(0, int((ingest_dt - pd.to_datetime(obs_date_str)).days))
            comp = classify_completeness(row_vals)
            if comp["missing_count"] == len(ALL_35_VARS):
                rows_unpublished_skipped += 1
                unpublished_dates_skipped.append(obs_date_str)
                continue
            q_flags = compute_physical_quality_flags(row_vals)
            new_hash = compute_row_value_hash(row_vals)
            prev_row = existing_latest.get(obs_date_str)

            if prev_row is None:
                version_num = 0
                change_summary = "INITIAL_NRT"
                late_fill_vars: List[str] = []
                value_change_details: Dict[str, List[Any]] = {}
                rows_inserted_v0 += 1
            elif str(prev_row["row_value_hash"]) == new_hash:
                rows_unchanged += 1
                continue
            else:
                prev_ver = int(prev_row["version_num"])
                version_num = prev_ver + 1
                late_fill_vars = []
                value_change_details = {}
                for var in ALL_35_VARS:
                    old_v = _normalize_float_token(prev_row.get(var))
                    new_v = row_vals.get(var)
                    if old_v is None and new_v is not None:
                        late_fill_vars.append(var)
                    elif old_v != new_v:
                        value_change_details[var] = [old_v, new_v]

                if late_fill_vars and not value_change_details:
                    change_summary = "LATE_FILL_ONLY"
                    rows_late_fill_only += 1
                elif value_change_details and not late_fill_vars:
                    change_summary = "VALUE_CHANGE_ONLY"
                    rows_value_change += 1
                else:
                    change_summary = "MIXED_FILL_AND_CHANGE"
                    rows_value_change += 1

                rows_new_vintage += 1
                revision_records.append(
                    {
                        "obs_date": obs_date_str,
                        "prev_version": prev_ver,
                        "new_version": version_num,
                        "prev_ingest_origin": prev_row.get("ingest_origin"),
                        "change_type_summary": change_summary,
                        "late_fill_count": len(late_fill_vars),
                        "late_fill_vars": late_fill_vars,
                        "value_change_count": len(value_change_details),
                        "value_change_details": value_change_details,
                    }
                )

            obs_inserts.append(
                (
                    location_id,
                    obs_date_str,
                    version_num,
                    ingest_ts,
                    run_id,
                    age_days,
                    "NASA_POWER_API",
                    comp["missing_count"],
                    comp["missing_meteo_count"],
                    comp["missing_flashflux_count"],
                    comp["missing_ceres_count"],
                    comp["completeness_regime"],
                    comp["missing_vars_json"],
                    change_summary,
                    len(late_fill_vars),
                    json.dumps(late_fill_vars),
                    len(value_change_details),
                    json.dumps(value_change_details),
                    q_flags["anomaly_count"],
                    q_flags["quality_flags_json"],
                    new_hash,
                    *[row_vals.get(v) for v in ALL_35_VARS],
                )
            )

        latest_valid_by_group = _compute_latest_valid_by_group(parsed_rows, win_end)
        revision_stats = {
            "mixed_fill_and_change_count": sum(1 for r in revision_records if r["change_type_summary"] == "MIXED_FILL_AND_CHANGE"),
            "unpublished_dates_skipped": unpublished_dates_skipped,
            "revisions": revision_records,
        }
        latest_group_json = json.dumps(latest_valid_by_group)
        revision_stats_json = json.dumps(revision_stats)
        finished_at = utc_now_iso()

        obs_col_list = [
            "location_id", "obs_date", "version_num", "ingested_at_utc", "run_id",
            "age_days_at_ingest", "ingest_origin",
            "missing_count", "missing_meteo_count", "missing_flashflux_count", "missing_ceres_count",
            "completeness_regime", "missing_vars_json",
            "change_type_summary", "late_fill_count", "late_fill_vars_json",
            "value_change_count", "value_change_details_json",
            "anomaly_count", "quality_flags_json", "row_value_hash",
            *ALL_35_VARS,
        ]
        insert_obs_sql = f"INSERT INTO observations_vintage ({', '.join(obs_col_list)}) VALUES ({', '.join(['?'] * len(obs_col_list))})"

        conn.execute("BEGIN IMMEDIATE;")
        conn.execute(
            """
            INSERT INTO ingestion_runs (
                run_id, run_type, started_at_utc, finished_at_utc, status,
                location_id, scan_window_start, scan_window_end,
                nasa_api_version, nasa_sources_header_json,
                rows_scanned, rows_inserted_v0, rows_new_vintage,
                rows_late_fill_only, rows_value_change, rows_unchanged,
                rows_unpublished_skipped, latest_valid_by_group_json, revision_stats_json, git_sha
            ) VALUES (?, ?, ?, ?, 'SUCCESS', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, run_type, started_at, finished_at, location_id, win_start, win_end,
                api_version, sources_json,
                rows_scanned, rows_inserted_v0, rows_new_vintage,
                rows_late_fill_only, rows_value_change, rows_unchanged,
                rows_unpublished_skipped, latest_group_json, revision_stats_json, sha,
            ),
        )
        if obs_inserts:
            conn.executemany(insert_obs_sql, obs_inserts)
        conn.execute("COMMIT;")
        finalize_db_for_read_only(conn)

        return {
            "run_id": run_id,
            "run_type": run_type,
            "status": "SUCCESS",
            "location_id": location_id,
            "scan_window_start": win_start,
            "scan_window_end": win_end,
            "nasa_api_version": api_version,
            "nasa_sources_header": sources_list,
            "rows_scanned": rows_scanned,
            "rows_inserted_v0": rows_inserted_v0,
            "rows_new_vintage": rows_new_vintage,
            "rows_late_fill_only": rows_late_fill_only,
            "rows_value_change": rows_value_change,
            "rows_unchanged": rows_unchanged,
            "rows_unpublished_skipped": rows_unpublished_skipped,
            "latest_valid_by_group_json": latest_valid_by_group,
            "revision_stats_json": revision_stats,
        }
    except Exception as exc:
        try:
            conn.execute("ROLLBACK;")
        except sqlite3.Error:
            pass
        conn.execute("BEGIN IMMEDIATE;")
        conn.execute(
            """
            INSERT INTO ingestion_runs (
                run_id, run_type, started_at_utc, finished_at_utc, status,
                location_id, scan_window_start, scan_window_end,
                rows_scanned, rows_inserted_v0, rows_new_vintage,
                rows_late_fill_only, rows_value_change, rows_unchanged,
                rows_unpublished_skipped, git_sha, error_message
            ) VALUES (?, ?, ?, ?, 'FAILED', ?, ?, ?, 0, 0, 0, 0, 0, 0, 0, ?, ?)
            """,
            (run_id, run_type, started_at, utc_now_iso(), location_id, win_start, win_end, sha, str(exc)),
        )
        conn.execute("COMMIT;")
        finalize_db_for_read_only(conn)
        raise


def query_observations_as_of(
    conn: sqlite3.Connection,
    *,
    location_id: str,
    input_window_start: str,
    input_window_end: str,
    issued_at_utc: str,
) -> List[Dict[str, Any]]:
    """Reconstruct exact 30-day vintage snapshot known at `issued_at_utc`."""
    issued_norm = normalize_utc_iso(issued_at_utc)
    rows = conn.execute(
        """
        SELECT o.*
        FROM observations_vintage o
        INNER JOIN (
            SELECT location_id, obs_date, MAX(version_num) AS as_of_ver
            FROM observations_vintage
            WHERE location_id = ?
              AND obs_date BETWEEN ? AND ?
              AND julianday(ingested_at_utc) <= julianday(?)
            GROUP BY location_id, obs_date
        ) m ON o.location_id = m.location_id
           AND o.obs_date = m.obs_date
           AND o.version_num = m.as_of_ver
        ORDER BY o.obs_date ASC
        """,
        (location_id, input_window_start, input_window_end, issued_norm),
    ).fetchall()
    return [dict(r) for r in rows]
