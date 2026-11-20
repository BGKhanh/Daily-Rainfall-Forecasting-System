"""
Round 10 — Task 1 (Item C): Probe NASA POWER `header.sources` per single day
========================================================================
Tests whether querying NASA POWER Daily API (`v2.10.3`, `community="RE"`,
`time-standard="LST"`) with a 1-day window returns day-specific `header.sources`:
  - Series 1 (Meteo only): `PRECTOTCORR`
  - Series 2 (Radiation only): `ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN`

If `header.sources` shifts across dates, binary-searches between consecutive
10-day sparse samples to identify exact cutoff dates:
  - `meteo_merra2_last_date` & `meteo_geosit_first_date`
  - `rad_syn1deg_last_date` & `rad_flashflux_first_date`
Persists results to `reports/storage_audit/nasa_header_sources_experiment.json`
and optionally appends an `audit_probe` row into `ingestion_runs` (DDL V3.2)
preserving existing `METEO_GEOSIT` / `RAD_FLASHFLUX` / `RAD_CERES_SYN1DEG` keys.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import uuid

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import Config
from src.data.vintage_store import (
    DEFAULT_PROVISIONAL_WINDOW_DAYS,
    HCMC_LOCATION_ID,
    binary_search_source_transition as binary_search_transition,
    connect_sqlite,
    finalize_db_for_read_only,
    resolve_git_sha,
    utc_now_iso,
)

DAILY_ENDPOINT = Config.NASA_API_BASE_URL
HCMC_LAT, HCMC_LON = Config.province_coordinates["TP. Hồ Chí Minh"]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "storage_audit" / "nasa_header_sources_experiment.json"
DEFAULT_LIVE_DB = PROJECT_ROOT / "data" / "vintage" / "rainfall_vintage.db"
DEFAULT_DEMO_DB = PROJECT_ROOT / "artifacts" / "models" / "demo_rainfall_vintage.db"
METADATA_JSON_PATH = PROJECT_ROOT / "artifacts" / "models" / "metadata.json"


def _is_published_value(val: Any) -> bool:
    if val is None:
        return False
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return False
    return fval != -999.0 and fval == fval


def build_sparse_probe_dates(
    start_iso: str = "2026-04-25",
    end_iso: str = "2026-11-18",
    step_days: int = 10,
) -> List[str]:
    """Return 10-day sparse dates from `start_iso` to `end_iso` inclusive of both endpoints."""
    start_d = date.fromisoformat(start_iso)
    end_d = date.fromisoformat(end_iso)
    dates: List[str] = []
    cur = start_d
    while cur <= end_d:
        dates.append(cur.isoformat())
        cur += timedelta(days=step_days)
    if dates[-1] != end_iso:
        dates.append(end_iso)
    return dates


def fetch_nasa_single_day(
    session: requests.Session,
    obs_date: str,
    parameters: str,
    *,
    lat: float = HCMC_LAT,
    lon: float = HCMC_LON,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Query NASA POWER Daily API (`v2.10.3`) with a 1-day window (`start == end`)."""
    ymd = obs_date.replace("-", "")
    params = {
        "parameters": parameters,
        "community": "RE",
        "time-standard": "LST",
        "longitude": lon,
        "latitude": lat,
        "start": ymd,
        "end": ymd,
        "format": "JSON",
    }
    resp = session.get(DAILY_ENDPOINT, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    header = payload.get("header") if isinstance(payload.get("header"), dict) else {}
    api_info = header.get("api") if isinstance(header.get("api"), dict) else {}
    sources = list(header.get("sources") or [])
    param_map = ((payload.get("properties") or {}).get("parameter")) or {}

    values: Dict[str, Optional[float]] = {}
    raw_values: Dict[str, Any] = {}
    for p_name in parameters.split(","):
        p_dict = param_map.get(p_name) or {}
        raw_v = p_dict.get(ymd)
        raw_values[p_name] = raw_v
        values[p_name] = float(raw_v) if _is_published_value(raw_v) else None

    return {
        "date": obs_date,
        "api_version": api_info.get("version"),
        "sources": sources,
        "sources_normalized": sorted(str(s).upper() for s in sources),
        "values": values,
        "raw_values": raw_values,
        "messages": payload.get("messages") or [],
    }


def _classify_meteo_source(entry: Dict[str, Any]) -> str:
    val = entry.get("values", {}).get("PRECTOTCORR")
    if val is None:
        return "UNPUBLISHED"
    srcs = [str(s).upper() for s in (entry.get("sources") or [])]
    if "GEOSIT" in srcs or "GEOS-IT" in srcs:
        return "GEOSIT"
    if "MERRA2" in srcs or "MERRA-2" in srcs:
        return "MERRA2"
    return "+".join(sorted(srcs)) or "UNKNOWN"


def _classify_radiation_source(entry: Dict[str, Any]) -> str:
    vals = entry.get("values", {})
    allsky = vals.get("ALLSKY_SFC_SW_DWN")
    clrsky = vals.get("CLRSKY_SFC_SW_DWN")
    if allsky is None and clrsky is None:
        return "UNPUBLISHED"
    srcs = [str(s).upper() for s in (entry.get("sources") or [])]
    if "SYN1DEG" in srcs and "FLASHFLUX" not in srcs:
        return "SYN1DEG"
    if "FLASHFLUX" in srcs and "SYN1DEG" not in srcs:
        return "FLASHFLUX"
    if srcs:
        return "+".join(sorted(srcs))
    return "UNKNOWN"


def _classify_header_signature(entry: Dict[str, Any]) -> str:
    srcs = sorted(str(s).upper() for s in (entry.get("sources") or []))
    return "+".join(srcs) if srcs else "EMPTY"


def persist_cutoffs_to_db(
    db_path: Path,
    cutoffs_payload: Dict[str, Any],
    *,
    experiment_run_date: str,
) -> Dict[str, Any]:
    """
    Append an `audit_probe` row into `ingestion_runs` (DDL V3.2 append-only safe) carrying forward
    existing `METEO_GEOSIT` / `RAD_FLASHFLUX` / `RAD_CERES_SYN1DEG` latency keys plus
    `source_transition_cutoffs`, then finalize WAL cleanly.
    """
    conn = connect_sqlite(db_path, read_only=False)
    try:
        prev = conn.execute(
            """
            SELECT scan_window_start, scan_window_end, nasa_api_version, latest_valid_by_group_json
            FROM ingestion_runs
            WHERE location_id = ? AND status = 'SUCCESS' AND latest_valid_by_group_json IS NOT NULL
            ORDER BY julianday(finished_at_utc) DESC, rowid DESC
            LIMIT 1
            """,
            (HCMC_LOCATION_ID,),
        ).fetchone()
        if prev is not None and prev["latest_valid_by_group_json"]:
            merged_json = json.loads(str(prev["latest_valid_by_group_json"]))
            scan_start = str(prev["scan_window_start"])
            scan_end = str(prev["scan_window_end"])
            api_ver = prev["nasa_api_version"] or cutoffs_payload.get("api_version") or "v2.10.3"
        else:
            merged_json = {}
            scan_start = "2026-04-25"
            scan_end = experiment_run_date
            api_ver = cutoffs_payload.get("api_version") or "v2.10.3"

        merged_json["source_transition_cutoffs"] = cutoffs_payload
        now_utc = utc_now_iso()
        run_id = f"audit_probe_header_sources_{uuid.uuid4().hex[:12]}"
        try:
            from src.config.settings import compute_config_sha256

            cfg_sha = compute_config_sha256()
        except Exception:
            cfg_sha = "unknown"
        conn.execute(
            """
            INSERT INTO ingestion_runs (
                run_id, run_type, started_at_utc, finished_at_utc, status, location_id,
                scan_window_start, scan_window_end, nasa_api_version, nasa_sources_header_json,
                rows_scanned, rows_inserted_v0, rows_new_vintage, rows_late_fill_only,
                rows_value_change, rows_unchanged, rows_unpublished_skipped,
                latest_valid_by_group_json, revision_stats_json, git_sha, config_sha256, error_message
            ) VALUES (?, 'audit_probe', ?, ?, 'SUCCESS', ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0, 0, ?, ?, ?, ?, NULL)
            """,
            (
                run_id,
                now_utc,
                now_utc,
                HCMC_LOCATION_ID,
                scan_start,
                scan_end,
                api_ver,
                json.dumps(cutoffs_payload.get("observed_unique_header_sources", [])),
                json.dumps(merged_json, ensure_ascii=False),
                json.dumps({"experiment": "round10_task1_header_sources"}),
                resolve_git_sha(),
                cfg_sha,
            ),
        )
        conn.commit()
        finalize_db_for_read_only(conn)
        return {"db_path": str(db_path), "run_id": run_id}
    finally:
        conn.close()


def run_experiment(
    *,
    start_iso: str = "2026-04-25",
    end_iso: str = "2026-11-18",
    experiment_today: str = "2026-11-19",
    persist_dbs: bool = True,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Dict[str, Any]:
    sparse_dates = build_sparse_probe_dates(start_iso=start_iso, end_iso=end_iso, step_days=10)
    session = requests.Session()
    try:
        meteo_sparse = [fetch_nasa_single_day(session, d, "PRECTOTCORR") for d in sparse_dates]
        rad_sparse = [
            fetch_nasa_single_day(session, d, "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN")
            for d in sparse_dates
        ]

        # Inspect whether header.sources changes across published dates
        meteo_pub_sigs = {_classify_header_signature(r) for r in meteo_sparse if r["values"]["PRECTOTCORR"] is not None}
        rad_pub_sigs = {
            _classify_header_signature(r)
            for r in rad_sparse
            if r["values"]["ALLSKY_SFC_SW_DWN"] is not None or r["values"]["CLRSKY_SFC_SW_DWN"] is not None
        }

        header_changes_meteo = len(meteo_pub_sigs) > 1
        header_changes_rad = len(rad_pub_sigs) > 1

        meteo_transition, meteo_bsearch_probes = binary_search_transition(
            meteo_sparse,
            lambda d: fetch_nasa_single_day(session, d, "PRECTOTCORR"),
            classifier=_classify_header_signature if header_changes_meteo else _classify_meteo_source,
        )
        rad_transition, rad_bsearch_probes = binary_search_transition(
            rad_sparse,
            lambda d: fetch_nasa_single_day(session, d, "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN"),
            classifier=_classify_header_signature if header_changes_rad else _classify_radiation_source,
        )

        # Also check unpublished boundary for PRECTOTCORR and CERES CLRSKY
        meteo_pub_to_unpub, meteo_unpub_probes = binary_search_transition(
            meteo_sparse,
            lambda d: fetch_nasa_single_day(session, d, "PRECTOTCORR"),
            classifier=lambda r: "PUBLISHED" if r["values"]["PRECTOTCORR"] is not None else "UNPUBLISHED",
            from_label="PUBLISHED",
            to_label="UNPUBLISHED",
        )
        ceres_pub_to_unpub, ceres_unpub_probes = binary_search_transition(
            rad_sparse,
            lambda d: fetch_nasa_single_day(session, d, "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN"),
            classifier=lambda r: "SYN1DEG_PRESENT" if r["values"]["CLRSKY_SFC_SW_DWN"] is not None else "SYN1DEG_MISSING",
            from_label="SYN1DEG_PRESENT",
            to_label="SYN1DEG_MISSING",
        )
    finally:
        session.close()

    today_d = date.fromisoformat(experiment_today)
    meteo_merra2_last = meteo_transition["last_from_date"] if (header_changes_meteo and meteo_transition) else None
    meteo_geosit_first = meteo_transition["first_to_date"] if (header_changes_meteo and meteo_transition) else None
    rad_syn1deg_last = rad_transition["last_from_date"] if (header_changes_rad and rad_transition) else (
        ceres_pub_to_unpub["last_from_date"] if ceres_pub_to_unpub else None
    )
    rad_flashflux_first = rad_transition["first_to_date"] if (header_changes_rad and rad_transition) else (
        ceres_pub_to_unpub["first_to_date"] if ceres_pub_to_unpub else None
    )

    meteo_cutoff_lag = (today_d - date.fromisoformat(meteo_merra2_last)).days if meteo_merra2_last else None
    rad_cutoff_lag = (today_d - date.fromisoformat(rad_syn1deg_last)).days if rad_syn1deg_last else None

    if header_changes_meteo and meteo_merra2_last:
        finality_mode = "source_header_transition"
        finality_cutoff_date = meteo_merra2_last
        finality_rule = (
            f"FINALIZED when PRECTOTCORR IS NOT NULL and obs_date <= {meteo_merra2_last} "
            f"(verified MERRA2 -> GEOSIT header transition cutoff, lag={meteo_cutoff_lag}d vs {experiment_today}); "
            "otherwise PROVISIONAL (or NOT_PUBLISHED when PRECTOTCORR IS NULL)."
        )
    else:
        finality_mode = "provisional_window_fallback"
        finality_cutoff_date = (today_d - timedelta(days=DEFAULT_PROVISIONAL_WINDOW_DAYS)).isoformat()
        finality_rule = (
            f"header.sources did not vary across published PRECTOTCORR dates; fallback applies FINALIZED when "
            f"completeness_regime == 'COMPLETE_ALL_35' and age > {DEFAULT_PROVISIONAL_WINDOW_DAYS} days."
        )

    observed_unique_sources = sorted(
        {tuple(r["sources"]) for r in (meteo_sparse + rad_sparse)}
    )
    cutoffs_payload = {
        "experiment_date": experiment_today,
        "api_version": (meteo_sparse[0].get("api_version") if meteo_sparse else "v2.10.3") or "v2.10.3",
        "header_sources_varies_per_day_meteo": header_changes_meteo,
        "header_sources_varies_per_day_radiation": header_changes_rad,
        "meteo_merra2_last_date": meteo_merra2_last,
        "meteo_geosit_first_date": meteo_geosit_first,
        "meteo_merra2_lag_days_vs_experiment": meteo_cutoff_lag,
        "rad_syn1deg_last_date": rad_syn1deg_last,
        "rad_flashflux_first_date": rad_flashflux_first,
        "rad_syn1deg_lag_days_vs_experiment": rad_cutoff_lag,
        "meteo_last_published_date": meteo_pub_to_unpub["last_from_date"] if meteo_pub_to_unpub else None,
        "finality_mode": finality_mode,
        "finality_cutoff_date": finality_cutoff_date,
        "finality_rule": finality_rule,
        "observed_unique_header_sources": [list(s) for s in observed_unique_sources],
    }

    db_updates: List[Dict[str, Any]] = []
    if persist_dbs:
        for dbp in (DEFAULT_LIVE_DB, DEFAULT_DEMO_DB):
            if dbp.exists():
                db_updates.append(persist_cutoffs_to_db(dbp, cutoffs_payload, experiment_run_date=end_iso))
        # Update demo_rainfall_vintage.db SHA-256 in metadata.json
        if DEFAULT_DEMO_DB.exists() and METADATA_JSON_PATH.exists():
            sha = hashlib.sha256(DEFAULT_DEMO_DB.read_bytes()).hexdigest()
            meta = json.loads(METADATA_JSON_PATH.read_text(encoding="utf-8"))
            meta.setdefault("sha256_checksums", {})["demo_rainfall_vintage.db"] = sha
            METADATA_JSON_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            cutoffs_payload["demo_db_sha256"] = sha

    report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "location": {"location_id": HCMC_LOCATION_ID, "latitude": HCMC_LAT, "longitude": HCMC_LON},
        "sparse_probe_window": {"start": start_iso, "end": end_iso, "count": len(sparse_dates), "dates": sparse_dates},
        "summary": cutoffs_payload,
        "meteo_sparse_series": meteo_sparse,
        "radiation_sparse_series": rad_sparse,
        "binary_search": {
            "meteo_transition": meteo_transition,
            "meteo_extra_probes": meteo_bsearch_probes,
            "radiation_transition": rad_transition,
            "radiation_extra_probes": rad_bsearch_probes,
            "meteo_published_boundary": meteo_pub_to_unpub,
            "meteo_unpublished_extra_probes": meteo_unpub_probes,
            "ceres_syn1deg_boundary": ceres_pub_to_unpub,
            "ceres_extra_probes": ceres_unpub_probes,
        },
        "db_updates": db_updates,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe NASA POWER per-day header.sources and transition dates.")
    parser.add_argument("--no-persist-dbs", action="store_true", help="Do not write audit_probe rows to SQLite DBs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    res = run_experiment(persist_dbs=not args.no_persist_dbs, report_path=args.output)
    print(json.dumps(res["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
