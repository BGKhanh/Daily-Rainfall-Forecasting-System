"""
SQLite B-Tree Page & Column Storage Footprint Audit (DS108 Round 13-Closeout)
=============================================================================
Inspects `data/vintage/rainfall_vintage.db` (strictly read-only; simulated VACUUM on temp copy)
and `artifacts/models/demo_rainfall_vintage.db` using:
  1. SQLite `PRAGMA page_count`, `PRAGMA freelist_count`, `PRAGMA page_size`.
  2. Deterministic binary SQLite B-tree page walker (handles interior/leaf + overflow chains;
     does not require `SQLITE_ENABLE_DBSTAT_VTAB` in Windows `sqlite3.dll`).
  3. Per-column byte breakdown across `forecasts_ledger`.
  4. Optional `--vacuum-demo` execution (`VACUUM` + `finalize_db_for_read_only`), verifying 29/29
     column SHA-256 bit-identity against `immutable_ledger_migration_proof_v35.json` and updating
     `db_file_sha256` in `immutable_ledger_migration_proof_v35.json` & `artifacts/models/metadata.json`.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import struct
import sys
import tempfile
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlops.migrate_ledger_v35_audited import compute_forecasts_column_hashes  # noqa: E402
from src.config.clock import now_utc_iso  # noqa: E402
from src.data.vintage_store import (  # noqa: E402
    FORECASTS_LEDGER_29_ORIGINAL_COLS,
    connect_sqlite,
    finalize_db_for_read_only,
)

DEFAULT_PROD_DB = PROJECT_ROOT / "data" / "vintage" / "rainfall_vintage.db"
DEFAULT_DEMO_DB = PROJECT_ROOT / "artifacts" / "models" / "demo_rainfall_vintage.db"
DEFAULT_PROOF_JSON = PROJECT_ROOT / "reports" / "storage_audit" / "immutable_ledger_migration_proof_v35.json"
DEFAULT_METADATA_JSON = PROJECT_ROOT / "artifacts" / "models" / "metadata.json"
DEFAULT_AUDIT_OUT = PROJECT_ROOT / "reports" / "storage_audit" / "db_storage_footprint_audit_r13_closeout.json"


def _read_varint(buf: bytes, offset: int) -> Tuple[int, int]:
    """Decode SQLite big-endian 1..9 byte varint starting at `offset`."""
    buf_len = len(buf)
    val = 0
    for i in range(8):
        if offset + i >= buf_len:
            raise ValueError(f"Truncated SQLite varint at offset {offset} (buf_len={buf_len}).")
        b = buf[offset + i]
        val = (val << 7) | (b & 0x7F)
        if not (b & 0x80):
            return val, offset + i + 1
    if offset + 8 >= buf_len:
        raise ValueError(f"Truncated 9-byte SQLite varint at offset {offset} (buf_len={buf_len}).")
    b8 = buf[offset + 8]
    val = (val << 8) | b8
    return val, offset + 9


def _calc_overflow_first_page(
    page_bytes: bytes,
    cell_offset: int,
    page_type: int,
    usable_size: int,
) -> int:
    """Return first overflow page index (0 if payload fits completely on the B-tree page)."""
    if usable_size <= 35:
        raise ValueError(f"Invalid SQLite usable_size={usable_size} (must exceed 35 bytes).")
    if page_type == 0x05:  # table interior: only 4-byte left child + rowid varint
        return 0
    pos = cell_offset
    if page_type == 0x02:  # index interior: skip 4-byte left child
        pos += 4
    payload_len, pos = _read_varint(page_bytes, pos)
    if page_type == 0x0D:  # table leaf: skip rowid varint
        _, pos = _read_varint(page_bytes, pos)

    if page_type == 0x0D:
        max_local = usable_size - 35
        min_local = ((usable_size - 12) * 32 // 255) - 23
    else:  # 0x02 or 0x0A index
        max_local = ((usable_size - 12) * 64 // 255) - 23
        min_local = ((usable_size - 12) * 32 // 255) - 23

    if payload_len <= max_local:
        return 0

    surplus = min_local + ((payload_len - min_local) % (usable_size - 4))
    local_len = min_local if surplus > max_local else surplus
    overflow_ptr_offset = pos + local_len
    if overflow_ptr_offset + 4 > len(page_bytes):
        return 0
    return struct.unpack(">I", page_bytes[overflow_ptr_offset : overflow_ptr_offset + 4])[0]


def walk_btree_pages(db_bytes: bytes, rootpage: int, page_size: int, usable_size: int) -> Set[int]:
    """Deterministically collect all B-tree + overflow page numbers reachable from `rootpage`."""
    if len(db_bytes) < 100 or db_bytes[:16] != b"SQLite format 3\x00":
        raise ValueError("Invalid or truncated SQLite database header (< 100 bytes or bad magic).")
    if page_size < 512 or usable_size <= 35:
        raise ValueError(f"Invalid SQLite page geometry: page_size={page_size}, usable_size={usable_size}.")
    total_pages = len(db_bytes) // page_size
    visited: Set[int] = set()
    queue: deque[int] = deque([rootpage])

    while queue:
        p = queue.popleft()
        if p <= 0 or p > total_pages or p in visited:
            continue
        visited.add(p)

        page_start = (p - 1) * page_size
        page_slice = db_bytes[page_start : page_start + page_size]
        if len(page_slice) < 12:
            continue
        bh = 100 if p == 1 else 0
        if bh + 8 > len(page_slice):
            continue
        page_type = page_slice[bh]
        if page_type not in (0x02, 0x05, 0x0A, 0x0D):
            continue

        is_interior = page_type in (0x02, 0x05)
        num_cells = struct.unpack(">H", page_slice[bh + 3 : bh + 5])[0]
        if is_interior:
            if bh + 12 > len(page_slice):
                continue
            right_child = struct.unpack(">I", page_slice[bh + 8 : bh + 12])[0]
            if right_child > 0:
                queue.append(right_child)
            ptr_base = bh + 12
        else:
            ptr_base = bh + 8

        if ptr_base + num_cells * 2 > len(page_slice):
            raise ValueError(f"Corrupt cell pointer table on SQLite page {p}.")

        for idx in range(num_cells):
            cell_off = struct.unpack(">H", page_slice[ptr_base + idx * 2 : ptr_base + idx * 2 + 2])[0]
            if cell_off >= len(page_slice):
                raise ValueError(f"Out-of-bounds cell offset {cell_off} on SQLite page {p}.")
            if is_interior:
                if cell_off + 4 > len(page_slice):
                    raise ValueError(f"Truncated interior cell header on SQLite page {p}.")
                left_child = struct.unpack(">I", page_slice[cell_off : cell_off + 4])[0]
                if left_child > 0:
                    queue.append(left_child)
            ov_page = _calc_overflow_first_page(page_slice, cell_off, page_type, usable_size)
            while ov_page > 0 and ov_page <= total_pages and ov_page not in visited:
                visited.add(ov_page)
                ov_start = (ov_page - 1) * page_size
                if ov_start + 4 > len(db_bytes):
                    break
                ov_page = struct.unpack(">I", db_bytes[ov_start : ov_start + 4])[0]

    return visited


def inspect_sqlite_footprint(db_path: Path) -> Dict[str, Any]:
    """Compute file size, PRAGMA counts, per-object B-tree page allocations, and forecasts_ledger column bytes."""
    file_bytes = db_path.read_bytes()
    if len(file_bytes) < 100 or file_bytes[:16] != b"SQLite format 3\x00":
        raise ValueError(f"{db_path} is not a valid hydrated SQLite 3 database file.")
    reserved = int(file_bytes[20])
    conn = connect_sqlite(db_path, read_only=True)
    try:
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
        freelist_count = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0])
        usable_size = page_size - reserved

        master_rows = conn.execute(
            "SELECT type, name, tbl_name, rootpage FROM sqlite_master WHERE rootpage > 0 ORDER BY rootpage"
        ).fetchall()

        objects: List[Dict[str, Any]] = []
        accounted_pages: Set[int] = {1}  # sqlite_schema rootpage=1
        schema_pages = walk_btree_pages(file_bytes, 1, page_size, usable_size)
        accounted_pages.update(schema_pages)
        objects.append(
            {
                "type": "table",
                "name": "sqlite_schema",
                "tbl_name": "sqlite_schema",
                "rootpage": 1,
                "pages": len(schema_pages),
                "bytes": len(schema_pages) * page_size,
            }
        )

        for r in master_rows:
            rp = int(r["rootpage"])
            pages = walk_btree_pages(file_bytes, rp, page_size, usable_size)
            accounted_pages.update(pages)
            objects.append(
                {
                    "type": str(r["type"]),
                    "name": str(r["name"]),
                    "tbl_name": str(r["tbl_name"]),
                    "rootpage": rp,
                    "pages": len(pages),
                    "bytes": len(pages) * page_size,
                }
            )

        table_row_counts: Dict[str, int] = {}
        for r in master_rows:
            if r["type"] == "table":
                tname = str(r["name"])
                table_row_counts[tname] = int(conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0])

        fl_cols = [str(r["name"]) for r in conn.execute("PRAGMA table_info(forecasts_ledger)").fetchall()]
        fl_count = table_row_counts.get("forecasts_ledger", 0)
        col_bytes: Dict[str, Dict[str, Any]] = {}
        if fl_count > 0:
            sum_exprs = [f'COALESCE(SUM(LENGTH(CAST("{c}" AS BLOB))), 0) AS "{c}"' for c in fl_cols]
            row_sums = conn.execute(f"SELECT {', '.join(sum_exprs)} FROM forecasts_ledger").fetchone()
            for c in fl_cols:
                total_b = int(row_sums[c])
                col_bytes[c] = {
                    "total_bytes": total_b,
                    "total_mb": round(total_b / 1e6, 4),
                    "avg_bytes_per_row": round(total_b / fl_count, 2),
                }
            # Include degraded ML vs baseline split stats for imputed_values_json
            deg_row = conn.execute(
                """
                SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(CAST(imputed_values_json AS BLOB))), 0) AS b
                FROM forecasts_ledger
                WHERE is_degraded_input = 1
                """
            ).fetchone()
            col_bytes["imputed_values_json"]["degraded_rows_count"] = int(deg_row["n"])
            col_bytes["imputed_values_json"]["degraded_rows_total_bytes"] = int(deg_row["b"])
            col_bytes["imputed_values_json"]["degraded_rows_avg_bytes"] = (
                round(int(deg_row["b"]) / int(deg_row["n"]), 2) if int(deg_row["n"]) > 0 else 0.0
            )
    finally:
        conn.close()

    by_table: Dict[str, int] = {}
    for obj in objects:
        t = obj["tbl_name"]
        by_table[t] = by_table.get(t, 0) + int(obj["bytes"])

    return {
        "file_size_bytes": len(file_bytes),
        "file_size_mb_decimal": round(len(file_bytes) / 1e6, 4),
        "file_size_mib": round(len(file_bytes) / (1024 * 1024), 4),
        "file_sha256": hashlib.sha256(file_bytes).hexdigest(),
        "journal_mode": journal_mode,
        "page_size_bytes": page_size,
        "page_count": page_count,
        "freelist_count": freelist_count,
        "freelist_bytes": freelist_count * page_size,
        "freelist_mb_decimal": round((freelist_count * page_size) / 1e6, 4),
        "active_btree_pages_walked": len(accounted_pages),
        "expected_active_pages": page_count - freelist_count,
        "table_row_counts": table_row_counts,
        "objects": objects,
        "total_bytes_by_table_including_indexes": by_table,
        "forecasts_ledger_column_bytes": dict(
            sorted(col_bytes.items(), key=lambda kv: kv[1]["total_bytes"], reverse=True)
        ),
    }


def measure_vacuum_on_temp_copy(db_path: Path) -> Dict[str, Any]:
    """Measure exact bytes and per-table page allocations after VACUUM on a temp copy."""
    with tempfile.TemporaryDirectory() as td:
        tmp_db = Path(td) / "vacuum_probe.db"
        shutil.copy2(db_path, tmp_db)
        conn = sqlite3.connect(str(tmp_db))
        try:
            conn.execute("VACUUM;")
            conn.commit()
        finally:
            conn.close()

        vacuumed_conn = connect_sqlite(tmp_db, read_only=False)
        try:
            finalize_db_for_read_only(vacuumed_conn)
        finally:
            vacuumed_conn.close()

        after_info = inspect_sqlite_footprint(tmp_db)
        return {
            "after_vacuum_file_size_bytes": after_info["file_size_bytes"],
            "after_vacuum_file_size_mb_decimal": after_info["file_size_mb_decimal"],
            "after_vacuum_file_size_mib": after_info["file_size_mib"],
            "after_vacuum_page_count": after_info["page_count"],
            "after_vacuum_freelist_count": after_info["freelist_count"],
            "after_vacuum_by_table_bytes": after_info["total_bytes_by_table_including_indexes"],
        }


def run_audit(
    *,
    prod_db: Path = DEFAULT_PROD_DB,
    demo_db: Path = DEFAULT_DEMO_DB,
    proof_json: Path = DEFAULT_PROOF_JSON,
    metadata_json: Path = DEFAULT_METADATA_JSON,
    audit_out: Path = DEFAULT_AUDIT_OUT,
    vacuum_demo: bool = False,
) -> Dict[str, Any]:
    """Run full audit and optionally VACUUM `demo_rainfall_vintage.db` with 29-col SHA-256 verification."""
    report: Dict[str, Any] = {
        "audited_at_utc": now_utc_iso(),
        "prod_db": None,
        "demo_db": {},
    }

    if prod_db.exists():
        prod_before = inspect_sqlite_footprint(prod_db)
        prod_vac_sim = measure_vacuum_on_temp_copy(prod_db)
        report["prod_db"] = {
            "path": str(prod_db.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "read_only_snapshot": prod_before,
            "temp_copy_vacuum_simulation": prod_vac_sim,
        }

    demo_before = inspect_sqlite_footprint(demo_db)
    demo_entry: Dict[str, Any] = {
        "path": str(demo_db.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "before_vacuum": demo_before,
        "vacuum_executed": False,
    }

    if vacuum_demo:
        proof_data = json.loads(proof_json.read_text(encoding="utf-8"))
        expected_hashes = proof_data["after_column_sha256"]

        # Fail fast before VACUUM if pre-existing 29 columns already diverge from proof_v35
        conn_ro = connect_sqlite(demo_db, read_only=True)
        try:
            row_cnt_before, pre_hashes = compute_forecasts_column_hashes(conn_ro, FORECASTS_LEDGER_29_ORIGINAL_COLS)
        finally:
            conn_ro.close()

        pre_mismatches = [
            col for col in FORECASTS_LEDGER_29_ORIGINAL_COLS if pre_hashes[col] != expected_hashes[col]
        ]
        if pre_mismatches:
            raise RuntimeError(f"Pre-VACUUM verification aborted; mismatched columns={pre_mismatches}")

        conn_rw = connect_sqlite(demo_db, read_only=False)
        try:
            conn_rw.execute("VACUUM;")
            conn_rw.commit()
            finalize_db_for_read_only(conn_rw)
        finally:
            conn_rw.close()

        conn_post = connect_sqlite(demo_db, read_only=True)
        try:
            row_cnt_after, post_hashes = compute_forecasts_column_hashes(
                conn_post, FORECASTS_LEDGER_29_ORIGINAL_COLS
            )
            integrity = str(conn_post.execute("PRAGMA integrity_check;").fetchone()[0])
        finally:
            conn_post.close()

        mismatches = [
            col for col in FORECASTS_LEDGER_29_ORIGINAL_COLS if post_hashes[col] != expected_hashes[col]
        ]
        if row_cnt_before != row_cnt_after or mismatches or integrity.lower() != "ok":
            raise RuntimeError(
                f"Post-VACUUM verification failed: rows={row_cnt_before}->{row_cnt_after}, "
                f"mismatches={mismatches}, integrity={integrity}"
            )

        demo_after = inspect_sqlite_footprint(demo_db)
        new_sha = demo_after["file_sha256"]

        proof_data["db_file_sha256"] = new_sha
        proof_data["post_vacuum_audit"] = {
            "vacuumed_at_utc": now_utc_iso(),
            "file_size_bytes_before": demo_before["file_size_bytes"],
            "file_size_bytes_after": demo_after["file_size_bytes"],
            "freelist_count_after": demo_after["freelist_count"],
            "all_29_columns_bit_identical_after_vacuum": True,
        }
        proof_json.write_text(json.dumps(proof_data, indent=2) + "\n", encoding="utf-8")

        meta_data = json.loads(metadata_json.read_text(encoding="utf-8"))
        meta_data.setdefault("sha256_checksums", {})["demo_rainfall_vintage.db"] = new_sha
        metadata_json.write_text(json.dumps(meta_data, indent=2) + "\n", encoding="utf-8")

        demo_entry["vacuum_executed"] = True
        demo_entry["after_vacuum"] = demo_after
        demo_entry["reclaimed_bytes"] = demo_before["file_size_bytes"] - demo_after["file_size_bytes"]
        demo_entry["all_29_columns_bit_identical_after_vacuum"] = len(pre_hashes) == len(post_hashes) == 29 and (
            pre_hashes == post_hashes == expected_hashes
        )
    else:
        demo_entry["temp_copy_vacuum_simulation"] = measure_vacuum_on_temp_copy(demo_db)

    report["demo_db"] = demo_entry
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit DS108 SQLite B-tree storage footprint.")
    parser.add_argument("--vacuum-demo", action="store_true", help="Run VACUUM + finalize_db_for_read_only on demo DB.")
    parser.add_argument("--out", type=Path, default=DEFAULT_AUDIT_OUT)
    args = parser.parse_args()

    res = run_audit(audit_out=args.out, vacuum_demo=args.vacuum_demo)
    demo_summary = res["demo_db"]
    print(
        json.dumps(
            {
                "prod_bytes": res["prod_db"]["read_only_snapshot"]["file_size_bytes"] if res["prod_db"] else None,
                "demo_before_bytes": demo_summary["before_vacuum"]["file_size_bytes"],
                "demo_before_freelist": demo_summary["before_vacuum"]["freelist_count"],
                "demo_vacuum_executed": demo_summary["vacuum_executed"],
                "demo_after_bytes": demo_summary.get("after_vacuum", {}).get("file_size_bytes"),
                "demo_after_sha256": demo_summary.get("after_vacuum", {}).get("file_sha256"),
                "29_cols_identical": demo_summary.get("all_29_columns_bit_identical_after_vacuum"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
