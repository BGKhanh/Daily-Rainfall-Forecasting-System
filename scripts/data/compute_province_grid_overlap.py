"""CLI generator for `frontend/public/map/province_grid_overlap.json`."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.grid_overlap import compute_all_provinces_grid_overlap  # noqa: E402

DEFAULT_GEOJSON = PROJECT_ROOT / "frontend" / "public" / "map" / "vietnam_63_provinces.geojson"
DEFAULT_OUT = PROJECT_ROOT / "frontend" / "public" / "map" / "province_grid_overlap.json"


def main() -> int:
    payload = compute_all_provinces_grid_overlap(DEFAULT_GEOJSON)
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {DEFAULT_OUT} with {payload['unit_count']} units.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
