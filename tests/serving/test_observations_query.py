"""TDD tests for GET /observations (Seam A), OpenAPI tags, and header.sources binary search (Seam B)."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from fastapi.testclient import TestClient
import pytest

import src.config.clock as clock_mod
from src.api.main import app
from src.data.vintage_store import binary_search_source_transition


def test_observations_default_90d_and_not_published_gaps(monkeypatch):
    monkeypatch.setattr(clock_mod, "now_utc", lambda: datetime(2026, 11, 19, 8, 0, tzinfo=timezone.utc))
    with TestClient(app) as client:
        resp = client.get("/observations")
        assert resp.status_code == 200
        data = resp.json()

        assert data["location_id"] == "HCMC"
        assert data["has_data"] is True
        assert data["empty_reason"] is None
        assert data["granularity"] == "daily"
        assert data["valid_range"]["min_date"] == "2001-01-01"
        assert data["valid_range"]["max_date"] == "2026-11-19"
        assert data["end_date"] == "2026-11-19"
        assert len(data["series"]) == 90

        # Cutoff verified in Task 1: MERRA2 <= 2026-09-30, GEOSIT >= 2026-10-01
        by_date = {row["date"]: row for row in data["series"]}
        assert by_date["2026-09-30"]["status"] == "FINALIZED"
        assert by_date["2026-09-30"]["precip_mm"] is not None
        assert by_date["2026-10-01"]["status"] == "PROVISIONAL"
        assert by_date["2026-10-01"]["precip_mm"] is not None

        # Trailing unpublished days on 2026-11-19 (at least 2026-11-16..2026-11-19)
        assert by_date["2026-11-19"]["status"] == "NOT_PUBLISHED"
        assert by_date["2026-11-19"]["precip_mm"] is None

        # Serving forecast overlay for latest complete anchor (2026-08-30 -> 2026-08-31..2026-09-06)
        overlay = data["forecast_overlay"]
        assert len(overlay) == 7
        assert overlay[0]["date"] == "2026-08-31"
        assert overlay[-1]["date"] == "2026-09-06"
        assert all(pt["pred_mm"] >= 0.0 for pt in overlay)

        # Vocabulary check: zero occurrences of the banned word "normalized"
        assert "normalized" not in json.dumps(data).lower()


def test_observations_validation_errors_and_point_limits(monkeypatch):
    monkeypatch.setattr(clock_mod, "now_utc", lambda: datetime(2026, 11, 19, 8, 0, tzinfo=timezone.utc))
    with TestClient(app) as client:
        # Invalid date format
        r_bad_fmt = client.get("/observations?start_date=2026/08/01&end_date=2026-08-30")
        assert r_bad_fmt.status_code == 422
        assert r_bad_fmt.json()["error"]["code"] == "INVALID_QUERY_PARAMETER"

        # Reversed range
        r_rev = client.get("/observations?start_date=2026-09-10&end_date=2026-09-01")
        assert r_rev.status_code == 422
        assert r_rev.json()["error"]["code"] == "INVALID_QUERY_PARAMETER"

        # Before valid min_date (2001-01-01)
        r_early = client.get("/observations?start_date=2000-12-31&end_date=2001-02-01")
        assert r_early.status_code == 422
        assert r_early.json()["error"]["code"] == "INVALID_QUERY_PARAMETER"

        # Exceeds daily limit (366 days)
        r_daily_over = client.get("/observations?granularity=daily&start_date=2025-01-01&end_date=2026-02-01")
        assert r_daily_over.status_code == 422
        body_over = r_daily_over.json()
        assert body_over["error"]["code"] == "INVALID_QUERY_PARAMETER"
        assert "weekly" in body_over["error"]["message"].lower()

        # Exceeds weekly limit (2555 days)
        r_weekly_over = client.get("/observations?granularity=weekly&start_date=2018-01-01&end_date=2026-04-30")
        assert r_weekly_over.status_code == 422
        assert r_weekly_over.json()["error"]["code"] == "INVALID_QUERY_PARAMETER"

        # Unknown location -> 422 UNSUPPORTED_LOCATION
        r_bad_loc = client.get("/observations?location_id=ATLANTIS")
        assert r_bad_loc.status_code == 422
        assert r_bad_loc.json()["error"]["code"] == "UNSUPPORTED_LOCATION"


def test_observations_unpopulated_map_province_returns_200_empty(monkeypatch):
    monkeypatch.setattr(clock_mod, "now_utc", lambda: datetime(2026, 11, 19, 8, 0, tzinfo=timezone.utc))
    with TestClient(app) as client:
        resp = client.get("/observations?location_id=HANOI&start_date=2026-09-01&end_date=2026-11-19")
        assert resp.status_code == 200
        data = resp.json()
        assert data["location_id"] == "HANOI"
        assert data["display_name"] == "Ha Noi"
        assert data["has_data"] is False
        assert data["empty_reason"] == "No observations have been ingested for this location yet."
        assert data["series"] == []


def test_observations_weekly_and_monthly_exact_parity_with_daily(monkeypatch):
    monkeypatch.setattr(clock_mod, "now_utc", lambda: datetime(2026, 11, 19, 8, 0, tzinfo=timezone.utc))
    start_iso, end_iso = "2026-08-24", "2026-11-19"
    with TestClient(app) as client:
        d_res = client.get(f"/observations?granularity=daily&start_date={start_iso}&end_date={end_iso}").json()
        w_res = client.get(f"/observations?granularity=weekly&start_date={start_iso}&end_date={end_iso}").json()
        m_res = client.get(f"/observations?granularity=monthly&start_date={start_iso}&end_date={end_iso}").json()

        daily_sum = sum(r["precip_mm"] for r in d_res["series"] if r["precip_mm"] is not None)
        weekly_sum = sum(r["precip_total_mm"] for r in w_res["series"] if r["precip_total_mm"] is not None)
        monthly_sum = sum(r["precip_total_mm"] for r in m_res["series"] if r["precip_total_mm"] is not None)

        assert daily_sum == pytest.approx(d_res["summary"]["total_precip_mm"], abs=1e-4)
        assert weekly_sum == pytest.approx(daily_sum, abs=1e-4)
        assert monthly_sum == pytest.approx(daily_sum, abs=1e-4)

        # Check mixed week 2026-09-28..2026-10-04 (spans FINALIZED <= 09-30 & PROVISIONAL >= 10-01)
        mixed_week = next(r for r in w_res["series"] if r["date"] == "2026-09-28")
        assert mixed_week["status"] == "PROVISIONAL"

        # Month August (clipped 2026-08-24..2026-08-31) is 100% FINALIZED
        aug_month = next(r for r in m_res["series"] if r["date"].startswith("2026-08"))
        assert aug_month["status"] == "FINALIZED"

        # Month October is 100% PROVISIONAL
        oct_month = next(r for r in m_res["series"] if r["date"].startswith("2026-10"))
        assert oct_month["status"] == "PROVISIONAL"


def test_openapi_tags_developer_and_evaluation_and_no_normalized():
    spec = app.openapi()
    assert spec["paths"]["/scores"]["get"]["tags"] == ["Developer & Evaluation"]
    assert spec["paths"]["/benchmark/leaderboard"]["get"]["tags"] == ["Developer & Evaluation"]
    assert "/observations" in spec["paths"]
    assert spec["paths"]["/observations"]["get"]["tags"] == ["Data Retrieval"]
    assert "normalized" not in json.dumps(spec["paths"]["/observations"]).lower()


def test_probe_nasa_header_sources_binary_search_mock():
    from datetime import date, timedelta

    cur = date(2026, 9, 1)
    sparse_dates = []
    while cur <= date(2026, 10, 21):
        sparse_dates.append(cur.isoformat())
        cur += timedelta(days=10)

    def fake_fetch(iso_d: str):
        srcs = ["MERRA2"] if iso_d <= "2026-09-30" else ["GEOSIT"]
        return {"date": iso_d, "sources": srcs, "values": {"PRECTOTCORR": 1.2}}

    sparse_entries = [fake_fetch(d) for d in sparse_dates]
    trans, extra = binary_search_source_transition(
        sparse_entries,
        fake_fetch,
        classifier=lambda r: "+".join(sorted(r.get("sources") or [])),
    )
    assert trans is not None
    assert trans["last_from_date"] == "2026-09-30"
    assert trans["first_to_date"] == "2026-10-01"
    assert len(extra) <= 4
