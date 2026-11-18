"""
Canonical Operational Clock Utilities (DS108)
=============================================
Single source of truth for Ho Chi Minh City operational dates (`Asia/Ho_Chi_Minh`)
and timezone-aware UTC timestamps (`now_utc`). Centralizing clock reads enables
deterministic monkeypatching in tests across UTC/HCMC day boundaries.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

HCMC_TIMEZONE_NAME = "Asia/Ho_Chi_Minh"
HCMC_TZ = ZoneInfo(HCMC_TIMEZONE_NAME)


def now_utc() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def today_hcmc() -> date:
    """Return current calendar date in Asia/Ho_Chi_Minh derived from `now_utc()`."""
    return now_utc().astimezone(HCMC_TZ).date()


def now_utc_iso() -> str:
    """Return canonical UTC ISO-8601 string (`YYYY-MM-DDTHH:MM:SS.ffffffZ`)."""
    return now_utc().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
