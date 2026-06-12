"""
YourEye branch scheduled hours — fetch from app.youreye.co.uk API.

GET /api/Branches/{branchId}/hours
→ {"branchId": "...", "openingTime": "09:00", "closingTime": "21:00"}

Customise:
  • get_branch_hours_request_headers() — add auth / API headers on fetch
  • build_hours_difference_data()       — scheduled vs actual report fields
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional

DEFAULT_API_BASE = "https://app.youreye.co.uk/api/Branches"
DEFAULT_TIMEOUT_SEC = 15.0


@dataclass(frozen=True)
class BranchHours:
    branch_id: str
    opening_time: time
    closing_time: time
    raw_opening: str
    raw_closing: str

    @classmethod
    def from_api_payload(cls, data: dict) -> BranchHours:
        branch_id = str(data.get("branchId", ""))
        raw_open = str(data.get("openingTime", "")).strip()
        raw_close = str(data.get("closingTime", "")).strip()
        if not branch_id or not raw_open or not raw_close:
            raise ValueError(f"Incomplete branch hours payload: {data!r}")
        return cls(
            branch_id=branch_id,
            opening_time=_parse_hhmm(raw_open),
            closing_time=_parse_hhmm(raw_close),
            raw_opening=raw_open,
            raw_closing=raw_close,
        )


def _parse_hhmm(value: str) -> time:
    s = value.strip()
    if ":" in s:
        parts = s.split(":")
        h = int(parts[0]) % 24
        m = int(parts[1]) % 60 if len(parts) > 1 else 0
        sec = int(parts[2]) % 60 if len(parts) > 2 else 0
        return time(h, m, sec)
    return time(int(s) % 24, 0)


def get_branch_hours_request_headers(
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    HTTP headers for the branch hours GET request.

    Edit this function (or pass extra via config branch_hours_headers) to add
    Authorization, API keys, etc.
    """
    headers = {
        "Accept": "application/json",
        "User-Agent": "YourEye-ShutterMarketHours/1.0",
    }
    if extra:
        headers.update({str(k): str(v) for k, v in extra.items()})
    return headers


def combine_date_time(day: datetime, hhmm: time) -> datetime:
    return datetime(day.year, day.month, day.day, hhmm.hour, hhmm.minute, hhmm.second)


def format_signed_delta(actual: datetime, scheduled: time) -> str:
    """
    actual − scheduled on the same calendar day as *actual*.

    Negative → earlier than scheduled (e.g. 08:10 vs 09:00 → "00:49:40 erken").
    Positive → later than scheduled (e.g. "04:00:45 geç").
    """
    sched_dt = combine_date_time(actual, scheduled)
    delta: timedelta = actual - sched_dt
    total_sec = int(delta.total_seconds())
    label = "Late" if total_sec >= 0 else "Early"
    total_sec = abs(total_sec)
    hours, rem = divmod(total_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d} {label}"


def build_hours_difference_data(
    *,
    opening_time: Optional[datetime],
    closing_time: Optional[datetime],
    scheduled: Optional[BranchHours],
) -> dict[str, str]:
    """
    Build scheduled vs actual hour-difference fields for get_data().

    Customise this function to add fields, rename keys, or change formatting.
    Returns only string values (empty when not yet available).
    """
    sched_open = scheduled.raw_opening if scheduled else ""
    sched_close = scheduled.raw_closing if scheduled else ""
    sched_open_t = scheduled.opening_time if scheduled else None
    sched_close_t = scheduled.closing_time if scheduled else None

    open_diff = (
        format_signed_delta(opening_time, sched_open_t)
        if opening_time is not None and sched_open_t is not None
        else ""
    )
    close_diff = (
        format_signed_delta(closing_time, sched_close_t)
        if closing_time is not None and sched_close_t is not None
        else ""
    )

    return {
        "Scheduled Opening Time": sched_open,
        "Scheduled Closing Time": sched_close,
        "Opening Time Difference": open_diff,
        "Closing Time Difference": close_diff,
    }


def fetch_branch_hours(
    branch_id: str,
    *,
    api_base: str = DEFAULT_API_BASE,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    request_headers: dict[str, str] | None = None,
) -> BranchHours:
    url = f"{api_base.rstrip('/')}/{branch_id}/hours"
    headers = get_branch_hours_request_headers(request_headers)
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Branch hours request failed: {url} — {exc}") from exc
    return BranchHours.from_api_payload(data)
