"""Weekly q50 crossing event aggregates for Grafana."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
import math
from zoneinfo import ZoneInfo

import numpy as np

from .data import ns_to_datetime
from .events import CrossingEvents
from .sessions import classify_session


@dataclass(frozen=True)
class WeeklySessionStat:
    week_start_ts: datetime
    session_sort_order: int
    session_label: str
    session_start_local_time: time
    session_end_local_time: time
    crossings_total: int
    week_first_cross_ts: datetime
    week_last_cross_ts: datetime
    min_range_points: float | None
    avg_range_points: float | None
    median_range_points: float | None
    p75_range_points: float | None
    p95_range_points: float | None
    max_range_points: float | None


def aggregate_weekly_session_stats(
    events: CrossingEvents,
    timezone_name: str = "America/New_York",
) -> list[WeeklySessionStat]:
    tz = ZoneInfo(timezone_name)
    values_by_key: dict[tuple[datetime, int], list[float]] = {}
    info_by_key: dict[tuple[datetime, int], dict[str, object]] = {}

    ranges = np.asarray(events.profile_range_points, dtype=np.float64)
    for idx, value in enumerate(ranges):
        if not math.isfinite(float(value)):
            continue
        cross_dt_utc = ns_to_datetime(int(events.cross_ts_ns[idx]))
        local_dt = cross_dt_utc.astimezone(tz)
        week_start_local_date = local_dt.date() - timedelta(days=local_dt.weekday())
        week_start_local = datetime.combine(week_start_local_date, time(0, 0), tzinfo=tz)
        week_start_ts = week_start_local.astimezone(timezone.utc)
        session = classify_session(local_dt.time())
        key = (week_start_ts, session.sort_order)
        values_by_key.setdefault(key, []).append(float(value))

        info = info_by_key.get(key)
        cross_ns = int(events.cross_ts_ns[idx])
        if info is None:
            info_by_key[key] = {
                "session_label": session.label,
                "session_start_local_time": session.start,
                "session_end_local_time": session.end,
                "week_first_cross_ns": cross_ns,
                "week_last_cross_ns": cross_ns,
            }
            continue
        if cross_ns < int(info["week_first_cross_ns"]):
            info["week_first_cross_ns"] = cross_ns
        if cross_ns > int(info["week_last_cross_ns"]):
            info["week_last_cross_ns"] = cross_ns

    stats: list[WeeklySessionStat] = []
    for key in sorted(values_by_key.keys(), key=lambda item: (item[0], item[1])):
        week_start_ts, session_sort_order = key
        info = info_by_key[key]
        values = values_by_key.get(key, [])
        computed = _compute_range_stats(values)
        stats.append(
            WeeklySessionStat(
                week_start_ts=week_start_ts,
                session_sort_order=session_sort_order,
                session_label=str(info["session_label"]),
                session_start_local_time=info["session_start_local_time"],
                session_end_local_time=info["session_end_local_time"],
                crossings_total=len(values),
                week_first_cross_ts=ns_to_datetime(int(info["week_first_cross_ns"])),
                week_last_cross_ts=ns_to_datetime(int(info["week_last_cross_ns"])),
                min_range_points=computed["min"],
                avg_range_points=computed["avg"],
                median_range_points=computed["median"],
                p75_range_points=computed["p75"],
                p95_range_points=computed["p95"],
                max_range_points=computed["max"],
            )
        )
    return stats


def _compute_range_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "avg": None, "median": None, "p75": None, "p95": None, "max": None}
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "avg": float(np.mean(arr)),
        "median": float(np.quantile(arr, 0.5)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }
