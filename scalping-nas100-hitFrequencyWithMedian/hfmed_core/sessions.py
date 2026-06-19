"""Entry session classification shared by simulation and reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd

ENTRY_SESSION_COLUMN = "entry_session"

PRE_MARKET_START_MINUTE = 4 * 60
NY_OPEN_POWER_START_MINUTE = 9 * 60 + 30
NY_MIDDAY_START_MINUTE = 11 * 60 + 30
NY_LATE_START_MINUTE = 14 * 60
NY_POWER_HOUR_START_MINUTE = 15 * 60
AFTER_HOURS_START_MINUTE = 16 * 60
OVERNIGHT_START_MINUTE = 20 * 60

SESSION_TYPES = (
    ("pre_market", "Pre-Market", 1),
    ("ny_open_power", "NY Open Power", 2),
    ("ny_midday", "NY Midday", 3),
    ("ny_late", "NY Late", 4),
    ("ny_power_hour", "NY Power Hour", 5),
    ("after_hours", "After Hours", 6),
    ("overnight", "Overnight", 7),
)

SESSION_LABELS = {key: label for key, label, _sort_order in SESSION_TYPES}
SESSION_SORT_ORDERS = {key: sort_order for key, _label, sort_order in SESSION_TYPES}
SESSION_CODE_BY_KEY = {key: idx for idx, (key, _label, _sort_order) in enumerate(SESSION_TYPES)}
SESSION_KEY_BY_CODE = {code: key for key, code in SESSION_CODE_BY_KEY.items()}


def classify_minutes(minute_of_day: np.ndarray) -> np.ndarray:
    sessions = np.full(len(minute_of_day), "overnight", dtype=object)
    sessions[(PRE_MARKET_START_MINUTE <= minute_of_day) & (minute_of_day < NY_OPEN_POWER_START_MINUTE)] = "pre_market"
    sessions[(NY_OPEN_POWER_START_MINUTE <= minute_of_day) & (minute_of_day < NY_MIDDAY_START_MINUTE)] = "ny_open_power"
    sessions[(NY_MIDDAY_START_MINUTE <= minute_of_day) & (minute_of_day < NY_LATE_START_MINUTE)] = "ny_midday"
    sessions[(NY_LATE_START_MINUTE <= minute_of_day) & (minute_of_day < NY_POWER_HOUR_START_MINUTE)] = "ny_late"
    sessions[(NY_POWER_HOUR_START_MINUTE <= minute_of_day) & (minute_of_day < AFTER_HOURS_START_MINUTE)] = "ny_power_hour"
    sessions[(AFTER_HOURS_START_MINUTE <= minute_of_day) & (minute_of_day < OVERNIGHT_START_MINUTE)] = "after_hours"
    return sessions


def classify_minutes_codes(minute_of_day: np.ndarray) -> np.ndarray:
    sessions = np.full(len(minute_of_day), SESSION_CODE_BY_KEY["overnight"], dtype=np.uint8)
    sessions[(PRE_MARKET_START_MINUTE <= minute_of_day) & (minute_of_day < NY_OPEN_POWER_START_MINUTE)] = SESSION_CODE_BY_KEY["pre_market"]
    sessions[(NY_OPEN_POWER_START_MINUTE <= minute_of_day) & (minute_of_day < NY_MIDDAY_START_MINUTE)] = SESSION_CODE_BY_KEY["ny_open_power"]
    sessions[(NY_MIDDAY_START_MINUTE <= minute_of_day) & (minute_of_day < NY_LATE_START_MINUTE)] = SESSION_CODE_BY_KEY["ny_midday"]
    sessions[(NY_LATE_START_MINUTE <= minute_of_day) & (minute_of_day < NY_POWER_HOUR_START_MINUTE)] = SESSION_CODE_BY_KEY["ny_late"]
    sessions[(NY_POWER_HOUR_START_MINUTE <= minute_of_day) & (minute_of_day < AFTER_HOURS_START_MINUTE)] = SESSION_CODE_BY_KEY["ny_power_hour"]
    sessions[(AFTER_HOURS_START_MINUTE <= minute_of_day) & (minute_of_day < OVERNIGHT_START_MINUTE)] = SESSION_CODE_BY_KEY["after_hours"]
    return sessions


def classify_timestamps(timestamps: pd.Series, timezone: str) -> np.ndarray:
    local = pd.to_datetime(timestamps, utc=True).dt.tz_convert(timezone)
    minute_of_day = local.dt.hour.to_numpy(dtype=np.int16) * 60 + local.dt.minute.to_numpy(dtype=np.int16)
    return classify_minutes(minute_of_day)


def classify_timestamp_codes(timestamps, timezone: str) -> np.ndarray:
    local = pd.to_datetime(timestamps, utc=True).tz_convert(timezone)
    minute_of_day = local.hour.to_numpy(dtype=np.int16) * 60 + local.minute.to_numpy(dtype=np.int16)
    return classify_minutes_codes(minute_of_day)


def session_key_for_code(code: int) -> str:
    return SESSION_KEY_BY_CODE.get(int(code), "overnight")


def add_entry_session_column(ticks: pd.DataFrame, timezone: str) -> pd.DataFrame:
    if ENTRY_SESSION_COLUMN in ticks.columns:
        return ticks
    out = ticks.copy()
    out[ENTRY_SESSION_COLUMN] = classify_timestamps(out["tick_time"], timezone)
    return out
