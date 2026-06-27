"""NY session classification for range aggregates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time


@dataclass(frozen=True)
class SessionDef:
    sort_order: int
    label: str
    start: time
    end: time


SESSIONS: tuple[SessionDef, ...] = (
    SessionDef(10, "Asia Early 20:00-00:00", time(20, 0), time(0, 0)),
    SessionDef(20, "Asia Late 00:00-03:00", time(0, 0), time(3, 0)),
    SessionDef(30, "London Open 03:00-04:00", time(3, 0), time(4, 0)),
    SessionDef(40, "Pre Market Early 04:00-07:00", time(4, 0), time(7, 0)),
    SessionDef(50, "Pre Market Active 07:00-08:30", time(7, 0), time(8, 30)),
    SessionDef(60, "Pre Market Macro 08:30-09:30", time(8, 30), time(9, 30)),
    SessionDef(70, "NY Open Impulse 09:30-10:00", time(9, 30), time(10, 0)),
    SessionDef(80, "NY Morning 10:00-11:30", time(10, 0), time(11, 30)),
    SessionDef(90, "NY Midday 11:30-14:00", time(11, 30), time(14, 0)),
    SessionDef(100, "NY Late 14:00-15:00", time(14, 0), time(15, 0)),
    SessionDef(110, "NY Power Hour 15:00-16:00", time(15, 0), time(16, 0)),
    SessionDef(120, "After Close Shock 16:00-17:00", time(16, 0), time(17, 0)),
    SessionDef(130, "After Hours Late 17:00-20:00", time(17, 0), time(20, 0)),
)


def classify_session(local_time: time) -> SessionDef:
    if local_time >= time(20, 0):
        return SESSIONS[0]
    for session in SESSIONS[1:]:
        if local_time < session.end:
            return session
    return SESSIONS[-1]
