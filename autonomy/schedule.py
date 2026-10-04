"""Active-hours window: the local-time span in which the agent is allowed to run.

Outside the window the whole process goes dormant — no conscious cycles and
no daemon ticks (so no local-model loads on a GPU that other work needs).
Both loops consult this module independently, so each thread stops on its own.

Spec format: "HH:MM-HH:MM" in the machine's local time. The window may wrap
midnight ("22:00-02:00"). "always", "off", "" or None mean no restriction.
"""

import datetime as _dt
import re
from typing import Optional, Tuple

_PATTERN = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")
_ALWAYS = {"", "always", "off", "none", "24/7"}


def parse_active_hours(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse a spec into (start_minute, end_minute) of the day, or None for "always".

    Raises ValueError on a malformed spec so callers can surface the mistake
    rather than silently running 24/7.
    """
    if spec is None or str(spec).strip().lower() in _ALWAYS:
        return None
    m = _PATTERN.match(str(spec))
    if not m:
        raise ValueError(f"active_hours must look like 'HH:MM-HH:MM' or 'always', got {spec!r}")
    h1, m1, h2, m2 = (int(x) for x in m.groups())
    if not (0 <= h1 <= 23 and 0 <= h2 <= 23 and 0 <= m1 <= 59 and 0 <= m2 <= 59):
        raise ValueError(f"active_hours has an out-of-range time: {spec!r}")
    start, end = h1 * 60 + m1, h2 * 60 + m2
    if start == end:
        return None  # a zero-length window is read as "always on"
    return start, end


def describe(spec: Optional[str]) -> str:
    window = parse_active_hours(spec)
    if window is None:
        return "always on"
    s, e = window
    return f"{s // 60:02d}:{s % 60:02d}-{e // 60:02d}:{e % 60:02d} local"


def _minute_of_day(now: _dt.datetime) -> int:
    return now.hour * 60 + now.minute


def is_active(spec: Optional[str], now: Optional[_dt.datetime] = None) -> bool:
    """True when `now` (local time) falls inside the window. Always True for "always"."""
    window = parse_active_hours(spec)
    if window is None:
        return True
    start, end = window
    cur = _minute_of_day(now or _dt.datetime.now())
    if start < end:
        return start <= cur < end
    return cur >= start or cur < end  # wraps midnight


def next_active_at(spec: Optional[str], now: Optional[_dt.datetime] = None) -> _dt.datetime:
    """The next moment the window opens (or `now` itself when already active / always on)."""
    now = now or _dt.datetime.now()
    window = parse_active_hours(spec)
    if window is None or is_active(spec, now):
        return now
    start, _ = window
    candidate = now.replace(hour=start // 60, minute=start % 60, second=0, microsecond=0)
    if candidate <= now:
        candidate += _dt.timedelta(days=1)
    return candidate


def next_inactive_at(spec: Optional[str], now: Optional[_dt.datetime] = None) -> Optional[_dt.datetime]:
    """The next moment the window closes; None when always on; `now` when already inactive."""
    now = now or _dt.datetime.now()
    window = parse_active_hours(spec)
    if window is None:
        return None
    if not is_active(spec, now):
        return now
    _, end = window
    candidate = now.replace(hour=end // 60, minute=end % 60, second=0, microsecond=0)
    if candidate <= now:
        candidate += _dt.timedelta(days=1)
    return candidate


def seconds_until_active(spec: Optional[str], now: Optional[_dt.datetime] = None) -> int:
    """Seconds until the window opens; 0 when active or always on."""
    now = now or _dt.datetime.now()
    return max(0, int((next_active_at(spec, now) - now).total_seconds()))


def seconds_until_inactive(spec: Optional[str], now: Optional[_dt.datetime] = None) -> Optional[int]:
    """Seconds until the window closes; None when always on; 0 when already inactive."""
    now = now or _dt.datetime.now()
    at = next_inactive_at(spec, now)
    if at is None:
        return None
    return max(0, int((at - now).total_seconds()))
