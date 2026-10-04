"""Tests for autonomy.schedule (active-hours window)."""

import datetime as dt
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autonomy.schedule import (  # noqa: E402
    describe, is_active, next_active_at, next_inactive_at, parse_active_hours,
    seconds_until_active, seconds_until_inactive,
)

T = lambda h, m=0: dt.datetime(2026, 10, 5, h, m)  # a Monday  # noqa: E731


def test_parse():
    assert parse_active_hours("09:00-17:00") == (540, 1020)
    assert parse_active_hours(" 9:05 - 17:30 ") == (545, 1050)
    assert parse_active_hours("22:00-02:00") == (1320, 120)
    for s in ("", None, "always", "ALWAYS", "off"):
        assert parse_active_hours(s) is None
    assert parse_active_hours("09:00-09:00") is None
    for bad in ("9-17", "09:00", "25:00-17:00", "09:60-17:00", "nine to five"):
        with pytest.raises(ValueError):
            parse_active_hours(bad)


def test_describe():
    assert describe("9:00-17:00") == "09:00-17:00 local"
    assert describe("always") == "always on"


def test_is_active_day_window():
    spec = "09:00-17:00"
    assert not is_active(spec, T(8, 59))
    assert is_active(spec, T(9, 0))
    assert is_active(spec, T(16, 59))
    assert not is_active(spec, T(17, 0))
    assert not is_active(spec, T(23, 30))
    assert is_active("always", T(3))


def test_is_active_overnight_window():
    spec = "22:00-02:00"
    assert is_active(spec, T(23))
    assert is_active(spec, T(1, 59))
    assert not is_active(spec, T(2, 0))
    assert not is_active(spec, T(12))
    assert is_active(spec, T(22, 0))


def test_next_active_and_seconds():
    spec = "09:00-17:00"
    assert next_active_at(spec, T(8, 30)) == T(9, 0)
    assert seconds_until_active(spec, T(8, 30)) == 1800
    assert next_active_at(spec, T(18)) == dt.datetime(2026, 10, 6, 9, 0)
    assert seconds_until_active(spec, T(18)) == 15 * 3600
    assert next_active_at(spec, T(12)) == T(12)
    assert seconds_until_active(spec, T(12)) == 0
    assert seconds_until_active("always", T(3)) == 0


def test_next_inactive_and_seconds():
    spec = "09:00-17:00"
    assert next_inactive_at(spec, T(12)) == T(17)
    assert seconds_until_inactive(spec, T(16, 59)) == 60
    assert seconds_until_inactive(spec, T(18)) == 0
    assert seconds_until_inactive("always", T(18)) is None
    # Overnight window: active at 23:00 closes at 02:00 next day.
    assert next_inactive_at("22:00-02:00", T(23)) == dt.datetime(2026, 10, 6, 2, 0)
    assert seconds_until_inactive("22:00-02:00", T(23)) == 3 * 3600


def test_window_end_is_exclusive_and_consistent():
    spec = "09:00-17:00"
    at = next_inactive_at(spec, T(10))
    assert not is_active(spec, at)
    assert is_active(spec, at - dt.timedelta(minutes=1))
