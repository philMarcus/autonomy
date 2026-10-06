"""Cost estimation with context-cache pricing."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autonomy.llm import budget as B  # noqa: E402
from autonomy.llm.base import LLMResponse  # noqa: E402


def test_estimate_cost_prices_cached_input_at_cache_rate():
    full = B.estimate_cost("gemini-3.8-flash", 300_000, 5_000)
    mostly_cached = B.estimate_cost("gemini-3.8-flash", 300_000, 5_000, cached_tokens=270_000)
    assert full > mostly_cached * 3
    # 30K fresh @0.00075 + 270K cached @0.000075 + 5K out @0.00375
    assert abs(mostly_cached - (30 * 0.00075 + 270 * 0.000075 + 5 * 0.00375)) < 1e-6
    # cached never exceeds input; models without a cache rate fall back to 10%
    assert B.estimate_cost("gemini-3.8-flash", 1000, 0, cached_tokens=5000) == B.estimate_cost("gemini-3.8-flash", 1000, 0, cached_tokens=1000)
    assert abs(B.estimate_cost("gemini-3.5-flash-lite", 1000, 0, cached_tokens=1000) - 0.0003 * 0.1) < 1e-9
    assert B.estimate_cost("unknown-model", 1000, 1000) == 0.0


def test_record_usage_uses_cached_tokens():
    b = B.DailyBudget(daily_limit_usd=1.0)
    b.record_usage("gemini-3.8-flash", LLMResponse(text="", input_tokens=300_000, output_tokens=5_000,
                                                   cached_tokens=270_000, model_id="gemini-3.8-flash"))
    spent = 1.0 - b.remaining_usd()
    assert abs(spent - B.estimate_cost("gemini-3.8-flash", 300_000, 5_000, 270_000)) < 1e-6


def test_load_from_state_rules():
    import datetime
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    b = B.DailyBudget(daily_limit_usd=1.0)
    # Same day: exact restore, prorate line item dropped.
    b.load_from_state({"date": today, "spend_by_model": {"__prorate__": 0.95, "gemini-3.8-flash": 0.24}})
    assert abs(b.remaining_usd() - 0.76) < 1e-9
    # Prior day: start at zero (no prorating).
    b.load_from_state({"date": "2026-01-01", "spend_by_model": {"gemini-3.8-flash": 0.9}})
    assert b.remaining_usd() == 1.0
    # Nothing saved: zero.
    b.load_from_state({})
    assert b.remaining_usd() == 1.0
    # Explicit reset ignores same-day spend.
    b.load_from_state({"date": today, "spend_by_model": {"gemini-3.8-flash": 0.9}}, reset=True)
    assert b.remaining_usd() == 1.0
