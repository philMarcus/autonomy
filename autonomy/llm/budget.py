"""Daily cost budget tracker for multi-model usage.

Tracks per-model USD spend and provides can_afford() checks for the
subconscious gate and conscious loop.  Resets at midnight UTC.

Pricing is loaded from pricing.json (easy to update separately).
"""

import datetime
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict

from .base import LLMResponse, ModelInfo


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def _load_pricing() -> Dict[str, Dict[str, float]]:
    """Load pricing from pricing.json next to this file."""
    pricing_path = os.path.join(os.path.dirname(__file__), "pricing.json")
    try:
        with open(pricing_path, "r") as f:
            data = json.load(f)
        # Filter out _note / _url keys, and strip _deprecated from model entries
        result = {}
        for k, v in data.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                result[k] = {ck: cv for ck, cv in v.items() if not ck.startswith("_")}
            else:
                result[k] = v
        return result
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# USD per 1K tokens.  Loaded from pricing.json at import time.
COST_TABLE: Dict[str, Dict[str, float]] = _load_pricing()


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """Estimate USD cost for a model and token counts.

    `cached_tokens` is the part of `input_tokens` served from a context cache,
    priced at the model's `cached_input` rate (10% of input when not listed).
    """
    costs = COST_TABLE.get(model_id)
    if not costs:
        return 0.0
    cached = max(0, min(int(cached_tokens or 0), int(input_tokens or 0)))
    fresh = max(0, int(input_tokens or 0) - cached)
    cached_rate = costs.get("cached_input", costs["input"] * 0.1)
    return ((fresh / 1000) * costs["input"] + (cached / 1000) * cached_rate
            + (int(output_tokens or 0) / 1000) * costs["output"])


def pricing_age_days() -> int:
    """Days since pricing.json was last modified. -1 if file not found."""
    pricing_path = os.path.join(os.path.dirname(__file__), "pricing.json")
    try:
        mtime = os.path.getmtime(pricing_path)
        age = datetime.datetime.now().timestamp() - mtime
        return int(age / 86400)
    except FileNotFoundError:
        return -1


@dataclass
class DailyBudget:
    """Tracks daily USD spend across all models.

    Thread-safe: all public methods acquire _lock so the daemon and conscious
    threads can call record_usage / can_afford concurrently.
    """

    daily_limit_usd: float = 1.0
    _spend_by_model: Dict[str, float] = field(default_factory=dict)
    _spend_date: str = ""  # YYYY-MM-DD, resets when date changes
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _ensure_today(self) -> None:
        """Reset counters if the UTC date has changed.  Caller must hold _lock."""
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        if self._spend_date != today:
            self._spend_by_model.clear()
            self._spend_date = today

    def record_usage(self, model_id: str, response: LLMResponse) -> None:
        """Record tokens and cost from an LLM response."""
        with self._lock:
            self._ensure_today()
            cost = response.cost_usd
            if cost <= 0:
                cost = estimate_cost(model_id, response.input_tokens, response.output_tokens,
                                     getattr(response, "cached_tokens", 0))
            self._spend_by_model[model_id] = self._spend_by_model.get(model_id, 0.0) + cost

    def remaining_usd(self) -> float:
        """How much budget remains today."""
        with self._lock:
            self._ensure_today()
            spent = sum(self._spend_by_model.values())
            return max(0.0, self.daily_limit_usd - spent)

    def remaining_fraction(self) -> float:
        """Fraction of daily budget remaining (0.0-1.0)."""
        with self._lock:
            self._ensure_today()
            spent = sum(self._spend_by_model.values())
            if self.daily_limit_usd <= 0:
                return 0.0
            return max(0.0, min(1.0, 1.0 - spent / self.daily_limit_usd))

    def spent_today_usd(self) -> float:
        with self._lock:
            self._ensure_today()
            return sum(self._spend_by_model.values())

    def can_afford(self, model_id: str, est_input_tokens: int = 1000,
                   est_output_tokens: int = 500) -> bool:
        """Check if estimated call fits within remaining budget."""
        with self._lock:
            self._ensure_today()
            est_cost = estimate_cost(model_id, est_input_tokens, est_output_tokens)
            spent = sum(self._spend_by_model.values())
            remaining = max(0.0, self.daily_limit_usd - spent)
            return remaining >= est_cost

    def spend_summary(self) -> Dict[str, Any]:
        """For telemetry and dashboard: per-model breakdown."""
        with self._lock:
            self._ensure_today()
            spent = sum(self._spend_by_model.values())
            return {
                "date": self._spend_date,
                "daily_limit_usd": self.daily_limit_usd,
                "spent_usd": round(spent, 6),
                "remaining_usd": round(max(0.0, self.daily_limit_usd - spent), 6),
                "by_model": {k: round(v, 6) for k, v in self._spend_by_model.items()},
            }

    def spend_summary_text(self) -> str:
        """Human-readable budget summary for inclusion in LLM prompt."""
        s = self.spend_summary()
        lines = [
            f"Daily limit: ${s['daily_limit_usd']:.2f} | "
            f"Spent: ${s['spent_usd']:.4f} | "
            f"Remaining: ${s['remaining_usd']:.4f}"
        ]
        for model, cost in sorted(s.get("by_model", {}).items()):
            lines.append(f"  {model}: ${cost:.4f}")
        return "\n".join(lines)

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize current spend for persistence to state.

        Called on each cycle_end so a restart mid-day resumes with the
        correct amount already spent (rather than resetting to $0).
        """
        with self._lock:
            return {
                "date": self._spend_date,
                "spend_by_model": dict(self._spend_by_model),
            }

    def load_from_state(self, saved: Dict[str, Any], reset: bool = False) -> None:
        """Restore today's spend from saved state.

          - saved date == today (UTC): restore the per-model spend exactly, so a
            restart mid-day resumes with the right amount already spent.
          - otherwise (prior day, or nothing saved): start at $0. The daily
            limit is a cap, not an entitlement, so a late-day restart simply
            has the whole cap available for what is left of the day.
          - reset=True (--reset-budget): start at $0 regardless.

        Earlier versions wrote a synthetic "__prorate__" line item that charged
        the budget for hours of the UTC day that had already passed; it is
        dropped on load if present.
        """
        import datetime
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        saved_date = (saved or {}).get("date", "")
        with self._lock:
            self._spend_date = today
            self._spend_by_model.clear()
            if reset or saved_date != today:
                return
            for k, v in (saved or {}).get("spend_by_model", {}).items():
                if k == "__prorate__":
                    continue
                try:
                    self._spend_by_model[k] = float(v)
                except (TypeError, ValueError):
                    pass

    def spend_summary_for_planning(self, registry=None) -> str:
        """Extended budget summary with cost projections for budget planning.

        Includes hours remaining in UTC day, model cost table, and
        projected spend extrapolation.
        """
        s = self.spend_summary()
        now = datetime.datetime.now(datetime.timezone.utc)
        hours_left = max(0.0, 24.0 - (now.hour + now.minute / 60))

        lines = [
            f"Daily limit: ${s['daily_limit_usd']:.2f} | "
            f"Spent: ${s['spent_usd']:.4f} | "
            f"Remaining: ${s['remaining_usd']:.4f}",
            f"Hours remaining today (UTC): {hours_left:.1f}",
        ]

        # Per-model spend
        if s.get("by_model"):
            lines.append("Spend by model:")
            for model, cost in sorted(s["by_model"].items()):
                lines.append(f"  {model}: ${cost:.4f}")

        # Available model costs
        if COST_TABLE:
            lines.append("")
            lines.append("Available model costs (per 1K tokens):")
            for model_id, costs in sorted(COST_TABLE.items()):
                if costs["input"] == 0 and costs["output"] == 0:
                    lines.append(f"  {model_id}: FREE")
                else:
                    lines.append(
                        f"  {model_id}: "
                        f"${costs['input']:.5f} in / ${costs['output']:.5f} out"
                    )

        return "\n".join(lines)
