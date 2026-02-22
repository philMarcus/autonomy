"""Daily cost budget tracker for multi-model usage.

Tracks per-model USD spend and provides can_afford() checks for the
subconscious gate and conscious loop.  Resets at midnight UTC.
"""

import datetime
import threading
from dataclasses import dataclass, field
from typing import Any, Dict

from .base import LLMResponse, ModelInfo


# Built-in cost table (USD per 1K tokens).  Updated as prices change.
COST_TABLE: Dict[str, Dict[str, float]] = {
    # Gemini
    "gemini-2.5-flash":          {"input": 0.00015,  "output": 0.0006},
    "gemini-2.5-pro":            {"input": 0.00125,  "output": 0.005},
    "gemini-2.0-flash-lite":     {"input": 0.0,      "output": 0.0},
    "gemini-3-pro-preview":      {"input": 0.00125,  "output": 0.005},
    "gemini-3-flash-preview":    {"input": 0.00015,  "output": 0.0006},
    # Anthropic
    "claude-haiku-4-5":          {"input": 0.001,    "output": 0.005},
    "claude-sonnet-4-5":         {"input": 0.003,    "output": 0.015},
    "claude-opus-4-6":           {"input": 0.005,    "output": 0.025},
    # OpenAI
    "gpt-5-nano":                {"input": 0.00005,  "output": 0.0004},
    "gpt-5-mini":                {"input": 0.00025,  "output": 0.002},
    "gpt-5.1":                   {"input": 0.00125,  "output": 0.01},
    "gpt-5.2":                   {"input": 0.00175,  "output": 0.014},
    "gpt-5-pro":                 {"input": 0.015,    "output": 0.12},
    "gpt-5.2-pro":               {"input": 0.021,    "output": 0.168},
    # Mistral
    "mistral-small-latest":      {"input": 0.0002,   "output": 0.0006},
    "mistral-large-latest":      {"input": 0.002,    "output": 0.006},
    # Local (free)
}


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a given model and token counts."""
    costs = COST_TABLE.get(model_id)
    if not costs:
        return 0.0
    return (input_tokens / 1000) * costs["input"] + (output_tokens / 1000) * costs["output"]


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
                cost = estimate_cost(model_id, response.input_tokens, response.output_tokens)
            self._spend_by_model[model_id] = self._spend_by_model.get(model_id, 0.0) + cost

    def remaining_usd(self) -> float:
        """How much budget remains today."""
        with self._lock:
            self._ensure_today()
            spent = sum(self._spend_by_model.values())
            return max(0.0, self.daily_limit_usd - spent)

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
