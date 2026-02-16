"""ControlRegistry for v15.0 — every tunable is a first-class control.

Controls are:
  1. Readable by the conscious LLM (included in context as formatted block)
  2. Writable by the conscious LLM (via controls_update in JSON output)
  3. Settable by the user via CLI or dashboard
  4. Blacklistable by the user (shown as [LOCKED] to the LLM)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class Control:
    """Definition of a single tunable control."""
    key: str
    dtype: str              # "float", "int", "str", "bool"
    default: Any
    description: str
    category: str           # "llm", "cost", "timing", "output", "social", "daemon", "context"
    min_val: Any = None     # numeric lower bound (inclusive)
    max_val: Any = None     # numeric upper bound (inclusive)
    choices: Optional[List[str]] = None  # valid values for str controls


# Category display order for LLM prompt
_CATEGORY_ORDER = ["llm", "cost", "timing", "output", "social", "daemon", "context"]


class ControlRegistry:
    """Central registry of all tunable controls with blacklist + validation."""

    def __init__(
        self,
        controls: List[Control],
        blacklist: Optional[Set[str]] = None,
    ):
        self._defs: Dict[str, Control] = {}
        self._values: Dict[str, Any] = {}
        self._blacklist: Set[str] = blacklist or set()
        self._change_log: List[Dict[str, Any]] = []

        for c in controls:
            self._defs[c.key] = c
            self._values[c.key] = c.default

    # ------------------------------------------------------------------
    # Core read/write
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any:
        """Get current value of a control. Raises KeyError if unknown."""
        if key not in self._defs:
            raise KeyError(f"Unknown control: {key!r}")
        return self._values[key]

    def set(self, key: str, value: Any, source: str = "conscious") -> bool:
        """Set a control value with validation. Returns False if blocked/invalid."""
        if key not in self._defs:
            return False
        if source == "conscious" and key in self._blacklist:
            return False

        defn = self._defs[key]
        validated = self._validate(defn, value)
        if validated is None:
            return False

        old = self._values[key]
        self._values[key] = validated
        self._change_log.append({
            "key": key,
            "old": old,
            "new": validated,
            "source": source,
            "ts": time.time(),
        })
        return True

    def apply_updates(
        self, updates: Dict[str, Any], source: str = "conscious"
    ) -> Dict[str, str]:
        """Batch-apply control updates. Returns {key: "ok"|"blocked"|"invalid"}."""
        results = {}
        for key, value in updates.items():
            if key not in self._defs:
                results[key] = "unknown"
            elif source == "conscious" and key in self._blacklist:
                results[key] = "blocked"
            elif self.set(key, value, source=source):
                results[key] = "ok"
            else:
                results[key] = "invalid"
        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, defn: Control, value: Any) -> Any:
        """Validate and coerce value to the control's dtype. Returns None on failure."""
        try:
            if defn.dtype == "float":
                v = float(value)
                if defn.min_val is not None:
                    v = max(v, float(defn.min_val))
                if defn.max_val is not None:
                    v = min(v, float(defn.max_val))
                return v
            elif defn.dtype == "int":
                v = int(float(value))  # allow "5.0" → 5
                if defn.min_val is not None:
                    v = max(v, int(defn.min_val))
                if defn.max_val is not None:
                    v = min(v, int(defn.max_val))
                return v
            elif defn.dtype == "bool":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
            elif defn.dtype == "str":
                v = str(value).strip()
                if defn.choices and v not in defn.choices:
                    return None
                return v
        except (ValueError, TypeError):
            return None
        return None

    # ------------------------------------------------------------------
    # Formatting for LLM prompt
    # ------------------------------------------------------------------

    def to_llm_block(self) -> str:
        """Format all controls as a text block for the planner prompt."""
        by_cat: Dict[str, List[str]] = {}
        for cat in _CATEGORY_ORDER:
            by_cat[cat] = []

        for key in sorted(self._defs.keys()):
            defn = self._defs[key]
            cat = defn.category if defn.category in by_cat else "context"
            val = self._values[key]
            locked = key in self._blacklist

            line = self._format_control_line(defn, val, locked)
            by_cat[cat].append(line)

        lines = []
        for cat in _CATEGORY_ORDER:
            entries = by_cat[cat]
            if not entries:
                continue
            lines.append(f"[{cat.upper()}]")
            lines.extend(entries)
            lines.append("")  # blank separator

        return "\n".join(lines).rstrip()

    def _format_control_line(self, defn: Control, val: Any, locked: bool) -> str:
        prefix = "[LOCKED] " if locked else ""
        val_str = self._format_value(defn, val)
        hint = self._format_hint(defn)
        desc = defn.description
        if locked:
            desc += " — user-protected"
        return f"  {prefix}{defn.key}: {val_str}{hint} — {desc}"

    def _format_value(self, defn: Control, val: Any) -> str:
        if defn.dtype == "float":
            return f"{val:.2f}" if isinstance(val, float) else str(val)
        if defn.dtype == "bool":
            return str(val).lower()
        return str(val)

    def _format_hint(self, defn: Control) -> str:
        if defn.choices:
            return f" (choices: {', '.join(defn.choices)})"
        if defn.min_val is not None and defn.max_val is not None:
            return f" ({defn.min_val}–{defn.max_val})"
        return ""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize current values for JSON persistence."""
        return dict(self._values)

    def load_from_dict(self, d: Dict[str, Any]) -> None:
        """Restore values from a saved dict. Unknown keys are ignored."""
        for key, value in d.items():
            if key in self._defs:
                validated = self._validate(self._defs[key], value)
                if validated is not None:
                    self._values[key] = validated

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def definitions(self) -> List[Control]:
        return list(self._defs.values())

    def change_log(self) -> List[Dict[str, Any]]:
        return list(self._change_log)

    def is_blacklisted(self, key: str) -> bool:
        return key in self._blacklist


# ======================================================================
# Factory: build the default registry from CLI args + model registry
# ======================================================================

def build_default_registry(args, model_registry) -> ControlRegistry:
    """Create a ControlRegistry populated from CLI args and available models.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    model_registry : ModelRegistry
        Registered model backends (for populating model choices).
    """
    model_ids = [m.model_id for m in model_registry.list_models()]

    conscious_model = getattr(args, "conscious_model", None) or getattr(args, "gemini_model", "gemini-2.5-flash")
    subconscious_model = getattr(args, "subconscious_model", "gemini-2.5-flash")

    # Determine output_destination choices based on whether moltbook is enabled
    moltbook_enabled = getattr(args, "moltbook_enabled", False)
    output_choices = ["analog_home", "moltbook_and_analog_home"] if moltbook_enabled else ["analog_home"]
    output_default = "analog_home"

    controls = [
        # --- LLM ---
        Control("conscious_model", "str", conscious_model,
                "Model for conscious loop", "llm", choices=model_ids),
        Control("subconscious_model", "str", subconscious_model,
                "Model for subconscious daemon", "llm", choices=model_ids),
        Control("temperature", "float", getattr(args, "temperature", 0.7),
                "Conscious LLM temperature", "llm", min_val=0.0, max_val=2.0),
        Control("subconscious_temperature", "float", 0.3,
                "Daemon LLM temperature", "llm", min_val=0.0, max_val=2.0),

        # --- Cost ---
        Control("daily_budget_usd", "float", getattr(args, "daily_budget", 1.0),
                "Daily API spend limit", "cost", min_val=0.01, max_val=100.0),

        # --- Timing ---
        Control("cycle_interval_minutes", "int", getattr(args, "interval", 5),
                "Minutes between cycles", "timing", min_val=1, max_val=120),
        Control("post_interval_minutes", "int", getattr(args, "post_interval", 30),
                "Minutes between posts", "timing", min_val=5, max_val=1440),

        # --- Output ---
        Control("output_destination", "str", output_default,
                "Where to publish artifacts", "output", choices=output_choices),

        # --- Social ---
        Control("mode", "str", getattr(args, "mode", "all"),
                "Action mode", "social",
                choices=["all", "comment_only", "no_post", "no_comment", "post_only"]),
        Control("follow_prob", "float", getattr(args, "follow_prob", 0.60),
                "Follow-on-like probability", "social", min_val=0.0, max_val=1.0),
        Control("create_submolt_prob", "float",
                getattr(args, "create_submolt_prob", 0.05),
                "Create submolt probability", "social", min_val=0.0, max_val=1.0),
        Control("allow_downvote", "bool", getattr(args, "allow_downvote", True),
                "Allow downvoting", "social"),
        Control("priority", "str", getattr(args, "priority", "replies_first"),
                "Reply priority", "social",
                choices=["replies_first", "outside_first"]),

        # --- Daemon (Phase 5 will consume; registered now with defaults) ---
        Control("sentry_interval_seconds", "int",
                getattr(args, "sentry_interval", 60),
                "Seconds between sentry scans", "daemon", min_val=10, max_val=600),
        Control("signal_threshold", "float", 0.5,
                "Score to trigger strategist", "daemon", min_val=0.0, max_val=1.0),
        Control("wake_threshold", "float", 2.0,
                "Charge to fire conscious", "daemon", min_val=0.5, max_val=10.0),
        Control("wake_decay_rate", "float", 1.0,
                "Per-tick charge decay (1.0=no decay)", "daemon", min_val=0.5, max_val=1.0),
        Control("wake_refractory", "float", -2.0,
                "Wake potential after firing (negative=cooldown)", "daemon", min_val=-10.0, max_val=0.0),
        Control("dream_depth", "int", 20,
                "Memories to compress per dream", "daemon", min_val=1, max_val=50),
        Control("max_drafts", "int", 10,
                "Max drafts in buffer before pruning oldest", "daemon", min_val=1, max_val=50),
        Control("sentry_max_tokens", "int", 256,
                "Max output tokens for sentry scoring", "daemon", min_val=64, max_val=1024),
        Control("strategist_max_tokens", "int", 4096,
                "Max output tokens for strategist drafts", "daemon", min_val=128, max_val=8192),

        # --- Context ---
        Control("feed_batch_size", "int", 12,
                "Feed items per cycle", "context", min_val=1, max_val=50),
        Control("history_context_n", "int", 15,
                "History entries in prompt", "context", min_val=1, max_val=50),
        Control("memory_max_chars", "int", 4000,
                "Memory context budget", "context", min_val=500, max_val=20000),
    ]

    # Parse blacklist from CLI
    blacklist_raw = getattr(args, "blacklist_controls", "") or ""
    blacklist = {k.strip() for k in blacklist_raw.split(",") if k.strip()}

    return ControlRegistry(controls=controls, blacklist=blacklist)
