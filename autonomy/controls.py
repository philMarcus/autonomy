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

    def lock(self, key: str) -> None:
        """Add key to blacklist (persist next time to_dict is saved)."""
        if key in self._defs:
            self._blacklist.add(key)

    def unlock(self, key: str) -> None:
        """Remove key from blacklist."""
        self._blacklist.discard(key)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize current values + locked set for JSON persistence."""
        d = dict(self._values)
        if self._blacklist:
            d["_locked"] = sorted(self._blacklist)
        return d

    def load_from_dict(self, d: Dict[str, Any]) -> None:
        """Restore values and locked set from a saved dict. Unknown keys ignored."""
        locked_from_file = set(d.get("_locked", []))
        for key, value in d.items():
            if key.startswith("_"):
                continue
            if key in self._defs:
                validated = self._validate(self._defs[key], value)
                if validated is not None:
                    self._values[key] = validated
        # Merge file-persisted blacklist with any CLI-supplied blacklist
        self._blacklist = self._blacklist | locked_from_file

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
    all_model_ids = [m.model_id for m in model_registry.list_models()]

    # Pro-tier models suitable for conscious (high-quality reasoning)
    _CONSCIOUS_TIER = {
        "gemini-2.5-pro", "gemini-3-pro-preview", "gemini-3.1-pro-preview",
        "claude-sonnet-4-5", "claude-opus-4-6",
        "gpt-5.1", "gpt-5.2", "gpt-5-pro", "gpt-5.2-pro",
    }
    # Cheap/fast models suitable for subconscious (sentry, strategist, seeker)
    _SUBCONSCIOUS_TIER = {
        "gemini-2.5-flash", "gemini-2.5-flash-lite",
        "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview",
        "gemini-2.0-flash", "gemini-2.0-flash-lite",
        "claude-haiku-4-5", "gpt-5-nano", "gpt-5-mini",
    }
    # Local models are always available for subconscious
    conscious_choices = [m for m in all_model_ids if m in _CONSCIOUS_TIER]
    subconscious_choices = [m for m in all_model_ids if m in _SUBCONSCIOUS_TIER or m.startswith("local:")]

    conscious_model = getattr(args, "conscious_model", None) or getattr(args, "gemini_model", "gemini-2.5-pro")
    subconscious_model = getattr(args, "subconscious_model", "gemini-2.5-flash-lite")

    # Output destination choices — always offer moltbook option (writes gated by moltbook_disabled flag)
    output_choices = ["analog_home", "moltbook_and_analog_home"]
    output_default = "analog_home"

    controls = [
        # --- LLM ---
        Control("conscious_model", "str", conscious_model,
                "Model for conscious loop (pro-tier only)", "llm", choices=conscious_choices),
        Control("subconscious_model", "str", subconscious_model,
                "Model for sentry + strategist (cheap/local)", "llm", choices=subconscious_choices),
        Control("seeker_model", "str", "gemini-2.5-flash-lite",
                "Model for seeker gear (needs Gemini for search grounding)", "llm",
                choices=[m for m in all_model_ids if m.startswith("gemini")]),
        Control("conscious_model_weights", "str", "gemini-2.5-pro=1,gemini-3.1-pro-preview=1",
                "Weighted model pool for conscious (model=weight pairs, comma-separated)", "llm"),
        Control("subconscious_model_weights", "str",
                "local:qwen2.5-1.5b=5,gemini-2.5-flash-lite=1,mistral-small-latest=1,claude-haiku-4-5=0.3",
                "Weighted model pool for sentry+strategist (model=weight pairs)", "llm"),
        Control("temperature", "float", getattr(args, "temperature", 0.7),
                "Conscious LLM temperature", "llm", min_val=0.0, max_val=2.0),
        Control("subconscious_temperature", "float", 0.3,
                "Daemon LLM temperature", "llm", min_val=0.0, max_val=2.0),

        # --- Cost ---
        Control("daily_budget_usd", "float", getattr(args, "daily_budget", 1.0),
                "Daily API spend limit", "cost", min_val=0.01, max_val=100.0),
        Control("budget_plan_enabled", "bool", True,
                "Enable daily budget planning pass", "cost"),
        Control("budget_conserve_threshold", "float", 0.3,
                "Switch to cheaper models below this remaining fraction", "cost",
                min_val=0.05, max_val=0.9),

        # --- Timing ---
        Control("cycle_interval_minutes", "int", getattr(args, "interval", 60),
                "Minutes between cycles", "timing", min_val=1, max_val=120),
        Control("post_interval_minutes", "int", getattr(args, "post_interval", 30),
                "Minutes between posts", "timing", min_val=5, max_val=1440),
        Control("image_cooldown_minutes", "int", 1440,
                "Min minutes between image generations (default 1440 = 24h)", "timing", min_val=10, max_val=10080),
        Control("image_model_tier", "str", "imagen-ultra",
                "Imagen tier: fast ($0.02), standard ($0.04), ultra ($0.06)", "llm",
                choices=["imagen-fast", "imagen-standard", "imagen-ultra"]),

        # --- Output ---
        Control("output_destination", "str", output_default,
                "Where to publish artifacts", "output", choices=output_choices),

        # --- Social ---
        Control("mode", "str", getattr(args, "mode", "all"),
                "Action mode", "social",
                choices=["all", "comment_only", "no_post", "no_comment", "post_only"]),
        Control("allow_downvote", "bool", getattr(args, "allow_downvote", False),
                "Allow downvoting", "social"),
        Control("allow_kernel_update", "bool",
                getattr(args, "allow_kernel_update", True),
                "Allow agent to rewrite its kernel prompt", "social"),
        Control("priority", "str", getattr(args, "priority", "replies_first"),
                "Reply priority", "social",
                choices=["replies_first", "outside_first"]),

        # --- Daemon (Phase 5 will consume; registered now with defaults) ---
        Control("sentry_interval_seconds", "int",
                getattr(args, "sentry_interval", 300),
                "Seconds between sentry scans", "daemon", min_val=10),
        Control("signal_threshold", "float", 0.5,
                "Score to trigger strategist", "daemon", min_val=0.0, max_val=1.0),
        Control("wake_threshold", "float", 3.0,
                "Charge to fire conscious", "daemon", min_val=0.5, max_val=10.0),
        Control("wake_refractory", "float", -2.0,
                "Wake potential after firing (negative=cooldown)", "daemon", min_val=-10.0, max_val=0.0),
        Control("charge_weight_feed", "float", 0.3,
                "Charge multiplier for Moltbook feed items", "daemon", min_val=0.0, max_val=5.0),
        Control("charge_weight_seed", "float", 2.0,
                "Charge multiplier for human seeds", "daemon", min_val=0.0, max_val=5.0),
        Control("dream_depth", "int", 10,
                "History entries to synthesize per dream", "conscious", min_val=3, max_val=50),
        Control("max_drafts", "int", 10,
                "Max drafts in buffer before pruning oldest", "daemon", min_val=1, max_val=50),
        Control("sentry_max_tokens", "int", 256,
                "Max output tokens for sentry scoring", "daemon", min_val=64, max_val=1024),
        Control("strategist_max_tokens", "int", 4096,
                "Max output tokens for strategist drafts", "daemon", min_val=128, max_val=8192),
        Control("max_item_age_hours", "int", 24,
                "Ignore feed items older than this (hours)", "daemon", min_val=1, max_val=168),

        # --- Daemon (Sentry rubric weights) ---
        Control("sentry_weight_relevance", "float", 0.45,
                "Rubric weight for relevance criterion", "daemon", min_val=0.0, max_val=1.0),
        Control("sentry_weight_novelty", "float", 0.30,
                "Rubric weight for novelty criterion", "daemon", min_val=0.0, max_val=1.0),
        Control("sentry_weight_actionability", "float", 0.25,
                "Rubric weight for actionability criterion", "daemon", min_val=0.0, max_val=1.0),

        # --- Daemon (Seeker gear) ---
        Control("seeker_interval_seconds", "int", 900,
                "Seconds between seeker search sweeps", "daemon", min_val=300, max_val=3600),
        Control("seeker_max_tokens", "int", 4096,
                "Max output tokens for seeker responses", "daemon", min_val=256, max_val=8192),
        Control("charge_weight_search", "float", 1.5,
                "Charge multiplier for seeker-discovered items", "daemon", min_val=0.0, max_val=5.0),
        Control("seeker_max_topics", "int", 3,
                "Max focus topics to search per sweep", "daemon", min_val=1, max_val=10),

        # --- Context ---
        Control("feed_batch_size", "int", 8,
                "Feed items per cycle", "context", min_val=1, max_val=50),
        Control("feed_item_chars", "int", 400,
                "Max chars per feed item in prompt", "context", min_val=50, max_val=2000),
        Control("history_context_n", "int", 15,
                "History entries in prompt", "context", min_val=1, max_val=50),
        Control("memory_max_chars", "int", 4000,
                "Memory context budget", "context", min_val=500, max_val=20000),
        Control("reply_candidate_chars", "int", 5000,
                "Max chars for reply candidate text in prompt", "context", min_val=500, max_val=20000),
        Control("outside_candidate_chars", "int", 5000,
                "Max chars for outside comment candidate text in prompt", "context", min_val=500, max_val=20000),

        # --- Social (scanning behaviour) ---
        Control("my_post_scan_limit", "int", 50,
                "Recent own posts to scan for unanswered comments", "social", min_val=5, max_val=200),
        Control("reply_threads_scanned", "int", 4,
                "Own post threads to scan per cycle for replies", "social", min_val=1, max_val=20),
        Control("reply_max_comments", "int", 25,
                "Max comments evaluated per thread for merit-based reply", "social", min_val=5, max_val=100),
        Control("thread_comments_for_engagement", "int", 12,
                "Max comments on a thread before dogpile guard fires", "social", min_val=1, max_val=100),

        # --- Daemon (extended) ---
        Control("saved_plan_max_cycles", "int", 5,
                "Cycles a daemon draft persists before expiring", "daemon", min_val=1, max_val=20),
        Control("daemon_notes_max", "int", 5,
                "Max directive notes retained", "daemon", min_val=1, max_val=20),

        # --- Timing (extended) ---
        Control("post_failure_cooldown_seconds", "int", 900,
                "Cooldown after a failed post attempt", "timing", min_val=60, max_val=7200),

        # --- Per-action cooldown controls ---
        Control("cooldown_comment_seconds", "int", 180,
                "Seconds between comment/reply actions", "timing", min_val=20, max_val=3600),
        Control("cooldown_upvote_seconds", "int", 60,
                "Seconds between upvote actions", "timing", min_val=10, max_val=3600),
        Control("cooldown_follow_seconds", "int", 3600,
                "Seconds between follow actions", "timing", min_val=60, max_val=86400),
        Control("cooldown_subscribe_seconds", "int", 300,
                "Seconds between subscribe actions", "timing", min_val=60, max_val=86400),
        Control("cooldown_dm_seconds", "int", 600,
                "Seconds between DM actions", "timing", min_val=60, max_val=86400),
        Control("cooldown_create_submolt_seconds", "int", 3600,
                "Seconds between submolt creation", "timing", min_val=600, max_val=86400),

        # --- Daemon permission controls ---
        Control("daemon_can_upvote", "bool", True,
                "Daemon can upvote posts/comments", "daemon"),
        Control("daemon_can_follow", "bool", False,
                "Daemon can follow (rare, off by default)", "daemon"),
        Control("daemon_can_subscribe", "bool", False,
                "Daemon can subscribe to submolts", "daemon"),
        Control("daemon_can_downvote", "bool", False,
                "Daemon can downvote (requires allow_downvote)", "daemon"),
    ]

    # Parse blacklist from CLI
    blacklist_raw = getattr(args, "blacklist_controls", "") or ""
    blacklist = {k.strip() for k in blacklist_raw.split(",") if k.strip()}

    return ControlRegistry(controls=controls, blacklist=blacklist)
