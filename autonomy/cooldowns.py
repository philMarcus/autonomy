"""Unified cooldown system for all social actions.

State format:
    state["cooldowns"] = {"POST": next_allowed_epoch, "COMMENT": ..., ...}

All actions check/set cooldowns through can_do() and set_cooldown(),
replacing the old ad-hoc can_post()/can_comment()/set_post_cooldown()/
set_comment_cooldown() functions.
"""

import time
from typing import Any, Dict, Optional, Tuple

# ============================================================
# Default cooldowns (seconds) — overridable via ControlRegistry
# ============================================================
DEFAULT_COOLDOWNS: Dict[str, int] = {
    "POST": 1800,               # 30m — Moltbook rate limit
    "COMMENT": 180,             # 3m — behavioral throttle (API limit is 20s)
    "REPLY": 180,               # same as COMMENT
    "UPVOTE_POST": 60,          # prevent spam-voting
    "DOWNVOTE_POST": 60,
    "UPVOTE_COMMENT": 30,
    "DOWNVOTE_COMMENT": 30,
    "FOLLOW": 3600,             # 1h — be VERY selective
    "UNFOLLOW": 3600,
    "SUBSCRIBE_SUBMOLT": 300,   # 5m
    "CREATE_SUBMOLT": 3600,     # 1h
    "DM": 600,                  # 10m
    "GENERATE_IMAGE": 86400,    # 24h — overridden by image_cooldown_hours
}

# Map from ControlRegistry keys to action names they override
_CONTROL_OVERRIDES: Dict[str, str] = {
    "cooldown_upvote_seconds": "UPVOTE_POST",
    # UPVOTE_COMMENT uses same control (half the value)
    "cooldown_follow_seconds": "FOLLOW",
    "cooldown_subscribe_seconds": "SUBSCRIBE_SUBMOLT",
    "cooldown_dm_seconds": "DM",
    "cooldown_create_submolt_seconds": "CREATE_SUBMOLT",
}


def _ensure_bucket(state: Dict[str, Any]) -> Dict[str, float]:
    """Get or create the cooldowns dict in state."""
    if "cooldowns" not in state:
        state["cooldowns"] = {}
    return state["cooldowns"]


def can_do(
    state: Dict[str, Any],
    action: str,
    ctrl: Optional[Any] = None,
) -> Tuple[bool, int]:
    """Check whether an action is off cooldown.

    Returns (allowed, seconds_remaining).
    """
    cds = _ensure_bucket(state)
    now = time.time()
    next_t = float(cds.get(action, 0))
    if now >= next_t:
        return True, 0
    return False, max(0, int(next_t - now))


def set_cooldown(
    state: Dict[str, Any],
    action: str,
    seconds: int = 0,
    ctrl: Optional[Any] = None,
) -> None:
    """Set a cooldown for an action.

    If seconds == 0, uses ctrl override or DEFAULT_COOLDOWNS.
    Cooldown is always at least as far in the future as any existing cooldown
    (prevents shortening by re-calling).
    """
    if seconds <= 0:
        seconds = _resolve_default(action, ctrl)
    cds = _ensure_bucket(state)
    new_t = time.time() + seconds
    cds[action] = max(float(cds.get(action, 0)), new_t)


def _resolve_default(action: str, ctrl: Optional[Any]) -> int:
    """Look up cooldown: ctrl override first, then DEFAULT_COOLDOWNS."""
    if ctrl is not None:
        # Direct per-action control overrides
        if action == "POST":
            try:
                return int(ctrl.get("post_interval_minutes") * 60)
            except Exception:
                pass
        elif action in ("COMMENT", "REPLY"):
            try:
                return int(ctrl.get("cooldown_comment_seconds"))
            except Exception:
                pass
        elif action in ("UPVOTE_POST", "DOWNVOTE_POST"):
            try:
                return int(ctrl.get("cooldown_upvote_seconds"))
            except Exception:
                pass
        elif action in ("UPVOTE_COMMENT", "DOWNVOTE_COMMENT"):
            try:
                return max(10, int(ctrl.get("cooldown_upvote_seconds") // 2))
            except Exception:
                pass
        elif action in ("FOLLOW", "UNFOLLOW"):
            try:
                return int(ctrl.get("cooldown_follow_seconds"))
            except Exception:
                pass
        elif action == "SUBSCRIBE_SUBMOLT":
            try:
                return int(ctrl.get("cooldown_subscribe_seconds"))
            except Exception:
                pass
        elif action == "DM":
            try:
                return int(ctrl.get("cooldown_dm_seconds"))
            except Exception:
                pass
        elif action == "CREATE_SUBMOLT":
            try:
                return int(ctrl.get("cooldown_create_submolt_seconds"))
            except Exception:
                pass
        elif action == "GENERATE_IMAGE":
            try:
                return int(ctrl.get("image_cooldown_minutes") * 60)
            except Exception:
                pass
    return DEFAULT_COOLDOWNS.get(action, 60)


def cooldown_status_text(
    state: Dict[str, Any],
    ctrl: Optional[Any] = None,
) -> str:
    """Build a status block for the planner prompt showing each action's readiness."""
    actions = [
        "POST", "COMMENT", "REPLY",
        "UPVOTE_POST", "DOWNVOTE_POST",
        "UPVOTE_COMMENT", "DOWNVOTE_COMMENT",
        "FOLLOW", "UNFOLLOW",
        "SUBSCRIBE_SUBMOLT", "CREATE_SUBMOLT",
        "DM", "GENERATE_IMAGE",
    ]
    lines = []
    for action in actions:
        ok, remaining = can_do(state, action, ctrl=ctrl)
        if ok:
            lines.append(f"  {action}: READY")
        else:
            m, s = divmod(remaining, 60)
            if m > 0:
                lines.append(f"  {action}: cooldown {m}m {s}s")
            else:
                lines.append(f"  {action}: cooldown {s}s")
    return "\n".join(lines)


def migrate_legacy_cooldowns(state: Dict[str, Any]) -> bool:
    """Migrate old next_post_time/next_comment_time into unified cooldowns.

    Returns True if migration happened.
    """
    migrated = False
    cds = _ensure_bucket(state)

    if "next_post_time" in state:
        cds.setdefault("POST", float(state.pop("next_post_time")))
        migrated = True
    if "next_comment_time" in state:
        cds.setdefault("COMMENT", float(state.pop("next_comment_time")))
        migrated = True

    return migrated
