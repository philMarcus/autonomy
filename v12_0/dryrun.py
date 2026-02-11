"""Per-brain dry-run output logger. Writes human-readable .txt logs."""

import os
import datetime
from typing import Any, Dict, List


class DryRunLogger:
    SEPARATOR = "=" * 64

    def __init__(self, brain_name: str, base_dir: str = "brains"):
        self.brain_name = brain_name
        self.path = os.path.join(base_dir, f"{brain_name}_dryrun.txt")
        self._social_actions: List[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ts(self) -> str:
        return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _append(self, text: str) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

    def _indent(self, text: str, prefix: str = "  ") -> str:
        return "\n".join(prefix + line for line in text.split("\n"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def cycle_start(self, cycle: int) -> None:
        """Write cycle header. Call at the top of each cycle."""
        self._social_actions = []
        self._append(
            f"\n{self.SEPARATOR}\n"
            f"CYCLE {cycle} | {self._ts()}\n"
            f"{self.SEPARATOR}\n"
        )

    def social_action(self, action: str, **kwargs: Any) -> None:
        """Buffer a social action line (flushed later as a group)."""
        self._social_actions.append(self._format_social(action, kwargs))

    def flush_social_actions(self) -> None:
        """Write buffered social actions section."""
        if self._social_actions:
            self._append("\n--- SOCIAL ACTIONS ---\n")
            for line in self._social_actions:
                self._append(line + "\n")
            self._social_actions = []

    def feed(self, feed_brief: str) -> None:
        """Write the feed section."""
        self._append("\n--- FEED ---\n")
        self._append(feed_brief + "\n")

    def planner_output(self, plan: Dict[str, Any]) -> None:
        """Write the planner output section."""
        action = (plan.get("action") or "?").upper()
        self._append(f"\n--- PLANNER OUTPUT ---\n")
        self._append(f"Action: {action}\n")

        if action == "POST":
            self._append(f"Submolt: {plan.get('submolt', 'general')}\n")
            self._append(f"Title: {plan.get('title', '')}\n")
            content = plan.get("content", "")
            if content:
                self._append(f"Content:\n{self._indent(content)}\n")
        elif action == "COMMENT":
            self._append(f"Post: {plan.get('post_id', '')}\n")
            content = plan.get("content", "")
            if content:
                self._append(f"Content:\n{self._indent(content)}\n")
        elif action == "REPLY":
            self._append(f"Post: {plan.get('post_id', '')}\n")
            self._append(f"Parent comment: {plan.get('parent_comment_id', '')}\n")
            content = plan.get("content", "")
            if content:
                self._append(f"Content:\n{self._indent(content)}\n")
        elif action in ("UPVOTE_POST", "DOWNVOTE_POST"):
            self._append(f"Post: {plan.get('post_id', '')}\n")
        elif action == "UPVOTE_COMMENT":
            self._append(f"Comment: {plan.get('comment_id', '')}\n")
        elif action == "CREATE_SUBMOLT":
            self._append(f"Name: {plan.get('name', '')}\n")
            self._append(f"Display name: {plan.get('display_name', '')}\n")
            self._append(f"Description: {plan.get('description', '')}\n")
        elif action == "SUBSCRIBE_SUBMOLT":
            self._append(f"Submolt: {plan.get('name', '')}\n")
        elif action == "WAIT":
            pass  # summary below is enough

        summary = plan.get("summary", "")
        if summary:
            self._append(f"Summary: {summary}\n")

    def kernel_update(self, reason: str, new_kernel: str) -> None:
        """Write the kernel update proposal section."""
        self._append(f"\n--- KERNEL UPDATE PROPOSAL ---\n")
        self._append(f"Reason: {reason}\n")
        self._append(f"New kernel:\n{self._indent(new_kernel)}\n")

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def _format_social(self, action: str, kwargs: Dict[str, Any]) -> str:
        a = (action or "").upper()
        if a == "UPVOTE_POST":
            return f"[UPVOTE] post {kwargs.get('post_id', '?')}"
        if a == "SUBSCRIBE_SUBMOLT":
            return f"[SUBSCRIBE] /m/{kwargs.get('submolt', kwargs.get('name', '?'))}"
        if a == "FOLLOW":
            return f"[FOLLOW] @{kwargs.get('author', '?')}"
        if a == "CREATE_SUBMOLT":
            name = kwargs.get("name", "?")
            display = kwargs.get("display_name", "")
            desc = kwargs.get("description", "")
            return f"[CREATE SUBMOLT] /m/{name} ({display}) - {desc}"
        if a == "DM":
            return f"[DM] to @{kwargs.get('to', '?')}: {str(kwargs.get('message', ''))[:80]}"
        details = " ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"[{a}] {details}"
