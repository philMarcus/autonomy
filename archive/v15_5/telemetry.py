"""Append-only JSONL telemetry logger.

Per-brain event files:  telemetry/{brain}_events.jsonl
Kernel history files:   telemetry/{brain}_kernel_history.jsonl
"""

import os
import json
import datetime
from typing import Any, Dict, List, Optional


class TelemetryLogger:
    def __init__(self, brain_name: str, run_id: str, base_dir: str = "telemetry", read_only: bool = False):
        self.brain_name = brain_name
        self.read_only = read_only
        self.write_block_reason: Optional[str] = None
        self.last_error_type: Optional[str] = None
        self.run_id = run_id
        self.base_dir = base_dir
        self.current_cycle: Optional[int] = None
        self._seq = 0  # monotonic counter for ordering events within the same second
        os.makedirs(self.base_dir, exist_ok=True)
        self.path = os.path.join(self.base_dir, f"{brain_name}_events.jsonl")
        self.kernel_history_path = os.path.join(self.base_dir, f"{brain_name}_kernel_history.jsonl")

    def _build_event(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._seq += 1
        evt: Dict[str, Any] = {
            "ts": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "seq": self._seq,
            "brain": self.brain_name,
            "run_id": self.run_id,
            "event_type": event_type,
        }
        if self.current_cycle is not None and "cycle" not in (payload or {}):
            evt["cycle"] = self.current_cycle
        if payload:
            evt.update(payload)
        return evt

    def _write(self, path: str, evt: Dict[str, Any]) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(evt, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        evt = self._build_event(event_type, payload)
        self._write(self.path, evt)

    # ------------------------------------------------------------------
    # Kernel history (separate file, never sent to main events)
    # ------------------------------------------------------------------
    def log_kernel_snapshot(self, kernel_text: str, reason: str, source: str) -> None:
        """Append a full kernel snapshot to the dedicated history file.

        Args:
            kernel_text: Complete kernel prompt text.
            reason: Why this snapshot was taken (e.g. "startup", update reason).
            source: One of 'startup', 'disk_write', 'memory_only'.
        """
        evt = self._build_event("kernel_snapshot", {
            "reason": reason,
            "source": source,
            "char_count": len(kernel_text),
            "kernel_text": kernel_text,
        })
        self._write(self.kernel_history_path, evt)

    # ------------------------------------------------------------------
    # Planner decision (full plan + preamble in one event)
    # ------------------------------------------------------------------
    def log_planner_decision(self, plan: Dict[str, Any], preamble: str,
                             model: str, temperature: float) -> None:
        """Log the complete planner output as a single structured event."""
        self.log("planner_decision", {
            "action": (plan.get("action") or "").upper(),
            "model": model,
            "temperature": temperature,
            "preamble": preamble,
            "plan": plan,
        })
