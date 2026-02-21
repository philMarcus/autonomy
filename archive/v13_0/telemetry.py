"""Append-only JSONL telemetry logger."""

import os
import json
import datetime
from typing import Any, Dict, Optional


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
        self.path = os.path.join(self.base_dir, "events.jsonl")

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        self._seq += 1
        evt = {
            "ts": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "seq": self._seq,
            "brain": self.brain_name,
            "run_id": self.run_id,
            "event_type": event_type,
        }
        # Auto-inject cycle if set and not already in payload
        if self.current_cycle is not None and "cycle" not in (payload or {}):
            evt["cycle"] = self.current_cycle
        if payload:
            evt.update(payload)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(evt, ensure_ascii=False) + "\n")
        except Exception:
            pass
