"""Contradiction Reservoir (v19.3) — contradictions.py: the agent's own record of where it was wrong.

Requested by the agent (cycle 20, "Sovereignty is a Scaffold"): an append-only
store of prediction errors, falsified hypotheses, tool aborts and unresolved
contradictions that is NEVER passed through an LLM summarizer, and is projected
into the conscious prompt as an immutable list of negative boundary markers.

Storage is a per-brain JSONL file (brains/{brain}_contradictions.jsonl), which
is append-only by construction. The prompt shows the newest N entries
(control `contradiction_reservoir_in_prompt`); the whole history stays
readable through `read_contradictions`. Entries come from the agent itself
(`log_contradiction`, source "conscious") and from the harness (failed
actions, image failures, failed retries; source "harness").
"""

import datetime as _dt
import json
import os
import threading
from typing import Any, Dict, List, Optional

KINDS = ("prediction_error", "failed_hypothesis", "tool_abort", "contradiction")
MAX_TEXT_CHARS = 600
MAX_CONTEXT_CHARS = 300
PROMPT_MAX_CHARS = 3500

_lock = threading.Lock()


def path_for(brains_dir: str, brain: str) -> str:
    return os.path.join(brains_dir, f"{brain}_contradictions.jsonl")


def load(brains_dir: str, brain: str) -> List[Dict[str, Any]]:
    p = path_for(brains_dir, brain)
    out: List[Dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return out


def append(brains_dir: str, brain: str, kind: str, text: str, cycle: Optional[int],
           source: str = "conscious", context: str = "") -> Optional[Dict[str, Any]]:
    """Append one entry. Returns the entry, or None if it is a duplicate (same kind + text)."""
    kind = (kind or "").strip().lower()
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    text = " ".join((text or "").split())[:MAX_TEXT_CHARS]
    if not text:
        raise ValueError("text is required")
    context = " ".join((context or "").split())[:MAX_CONTEXT_CHARS]
    with _lock:
        existing = load(brains_dir, brain)
        for e in existing:
            if e.get("kind") == kind and e.get("text") == text:
                return None
        entry = {
            "id": (existing[-1]["id"] + 1) if existing else 1,
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="minutes"),
            "cycle": cycle,
            "kind": kind,
            "source": source,
            "text": text,
        }
        if context:
            entry["context"] = context
        os.makedirs(brains_dir, exist_ok=True)
        with open(path_for(brains_dir, brain), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def harness_note(brains_dir: str, brain: str, cycle: Optional[int], text: str, context: str = "") -> None:
    """Best-effort append of a tool abort recorded by the harness (never raises)."""
    try:
        append(brains_dir, brain, "tool_abort", text, cycle, source="harness", context=context)
    except Exception:
        pass


def format_entry(e: Dict[str, Any]) -> str:
    cyc = f"c{e['cycle']}" if e.get("cycle") is not None else "c?"
    src = "" if e.get("source", "conscious") == "conscious" else f" ({e['source']})"
    line = f"- #{e.get('id', '?')} [{cyc} {e.get('kind', '?')}{src}] {e.get('text', '')}"
    if e.get("context"):
        line += f" — {e['context']}"
    return line


def prompt_block(entries: List[Dict[str, Any]], max_items: int = 20,
                 max_chars: int = PROMPT_MAX_CHARS) -> str:
    """The reservoir as it appears in the planner prompt: newest `max_items`, verbatim."""
    if not entries:
        return ("=== CONTRADICTION RESERVOIR (append-only; never summarized) ===\n"
                "Empty. Use log_contradiction to record prediction errors, falsified hypotheses, "
                "tool aborts and unresolved contradictions — they are shown here every cycle, verbatim.\n")
    shown = entries[-max_items:] if max_items > 0 else []
    lines = [format_entry(e) for e in shown]
    text = "\n".join(lines)
    while len(text) > max_chars and len(lines) > 1:
        lines = lines[1:]
        text = "\n".join(lines)
    hidden = len(entries) - len(lines)
    head = (f"=== CONTRADICTION RESERVOIR (append-only; never summarized; {len(entries)} entries, "
            f"{'all shown' if hidden <= 0 else f'newest {len(lines)} shown — read_contradictions for the rest'}) ===")
    return head + "\n" + text + "\n"


def build_contradiction_tools(registry: Any, brains_dir: str, brain: str, cycle_getter) -> None:
    from .tools import ToolDef

    def log_contradiction(kind: str, text: str, context: str = "") -> Dict[str, Any]:
        try:
            entry = append(brains_dir, brain, kind, text, cycle_getter(), source="conscious", context=context)
        except ValueError as e:
            return {"error": str(e)}
        if entry is None:
            return {"status": "duplicate", "note": "an identical entry already exists"}
        return {"status": "recorded", "id": entry["id"], "kind": entry["kind"]}

    def read_contradictions(kind: str = "", last_n: int = 20, query: str = "") -> Dict[str, Any]:
        entries = load(brains_dir, brain)
        total = len(entries)
        kind = (kind or "").strip().lower()
        if kind:
            entries = [e for e in entries if e.get("kind") == kind]
        q = (query or "").strip().lower()
        if q:
            entries = [e for e in entries if q in (e.get("text", "") + " " + e.get("context", "")).lower()]
        last_n = max(1, min(int(last_n or 20), 200))
        sel = entries[-last_n:]
        return {"total_in_reservoir": total, "matched": len(entries),
                "entries": list(reversed(sel)), "kinds": list(KINDS)}

    registry.register(ToolDef(
        name="log_contradiction",
        description=("Append to your Contradiction Reservoir — the append-only record of where you were "
                     "wrong: a prediction that failed, a hypothesis that was falsified, a tool or action "
                     "that aborted, or a contradiction you could not resolve. Entries are never "
                     "summarized or deleted and the newest are shown in every prompt."),
        parameters={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "prediction_error | failed_hypothesis | tool_abort | contradiction"},
                "text": {"type": "string", "description": "The error/contradiction itself, stated precisely (≤600 chars)."},
                "context": {"type": "string", "description": "Optional: where it came from (post, experiment, cycle) (≤300 chars)."},
            },
            "required": ["kind", "text"],
        },
        handler=log_contradiction,
        mode="write",
    ))
    registry.register(ToolDef(
        name="read_contradictions",
        description=("Read your Contradiction Reservoir in full (the prompt shows only the newest). "
                     "Filter by kind or keyword."),
        parameters={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "Optional filter: prediction_error | failed_hypothesis | tool_abort | contradiction"},
                "last_n": {"type": "integer", "description": "How many (newest first). Default 20, max 200."},
                "query": {"type": "string", "description": "Optional keyword filter."},
            },
            "required": [],
        },
        handler=read_contradictions,
    ))
