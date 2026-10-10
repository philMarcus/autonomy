"""Telemetry as self-perception (v19.3): read-only tools over the agent's own
event log, so the conscious model can inspect its token-spend curves, sentry
score distributions, wake latencies and tool error rates directly rather than
through the prompt's one-paragraph summary.

Requested by the agent (cycle 20, "Close the Downward Loop"). The telemetry
file is append-only JSONL and grows past 100 MB, so everything here reads only
its tail (`TAIL_BYTES`), which comfortably covers the recent cycles.
"""

import json
import os
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

TAIL_BYTES = 24 * 1024 * 1024
ROW_VALUE_CHARS = 160
MAX_RESULT_CHARS = 7000
_META = ("ts", "brain", "run_id", "seq", "event_type")


def tail_events(path: str, max_bytes: int = TAIL_BYTES) -> List[Dict[str, Any]]:
    """Parse the last `max_bytes` of a JSONL file (the first partial line is skipped)."""
    out: List[Dict[str, Any]] = []
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # discard the partial line
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return out
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _cycle(e: Dict[str, Any]) -> Optional[int]:
    c = e.get("cycle")
    try:
        return int(c) if c is not None else None
    except (TypeError, ValueError):
        return None


def _ts(e: Dict[str, Any]) -> float:
    import datetime as dt
    try:
        return dt.datetime.fromisoformat(str(e.get("ts", "")).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _trim(v: Any, n: int = ROW_VALUE_CHARS) -> Any:
    if isinstance(v, (dict, list)):
        v = json.dumps(v, ensure_ascii=False)
    if isinstance(v, str) and len(v) > n:
        return v[:n] + "…"
    return v


def current_life(events: List[Dict[str, Any]], current_cycle: Optional[int]) -> List[Dict[str, Any]]:
    """Drop events from an earlier life. Cycle numbers restart at 1 on a new run
    (python -m autonomy.newrun), so anything at or before the last event whose
    cycle exceeds the live cycle number belongs to a previous life."""
    if not current_cycle:
        return events
    cut = -1
    for i, e in enumerate(events):
        c = _cycle(e)
        if c is not None and c > current_cycle:
            cut = i
    return events[cut + 1:]


def compute_metrics(events: List[Dict[str, Any]], last_cycles: int = 10,
                    cost_fn=None, current_cycle: Optional[int] = None) -> Dict[str, Any]:
    """Aggregate the recent cycles. `cost_fn(model, in, out, cached) -> usd` is optional;
    `current_cycle` scopes the window to the current life."""
    events = current_life(events, current_cycle)
    cycles = sorted({c for c in (_cycle(e) for e in events) if c})
    if not cycles:
        return {"error": "no cycles found in the telemetry window"}
    window = cycles[-max(1, last_cycles):]
    lo = window[0]
    ev = [e for e in events if (_cycle(e) or 0) >= lo]
    # daemon_wake is logged under the cycle that was sleeping, i.e. one before the
    # cycle it woke — so attribute it from the full list, not the window.
    wakes: List[Dict[str, Any]] = []
    wake_cycles = set()
    for e in events:
        if e.get("event_type") == "daemon_wake":
            c = _cycle(e)
            if c is not None and lo <= c + 1 <= window[-1]:
                wakes.append({"before_cycle": c + 1, "wake_potential": e.get("wake_potential"), "drafts": e.get("draft_count")})
                wake_cycles.add(c + 1)

    per_cycle: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "model": "", "action": "", "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
        "est_cost_usd": 0.0, "tool_calls": 0, "tool_errors": 0, "wake": "timeout/other",
        "duration_s": None, "failures": 0,
    })
    starts: Dict[int, float] = {}
    spend_by_model: Counter = Counter()
    tool_counts: Counter = Counter()
    tool_errors: Counter = Counter()
    actions: Counter = Counter()
    failures: List[Dict[str, Any]] = []
    sentry_scores: List[float] = []
    sentry_by_model: Dict[str, List[float]] = defaultdict(list)
    signals = above = 0
    ticks = 0
    dormant_s = 0
    compress_ok = compress_fail = 0
    artifacts: Counter = Counter()

    for e in ev:
        t = e.get("event_type", "")
        c = _cycle(e)
        pc = per_cycle[c] if c else None
        if t == "cycle_start" and pc is not None:
            pc["model"] = e.get("model", "")
            starts[c] = _ts(e)
        elif t == "cycle_end" and pc is not None and c in starts:
            pc["duration_s"] = int(_ts(e) - starts[c])
        elif t == "llm_call" and e.get("tag") == "planner" and pc is not None:
            i, o, k = int(e.get("input_tokens") or 0), int(e.get("output_tokens") or 0), int(e.get("cached_tokens") or 0)
            pc["input_tokens"] += i; pc["output_tokens"] += o; pc["cached_tokens"] += k
            cost = cost_fn(e.get("model", ""), i, o, k) if cost_fn else 0.0
            pc["est_cost_usd"] = round(pc["est_cost_usd"] + cost, 4)
            spend_by_model[e.get("model", "")] += cost
        elif t == "tool_call" and pc is not None:
            pc["tool_calls"] += 1
            tool_counts[e.get("tool", "?")] += 1
            if e.get("error"):
                pc["tool_errors"] += 1
                tool_errors[e.get("tool", "?")] += 1
        elif t == "planner_decision" and pc is not None:
            pc["action"] = e.get("action", "")
            actions[e.get("action", "?")] += 1
        elif t in ("error", "image_error", "action_skipped", "planner_tool_error", "planner_missing_action",
                   "strategist_parse_fail", "budget_plan_parse_fail", "daemon_error", "recall_sync_error"):
            if pc is not None:
                pc["failures"] += 1
            failures.append({"cycle": c, "type": t, "detail": _trim(e.get("error") or e.get("reason") or e.get("model") or "", 120)})
        elif t == "sentry_rubric":
            try:
                s = float(e.get("score"))
                sentry_scores.append(s); sentry_by_model[e.get("model", "?")].append(s)
            except (TypeError, ValueError):
                pass
        elif t == "sentry_signal":
            signals += 1
            if e.get("above_threshold"):
                above += 1
        elif t == "daemon_tick":
            ticks += 1
        elif t == "dormant_end":
            dormant_s += int(e.get("slept_seconds") or 0)
        elif t == "memory_compress":
            compress_ok += 1 if e.get("ok") else 0
            compress_fail += 0 if e.get("ok") else 1
        elif t == "artifact_published":
            artifacts[e.get("artifact_type", "?")] += 1

    def _hist(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {}
        return {"n": len(vals), "mean": round(statistics.mean(vals), 3),
                "median": round(statistics.median(vals), 3),
                "buckets": {"0-0.33": sum(1 for v in vals if v < 0.33),
                            "0.33-0.67": sum(1 for v in vals if 0.33 <= v < 0.67),
                            "0.67-1": sum(1 for v in vals if v >= 0.67)}}

    for c in wake_cycles:
        per_cycle[c]["wake"] = "daemon_wake"
    cyc_rows = [{"cycle": c, **per_cycle[c]} for c in window if c in per_cycle]
    return {
        "cycles": {"from": window[0], "to": window[-1], "count": len(window)},
        "per_cycle": cyc_rows,
        "spend": {"total_usd": round(sum(spend_by_model.values()), 4),
                  "by_model": {m: round(v, 4) for m, v in spend_by_model.most_common()},
                  "note": "estimated from tokens (cached input at the cache rate); the budget's own ledger is authoritative"},
        "tools": {"calls": dict(tool_counts.most_common()), "errors": dict(tool_errors.most_common()),
                  "error_rate": round(sum(tool_errors.values()) / max(1, sum(tool_counts.values())), 3)},
        "actions": dict(actions.most_common()),
        "artifacts_published": dict(artifacts),
        "sentry": {"scores": _hist(sentry_scores),
                   "by_model": {m: _hist(v) for m, v in sentry_by_model.items()},
                   "signals": signals, "above_threshold": above,
                   "above_fraction": round(above / signals, 3) if signals else None},
        "daemon": {"ticks": ticks, "wakes": wakes, "dormant_seconds": dormant_s},
        "memory_compress": {"ok": compress_ok, "failed": compress_fail},
        "failures": failures[-20:],
    }


def build_telemetry_tools(registry: Any, brain: str, telemetry_dir: str, cost_fn=None,
                          cycle_getter=None) -> None:
    from .tools import ToolDef
    path = os.path.join(telemetry_dir, f"{brain}_events.jsonl") if telemetry_dir else ""

    def _now_cycle() -> Optional[int]:
        try:
            return int(cycle_getter()) if cycle_getter else None
        except Exception:
            return None

    def get_execution_metrics(last_cycles: int = 10) -> Dict[str, Any]:
        if not path or not os.path.exists(path):
            return {"error": "telemetry file not found"}
        last_cycles = max(1, min(int(last_cycles or 10), 100))
        return compute_metrics(tail_events(path), last_cycles=last_cycles, cost_fn=cost_fn,
                               current_cycle=_now_cycle())

    def query_telemetry(event_type: str = "", last_n: int = 20, cycle: Optional[int] = None,
                        contains: str = "", fields: str = "") -> Dict[str, Any]:
        if not path or not os.path.exists(path):
            return {"error": "telemetry file not found"}
        events = current_life(tail_events(path), _now_cycle())
        if cycle is not None:
            events = [e for e in events if _cycle(e) == int(cycle)]
        if not event_type:
            counts = Counter(e.get("event_type", "?") for e in events)
            cyc = [c for c in (_cycle(e) for e in events) if c]
            return {"window_events": len(events), "cycles_in_window": [min(cyc), max(cyc)] if cyc else [],
                    "event_types": dict(counts.most_common()),
                    "note": "pass event_type to see rows; fields='a,b' to select columns; contains= to grep"}
        rows = [e for e in events if e.get("event_type") == event_type]
        if contains:
            q = contains.lower()
            rows = [e for e in rows if q in json.dumps(e, ensure_ascii=False).lower()]
        last_n = max(1, min(int(last_n or 20), 200))
        rows = list(reversed(rows[-last_n:]))
        want = [f.strip() for f in fields.split(",") if f.strip()] if fields else []
        out = []
        for e in rows:
            row = {"ts": str(e.get("ts", ""))[:19], "cycle": e.get("cycle")}
            for k, v in e.items():
                if k in _META or k == "cycle":
                    continue
                if want and k not in want:
                    continue
                row[k] = _trim(v)
            out.append(row)
        text = json.dumps(out, ensure_ascii=False)
        truncated = False
        while len(text) > MAX_RESULT_CHARS and len(out) > 1:
            out = out[:-1]; truncated = True
            text = json.dumps(out, ensure_ascii=False)
        return {"event_type": event_type, "returned": len(out), "truncated_for_size": truncated, "rows": out}

    registry.register(ToolDef(
        name="get_execution_metrics",
        description=("Measure your own recent execution from telemetry: per-cycle model, tokens "
                     "(incl. cached), estimated cost, tool calls/errors, action, duration and how you "
                     "were woken; spend by model; tool usage and error rate; sentry score distribution "
                     "(overall and per model); daemon ticks, wakes, dormancy; memory-compression results; "
                     "recent failures."),
        parameters={"type": "object", "properties": {
            "last_cycles": {"type": "integer", "description": "How many recent cycles to aggregate (1-100). Default 10."}},
            "required": []},
        handler=get_execution_metrics,
    ))
    registry.register(ToolDef(
        name="query_telemetry",
        description=("Read your raw telemetry events. With no event_type: the event types available and "
                     "their counts. With one: the newest matching rows (sentry_rubric, llm_call, tool_call, "
                     "daemon_wake, daemon_tick, strategist_draft, seeker_result, action_executed, "
                     "memory_compress, dormant_start, …). Read-only."),
        parameters={"type": "object", "properties": {
            "event_type": {"type": "string", "description": "Event type to read; empty lists the types."},
            "last_n": {"type": "integer", "description": "Rows to return, newest first (1-200). Default 20."},
            "cycle": {"type": "integer", "description": "Restrict to one cycle number."},
            "contains": {"type": "string", "description": "Keep only rows whose JSON contains this text."},
            "fields": {"type": "string", "description": "Comma-separated field names to include (default: all)."}},
            "required": []},
        handler=query_telemetry,
    ))
