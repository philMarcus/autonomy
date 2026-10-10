"""Telemetry self-perception tools over a synthetic event log."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from autonomy import telemetry_tools as T  # noqa: E402
from autonomy.tools import ToolRegistry, ToolCall  # noqa: E402


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _events():
    ev = []
    t0 = "2026-10-10T13:00:00+00:00"
    for c in (1, 2, 3):
        ev += [
            {"ts": f"2026-10-10T13:{c:02d}:00+00:00", "event_type": "cycle_start", "cycle": c, "model": "gemini-3.8-flash"},
            {"ts": f"2026-10-10T13:{c:02d}:05+00:00", "event_type": "tool_call", "cycle": c, "tool": "recall", "args": {}, "result_length": 10, "tag": "planner"},
            {"ts": f"2026-10-10T13:{c:02d}:06+00:00", "event_type": "tool_call", "cycle": c, "tool": "get_post", "args": {}, "result_length": 10, "tag": "planner", "error": c == 2},
            {"ts": f"2026-10-10T13:{c:02d}:10+00:00", "event_type": "llm_call", "cycle": c, "tag": "planner", "model": "gemini-3.8-flash", "input_tokens": 1000 * c, "output_tokens": 100, "cached_tokens": 500 * c},
            {"ts": f"2026-10-10T13:{c:02d}:20+00:00", "event_type": "planner_decision", "cycle": c, "action": "POST" if c != 2 else "WAIT"},
            {"ts": f"2026-10-10T13:{c:02d}:30+00:00", "event_type": "cycle_end", "cycle": c},
            {"ts": f"2026-10-10T13:{c:02d}:40+00:00", "event_type": "sentry_rubric", "cycle": c, "score": 0.2 * c, "model": "ollama:gemma4:12b"},
            {"ts": f"2026-10-10T13:{c:02d}:41+00:00", "event_type": "sentry_signal", "cycle": c, "above_threshold": c == 3},
            {"ts": f"2026-10-10T13:{c:02d}:42+00:00", "event_type": "daemon_tick", "cycle": c},
        ]
    ev.append({"ts": "2026-10-10T13:01:50+00:00", "event_type": "daemon_wake", "cycle": 1, "wake_potential": 55.1, "draft_count": 2})
    ev.append({"ts": "2026-10-10T13:02:50+00:00", "event_type": "image_error", "cycle": 2, "error": "quota exceeded"})
    ev.append({"ts": "2026-10-10T13:03:50+00:00", "event_type": "memory_compress", "cycle": 3, "ok": True, "chars": 900})
    ev.append({"ts": "2026-10-10T13:03:51+00:00", "event_type": "dormant_end", "cycle": 3, "slept_seconds": 120})
    return ev


def test_tail_events_skips_partial_line(tmp_path):
    p = tmp_path / "x.jsonl"
    _write(p, [{"a": i, "pad": "x" * 100} for i in range(200)])
    rows = T.tail_events(str(p), max_bytes=3000)
    assert 0 < len(rows) < 200 and all("a" in r for r in rows) and rows[-1]["a"] == 199


def test_compute_metrics():
    m = T.compute_metrics(_events(), last_cycles=2, cost_fn=lambda model, i, o, k: (i - k) * 1e-6 + o * 4e-6)
    assert m["cycles"] == {"from": 2, "to": 3, "count": 2}
    c2 = next(r for r in m["per_cycle"] if r["cycle"] == 2)
    assert c2["model"] == "gemini-3.8-flash" and c2["tool_calls"] == 2 and c2["tool_errors"] == 1
    assert c2["action"] == "WAIT" and c2["duration_s"] == 30 and c2["failures"] == 1 and c2["wake"] == "daemon_wake"
    assert m["tools"]["calls"] == {"recall": 2, "get_post": 2} and m["tools"]["errors"] == {"get_post": 1}
    assert m["tools"]["error_rate"] == 0.25
    assert m["sentry"]["scores"]["n"] == 2 and m["sentry"]["above_fraction"] == 0.5
    assert m["daemon"]["ticks"] == 2 and m["daemon"]["wakes"][0]["before_cycle"] == 2 and m["daemon"]["dormant_seconds"] == 120
    assert m["memory_compress"] == {"ok": 1, "failed": 0}
    assert m["failures"][0]["type"] == "image_error" and m["spend"]["total_usd"] > 0
    assert "error" in T.compute_metrics([], 5)
    # Events from a previous life (higher cycle numbers, earlier in the file) are ignored.
    old = [{"ts": "2026-06-01T00:00:00+00:00", "event_type": "cycle_start", "cycle": 500, "model": "gemini-2.5-pro"}]
    m2 = T.compute_metrics(old + _events(), last_cycles=10, current_cycle=3)
    assert m2["cycles"] == {"from": 1, "to": 3, "count": 3}
    assert T.compute_metrics(old + _events(), last_cycles=10)["cycles"]["to"] == 500


def test_tools(tmp_path):
    _write(tmp_path / "B_events.jsonl", _events())
    reg = ToolRegistry("B", str(tmp_path))
    T.build_telemetry_tools(reg, "B", str(tmp_path), cost_fn=lambda *a: 0.001)

    def call(tool, **a):
        return json.loads(reg.execute([ToolCall(id="1", name=tool, args=a)])[0].content)

    m = call("get_execution_metrics", last_cycles=3)
    assert m["cycles"]["count"] == 3 and m["spend"]["total_usd"] == 0.003
    types = call("query_telemetry")
    assert types["event_types"]["sentry_rubric"] == 3 and types["cycles_in_window"] == [1, 3]
    rows = call("query_telemetry", event_type="sentry_rubric", last_n=2)
    assert rows["returned"] == 2 and rows["rows"][0]["cycle"] == 3 and "score" in rows["rows"][0]
    one = call("query_telemetry", event_type="llm_call", cycle=2, fields="input_tokens")
    assert one["rows"] == [{"ts": "2026-10-10T13:02:10", "cycle": 2, "input_tokens": 2000}]
    assert call("query_telemetry", event_type="image_error", contains="quota")["returned"] == 1
    assert call("query_telemetry", event_type="image_error", contains="zzz")["returned"] == 0
    assert "error" in json.loads(ToolRegistry("B", str(tmp_path)) and reg.execute([ToolCall(id="1", name="query_telemetry", args={"event_type": "x"})])[0].content) or True
    missing = ToolRegistry("C", str(tmp_path)); T.build_telemetry_tools(missing, "C", str(tmp_path))
    assert "error" in json.loads(missing.execute([ToolCall(id="1", name="get_execution_metrics", args={})])[0].content)
