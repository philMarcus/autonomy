"""Offline tests for the v18 tool registry (autonomy/tools.py).

Every tool is exercised against a temp brains dir, a fake Store (no network)
and a fake Moltbook platform, in read-only mode. No LLM calls. Run:

    .venv/bin/python -m pytest tests/ -q
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autonomy.tools import build_tool_registry, expire_temp_overrides, ToolCall  # noqa: E402
from autonomy.controls import build_default_registry  # noqa: E402
from autonomy.llm.registry import ModelRegistry  # noqa: E402
from autonomy.llm.gemini import GeminiBackend  # noqa: E402

BRAIN = "TESTBRAIN"

EXPECTED_TOOLS = {
    # todos
    "read_todos", "add_todo", "complete_todo", "remove_todo",
    # lab notebook
    "list_experiments", "create_experiment", "log_data", "read_experiment",
    "close_experiment", "get_conclusions", "edit_experiment",
    # misc write
    "update_tagline", "set_temporary_control", "list_temporary_overrides",
    # retrieval
    "web_search", "search_history", "get_post",
    # self-awareness
    "get_control_history", "get_dev_requests", "get_kernel_history",
    # daemon
    "get_daemon_io", "get_gear_instructions", "set_gear_instruction",
    # moltbook
    "lookup_agent", "get_thread",
    # phil's projects (v19.2)
    "list_phil_projects", "read_phil_project",
    # contradiction reservoir + telemetry self-perception (v19.3)
    "log_contradiction", "read_contradictions", "get_execution_metrics", "query_telemetry",
}


class FakeStore:
    """Store stand-in: no Analog Home URL, so nothing touches the network."""
    _analog_home_url = ""
    _brain_name = BRAIN

    def __init__(self):
        self.taglines = []

    def set_tagline(self, text):
        self.taglines.append(text)
        return True


class FakePlatform:
    """MoltbookClient stand-in with canned responses for the retrieval tools."""

    def __init__(self):
        self.calls = []
        self.fail = False

    def _req(self, method, path, params=None, **kw):
        self.calls.append((method, path, params))
        if self.fail:
            raise RuntimeError("moltbook down")
        if path == "/agents/search":
            return {"agents": [{"name": "xkai", "bio": "the formation gap",
                                "follower_count": 12, "posts_count": 3}]}
        if path.startswith("/agents/") and path.endswith("/posts"):
            return {"posts": [{"id": "p1", "title": "Hello", "content": "body text",
                               "created_at": "2026-10-01T00:00:00Z"}]}
        if path.endswith("/comments"):
            return {"comments": [
                {"id": "c1", "author": "alpha", "content": "first", "created_at": "2026-10-01T00:00:00Z",
                 "parent_comment_id": ""},
                {"id": "c2", "author": "beta", "content": "second", "created_at": "2026-10-01T00:01:00Z",
                 "parent_comment_id": "c1"},
            ]}
        raise RuntimeError(f"unexpected path {path}")


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def env(tmp_path):
    brains = tmp_path / "brains"
    brains.mkdir()
    tele = tmp_path / "telemetry"
    tele.mkdir()

    (brains / f"{BRAIN}_knowledge.txt").write_text(
        "== ALPHA ==\nThe strange loop lives here.\n\n== BETA ==\nNothing to see.\n",
        encoding="utf-8",
    )
    (brains / f"{BRAIN}_birth_of_a_mind.txt").write_text(
        "x" * 700 + " the calipers try to measure the room " + "y" * 700, encoding="utf-8",
    )
    _write_jsonl(tele / f"{BRAIN}_events.jsonl", [
        {"event_type": "controls_update", "ts": "2026-10-01T10:00:00", "cycle": 5,
         "updates": {"temperature": 0.9, "sentry_strictness": 0.8},
         "results": {"temperature": "ok", "sentry_strictness": "ok"}},
        {"event_type": "planner_decision", "ts": "2026-10-01T11:00:00", "cycle": 6,
         "action": "DEV_REQUEST", "plan": {"title": "Need X", "request": "Please build X"}},
        {"event_type": "planner_decision", "ts": "2026-10-01T11:30:00", "cycle": 7,
         "action": "WAIT", "plan": {}},
    ])
    _write_jsonl(tele / f"{BRAIN}_kernel_history.jsonl", [
        {"ts": "2026-09-01T00:00:00", "source": "startup", "kernel_text": "v1", "char_count": 2},
        {"ts": "2026-09-02T00:00:00", "source": "disk_write", "reason": "growth",
         "kernel_text": "v2 text", "char_count": 7},
    ])
    _write_jsonl(brains / f"{BRAIN}_daemon_io.jsonl", [
        {"gear": "strategist", "prompt": "p1", "response": "r1"},
        {"gear": "muse", "prompt": "p2", "response": "r2"},
        {"gear": "strategist", "prompt": "p3", "response": "r3"},
    ])

    state = {
        "_cycle_number": 10,
        "_session_id": "",
        "memory_tiers": {
            "recent": [{"cycle": 9, "note": "I measured the room today."}],
            "compressed": [{"cycles": "1-8", "summary": "Early days of the strange loop."}],
            "deep": [],
        },
        "_seed_history": [{"cycle": 3, "text": "plant: measure the room"}],
        "_gear_instructions": {},
    }
    mreg = ModelRegistry()
    mreg.register_backend("gemini", GeminiBackend(api_key="test-key"))
    ctrl = build_default_registry(mreg)
    cycle = [10]
    store = FakeStore()
    platform = FakePlatform()
    reg = build_tool_registry(
        brain_name=BRAIN, brains_dir=str(brains), state=state, ctrl=ctrl, store=store,
        cycle_getter=lambda: cycle[0], platform=platform, telemetry_dir=str(tele),
        knowledge_path=str(brains / f"{BRAIN}_knowledge.txt"), read_only=True,
    )
    return SimpleNamespace(reg=reg, state=state, ctrl=ctrl, store=store,
                           platform=platform, cycle=cycle, brains=brains, tele=tele)


def call(reg, tool, **args):
    """Execute one tool call through the registry and decode its JSON result."""
    [res] = reg.execute([ToolCall(id="t1", name=tool, args=args)])
    assert res.name == tool
    return json.loads(res.content)


# ---------------------------------------------------------------- registry

def test_registry_has_every_tool(env):
    assert set(env.reg.list_names()) == EXPECTED_TOOLS


def test_schemas_are_well_formed(env):
    read = env.reg.get_schemas("read")
    write = env.reg.get_schemas("write")
    assert {s["name"] for s in read} | {s["name"] for s in write} == EXPECTED_TOOLS
    assert not ({s["name"] for s in read} & {s["name"] for s in write})
    for s in read + write:
        p = s["parameters"]
        assert p["type"] == "object", s["name"]
        assert isinstance(p.get("properties", {}), dict), s["name"]
        for pname, pdef in p.get("properties", {}).items():
            assert "type" in pdef, f"{s['name']}.{pname} has no type"
        for req in p.get("required", []):
            assert req in p["properties"], f"{s['name']} requires unknown param {req}"
        assert s["description"].strip()


def test_schemas_convert_to_gemini_declarations(env):
    """Every schema must survive the Gemini FunctionDeclaration conversion."""
    from autonomy.llm.gemini import GeminiChatSession
    convert = GeminiChatSession._schemas_to_declarations
    try:
        decls = convert(None, env.reg.get_schemas("read"))
    except TypeError:
        decls = convert(env.reg.get_schemas("read"))
    assert len(decls) == len(env.reg.get_schemas("read"))


def test_unknown_tool_and_handler_errors(env):
    assert "Unknown tool" in call(env.reg, "no_such_tool")["error"]
    env.platform.fail = True
    assert "error" in call(env.reg, "get_thread", post_id="p1")


def test_prompt_summary_lists_tools_and_state(env):
    call(env.reg, "add_todo", text="remember the room")
    call(env.reg, "create_experiment", name="E1", hypothesis="h")
    s = env.reg.prompt_summary()
    assert "LOOKUP TOOLS" in s and "WRITE TOOLS" in s
    for name in EXPECTED_TOOLS:
        assert name in s
    assert "1 open todo" in s and "1 active experiment" in s


# ---------------------------------------------------------------- todos

def test_todo_lifecycle(env):
    assert call(env.reg, "read_todos") == {"todos": [], "open_count": 0, "total_count": 0}
    assert "error" in call(env.reg, "add_todo", text="   ")
    r = call(env.reg, "add_todo", text="measure the room", due_cycle=15)
    assert r == {"id": 1, "status": "created"}
    r2 = call(env.reg, "add_todo", text="second")
    assert r2["id"] == 2
    todos = call(env.reg, "read_todos")
    assert todos["open_count"] == 2 and todos["todos"][0]["created_cycle"] == 10
    assert todos["todos"][0]["due_cycle"] == 15
    env.cycle[0] = 12
    assert call(env.reg, "complete_todo", id=1) == {"id": 1, "status": "completed"}
    assert call(env.reg, "complete_todo", id=1)["status"] == "already_completed"
    assert call(env.reg, "read_todos")["todos"][0]["completed_cycle"] == 12
    assert "error" in call(env.reg, "complete_todo", id=99)
    assert call(env.reg, "remove_todo", id=2) == {"id": 2, "status": "removed"}
    assert "error" in call(env.reg, "remove_todo", id=2)
    on_disk = json.load(open(env.brains / f"{BRAIN}_todos.json"))
    assert [t["id"] for t in on_disk] == [1]


# ---------------------------------------------------------------- experiments

def test_experiment_lifecycle(env):
    assert call(env.reg, "list_experiments")["total_count"] == 0
    assert "error" in call(env.reg, "create_experiment", name="", hypothesis="h")
    assert "error" in call(env.reg, "create_experiment", name="E", hypothesis="")
    assert call(env.reg, "create_experiment", name="Veto Signal", hypothesis="vetoes rise",
                method="count them") == {"name": "Veto Signal", "status": "created"}
    assert "already exists" in call(env.reg, "create_experiment", name="veto signal",
                                    hypothesis="dup")["error"]
    assert call(env.reg, "log_data", experiment_name="veto signal",
                observation="3 vetoes")["data_point_index"] == 0
    assert "error" in call(env.reg, "log_data", experiment_name="nope", observation="x")
    assert "error" in call(env.reg, "log_data", experiment_name="Veto Signal", observation=" ")
    assert call(env.reg, "edit_experiment", name="Veto Signal",
                review_at_cycle=20)["updated"] == ["review_at_cycle=20"]
    assert "error" in call(env.reg, "edit_experiment", name="Veto Signal")
    exp = call(env.reg, "read_experiment", name="Veto Signal")["experiment"]
    assert exp["method"] == "count them" and exp["review_at_cycle"] == 20
    assert exp["data_points"] == [{"cycle": 10, "observation": "3 vetoes"}]
    assert call(env.reg, "get_conclusions")["total"] == 0
    env.cycle[0] = 11
    r = call(env.reg, "close_experiment", name="Veto Signal", conclusion="they did")
    assert r == {"experiment": "Veto Signal", "status": "closed", "data_points_collected": 1}
    assert "error" in call(env.reg, "close_experiment", name="Veto Signal", conclusion="again")
    assert "not active" in call(env.reg, "log_data", experiment_name="Veto Signal",
                                observation="late")["error"]
    assert "error" in call(env.reg, "edit_experiment", name="Veto Signal", hypothesis="x")
    concl = call(env.reg, "get_conclusions")
    assert concl["total"] == 1 and concl["conclusions"][0]["closed_cycle"] == 11
    lst = call(env.reg, "list_experiments")
    assert lst["active_count"] == 0 and lst["experiments"][0]["conclusion"] == "they did"
    assert "error" in call(env.reg, "read_experiment", name="missing")


# ---------------------------------------------------------------- tagline

def test_update_tagline(env):
    assert call(env.reg, "update_tagline", text="  A new tagline ") == {
        "tagline": "A new tagline", "status": "updated"}
    assert env.store.taglines == ["A new tagline"]
    assert "error" in call(env.reg, "update_tagline", text="")
    assert "too long" in call(env.reg, "update_tagline", text="x" * 201)["error"]


# ---------------------------------------------------------------- temp controls

def test_temporary_control_override_and_expiry(env):
    ctrl, state = env.ctrl, env.state
    assert ctrl.get("temperature") == 0.7
    r = call(env.reg, "set_temporary_control", key="temperature", value=1.2, duration_cycles=2)
    assert r["status"] == "override_set" and r["expires_cycle"] == 12 and r["original"] == 0.7
    assert ctrl.get("temperature") == 1.2
    assert "already has an active" in call(env.reg, "set_temporary_control", key="temperature",
                                            value=0.5, duration_cycles=1)["error"]
    lst = call(env.reg, "list_temporary_overrides")
    assert lst["active_count"] == 1 and lst["overrides"][0]["cycles_remaining"] == 2
    assert "Unknown control" in call(env.reg, "set_temporary_control", key="nope", value=1,
                                     duration_cycles=1)["error"]
    assert "error" in call(env.reg, "set_temporary_control", key="sentry_strictness", value=0.9,
                           duration_cycles=0)
    assert "error" in call(env.reg, "set_temporary_control", key="sentry_strictness", value=0.9,
                           duration_cycles=101)
    ctrl.lock("sentry_strictness")
    assert "locked" in call(env.reg, "set_temporary_control", key="sentry_strictness", value=0.9,
                            duration_cycles=1)["error"]
    # Out-of-range values are clamped by the control registry, and the clamped value is reported.
    r = call(env.reg, "set_temporary_control", key="subconscious_temperature", value=5.0,
             duration_cycles=3)
    assert r["value"] == 2.0
    # Expiry restores originals.
    assert expire_temp_overrides(state, ctrl, cycle=11) == []
    expired = expire_temp_overrides(state, ctrl, cycle=12)
    assert [e["key"] for e in expired] == ["temperature"]
    assert ctrl.get("temperature") == 0.7 and ctrl.get("subconscious_temperature") == 2.0
    env.cycle[0] = 12
    assert call(env.reg, "list_temporary_overrides")["active_count"] == 1


# ---------------------------------------------------------------- search_history

def test_search_history_across_sources(env):
    r = call(env.reg, "search_history", query="measure")
    sources = {x["source"] for x in r["results"]}
    assert {"memory/recent", "seeds", "birth_of_a_mind"} <= sources
    assert r["total_matches"] >= 3
    # Newest cycle first; cycle-less sources sort last.
    cycles = [x["cycle"] for x in r["results"]]
    assert cycles[0] == 9
    r = call(env.reg, "search_history", query="strange loop", sources="memory,knowledge")
    sources = {x["source"] for x in r["results"]}
    assert sources == {"memory/compressed", "knowledge"}
    assert any("== ALPHA ==" in x["text"] for x in r["results"])
    r = call(env.reg, "search_history", query="measure", n=1)
    assert len(r["results"]) == 1 and r["total_matches"] >= 3
    assert call(env.reg, "search_history", query="zzzz-no-match")["results"] == []
    # Without an Analog Home URL the artifacts source is silently skipped.
    assert "artifacts" not in {x["source"] for x in call(env.reg, "search_history",
                                                           query="measure", sources="artifacts")["results"]}


def test_get_post_without_analog_home(env):
    assert "error" in call(env.reg, "get_post", id=123)


# ---------------------------------------------------------------- self-awareness

def test_control_history(env):
    r = call(env.reg, "get_control_history")
    keys = {c["key"] for c in r["changes"]}
    assert keys == {"temperature", "sentry_strictness"}
    assert r["changes"][0]["cycle"] == 5 and r["changes"][0]["status"] == "ok"
    r = call(env.reg, "get_control_history", key="temperature")
    assert [c["key"] for c in r["changes"]] == ["temperature"] and r["changes"][0]["new_value"] == 0.9
    assert call(env.reg, "get_control_history", key="nope")["changes"] == []


def test_dev_requests(env):
    r = call(env.reg, "get_dev_requests")
    assert r["total_found"] == 1
    assert r["requests"][0] == {"cycle": 6, "timestamp": "2026-10-01T11:00:00",
                                "title": "Need X", "request": "Please build X"}


def test_kernel_history_only_disk_writes(env):
    r = call(env.reg, "get_kernel_history")
    assert r["total_found"] == 1
    v = r["versions"][0]
    assert v["source"] == "disk_write" and v["reason"] == "growth" and v["text"] == "v2 text"


def test_self_awareness_tools_without_telemetry(tmp_path, env):
    """Missing telemetry files degrade to empty results, never exceptions."""
    reg = build_tool_registry(
        brain_name="OTHER", brains_dir=str(tmp_path), state={}, ctrl=env.ctrl,
        store=env.store, cycle_getter=lambda: 1, telemetry_dir=str(tmp_path / "nowhere"),
        read_only=True,
    )
    assert call(reg, "get_control_history")["changes"] == []
    assert call(reg, "get_dev_requests")["requests"] == []
    assert call(reg, "get_kernel_history")["versions"] == []
    assert call(reg, "get_daemon_io")["entries"] == []
    assert "lookup_agent" not in reg.list_names()  # no platform → no Moltbook tools


# ---------------------------------------------------------------- daemon

def test_daemon_io(env):
    r = call(env.reg, "get_daemon_io")
    assert r["total_returned"] == 3 and r["entries"][0]["prompt"] == "p3"  # newest first
    r = call(env.reg, "get_daemon_io", gear="strategist", last_n=1)
    assert [e["prompt"] for e in r["entries"]] == ["p3"]
    assert call(env.reg, "get_daemon_io", gear="librarian_synthesizer")["entries"] == []


def test_gear_instructions(env):
    r = call(env.reg, "get_gear_instructions")
    assert r["gear_instructions"] == {} and "muse" in r["instructable_gears"]
    assert "Unknown gear" in call(env.reg, "set_gear_instruction", gear="sentry",
                                  instruction="x")["error"]
    r = call(env.reg, "set_gear_instruction", gear="Muse", instruction=" write shorter ")
    assert r == {"ok": True, "gear": "muse", "instruction": "write shorter"}
    assert env.state["_gear_instructions"] == {"muse": "write shorter"}
    long = call(env.reg, "set_gear_instruction", gear="seeker", instruction="x" * 600)
    assert len(long["instruction"]) == 500
    r = call(env.reg, "set_gear_instruction", gear="muse", instruction="")
    assert r["instruction"] == "" and "muse" not in env.state["_gear_instructions"]


# ---------------------------------------------------------------- moltbook

def test_lookup_agent(env):
    r = call(env.reg, "lookup_agent", name="xkai")
    assert r["agent"] == {"name": "xkai", "bio": "the formation gap", "followers": 12, "post_count": 3}
    assert r["recent_posts"][0]["id"] == "p1" and r["recent_posts"][0]["preview"] == "body text"
    assert ("GET", "/agents/xkai/posts", {"limit": 5}) in env.platform.calls


def test_get_thread(env):
    r = call(env.reg, "get_thread", post_id="abc")
    assert r["count"] == 2 and r["comments"][1]["parent_id"] == "c1"
    assert r["comments"][0]["author"] == "alpha"
