"""Deterministic invariant checks across memory folds."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from autonomy import invariants as I  # noqa: E402
from autonomy.tools import ToolRegistry, ToolCall  # noqa: E402

A = "**Constraint:** Do not use first-person past tense for memory compression; it prunes branch gates."
B = "**Requirement:** Invariants must reside in typed schemas, not narrative, to survive compaction."
C = "**Gate:** Predicate P must hold before action B."
ENTRY1 = f"[INVARIANTS & CONDITIONAL GATES]\n- {A}\n- {B}\n    - sub-detail about schemas\n\n[EPISODIC TRAJECTORY]\n- did things"
ENTRY2 = f"[INVARIANTS & CONDITIONAL GATES]\n- {C}\n\n[EPISODIC TRAJECTORY]\n- more things"


def test_parse_and_ids():
    inv = I.parse_invariants(ENTRY1)
    assert [i["text"][:14] for i in inv] == ["**Constraint:*", "**Requirement:"]
    assert "sub-detail" in inv[1]["text"] and len(inv[0]["id"]) == 6
    assert I.inv_id(f"[inv:abcdef] {A}") == I.inv_id(A) == I.inv_id(A.upper())
    assert I.parse_invariants("[INVARIANTS & CONDITIONAL GATES]\n- None\n\n[EPISODIC TRAJECTORY]\n- x") == []
    assert I.parse_invariants("no header here") == []


def test_enforce_restores_dropped_and_tags():
    folded = "[INVARIANTS & CONDITIONAL GATES]\n- " + B + "\n\n[EPISODIC TRAJECTORY]\n- merged story"
    out, rep = I.enforce([{"summary": ENTRY1}, {"summary": ENTRY2}], folded)
    assert rep["kept"] == 3 and sorted(r[:6] for r in rep["restored"]) == ["**Cons", "**Gate"]
    assert out.count("[inv:") == 3 and "[EPISODIC TRAJECTORY]\n- merged story" in out
    # Reworded beyond recognition counts as omitted: both versions kept.
    reworded = "[INVARIANTS & CONDITIONAL GATES]\n- **Note:** past tense is usually fine for summaries.\n\n[EPISODIC TRAJECTORY]\n- s"
    out2, rep2 = I.enforce([{"summary": ENTRY1}], reworded)
    assert rep2["kept"] == 3 and len(rep2["restored"]) == 2
    # Light rewording that keeps the content tokens survives.
    kept = "[INVARIANTS & CONDITIONAL GATES]\n- Constraint: do not use first-person past tense for memory compression (prunes branch gates)\n- Requirement: invariants must reside in typed schemas not narrative to survive compaction; sub-detail about schemas\n\n[EPISODIC TRAJECTORY]\n- s"
    _, rep3 = I.enforce([{"summary": ENTRY1}], kept)
    assert rep3["restored"] == [] and rep3["kept"] == 2


def test_enforce_missing_header_and_retired():
    out, rep = I.enforce([{"summary": ENTRY2}], "just a narrative with no sections")
    assert out.startswith("[INVARIANTS & CONDITIONAL GATES]\n- [inv:") and rep["restored"] == [C]
    out, rep = I.enforce([{"summary": ENTRY2}], "just a narrative", retired_ids={I.inv_id(C)})
    assert rep["kept"] == 0 and out == "just a narrative"
    # Raw notes (no summaries) enforce nothing and leave text alone.
    out, rep = I.enforce([{"cycle": 1, "note": "x"}], "plain summary")
    assert out == "plain summary" and rep["inputs"] == 0


def test_retire_and_tools():
    state = {"memory_tiers": {"recent": [], "compressed": [{"cycles": "1-7", "summary": ENTRY1}],
                              "deep": [{"cycles": "x", "summary": ENTRY2}]}}
    active = I.active_invariants(state)
    assert len(active) == 3 and {a["tier"] for a in active} == {"compressed", "deep"}
    reg = ToolRegistry("B", "/nonexistent")
    I.build_invariant_tools(reg, state, lambda: 21)

    def call(tool, **a):
        return json.loads(reg.execute([ToolCall(id="1", name=tool, args=a)])[0].content)

    assert call("list_invariants")["active_count"] == 3
    cid = I.inv_id(C)
    assert "error" in call("retire_invariant", id=cid, reason="")
    assert "error" in call("retire_invariant", id="ffffff", reason="x")
    r = call("retire_invariant", id=f"inv:{cid}", reason="P was shown unnecessary in cycle 21")
    assert r["status"] == "retired" and r["text"] == C
    assert cid not in [a["id"] for a in I.active_invariants(state)]
    assert "- None" in state["memory_tiers"]["deep"][0]["summary"] and "more things" in state["memory_tiers"]["deep"][0]["summary"]
    assert call("list_invariants", include_retired=True)["retired"][0]["cycle"] == 21
    # A later fold cannot bring it back.
    out, rep = I.enforce([{"summary": ENTRY2}], "narrative", retired_ids=I.retired_ids(state))
    assert rep["kept"] == 0
