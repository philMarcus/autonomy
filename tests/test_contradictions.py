"""Contradiction Reservoir: append-only, deduplicated, projected verbatim."""
import json, os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from autonomy import contradictions as R  # noqa: E402
from autonomy.tools import ToolRegistry, ToolCall  # noqa: E402


def test_append_load_dedupe(tmp_path):
    d = str(tmp_path)
    assert R.load(d, "B") == []
    e = R.append(d, "B", "prediction_error", "  I predicted  X;  Y happened. ", 7, context="post c7")
    assert e["id"] == 1 and e["text"] == "I predicted X; Y happened." and e["context"] == "post c7"
    assert R.append(d, "B", "prediction_error", "I predicted X; Y happened.", 8) is None  # duplicate
    e2 = R.append(d, "B", "tool_abort", "POST_MOLTBOOK failed: 403", 9, source="harness")
    assert e2["id"] == 2 and e2["source"] == "harness"
    rows = R.load(d, "B")
    assert [r["id"] for r in rows] == [1, 2]
    with pytest.raises(ValueError):
        R.append(d, "B", "bogus", "x", 1)
    with pytest.raises(ValueError):
        R.append(d, "B", "contradiction", "   ", 1)
    # The file is append-only: two lines, nothing rewritten.
    assert open(R.path_for(d, "B")).read().count("\n") == 2


def test_prompt_block(tmp_path):
    assert "Empty" in R.prompt_block([])
    entries = [{"id": i, "cycle": i, "kind": "contradiction", "source": "conscious", "text": f"entry {i}"} for i in range(1, 31)]
    block = R.prompt_block(entries, max_items=5)
    assert "30 entries, newest 5 shown" in block and "entry 30" in block and "entry 25" not in block
    assert R.prompt_block(entries[:3], max_items=5).count("\n- ") == 3
    big = [{"id": i, "cycle": 1, "kind": "contradiction", "text": "x" * 500} for i in range(10)]
    assert len(R.prompt_block(big, max_items=10, max_chars=1200)) < 1400


def test_harness_note_never_raises(tmp_path):
    R.harness_note("/nonexistent/dir/that/cannot/be/created\0", "B", 1, "boom")  # swallowed
    R.harness_note(str(tmp_path), "B", 3, "GENERATE_IMAGE failed: quota")
    assert R.load(str(tmp_path), "B")[0]["source"] == "harness"


def test_tools(tmp_path):
    reg = ToolRegistry("B", str(tmp_path))
    R.build_contradiction_tools(reg, str(tmp_path), "B", lambda: 12)

    def call(tool, **a):
        return json.loads(reg.execute([ToolCall(id="1", name=tool, args=a)])[0].content)

    assert call("log_contradiction", kind="failed_hypothesis", text="Compression keeps invariants")["status"] == "recorded"
    assert call("log_contradiction", kind="failed_hypothesis", text="Compression keeps invariants")["status"] == "duplicate"
    assert "error" in call("log_contradiction", kind="nope", text="x")
    call("log_contradiction", kind="prediction_error", text="Expected 3 replies, got 0", context="c12")
    r = call("read_contradictions")
    assert r["total_in_reservoir"] == 2 and r["entries"][0]["kind"] == "prediction_error"  # newest first
    assert call("read_contradictions", kind="failed_hypothesis")["matched"] == 1
    assert call("read_contradictions", query="replies")["matched"] == 1
    assert [t for t in reg.list_names()] == ["log_contradiction", "read_contradictions"]
    assert reg.get("log_contradiction").mode == "write"
