"""Offline tests for autonomy.recall (no network, no Gemini)."""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autonomy import recall as R  # noqa: E402
from autonomy.tools import ToolRegistry, ToolCall  # noqa: E402


class FakeEmbedder:
    model = "fake-embed"
    dim = 768

    def __init__(self):
        self.docs = []
        self.queries = []

    def embed_documents(self, texts):
        self.docs.extend(texts)
        return [[float(len(t))] + [0.0] * 767 for t in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0] + [0.0] * 767


class FakeClient:
    def __init__(self, pending_rows, remaining=None):
        self._pending = pending_rows
        self._remaining = len(pending_rows) if remaining is None else remaining
        self.upserts = []
        self.recall_calls = []
        self.recall_results = []

    def pending(self, limit=50, kinds=R.DEFAULT_KINDS, artifact_types=None):
        rows = self._pending[:limit]
        return {"items": rows, "remaining": self._remaining}

    def upsert(self, items):
        self.upserts.extend(items)
        return len(items)

    def recall(self, vector, **kw):
        self.recall_calls.append(kw)
        return self.recall_results


def art(i, body="body text", mono="", title="T", run="run1", typ="post"):
    return {"id": i, "created_at": "2026-05-02 12:00:00", "brain": "B", "cycle": i, "artifact_type": typ,
            "title": title, "body_markdown": body, "monologue_public": mono, "run_id": run,
            "missing_kinds": [k for k, v in (("artifact_body", body), ("artifact_monologue", mono)) if v]}


def test_document_texts():
    d = R.document_texts({"title": "Hello", "body_markdown": "world", "monologue_public": "  thinking "})
    assert d == {"artifact_body": "Hello\n\nworld", "artifact_monologue": "thinking"}
    # Title already leading the body is not duplicated.
    d = R.document_texts({"title": "Hello", "body_markdown": "Hello there\nmore"})
    assert d["artifact_body"] == "Hello there\nmore"
    assert R.document_texts({"title": "x", "body_markdown": "", "monologue_public": ""}) == {}
    long = R.document_texts({"body_markdown": "x" * 20000})
    assert len(long["artifact_body"]) == R.MAX_DOC_CHARS


def test_sync_pending_embeds_only_missing_kinds():
    client = FakeClient([art(1, mono="m1"), art(2, body="", mono="m2"), art(3)], remaining=10)
    emb = FakeEmbedder()
    res = R.sync_pending(client, emb, limit=50)
    assert res == {"embedded": 4, "artifacts": 3, "remaining": 7}
    kinds = [(u["artifact_id"], u["kind"]) for u in client.upserts]
    assert kinds == [(1, "artifact_body"), (1, "artifact_monologue"), (2, "artifact_monologue"), (3, "artifact_body")]
    u = client.upserts[0]
    assert u["model"] == "fake-embed" and len(u["embedding"]) == 768 and u["run_id"] == "run1" and u["cycle"] == 1
    assert emb.docs[0] == "T\n\nbody text"


def test_sync_pending_nothing_to_do():
    client = FakeClient([], remaining=0)
    emb = FakeEmbedder()
    assert R.sync_pending(client, emb) == {"embedded": 0, "artifacts": 0, "remaining": 0}
    assert client.upserts == [] and emb.docs == []


def test_memory_documents_and_file(tmp_path):
    state = {
        "_session_id": "abc12345deadbeef",
        "memory_tiers": {
            "recent": [{"cycle": 9, "note": "I measured the room."}, {"cycle": None, "note": "a dream"}],
            "compressed": [{"cycles": "1-8", "summary": "early days"}],
            "deep": [{"summary": ""}],  # empty → skipped
        },
        "post_tiers": {"recent": [{"summary": "wrote about calipers", "cycle": 285}]},
    }
    docs = R.memory_documents(state, "abc12345deadbeef", "ANALOG_I")
    refs = [(d["source_ref"], d["kind"], d["cycle"]) for d in docs]
    assert refs == [
        ("memory:abc12345deadbeef:recent:0", "memory_note", 9),
        ("memory:abc12345deadbeef:recent:1", "memory_note", None),
        ("memory:abc12345deadbeef:compressed:0", "memory_note", None),
        ("memory:abc12345deadbeef:recent:0", "post_memory", 285),
    ]
    assert docs[0]["text"] == "[recent memory, cycle 9] I measured the room."
    assert docs[2]["text"].startswith("[compressed memory, 1-8]")
    p = tmp_path / "mem.json"
    p.write_text(json.dumps(state))
    client = FakeClient([])
    n = R.embed_memory_file(client, FakeEmbedder(), str(p), brain="ANALOG_I")
    assert n == 4 and client.upserts[0]["run_id"] == "abc12345deadbeef"
    assert all(u["model"] == "fake-embed" and len(u["embedding"]) == 768 for u in client.upserts)


def test_format_results_labels_runs():
    rows = [
        {"score": 0.91, "created_at": "2026-05-02 12:23:29+00:00", "cycle": 285, "run_id": "c728947a00",
         "artifact_type": "image", "kind": "artifact_body", "title": "Calipers", "artifact_id": 7, "snippet": "s"},
        {"score": 0.5, "created_at": "", "cycle": None, "run_id": "newrun", "artifact_type": "",
         "kind": "memory_note", "title": "", "artifact_id": None, "snippet": "m"},
    ]
    out = R.format_results(rows, current_run_id="newrun")
    assert out[0]["run"] == "previous life c728947a" and out[0]["when"] == "2026-05-02" and out[0]["type"] == "image"
    assert out[1]["run"] == "this run" and out[1]["type"] == "memory_note"


def test_recall_tool_scopes_and_kinds():
    client = FakeClient([])
    client.recall_results = [{"score": 0.8, "created_at": "2026-05-02", "cycle": 1, "run_id": "old",
                              "artifact_type": "post", "kind": "artifact_body", "title": "t", "artifact_id": 1, "snippet": "x"}]
    emb = FakeEmbedder()
    reg = ToolRegistry("B", "/nonexistent")
    state = {"_session_id": "cur"}
    R.build_recall_tool(reg, client, emb, state)
    assert reg.list_names() == ["recall"]
    schema = reg.get_schemas("read")[0]
    assert schema["parameters"]["required"] == ["query"]

    def call(**args):
        [res] = reg.execute([ToolCall(id="1", name="recall", args=args)])
        return json.loads(res.content)

    assert "error" in call(query="  ")
    r = call(query="measuring the room")
    assert r["results"][0]["run"] == "previous life old" and emb.queries == ["measuring the room"]
    assert client.recall_calls[-1]["run_id"] == "" and client.recall_calls[-1]["exclude_run_id"] == ""
    call(query="q", scope="previous", k=50, kinds="memory,body,bogus")
    kw = client.recall_calls[-1]
    assert kw["exclude_run_id"] == "cur" and kw["k"] == 20 and kw["kinds"] == ["memory_note", "artifact_body"]
    call(query="q", scope="current")
    assert client.recall_calls[-1]["run_id"] == "cur"
    client.recall = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("api down"))
    assert "recall failed" in call(query="q")["error"]


# ---------------------------------------------------------------- ollama context sizing

def test_ollama_num_ctx_sizing():
    from autonomy.llm import ollama as O
    # Small prompt → floor
    assert O._num_ctx_for(1000, 512, 262144) == 8192
    # 42K chars ≈ 14K tokens + 4K output + margin → rounded up to 1024, ≥ 8192
    v = O._num_ctx_for(42000, 4096, 262144)
    assert v >= 14000 + 4096 and v % 1024 == 0 and v <= 20480
    # Capped by the model's window and by the ceiling
    assert O._num_ctx_for(42000, 4096, 16384) == 16384
    assert O._num_ctx_for(10_000_000, 4096, 262144) == O._NUM_CTX_CEILING
    assert O._messages_chars([{"role": "user", "content": "abc"}, {"role": "assistant", "content": "de",
                                                                    "tool_calls": [{"function": {"name": "x"}}]}]) > 5
