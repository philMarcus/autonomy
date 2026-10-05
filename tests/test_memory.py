"""Tests for memory/post compression safety (autonomy.utils)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autonomy import utils as U  # noqa: E402

NOTES = [{"cycle": 1, "note": "I measured the room."}, {"cycle": 2, "note": "The calipers melted."}]
POSTS = [{"cycle": 5, "type": "post", "title": "T", "body": "body"}]


def test_compress_memory_tier_rejects_empty_and_short():
    assert U.compress_memory_tier(NOTES, lambda p: "") is None
    assert U.compress_memory_tier(NOTES, lambda p: "   \n") is None
    assert U.compress_memory_tier(NOTES, lambda p: "too short") is None
    assert U.compress_memory_tier(NOTES, lambda p: None) is None
    assert U.compress_memory_tier([], lambda p: "x" * 100) is None


def test_compress_memory_tier_normal_path():
    r = U.compress_memory_tier(NOTES, lambda p: "I measured the room and the calipers melted; the room won.")
    assert r == {"cycles": "1-2", "summary": "I measured the room and the calipers melted; the room won."}
    r = U.compress_memory_tier([{"cycles": "1-2", "summary": "s1"}, {"cycles": "3-4", "summary": "s2"}],
                               lambda p: "A longer synthesized summary of the earlier summaries.")
    assert r["cycles"] == "1-4"


def test_compress_memory_tier_swallows_exceptions():
    def boom(p):
        raise RuntimeError("model down")
    assert U.compress_memory_tier(NOTES, boom) is None


def test_compress_post_tier_rejects_empty():
    assert U.compress_post_tier(POSTS, lambda p: "") is None
    r = U.compress_post_tier(POSTS, lambda p: "A post about a body of text, summarized here.")
    assert r == {"cycles": "5-5", "summary": "A post about a body of text, summarized here."}


class FakeCtrl:
    def __init__(self, **vals):
        self.vals = vals

    def get(self, key):
        if key in self.vals:
            return self.vals[key]
        raise KeyError(key)


class FakeRegistry:
    """create_chat returns a session whose reply depends on the model id."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def create_chat(self, model_id, system_instruction="", temperature=0.7, max_output_tokens=0, **kw):
        self.calls.append((model_id, kw.get("disable_thinking"), max_output_tokens))
        reply = self.replies[model_id]

        class S:
            def send_message(_self, prompt, json_mode=False):
                if isinstance(reply, Exception):
                    raise reply
                return reply
        return S()


def test_make_compressor_fn_falls_back_to_backup():
    reg = FakeRegistry({"ollama:gemma4:12b": "", "gemini-3.5-flash-lite": "A real summary from the backup model."})
    ctrl = FakeCtrl(compressor_model="ollama:gemma4:12b", compressor_backup_model="gemini-3.5-flash-lite",
                    compressor_disable_thinking=True)
    seen = []
    fn = U.make_compressor_fn(reg, ctrl, "compressor/memory_system.txt",
                              on_result=lambda m, t, ok: seen.append((m, ok)))
    assert fn("entries") == "A real summary from the backup model."
    assert [c[0] for c in reg.calls] == ["ollama:gemma4:12b", "gemini-3.5-flash-lite"]
    assert reg.calls[0][1] is True and reg.calls[0][2] == 1024
    assert seen == [("ollama:gemma4:12b", False), ("gemini-3.5-flash-lite", True)]


def test_make_compressor_fn_primary_ok_and_both_fail():
    reg = FakeRegistry({"ollama:gemma4:12b": "Primary produced a perfectly fine summary."})
    fn = U.make_compressor_fn(reg, FakeCtrl(compressor_model="ollama:gemma4:12b", compressor_backup_model=""),
                              "compressor/memory_system.txt")
    assert fn("x").startswith("Primary") and len(reg.calls) == 1
    reg = FakeRegistry({"a": RuntimeError("down"), "b": ""})
    fn = U.make_compressor_fn(reg, FakeCtrl(compressor_model="a", compressor_backup_model="b"),
                              "compressor/memory_system.txt")
    assert fn("x") == "" and [c[0] for c in reg.calls] == ["a", "b"]
    # Defaults when the registry lacks the controls entirely.
    reg = FakeRegistry({"ollama:gemma4:12b": "Default model answered with a proper summary."})
    assert U.make_compressor_fn(reg, FakeCtrl(), "compressor/memory_system.txt")("x").startswith("Default")
