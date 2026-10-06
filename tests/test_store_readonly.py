"""--read-only must keep every Analog Home write inside the store from happening."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import requests  # noqa: E402

from autonomy import store as S  # noqa: E402


@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted in read-only mode")
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(requests, "delete", boom)
    monkeypatch.setattr(requests, "get", boom)


def test_read_only_store_never_writes(tmp_path, no_network, capsys):
    st = S.LocalFileStore(str(tmp_path / "B_memories.json"), analog_home_url="https://example.invalid/",
                          run_id="r", read_only=True)
    assert st.set_trajectory("a", "b", "c", reason="x") is False
    assert st.set_tagline("t") is False
    assert st.set_default_temperature(0.9) is False
    assert st.consume_seeds([1, 2]) is False
    assert st.write_artifact(1, {"brain": "B", "artifact_type": "post", "body_markdown": "x"}) is None
    assert st.push_daemon_tick(1, ["line"]) is None
    out = capsys.readouterr().out
    assert out.count("[READ-ONLY] skipped") >= 6
    # Local state still works.
    st.save_state({"k": 1})
    assert st.load_state()["k"] == 1


def test_writable_store_calls_network(tmp_path, monkeypatch):
    calls = []

    class R:
        ok = True
        status_code = 200
        def json(self):
            return {"ok": True}
        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "post", lambda url, **k: calls.append(url) or R())
    st = S.LocalFileStore(str(tmp_path / "B_memories.json"), analog_home_url="https://example.invalid/", run_id="r")
    st.set_tagline("t")
    assert calls and calls[0].endswith("/tagline")
