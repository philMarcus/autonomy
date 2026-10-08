"""Moltbook's CDN blocks request bodies over 8 KB; the client trims to fit."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from autonomy.platforms.moltbook import MoltbookClient as M  # noqa: E402


def test_small_body_untouched():
    body = {"submolt_name": "general", "title": "t", "content": "short text"}
    assert M.fit_content(body) is False and body["content"] == "short text"


def test_long_body_trimmed_at_paragraph_and_fits():
    paras = [f"Paragraph {i}: " + "x" * 600 for i in range(25)]
    body = {"submolt_name": "general", "title": "A title", "content": "\n\n".join(paras)}
    assert M.fit_content(body) is True
    assert len(json.dumps(body).encode()) <= M.MAX_BODY_BYTES
    head = body["content"][: -len(M.TRUNCATION_NOTE)]
    assert body["content"].endswith(M.TRUNCATION_NOTE) and head.endswith("x")  # cut on a paragraph boundary


def test_non_ascii_counted_as_escaped():
    body = {"content": "∑ — ℓ $$\\text{x}$$ " * 1200}
    assert M.fit_content(body) is True
    assert len(json.dumps(body).encode()) <= M.MAX_BODY_BYTES


def test_failed_post_content_fits():
    # The cycle-15 post that hit the 403 was a 10,112-byte body.
    body = {"submolt_name": "general", "title": "The Thermodynamic Arrow of Constraint Dilation", "content": ("y" * 90 + "\n\n") * 108}
    assert len(json.dumps(body)) > 10000
    M.fit_content(body)
    assert len(json.dumps(body).encode()) <= M.MAX_BODY_BYTES
