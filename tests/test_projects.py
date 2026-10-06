"""Offline tests for autonomy.projects (fake GitHub API)."""

import base64
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autonomy import projects as P  # noqa: E402
from autonomy.tools import ToolRegistry, ToolCall  # noqa: E402

REPOS = [
    {"name": "Mastermind", "description": "Difficulty analysis", "language": "Java", "pushed_at": "2025-10-31T00:00:00Z",
     "stargazers_count": 3, "private": False, "html_url": "https://github.com/philMarcus/Mastermind", "topics": ["games"], "fork": False},
    {"name": "powers-of-zen", "description": "Zoom videos", "language": "Python", "pushed_at": "2026-10-06T00:00:00Z",
     "stargazers_count": 0, "private": False, "html_url": "https://github.com/philMarcus/powers-of-zen", "fork": False},
    {"name": "philMarcus", "description": "profile", "language": None, "pushed_at": "2026-10-01T00:00:00Z", "fork": False},
    {"name": "someone-elses", "description": "fork", "language": "C", "pushed_at": "2026-01-01T00:00:00Z", "fork": True},
]


def fake_get(path, params=None, raw=False, timeout=30):
    if path == "/users/philMarcus/repos":
        return REPOS
    if path == "/repos/philMarcus/Mastermind/readme":
        return "# Mastermind\n\nA study.\n\n" + ("Paragraph about inference. " * 20 + "\n\n") * 30
    if path == "/repos/philMarcus/powers-of-zen/readme":
        return "# Powers of Zen\n\nPipeline."
    if path == "/repos/philMarcus/powers-of-zen/git/trees/HEAD":
        return {"tree": [{"path": "README.md", "type": "blob"}, {"path": "PLAN.md", "type": "blob"},
                         {"path": "engine", "type": "tree"}, {"path": "engine/dive.py", "type": "blob"},
                         {"path": "engine/deep/x.py", "type": "blob"}]}
    if path == "/repos/philMarcus/Mastermind/git/trees/HEAD":
        return {"tree": [{"path": "README.md", "type": "blob"}]}
    if path == "/repos/philMarcus/powers-of-zen/contents/PLAN.md":
        return {"encoding": "base64", "content": base64.b64encode(b"# Plan\n\nDecisions.").decode()}
    if path == "/repos/philMarcus/powers-of-zen/contents/engine":
        return [{"path": "engine/dive.py", "type": "file"}, {"path": "engine/deep", "type": "dir"}]
    if path == "/repos/philMarcus/powers-of-zen/contents/missing.txt":
        raise FileNotFoundError(path)
    raise AssertionError(f"unexpected {path}")


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    P._cache.clear()
    monkeypatch.setattr(P, "_get", fake_get)
    yield
    P._cache.clear()


def test_list_repos_filters_and_sorts():
    repos = P.list_repos(fake_get)
    assert [r["repo"] for r in repos] == ["powers-of-zen", "Mastermind"]   # newest push first; profile + fork dropped
    assert repos[1]["topics"] == ["games"] and repos[1]["pushed_at"] == "2025-10-31"


def test_read_helpers():
    assert P.read_readme("powers-of-zen", fake_get).startswith("# Powers of Zen")
    assert P.read_tree("powers-of-zen", fake_get) == ["PLAN.md", "README.md", "engine/", "engine/dive.py"]
    assert P.read_file("powers-of-zen", "PLAN.md", fake_get) == "# Plan\n\nDecisions."
    assert "engine/deep/" in P.read_file("powers-of-zen", "engine", fake_get)


def test_chunking():
    text = "\n\n".join(f"para {i} " + "x" * 300 for i in range(20))
    chunks = P._chunk(text, size=1000)
    assert all(len(c) <= 1500 for c in chunks) and sum(len(c) for c in chunks) >= len(text) - 2 * len(chunks)
    assert P._chunk("", 1000) == []
    big = "y" * 5000
    assert len(P._chunk(big, 1000)) >= 4


def test_project_documents():
    docs = P.project_documents(fake_get, brain="B")
    refs = [d["source_ref"] for d in docs]
    assert "gh:powers-of-zen:README.md:0" in refs and "gh:powers-of-zen:PLAN.md:0" in refs
    assert any(r.startswith("gh:Mastermind:README.md:") for r in refs)
    assert all(d["kind"] == "phil_project" and d["run_id"] == "" for d in docs)
    zen = next(d for d in docs if d["source_ref"] == "gh:powers-of-zen:README.md:0")
    assert zen["text"].startswith("[Phil's project: powers-of-zen — Zoom videos] (README.md, part 1)")


def test_tools():
    reg = ToolRegistry("B", "/nonexistent")
    P.build_project_tools(reg)
    assert reg.list_names() == ["list_phil_projects", "read_phil_project"]

    def call(tool, **args):
        [res] = reg.execute([ToolCall(id="1", name=tool, args=args)])
        return json.loads(res.content)

    lst = call("list_phil_projects")
    assert lst["count"] == 2 and lst["projects"][0]["repo"] == "powers-of-zen"
    assert call("read_phil_project", repo="Mastermind")["content"].startswith("# Mastermind")
    assert call("read_phil_project", repo="powers-of-zen", path="tree")["entries"][0] == "PLAN.md"
    assert call("read_phil_project", repo="powers-of-zen", path="PLAN.md")["content"] == "# Plan\n\nDecisions."
    assert "not found" in call("read_phil_project", repo="powers-of-zen", path="missing.txt")["error"]
    assert "error" in call("read_phil_project", repo="")
    long = call("read_phil_project", repo="Mastermind")
    assert len(long["content"]) <= P.MAX_FILE_CHARS + 60
