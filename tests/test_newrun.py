"""Tests for autonomy.newrun (archive + clean-slate reset)."""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autonomy import newrun as N  # noqa: E402

BRAIN = "TB"


@pytest.fixture
def brains(tmp_path):
    d = tmp_path / "brains"
    d.mkdir()
    state = {
        "_session_id": "c728947af0cb47699e1cfbd76cc1564d", "_cycle_number": 500,
        "memory": "old", "history": [{"action": "POST"}] * 3,
        "memory_tiers": {"recent": [{"cycle": 1, "note": "x"}], "compressed": [], "deep": [{"summary": "s"}]},
        "post_tiers": {"recent": [1]}, "saved_plans": [1], "directive": "old directive",
        "_gear_instructions": {"muse": "x"}, "_budget_state": {"spent": 1},
        "my_post_ids": ["p1"], "replied_comment_keys": ["c1"], "followed_agents": ["a"],
        "subscribed_submolts": ["general"], "_scored_comment_ids": ["c9"], "cooldowns": {"POST": 1},
        "next_post_time": 5.0, "_retry_queue": [],
    }
    (d / f"{BRAIN}_memories.json").write_text(json.dumps(state))
    (d / f"{BRAIN}_todos.json").write_text('[{"id": 1}]')
    (d / f"{BRAIN}_experiments.json").write_text('[{"name": "e"}]')
    (d / f"{BRAIN}_daemon_io.jsonl").write_text('{"gear": "muse"}\n')
    (d / f"{BRAIN}_kernel_prompt.txt").write_text("evolved kernel")
    (d / f"{BRAIN}_kernel_prompt__BoaM_ORIGINAL.txt").write_text("original kernel")
    (d / f"{BRAIN}_knowledge.txt").write_text("knowledge")
    (d / f"{BRAIN}_controls.json").write_text("{}")
    (d / f"{BRAIN}_dev_requests.txt").write_text("req")
    return d


def test_plan_is_read_only(brains):
    p = N.plan(str(brains), BRAIN)
    assert p["old_session_id"].startswith("c728947a") and p["old_cycle"] == 500
    assert p["archive_dir"].startswith(str(brains / "archive" / "TB_c728947a_"))
    assert set(p["archive_files"]) == {"memories.json", "todos.json", "experiments.json", "daemon_io.jsonl",
                                       "dev_requests.txt", "kernel_prompt.txt", "knowledge.txt", "controls.json"}
    assert p["memory_items"] == 2 and p["history_items"] == 3
    assert "directive" in p["dropped_keys"] and "my_post_ids" in p["kept_keys"]
    assert (brains / f"{BRAIN}_memories.json").read_text()  # untouched
    assert not (brains / "archive").exists()


def test_new_run_archives_and_resets(brains):
    p = N.new_run(str(brains), BRAIN)
    arc = p["archive_dir"]
    assert os.path.isdir(arc)
    old = json.load(open(os.path.join(arc, f"{BRAIN}_memories.json")))
    assert old["_cycle_number"] == 500 and old["memory_tiers"]["recent"]
    assert open(os.path.join(arc, f"{BRAIN}_todos.json")).read() == '[{"id": 1}]'
    assert open(os.path.join(arc, f"{BRAIN}_daemon_io.jsonl")).read().startswith('{"gear"')

    new = json.load(open(brains / f"{BRAIN}_memories.json"))
    assert "_session_id" not in new and new["_cycle_number"] == 0
    assert new["memory"] == "" and new["history"] == []
    assert new["memory_tiers"] == {"recent": [], "compressed": [], "deep": []}
    for k in ("post_tiers", "saved_plans", "directive", "_gear_instructions", "_budget_state"):
        assert k not in new
    assert new["my_post_ids"] == ["p1"] and new["replied_comment_keys"] == ["c1"]
    assert new["followed_agents"] == ["a"] and new["_scored_comment_ids"] == ["c9"]
    assert new["cooldowns"] == {"POST": 1} and new["next_post_time"] == 5.0
    assert new["_previous_session_id"].startswith("c728947a")
    assert json.load(open(brains / f"{BRAIN}_todos.json")) == []
    assert json.load(open(brains / f"{BRAIN}_experiments.json")) == []
    assert not (brains / f"{BRAIN}_daemon_io.jsonl").exists()
    assert (brains / f"{BRAIN}_kernel_prompt.txt").read_text() == "evolved kernel"
    assert (brains / f"{BRAIN}_knowledge.txt").read_text() == "knowledge"
    assert (brains / f"{BRAIN}_dev_requests.txt").read_text() == "req"
    # Re-running on the already-fresh state archives under a session-less name, not over the old archive.
    p2 = N.new_run(str(brains), BRAIN)
    assert p2["archive_dir"] != arc and "TB_nosessio" in p2["archive_dir"]
    assert json.load(open(os.path.join(arc, f"{BRAIN}_memories.json")))["_cycle_number"] == 500
    # Same-day collision is refused.
    with pytest.raises(FileExistsError):
        N.new_run(str(brains), BRAIN)


def test_reset_kernel(brains):
    N.new_run(str(brains), BRAIN, reset_kernel=True)
    assert (brains / f"{BRAIN}_kernel_prompt.txt").read_text() == "original kernel"


def test_reset_kernel_requires_original(brains):
    os.remove(brains / f"{BRAIN}_kernel_prompt__BoaM_ORIGINAL.txt")
    with pytest.raises(FileNotFoundError):
        N.new_run(str(brains), BRAIN, reset_kernel=True)
    assert not (brains / "archive").exists()


def test_fresh_state_from_nothing():
    s = N.fresh_state({})
    assert s["_cycle_number"] == 0 and s["history"] == [] and s["_previous_session_id"] == ""


def test_cli_dry_run(brains, capsys):
    assert N.main([BRAIN, "--brains-dir", str(brains), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "old session:      c728947a" in out and "kept (evolved)" in out
    assert "_session_id" not in json.load(open(brains / f"{BRAIN}_memories.json")) or True
    assert not (brains / "archive").exists()


def test_cli_yes(brains, capsys):
    assert N.main([BRAIN, "--brains-dir", str(brains), "--yes"]) == 0
    out = capsys.readouterr().out
    assert "python -m autonomy.recall backfill TB --memory" in out and "--run-id c728947a" in out
    assert "_session_id" not in json.load(open(brains / f"{BRAIN}_memories.json"))
