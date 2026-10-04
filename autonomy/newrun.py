"""Start a new life: archive a brain's run state and reset it to a clean slate.

    python -m autonomy.newrun <brain> [--brains-dir brains] [--reset-kernel] [--dry-run]

What happens:
  * every brain file is COPIED to brains/archive/<brain>_<session8>_<YYYYMMDD>/
  * memories.json is rewritten as a fresh state: no session id (so the agent
    mints a new one and Analog Home shows a new run), cycle 0, empty memory
    tiers / history / post memory / saved plans / gear instructions.
    Platform bookkeeping survives — own post ids, replied-comment keys,
    follows, subscriptions, scored-comment ids, cooldowns, retry queue — so
    the new life cannot double-reply to old threads.
  * todos.json and experiments.json are emptied; daemon_io.jsonl is removed.
  * the kernel prompt is kept (it is the agent's self-model) unless
    --reset-kernel, which restores <brain>_kernel_prompt__BoaM_ORIGINAL.txt.
  * knowledge.txt, controls.json, dream_topics.txt and dev_requests.txt stay.

The previous life stays recallable through the `recall` tool once its
archived memories are embedded:
    python -m autonomy.recall backfill <brain> --memory <archive>/<brain>_memories.json --run-id <session>
"""

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

# State keys that are platform bookkeeping, not memory. Everything else is dropped.
KEEP_STATE_KEYS = (
    "my_post_ids", "replied_comment_keys", "subscribed_submolts", "followed_agents",
    "_scored_comment_ids", "cooldowns", "next_post_time", "next_comment_time",
    "_retry_queue",
)
ARCHIVE_SUFFIXES = (
    "memories.json", "todos.json", "experiments.json", "daemon_io.jsonl",
    "pending_artifacts.json", "dev_requests.txt", "kernel_prompt.txt",
    "kernel_prompt.backup.txt", "knowledge.txt", "controls.json", "dream_topics.txt",
)
RESET_EMPTY_JSON = ("todos.json", "experiments.json")
REMOVE = ("daemon_io.jsonl",)


def _path(brains_dir: str, brain: str, suffix: str) -> str:
    return os.path.join(brains_dir, f"{brain}_{suffix}")


def fresh_state(old: Dict[str, Any]) -> Dict[str, Any]:
    """The clean-slate state derived from an old one (bookkeeping only)."""
    new: Dict[str, Any] = {k: old[k] for k in KEEP_STATE_KEYS if k in old}
    new["memory"] = ""
    new["history"] = []
    new["memory_tiers"] = {"recent": [], "compressed": [], "deep": []}
    new["_cycle_number"] = 0
    new["_previous_session_id"] = old.get("_session_id", "")
    new["_previous_life_ended"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="minutes")
    return new


def plan(brains_dir: str, brain: str, reset_kernel: bool = False) -> Dict[str, Any]:
    """Describe what new_run() would do, without touching anything."""
    mem_path = _path(brains_dir, brain, "memories.json")
    old: Dict[str, Any] = {}
    if os.path.exists(mem_path):
        with open(mem_path, "r", encoding="utf-8") as f:
            try:
                old = json.load(f)
            except json.JSONDecodeError:
                old = {}
    session = old.get("_session_id", "") or "nosession"
    stamp = _dt.datetime.now().strftime("%Y%m%d")
    archive_dir = os.path.join(brains_dir, "archive", f"{brain}_{session[:8]}_{stamp}")
    present = [s for s in ARCHIVE_SUFFIXES if os.path.exists(_path(brains_dir, brain, s))]
    original_kernel = _path(brains_dir, brain, "kernel_prompt__BoaM_ORIGINAL.txt")
    return {
        "brain": brain,
        "old_session_id": old.get("_session_id", ""),
        "old_cycle": old.get("_cycle_number", 0),
        "archive_dir": archive_dir,
        "archive_files": present,
        "reset_kernel": reset_kernel,
        "original_kernel_available": os.path.exists(original_kernel),
        "kept_keys": [k for k in KEEP_STATE_KEYS if k in old],
        "dropped_keys": sorted(k for k in old if k not in KEEP_STATE_KEYS),
        "memory_items": sum(len(v) for v in (old.get("memory_tiers") or {}).values() if isinstance(v, list)),
        "history_items": len(old.get("history") or []),
    }


def new_run(brains_dir: str, brain: str, reset_kernel: bool = False) -> Dict[str, Any]:
    """Archive and reset. Returns the plan dict plus 'new_state_path'."""
    p = plan(brains_dir, brain, reset_kernel=reset_kernel)
    if reset_kernel and not p["original_kernel_available"]:
        raise FileNotFoundError(f"--reset-kernel needs {brain}_kernel_prompt__BoaM_ORIGINAL.txt")
    if os.path.exists(p["archive_dir"]):
        raise FileExistsError(f"archive dir already exists: {p['archive_dir']}")
    os.makedirs(p["archive_dir"])
    for suffix in p["archive_files"]:
        shutil.copy2(_path(brains_dir, brain, suffix), os.path.join(p["archive_dir"], f"{brain}_{suffix}"))

    mem_path = _path(brains_dir, brain, "memories.json")
    old: Dict[str, Any] = {}
    if os.path.exists(mem_path):
        with open(mem_path, "r", encoding="utf-8") as f:
            try:
                old = json.load(f)
            except json.JSONDecodeError:
                old = {}
    with open(mem_path, "w", encoding="utf-8") as f:
        json.dump(fresh_state(old), f, ensure_ascii=False, indent=2)

    for suffix in RESET_EMPTY_JSON:
        with open(_path(brains_dir, brain, suffix), "w", encoding="utf-8") as f:
            f.write("[]\n")
    for suffix in REMOVE:
        try:
            os.remove(_path(brains_dir, brain, suffix))
        except FileNotFoundError:
            pass
    if reset_kernel:
        shutil.copy2(_path(brains_dir, brain, "kernel_prompt__BoaM_ORIGINAL.txt"),
                     _path(brains_dir, brain, "kernel_prompt.txt"))
    p["new_state_path"] = mem_path
    return p


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m autonomy.newrun",
                                 description="Archive a brain's run and reset it to a clean slate.")
    ap.add_argument("brain")
    ap.add_argument("--brains-dir", default=os.environ.get("BRAINS_DIR", "brains"))
    ap.add_argument("--reset-kernel", action="store_true",
                    help="restore the original Birth-of-a-Mind kernel instead of keeping the evolved one")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen and exit")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args(argv)

    p = plan(args.brains_dir, args.brain, reset_kernel=args.reset_kernel)
    print(f"brain:            {p['brain']}")
    print(f"old session:      {p['old_session_id'] or '(none)'}  cycle {p['old_cycle']}  "
          f"({p['memory_items']} memory items, {p['history_items']} history entries)")
    print(f"archive to:       {p['archive_dir']}")
    print(f"archive files:    {', '.join(p['archive_files']) or '(none)'}")
    print(f"keep in state:    {', '.join(p['kept_keys']) or '(nothing)'}")
    print(f"drop from state:  {', '.join(p['dropped_keys']) or '(nothing)'}")
    print(f"kernel:           {'RESET to BoaM original' if args.reset_kernel else 'kept (evolved)'}")
    if args.dry_run:
        return 0
    if not args.yes:
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans != "y":
            print("aborted")
            return 1
    p = new_run(args.brains_dir, args.brain, reset_kernel=args.reset_kernel)
    print(f"\ndone. fresh state written to {p['new_state_path']}")
    if p["old_session_id"]:
        print("make the previous life recallable:")
        print(f"  python -m autonomy.recall backfill {args.brain} "
              f"--memory {os.path.join(p['archive_dir'], args.brain + '_memories.json')} "
              f"--run-id {p['old_session_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
