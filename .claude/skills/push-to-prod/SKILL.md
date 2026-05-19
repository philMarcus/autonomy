---
name: push-to-prod
description: Use this skill after committing code changes to the autonomy agent (main branch) that affect agent behavior. Covers both small bugfixes and larger merges from dev. Ensures version bump, architect memory note, knowledge refresh, and design principle compliance.
disable-model-invocation: true
---

# Push to Prod — Post-Commit Hygiene

Run this after every commit to `main` that changes how the autonomy agent operates
(edits to `autonomy/*.py`, controls, wake mechanics, LLM backends, tools, terminal output, etc.).

Skip for doc-only or cosmetic changes (CLAUDE.md, README, minor log formatting).

## Current state

Git branch: !`git branch --show-current`
Last commit: !`git log --oneline -1`
Current VERSION: !`cat autonomy/__init__.py`

## Steps

### 1. Version bump

Bump `VERSION` in `autonomy/__init__.py`:
- **Bugfix** (one-line fix, error handling): increment patch (18.0.1 → 18.0.2)
- **Feature** (new tool, new control, behavior change): increment minor (18.0 → 18.1)
- **Architectural** (new sprint merge, major redesign): increment major (18.x → 19.0)

If this is a **merge from dev branch**, use the major/minor bump (not patch).

### 2. Architect memory note

Append to `brains/{BRAIN}_memories.json` → `state["memory_tiers"]["recent"]`:

```json
{"cycle": null, "note": "[ARCHITECT update YYYY-MM-DDTHH:MM UTC] ...summary..."}
```

Use the **brain name from the commit context** (usually ANALOG_I). Do NOT hardcode brain names.
Keep recent tier capped at ~20 entries.

The note should:
- Summarize what changed in 2-4 sentences
- Note if restart is required
- Mention any new tools, controls, or behavioral changes the agent should know about

### 3. Knowledge file refresh

Scan `brains/{BRAIN}_knowledge.txt` for sections that contradict the new code:
- `YOUR ARCHITECTURE` — does it describe features that now exist or were removed?
- `TOOLS` — are all tools listed? Any new ones missing?
- `THE MODEL CADRE` — are model pool names current?
- `BUDGET` — any budget behavior changes?

Rewrite stale sections (don't append endlessly). If nothing is stale, skip this step.

### 4. Design principles check

Before committing, verify:
- [ ] **Brain-name prefix pattern**: env vars use `{PREFIX}_GEMINI_API_KEY`, not hardcoded `ANALOG_I_GEMINI_API_KEY`. File paths use `{brain_name}_memories.json`, not hardcoded.
- [ ] **Model agnosticism**: new features work across Gemini/OpenAI/Anthropic where applicable.
- [ ] **.venv is gitignored**: never let venv files into git operations.
- [ ] **No `--include-untracked`** in git stash without checking what's untracked first.
- [ ] **Controls as source of truth**: new tunables are registered in controls.py, not hardcoded.

### 5. If this is a merge from dev

Additional steps:
- Update CLAUDE.md with new version section describing what was added
- Update the plan file if one exists
- Run a smoke test: `python -m autonomy {BRAIN} "test" --no-moltbook --read-only --no-subconscious`

### 6. Commit hygiene changes

Include version bump + memory note in the same commit (or immediately after):
```
git add autonomy/__init__.py
git commit -m "vX.Y.Z: post-commit hygiene — version bump + architect memory note"
git push
```

### 7. Update agent_updates memory

Add an entry to the agent code updates memory log for this session.
