---
name: plan
description: Use this skill when starting a planning session for new features, sprints, or architectural decisions. Creates/updates the plan file and syncs to OneDrive so Phil can access from both desktop and laptop.
disable-model-invocation: true
---

# Planning Session

Use this when Phil wants to plan a new feature, sprint, or architectural change.

## Current state

Active plan: !`cat /root/.claude/plans/*.md 2>/dev/null | head -5 || echo "No active plan"`
OneDrive plan: !`ls -la "/mnt/c/Users/Phil/OneDrive/Documents/v18_plan.md" 2>/dev/null || echo "No OneDrive copy"`

## Steps

### 1. Enter plan mode

Use `EnterPlanMode` to explore the codebase and design the approach.
Follow the plan mode workflow (explore → design → review → finalize).

### 2. Write the plan

Write to the plan file at `/root/.claude/plans/<name>.md`.

Plan should include:
- **Context**: what problem this solves, what prompted it
- **Implementation details**: files to modify, approach, code sketches
- **Verification**: how to test the changes

### 3. Sync to OneDrive

After writing or updating the plan, copy to OneDrive so Phil can access from his laptop:

```bash
cp /root/.claude/plans/<plan-file>.md "/mnt/c/Users/Phil/OneDrive/Documents/<descriptive-name>.md"
```

### 4. Update CLAUDE.md

If the plan represents a new version or architectural direction, add a section
to CLAUDE.md documenting the planned changes. Mark them as planned/in-progress
(not complete) until they're merged.

### 5. Save architecture decisions to memory

Key architectural decisions from the planning session should be saved to project
memory (`/root/.claude/projects/-mnt-c-Users-Phil-autonomy-prod/memory/`) so
future sessions know the direction without re-reading the full plan.

## Design principles to maintain

These are permanent — they don't change between versions:

- **Daemon pre-loads, conscious fine-tunes**: subconscious shapes coarse context,
  conscious has ad-hoc tool access for deeper retrieval.
- **Controls are source of truth**: controls.py has defaults, controls.json overrides,
  CLI overrides both. No competing defaults.
- **Brain-name prefix pattern**: env vars use `{PREFIX}_GEMINI_API_KEY`, file paths
  use `{brain_name}_*.json`. Never hardcode a specific brain name.
- **Model agnosticism**: features work across providers where possible. Custom tools
  are provider-agnostic Python functions.
- **Don't develop on main**: use a dev branch + worktree for non-trivial changes.
  Test with `--read-only --no-moltbook` before merging.
- **.venv is gitignored**: never let venv files into git operations.
- **Post-commit hygiene**: version bump + architect memory note + knowledge refresh
  on every prod commit (use `/push-to-prod` skill).
- **Audience field on controls**: conscious/accountant/both/operator determines who
  sees and can modify each control.
- **CEO/CFO split**: conscious owns content + signal/noise; accountant owns wake
  mechanics + budget. sentry_strictness is conscious, not accountant.
