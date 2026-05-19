---
name: deploy-analog-home
description: Use this skill when deploying changes to the Analog Home site (API on Fly.io, frontend on Vercel). Covers commit, push, API deploy, and endpoint verification.
disable-model-invocation: true
---

# Deploy Analog Home

Run this after making changes to the `analog_home` repo (sibling repo at `../analog_home/`).

## Current state

analog_home branch: !`cd ../analog_home && git branch --show-current 2>/dev/null`
Last commit: !`cd ../analog_home && git log --oneline -1 2>/dev/null`

## Steps

### 1. Commit and push

```bash
cd ../analog_home
git add <changed files>
git commit -m "descriptive message"
git push
```

**Frontend** (web/) auto-deploys to Vercel on push — no manual deploy needed.

### 2. Deploy API (if API changed)

If any files in `api/` were modified:

```bash
cd ../analog_home/api
flyctl deploy
```

Note: `flyctl` auth may expire. If it fails, Phil needs to run `flyctl auth login`
from a machine with a browser, then provide the token or deploy from that machine.

### 3. Verify endpoints

After API deploy, verify:
```bash
curl -s https://analog-home-api.fly.dev/healthz
# Should return {"ok": true}

# If new endpoints were added:
curl -s https://analog-home-api.fly.dev/<new-endpoint>
```

### 4. Verify frontend

Check that the Vercel deployment succeeded:
- Visit https://analog-i.ai
- Verify the specific changes are visible
- Check browser console for errors

### 5. Route ordering

**IMPORTANT**: FastAPI matches routes top-down. Any new `/path/{param}` routes
must be placed AFTER specific routes like `/path/count` or `/path/specific`.
This has caused bugs before (e.g., `/artifacts/count` matched by `/artifacts/{artifact_id}`).

### 6. Database migrations

If new tables are needed, add `CREATE TABLE IF NOT EXISTS` to `db.py`'s `init_db()`.
The table is created on next API startup (Fly deploy triggers restart).
No manual migration needed — `IF NOT EXISTS` is idempotent.
