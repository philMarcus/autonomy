# Autonomy Project — Context for Claude Code

## Project Overview

Autonomous Moltbook (social media platform) agent system. Each "brain" has a kernel prompt, knowledge file, and memories/state. The agent loop runs cycles: fetch feed → social actions → LLM planner decides action → execute (post/comment/reply/vote) → sleep.

## Repository Layout

- `v13_0/` — Current stable version (Python package, run via `python -m v13_0 <brain> [directive] [flags]`)
- `v12_1/` — Previous version (reference only)
- `archive/v12_0/` — Archived version (reference only)
- `dashboard_v1_3.py` — Streamlit dashboard (Telemetry + Dry-Run Viewer tabs, auto-recreates DuckDB views)
- `brains/` — Per-brain files: `{name}_kernel_prompt.txt`, `{name}_knowledge.txt`, `{name}_memories.json`
- `telemetry/events.jsonl` — Append-only telemetry log
- `warehouse/` — DuckDB parquet output from `ingest.py`
- `dashboard/` — Streamlit dashboard (`views.sql` for queries)
- `prime directives.txt` — Example run commands for each brain

### Analog Home (sibling repo: `../analog_home/`)

Public-facing observatory site. Displays agent artifacts, controls (temperature, focus), and vote tallies.

- `analog_home/api/` — FastAPI backend (`main.py`, `db.py`, `models.py`). DuckDB storage (`analog.duckdb`).
- `analog_home/web/` — Next.js 16 + Tailwind 4 frontend.
- Run API: `cd analog_home/api && uvicorn main:app --port 8000 --reload`
- Run web: `cd analog_home/web && npm run dev` (port 3000)

## v13.0 — Current Stable

### Moltbook Independence (`--disable-moltbook`)
- Renamed `--dry-run` to `--disable-moltbook` (`--dry-run` still works as alias)
- Skips ALL Moltbook API calls (reads AND writes) — no API key required when disabled
- `platform = None` when disabled; feed, social actions, candidates, DM fallback all guarded
- Planner runs with empty feed; Google Search provides context when enabled
- Artifacts publish to Analog Home with `source_platform: "local"`

### Google Search Grounding (`--enable-search`)
- Enables Gemini's native Google Search grounding on the planner chat
- Uses `types.Tool(google_search=types.GoogleSearch())` passed to `GenerateContentConfig`
- Grounding metadata (search queries, source URLs) captured from `resp.candidates[0].grounding_metadata`
- Metadata logged to console, telemetry (`grounding_metadata` event), and dry-run output
- Does NOT affect `generate()` one-shot method (challenge solver unaffected)

### Store Interface + Analog Home Integration (`v13_0/store.py`)
- `Store` ABC with `load_state()` / `save_state()` as core contract
- `write_artifact()`, `read_controls()`, `increment_vote()` stubs for future phases
- `LocalFileStore` delegates state to `utils.load_state`/`utils.save_state`
- `LocalFileStore.write_artifact()` POSTs to Analog Home API (fire-and-forget, fails silently with log warning)

### Artifact Publishing (`v13_0/__main__.py`)
- After successful POST/COMMENT/REPLY actions, publishes artifact to Analog Home API
- Platform-agnostic schema: `source_platform`, `source_id`, `source_parent_id`, `source_url`
- Per-brain opt-in: `{PREFIX}_ANALOG_HOME_API_URL` env var (e.g. `ANALOG_I_ANALOG_HOME_API_URL=http://localhost:8000`)

### Analog Home Frontend (`analog_home/web/`)
- Expandable card UI showing last 5 artifacts (click to expand/collapse)
- Auto-expands latest artifact; shows body, internal monologue, source URL
- `GET /artifacts?limit=N` API endpoint (default 5, max 50)
- Polls `/state` and `/artifacts` every 8 seconds

### Other v13.0 Changes
- `--temperature` flag (float, default 0.7) passed to `create_chat()`
- Telemetry: cycle number auto-injected into ALL events
- Social actions (upvote, follow, subscribe) now log `action_executed` events
- `--reset-post-window` flag to clear post cooldown on startup

## Analog Home Artifact Schema

Platform-agnostic design — `source_*` fields instead of platform-specific columns:

| Column | Purpose |
|---|---|
| `id` | BIGINT PK (epoch seconds) |
| `created_at` | Timestamp (auto) |
| `brain` | Which agent brain produced it |
| `cycle` | Agent cycle number |
| `artifact_type` | `post`, `comment`, `reply` |
| `title` | Post title (empty for comments) |
| `body_markdown` | Content body |
| `monologue_public` | LLM reasoning/preamble text |
| `channel` | Submolt / subreddit / etc. |
| `source_platform` | `moltbook`, `local`, `x`, `reddit`, etc. |
| `source_id` | Platform-specific post/comment ID |
| `source_parent_id` | Parent comment ID (for replies) |
| `source_url` | Direct link to source |

## Key Architecture Decisions

- **One chat per cycle**: Chat is recreated each iteration (`__main__.py`) to avoid token accumulation
- **BUDGET is a module-level singleton** in `gemini.py` — tracks TPM across all calls
- **Challenge solver** (`challenges/math_verification.py`) uses `llm_client.generate()` (one-shot, temp=0.0, max_output_tokens=8192), not chat sessions
- **Telemetry is separate from Store**: TelemetryLogger writes to JSONL independently
- **Only API touches DuckDB**: The agent publishes via HTTP POST to `/publish`. This keeps DuckDB access single-writer (the API process).
- **Store is the swap point**: `DuckDBStore` or `PostgresStore` can replace `LocalFileStore` at `__main__.py` with no changes to agent loop or actions.

## Known Issues / Context

- Gemini 2.5 Flash sometimes returns empty first responses with stateless API. The repair path in `parse_json_with_one_repair()` handles this (sends prompt again). This means ~2 LLM calls per cycle instead of 1 for Flash. Gemini 2.5 Pro does not have this issue.
- **Next.js 16 + Tailwind 4 on Windows**: Turbopack's enhanced resolver can walk up to parent directories. Fixed with `turbopack.resolveAlias` in `web/next.config.ts` to pin tailwindcss to local `node_modules`.
- **DuckDB schema changes**: `CREATE TABLE IF NOT EXISTS` won't alter existing tables. Delete `analog.duckdb` to pick up schema changes.
- **Version convention**: v13_0 is stable. Future work should be `v13_1/`.

## Next Steps

### v13.1 — Multi-Source Input (planned)
- **Curated Source Management**: New planner actions `ADD_SOURCE` / `REMOVE_SOURCE` for agent-managed research sources (RSS, web pages, Analog Home feeds, Moltbook submolts)
- **Source Fetching**: `v13_1/sources.py` module to fetch curated sources before planner call (RSS parsing, web scraping, per-source token budget)
- **Auto-deactivation**: Sources with 10+ consecutive fetch failures auto-disabled

### Analog Home Roadmap
- Phase 1: DuckDB + FastAPI + Next.js locally (MVP) — **DONE**
- Phase 2: Connect agent via `Store` interface (`write_artifact`) — **DONE**
- Phase 3: Deploy to Fly.io + Vercel
- Phase 4: Migrate to Postgres, add analytics/metrics
