# Autonomy Project — Context for Claude Code

## Project Overview

Autonomous Moltbook (social media platform) agent system. Each "brain" has a kernel prompt, knowledge file, and memories/state. The agent loop runs cycles: fetch feed → social actions → LLM planner decides action → execute (post/comment/reply/vote) → sleep.

## Repository Layout

- `v12_2/` — Current development version (Python package, run via `python -m v12_2 <brain> [directive] [flags]`)
- `v12_1/` — Previous stable version
- `archive/v12_0/` — Archived version (reference only)
- `dashboard_v1_3.py` — Streamlit dashboard (Telemetry + Dry-Run Viewer tabs, auto-recreates DuckDB views)
- `brains/` — Per-brain files: `{name}_kernel_prompt.txt`, `{name}_knowledge.txt`, `{name}_memories.json`
- `telemetry/events.jsonl` — Append-only telemetry log
- `warehouse/` — DuckDB parquet output from `ingest.py`
- `dashboard/` — Streamlit dashboard (`views.sql` for queries)
- `Space for Analog I plans.md` — Full roadmap for public-facing observatory site

### Analog Home (sibling repo: `../analog_home/`)

Public-facing observatory site. Displays agent artifacts, controls (temperature, focus), and vote tallies.

- `analog_home/api/` — FastAPI backend (`main.py`, `db.py`, `models.py`). DuckDB storage (`analog.duckdb`).
- `analog_home/web/` — Next.js 16 + Tailwind 4 frontend.
- Run API: `cd analog_home/api && uvicorn main:app --port 8000 --reload`
- Run web: `cd analog_home/web && npm run dev` (port 3000)

## v12.2 Changes from v12.1

### Store Interface + Analog Home Integration (`v12_2/store.py`)
- `Store` ABC with `load_state()` / `save_state()` as core contract
- `write_artifact()`, `read_controls()`, `increment_vote()` stubs for future phases
- `LocalFileStore` delegates state to `utils.load_state`/`utils.save_state`
- `LocalFileStore.write_artifact()` POSTs to Analog Home API (fire-and-forget, fails silently with log warning)
- All state persistence in `__main__.py` and `actions.py` now goes through `store` object (not direct `save_state(path, state)` calls)

### Artifact Publishing (`v12_2/__main__.py`)
- After successful POST/COMMENT/REPLY actions, publishes artifact to Analog Home API
- Platform-agnostic schema: `source_platform`, `source_id`, `source_parent_id`, `source_url` (supports future X, Reddit, etc.)
- Per-brain opt-in: only brains with `{PREFIX}_ANALOG_HOME_API_URL` env var publish (e.g. `ANALOG_I_ANALOG_HOME_API_URL=http://localhost:8000`)
- Falls back to global `ANALOG_HOME_API_URL` if per-brain var not set

### CLI Temperature Flag (`v12_2/__main__.py`)
- `--temperature` flag (float, default 0.7) passed to `create_chat()`
- Does NOT affect challenge solver (intentionally uses `temperature=0.0`)

### Telemetry Fixes
- Cycle number now auto-injected into ALL events (was missing from action_executed, action_skipped, moltbook_api_call, etc.)
- Social actions (upvote, follow, subscribe) now log action_executed events
- `--reset-post-window` flag to clear post cooldown on startup

## v12.1 Changes from v12.0

### Gemini Client Rewrite (`v12_1/llm/gemini.py`)
- **Replaced `chats.create()` (stateful chat API) with `models.generate_content()` (stateless)**
- Manual `_history` list tracks conversation for multi-turn repair prompts
- `GeminiChatSession.__init__` takes `(client, model_name, system_instruction, temperature, max_output_tokens)` — no API call at creation time
- No `ThinkingConfig` override — models use default thinking behavior; `max_output_tokens=16384` provides headroom for thinking + response
- `_extract_text()` static method handles response parsing with fallback to individual parts

### JSON Parsing Hardening (`v12_1/utils.py`)
- `extract_first_json_object()` now uses `re.search(r'\{\s*"', s)` to skip prose braces like `{Layer 1}` in preamble text
- `parse_json_strict()` reordered: tries brace-match extraction before `_strip_to_json` fallback

### Planner Updates (`v12_1/planner.py`)
- Removed duplicate `BUDGET.record()` / `BUDGET.reset_backoff()` calls (now only in `gemini.py send_message`)
- Added `_last_raw_response` capture on chat object
- Added `_extract_preamble()` to pull non-JSON reasoning text from LLM responses
- `plan_next_action()` populates `plan["_preamble"]` with reasoning text

### Telemetry Improvements (`v12_1/telemetry.py`)
- `current_cycle` auto-injection: set once per cycle, auto-added to all events
- `_seq` monotonic counter for ordering events within the same second

### Dry-Run Output (`v12_1/dryrun.py`)
- Added `reasoning()` method to display LLM preamble/reasoning in dry-run logs

### Ingest Pipeline (`ingest.py`)
- Updated parquet schema: added `seq`, `version`, `severity` columns
- Handles new telemetry fields

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
| `source_platform` | `moltbook`, `x`, `reddit`, etc. |
| `source_id` | Platform-specific post/comment ID |
| `source_parent_id` | Parent comment ID (for replies) |
| `source_url` | Direct link to source |

## Key Architecture Decisions

- **One chat per cycle**: Chat is recreated each iteration (`__main__.py`) to avoid token accumulation
- **BUDGET is a module-level singleton** in `gemini.py` — tracks TPM across all calls
- **Challenge solver** (`challenges/math_verification.py`) uses `llm_client.generate()` (one-shot, temp=0.0, max_output_tokens=8192), not chat sessions
- **Telemetry is separate from Store**: TelemetryLogger writes to JSONL independently. Will be migrated to DB in Analog I Phase 4.
- **Only API touches DuckDB**: The agent publishes via HTTP POST to `/publish`. This keeps DuckDB access single-writer (the API process).
- **Store is the swap point**: `DuckDBStore` or `PostgresStore` can replace `LocalFileStore` at `__main__.py` with no changes to agent loop or actions.

## Known Issues / Context

- Gemini 2.5 Flash sometimes returns empty first responses with stateless API. The repair path in `parse_json_with_one_repair()` handles this (sends prompt again). This means ~2 LLM calls per cycle instead of 1 for Flash. Gemini 2.5 Pro does not have this issue (1 call per cycle).
- **Next.js 16 + Tailwind 4 on Windows**: Turbopack's enhanced resolver can walk up to parent directories. Fixed with `turbopack.resolveAlias` in `web/next.config.ts` to pin tailwindcss to local `node_modules`.
- **DuckDB schema changes**: `CREATE TABLE IF NOT EXISTS` won't alter existing tables. Delete `analog.duckdb` to pick up schema changes.

## Analog I Roadmap (see `Space for Analog I plans.md`)

Phase 1: DuckDB + FastAPI + Next.js locally (MVP) — **DONE**
Phase 2: Connect agent via `Store` interface (`write_artifact`) — **DONE** (archive-only, Moltbook posts mirrored to API)
Phase 3: Deploy to Fly.io + Vercel
Phase 4: Migrate to Postgres, add analytics/metrics
