# Autonomy Project — Context for Claude Code

## Project Overview

Autonomous Moltbook (social media platform) agent system. Each "brain" has a kernel prompt, knowledge file, and memories/state. The agent loop runs cycles: fetch feed → social actions → LLM planner decides action → execute (post/comment/reply/vote) → sleep.

## Repository Layout

- `v14_0/` — Current stable version (Python package, run via `python -m v14_0 <brain> [directive] [flags]`)
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

- `analog_home/api/` — FastAPI backend (`main.py`, `db.py`, `models.py`). **Neon Postgres** (was DuckDB, migrated Feb 2026).
- `analog_home/web/` — Next.js 16 + Tailwind 4 frontend. Cyberpunk CRT aesthetic.
- API deployed on **Fly.io** (`analog-home-api.fly.dev`)
- Web deployed on **Vercel** (`marcusrecursives.com`)
- Run API locally: `cd analog_home/api && DATABASE_URL="postgresql://..." uvicorn main:app --port 8000 --reload`
- Run web locally: `cd analog_home/web && npm run dev` (port 3000)

## v14.0 — Current Stable

### Moltbook as opt-in (`--enable-moltbook`)
- Default mode: Moltbook disabled. Output goes to local dry-run log + Analog Home only.
- `--enable-moltbook` enables Moltbook API calls (requires API key)
- Skips ALL Moltbook API calls when disabled — no API key required
- Planner runs with empty feed; Google Search provides context when enabled
- Artifacts publish to Analog Home with `source_platform: "local"`

### Analog Home Integration (bidirectional)
- **Controls**: Agent reads temperature, trajectory votes, and seeds from Analog Home each cycle
- **Seeds**: Users plant text seeds via the UI; agent consumes them after reading
- **Trajectory voting**: Users vote on 3 creative direction labels; agent sees vote tallies
- **Trajectory updates**: Agent can reset vote labels via `set_trajectory` in planner JSON
- **Temperature**: Users adjust temperature via slider (clamped ±0.5, one per IP per cycle, decays toward default over 3h)
- **Default temperature**: Agent can set `default_temperature` via trajectory updates (requires `--enable-default-temp`)
- **Artifact temperature**: Each artifact records the cycle temperature at time of creation

### Google Search Grounding (`--enable-search`)
- Enables Gemini's native Google Search grounding on the planner chat
- Uses `types.Tool(google_search=types.GoogleSearch())` passed to `GenerateContentConfig`
- Grounding metadata (search queries, source URLs) captured from `resp.candidates[0].grounding_metadata`
- Search queries stored in artifact `search_queries` field

### Store Interface (`v14_0/store.py`)
- `Store` ABC with `load_state()` / `save_state()` as core contract
- `write_artifact()` POSTs to Analog Home API (with retry queue for failures)
- `read_controls()` fetches temperature, votes, seeds from Analog Home
- `consume_seeds()` DELETEs seeds by ID after the agent has read them
- `set_trajectory()` POSTs new vote labels (optionally with `default_temperature`)
- `LocalFileStore` delegates state to `utils.load_state`/`utils.save_state`

### CLI Flags
- `--enable-moltbook` — Enable Moltbook API calls
- `--enable-search` — Enable Google Search grounding
- `--enable-default-temp` — Allow agent to set default temperature
- `--temperature` — LLM temperature (default 0.7)
- `--interval` — Minutes between cycles (default 5)
- `--post-interval` — Minutes between posts (default 30)
- `--reset-post-window` — Clear post cooldown on startup
- `--allow-kernel-update` — Allow planner to rewrite kernel prompt
- `--no-kernel-disk-write` — Kernel updates stay in-memory only
- `--inject-espn` — Inject ESPN data into planner context
- `--mode` — `all`, `comment_only`, `no_post`

## Analog Home Architecture

### Backend (`analog_home/api/`)
- **Database**: Neon serverless Postgres (connection pooled via PgBouncer)
- **Connection**: `psycopg3` with `ConnectionPool` (min 2, max 10)
- **Config**: `DATABASE_URL` env var (set as Fly.io secret)
- **Tables**: `controls`, `seeds`, `artifacts`, `ip_rate_limits`
- **Rate limiting**: Unified `ip_rate_limits` table (composite PK: `ip` + `action`), resets each trajectory cycle
- **Temperature decay**: Linear interpolation toward `default_temperature` over `TEMP_DECAY_HOURS` (3h)

### Artifact Schema

| Column | Type | Purpose |
|---|---|---|
| `id` | BIGINT PK | Epoch seconds |
| `created_at` | TIMESTAMPTZ | Auto-set |
| `brain` | VARCHAR | Which agent brain produced it |
| `cycle` | INTEGER | Agent cycle number |
| `artifact_type` | VARCHAR | `post`, `comment`, `reply` |
| `title` | VARCHAR | Post title (empty for comments) |
| `body_markdown` | TEXT | Content body |
| `monologue_public` | TEXT | LLM reasoning/preamble text |
| `channel` | VARCHAR | Submolt / subreddit / etc. |
| `source_platform` | VARCHAR | `moltbook`, `local`, etc. |
| `source_id` | VARCHAR | Platform-specific post/comment ID |
| `source_parent_id` | VARCHAR | Parent comment ID (for replies) |
| `source_url` | VARCHAR | Direct link to source |
| `search_queries` | VARCHAR | Google Search grounding queries |
| `temperature` | DOUBLE PRECISION | LLM temperature at time of creation |

### Frontend (`analog_home/web/`)
- Cyberpunk CRT aesthetic: icosahedron crystal, NavBeams, scan lines
- Orbitron font for "Analog_I" title
- Components: `Crystal`, `NavBeams`, `VotingBox`, `Controls`, `CrtTerminal`, `SeedInput`
- CrtTerminal: expandable artifact cards with cycle/temp/date meta line
- Archives page: paginated artifact history (`/archives`)
- Temperature slider with ±0.5 clamping, 429 error handling

## Key Architecture Decisions

- **One chat per cycle**: Chat is recreated each iteration (`__main__.py`) to avoid token accumulation
- **BUDGET is a module-level singleton** in `gemini.py` — tracks TPM across all calls
- **Challenge solver** (`challenges/math_verification.py`) uses `llm_client.generate()` (one-shot, temp=0.0, max_output_tokens=8192), not chat sessions
- **Telemetry is separate from Store**: TelemetryLogger writes to JSONL independently
- **Only API touches Postgres**: The agent publishes via HTTP POST to `/publish`. This keeps DB access through the API.
- **Store is the swap point**: `DuckDBStore` or `PostgresStore` can replace `LocalFileStore` at `__main__.py` with no changes to agent loop or actions.

## Known Issues / Context

- Gemini 2.5 Flash sometimes returns empty first responses with stateless API. The repair path in `parse_json_with_one_repair()` handles this (sends prompt again). This means ~2 LLM calls per cycle instead of 1 for Flash. Gemini 2.5 Pro does not have this issue.
- **Next.js 16 + Tailwind 4 on Windows**: Turbopack's enhanced resolver can walk up to parent directories. Fixed with `turbopack.resolveAlias` in `web/next.config.ts` to pin tailwindcss to local `node_modules`.
- **Version convention**: v14_0 is stable. Future work should be `v14_1/` or `v15_0/`.

## Deployment

### Analog Home API (Fly.io)
- App: `analog-home-api` (region: `iad`)
- Secret: `DATABASE_URL` (Neon Postgres connection string)
- Deploy: `cd analog_home/api && flyctl deploy`
- Health check: `GET /healthz`
- No volumes needed (stateless — Postgres is external)

### Analog Home Web (Vercel)
- Domain: `marcusrecursives.com`
- API URL configured in Next.js env

### Agent (local)
- Runs locally via `python -m v14_0 <brain> [flags]`
- Connects to Analog Home API via `{PREFIX}_ANALOG_HOME_API_URL` env var
- Uses local DuckDB for Streamlit dashboard (separate from Analog Home Postgres)
