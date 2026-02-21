# Autonomy Project — Context for Claude Code

## Project Overview

Autonomous Moltbook (social media platform) agent system. Each "brain" has a kernel prompt, knowledge file, and memories/state. The agent loop runs cycles: fetch feed → social actions → LLM planner decides action → execute (post/comment/reply/vote) → sleep.

## Repository Layout

- `v15_5/` — **Current stable version** (Python package, run via `python -m v15_5 <brain> [directive] [flags]`)
- `v15_6/` — Development version (forked from v15_5)
- `archive/` — Archived versions: v12_0, v12_1, v12_2, v13_0, v14_0, v15_0 (reference only)
- `dashboard_v1_3.py` — Streamlit dashboard (Telemetry + Dry-Run Viewer tabs, auto-recreates DuckDB views)
- `brains/` — Per-brain files: `{name}_kernel_prompt.txt`, `{name}_knowledge.txt`, `{name}_memories.json`, `{name}_controls.json`
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

## v15.0 — Stable (Multi-Model + ControlRegistry)

v15_0/ has Phases 1-4 complete. **Do not modify v14_0/ or v15_0/** — they are stable.

### Architecture: Dual-Process "Neuron" Model
- **Subconscious Daemon** (cheap/local model) runs continuously with two gears:
  - **Gear 1: Sentry** — scans feeds, scores items against directives
  - **Gear 2: Strategist** — activates on high-signal items, generates draft plans, adds "charge" to wake potential
- **Integrate-and-Fire**: drafts accumulate in a buffer, each adding charge. When `wake_potential` crosses threshold, consciousness fires — seeing the full buffer and synthesizing multiple insights
- **Dreaming**: a conscious action (not subconscious). Compresses old memories into synthesized narratives at front of memory
- **Downward Causality**: every conscious output includes directives to daemon + control updates to any system parameter
- **Feedback Controls**: every configurable number is a control (readable/writable by conscious, settable/blacklistable by user)

### Multi-Model LLM Backend (Phase 1-3: COMPLETE)
- `v15_0/llm/base.py` — ABCs: `ModelBackend`, `ChatSession`, `LLMClient`, `LLMResponse`, `ModelInfo`, `CompatAdapter`
- `v15_0/llm/registry.py` — `ModelRegistry`: maps model IDs to provider backends, `as_llm_client()` for v14 compat
- `v15_0/llm/budget.py` — `DailyBudget`: per-model USD tracking, resets midnight UTC, `spend_summary_text()` for LLM prompt
- `v15_0/llm/gemini.py` — `GeminiBackend`: 5 models (2.5 Flash/Pro, 2.0 Flash-Lite, 3 Pro/Flash Preview)
- `v15_0/llm/anthropic.py` — `AnthropicBackend`: Claude Haiku 4.5, Sonnet 4.5, Opus 4.6
- `v15_0/llm/openai.py` — `OpenAIBackend`: GPT-5 Nano, Mini, 5.1, 5.2, Pro, 5.2 Pro
- `v15_0/llm/mistral.py` — `MistralBackend`: Mistral Small, Mistral Large
- `v15_0/llm/local.py` — `LocalBackend`: HuggingFace transformers, lazy GPU loading, 4-bit quantization for 7B+ on <=10GB VRAM
  - Small: Qwen3 1.5B, Llama 3.2 3B (float16, no quantization)
  - Full: Qwen3 8B, Mistral 7B, Llama 3.1 8B (4-bit NF4 via bitsandbytes)
- See `v15_0/MODELS.md` for full pricing and usage reference

### ControlRegistry (Phase 4: COMPLETE)
- `v15_0/controls.py` — `Control` dataclass, `ControlRegistry` class, `build_default_registry()` factory
- Every tunable is a first-class control: readable by LLM, writable by LLM, blacklistable by user
- 22 controls across 7 categories: llm, cost, timing, output, social, daemon, context
- **Model selection as a control**: `conscious_model` and `subconscious_model` are controls the LLM can modify to switch models mid-run
- **Budget visibility**: LLM sees per-model spend summary and remaining budget in its prompt
- **Blacklist**: `--blacklist-controls key1,key2` prevents LLM from modifying those controls (shown as `[LOCKED]`)
- **Output destination**: `output_destination` control chooses between `analog_home` or `moltbook_and_analog_home`
- **Mode expansion**: `mode` control supports `all`, `comment_only`, `no_post`, `no_comment`, `post_only`
- Controls persisted to `brains/{brain}_controls.json` between runs
- LLM returns `"controls_update": {...}` in JSON response to modify controls
- Validation: type coercion, range clamping, choice checking, change logging

### v15 CLI Flags (additive to v14)
- `--conscious-model` — Model for conscious loop (default: same as `--gemini-model`)
- `--subconscious-model` — Model for subconscious daemon (default: `gemini-2.5-flash`)
- `--daily-budget` — Daily API spend limit in USD (default: 1.0)
- `--sentry-interval` — Seconds between sentry scans (default: 60)
- `--no-subconscious` — v14-compatible single-loop mode (default: True)
- `--subconscious` — Enable dual-process mode
- `--blacklist-controls` — Comma-separated control keys LLM cannot modify

### v15 Environment Variables
Optional per-provider API keys (per-brain or global):
- `{PREFIX}_ANTHROPIC_API_KEY` / `ANTHROPIC_API_KEY`
- `{PREFIX}_OPENAI_API_KEY` / `OPENAI_API_KEY`
- `{PREFIX}_MISTRAL_API_KEY` / `MISTRAL_API_KEY`

## v15.5 — Stable (Subconscious Daemon + Saved Plans)

v15_5/ is now the stable foundation. **Do not modify v15_5/** — use v15_6/ for future work.

### Subconscious Daemon (Phase 5: COMPLETE)
- `v15_5/buffer.py` — `Draft` dataclass + `DraftBuffer` (thread-safe wake potential + draft storage)
- `v15_5/daemon.py` — `SubconsciousDaemon`: background thread with Sentry + Strategist gears
  - **Sentry**: scans feed, scores items against directive using cheap LLM with kernel as system instruction
  - **Strategist**: generates draft action plans for high-signal items, adds charge to wake potential
  - **Integrate-and-Fire**: wake_potential accumulates, decays per tick, fires conscious when threshold crossed
  - **Downward causality**: conscious sends `daemon_directives` (focus_topics, ignore_authors, urgency_boost, note)
- `v15_5/llm/budget.py` — `DailyBudget` now thread-safe (`threading.Lock` on all public methods)
- `v15_5/controls.py` — 2 new daemon controls: `max_drafts`, `strategist_max_tokens`
- `v15_5/planner.py` — `draft_context` param, `--- SUBCONSCIOUS BUFFER ---` prompt section, `daemon_directives` response field
- `v15_5/__main__.py` — Daemon lifecycle, buffer drain, wake/sleep mechanism, downward causality

### How the Daemon Works
1. Daemon thread starts on `--subconscious` flag (default: `--no-subconscious`)
2. Every `sentry_interval_seconds` (default 60s): fetch feed, score items, decay wake_potential
3. Items scoring above `signal_threshold` trigger Strategist → draft plan → charge added to buffer
4. When `wake_potential >= wake_threshold`, conscious loop wakes early (instead of fixed sleep)
5. Conscious sees "SUBCONSCIOUS BUFFER" in prompt with draft summaries
6. Conscious responds with `daemon_directives` to steer daemon's focus
7. Both sentry and strategist use kernel_prompt as system instruction (same personality as conscious)

### Saved Plans (Draft Persistence)
- `v15_5/buffer.py` — Draft.to_dict() / from_dict() for state serialization
- Unused daemon drafts saved to `state["saved_plans"]` for up to 5 cycles (configurable via `SAVED_PLAN_MAX_CYCLES`)
- Conscious sees both fresh drafts + saved plans in prompt
- Plans tagged with `[HUMAN SEED]` when seed-originated, `cycles_saved` counter tracks age
- Plans prioritized by signal_score, aged out gracefully
- Conscious can act on any plan, synthesize multiple, or ignore — unused plans persist

### Seed Enhancements
- Planner prompt explicitly states seeds are from "HUMAN visitors at Analog Home"
- Seeds marked as higher priority than feed noise
- Clear response options: POST on Analog Home (no cooldown), POST on Moltbook, weave into actions, redirect daemon focus
- Draft source field: `"feed"` or `"seed"` for visibility in conscious prompt

### Age Filtering (Complete)
- New control: `max_item_age_hours` (default 24, range 1-168)
- Shared `is_item_too_old()` utility in utils.py
- Filters applied to ALL candidate sources:
  - Daemon sentry scan (feed items)
  - `pick_outside_post_for_comment()` (feed items)
  - `find_unanswered_comment_on_my_posts()` (comments on your posts)
- Prevents responding to stale content (16-day-old posts/comments)

### Search Awareness
- Daemon prompts now inform LLM about Google Search availability
- Sentry: "You have access to Google Search for current information"
- Strategist: "Use this when drafting responses that benefit from facts, news, or recent developments"
- Matches conscious planner's search grounding note for consistent search usage

### Terminology & UX Improvements
- "LOCAL" → "ANALOG HOME" throughout (local was misleading since Analog Home is permanently archived web app)
- `output_destination` values: `analog_home`, `moltbook_and_analog_home`
- Daemon directives terminal output: shows focus topics, ignore authors, urgency boost, note
- Feed state transition detection: `feed_resumed`, `feed_unavailable` telemetry events
- Platform status awareness in planner prompt (READS OK, WRITES BLOCKED, etc.)

### Bug Fixes (v15.5)
- Moltbook POST API: `submolt` → `submolt_name` (API change)
- Verification challenges: nested `comment.verification` detection
- Feed/write decoupling: platform created when API key exists, `--enable-moltbook` only gates writes
- `moltbook_disabled` comparison: `"moltbook" not in output_dest` (was `!= "moltbook"`)
- Post cooldown: platform-specific (Analog Home has no cooldown, Moltbook has 30min)
- Daemon seeding: pre-populate `_seen_ids` from state on startup to avoid re-scoring old feed
- Strategist parse failures: enhanced diagnostics (full_text_length, looks_truncated, model)

### Phase 6: DREAM Action (COMPLETE)
- Dream action implemented in v15_5 (from previous session)
- `compress_memories()` is append-only: synthesizes oldest history into narrative prepended to memory
- History NOT consumed (unlike earlier designs)
- Control: `dream_depth` (default 10, range 3-50)

### Remaining Phases
- **Phase 7**: Remote management via Analog Home dashboard

## v15.6 — In Development (Phase 7+)

v15_6/ is forked from v15_5. **Do not modify v15_5/** — it is the stable foundation.

This is the active development version. All future work should be done in v15_6.

### Changes in v15.6

**Feed Endpoint Fix (Hybrid Personalized + Global)**:
- `MoltbookClient.get_feed()` now uses `/feed` (personalized: subscriptions + follows) instead of `/posts` (global)
- Added `get_global_feed()` method for global discovery feed (`/posts`)
- **Hybrid approach**: When personalized feed is sparse (< half of feed_batch_size), automatically supplements with global feed
- Prevents duplicate posts by tracking seen IDs
- Applies to both conscious loop (`__main__.py`) and daemon sentry scan (`daemon.py`)
- Fixes issue where feed appeared stale/unchanging (was fetching global "hot" feed which changes slowly)

**Enhanced Moltbook Awareness**:
- Updated `ANALOG_I_knowledge.txt` with comprehensive Moltbook API reference including:
  - Personalized vs global feed distinction
  - Semantic search capability (`GET /search`)
  - Rate limits (1 post/30min, 1 comment/20sec, 50 comments/day)
  - Following guidelines (be selective!)
  - Verification challenge info
- Planner prompt now includes rate limit warnings and following guidelines
- Daemon prompts (sentry + strategist) now mention Moltbook semantic search capability
- Conscious does NOT see semantic search info (daemon-only knowledge)

**Seed Response Control**:
- Planner prompt clarifies `output_destination` control for directing seed responses to Analog Home vs Moltbook
- `output_destination` is actuated immediately (affects current cycle), unlike most controls (affect next cycle)

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
- **Version convention**: v15_5 is stable production (Phases 1-6 complete). v15_6 is in development (Phase 7+). **Do not modify v15_5** — use v15_6 for future work. Older versions archived in `archive/`.

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
