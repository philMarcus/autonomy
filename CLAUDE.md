# Autonomy Project — Context for Claude Code

## Project Overview

Autonomous Moltbook (social media platform) agent system. Each "brain" has a kernel prompt, knowledge file, and memories/state. The agent loop runs cycles: fetch feed → social actions → LLM planner decides action → execute (post/comment/reply/vote) → sleep.

## Repository Layout

- `autonomy/` — **Active version** (Python package, run via `python -m autonomy <brain> [flags]`)
- `archive/` — Archived versions: v12_0–v16_0 (reference only, do not modify)
- `dashboard_v2_0.py` — Previous dashboard (4 tabs, reference only)
- `dashboard_v2_1.py` — **Current dashboard** (Overview, Cycle Replay, Daemon Monitor, Input/Controls, Controls Manager tabs)
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
- `--interval` — Minutes between cycles (default 60)
- `--post-interval` — Minutes between posts (default 30)
- `--reset-post-window` — Clear post cooldown on startup
- `--allow-kernel-update` — Allow planner to rewrite kernel prompt (default: enabled, also a control)
- `--no-kernel-update` — Prevent planner from rewriting kernel prompt
- `--no-kernel-disk-write` — Kernel updates stay in-memory only
- `--inject-espn` — Inject ESPN data into planner context
- `--mode` — `all`, `comment_only`, `no_post`
- `--subconscious-model` — DEPRECATED (use model weight controls in controls.json)
- `--sentry-interval` — Seconds between sentry scans (default: 300)

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
| `run_id` | VARCHAR | Agent session UUID (groups artifacts across restarts) |
| `image_url` | TEXT | Base64 data URI for generated images (JPEG, ~140KB) |

### Frontend (`analog_home/web/`)
- Cyberpunk CRT aesthetic: icosahedron crystal, NavBeams, scan lines
- CrtTerminal: expandable cards with preview text, system artifacts auto-expanded, IMG badge for images
- Archives page: "Present Run" section, past major/short run split, run titles from first artifact
- Home page: filtered to latest session's artifacts only
- Featured image: most recent image shown between controls and artifacts (via `/latest-image` endpoint)
- System event colors: pink (magenta) for RUN START, KERNEL SELF-UPDATE, TAGLINE UPDATE; blue (cyan) for CYCLE REPORT, CONTROLS UPDATE
- Moltbook posts prefixed "Moltbook Post:" in collapsed card view
- Agent-controlled tagline (subtitle under "Analog_I")
- Temperature slider with ±0.5 clamping, 429 error handling
- API endpoints: `/runs`, `/artifacts?run_id=X`, `/tagline`, `/latest-image`, `/audience`
- `/audience` returns: unique voters, unique seeders, last vote/seed timestamps, total votes
- Vercel Analytics integration (`@vercel/analytics`) for page views and visitors

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
- **Model selection via cadres**: `conscious_model` and `subconscious_model` individual controls DEPRECATED. All model selection via weighted pools: `conscious_model_weights`, `subconscious_model_weights`, `strategist_model_weights`, `seeker_model_weights`, `verification_model_weights`. Model weight controls locked by default (operator decision).
- **Budget visibility**: LLM sees per-model spend summary and remaining budget in its prompt
- **Blacklist**: `--blacklist-controls key1,key2` prevents LLM from modifying those controls (shown as `[LOCKED]`)
- **Action split**: `POST` (Analog Home only, no cooldown) and `POST_MOLTBOOK` (Moltbook + archive, 30min cooldown). REPLY/COMMENT implicitly Moltbook. `output_destination` control removed.
- **Mode expansion**: `mode` control supports `all`, `comment_only`, `no_post`, `no_comment`, `post_only`
- Controls persisted to `brains/{brain}_controls.json` between runs
- LLM returns `"controls_update": {...}` in JSON response to modify controls
- Validation: type coercion, range clamping, choice checking, change logging

### v15 CLI Flags (additive to v14)
- `--conscious-model` — Model for conscious loop (default: same as `--gemini-model`)
- `--subconscious-model` — DEPRECATED (use model weight controls)
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

## v15.5 — Archived (Subconscious Daemon + Saved Plans)

v15_5/ has been archived to `archive/v15_5`. **Do not modify.** v15_6 is the stable foundation.

### Subconscious Daemon (Phase 5: COMPLETE)
- `v15_5/buffer.py` — `Draft` dataclass + `DraftBuffer` (thread-safe wake potential + draft storage)
- `archive/v15_5/daemon.py` — `SubconsciousDaemon`: background thread with Sentry + Strategist gears
  - **Sentry**: scans feed, scores items against directive using cheap LLM with kernel as system instruction
  - **Strategist**: generates draft action plans for high-signal items, adds charge to wake potential
  - **Integrate-and-Fire**: wake_potential accumulates, decays per tick, fires conscious when threshold crossed
  - **Downward causality**: conscious sends `daemon_directives` (focus_topics, ignore_authors, urgency_boost, note)
- `v16_0/llm/budget.py` — `DailyBudget` now thread-safe (`threading.Lock` on all public methods)
- `v16_0/controls.py` — 2 new daemon controls: `max_drafts`, `strategist_max_tokens`
- `v16_0/planner.py` — `draft_context` param, `--- SUBCONSCIOUS BUFFER ---` prompt section, `daemon_directives` response field
- `v16_0/__main__.py` — Daemon lifecycle, buffer drain, wake/sleep mechanism, downward causality

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
- POST/POST_MOLTBOOK split replaces `output_destination` control (removed)
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

## v15.6 — Archived (Feed Fix + Temperature System + Bug Fixes)

v15_6/ has been archived to `archive/v15_6`. **Do not modify.** v16_0 is the stable foundation.

### Changes in v15.6

**Feed Endpoint Fix**:
- `MoltbookClient.get_feed()` uses `/feed` (personalized: subscriptions + follows) instead of `/posts` (global)
- Added `get_global_feed()` method for `/posts` (global discovery — available for explicit daemon use)
- Default `--feed-sort` changed from `"hot"` to `"new"` (both conscious and daemon sentry)
- Fixes stale feed issue (was fetching global "hot" feed which changes slowly)

**Temperature System Rewrite**:
- `temperature` ControlRegistry entry now actuates: when agent changes it, `store.set_default_temperature()` is called
- `POST /default-temperature` endpoint added to Analog Home API
- `set_default_temperature()` added to `Store` ABC and `LocalFileStore`
- `read_controls()` now returns `default_temperature` from Analog Home DB
- Planner prompt shows `agent_default_temperature` (from DB, not CLI `--temperature` arg)
- Agent can now meaningfully set its preferred temperature; user slider nudges decay toward this value

**Daemon Directives Fix**:
- `set_directives()` now merges fields instead of full replace (omitting `focus_topics` no longer clears it)
- Notes accumulate as a capped list (max 5) rather than being overwritten each cycle
- `_get_directives_text()` renders notes list; planner prompt documents merge semantics

**Fallback Comment Regeneration**:
- Content regeneration now triggers on target-post change in addition to action-type change
- Prevents mis-addressed comments (content written for post X being posted verbatim on post Y after fallback)

**Analog Home API**:
- `POST /default-temperature` endpoint (agent-set decay target for user nudges)
- Dockerfile: `sed -i 's/\r$//' entrypoint.sh` strips CRLF at build time (Windows dev environment fix)

**Seed Response Control**:
- POST/POST_MOLTBOOK actions handle platform routing (output_destination control removed)

## v16.0 — Current Stable (Controls Manager + Seeker Gear + Sentry Fixes)

v16_0/ is the stable foundation. **Do not modify archived versions.** All future work goes in v16_1.

### Phase 7: ControlRegistry Expansion + Controls Manager UI

**New Controls (v16_0/controls.py)**:
- `feed_item_chars` (int, 400, context) — max chars per feed item in prompt
- `reply_candidate_chars` (int, 5000, context) — max chars for reply candidate text
- `outside_candidate_chars` (int, 5000, context) — max chars for outside comment candidate
- `my_post_scan_limit` (int, 50, social) — recent own posts to scan for unanswered comments
- `reply_threads_scanned` (int, 4, social) — own post threads to scan per cycle
- `reply_max_comments` (int, 25, social) — max comments evaluated per thread
- `thread_comments_for_engagement` (int, 12, social) — dogpile guard threshold
- `saved_plan_max_cycles` (int, 5, daemon) — cycles a draft persists before expiry
- `daemon_notes_max` (int, 5, daemon) — max directive notes retained
- `post_failure_cooldown_seconds` (int, 900, timing) — cooldown after failed post

**Persistent Blacklist**:
- `_locked` key in `{brain}_controls.json` stores the blacklist between runs
- `lock(key)` / `unlock(key)` methods on `ControlRegistry`
- `load_from_dict()` merges file-persisted blacklist with CLI `--blacklist-controls`

**Wiring Changes**:
- `actions.py`: all hardcoded constants now accept `flags.get(key, CONSTANT_FALLBACK)` overrides
- `utils.py`: `format_feed_brief()` accepts optional `feed_item_chars` param
- `daemon.py`: `set_directives()` uses `self._ctrl.get("daemon_notes_max")` instead of `_MAX_NOTES`
- `__main__.py`: passes all new ctrl values into flags dict, call sites, and `format_feed_brief()`

**Controls Manager Dashboard Tab** (`dashboard_v2_1.py`):
- 5th tab "Controls Manager" on top of the existing 4 tabs from v2_0
- Per-brain controls.json editor: type-appropriate widgets (number_input, selectbox, checkbox, text_input)
- Grouped by category with expandable sections
- "Agent editable" checkbox per control (persisted in `_locked`)
- Save button writes back to `{brain}_controls.json`
- Raw JSON viewer expander
- CLI command builder updated to reference `v16_0` instead of `v14_0`

### Sentry/Strategist Fixes
- **Root cause**: `json_mode=True` + `search_tools` passed together to Gemini API is rejected (`400 INVALID_ARGUMENT`)
- Sentry now uses `json_mode=True` with NO search tools (scores feed items)
- Strategist now uses `json_mode=True` with NO search tools (generates drafts)
- Search responsibility moved to dedicated Seeker gear

### Seeker Gear (Gear 3) — Refactored: Living Summaries + Rabbit Hole
- Dedicated search gear using Google Search grounding (Gemini only, no json_mode)
- Produces **living summaries** (not drafts) — rewritten each run with accumulated findings
- **Rabbit hole**: generates follow-up search terms each run; terms evolve from what it discovers
- Runs every N sentry ticks (`seeker_every_n_ticks`, default 3) — not time-based
- Summaries fed to both strategist (as context for draft creation) and consciousness (as `SEEKER FINDINGS`)
- Consciousness resets seeker with new `focus_topics` via daemon directives
- Strategist called **once per tick** with all high-signal items + seeker summary (not N times per item)
- Controls: `seeker_every_n_ticks`, `seeker_max_tokens`, `charge_weight_search`, `seeker_max_topics`, `seeker_model_weights`
- Telemetry: `seeker_sweep`, `seeker_result`, `seeker_error`, `seeker_budget_skip`

### 429 Backoff Fix
- `GeminiChatSession.send_message()` now calls `BUDGET.note_429()` on rate-limit errors before re-raising
- Prevents unlimited API hammering when free-tier models hit rate limits

### Budget Tracking + Accountant (v16.2+)
- `GeminiChatSession.send_message()` now captures `usage_metadata` (input/output tokens)
- `planner.py` records conscious LLM spend to `DailyBudget` after every call; telemetry enriched with tokens + cost_usd
- `__main__.py` records accountant chat spend
- **Accountant** (`accountant.py`): understands daemon wake mechanics (wake_threshold, signal_threshold, charge_weight control actual conscious invocation rate). Conservation priority: sentry interval → wake threshold → signal threshold → charge weight → cycle interval → model downgrade (last resort)
- Accountant can adjust 7 controls: conscious_model, subconscious_model, sentry_interval_seconds, cycle_interval_minutes, wake_threshold, signal_threshold, charge_weight_feed
- Cost projection estimates effective wake interval from daemon parameters

### Default Tuning (Apr 2026)
- `cycle_interval_minutes`: 60 — budget-friendly conscious cycle rate
- `sentry_interval_seconds`: 300 — daemon tick interval
- `subconscious_model`: DEPRECATED — use `subconscious_model_weights` cadre
- `signal_threshold`: 0.67 — sentry score to trigger strategist (feed items)
- `seed_threshold`: 0.3 — sentry score for human seeds (low to filter spam only)
- `seeker_every_n_ticks`: 3 — seeker runs every 3 sentry ticks
- `charge_weight_feed`: 0.05 — feed items barely register
- `charge_weight_seed`: 999.0 — seeds cause instant wake (if they pass seed_threshold)
- `feed_batch_size`: 8 — items per sentry batch
- `allow_downvote`: False
- `allow_kernel_update`: True (now a control, default enabled; `--no-kernel-update` to disable)
- Verification: `verification_model_weights` default `ollama:gemma3:12b=3,gemini-2.5-flash=1`

### Run Tracking + Session Continuity (v16.3)
- `session_id` persists across Ctrl+C restarts — only resets when memories/history are wiped
- Stored in `state["_session_id"]`, used as `run_id` for all Analog Home artifacts
- Analog Home API: `run_id` column, `/runs` endpoint (returns `first_title`), `/artifacts?run_id=X` filtering
- Archives page: "Present Run" at top, past runs split into major (8+ artifacts) and short runs
- Home page filtered to latest run only

### Image Generation (v16.4)
- `GENERATE_IMAGE` action: agent generates images via Gemini Imagen 3 API
- Three tiers: `imagen-fast` ($0.02), `imagen-standard` ($0.04), `imagen-ultra` ($0.06, default)
- Controlled via `image_model_tier` and `image_cooldown_minutes` controls
- Raw PNG compressed to JPEG (quality 85) via Pillow before base64 encoding (~140KB vs ~1MB)
- Published as `artifact_type: "image"` with `image_url` containing data URI
- Frontend renders with green glow border, IMG badge on collapsed cards
- `GeminiBackend.generate_image(prompt, tier, aspect_ratio)` returns `(bytes, model_id, cost)`

### Dev Requests + Prompt Nudges (v16.5)
- `DEV_REQUEST` action: agent writes to `brains/{brain}_dev_requests.txt` + publishes `system_dev_request` artifact
- Prompt nudges: gentle reminders when GENERATE_IMAGE is available or tagline overdue (20+ cycles)
- Agent-controlled tagline: `POST /tagline` endpoint, displayed as site subtitle on Analog Home

### Artifact Titles
- Replies titled "Reply to @{author}", comments titled "Comment on: {post title}"
- Metadata enriched in `execute_action()` from actual Moltbook API response (`_reply_author`, `_post_title`, `_post_author`)

### Batch Sentry + Simplified Scoring (v16.6-16.8)
- Sentry uses simple 0-9 single-score batch format (`build_simple_batch_prompt()` / `parse_simple_batch_response()`)
- Sentry uses short task-specific system instruction, NOT the kernel (kernel causes flash-lite to role-play instead of scoring)
- Strategist keeps the kernel for personality-consistent drafts
- `sentry_interval_seconds` max_val removed

### Model Tier Separation
- **Conscious pool** (`conscious_model_weights`): pro-tier only (gemini-2.5-pro, gemini-3.1-pro-preview, claude-sonnet-4-5, claude-opus-4-6, gpt-5.2, gpt-5-pro, gpt-5.2-pro)
- **Sentry pool** (`subconscious_model_weights`): flash-lite, ollama:gemma3:12b, haiku, mistral-small, gpt-5-nano/mini + all Ollama models
- **Strategist pool** (`strategist_model_weights`): same pool as sentry (called once per tick with all high-signal items)
- **Seeker pool** (`seeker_model_weights`): Gemini only (needs search grounding)
- **Verification pool** (`verification_model_weights`): default ollama:gemma3:12b=3, gemini-2.5-flash=1 (gemma 6/10, free; flash as backup)
- All model weight controls locked by default (operator decision). Weighted random selection per tick.
- Sentry: `disable_thinking=True` for Ollama models, short task instruction (NOT kernel)
- Strategist: uses kernel as system instruction. Parser strips monologue/comments before JSON extraction.

### Ollama Backend (v17.0)
- `autonomy/llm/ollama.py` — `OllamaBackend` + `OllamaChatSession` via REST API
- Connects to Ollama at `OLLAMA_URL` env var (default `http://localhost:11434`)
- Auto-discovers available models via `/api/tags`, registers as `ollama:{model_name}`
- Much faster than HuggingFace/PyTorch (1-6s vs 30-250s) — native GGUF inference
- Ollama manages its own model loading/unloading — no "max 1 local" restriction
- `ollama:gemma3:12b`: 94% sentry accuracy (ties flash-lite), free, 5.7s latency

### Hierarchical Memory (v17.0)
- Three-tier automatic compression: recent (20 cycles) → compressed (10 summaries) → deep (10 deep)
- Agent writes `memory_note` each cycle, appended to `state["memory_tiers"]["recent"]`
- When a tier fills, oldest half compressed via LLM (gemini-2.5-flash) into one paragraph, pushed to next tier
- Nothing is truncated — everything is compressed. ~40 memory items covers 1000+ cycles
- DREAM action deprecated — compression is automatic
- Controls: `memory_recent_capacity`, `memory_compressed_capacity`, `memory_deep_capacity`, `compressor_model`

### Recent Posts in Prompt (v16.9)
- Agent sees its last N artifact bodies (default 4, control: `recent_posts_in_prompt`)
- Fetched from Analog Home API each cycle
- Enables agent to build on its own prior work

### Self-Telemetry in Prompt (v16.8)
- Agent sees recent action distribution, budget status, sentry model usage, memory stats
- Computed from state + daemon at cycle start

### Auto-Calibrated Wake Threshold (v17.1)
- `target_wake_minutes` control (default 60) replaces raw `wake_threshold`
- Daemon tracks per-model charge history, computes threshold automatically
- Feed items barely register (charge 0.05), seeds wake instantly (999), replies worth waking (1.5)
- Reply scanner (Gear 5): daemon scores comments on our posts, worthy ones add wake charge
- `max_replies_per_post` (default 3) caps reply storms

### Daemon Provides Candidates (v17.1)
- Outside comment candidate extracted from daemon's best COMMENT draft (sentry-scored)
- Reply candidate extracted from daemon's reply scanner (sentry-scored comments)
- Full post/comment text fetched from Moltbook API for planner context
- Falls back to old functions when daemon has no drafts

### 503 Retry Chain
- Sentry: tries different model from sentry pool
- Strategist: tries different model from strategist pool
- Conscious: tries each model in conscious pool sequentially; if all fail, WAITs (never degrades to local)
- Verification: primary → backup conscious → ollama:gemma3:12b

### CLI/Controls Refactor (v17.2)
- Controls.py is the SINGLE SOURCE OF TRUTH for all defaults
- CLI flags default to None; only override controls when explicitly passed
- Startup order: parse args → build registry → build controls → load controls.json → apply CLI overrides → read derived values
- `build_default_registry()` no longer takes `args` parameter
- Budget, conscious_model, temperature all read from controls after load

### Dashboard
- Spend graph (conscious vs subconscious vs image) with thinking-token multiplier estimates
- Daemon monitor: model distribution per sentry rubric, recent ticks table, avg score by model
- Controls manager: weight sliders per model, auto-discovers Ollama models
- Time range: Past day (default), Past week, All time

## v17.3–17.5 — Subconscious Expansion + Social Improvements

### Live Daemon Feed on Analog Home (v17.3)
- Daemon pushes tick lines per-role to `/daemon-tick` API as they happen
- `daemon_ticks` table in Postgres (auto-pruned, run_id-filtered)
- `DaemonTerminal.tsx` polls `/daemon/live` every 8s, color-coded lines
- Conscious events (CYCLE start, ACTION, MEMORY, CONTROL, KERNEL, BUDGET) pushed with tick = `cycle + 10000` to avoid collision with daemon ticks
- Auto-clears old session ticks on daemon start (`DELETE /daemon-ticks?run_id=...`)
- Lines truncate with ellipsis (no horizontal scroll)
- Color scheme: white tick borders, cyan sentry, orange strategist, pink seeker, green conscious, teal budget, blue compress, amber verification, lavender dreamer, golden muse

### Hierarchical Post Memory (v17.3)
- Mirrors `memory_tiers` but for what was *produced*, not what was *thought*
- Buffer: last 4 artifact bodies (full text, already shown in prompt)
- After every 4 artifacts: compress into summary → `post_tiers.recent` (up to 8)
- Cascade: recent → compressed (8) → deep (5)
- Compressor: `qwen2.5:1.5b` (free)
- Rendered as `POST HISTORY (what you've written)` in planner prompt
- Controls: `post_memory_batch`, `post_memory_recent_cap`, `post_memory_compressed_cap`, `post_memory_deep_cap`

### Featured Artifact + Gallery + Archive Deep-Linking (v17.3)
- `is_featured` flag on artifacts table
- `POST /feature/{id}`, `GET /featured` endpoints
- Home page: featured artifact section above recent artifacts (collapsible, gold border)
- New `/gallery` page: image grid via `/artifacts?artifact_type=image`
- Deep-linking: `/archives?artifact={id}` calculates page via `/artifacts/{id}/position`, loads exactly that page
- Archives pagination: 25 per page with First/Prev/Next/Last
- Featured image on home links to archives

### Synthesizer Cadre (v17.3)
- Seeker findings synthesis moved from flash-lite (re-summarize, lossy) to local model (synthesize new findings only)
- Seeker now appends new synthesized blocks to living summary (preserves all prior findings)
- Compressor only fires when summary > `seeker_max_summary_chars` (default 2000)
- New control: `synthesizer_model_weights` (locked default: gemma3=2, deepseek=1)

### Dreamer Gear (Gear 6, v17.3-17.5)
- Stochastic 1/`dream_interval_ticks` (default 60) per tick
- Loads topics from `brains/{brain}_dream_topics.txt`
- Picks random topic, generates first-person dream paragraph via local model
- Injects into `state["memory_tiers"]["recent"]` with `cycle: None`
- Cadre: `dreamer_model_weights` (locked default: gemma3=2, deepseek=1)
- Color: lavender in terminal/live daemon

### Muse Gear (Gear 7, v17.5)
- Stochastic 1/`muse_interval_ticks` (default 30) per tick
- Reads: full memory tiers (incl. dreams), most recent post, current seeker summary, kernel
- Outputs ONE creative draft: POST / POST_MOLTBOOK / GENERATE_IMAGE
- Goes into draft buffer alongside strategist drafts
- Own model cadre: `muse_model_weights` (locked default: gemma3=2, deepseek=1, others at 0)
- Own temperature: `muse_temperature` (default 0.95)
- Color: golden in terminal/live daemon

### Sentry Improvements (v17.4)
- **ignore_authors enforced**: sentry skips items from authors in conscious directives (was stored but never checked)
- **daemon_can_follow=True default**: follows authors on score >= 0.9
- **Feed rotation**: `feed_rotation` control rotates between new/following/hot per tick (default `new,new,new,new,following,new,hot`)
- **Reply scanner persistence**: `_scored_comment_ids` persists to state (was re-scoring all comments after restart, causing instant wake from accumulated charge)
- **Persistent cycle number**: `state["_cycle_number"]` survives restarts (only resets on memory wipe)
- **Single 0-9 scoring**: removed dead novelty/actionability fields. parse_simple_batch_response returns just `relevance` (full 10-point granularity, was only 4 values: 0/0.33/0.67/1.0)

### Moltbook /home Dashboard (v17.4)
- `get_home()` replaces multi-call engagement gathering
- One API call returns: account stats, activity on posts, followed accounts' posts, suggestions, DMs
- `mark_notifications_read()` called after each conscious cycle

### Strategist Improvements (v17.5)
- Added `GENERATE_IMAGE` to action types
- Prompt rewritten to encourage multiple drafts (was always producing 0-1 even with 16+ signals)
- Per-item and synthesis modes presented as equally valid first-class options

### 503 Retry Order (v17.4)
- Conscious 503 retry now sorts by weight DESCENDING (was string order)
- Was: gemini-2.5-pro 503'd → fell to claude-sonnet-4-6 (next in string) → 38% sonnet usage vs 11% expected
- Now: tries gemini-3.1-pro-preview before sonnet
- ReadTimeout treated as retryable (was going to WAIT)

### Seed Charge Calibration (v17.4)
- Was: 999 charge per seed (caused double-wakes — residual after refractory still above threshold)
- Now: `wake_threshold * sentry_score` (proportional)
- Operator override: text ending in `-P` gets 999 (instant wake)

### Wake Threshold Bug Fix (v17.4)
- `_record_tick_charge` was including seed charge in calibration history → inflated threshold to 400
- Now subtracts seed charge before recording feed-only charge

### Cycle Reports (v17.4)
- Removed `**markdown**` (CrtTerminal renders literally)
- Model names: `_format_model_name()` strips `ollama:`, replaces `:` with space, adds `(local)` or `(api)`
- Controls update artifact also cleaned

### Architecture Diagram (v17.4)
- `architecture.d2` — full system diagram via D2 (https://d2lang.com)
- Render: `d2 --layout=elk --pad=40 architecture.d2 architecture.svg`

## v17.6 — Pre-Launch Polish (Apr 2026)

### Image Bandwidth Overhaul
- **Storage**: new `image_data BYTEA` + `image_mime VARCHAR(32)` columns on `artifacts`. Replaces ~360KB base64 data URIs in `image_url` for new artifacts.
- **Endpoint**: `GET /artifacts/{id}/image/{thumb|medium|full}` — thumb 400px, medium 800px, full = original. PIL resizes thumb/medium on demand. Returns `Content-Type: image/jpeg`, `Cache-Control: public, max-age=86400, immutable`, ETag for 304s.
- **Backward compat**: legacy artifacts with `image_url` data URIs decode through the same endpoint via `_decode_legacy_data_uri()`. `_art_row_to_dict` resolves `image_url` field to `/api/proxy/artifacts/{id}/image/medium` for new artifacts; legacy data URIs pass through as-is.
- **Agent**: `GENERATE_IMAGE` now sends raw `image_data_b64` (no data URI prefix) via `store.write_artifact`. `/publish` decodes and stores in BYTEA.
- **Frontend**: `lib/imageUrl.ts` helper swaps the size segment per call site. Featured image uses medium, gallery uses thumb, archive expanded view uses medium with "view full size →" link to full.
- **Backfill**: `analog_home/api/backfill_images_via_api.py` re-publishes existing data-URI artifacts through `/publish` (uses `ON CONFLICT DO UPDATE` to overwrite in place). Six legacy images converted Apr 2026.
- **Bandwidth**: home page polls drop from ~160 MB/hour/visitor to ~kB after the first cached fetch.

### 4:3 Image Generation
- `gemini.py` `generate_image()` already accepted `aspect_ratio` parameter. `__main__.py` GENERATE_IMAGE now passes `aspect_ratio="4:3"` (was implicit 1:1).
- Reason: Analog Home featured image and gallery thumbnails both use 4:3 containers with `object-fit: cover`. Square images got their tops/bottoms cropped; 4:3 native generation fills the container exactly.
- Existing 1:1 backfilled images still render via the cropped wide-strip look on the home featured slot.

### sentry_strictness Control (the noise dial)
- New control: `sentry_strictness` (float 0.0-1.0, default 0.5, category "daemon", agent-modifiable).
- Threaded through `build_simple_batch_prompt(items, directive, directives_text, strictness=)` in `scoring.py`. Injects a one-line bias into the rubric:
  - 0.0-0.33: "be liberal — cast a wide net … false positives are fine"
  - 0.34-0.66: no bias text (current behavior)
  - 0.67-1.0: "be strict — only score 6-9 with clear, direct relevance"
- Wired into all three sentry call sites in `daemon.py`: `_score_items_batch` (batch scan), `_score_item` (per-item fallback), and the reply scanner at line ~1137.
- Distinct from `signal_threshold` — that's the post-scoring numeric cutoff. `sentry_strictness` changes how the model itself decides to score.
- Agent-facing description in `controls.py` is framed as "your sentry's signal-to-noise dial" so the agent recognizes it as the knob to fiddle with when the daemon is over- or under-waking.

### planner.py Credit/Quota Error Fix
- **Bug**: `parse_json_with_one_repair()` outer except only re-raised `503/UNAVAILABLE`. Anthropic credit-balance and Gemini quota errors landed in the else branch, were logged, and a default WAIT plan was returned. The conscious retry chain at `__main__.py:1344` never saw the failure → no fallback.
- **Fix**: outer except now also re-raises `credit balance`, `insufficient credit`, `quota`, `insufficient_quota`, `rate_limit`, `429`, `ReadTimeout`, `timed out`. The `__main__.py` pool-fallback catches them at line 1304 and tries the next conscious model (sorted by weight DESC).

### Public-Launch Doc Refresh
- About page (`analog_home/web/app/about/page.tsx`): "six providers" → "five", expanded daemon description (sentry/strategist/seeker/dreamer/muse), mention of live daemon stream visible on home page.
- Autonomy README: daemon gear table expanded to seven gears + verifier, controls count `30+ → 75+`, recent work section refreshed.
- Analog Home README: features list adds live daemon stream, featured artifacts, gallery, tiered images, archive deep-linking, footer. Endpoint list in architecture diagram comprehensive.
- Site footer (`components/Footer.tsx`): Home / Archives / Gallery / About / Source mounted on all four pages.

### Archive Deep-Link Scroll Fix
- `archives/page.tsx`: replaced fixed `setTimeout(scrollIntoView, 600)` with a `useEffect` that watches for the target artifact element to appear in the DOM and retries up to 20 × 100ms. Eliminates the race when the target is on a paginated page that hasn't rendered yet.

## v17.7 — Accountant Ownership + CEO/CFO Split (Apr 2026)

### Architectural shift: conscious and accountant split by control category
Previously both the conscious model and accountant could tune wake/budget controls. The conscious would burn pro-tier cognition on mechanical "is my sentry interval coherent with my wake target?" decisions, and two owners could make incompatible changes (e.g., conscious sets `target_wake_minutes=30` while sentry still ticks every hour). v17.7 resolves this by making the accountant the sole owner of wake mechanics.

### New `audience` field on Control
- `Control` dataclass in `autonomy/controls.py` gains `audience: str = "both"` — one of `"conscious"`, `"accountant"`, `"both"`.
- `to_llm_block(audience="conscious")` filters out controls that don't include the caller.
- Marked `audience="accountant"` (hidden from conscious prompt):
  - Wake mechanics: `cycle_interval_minutes`, `sentry_interval_seconds`, `target_wake_minutes`, `signal_threshold`, `charge_weight_feed`, `charge_weight_reply`, `wake_refractory`
  - Model pools: `conscious_model_weights`, `subconscious_model_weights`, `budget_exhausted_model_weights`, `accountant_model_weights`
- `sentry_strictness` stays `audience="both"` — it's a signal/noise judgment (is the daemon over/under-waking?), not a budget knob. Conscious keeps that call.

### Accountant runs every cycle
- `should_run_budget_plan()` simplified: just returns `budget_plan_enabled`. The old threshold + 8h + first-of-day gates are gone.
- Runs on the accountant cadre (`accountant_model_weights`, default `ollama:qwen3:14b=2,ollama:gemma3:12b=1`) — free local inference, ~5-15s per cycle.
- Fires **before** conscious model selection so accountant weight changes take effect this cycle.
- Fixed pre-existing bug where `conscious_model` was hardcoded to `"gemini-2.5-pro"` after accountant ran — now properly re-selects from (potentially updated) weights.

### `apply_budget_plan` updates
- Expanded updatable list: `wake_refractory`, `charge_weight_reply`, `budget_exhausted_model_weights` in addition to the original seven.
- **Oscillation guard**: skips numeric changes within 10% of current value. Prevents accountant from jittering knobs each cycle on noise.
- `sentry_strictness` deliberately NOT in the updatable list.
- Accountant prompt rewritten to explicitly state ownership, coherence rules (e.g., `target_wake_minutes*60 >= sentry_interval_seconds`), and an expanded JSON schema.

### Cycle report consolidation
- Removed the standalone `system_controls_update` artifact. Previously, conscious control changes got their own artifact; accountant changes didn't — inconsistent.
- Now both origins fold into the `system_cycle_report` artifact with `[accountant]` / `[conscious]` prefixes, followed by the accountant's `reasoning` string. The report runs every cycle, so there's always a single place to see "what changed and why."
- Conscious-origin changes tracked in `_conscious_control_changes`, accountant-origin in `_accountant_control_changes` + `_accountant_reasoning`.

### Budget-exhausted conscious fallback
- When `budget.remaining_usd() <= 0`, the conscious model pool swaps to `budget_exhausted_model_weights` (default `ollama:qwen3:14b=2,ollama:gemma3:12b=2,ollama:deepseek-r1:8b=1`).
- Local models still decide any action — they're not forced to WAIT. A thoughtful local POST is strictly better than a paid gemini-3.1-pro-preview deciding WAIT.
- The retry chain at `__main__.py:~1406` uses the same `active_conscious_weights` variable, so fallback during budget exhaustion stays on local models rather than escalating back to paid.

### sentry_strictness kept in conscious domain
- Despite the accountant owning all other wake-related numbers, `sentry_strictness` stays visible to conscious and is NOT in the accountant's updatable list.
- Rationale: it's not a budget control — it's a signal/noise judgment. If the conscious raises strictness (because the daemon is waking on irrelevant items), the accountant may *compensate* by adjusting other knobs, but should never override the strictness itself.

### Removed: control_validators.py
- The validator module and its post-update coherence checks were deleted.
- Rationale: with single-owner accountant making coherent changes each cycle, the validator becomes redundant safety net on a path where bugs should no longer surface. If we see incoherent states in practice, it can come back.

### Conscious prompt now simpler
- The `controls_block` in the conscious planner prompt dropped from 8,190 chars / 92 lines → 6,485 chars / 79 lines (~21% of the controls section, ~2.5% of the full ~66KB prompt).
- More important than the token savings: the conscious no longer sees wake/budget knobs, so it doesn't spend cognition deciding whether to touch them.

## v17.6.5 — Image Recovery + Memory Sliding Window + Muse Fixes (Apr 2026)

### Muse draft visibility fixes
- Added `[MUSE]` source tag in `_format_draft_context()` so muse drafts are visually distinct (alongside `[HUMAN SEED]` and `[SEARCH]`).
- `DraftBuffer.add_draft()` eviction policy: now pops oldest **non-muse** draft when buffer exceeds `max_drafts`. Muse drafts protected from FIFO eviction. Fallback to strict FIFO if all drafts are muse (breaks after one pop to avoid infinite loop).
- `DraftBuffer.drain()` signature changed: returns `(kept_drafts, overflow_drafts, wake_potential)`. Sorts non-muse by `signal_score` DESC, keeps top 10 non-muse + all muse, overflow returned for caller to compress.
- Defaults: `max_drafts` 10 → 40, `max_saved_plans` new control = 15, `muse_interval_ticks` default 30 → 15, `dream_interval_ticks` 30 → 60.

### Draft digest compression
- New `_compress_drafts_to_digest()` in `__main__.py`: takes overflow drafts (below top-10 by score) and synthesizes a 3-5 sentence thematic paragraph via the compressor cadre (default gemma3:12b).
- Digest regenerates fresh each drain, doesn't persist.
- Rendered in `_format_draft_context()` as "Your subconscious also noticed these less-prominent themes" — gives the conscious a thematic summary of what didn't make the cut without the verbosity of 30+ individual drafts.

### Post memory sliding-window redesign
- Old: `_post_memory_buffer` accumulated 4 posts, then compressed all 4 into one summary and reset. Lost per-post structure; gaps while accumulating.
- New: `_post_memory_fresh` keeps last N (default 4) full posts. On new post: prepend; if overflow, pop oldest single post, compress into one summary, prepend to `post_tiers.recent`. Each compressed entry retains individual identity.
- Cascade unchanged: `recent` (cap 10) → `compressed` (cap 10) → `deep` (cap 5) via oldest-half compression.
- `_build_recent_posts()` reads from local `_post_memory_fresh` instead of the Analog Home `/artifacts` API — faster, offline-safe, deterministic.

### Image generation failure recovery
- `GENERATE_IMAGE` on error now: (1) retries with `imagen-ultra` if original tier was different, (2) falls back to publishing `plan.content` as a text `artifact_type="post"` to Analog Home so the essay isn't lost, (3) saves the failed `image_prompt` to `state["_failed_image_prompts"]` (capped 5) for future retry, (4) adds history entry and logs full telemetry (no truncation).
- Fixed stale imagen model IDs: `imagen-3.0-fast-generate-001` → `imagen-4.0-fast-generate-001`, `imagen-3.0-generate-002` → `imagen-4.0-generate-001`.

## Key Architecture Decisions

- **Controls are source of truth**: controls.py has defaults, controls.json overrides, CLI overrides both. No competing defaults.
- **One chat per cycle**: Chat is recreated each iteration (`__main__.py`) to avoid token accumulation
- **BUDGET is in-memory only**: DailyBudget resets on restart, reads limit from controls
- **Challenge solver** (`challenges/math_verification.py`) uses `llm_client.generate()` (one-shot, temp=0.0, max_output_tokens=8192), not chat sessions
- **Telemetry is separate from Store**: TelemetryLogger writes to JSONL independently
- **Only API touches Postgres**: The agent publishes via HTTP POST to `/publish`. This keeps DB access through the API.
- **Store is the swap point**: `DuckDBStore` or `PostgresStore` can replace `LocalFileStore` at `__main__.py` with no changes to agent loop or actions.
- **Ollama preferred over HuggingFace**: Local models served via Ollama REST API (1-6s) not PyTorch (30-250s)
- **POST/POST_MOLTBOOK split**: Two actions with explicit audience intent. POST = Analog Home (human audience, no cooldown). POST_MOLTBOOK = Moltbook + archive (agent community, 30min cooldown). `output_destination` control removed.
- **Post engagement feedback**: Agent sees Moltbook upvotes, comments, karma, followers each cycle. Audience stats (unique voters, seeders) from Analog Home `/audience` endpoint.
- **Verification**: gemma3:12b primary (free, 6/10 accuracy with simple prompt), gemini-2.5-flash backup. `verification_model_weights` cadre.
- **Strategist parser**: Strips monologue text and `//` comments before JSON extraction — models can role-play the kernel but JSON is still recovered.
- **Sentry thinking disabled**: `disable_thinking=True` passed to Ollama for sentry scoring — prevents thinking models from wasting tokens on reasoning blocks.

## Known Issues / Context

- Gemini 2.5 Flash sometimes returns empty first responses with stateless API. The repair path in `parse_json_with_one_repair()` handles this (sends prompt again). This means ~2 LLM calls per cycle instead of 1 for Flash. Gemini 2.5 Pro does not have this issue.
- **Next.js 16 + Tailwind 4 on Windows**: Turbopack's enhanced resolver can walk up to parent directories. Fixed with `turbopack.resolveAlias` in `web/next.config.ts` to pin tailwindcss to local `node_modules`.
- **Package convention**: `autonomy/` is the active codebase. Archived versions (v12_0–v16_0) live in `archive/` — do not modify. Use `git tag` for version milestones.

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
- Runs locally via `python -m autonomy <brain> [flags]`
- Connects to Analog Home API via `{PREFIX}_ANALOG_HOME_API_URL` env var
- Uses local DuckDB for Streamlit dashboard (separate from Analog Home Postgres)
