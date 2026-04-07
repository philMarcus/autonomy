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
- `--subconscious-model` — Model for daemon (default: `gemini-2.5-flash-lite`)
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
- Agent-controlled tagline (subtitle under "Analog_I")
- Temperature slider with ±0.5 clamping, 429 error handling
- API endpoints: `/runs`, `/artifacts?run_id=X`, `/tagline`
- API endpoints: `/runs` (list runs with metadata), `/artifacts?run_id=X` (filter by run)

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
- Planner prompt clarifies `output_destination` control for directing seed responses to Analog Home vs Moltbook
- `output_destination` is actuated immediately (affects current cycle), unlike most controls (affect next cycle)

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

### Seeker Gear (Gear 3)
- Dedicated search gear that uses Google Search grounding (no json_mode)
- Searches `focus_topics` from conscious directives on a configurable cadence (default 15min)
- Results go directly to strategist for draft generation (bypass sentry scoring)
- Drafts tagged with `source="search"`, displayed as `[SEARCH]` in conscious prompt
- New controls: `seeker_interval_seconds`, `seeker_max_tokens`, `charge_weight_search`, `seeker_max_topics`
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
- `cycle_interval_minutes`: 60 (was 5) — budget-friendly conscious cycle rate
- `sentry_interval_seconds`: 300 (was 60) — halves daemon LLM calls
- `subconscious_model`: `gemini-2.5-flash-lite` (was `gemini-2.5-flash`)
- `wake_threshold`: 3.0 (was 2.0) — prevents constant daemon waking
- `charge_weight_feed`: 0.3 (was 0.5) — feed items less likely to trigger wake
- `feed_batch_size`: 8 (was 12) — fewer sentry evaluations per tick
- `allow_downvote`: False (was True)
- `allow_kernel_update`: True (now a control, default enabled; `--no-kernel-update` to disable)

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
- **Conscious pool** (`conscious_model_weights`): pro-tier only (gemini-2.5-pro, gemini-3.1-pro-preview, claude-sonnet/opus, gpt-5.1+)
- **Sentry pool** (`subconscious_model_weights`): flash-lite=3, ollama:gemma3:12b=1, haiku=1, mistral-small=1
- **Strategist pool** (`strategist_model_weights`): mistral-small=2, flash-lite=2
- **Seeker** (`seeker_model`): Gemini only (needs search grounding)
- Weighted random selection per tick. Agent can adjust weights. Max 1 `local:` model enforced (not `ollama:`)
- Sentry uses short task instruction (NOT kernel) to prevent models from role-playing
- See `local_model_research.md` and `sentry_eval_v2_results.json` for benchmark data

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

### Dashboard
- Spend graph (conscious vs subconscious vs image) with thinking-token multiplier estimates
- Daemon monitor: model distribution per sentry rubric, recent ticks table, avg score by model
- Controls manager: weight sliders per model, all pool models shown (0-weight locked by default)
- Time range: Past day (default), Past week, All time

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
