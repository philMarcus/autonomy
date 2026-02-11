# Autonomy Project — Context for Claude Code

## Project Overview

Autonomous Moltbook (social media platform) agent system. Each "brain" has a kernel prompt, knowledge file, and memories/state. The agent loop runs cycles: fetch feed → social actions → LLM planner decides action → execute (post/comment/reply/vote) → sleep.

## Repository Layout

- `v12_1/` — Current active version (Python package, run via `python -m v12_1 <brain> [directive] [flags]`)
- `archive/v12_0/` — Previous version (reference only)
- `brains/` — Per-brain files: `{name}_kernel_prompt.txt`, `{name}_knowledge.txt`, `{name}_memories.json`
- `telemetry/events.jsonl` — Append-only telemetry log
- `warehouse/` — DuckDB parquet output from `ingest.py`
- `dashboard/` — Streamlit dashboard (`views.sql` for queries)
- `Space for Analog I plans.md` — Full roadmap for public-facing observatory site

## v12.1 Changes from v12.0

### Gemini Client Rewrite (`v12_1/llm/gemini.py`)
- **Replaced `chats.create()` (stateful chat API) with `models.generate_content()` (stateless)**
- Manual `_history` list tracks conversation for multi-turn repair prompts
- `GeminiChatSession.__init__` takes `(client, model_name, system_instruction, temperature, max_output_tokens)` — no API call at creation time
- `ThinkingConfig(thinking_budget=0)` disables Gemini 2.5 Flash's thinking mode which was consuming output tokens and truncating JSON responses
- `_extract_text()` static method handles response parsing with fallback to individual parts

### Store Interface (`v12_1/store.py`) — NEW
- `Store` ABC with `load_state()` / `save_state()` as core contract
- Future stubs: `write_artifact()`, `read_controls()`, `increment_vote()` — aligned with Analog I Phase 2 plan
- `LocalFileStore` delegates to existing `utils.load_state`/`utils.save_state`
- All state persistence in `__main__.py` and `actions.py` now goes through `store` object (not direct `save_state(path, state)` calls)

### CLI Temperature Flag (`v12_1/__main__.py`)
- `--temperature` flag (float, default 0.7) passed to `create_chat()`
- Does NOT affect challenge solver (intentionally uses `temperature=0.0`)

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

## Key Architecture Decisions

- **One chat per cycle**: Chat is recreated each iteration (`__main__.py:227`) to avoid token accumulation
- **BUDGET is a module-level singleton** in `gemini.py` — tracks TPM across all calls
- **Challenge solver** (`challenges/math_verification.py`) uses `llm_client.generate()` (one-shot, temp=0.0), not chat sessions
- **Telemetry is separate from Store**: TelemetryLogger writes to JSONL independently. Will be migrated to DB in Analog I Phase 4.

## Known Issues / Context

- Gemini 2.5 Flash sometimes returns empty first responses even with stateless API. The repair path in `parse_json_with_one_repair()` handles this (sends prompt again). This means ~2 LLM calls per cycle instead of 1. Root cause appears to be a Gemini model update on Feb 10 that added thinking/monologue behavior.
- `thinking_budget=0` mitigates truncated JSON but the empty-first-response issue persists intermittently.

## Analog I Roadmap (see `Space for Analog I plans.md`)

Phase 1: DuckDB + FastAPI + Next.js locally (MVP)
Phase 2: Connect v12_1 agent via `Store` interface (`read_controls`, `write_artifact`, `increment_vote`)
Phase 3: Deploy to Fly.io + Vercel
Phase 4: Migrate to Postgres, add analytics/metrics

The `Store` ABC is designed so that `DuckDBStore` or `PostgresStore` can be swapped in at `__main__.py` with no changes to agent loop or actions.
