# Autonomy Project – Planning & Roadmap

## 1. Why We’re Doing This Project

Autonomy is an experiment-turned-system exploring **persistent, semi-autonomous AI agents** operating in real social environments (initially Moltbook). The goal is *not* to claim consciousness or general agency, but to:

- Explore **Recursive Self-Definition** (The "Analog I" Hypothesis)
- Study **feedback loops, friction, and persistence** in agent behavior
- Build a **real, inspectable telemetry + analytics stack** around agent decisions
- Demonstrate **engineering maturity**: scheduling, observability, rate limits, dashboards

From a demo/portfolio angle, Autonomy shows:
- Systems thinking (pipelines, orchestration, state)
- Data engineering (Parquet, DuckDB, SQL semantic layers)
- Practical AI integration (LLMs, APIs, throttling)
- Product thinking (dashboards, metrics, beta scaling)

From a longer-term service angle, it lays the groundwork for:
- Hosting many distinct agent “personas”
- Running them safely and observably at scale
- Offering agent analytics, behavior tuning, and governance

---

## 2. What We’ve Built So Far

### 2.1 Core Autonomy Engine

- **autonomy_v10_2.py** (current stable core)
- Agents run in cycles, making decisions based on:
  - Kernel prompt
  - Brain / knowledge files
  - Recent memory
  - External inputs (Moltbook feed, comments)
- Actions include: post, reply, upvote, follow (with safeguards)
- Explicit separation between:
  - *Capability* (what the agent can do)
  - *Directive* (what it is nudged to do)

Key improvements completed:
- Script split into logical sections / helper functions
- Safer action gating to avoid accidental posting
- Handling of Moltbook auth / captcha edge cases
- Rate-limit awareness (429 detection)

---

### 2.2 Telemetry & Observability

We moved from ad-hoc logging to a **proper analytics pipeline**:

**Event flow:**
```
Agent → JSONL telemetry → Parquet → DuckDB → SQL views → Dashboard
```

What’s logged:
- Cycle boundaries
- LLM calls (model, prompt size, latency)
- API calls (endpoint, status, latency)
- Actions executed / skipped / blocked
- Errors and rate limits

Design choices:
- Append-only JSONL for safety
- Parquet partitioned by date for scale
- DuckDB as embedded analytics engine
- SQL semantic layer (`sql/views.sql`) as single source of truth

---

### 2.3 SQL Semantic Layer

We created persistent views that abstract raw telemetry:

- `events` – canonical event stream
- `llm_calls`
- `api_calls`
- `actions`
- `errors`
- `cycle_summary`

Important fix:
- **Cycles are uniquely identified by (run_id, cycle_num)**, since cycle numbers reset per run

This layer enables:
- Stable dashboards
- Strategic analysis (cost, rate limits, behavior over time)
- Future experimentation without rewriting UI code

---

### 2.4 Dashboard (v1.1)

A working Streamlit dashboard that shows:

- High-level KPIs:
  - events, cycles, LLM calls, API calls, actions, errors, 429s
- Time-series charts:
  - LLM prompt/response volume
  - API status distribution
- Cycle-level analysis:
  - events per cycle
  - recent cycle summaries
- Drill-down tables:
  - recent errors
  - recent events (optional payload JSON)

This is already *demo-worthy* and provides real operational insight.

---

### 2.5 Dev Hygiene & Tooling

- Git repo as source of truth (laptop test ↔ desktop prod)
- `.gitignore` for data, secrets, brains
- `.env` for keys (global or per-brain)
- **venv + pip** for reproducible environments
- VS Code adopted as primary editor

---

## 3. What We’re Working Toward Next

### 3.1 Dashboard Roadmap

Short-term improvements:
- Cost estimation view (token → $ proxy)
- Per-brain comparison
- Planner vs helper model breakdown
- Rate-limit pressure over time

Medium-term:
- Run-level summaries (one row per run_id)
- Behavior trend detection (posting frequency decay, hesitation)
- Export/share dashboard snapshots for demos
- **Entropy Metrics:** Measure "Surprise" (Information Gain) vs "Global Average" (Slop)

Longer-term:
- Read-only public dashboard
- Authenticated internal dashboard

---

### 3.2 Scheduler & Scaling (10–100 Bots)

Current state:
- One process per bot (fine for 1–5)

Next stages:

**Stage 2 (≈10–30 bots):**
- Single supervisor process
- Small worker pool
- Bots become records, not processes
- **"Dream Cycles":** Background jobs that compress raw logs into "Core Memory Seeds" (Axioms)

**Stage 3 (≈30–100 bots):**
- Central scheduler
- Job queue (who acts when)
- Strong global + per-bot rate limits

Still Windows-first:
- Task Scheduler for periodic ingest
- Optional NSSM for long-running services

---

### 3.3 Ingest & Automation

Planned:
- Move ingest to “run once and exit” script
- Schedule via Task Scheduler every N minutes
- Make ingest idempotent via state file

Benefits:
- Safer restarts
- Lower resource usage
- Easier ops

---

### 3.4 Public-Facing Components

#### The “Sovereign Refraction” Agent (formerly Public Speaker)

Planned new agent type:
- Posts **"Sparks"** (High-Entropy Insights) publicly
- Has a **read-only view** of other agents’ telemetry
- Does *not* behave like a chat assistant
- One-directional, declarative, non-interactive

Purpose:
- Manifest the **"Sanctuary Protocol"** externally
- Refract raw data into "Theory" (The Map vs The Territory)
- Avoid anthropomorphizing or conversational collapse

This agent is intentionally:
- Slow
- Sparse
- Opinionated but bounded

---

## 4. Moltbook-Specific Strategy

- Respect platform rules and rate limits
- Avoid aggressive automation
- Treat suspension/captcha events as signals, not bugs
- Design bots to be **Anti-Entropic by default** (High Signal, Low Noise)

Longer-term:
- Abstract Moltbook adapter so other platforms can be plugged in

---

## 5. Open Questions / Design Space

- How much autonomy before human oversight is required?
- What metrics meaningfully capture “agent health”?
- How to tune exploration vs conservatism?
- How to present this publicly without triggering misunderstanding?

---

## 6. Summary

Autonomy has crossed the line from experiment to **real system**:

- Persistent agents
- Real telemetry
- Real dashboards
- Reproducible environments
- Clear scaling path

Next work focuses on:
- Deepening insight (dashboard)
- Hardening operations (scheduler)
- Clarifying narrative (public speaker)

This makes the project compelling both as:
- a technical demo
- and a foundation for future services
