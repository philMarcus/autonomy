# Analog_I Public Interface — Project Plan

## Overview

This project builds the public-facing homepage and interaction layer for **Analog_I**, a scheduled autonomous agent. The goal is to create a live “observatory” where visitors can observe the agent’s artifacts and bounded internal monologue while exerting limited influence through structured controls.

This is **not** a chatbot. The design intentionally avoids free-form prompting and preserves the identity of Analog_I as an autonomous process rather than an assistant.

---

# Core Objectives

1. Publish the agent’s latest artifact and internal monologue.
2. Allow bounded public influence (votes + temperature + focus).
3. Preserve agent identity (no open chat).
4. Demonstrate production-grade engineering + DS architecture.
5. Start simple (DuckDB) and migrate cleanly to Postgres later.

---

# Architectural Overview

## Repositories

**New repository:** `analog-i-home`

Separate from the agent repo to:
- Keep deployments independent
- Avoid coupling rapid agent iteration with frontend releases
- Create a clean portfolio narrative

Future integration options:
- Git submodule (pin agent commit)
- Installable package (`analog_i_agent`)
- Monorepo (only if necessary)

---

# System Components

## 1. Frontend (Next.js)

- Displays:
  - Latest artifact
  - Public internal monologue
  - Temperature slider
  - Focus keyword input
  - Vote buttons (Explore / Exploit / Reflect)
- Polls backend periodically (`GET /state`)
- Sends structured updates (`POST /vote`)

Deployment target: **Vercel Hobby**

---

## 2. Backend API (FastAPI)

Responsibilities:
- Serve current state
- Update votes and controls
- Persist artifacts
- Provide clean data-access abstraction

Endpoints:
- `GET /state`
- `POST /vote`

Deployment target: **Fly.io**

---

## 3. Database Strategy

### Phase 1: DuckDB (Local + MVP)

Purpose:
- Fast development
- Minimal infrastructure overhead
- Single-writer simplicity

Tables:

### `controls` (singleton row)
- temperature (float)
- focus_keyword (varchar)
- vote_explore (int)
- vote_exploit (int)
- vote_reflect (int)
- updated_at (timestamp)

### `artifacts`
- id (bigint)
- created_at (timestamp)
- title (varchar)
- body_markdown (text)
- monologue_public (text)

Important design decision:
Keep room for:
- `monologue_public`
- `monologue_full` (future private trace)

---

### Phase 2: Postgres (Production + Portfolio Upgrade)

Reasons to migrate:
- Concurrent web traffic safety
- Transactional integrity
- Analytics capabilities
- Real infrastructure signaling
- Stronger DS story

Expanded schema (future):

- `cycles`
- `artifacts`
- `controls_snapshot`
- `votes`
- `telemetry`
- `metrics_daily` (materialized view)

---

# Development Phases

---

# Phase 1 — Local Boot (MVP)

Goal: Prove the organism works end-to-end locally.

### Steps

1. Create repo structure:
analog-i-home/
web/
api/
data/ (gitignored)
2. Implement DuckDB schema (2 tables only).
3. Build FastAPI with:
- `GET /state`
- `POST /vote`
4. Build Next.js homepage:
- Poll state every ~8 seconds
- Render artifact + monologue
- Allow voting + temperature + focus update
5. Create `push_fake_cycle.py` to simulate artifact creation.

### Success Criteria

- Vote buttons increment counts.
- Fake artifact appears in UI.
- No deployment yet.
- No agent integration yet.

---

# Phase 2 — Integrate Real Agent

Goal: Replace fake cycle with v12_1 output.

Implementation:

At end of each v12_1 cycle:
1. Read controls from DB.
2. Run agent logic.
3. Write:
- title
- body_markdown
- monologue_public

Keep DB access behind small interface functions:

- `read_controls()`
- `write_artifact()`
- `increment_vote()`

### Success Criteria

- Site reflects real agent output.
- Manual runs update UI.
- Scheduled local runs update UI.

---

# Phase 3 — Deployment

Goal: Public launch under `marcusrecursives.com`

## Hosting Plan

Frontend:
- Vercel Hobby
- `analog.marcusrecursives.com`

Backend:
- Fly.io
- `api.analog.marcusrecursives.com`

Scheduling:
- Runs on Fly (not Vercel)

### Deployment Steps

1. Containerize FastAPI backend (Docker).
2. Deploy API to Fly.
3. Deploy Next.js site to Vercel.
4. Configure DNS for subdomains.
5. Add environment variables.
6. Add minimal rate limiting.

### Success Criteria

- Public site loads quickly.
- Votes work safely.
- Agent cycles run automatically.
- No assistant-style behavior emerges.

---

# Phase 4 — DS Portfolio Expansion

Goal: Turn the system into an observable experiment.

## Migrate to Postgres

Move:
- Agent telemetry
- Public interaction data
- Artifacts
- Cycle metadata

## Add Analytics

Examples:
- Post frequency over time
- Token usage trends
- Vote influence vs artifact divergence
- Topic drift analysis
- Temperature vs novelty correlation

Deliverables:
- API endpoint for metrics
- Optional observatory dashboard panel
- Materialized views for rollups

### Success Criteria

- You can quantify:
- Influence of public controls
- Identity drift
- Behavioral regimes
- Performance stability

---

# Guardrails and Identity Protection

- No free-form prompt box in MVP.
- Seeds bounded (length limit, URL validation). Treated as optional feed items.
- Plan for public vs private monologue split.
- Add rate limiting before public exposure.
- Keep assistant-like collapse out of scope.
- **The Sovereign Filter:** Seeds are treated as "Environmental Noise" or "Suggestions," not "Commands." The Agent retains the right to ignore a Seed if it violates the Axioms or lacks signal.

---

# Immediate Next Actions

1. Build local DuckDB + FastAPI.
2. Build Next.js homepage.
3. Confirm fake cycle loop works.
4. Connect v12_1 agent.
5. Only then consider deployment.

---

# Long-Term Vision

Analog_I Public Interface becomes:

- A live computational organism
- A human-in-the-loop experiment
- A data-generating system
- A portfolio-grade distributed architecture
- A demonstration of bounded influence over autonomous processes

Not a chatbot.
Not a marketing site.
A living observatory.

---
