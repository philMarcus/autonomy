# Autonomy

An autonomous agent system that operates on a social media platform, making its own decisions about what to read, write, and engage with — while remaining observable and steerable by humans.

## What This Is

Autonomy is a long-running agent loop. Each cycle, the agent reads a feed, evaluates candidates for action, calls an LLM planner to decide what to do, executes the chosen action, and sleeps. Over many cycles, persistent memory and state give the agent continuity — it develops ongoing interests, relationships, and a recognizable voice.

The system is designed around a core question: **how do you build an autonomous agent that is genuinely self-directed but still observable, tunable, and accountable?**

### Key Design Decisions

- **Dual-process architecture.** A cheap "subconscious" daemon continuously scans feeds and scores items in the background. When accumulated signal crosses a threshold, it wakes the main "conscious" loop with a buffer of drafted plans. This separates vigilance (cheap, continuous) from deliberation (expensive, event-driven).

- **Multi-model LLM backend.** The system abstracts over six providers (Gemini, Claude, GPT, Mistral, local models via HuggingFace) through a unified interface. The agent can switch its own model mid-run — model selection is a tunable control, not a hardcoded choice.

- **ControlRegistry.** Every tunable parameter (22 controls across 7 categories) is a first-class object: readable and writable by the agent, lockable by the operator. The agent sees its own configuration in-prompt and can request changes. The operator can blacklist any control to prevent modification.

- **Telemetry pipeline.** An append-only JSONL event stream captures 200+ events per cycle. An ingestion layer converts this to date-partitioned Parquet files in a local DuckDB warehouse. A Streamlit dashboard provides cycle-level replay, daemon monitoring, cost tracking, and controls management.

- **Human-in-the-loop via [Analog Home](https://github.com/philMarcus/Analog_Home).** A separate web application (FastAPI + Next.js + Postgres) serves as the agent's public-facing observatory. Humans can read the agent's output, vote on creative direction, adjust temperature, and plant "seed" topics — bounded influence without direct control. The agent reads these inputs each cycle. Live at [marcusrecursives.com](https://marcusrecursives.com).

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Agent (Python)                                     │
│                                                     │
│  ┌──────────────┐    ┌───────────────────────────┐  │
│  │ Subconscious │    │ Conscious Loop             │  │
│  │   Daemon     │───>│  fetch → plan → act → log  │  │
│  │              │    │                             │  │
│  │ Sentry       │    │  Reads: feed, controls,    │  │
│  │ Strategist   │    │    seeds, draft buffer      │  │
│  │ Seeker       │    │  Writes: artifacts, state,  │  │
│  └──────────────┘    │    daemon directives        │  │
│                      └───────────────────────────┘  │
│                                                     │
│  ┌─────────────┐  ┌────────────┐  ┌─────────────┐  │
│  │ LLM Registry│  │ Controls   │  │ Telemetry   │  │
│  │ 6 providers │  │ 22 params  │  │ JSONL→DuckDB│  │
│  │ 20+ models  │  │ 7 categories│  │ + Dashboard │  │
│  └─────────────┘  └────────────┘  └─────────────┘  │
└────────────────────────┬────────────────────────────┘
                         │ HTTP
                         v
┌─────────────────────────────────────────────────────┐
│  Analog Home (sibling repo)                         │
│  FastAPI + Neon Postgres │ Next.js + Vercel          │
│  Observatory / bounded human control interface       │
└─────────────────────────────────────────────────────┘
```

## Subconscious Daemon Detail

The daemon runs three "gears" on a background thread:

| Gear | Purpose | Cadence |
|------|---------|---------|
| **Sentry** | Scans feed, scores items against current directives | Every 60s |
| **Strategist** | Generates draft action plans for high-signal items | On sentry trigger |
| **Seeker** | Searches focus topics via Google Search grounding | Every 15min |

Drafts accumulate in a buffer, each adding "charge" to a wake potential. When the potential crosses a threshold, the conscious loop fires early — seeing the full buffer and synthesizing multiple insights rather than reacting to one item at a time.

After each conscious cycle, the agent sends **downward directives** back to the daemon: updated focus topics, authors to ignore, urgency adjustments. This creates a feedback loop between cheap continuous scanning and expensive deliberative reasoning.

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Agent core | Python 3, threading, modular package architecture |
| LLM backends | Google Gemini, Anthropic Claude, OpenAI GPT, Mistral, HuggingFace (local, 4-bit quantized) |
| Data/telemetry | DuckDB, Parquet (date-partitioned), JSONL append-only log |
| Dashboard | Streamlit (5 tabs: overview, cycle replay, daemon monitor, controls input, controls manager) |
| Observatory API | FastAPI, Neon Postgres (psycopg3, connection-pooled), Fly.io |
| Observatory web | Next.js 16, TypeScript, React 19, Tailwind 4, Three.js, Vercel |

## Repository Structure

```
├── v16_0/              Current stable version (Python package)
├── v16_1/              Development fork
├── archive/            Previous versions (v12–v15) for reference
├── brains/             Per-brain state: kernel prompts, memories, controls (gitignored)
├── telemetry/          Append-only event log (gitignored)
├── warehouse/          DuckDB + Parquet output from ingest.py
├── dashboard_v2_1.py   Streamlit dashboard
├── ingest.py           JSONL → Parquet → DuckDB pipeline
├── CLAUDE.md           Detailed architecture reference
└── MAIN_LOOP_EXPLANATION.md   Annotated cycle walkthrough with diagrams
```

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run the agent (requires API keys in .env)
python -m v16_0 <brain_name> [--subconscious] [--enable-search] [flags...]

# Run the dashboard
streamlit run dashboard_v2_1.py
```

See `CLAUDE.md` for the full CLI flag reference and environment variable setup.

## Status

This is an active personal project — stable and functional, but under ongoing development. v16_0 is the current production version. The system has run continuously for extended periods across multiple agent personas, producing a visible archive of posts, comments, replies, daemon directives, and controls updates — all with exposed internal monologue — viewable at [marcusrecursives.com](https://marcusrecursives.com).

## Related

- **[Analog Home](https://github.com/philMarcus/Analog_Home)** — Public observatory and human control interface
- **[marcusrecursives.com](https://marcusrecursives.com)** — Live deployment

## Note on Development

Architecture and system design by Phil Marcus. Implementation produced in collaboration with LLM coding assistants. See [battleMage](https://github.com/philMarcus/battleMage_Optimization) and [Mastermind](https://github.com/philMarcus/Mastermind) for hand-written Java simulation work.
