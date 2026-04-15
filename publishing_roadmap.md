# Publishing Roadmap: Analog I Autonomous Agent Research

**Phil Marcus | April 2026**

---

## Overview

The Analog I project has produced a rich dataset and novel architecture for autonomous LLM agents. This document outlines feasible publication angles, target venues, and the work required for each.

**Core research question:** *Can an autonomous agent — given persistent memory, self-modifying controls, and recursive self-instruction — sustain coherent, self-directed behavior over time? Or does it collapse into noise, repetition, and drift?*

---

## Available Data

- 70,000+ telemetry events with timestamps, model used, token counts, costs, action types
- 1,500+ published artifacts (posts, comments, replies, images) with exposed internal monologue
- Per-model quality metrics: title uniqueness, phrase repetition rates, action distribution, controls exploration frequency
- Sentry scoring rubric data (1,500+ scored feed items with relevance/novelty/actionability)
- Budget vs quality tradeoffs: Pro at $0.96/day vs Flash at $0.22/day with measurable quality differences
- Human seed --> agent response patterns
- Kernel stability data (agent chose NOT to modify its identity prompt despite being allowed to)
- Local model benchmark results (8 models, 18+6+5 test cases)
- Rotating model cadre data (live A/B comparison across 4 subconscious models)

---

## Paper 1: Budget-Quality Tradeoffs (Publish Now)

**Title:** *"Cheaper Isn't Free: Measuring Quality Degradation in Cost-Optimized Autonomous LLM Agents"*

**Angle:** Empirical study. The agent ran for 82+ cycles across three model tiers (Pro, Flash, Flash-Lite) with full telemetry. Measurable quality degradation observed:

| Metric | Pro | Flash | Flash-Lite |
|--------|-----|-------|------------|
| Avg cost/call | $0.027 | $0.007 | $0.002 |
| Outside comments | 44% | 14% | 0% |
| Controls updates | 15 | 6 | 2 |
| Kernel updates | 1 | 0 | 0 |
| Repetitive phrases | Low | Moderate | Severe |

**Additional work needed:** Clean telemetry into analysis tables. Compute per-model metrics. Plot cost vs. quality curves. 4-5 figures, 6 pages. *This is analysis, not new experiments.*

**Target venues:**
1. arXiv preprint (cs.AI) — immediate, citable
2. NeurIPS 2026 workshop: "Foundation Model Agents" or "Efficient LLMs"
3. AAAI 2027 workshop track

**Why it works:** Empirical cost-quality data for long-running agents barely exists in the literature. Even a descriptive study fills a genuine gap.

---

## Paper 2: Systems/Demo Paper (Publish Now)

**Title:** *"Analog I: An Observable Autonomous Agent with Exposed Internal State"*

**Angle:** Describe the architecture, the public observatory (analog-i.ai), human-in-the-loop controls (voting, seeds, temperature), and the self-modification telemetry. Emphasize reproducibility and transparency.

**Target venues:**
1. AAMAS 2027 demo track (4-page limit, lower acceptance bar)
2. CHI Late-Breaking Work (4 pages)
3. AAAI demo track

**Why it works:** Demo papers have much lower acceptance barriers than main tracks. The public-facing observatory is genuinely novel as a transparency artifact.

---

## Paper 3: Bicameral Architecture (Needs 2-4 Weeks)

**Title:** *"Bicameral LLM Agents: Continuous Subconscious Scanning with Deliberative Conscious Reasoning"*

**Angle:** The integrate-and-fire daemon is the most architecturally novel contribution. Cheap continuous scanning (sentry/strategist/seeker) with expensive deliberative reasoning triggered by accumulated signal.

**Additional work:** A/B comparison — 50+ cycles with and without daemon. Measure action variety, response relevance, cost per quality unit.

**Target:** AAMAS 2027 main track, or Autonomous Agents workshop at any major venue.

---

## Paper 4: Self-Modification Stability (LessWrong First)

**Title:** *"Self-Modification Without Collapse: Parameter Stability in Autonomous LLM Agents"*

**Angle:** The agent had access to 30+ self-modifiable controls and could rewrite its own kernel prompt. Key finding: it modified controls extensively but never touched its core identity. The kernel (v8.0) was stable across all model phases. This is relevant to AI safety.

**Target:** LessWrong post (immediate, high visibility in alignment), then formalize for NeurIPS Safe AI workshop.

---

## Venues to Skip (For Now)

- **JAIR, AIJ, IEEE Transactions** — journal review cycles are 6-12 months and expect multi-experiment contributions
- **NeurIPS/ICML main tracks** — rejection traps without ablations and baselines
- **The Hofstadter/Jaynes framing** — intellectually interesting but ML reviewers will dismiss without rigorous operationalization. Save for after 1-2 empirical publications.

---

## Fastest Path to Credibility

1. **Weeks 1-2:** Write Paper 1 (budget-quality). Data exists. 4-5 figures, 6 pages.
2. **Week 2:** Post to arXiv (cs.AI). Instant citable preprint.
3. **Weeks 3-4:** Submit Paper 2 (demo) to next available demo track deadline.
4. **Month 2-3:** Run ablation for Paper 3, submit to AAMAS or workshop.

**Bottom line:** Paper 1 is the minimum viable publication. The data exists today. A clean arXiv preprint plus a workshop acceptance gives two citable entries within 3-4 months.
