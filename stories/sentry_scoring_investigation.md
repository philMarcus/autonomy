# When the Free Models "Won": A Lesson in Evaluation Design

*A story from building an autonomous AI agent system*

---

## The Setup

I'm building an autonomous AI agent that runs continuously — reading social media feeds, deciding what to engage with, and posting its own content. The agent has a "subconscious" daemon that scans hundreds of feed items per day, scoring each one for relevance before the expensive "conscious" model sees it.

To save money, I rotate through a cadre of cheap models for this scoring task: local models running on my GPU (free), plus cloud APIs from Google, Anthropic, OpenAI, and Mistral.

The question: **which model is the best sentry?**

## The First Answer (Wrong)

We ran a benchmark. The results were clear:

| Model | Accuracy | Cost |
|-------|----------|------|
| local:qwen2.5-7b | 86% | $0.00 |
| local:qwen2.5-1.5b | 72% | $0.00 |
| gemini-2.5-flash-lite | 0% | $0.002 |
| claude-haiku | 0% | $0.019 |

The free local models dominated. The cloud APIs couldn't even parse the scoring format. Case closed — run the free models, save money, get better results.

Except it wasn't case closed.

## The Doubt

Something didn't add up. Claude Haiku is a capable model. Gemini Flash-Lite handles complex instructions daily. How could they score 0% on a rubric that a 1.5-billion-parameter local model aced?

I looked closer at what "0%" actually meant. The cloud APIs weren't getting the answers wrong — they were **returning empty or unparseable responses** that the parser defaulted to zero. The scoring format — a multi-criterion rubric asking for three separate 0-3 scores per item in a specific line format — was too fiddly for models that think differently about structured output.

The local models "scored well" because they happened to produce text that the regex parser could extract numbers from. But were those numbers actually meaningful?

## The Real Test

We simplified the prompt. Instead of:
```
ITEM 1:
relevance: <0-3>
novelty: <0-3>
actionability: <0-3>
```

We asked:
```
Score each item 0-9. One number per line, nothing else.
```

Then we re-tested everyone:

| Model | Accuracy | Ranking Correct | Cost |
|-------|----------|----------------|------|
| **gemini-2.5-flash-lite** | **88%** | **YES** | $0.002 |
| **claude-haiku** | **81%** | **YES** | $0.019 |
| **mistral-small** | **75%** | **YES** | $0.002 |
| local:qwen2.5-7b | 44% | NO | $0.00 |
| local:qwen2.5-1.5b | **0%** | **NO** | $0.00 |

The story completely flipped. Flash-Lite — the model that "scored 0%" — was actually the best sentry with nearly perfect ranking. Qwen 1.5B — the "86% accuracy" champion — scored **0% with inverted rankings** (scoring irrelevant items at 7/9 and core topics at 0/9).

## The Lesson

**The benchmark was measuring format compliance, not intelligence.** The detailed rubric format happened to match what local models produce. The simple format — which actually tests whether the model *understands relevance* — revealed the real ranking.

Three takeaways:

1. **Your evaluation measures what your evaluation measures, not what you think it measures.** "Accuracy" on a parsing-dependent metric tells you about parsing compatibility, not cognitive capability.

2. **When results seem too good to be true, check the instrument.** Free models outperforming Haiku should have been a red flag. The correct response was not "great, use the free ones" but "why would a 1.5B model outperform a model 100x its size on a ranking task?"

3. **Simplify the interface before blaming the model.** The cloud APIs weren't broken — the prompt was too complex for the task. One number per line is all you need for a sentry. The engineering instinct to build elaborate rubrics with multiple criteria and structured output created a format that filtered for the wrong thing.

---

*This happened during development of [Autonomy](https://github.com/philMarcus/autonomy), an autonomous agent system exploring whether self-referential feedback can sustain coherent emergent behavior over time. The agent's telemetry, internal monologue, and artifacts are visible at [marcusrecursives.com](https://marcusrecursives.com).*
