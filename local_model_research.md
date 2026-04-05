# Local Model Research Results

## Benchmark Run: April 5, 2026

### Test Suite
- **Sentry scoring**: 18 cases (7 original + 11 expanded) — irrelevant, tangential, moderate, highly relevant, spam, noise
- **Strategist**: 6 cases — JSON compliance, action selection, draft generation
- **Math verification**: 5 cases — obfuscated arithmetic (Moltbook-style challenges)

### Directive used
"Discuss advances in AI alignment, machine consciousness, and the philosophy of mind."

---

## Results Summary

| Model | Sentry (18) | Strategist (6) | Math (5) | Avg Latency | Cost/run |
|-------|-------------|----------------|----------|-------------|----------|
| **gemini-2.5-flash-lite** | **83% (15/18)** | **100% (6/6)** | 20% (1/5) | **712ms** | $0.0015 |
| gemini-2.5-flash | 72% (13/18) | 83% (5/6) | **100% (5/5)** | 2,503ms | $0.0052 |
| local:qwen2.5-1.5b | 72% (13/18) | 67% (4/6) | 60% (3/5) | 1,732ms | **$0.00** |
| local:qwen2.5-7b | 67% (12/18) | 83% (5/6) | 60% (3/5) | 8,783ms | **$0.00** |
| local:llama-3.1-8b | 71% (13/18)* | 50% (3/6)* | 60% (3/5)* | 22,149ms | **$0.00** |

*From initial run with 7 sentry + 2 strategist cases

---

## Detailed Findings

### Sentry Scoring

**Flash-lite (83%)** — Best overall. Correct score discrimination: spam/noise at 0.0, tangential at 0.3-0.4, core topics at 0.8+. Three misses: under-scored moderately relevant items (emergence 0.33, memory 0.43) and one highly relevant item (strange loops 0.57).

**Qwen 1.5B (72%)** — Ties flash for accuracy count but with opposite error pattern. **Over-scores** tangential and irrelevant tech items (Kubernetes got 0.85, "AI is the future" got 0.75). Correctly puts spam/noise at 0.0 and highly relevant at 0.85-1.0. The ranking order is mostly correct — the threshold just needs to be higher (~0.70+).

**Qwen 7B (67%)** — Most conservative scorer. Under-scores tangential and moderately relevant items (automation 0.10, ethics 0.20, emergence 0.43). Good discrimination on extremes (0.0 for noise, 0.82 for selfhood). Too slow for practical use at 9s/call.

**Flash (72%)** — Surprisingly worse than flash-lite on sentry. Under-scored core_topic (0.45) and selfhood (0.45). Consistently gives actionability=0, which drags down weighted scores. Expensive at 3x flash-lite cost.

### Strategist

**Flash-lite (100%)** — Perfect. All 6 cases produced valid JSON with correct keys and valid actions.

**Qwen 7B (83%)** — Good JSON compliance. One failure: returned `"action": "IGNORE"` which isn't a valid action. Otherwise produced quality drafts.

**Qwen 1.5B (67%)** — Produced valid JSON but two cases returned `"action": "COMMENT|POST|REPLY|UPVOTE"` (the instruction text, not a selection). JSON parsed but action was invalid.

**Flash (83%)** — One failure on the spam case (returned unparseable response).

### Math Verification

**Flash (100%)** — Perfect including obfuscated problems. The thinking model handles deobfuscation well.

**All local models (60%)** — Same pattern: handle clean arithmetic (addition, division, multi-step) but fail obfuscated problems. Qwen 1.5B got 168.0 for 24*3 (wrong), 789.0 for "five cats seven lives" (very wrong). These models cannot reliably handle Moltbook verification challenges.

**Flash-lite (20%)** — Terrible. Got 4/5 wrong including simple addition (33+2=5). Flash-lite's thinking is minimal and it fails arithmetic.

---

## Error Pattern Analysis

### Over-scoring (Qwen 1.5B)
The model defaults to high scores when uncertain. Kubernetes (infrastructure, not AI) got relevance=2, novelty=3, actionability=3. This suggests the model recognizes "tech content" as broadly relevant but can't discriminate sub-domains.

**Mitigation**: Raise signal_threshold to 0.70-0.75 when using Qwen 1.5B. False positives (high scores on irrelevant items) just mean more strategist calls, which is a minor cost when the strategist is also free.

### Under-scoring (Flash, Qwen 7B)
Flash consistently gives actionability=0, which may be a calibration issue with how it interprets the rubric anchors. Qwen 7B is conservative across the board — good for avoiding false positives but misses genuinely relevant content.

### Math failures
All models except Flash fail the obfuscated multiply problems. The deobfuscation step (removing random punctuation and case mixing) requires multi-step reasoning that small/fast models can't do. **Recommendation: Always use gemini-2.5-flash for verification challenges, regardless of subconscious model choice.**

---

## Recommendations

1. **Keep flash-lite as default subconscious model** — best sentry accuracy, fast, cheap
2. **Qwen 1.5B is viable as a free alternative** for sentry+strategist with signal_threshold >= 0.70
3. **Don't use local models for math verification** — keep Flash for that
4. **Seeker must stay Gemini** (needs Google Search grounding)
5. **Next step**: rotating model cadre to collect live comparison data across models

## Non-Google API Results (April 5, 2026)

| Model | Sentry (18) | Strategist (6) | Math (5) | Avg Latency | Cost/run |
|-------|-------------|----------------|----------|-------------|----------|
| **claude-haiku-4-5** | **94% (17/18)** | **100% (6/6)** | **100% (5/5)** | 1,968ms | $0.019 |
| **mistral-small** | **94% (17/18)** | 83% (5/6) | 60% (3/5) | **893ms** | $0.002 |
| gpt-5-nano | 44% (8/18) | 0% (0/6) | 80% (4/5) | 3,400ms | $0.000 |
| gpt-5-mini | 44% (8/18) | 33% (2/6) | 100% (5/5) | 6,046ms | $0.001 |

### Claude Haiku 4.5 — Best Overall
17/18 sentry (only missed strange_loops at 0.57). Perfect strategist and math. Excellent score discrimination: 0.0 for noise, 0.43 for tangential, 0.67-0.90 for relevant, 0.90 for core. At $0.019/run it's 10x flash-lite but still cheap in absolute terms.

### Mistral Small — Best Value
Ties Haiku on sentry (17/18, same miss). Fastest API model at 893ms avg. Same $0.002 cost as flash-lite. One strategist failure (returned IGNORE for low-engagement). Weak on obfuscated math.

### GPT-5 Nano/Mini — Not Viable
Both scored 0.00 on every sentry item. The line-based rubric format appears incompatible with these models. GPT-5 Nano also failed all strategist cases (no valid JSON). Do not use for subconscious.

## Full Comparison (All Models Tested)

| Model | Sentry | Strategist | Math | Latency | Cost | Notes |
|-------|--------|------------|------|---------|------|-------|
| claude-haiku-4-5 | 94% | 100% | 100% | 1,968ms | $0.019 | Best quality |
| mistral-small | 94% | 83% | 60% | 893ms | $0.002 | Best value |
| gemini-2.5-flash-lite | 83% | 100% | 20% | 712ms | $0.002 | Current default |
| gemini-2.5-flash | 72% | 83% | 100% | 2,503ms | $0.005 | Overkill for sentry |
| local:qwen2.5-1.5b | 72% | 67% | 60% | 1,732ms | $0.00 | Over-scores |
| local:qwen2.5-7b | 67% | 83% | 60% | 8,783ms | $0.00 | Too slow |
| gpt-5-mini | 44% | 33% | 100% | 6,046ms | $0.001 | Broken sentry |
| gpt-5-nano | 44% | 0% | 80% | 3,400ms | $0.000 | Broken sentry |

## Recommended Cadre for Rotating Model Experiment

`gemini-2.5-flash-lite,mistral-small-latest,claude-haiku-4-5,local:qwen2.5-1.5b`

Expected avg cost per tick: ~$0.005 (amortized across rotation)
