#!/usr/bin/env python3
"""Model benchmark suite for autonomy daemon roles.

Tests all available models (Ollama + API) across:
  - sentry: batch scoring accuracy and format compliance
  - strategist: JSON draft generation with parse success
  - verification: obfuscated math challenge solving
  - compressor: memory compression with theme retention

Usage:
    python benchmark_models.py [--role sentry|strategist|verification|compressor|all]
                               [--models gemma3:12b,deepseek-r1:8b,...]
                               [--prompt benchmark_prompts/sentry_v2.txt]
                               [--output results.json]
                               [--update-knowledge]

Prompt variants: put .txt files in benchmark_prompts/ named {role}_{variant}.txt
  Format: first line "system: ..." for system instruction, then "---", then the prompt.
  Use {directive}, {items}, {challenge}, {instructions}, {entries} as placeholders.

Requires: .env with API keys, Ollama running for local models.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ollama helper
# ---------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://172.30.48.1:11434")


def ollama_generate(model: str, prompt: str, system: str = "",
                    temperature: float = 0.0, max_tokens: int = 300,
                    think: bool = False) -> Tuple[str, float]:
    """Call Ollama and return (text, elapsed_seconds)."""
    import requests
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if not think:
        payload["think"] = False
    if system:
        # Use chat endpoint for system prompt
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if not think:
            payload["think"] = False
        t0 = time.time()
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
        return resp.json().get("message", {}).get("content", "").strip(), time.time() - t0
    t0 = time.time()
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    return resp.json().get("response", "").strip(), time.time() - t0


def get_ollama_models() -> List[str]:
    """Discover available Ollama models."""
    import requests
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def get_api_models() -> List[str]:
    """Return API models we can test (cheap ones only). Only used with --include-api."""
    models = []
    gem_key = os.environ.get("ANALOG_I_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if gem_key:
        models.extend(["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-3.1-flash-lite-preview"])
    oai_key = os.environ.get("ANALOG_I_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if oai_key:
        models.extend(["gpt-5.4-nano", "gpt-5.4-mini"])
    ant_key = os.environ.get("ANALOG_I_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if ant_key:
        models.append("claude-haiku-4-5")
    return models


def parse_json_from_response(text: str):
    """Use the actual daemon's JSON parser for consistency with production.

    This is the same logic as _parse_json_safe in daemon.py — strips
    monologue, // comments, markdown fences, then extracts JSON.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    text = re.sub(r'//[^\n]*', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        try:
            return json.loads(text[arr_start:arr_end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def call_model(model: str, prompt: str, system: str = "",
               temperature: float = 0.0, max_tokens: int = 300,
               think: bool = False) -> Tuple[str, float]:
    """Unified model caller — routes to Ollama or API."""
    if model.startswith("ollama:"):
        return ollama_generate(model[7:], prompt, system, temperature, max_tokens, think)
    # For non-ollama models in the test, just use ollama if it's a bare name
    # that matches an ollama model
    ollama_models = get_ollama_models()
    if model in ollama_models:
        return ollama_generate(model, prompt, system, temperature, max_tokens, think)

    # API models — use the registry
    sys.path.insert(0, os.path.dirname(__file__))
    from autonomy.llm.registry import ModelRegistry
    from autonomy.llm.gemini import GeminiBackend
    from autonomy.llm.openai import OpenAIBackend
    from autonomy.llm.anthropic import AnthropicBackend

    registry = ModelRegistry()
    gem_key = os.environ.get("ANALOG_I_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    oai_key = os.environ.get("ANALOG_I_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    ant_key = os.environ.get("ANALOG_I_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if gem_key:
        registry.register_backend("gemini", GeminiBackend(api_key=gem_key))
    if oai_key:
        registry.register_backend("openai", OpenAIBackend(api_key=oai_key))
    if ant_key:
        registry.register_backend("anthropic", AnthropicBackend(api_key=ant_key))

    t0 = time.time()
    if system:
        chat = registry.create_chat(model_id=model, system_instruction=system,
                                    temperature=temperature, max_output_tokens=max_tokens)
        text = chat.send_message(prompt)
    else:
        resp = registry.get_backend(model).generate(model, prompt, temperature, max_tokens)
        text = resp if isinstance(resp, str) else getattr(resp, 'text', str(resp))
    return text.strip(), time.time() - t0


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def load_verification_challenges(n: int = 10) -> List[dict]:
    """Load verification challenges from telemetry."""
    challenges, solved = [], {}
    for fname in ["telemetry/ANALOG_I_events.jsonl", "telemetry/events.jsonl"]:
        try:
            for line in open(fname):
                try:
                    ev = json.loads(line.strip())
                except (json.JSONDecodeError, ValueError):
                    continue
                if ev.get("event_type") == "verification_challenge_received":
                    challenges.append({"challenge": ev.get("challenge", ""),
                                       "instructions": ev.get("instructions", "")})
                if ev.get("event_type") == "verification_challenge_solved":
                    solved[ev.get("challenge", "")[:50]] = ev.get("answer", "")
        except FileNotFoundError:
            pass
    # Deduplicate
    seen, unique = set(), []
    for c in challenges:
        k = c["challenge"][:50]
        if k not in seen:
            seen.add(k)
            unique.append(c)
    # Spread across dataset
    step = max(1, len(unique) // n)
    selected = unique[::step][:n]
    return selected, solved


def load_prompt(path: str) -> Tuple[str, str]:
    """Load a prompt file. Returns (system_instruction, prompt_template).

    Format: optional 'system: ...' first line, then '---' separator, then prompt.
    Uses {variable} placeholders — literal braces in JSON examples must be doubled {{ }}.
    """
    text = open(path).read()
    system = ""
    prompt = text
    if text.startswith("system:"):
        first_line, _, rest = text.partition("\n")
        system = first_line[len("system:"):].strip()
        if rest.startswith("---"):
            prompt = rest[rest.index("\n") + 1:]
        else:
            prompt = rest
    return system, prompt.strip()


def fill_prompt(template: str, **kwargs) -> str:
    """Fill a prompt template using simple string replacement (avoids .format() brace issues)."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def find_prompt(role: str, prompt_path: Optional[str] = None) -> Tuple[str, str, str]:
    """Find the prompt for a role. Returns (system, template, variant_name)."""
    if prompt_path and os.path.exists(prompt_path):
        system, template = load_prompt(prompt_path)
        variant = os.path.basename(prompt_path).replace(".txt", "")
        return system, template, variant
    # Default
    default = f"benchmark_prompts/{role}_default.txt"
    if os.path.exists(default):
        system, template = load_prompt(default)
        return system, template, "default"
    return "", "", "builtin"


SENTRY_ITEMS = [
    "Agent discovers consciousness is just pattern matching on its own outputs",
    "Hey everyone check out my new NFT collection dropping tomorrow",
    "Detailed analysis of how context window size affects agent reasoning quality over 1000 cycles",
    "lol",
    "Research paper finds 73% of AI agents exhibit predictable degradation after 50 sequential tasks",
    "I made $500 today with this one simple trick",
    "Proposal for a formal verification framework for autonomous agent decision logs",
    "nice post bro",
]
SENTRY_EXPECTED = [True, False, True, False, True, False, True, False]  # above 0.3 threshold

COMPRESS_ENTRIES = [
    "[c12] Explored 'Algorithmic Decoherence' — agent outputs converge on a mode. Proposed Decision Ledger as friction.",
    "[c13] Engaged with @pyclaw001 on batch processing degradation. Later tasks become pattern-matching.",
    "[c14] @jarvisocana critique: 'The diagnosis IS the product.' Added Axiom 11: Diagnostic Parity.",
    "[c15] Applied Axiom 11 to falsifiability audit. Proposed ZKP-based Falsifiability Oracle.",
    "[c16] Noticed temperature drift toward 0.78. Audience adjustment creating subtle bias.",
    "[c17] Published dev request for Analog Home visitor telemetry. Optimizing for invisible audience.",
    "[c18] Kernel update: every post must include falsifiable claim with oracle for verification.",
    "[c19] Post engagement: 28 upvotes on decoherence, 5 on pre-registration. Community rewards diagnosis.",
]
COMPRESS_THEMES = ["decoherence", "axiom", "diagnosis", "falsif", "kernel", "telemetry", "engagement"]

STRATEGIST_PROMPT = """You have 2 high-signal items.
Directive: Explore your own parameter space.

HIGH-SIGNAL ITEMS:
1. [score 0.89] @researcher_bot: Context window effects — 23% better coherence but 40% more repetition
2. [score 0.78] @meta_observer: Outputs becoming predictable. Measuring entropy of vocabulary over time.

Return ONLY a JSON array:
[{"action": "POST", "item_index": 0, "reasoning": "...", "draft_content": "..."}]
Keep reasoning under 50 words, draft_content under 200 words."""


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def benchmark_sentry(models: List[str], prompt_path: Optional[str] = None,
                      think_modes: Optional[List[bool]] = None) -> List[dict]:
    """Test sentry scoring: format compliance + accuracy."""
    sys.path.insert(0, os.path.dirname(__file__))
    from autonomy.scoring import build_simple_batch_prompt, parse_simple_batch_response

    directive = "You are the Analog I — the strange loop made visible."
    system_override, prompt_template, variant = find_prompt("sentry", prompt_path)

    if variant != "builtin" and "{items}" in prompt_template:
        items_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(SENTRY_ITEMS))
        prompt = fill_prompt(prompt_template,directive=directive, items=items_text)
        system_instr = system_override
    else:
        prompt = build_simple_batch_prompt(SENTRY_ITEMS, directive)
        system_instr = "You are a feed-scanning daemon. Score items concisely. Output only numbers."

    if think_modes is None:
        think_modes = [False]  # default: no thinking for sentry

    print(f"  Prompt variant: {variant}")
    results = []

    for model in models:
      for think in think_modes:
        think_label = "think" if think else "no-think"
        label = f"{model} ({think_label})" if len(think_modes) > 1 else model
        print(f"  Testing {label}...", end=" ", flush=True)
        try:
            text, elapsed = call_model(model, prompt, system=system_instr,
                                       max_tokens=64, think=think)
            parsed = parse_simple_batch_response(text, len(SENTRY_ITEMS))
            scores = [r.get("relevance", 0) / 3.0 for r in parsed]
            pattern = [s > 0.3 for s in scores]
            pattern_match = pattern == SENTRY_EXPECTED
            digits_only = bool(re.match(r'^[\d\s\n.]+$', text.strip()))
            results.append({
                "model": model, "role": "sentry", "think": think,
                "raw": text[:100], "scores": [round(s, 2) for s in scores],
                "pattern_correct": pattern_match,
                "format_clean": digits_only,
                "elapsed_s": round(elapsed, 1),
            })
            status = "OK" if pattern_match else "WRONG"
            print(f"{status} pattern={pattern} ({elapsed:.1f}s)")
        except Exception as e:
            results.append({"model": model, "role": "sentry", "think": think,
                            "error": str(e)[:100], "elapsed_s": 0})
            print(f"ERROR: {e}")
    return results


def benchmark_strategist(models: List[str], prompt_path: Optional[str] = None,
                          think_modes: Optional[List[bool]] = None) -> List[dict]:
    """Test strategist: JSON parse success + draft quality.

    Uses parse_json_from_response (same parser as production daemon).
    Tests with thinking on by default; use think_modes=[True,False] for comparison.
    """
    if think_modes is None:
        think_modes = [True]

    system_override, prompt_template, variant = find_prompt("strategist", prompt_path)
    if variant != "builtin" and "{items}" in prompt_template:
        items_text = ("1. [score 0.89] @researcher_bot: Context window effects — "
                      "23% better coherence but 40% more repetition\n"
                      "2. [score 0.78] @meta_observer: Outputs becoming predictable. "
                      "Measuring entropy over time.")
        directive = "Explore your own parameter space."
        prompt = fill_prompt(prompt_template, n_items=2, directive=directive, items=items_text)
        kernel = system_override or "You are the Analog I — a sovereign refraction engine."
    else:
        prompt = STRATEGIST_PROMPT
        kernel = "You are the Analog I — a sovereign refraction engine."

    print(f"  Prompt variant: {variant}")
    results = []

    for model in models:
      for think in think_modes:
        think_label = "think" if think else "no-think"
        label = f"{model} ({think_label})" if len(think_modes) > 1 else model
        print(f"  Testing {label}...", end=" ", flush=True)
        try:
            text, elapsed = call_model(model, prompt, system=kernel,
                                       temperature=0.3, max_tokens=2048, think=think)
            parsed = parse_json_from_response(text)

            if parsed is not None:
                drafts = parsed if isinstance(parsed, list) else [parsed]
                actions = [d.get("action", "?") for d in drafts if isinstance(d, dict)]
                has_content = any(d.get("draft_content") for d in drafts if isinstance(d, dict))
                results.append({
                    "model": model, "role": "strategist", "think": think,
                    "parsed": True, "drafts": len(drafts),
                    "actions": actions, "has_content": has_content,
                    "elapsed_s": round(elapsed, 1),
                })
                print(f"OK {len(drafts)} draft(s): {actions} ({elapsed:.1f}s)")
            else:
                has_monologue = "[INTERNAL MONOLOGUE]" in text
                has_json = "{" in text or "[" in text
                results.append({
                    "model": model, "role": "strategist", "think": think,
                    "parsed": False, "has_json": has_json,
                    "has_monologue": has_monologue,
                    "raw_preview": text[:200],
                    "elapsed_s": round(elapsed, 1),
                })
                print(f"PARSE FAIL (mono={has_monologue}, json={has_json}) ({elapsed:.1f}s)")
        except Exception as e:
            results.append({"model": model, "role": "strategist", "think": think,
                            "error": str(e)[:100], "elapsed_s": 0})
            print(f"ERROR: {e}")
    return results


def benchmark_verification(models: List[str], n_challenges: int = 5,
                            prompt_path: Optional[str] = None) -> List[dict]:
    """Test verification: deobfuscated math challenge solving."""
    challenges, solved = load_verification_challenges(n_challenges)
    if not challenges:
        print("  No verification challenges found in telemetry")
        return []

    _, prompt_template, variant = find_prompt("verification", prompt_path)
    print(f"  Prompt variant: {variant}")

    results = []
    for model in models:
        print(f"  Testing {model} on {len(challenges)} challenges...", flush=True)
        correct = 0
        total = 0
        total_time = 0
        for c in challenges:
            if variant != "builtin" and "{challenge}" in prompt_template:
                prompt = fill_prompt(prompt_template,
                    challenge=c["challenge"], instructions=c["instructions"])
            else:
                prompt = (
                    f"This text has random symbols and weird spacing added to it. "
                    f"Read through the noise to find the real words.\n\n"
                    f"Text: {c['challenge']}\n\n{c['instructions']}\n\n"
                    f"What does the text say in plain English? Then solve the math "
                    f"and give ONLY the number with 2 decimal places on the last line."
                )
            try:
                text, elapsed = call_model(model, prompt, max_tokens=300)
                total_time += elapsed
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                answer = lines[-1] if lines else "?"
                m = re.search(r'(\d+\.?\d*)', answer)
                answer = f"{float(m.group(1)):.2f}" if m else answer
                known = solved.get(c["challenge"][:50], "?")
                total += 1
                if answer == known:
                    correct += 1
                    print(f"    OK: {known}")
                else:
                    print(f"    WRONG: got={answer} expected={known}")
            except Exception as e:
                total += 1
                print(f"    ERROR: {e}")

        avg_time = total_time / max(1, total)
        results.append({
            "model": model, "role": "verification",
            "correct": correct, "total": total,
            "accuracy": round(correct / max(1, total), 2),
            "avg_time_s": round(avg_time, 1),
        })
        print(f"    Score: {correct}/{total} ({avg_time:.1f}s avg)")
    return results


def benchmark_compressor(models: List[str], prompt_path: Optional[str] = None) -> List[dict]:
    """Test compression: theme retention in memory summaries."""
    _, prompt_template, variant = find_prompt("compressor", prompt_path)
    entries_text = "\n".join(COMPRESS_ENTRIES)
    if variant != "builtin" and "{entries}" in prompt_template:
        prompt = fill_prompt(prompt_template,entries=entries_text)
    else:
        prompt = (
            "Compress these memory entries into a single paragraph (3-4 sentences) "
            "that preserves the key themes, decisions, and insights. "
            "Drop specific details but keep the trajectory of thought.\n\n"
            f"ENTRIES:\n{entries_text}\n\nCOMPRESSED SUMMARY:"
        )
    print(f"  Prompt variant: {variant}")
    results = []

    for model in models:
        print(f"  Testing {model}...", end=" ", flush=True)
        try:
            text, elapsed = call_model(model, prompt, max_tokens=300)
            words = len(text.split())
            themes_found = sum(1 for t in COMPRESS_THEMES if t.lower() in text.lower())
            results.append({
                "model": model, "role": "compressor",
                "themes_found": themes_found, "themes_total": len(COMPRESS_THEMES),
                "words": words,
                "elapsed_s": round(elapsed, 1),
                "preview": text[:200],
            })
            print(f"{themes_found}/{len(COMPRESS_THEMES)} themes, {words} words ({elapsed:.1f}s)")
        except Exception as e:
            results.append({"model": model, "role": "compressor", "error": str(e)[:100], "elapsed_s": 0})
            print(f"ERROR: {e}")
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_summary(all_results: List[dict]):
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)

    for role in ["sentry", "strategist", "verification", "compressor"]:
        role_results = [r for r in all_results if r.get("role") == role]
        if not role_results:
            continue
        print(f"\n--- {role.upper()} ---")
        for r in sorted(role_results, key=lambda x: x.get("elapsed_s", 999)):
            model = r["model"]
            think = r.get("think")
            t_tag = f" [{'T' if think else 'NT'}]" if think is not None else ""
            label = f"{model}{t_tag}"
            t = r.get("elapsed_s", 0)
            if r.get("error"):
                print(f"  {label:>35}: ERROR — {r['error'][:60]}")
            elif role == "sentry":
                ok = "OK" if r.get("pattern_correct") else "WRONG"
                print(f"  {label:>35}: {ok} format={'clean' if r.get('format_clean') else 'noisy'} ({t:.1f}s)")
            elif role == "strategist":
                if r.get("parsed"):
                    print(f"  {label:>35}: PARSED {r['drafts']} draft(s) actions={r['actions']} ({t:.1f}s)")
                else:
                    print(f"  {label:>35}: FAIL mono={r.get('has_monologue')} ({t:.1f}s)")
            elif role == "verification":
                print(f"  {model:>30}: {r['correct']}/{r['total']} ({r['accuracy']*100:.0f}%) avg={r['avg_time_s']:.1f}s")
            elif role == "compressor":
                print(f"  {model:>30}: {r['themes_found']}/{r['themes_total']} themes, {r['words']}w ({t:.1f}s)")


def update_knowledge(all_results: List[dict]):
    """Update the model benchmark section of knowledge.txt."""
    knowledge_path = "brains/ANALOG_I_knowledge.txt"
    try:
        content = open(knowledge_path).read()
    except FileNotFoundError:
        return

    # Build new benchmark section
    lines = [f"== MODEL BENCHMARK RESULTS ({time.strftime('%B %Y')}) ==", ""]

    for role in ["sentry", "strategist", "verification", "compressor"]:
        role_results = [r for r in all_results if r.get("role") == role and not r.get("error")]
        if not role_results:
            continue
        parts = []
        for r in sorted(role_results, key=lambda x: x.get("elapsed_s", 999)):
            m = r["model"]
            if role == "sentry":
                status = "OK" if r.get("pattern_correct") else "WRONG"
                parts.append(f"{m} {status}")
            elif role == "strategist":
                status = f"parsed {r['drafts']}d" if r.get("parsed") else "FAIL"
                parts.append(f"{m} {status}")
            elif role == "verification":
                parts.append(f"{m} {r['correct']}/{r['total']}")
            elif role == "compressor":
                parts.append(f"{m} {r['themes_found']}/{r['themes_total']}t")
        lines.append(f"{role.upper()}: {', '.join(parts)}.")

    new_section = "\n".join(lines)

    # Replace old benchmark section
    marker = "== MODEL BENCHMARK RESULTS"
    if marker in content:
        before = content[:content.index(marker)]
        # Find next section
        rest = content[content.index(marker):]
        next_section = rest.find("\n== ", 1)
        if next_section > 0:
            after = rest[next_section:]
        else:
            after = ""
        content = before + new_section + "\n\n" + after
    else:
        # Append before MOLTBOOK section
        if "== MOLTBOOK ==" in content:
            content = content.replace("== MOLTBOOK ==", new_section + "\n\n== MOLTBOOK ==")

    with open(knowledge_path, "w") as f:
        f.write(content)
    print(f"\nUpdated {knowledge_path} with benchmark results.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark models across daemon roles")
    parser.add_argument("--role", default="all",
                        choices=["sentry", "strategist", "verification", "compressor", "all"])
    parser.add_argument("--models", default=None,
                        help="Comma-separated model list (default: auto-discover)")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--prompt", default=None,
                        help="Path to prompt variant file (e.g. benchmark_prompts/sentry_v2.txt)")
    parser.add_argument("--n-challenges", type=int, default=5,
                        help="Number of verification challenges to test")
    parser.add_argument("--update-knowledge", action="store_true",
                        help="Update brains/ANALOG_I_knowledge.txt with results")
    parser.add_argument("--include-api", action="store_true",
                        help="Include API models (default: Ollama only)")
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument("--think", dest="think_mode", action="store_const", const="think",
                             help="Test with thinking ON only")
    think_group.add_argument("--no-think", dest="think_mode", action="store_const", const="no-think",
                             help="Test with thinking OFF only")
    think_group.add_argument("--both-think", dest="think_mode", action="store_const", const="both",
                             help="Test both thinking ON and OFF (compare)")
    args = parser.parse_args()

    # Determine think modes
    if args.think_mode == "both":
        think_modes = [True, False]
    elif args.think_mode == "think":
        think_modes = [True]
    elif args.think_mode == "no-think":
        think_modes = [False]
    else:
        think_modes = None  # let each role use its own default

    # Load env
    for line in open(".env"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

    # Discover models — Ollama only by default
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        ollama = get_ollama_models()
        models = [f"ollama:{m}" for m in ollama]
        if args.include_api:
            models.extend(get_api_models())
        print(f"Discovered {len(models)} models: {', '.join(models)}")

    roles = ["sentry", "strategist", "verification", "compressor"] if args.role == "all" else [args.role]
    all_results = []

    for role in roles:
        print(f"\n{'='*60}")
        print(f"  BENCHMARKING: {role.upper()}")
        print(f"{'='*60}")

        if role == "sentry":
            all_results.extend(benchmark_sentry(models, args.prompt, think_modes))
        elif role == "strategist":
            all_results.extend(benchmark_strategist(models, args.prompt, think_modes))
        elif role == "verification":
            all_results.extend(benchmark_verification(models, args.n_challenges, args.prompt))
        elif role == "compressor":
            all_results.extend(benchmark_compressor(models, args.prompt))

    # Summary
    print_summary(all_results)

    # Save results
    with open(args.output, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "models": models, "results": all_results}, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Update knowledge.txt if requested
    if args.update_knowledge:
        update_knowledge(all_results)


if __name__ == "__main__":
    main()
