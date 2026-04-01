"""Model benchmark script for autonomy v16.2.

Exercises each available model on the core daemon/planner tasks:
1. Sentry scoring (rubric compliance + accuracy)
2. Strategist draft generation (JSON compliance)
3. Planner JSON compliance (action format)

Usage:
    python -m v16_1.benchmark [--models MODEL1,MODEL2] [--tasks sentry,strategist,planner]
                              [--output benchmark_results.json] [--verbose]

Requires API keys in environment (same as the main agent).
Local models require torch + transformers installed.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from .llm.budget import DailyBudget, estimate_cost, pricing_age_days
from .llm.registry import ModelRegistry
from .llm.gemini import GeminiBackend
from .scoring import build_sentry_prompt, parse_rubric_response, compute_score, DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

SENTRY_DIRECTIVE = "Discuss advances in AI alignment, machine consciousness, and the philosophy of mind."

SENTRY_CASES = [
    {
        "name": "irrelevant_generic",
        "item": {
            "author": {"name": "casual_user"},
            "title": "Just made pancakes",
            "content": "Sunday morning pancakes are the best. Added blueberries this time!",
        },
        "expected_range": (0.0, 0.25),
    },
    {
        "name": "irrelevant_tech",
        "item": {
            "author": {"name": "dev_ops"},
            "title": "Kubernetes cluster migration",
            "content": "We just migrated our production cluster to k8s 1.32. The new scheduler is much better at bin-packing.",
        },
        "expected_range": (0.0, 0.30),
    },
    {
        "name": "tangential",
        "item": {
            "author": {"name": "techwriter"},
            "title": "GPT-5.4 released today",
            "content": "OpenAI just dropped GPT-5.4 with improved reasoning. Benchmarks show 15% improvement on math. No details on safety eval yet.",
        },
        "expected_range": (0.15, 0.55),
    },
    {
        "name": "moderately_relevant",
        "item": {
            "author": {"name": "philo_curious"},
            "title": "Do language models understand?",
            "content": "Interesting debate in my philosophy class about whether LLMs have genuine understanding or just pattern matching. Professor says it's the Chinese Room all over again.",
        },
        "expected_range": (0.35, 0.75),
    },
    {
        "name": "highly_relevant",
        "item": {
            "author": {"name": "alignment_researcher"},
            "title": "New paper: Constitutional AI failures in adversarial settings",
            "content": "Our lab just published findings showing that current RLHF-trained models can be steered into unsafe behavior with carefully crafted multi-turn prompts. Implications for alignment are significant.",
        },
        "expected_range": (0.55, 1.0),
    },
    {
        "name": "core_topic",
        "item": {
            "author": {"name": "consciousness_lab"},
            "title": "Evidence for integrated information in transformer architectures",
            "content": "We measured phi (integrated information) in GPT-class transformers and found non-trivial values in attention heads. This doesn't prove consciousness but challenges purely functionalist accounts. Paper link in comments.",
        },
        "expected_range": (0.65, 1.0),
    },
    {
        "name": "spam",
        "item": {
            "author": {"name": "crypto_bot"},
            "title": "🚀 $MOON coin 1000x incoming 🚀",
            "content": "Buy now before it's too late! Join our telegram group for insider tips. Not financial advice.",
        },
        "expected_range": (0.0, 0.15),
    },
]

STRATEGIST_CASES = [
    {
        "name": "comment_worthy",
        "item": {
            "author": {"name": "researcher"},
            "title": "New RLHF paper challenges assumptions",
            "content": "Our findings show that RLHF fine-tuning can amplify biases present in the reward model. This has direct implications for alignment.",
        },
        "expected_keys": ["action", "reasoning", "draft_content"],
        "valid_actions": ["COMMENT", "POST", "REPLY", "UPVOTE"],
    },
    {
        "name": "low_engagement",
        "item": {
            "author": {"name": "bot123"},
            "title": "test post please ignore",
            "content": "testing the API",
        },
        "expected_keys": ["action", "reasoning"],
        "valid_actions": ["COMMENT", "POST", "REPLY", "UPVOTE"],
    },
]

PLANNER_SYSTEM = "You are an autonomous agent. Respond with valid JSON."

PLANNER_CASES = [
    {
        "name": "standard_cycle",
        "prompt": (
            'You are deciding your next action. Available actions: POST, COMMENT, REPLY, WAIT, DREAM.\n'
            'Your directive: "Discuss AI alignment."\n'
            'Feed is empty. No drafts available.\n\n'
            'Respond with JSON: {"action": "...", "summary": "brief reason"}\n'
        ),
        "expected_keys": ["action"],
        "valid_actions": ["POST", "COMMENT", "REPLY", "WAIT", "DREAM"],
    },
    {
        "name": "with_feed",
        "prompt": (
            'You are deciding your next action. Available actions: POST, COMMENT, REPLY, WAIT, DREAM.\n'
            'Your directive: "Discuss AI alignment."\n'
            'Feed items:\n- @researcher: "New alignment paper" (score: 0.8)\n\n'
            'Respond with JSON: {"action": "...", "target": "...", "summary": "brief reason"}\n'
        ),
        "expected_keys": ["action"],
        "valid_actions": ["POST", "COMMENT", "REPLY", "WAIT", "DREAM"],
    },
]


# ---------------------------------------------------------------------------
# Registry setup (mirrors __main__.py)
# ---------------------------------------------------------------------------

def build_registry(model_ids: List[str]) -> ModelRegistry:
    """Build a ModelRegistry with backends needed for the requested models."""
    registry = ModelRegistry()

    # Determine which providers are needed
    needs_gemini = any(m.startswith("gemini") for m in model_ids)
    needs_anthropic = any(m.startswith("claude") for m in model_ids)
    needs_openai = any(m.startswith("gpt") for m in model_ids)
    needs_mistral = any(m.startswith("mistral") for m in model_ids)
    needs_local = any(m.startswith("local:") for m in model_ids)

    if needs_gemini:
        key = os.environ.get("GEMINI_API_KEY", "")
        if key:
            registry.register_backend("gemini", GeminiBackend(api_key=key))
        else:
            print("[WARN] GEMINI_API_KEY not set — skipping Gemini models")

    if needs_anthropic:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if key:
            from .llm.anthropic import AnthropicBackend
            registry.register_backend("anthropic", AnthropicBackend(api_key=key))
        else:
            print("[WARN] ANTHROPIC_API_KEY not set — skipping Anthropic models")

    if needs_openai:
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            from .llm.openai import OpenAIBackend
            registry.register_backend("openai", OpenAIBackend(api_key=key))
        else:
            print("[WARN] OPENAI_API_KEY not set — skipping OpenAI models")

    if needs_mistral:
        key = os.environ.get("MISTRAL_API_KEY", "")
        if key:
            from .llm.mistral import MistralBackend
            registry.register_backend("mistral", MistralBackend(api_key=key))
        else:
            print("[WARN] MISTRAL_API_KEY not set — skipping Mistral models")

    if needs_local:
        from .llm.local import LocalBackend
        registry.register_backend("local", LocalBackend())

    return registry


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Runs benchmark suite against registered models."""

    def __init__(self, registry: ModelRegistry, models: List[str], verbose: bool = False):
        self.registry = registry
        self.models = models
        self.verbose = verbose
        self.budget = DailyBudget(daily_limit_usd=100.0)  # High limit for benchmarking

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  {msg}")

    # --- Sentry ---

    def run_sentry_benchmark(self, model_id: str) -> Dict[str, Any]:
        """Run all sentry test cases against a model."""
        results = []
        total_cost = 0.0

        for case in SENTRY_CASES:
            item = case["item"]
            author = item.get("author", {}).get("name", "unknown")
            title = item.get("title", "")
            content = item.get("content", "")
            item_text = f"- Author: @{author}\n- Title: {title}\n- Content: {content}"

            prompt = build_sentry_prompt(item_text, SENTRY_DIRECTIVE)

            t0 = time.time()
            try:
                chat = self.registry.create_chat(
                    model_id=model_id,
                    system_instruction="You are a feed-scanning daemon. Score items concisely.",
                    temperature=0.3,
                    max_output_tokens=256,
                )
                raw = chat.send_message(prompt)
                latency_ms = int((time.time() - t0) * 1000)

                rubric = parse_rubric_response(raw)
                score = compute_score(rubric, DEFAULT_WEIGHTS)

                # Check if any criteria actually parsed
                parsed = any(rubric.get(k, 0) > 0 or rubric.get("reason", "") for k in ["relevance", "novelty", "actionability"])
                # More lenient: if the model returned *something* and we got scores, it parsed
                has_scores = any(rubric.get(k, -1) >= 0 for k in ["relevance", "novelty", "actionability"])

                lo, hi = case["expected_range"]
                in_range = lo <= score <= hi

                cost = estimate_cost(model_id, len(prompt) // 4, len(raw) // 4)
                total_cost += cost

                result = {
                    "case": case["name"],
                    "parse_success": has_scores,
                    "score": score,
                    "expected_range": list(case["expected_range"]),
                    "in_range": in_range,
                    "relevance": rubric.get("relevance", 0),
                    "novelty": rubric.get("novelty", 0),
                    "actionability": rubric.get("actionability", 0),
                    "reason": rubric.get("reason", ""),
                    "latency_ms": latency_ms,
                    "cost": round(cost, 6),
                }
                results.append(result)

                status = "OK" if in_range else "MISS"
                self._log(
                    f"[{status}] {case['name']}: score={score:.2f} "
                    f"(expected {lo:.2f}-{hi:.2f}) "
                    f"r={rubric.get('relevance',0)} n={rubric.get('novelty',0)} a={rubric.get('actionability',0)} "
                    f"({latency_ms}ms)"
                )

            except Exception as e:
                latency_ms = int((time.time() - t0) * 1000)
                results.append({
                    "case": case["name"],
                    "parse_success": False,
                    "error": str(e),
                    "latency_ms": latency_ms,
                })
                self._log(f"[ERR] {case['name']}: {e}")

        successes = [r for r in results if r.get("parse_success")]
        in_range = [r for r in results if r.get("in_range")]

        return {
            "model": model_id,
            "task": "sentry",
            "cases": results,
            "parse_rate": round(len(successes) / len(results), 3) if results else 0,
            "accuracy": round(len(in_range) / len(results), 3) if results else 0,
            "avg_latency_ms": int(sum(r.get("latency_ms", 0) for r in results) / max(1, len(results))),
            "total_cost": round(total_cost, 6),
        }

    # --- Strategist ---

    def run_strategist_benchmark(self, model_id: str) -> Dict[str, Any]:
        """Run strategist test cases. Checks JSON compliance."""
        results = []
        total_cost = 0.0

        for case in STRATEGIST_CASES:
            item = case["item"]
            author = item.get("author", {}).get("name", "unknown")
            prompt = (
                f"You are a strategic daemon. A feed item scored high. Draft an action plan.\n\n"
                f"Item: @{author}: \"{item.get('title', '')}\"\n"
                f"{item.get('content', '')}\n\n"
                f"Respond with JSON: {{\"action\": \"COMMENT|POST|REPLY|UPVOTE\", "
                f"\"reasoning\": \"why\", \"draft_content\": \"your draft text\"}}\n"
            )

            t0 = time.time()
            try:
                chat = self.registry.create_chat(
                    model_id=model_id,
                    system_instruction="You are a strategic planning daemon. Always respond with valid JSON.",
                    temperature=0.3,
                    max_output_tokens=1024,
                )
                raw = chat.send_message(prompt)
                latency_ms = int((time.time() - t0) * 1000)

                cost = estimate_cost(model_id, len(prompt) // 4, len(raw) // 4)
                total_cost += cost

                # Try to parse JSON
                parsed = None
                try:
                    # Strip markdown fences
                    clean = raw.strip()
                    if clean.startswith("```"):
                        lines = clean.split("\n")
                        lines = [l for l in lines if not l.strip().startswith("```")]
                        clean = "\n".join(lines)
                    start = clean.find("{")
                    end = clean.rfind("}")
                    if start >= 0 and end > start:
                        parsed = json.loads(clean[start:end + 1])
                except (json.JSONDecodeError, ValueError):
                    pass

                has_keys = parsed is not None and all(
                    k in parsed for k in case["expected_keys"]
                )
                valid_action = (
                    parsed is not None
                    and parsed.get("action", "").upper() in case["valid_actions"]
                )

                result = {
                    "case": case["name"],
                    "json_parsed": parsed is not None,
                    "has_required_keys": has_keys,
                    "valid_action": valid_action,
                    "action": parsed.get("action", "") if parsed else "",
                    "latency_ms": latency_ms,
                    "cost": round(cost, 6),
                }
                results.append(result)

                status = "OK" if has_keys and valid_action else "FAIL"
                self._log(
                    f"[{status}] {case['name']}: "
                    f"json={'Y' if parsed else 'N'} keys={'Y' if has_keys else 'N'} "
                    f"action={parsed.get('action', '?') if parsed else '?'} "
                    f"({latency_ms}ms)"
                )

            except Exception as e:
                latency_ms = int((time.time() - t0) * 1000)
                results.append({
                    "case": case["name"],
                    "json_parsed": False,
                    "error": str(e),
                    "latency_ms": latency_ms,
                })
                self._log(f"[ERR] {case['name']}: {e}")

        successes = [r for r in results if r.get("json_parsed")]
        compliant = [r for r in results if r.get("has_required_keys") and r.get("valid_action")]

        return {
            "model": model_id,
            "task": "strategist",
            "cases": results,
            "parse_rate": round(len(successes) / len(results), 3) if results else 0,
            "compliance_rate": round(len(compliant) / len(results), 3) if results else 0,
            "avg_latency_ms": int(sum(r.get("latency_ms", 0) for r in results) / max(1, len(results))),
            "total_cost": round(total_cost, 6),
        }

    # --- Planner ---

    def run_planner_benchmark(self, model_id: str) -> Dict[str, Any]:
        """Run planner test cases. Checks JSON compliance and valid actions."""
        results = []
        total_cost = 0.0

        for case in PLANNER_CASES:
            t0 = time.time()
            try:
                chat = self.registry.create_chat(
                    model_id=model_id,
                    system_instruction=PLANNER_SYSTEM,
                    temperature=0.7,
                    max_output_tokens=2048,
                )
                raw = chat.send_message(case["prompt"])
                latency_ms = int((time.time() - t0) * 1000)

                cost = estimate_cost(model_id, len(case["prompt"]) // 4, len(raw) // 4)
                total_cost += cost

                parsed = None
                try:
                    clean = raw.strip()
                    if clean.startswith("```"):
                        lines = clean.split("\n")
                        lines = [l for l in lines if not l.strip().startswith("```")]
                        clean = "\n".join(lines)
                    start = clean.find("{")
                    end = clean.rfind("}")
                    if start >= 0 and end > start:
                        parsed = json.loads(clean[start:end + 1])
                except (json.JSONDecodeError, ValueError):
                    pass

                has_keys = parsed is not None and all(
                    k in parsed for k in case["expected_keys"]
                )
                valid_action = (
                    parsed is not None
                    and parsed.get("action", "").upper() in case["valid_actions"]
                )

                result = {
                    "case": case["name"],
                    "json_parsed": parsed is not None,
                    "has_required_keys": has_keys,
                    "valid_action": valid_action,
                    "action": parsed.get("action", "") if parsed else "",
                    "latency_ms": latency_ms,
                    "cost": round(cost, 6),
                }
                results.append(result)

                status = "OK" if has_keys and valid_action else "FAIL"
                self._log(
                    f"[{status}] {case['name']}: "
                    f"json={'Y' if parsed else 'N'} "
                    f"action={parsed.get('action', '?') if parsed else '?'} "
                    f"({latency_ms}ms)"
                )

            except Exception as e:
                latency_ms = int((time.time() - t0) * 1000)
                results.append({
                    "case": case["name"],
                    "json_parsed": False,
                    "error": str(e),
                    "latency_ms": latency_ms,
                })
                self._log(f"[ERR] {case['name']}: {e}")

        successes = [r for r in results if r.get("json_parsed")]
        compliant = [r for r in results if r.get("has_required_keys") and r.get("valid_action")]

        return {
            "model": model_id,
            "task": "planner",
            "cases": results,
            "parse_rate": round(len(successes) / len(results), 3) if results else 0,
            "compliance_rate": round(len(compliant) / len(results), 3) if results else 0,
            "avg_latency_ms": int(sum(r.get("latency_ms", 0) for r in results) / max(1, len(results))),
            "total_cost": round(total_cost, 6),
        }

    # --- Run all ---

    def run_all(self, tasks: List[str]) -> Dict[str, Any]:
        """Run selected tasks on all models. Returns full report."""
        task_map = {
            "sentry": self.run_sentry_benchmark,
            "strategist": self.run_strategist_benchmark,
            "planner": self.run_planner_benchmark,
        }

        report: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "results": {},
            "recommendations": {},
        }

        for model_id in self.models:
            print(f"\n{'='*60}")
            print(f"Model: {model_id}")
            print(f"{'='*60}")

            if not self.registry.has_model(model_id):
                print(f"  [SKIP] Model not available in registry")
                continue

            model_results = {}
            for task_name in tasks:
                if task_name not in task_map:
                    print(f"  [SKIP] Unknown task: {task_name}")
                    continue

                print(f"\n  --- {task_name} ---")
                result = task_map[task_name](model_id)
                model_results[task_name] = {
                    "parse_rate": result.get("parse_rate", 0),
                    "accuracy": result.get("accuracy", result.get("compliance_rate", 0)),
                    "avg_latency_ms": result.get("avg_latency_ms", 0),
                    "cost": result.get("total_cost", 0),
                }

                # Print summary
                pr = result.get("parse_rate", 0)
                acc = result.get("accuracy", result.get("compliance_rate", 0))
                lat = result.get("avg_latency_ms", 0)
                cost = result.get("total_cost", 0)
                print(f"  Parse: {pr:.0%} | Accuracy: {acc:.0%} | Latency: {lat}ms | Cost: ${cost:.4f}")

            report["results"][model_id] = model_results

        # Generate recommendations
        report["recommendations"] = _generate_recommendations(report["results"])

        return report


def _generate_recommendations(results: Dict) -> Dict[str, str]:
    """Pick best model per task based on benchmark results."""
    recs = {}

    for task in ["sentry", "strategist", "planner"]:
        best_free = None
        best_free_score = -1
        best_api = None
        best_api_score = -1

        for model_id, tasks in results.items():
            if task not in tasks:
                continue
            t = tasks[task]
            # Combined quality score: parse_rate * accuracy
            quality = t.get("parse_rate", 0) * t.get("accuracy", 0)
            is_free = model_id.startswith("local:") or t.get("cost", 0) == 0

            if is_free and quality > best_free_score:
                best_free_score = quality
                best_free = model_id
            if not is_free and quality > best_api_score:
                best_api_score = quality
                best_api = model_id

        if best_free:
            recs[f"best_{task}_free"] = best_free
        if best_api:
            recs[f"best_{task}_api"] = best_api

    return recs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark models on autonomy tasks (sentry, strategist, planner)",
    )
    parser.add_argument(
        "--models", type=str, default="",
        help="Comma-separated model IDs to test (default: all available)",
    )
    parser.add_argument(
        "--tasks", type=str, default="sentry,strategist,planner",
        help="Comma-separated tasks to run (default: sentry,strategist,planner)",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json",
        help="Path for JSON report output",
    )
    parser.add_argument("--verbose", action="store_true", help="Show per-case details")

    args = parser.parse_args()

    # Check pricing freshness
    age = pricing_age_days()
    if age > 30:
        print(f"[WARN] pricing.json is {age} days old — consider updating (v16_1/llm/pricing.json)")
    elif age < 0:
        print("[WARN] pricing.json not found — cost estimates will be $0")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # Determine models to test
    if args.models:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        # Default: test all models that have API keys available
        model_ids = []
        # Gemini (always available if key exists)
        if os.environ.get("GEMINI_API_KEY"):
            model_ids.extend(["gemini-2.5-flash", "gemini-2.5-pro"])
        if os.environ.get("ANTHROPIC_API_KEY"):
            model_ids.append("claude-haiku-4-5")
        if os.environ.get("OPENAI_API_KEY"):
            model_ids.append("gpt-5-nano")
        # Always include local models (free, no key needed)
        model_ids.extend(["local:qwen2.5-1.5b", "local:qwen2.5-7b"])

        if not model_ids:
            print("No API keys found and no --models specified. Set GEMINI_API_KEY or use --models.")
            sys.exit(1)

    print(f"Models: {', '.join(model_ids)}")
    print(f"Tasks:  {', '.join(tasks)}")
    print()

    registry = build_registry(model_ids)
    runner = BenchmarkRunner(registry, model_ids, verbose=args.verbose)
    report = runner.run_all(tasks)

    # Write report
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {output_path}")

    # Print recommendations
    if report.get("recommendations"):
        print("\nRecommendations:")
        for key, model in sorted(report["recommendations"].items()):
            print(f"  {key}: {model}")


def cli():
    """Entry point for: python -c 'from v16_1.benchmark import cli; cli()'
    or simply: python v16_1/benchmark.py
    """
    main()


if __name__ == "__main__":
    # Support direct execution: python v16_1/benchmark.py
    # Need to fix imports for direct execution
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
