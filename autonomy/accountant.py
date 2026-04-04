"""Budget-aware model/frequency planner for v16.2.

Runs as a conscious-level daily planning pass (not a daemon gear).
Evaluates current spend rate and remaining budget, then recommends
model selections and interval adjustments.

Triggers: first cycle of the day, when budget crosses thresholds,
or after 8+ hours since last plan.
"""

import json
import os
import time
from typing import Any, Dict, Optional

from .llm.budget import DailyBudget, COST_TABLE, estimate_cost


def should_run_budget_plan(
    budget: DailyBudget,
    ctrl,
    last_plan_time: float,
) -> bool:
    """Return True if the accountant should run.

    Triggers:
    - budget_plan_enabled control is False -> never run
    - First cycle of the day (last_plan_time is 0 or from yesterday)
    - Budget crossed the conserve_threshold
    - 8+ hours since last plan
    """
    if not ctrl.get("budget_plan_enabled"):
        return False

    now = time.time()

    # First cycle of day (no plan yet or plan was yesterday)
    if last_plan_time <= 0:
        return True

    hours_since = (now - last_plan_time) / 3600
    if hours_since >= 8:
        return True

    # Budget threshold check
    remaining = budget.remaining_fraction()
    threshold = float(ctrl.get("budget_conserve_threshold"))
    if remaining <= threshold:
        return True

    return False


def estimate_daily_cost(ctrl) -> Dict[str, float]:
    """Estimate daily cost at current settings.

    Returns breakdown by role (sentry, strategist, conscious) and total.
    """
    sentry_interval = max(1, int(ctrl.get("sentry_interval_seconds")))
    cycle_interval = max(1, int(ctrl.get("cycle_interval_minutes")))
    sub_model = ctrl.get("subconscious_model")
    con_model = ctrl.get("conscious_model")

    # Estimated calls per 24 hours
    sentry_calls = (24 * 3600) / sentry_interval
    feed_batch = max(1, int(ctrl.get("feed_batch_size") or 8))
    signal_thresh = float(ctrl.get("signal_threshold") or 0.5)
    wake_thresh = float(ctrl.get("wake_threshold") or 3.0)
    charge_wt = float(ctrl.get("charge_weight_feed") or 0.3)

    # Estimate what fraction of feed items pass signal_threshold
    pass_rate = max(0.05, 1.0 - signal_thresh)  # rough: higher threshold = fewer pass
    strategist_calls = sentry_calls * pass_rate

    # Estimate daemon-triggered wakes: each sentry tick adds charge_wt * pass_rate * feed_batch
    charge_per_tick = charge_wt * pass_rate * feed_batch
    ticks_to_wake = wake_thresh / max(0.01, charge_per_tick)
    seconds_to_wake = ticks_to_wake * sentry_interval
    # Conscious fires at min(cycle_interval, daemon wake interval)
    effective_interval_min = min(cycle_interval, seconds_to_wake / 60)
    conscious_calls = (24 * 60) / max(1, effective_interval_min)

    # Estimated tokens per call
    sentry_cost = sentry_calls * estimate_cost(sub_model, 800, 50)
    strategist_cost = strategist_calls * estimate_cost(sub_model, 1200, 4096)
    conscious_cost = conscious_calls * estimate_cost(con_model, 6000, 1500)

    return {
        "sentry_cost": round(sentry_cost, 4),
        "strategist_cost": round(strategist_cost, 4),
        "conscious_cost": round(conscious_cost, 4),
        "total": round(sentry_cost + strategist_cost + conscious_cost, 4),
        "sentry_calls_per_day": int(sentry_calls),
        "conscious_calls_per_day": int(conscious_calls),
        "effective_interval_min": round(effective_interval_min, 1),
    }


def build_budget_plan_prompt(
    budget: DailyBudget,
    ctrl,
    registry=None,
) -> str:
    """Build a prompt for the conscious model to produce a budget allocation plan."""
    import datetime

    summary = budget.spend_summary_for_planning(registry)
    projection = estimate_daily_cost(ctrl)

    now = datetime.datetime.now(datetime.timezone.utc)
    hours_left = max(0.0, 24.0 - (now.hour + now.minute / 60))

    prompt_parts = [
        "You are reviewing your daily budget allocation. Analyze the current spend rate "
        "and recommend adjustments to stay within budget while maximizing output quality.\n",
        f"--- CURRENT BUDGET STATUS ---\n{summary}\n",
        f"--- COST PROJECTION AT CURRENT SETTINGS ---",
        f"Estimated daily cost: ${projection['total']:.4f}",
        f"  Sentry ({projection['sentry_calls_per_day']} calls/day): ${projection['sentry_cost']:.4f}",
        f"  Strategist: ${projection['strategist_cost']:.4f}",
        f"  Conscious ({projection['conscious_calls_per_day']} calls/day, ~{projection['effective_interval_min']:.0f}min effective interval): ${projection['conscious_cost']:.4f}",
        f"\nHours remaining today: {hours_left:.1f}",
        f"\n--- CURRENT SETTINGS ---",
        f"conscious_model: {ctrl.get('conscious_model')}",
        f"subconscious_model: {ctrl.get('subconscious_model')}",
        f"sentry_interval_seconds: {ctrl.get('sentry_interval_seconds')}",
        f"cycle_interval_minutes: {ctrl.get('cycle_interval_minutes')} (NOTE: this is the MAX sleep — the daemon usually wakes conscious earlier)",
        f"wake_threshold: {ctrl.get('wake_threshold')} (charge needed for daemon to wake conscious)",
        f"signal_threshold: {ctrl.get('signal_threshold')} (sentry score cutoff — higher = more selective = fewer strategist calls = less charge)",
        f"charge_weight_feed: {ctrl.get('charge_weight_feed')} (charge per qualifying feed item)",
        f"daily_budget_usd: {ctrl.get('daily_budget_usd')}",
    ]

    # Available model alternatives
    if COST_TABLE:
        prompt_parts.append("\n--- AVAILABLE MODELS (cost per 1K tokens) ---")
        for model_id, costs in sorted(COST_TABLE.items()):
            if costs["input"] == 0 and costs["output"] == 0:
                prompt_parts.append(f"  {model_id}: FREE")
            else:
                prompt_parts.append(
                    f"  {model_id}: ${costs['input']:.5f} in / ${costs['output']:.5f} out"
                )

    # Check for benchmark results
    bench_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_results.json")
    if os.path.exists(bench_path):
        try:
            with open(bench_path) as f:
                bench = json.load(f)
            recs = bench.get("recommendations", {})
            if recs:
                prompt_parts.append("\n--- BENCHMARK RECOMMENDATIONS ---")
                for key, model in sorted(recs.items()):
                    prompt_parts.append(f"  {key}: {model}")
        except (json.JSONDecodeError, OSError):
            pass

    prompt_parts.extend([
        "\n--- INSTRUCTIONS ---",
        "Based on the above, recommend settings adjustments.",
        "",
        "IMPORTANT — How conscious invocations actually work:",
        "The daemon's sentry scans the feed and scores items. High-scoring items trigger the",
        "strategist, which adds charge to wake_potential. When wake_potential >= wake_threshold,",
        "conscious fires — usually BEFORE cycle_interval_minutes elapses. This means the daemon",
        "wake mechanism is the primary driver of conscious cost, not the cycle interval.",
        "",
        "Budget conservation priority (try in this order):",
        "1. Increase sentry_interval_seconds — fewer scans = fewer charge events",
        "2. Raise wake_threshold — requires more accumulated charge to wake conscious",
        "3. Raise signal_threshold — sentry becomes more selective, fewer items reach strategist",
        "4. Reduce charge_weight_feed — each qualifying item contributes less wake charge",
        "5. Increase cycle_interval_minutes — only affects the guaranteed max sleep between wakes",
        "6. Downgrade conscious_model ONLY as a last resort — quality matters more than frequency",
        "",
        "Also consider:",
        "- Free models (Gemini 2.0 Flash, local models) have no cost but lower quality",
        "- If well under budget, you may decrease intervals or thresholds for more output",
        "",
        "Respond with ONLY valid JSON (include only fields you want to change):",
        '{',
        '  "conscious_model": "model-id",',
        '  "subconscious_model": "model-id",',
        '  "sentry_interval_seconds": <int>,',
        '  "cycle_interval_minutes": <int>,',
        '  "wake_threshold": <float>,',
        '  "signal_threshold": <float>,',
        '  "charge_weight_feed": <float>,',
        '  "reasoning": "brief explanation of your budget strategy"',
        '}',
    ])

    return "\n".join(prompt_parts)


def parse_budget_plan(text: str) -> Optional[Dict[str, Any]]:
    """Parse the budget plan JSON from the conscious model's response."""
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Find JSON object in text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def apply_budget_plan(plan: Dict[str, Any], ctrl) -> Dict[str, str]:
    """Apply the budget plan by updating controls.

    Only updates controls that are present in the plan and not locked.
    Returns dict of {key: "old -> new"} for logging.
    """
    changes = {}
    updatable = [
        "conscious_model", "subconscious_model",
        "sentry_interval_seconds", "cycle_interval_minutes",
        "wake_threshold", "signal_threshold", "charge_weight_feed",
    ]

    for key in updatable:
        if key not in plan:
            continue
        new_val = plan[key]
        old_val = ctrl.get(key)
        if str(new_val) != str(old_val):
            try:
                ctrl.set(key, new_val, source="accountant")
                changes[key] = f"{old_val} -> {new_val}"
            except (ValueError, KeyError):
                pass  # Locked or invalid

    return changes
