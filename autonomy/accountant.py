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

    The accountant owns wake/budget mechanics and runs every cycle when enabled.
    Running is cheap (local model, a few seconds) and keeps the controls coherent
    with changing spend rate / headroom. Gate only on the enabled control.
    """
    return bool(ctrl.get("budget_plan_enabled"))


def estimate_daily_cost(ctrl) -> Dict[str, float]:
    """Estimate daily cost at current settings.

    Returns breakdown by role (sentry, strategist, conscious) and total.
    """
    sentry_interval = max(1, int(ctrl.get("sentry_interval_seconds")))
    cycle_interval = max(1, int(ctrl.get("cycle_interval_minutes")))
    # Use first model from sentry weights as cost estimate proxy
    _sw = ctrl.get("subconscious_model_weights") or "gemini-2.5-flash-lite=1"
    sub_model = _sw.split("=")[0].strip() if "=" in _sw else "gemini-2.5-flash-lite"
    _cw = ctrl.get("conscious_model_weights") or "gemini-2.5-pro=1"
    con_model = _cw.split("=")[0].strip() if "=" in _cw else "gemini-2.5-pro"

    # Estimated calls per 24 hours
    sentry_calls = (24 * 3600) / sentry_interval
    feed_batch = max(1, int(ctrl.get("feed_batch_size") or 8))
    signal_thresh = float(ctrl.get("signal_threshold") or 0.5)
    target_wake = float(ctrl.get("target_wake_minutes") or 60)
    charge_wt = float(ctrl.get("charge_weight_feed") or 0.05)

    # Estimate what fraction of feed items pass signal_threshold
    pass_rate = max(0.05, 1.0 - signal_thresh)  # rough: higher threshold = fewer pass
    strategist_calls = sentry_calls * pass_rate

    # With auto-calibrated threshold, effective interval ≈ target_wake_minutes
    effective_interval_min = min(cycle_interval, target_wake)
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
        f"conscious_model_weights: {ctrl.get('conscious_model_weights')}",
        f"subconscious_model_weights: {ctrl.get('subconscious_model_weights')}",
        f"sentry_interval_seconds: {ctrl.get('sentry_interval_seconds')}",
        f"cycle_interval_minutes: {ctrl.get('cycle_interval_minutes')} (NOTE: this is the MAX sleep — the daemon usually wakes conscious earlier)",
        f"target_wake_minutes: {ctrl.get('target_wake_minutes')} (auto-calibrated wake interval)",
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
        "\n--- YOUR ROLE ---",
        "You are the accountant. You run every cycle and own the wake/budget mechanics.",
        "You set them coherently based on current state. The conscious planner is busy with",
        "content and relationships; it does NOT tune these knobs. If you leave a value alone,",
        "it stays. Only include fields in your JSON if you want to change them.",
        "",
        "How wake works:",
        "The daemon's sentry ticks every sentry_interval_seconds and scores feed items.",
        "Above-threshold items add charge_weight_feed to wake_potential. When wake_potential",
        "crosses an auto-calibrated threshold (tuned to target_wake_minutes on average),",
        "conscious fires. If no wake by cycle_interval_minutes, conscious fires anyway.",
        "",
        "COHERENCE RULES (check before outputting):",
        "- target_wake_minutes * 60 must be >= sentry_interval_seconds (can't wake faster than sentry ticks)",
        "- cycle_interval_minutes * 60 must be >= sentry_interval_seconds (sentry must tick at least once per cycle)",
        "- cycle_interval_minutes should be >= target_wake_minutes (otherwise target has no effect)",
        "",
        "Budget conservation priority (try in this order when spend is high):",
        "1. Raise sentry_interval_seconds — fewer scans = fewer charge events",
        "2. Raise target_wake_minutes — longer average intervals between conscious cycles",
        "3. Raise signal_threshold — sentry becomes more selective",
        "4. Reduce charge_weight_feed — each qualifying item contributes less charge",
        "5. Raise cycle_interval_minutes — affects the max-sleep floor",
        "6. Downgrade conscious_model_weights ONLY as a last resort",
        "",
        "Budget relaxation (when well under budget with headroom):",
        "- Lower sentry_interval_seconds or target_wake_minutes to get more engagement",
        "- Don't make changes <10% of current — skip trivial tweaks",
        "- If budget is 0 or negative remaining, the conscious already auto-swaps to the",
        "  budget_exhausted_model_weights pool. You don't need to force it.",
        "",
        "Respond with ONLY valid JSON (include only fields you want to change; empty JSON is fine):",
        '{',
        '  "conscious_model_weights": "model=weight,...",',
        '  "subconscious_model_weights": "model=weight,...",',
        '  "budget_exhausted_model_weights": "model=weight,...",',
        '  "sentry_interval_seconds": <int>,',
        '  "cycle_interval_minutes": <int>,',
        '  "target_wake_minutes": <int>,',
        '  "signal_threshold": <float>,',
        '  "charge_weight_feed": <float>,',
        '  "charge_weight_reply": <float>,',
        '  "wake_refractory": <float>,',
        '  "reasoning": "brief explanation"',
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


_ACCOUNTANT_UPDATABLE = [
    "conscious_model_weights", "subconscious_model_weights",
    "budget_exhausted_model_weights",
    "sentry_interval_seconds", "cycle_interval_minutes",
    "target_wake_minutes", "signal_threshold", "charge_weight_feed",
    "charge_weight_reply", "wake_refractory",
]


def apply_budget_plan(plan: Dict[str, Any], ctrl) -> Dict[str, str]:
    """Apply the budget plan by updating controls.

    - Only touches the accountant-owned control list (excludes sentry_strictness etc.)
    - Oscillation guard: skips numeric changes within 10% of the current value so the
      accountant can't jitter knobs each cycle on noise.
    Returns dict of {key: "old -> new"} for logging.
    """
    changes = {}

    for key in _ACCOUNTANT_UPDATABLE:
        if key not in plan:
            continue
        new_val = plan[key]
        old_val = ctrl.get(key)
        if str(new_val) == str(old_val):
            continue
        # Oscillation guard — skip tiny numeric nudges.
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            if old_val != 0 and abs(new_val - old_val) / abs(old_val) < 0.10:
                continue
        try:
            ctrl.set(key, new_val, source="accountant")
            changes[key] = f"{old_val} -> {new_val}"
        except (ValueError, KeyError):
            pass  # Locked or invalid

    return changes
