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
from .prompt_templates import load_template


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
    max_cycle_interval = max(1, int(ctrl.get("max_cycle_interval_minutes")))
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
    effective_interval_min = min(max_cycle_interval, target_wake)
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
        load_template("accountant/intro.txt") + "\n",
        f"--- CURRENT BUDGET STATUS ---\n{summary}\n",
        f"--- COST PROJECTION AT CURRENT SETTINGS ---",
        f"Estimated daily cost: ${projection['total']:.4f}",
        f"  Sentry ({projection['sentry_calls_per_day']} calls/day): ${projection['sentry_cost']:.4f}",
        f"  Strategist: ${projection['strategist_cost']:.4f}",
        f"  Conscious ({projection['conscious_calls_per_day']} calls/day, ~{projection['effective_interval_min']:.0f}min effective interval): ${projection['conscious_cost']:.4f}",
        f"\nHours remaining today: {hours_left:.1f}",
        f"\n--- CURRENT SETTINGS (time values shown in BOTH units) ---",
        f"conscious_model_weights: {ctrl.get('conscious_model_weights')}",
        f"subconscious_model_weights: {ctrl.get('subconscious_model_weights')}",
        f"budget_exhausted_model_weights: {ctrl.get('budget_exhausted_model_weights')}",
        f"sentry_interval_seconds: {int(ctrl.get('sentry_interval_seconds') or 300)} seconds  (= {int(ctrl.get('sentry_interval_seconds') or 300) / 60:.1f} minutes)",
        f"target_wake_minutes: {int(ctrl.get('target_wake_minutes') or 60)} minutes  (= {int(ctrl.get('target_wake_minutes') or 60) * 60} seconds)  (SHARED with conscious)",
        f"signal_threshold: {ctrl.get('signal_threshold')} (SHARED — signal/noise cutoff)",
        f"charge_weight_feed: {ctrl.get('charge_weight_feed')} (SHARED)",
        f"charge_weight_reply: {ctrl.get('charge_weight_reply')} (SHARED)",
        f"wake_refractory: {ctrl.get('wake_refractory')} (yours — wake potential reset after firing)",
        f"max_cycle_interval_minutes: {int(ctrl.get('max_cycle_interval_minutes') or 360)} minutes  (= {int(ctrl.get('max_cycle_interval_minutes') or 360) * 60} seconds)  (operator-managed, do not change)",
        f"daily_budget_usd: ${ctrl.get('daily_budget_usd')}",
    ]

    # Precomputed coherence status — local models are unreliable at the arithmetic,
    # so we hand them the answer. They just have to apply the fix (or not touch it).
    _tw = int(ctrl.get("target_wake_minutes") or 60)
    _si = int(ctrl.get("sentry_interval_seconds") or 300)
    _tw_sec = _tw * 60
    _rule1_ok = _tw_sec >= _si
    prompt_parts.append("")
    prompt_parts.append("--- COHERENCE STATUS (precomputed in SECONDS — use these numbers directly) ---")
    prompt_parts.append(f"  target_wake (in seconds):    {_tw_sec:>5}   [= {_tw} minutes x 60]")
    prompt_parts.append(f"  sentry_interval (in seconds):{_si:>5}   [already in seconds — no conversion needed]")
    if _rule1_ok:
        prompt_parts.append(load_template("accountant/coherence_satisfied.txt").format(
            tw_sec=_tw_sec, si=_si))
    else:
        _min_tw = (_si + 59) // 60  # ceiling division
        _max_si = _tw_sec
        prompt_parts.append(load_template("accountant/coherence_violated.txt").format(
            tw_sec=_tw_sec, si=_si, min_tw=_min_tw, max_si=_max_si))

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

    prompt_parts.append("\n" + load_template("accountant/role_and_rules.txt"))
    prompt_parts.append("\n" + load_template("accountant/json_schema.txt"))

    return "\n".join(prompt_parts)


def parse_budget_plan(text: str) -> Optional[Dict[str, Any]]:
    """Parse the budget plan JSON from the accountant's response.

    Handles common local-model quirks:
      - Markdown fences (```json ... ```)
      - Thinking models that leak <think>...</think> blocks (qwen3, deepseek-r1)
      - Plain prose preceding or following the JSON
    """
    import re
    text = text.strip()
    # Strip <think>...</think> blocks (thinking-model leakage)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Find last balanced JSON object in text (in case prose precedes it)
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
    "sentry_interval_seconds",
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

    # Programmatic Rule 1 guardrail: after applying whatever the model proposed,
    # if target_wake_minutes * 60 is still less than sentry_interval_seconds,
    # snap target_wake_minutes up to the minimum coherent value. Local models
    # occasionally flip the inequality in their reasoning and propose changes
    # that make coherence WORSE — this catches that.
    try:
        tw = int(ctrl.get("target_wake_minutes") or 60)
        si = int(ctrl.get("sentry_interval_seconds") or 300)
        if tw * 60 < si:
            min_tw = (si + 59) // 60  # ceiling division
            ctrl.set("target_wake_minutes", min_tw, source="accountant")
            changes["target_wake_minutes"] = (
                f"{tw} -> {min_tw} (auto-snap: rule 1 enforcement — tw*60 must be >= sentry_interval)"
            )
    except (ValueError, KeyError, TypeError):
        pass

    return changes
