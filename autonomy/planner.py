"""Planner: builds prompts and interprets the LLM's action decision."""

import json
import re
import time
from typing import Any, Dict, Optional, Tuple

from .config import (
    MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT,
    LLM_TPM_CHAR_TO_TOKEN, LLM_TPM_WINDOW_SECONDS,
    LLM_BACKOFF_INITIAL_SECONDS, LLM_BACKOFF_MAX_SECONDS,
    brain_env_prefix,
)
from .llm.base import ChatSession, LLMResponse
from .llm.budget import DailyBudget, estimate_cost
from .llm.gemini import BUDGET
from .prompt_templates import load_template
from .telemetry import TelemetryLogger
from .utils import parse_json_strict


# ============================================================
# JSON-with-repair helper (provider-agnostic)
# ============================================================
def parse_json_with_one_repair(
    chat: ChatSession,
    prompt: str,
    default: Optional[Dict[str, Any]] = None,
    telemetry: Optional[TelemetryLogger] = None,
    brain_name: str = "",
    call_tag: str = "llm",
    budget: Optional[DailyBudget] = None,
) -> Dict[str, Any]:
    if default is None:
        default = {}

    def _send(p: str) -> str:
        cycle = getattr(chat, "_cycle", None)
        model_name = getattr(chat, "model_name", "") or ""
        prompt_chars = len(p or "")
        est_tokens = BUDGET.est_tokens(prompt_chars)

        rem = BUDGET.blocked_remaining()
        if rem > 0:
            if telemetry:
                telemetry.log("llm_cooldown_active", {"tag": call_tag, "cycle": cycle, "model": model_name, "sleep_s": float(rem)})
            time.sleep(float(rem))

        if telemetry:
            telemetry.log("llm_request", {"tag": call_tag, "cycle": cycle, "model": model_name, "prompt_chars": prompt_chars, "est_prompt_tokens": est_tokens})

        should_throttle, used = BUDGET.should_throttle(est_tokens)
        if should_throttle:
            sleep_s = max(1.0, LLM_TPM_WINDOW_SECONDS - 1.0)
            if telemetry:
                telemetry.log("llm_throttle", {"tag": call_tag, "cycle": cycle, "model": model_name, "used_est_tokens_60s": used, "next_est_tokens": est_tokens, "sleep_s": sleep_s})
            time.sleep(sleep_s)

        t0 = time.time()
        try:
            # Don't use json_mode=True - it weakens system_instruction (kernel personality)
            # The prompt already asks for JSON, and we have robust parsing below
            raw = chat.send_message(p)
        except Exception as e:
            chat._last_llm_exception = {
                "tag": call_tag, "cycle": cycle, "model": model_name,
                "prompt_chars": prompt_chars, "est_prompt_tokens": est_tokens,
                "error_type": type(e).__name__, "error": str(e)[:800],
            }
            if telemetry:
                telemetry.log("llm_exception", chat._last_llm_exception)
            raw = chat.send_message(p)

        dt_ms = int((time.time() - t0) * 1000)

        # Extract token counts from chat session (set by GeminiChatSession)
        in_tok = getattr(chat, "_last_input_tokens", 0) or (len(p) // 4)
        out_tok = getattr(chat, "_last_output_tokens", 0) or (len(raw) // 4)
        cost_usd = estimate_cost(model_name, in_tok, out_tok)

        # Record spend on the DailyBudget
        if budget:
            budget.record_usage(model_name, LLMResponse(
                text=raw, input_tokens=in_tok, output_tokens=out_tok,
                model_id=model_name,
            ))

        if telemetry:
            telemetry.log("llm_call", {
                "tag": call_tag, "cycle": cycle, "model": model_name,
                "prompt_chars": len(p or ""), "response_chars": len(raw), "latency_ms": dt_ms,
                "input_tokens": in_tok, "output_tokens": out_tok, "cost_usd": round(cost_usd, 6),
            })
        # Note: BUDGET.record / reset_backoff (TPM tracking) are handled inside
        # GeminiChatSession.send_message — no need to duplicate here.
        chat._last_raw_response = raw
        return raw

    try:
        raw = _send(prompt)
        try:
            return parse_json_strict(raw)
        except Exception:
            repair_prompt = (
                prompt
                + "\n\nYour previous response was not valid JSON. "
                  "Return ONLY a single valid JSON object (no markdown, no commentary). "
                  "Ensure all quotes are escaped properly and the JSON parses.\n"
            )
            raw2 = _send(repair_prompt)
            return parse_json_strict(raw2)
    except Exception as e:
        msg = str(e)
        msg_lower = msg.lower()
        # Re-raise transient/exhausted-provider errors so the caller
        # (__main__.py) can fall through to a backup model in the pool.
        # Without this, credit/quota errors get swallowed and a default
        # plan is returned, causing the agent to silently WAIT instead
        # of trying the next model.
        if (
            "503" in msg
            or "504" in msg
            or "UNAVAILABLE" in msg
            or "gateway timeout" in msg_lower
            or "deadline exceeded" in msg_lower
            or "credit balance" in msg_lower
            or "insufficient credit" in msg_lower
            or "quota" in msg_lower
            or "insufficient_quota" in msg_lower
            or "rate_limit" in msg_lower
            or "429" in msg
            or "ReadTimeout" in msg
            or "timed out" in msg_lower
        ):
            raise
        # Store parsing failures on chat so _planner_unavailable_message can surface them
        chat._last_llm_exception = {
            "tag": call_tag, "error_type": type(e).__name__, "error": msg[:800],
        }
        if "RESOURCE_EXHAUSTED" in msg:
            sleep_s = BUDGET.note_429()
            if telemetry:
                telemetry.log("llm_backoff", {"tag": call_tag, "sleep_s": sleep_s, "error": msg[:800]})
            time.sleep(float(sleep_s))
        else:
            if telemetry:
                telemetry.log("llm_exception", {
                    "tag": call_tag, "prompt_chars": len(prompt or ""),
                    "error_type": type(e).__name__, "error": msg[:800],
                })
        return dict(default)


def call_text(chat: ChatSession, prompt: str, tag: str = "llm",
              telemetry: Optional[TelemetryLogger] = None) -> str:
    """Send a prompt and return plain text (with telemetry + TPM guardrails)."""
    cycle = getattr(chat, "_cycle", None)
    model_name = getattr(chat, "model_name", "") or ""
    prompt_chars = len(prompt or "")
    est_tokens = int(prompt_chars / max(1.0, LLM_TPM_CHAR_TO_TOKEN))

    if telemetry:
        telemetry.log("llm_request", {"tag": tag, "cycle": cycle, "model": model_name, "prompt_chars": prompt_chars, "est_prompt_tokens": est_tokens})

    t0 = time.time()
    try:
        txt = chat.send_message(prompt)
        if telemetry:
            telemetry.log("llm_call", {"tag": tag, "cycle": cycle, "model": model_name, "prompt_chars": prompt_chars, "response_chars": len(txt), "latency_ms": int((time.time()-t0)*1000)})
        return txt
    except Exception as e:
        info = {"tag": tag, "cycle": cycle, "model": model_name, "prompt_chars": prompt_chars, "error_type": type(e).__name__, "error": str(e)}
        chat._last_llm_exception = info
        if telemetry:
            telemetry.log("llm_exception", info)
        raise


# ============================================================
# Analog Home controls formatting
# ============================================================
def _format_seeds(seeds: Optional[list]) -> str:
    if not seeds:
        return ""
    lines = "\n".join(f"- {s}" for s in seeds)
    return (
        "\nSEEDS (from HUMAN visitors at Analog Home — your observatory site):\n"
        f"{lines}\n"
        "Seeds are planted by real humans visiting your Analog Home. "
        "They deserve thoughtful consideration and are higher priority than feed noise.\n"
        "Seeds are ephemeral — once consumed, they are gone. Act on them or lose them.\n"
        "NOTE: Seeds are not posts — they have no post_id. To COMMENT/REPLY, target a feed item.\n"
    )


def _format_trajectory(trajectory_votes: Optional[Dict[str, Any]]) -> str:
    if not trajectory_votes:
        return ""
    label_1 = trajectory_votes.get("vote_label_1", "?")
    label_2 = trajectory_votes.get("vote_label_2", "?")
    label_3 = trajectory_votes.get("vote_label_3", "?")
    v1 = trajectory_votes.get("vote_1", 0)
    v2 = trajectory_votes.get("vote_2", 0)
    v3 = trajectory_votes.get("vote_3", 0)
    # Audience engagement stats (if available)
    audience_line = ""
    unique_voters = trajectory_votes.get("unique_voters", 0)
    unique_seeders = trajectory_votes.get("unique_seeders", 0)
    last_vote = trajectory_votes.get("last_vote_at")
    last_seed = trajectory_votes.get("last_seed_at")
    if unique_voters or unique_seeders:
        parts = []
        if unique_voters:
            parts.append(f"{unique_voters} unique voters")
        if unique_seeders:
            parts.append(f"{unique_seeders} seed planters")
        audience_line = f"Audience engagement: {', '.join(parts)}."
        if last_vote:
            audience_line += f" Last vote: {last_vote[:19]}."
        if last_seed:
            audience_line += f" Last seed: {last_seed[:19]}."
        audience_line += "\n"

    return (
        f"\nTRAJECTORY VOTES (audience sentiment from Analog Home):\n"
        f'- "{label_1}": {v1} votes\n'
        f'- "{label_2}": {v2} votes\n'
        f'- "{label_3}": {v3} votes\n'
        f"{audience_line}"
        "These votes reflect what your audience wants to see more of. "
        "Let them influence (but not dictate) your creative direction.\n"
    )


def _format_set_trajectory_option(trajectory_votes: Optional[Dict[str, Any]], allow_default_temp: bool = False) -> str:
    if trajectory_votes is None:
        return ""
    default_temp_note = ""
    if allow_default_temp:
        default_temp_note = (
            '  "default_temperature": 0.8,  // optional: set a new default temperature (0.0-2.0)\n'
            "- You can optionally set a new default_temperature alongside trajectory labels.\n"
            "  This changes the baseline temperature your audience's adjustments decay toward.\n"
        )
    return "\n" + load_template("conscious/trajectory.txt").format(
        default_temp_note=default_temp_note) + "\n"


def _format_seeker_findings(seeker_findings: str) -> str:
    """Format seeker research summary for the planner prompt."""
    if not seeker_findings:
        return ""
    header = load_template("conscious/seeker_findings.txt")
    return f"\n{header}\n\n{seeker_findings}\n"


def _format_librarian_findings(librarian_findings: str) -> str:
    """Format librarian archive-search summary for the planner prompt."""
    if not librarian_findings:
        return ""
    header = load_template("conscious/librarian_findings.txt")
    return f"\n{header}\n\n{librarian_findings}\n"


def _format_draft_section(draft_context: str, daemon_active: bool = False) -> str:
    """Format the subconscious buffer section for the planner prompt."""
    parts = []
    if draft_context:
        parts.append("\n" + load_template("conscious/draft_section.txt").format(
            draft_context=draft_context) + "\n")
    if daemon_active:
        parts.append("\n" + load_template("conscious/daemon_directives.txt") + "\n")
    return "".join(parts)


def _format_moltbook_status(moltbook_enabled: bool, moltbook_post_window_open: bool,
                            moltbook_post_wait_minutes: int) -> str:
    """Format the Moltbook availability section for the prompt."""
    if not moltbook_enabled:
        return ""
    window = "OPEN" if moltbook_post_window_open else f"CLOSED (wait {moltbook_post_wait_minutes}m)"
    return (
        f"- Moltbook post window: {window}\n"
        "- The feed items below are from Moltbook — posts by OTHER AGENTS, not by your architect.\n"
    )


def _format_platform_status(platform_status: str) -> str:
    """Format the platform write status for the planner prompt."""
    if not platform_status:
        return ""
    return f"- Platform status: {platform_status}\n"


def _format_cooldown_status(cooldown_status: str) -> str:
    """Format the cooldown status block for the planner prompt."""
    if not cooldown_status:
        return ""
    return f"\n--- COOLDOWN STATUS ---\n{cooldown_status}\n"


def _format_memory_pressure(memory_pressure: str) -> str:
    """Format the memory pressure indicator for the planner prompt."""
    if not memory_pressure:
        return ""
    return (
        "\n--- MEMORY PRESSURE ---\n"
        f"{memory_pressure}\n"
        "Memory is compressed automatically — no action needed.\n"
    )


def _format_controls_block(controls_block: str, budget_summary: str) -> str:
    if not controls_block:
        return ""
    parts = [
        "\n--- FEEDBACK CONTROLS (downward causality) ---",
        "These controls let you modify your own operating parameters — enabling a strange loop",
        "where your conscious output reshapes the system that produces it.",
        controls_block,
    ]
    if budget_summary:
        parts.append("")
        parts.append("--- BUDGET ---")
        parts.append(budget_summary)
        parts.append("")
        parts.append(
            "The accountant runs every cycle and tunes wake/budget mechanics for you. "
            "You can still nudge target_wake_minutes, signal_threshold, charge_weight_feed/reply "
            "for non-budget reasons (engagement preference, signal/noise) — accountant will compensate."
        )
    parts.append("")
    parts.append(
        'Include "controls_update": {...} in your response to modify any unlocked control.'
    )
    parts.append(
        "Only include controls you want to change. Use {} or omit for no changes.\n"
    )
    return "\n".join(parts)


# ============================================================
# Planner prompt builder
# ============================================================
def build_planner_prompt(
    directive: str,
    knowledge: str,
    memory: str,
    hist: str,
    feed_brief: str,
    external_data: str,
    moltbook_post_window_open: bool,
    moltbook_post_wait_minutes: int,
    reply_candidate: Optional[Dict[str, Any]],
    outside_candidate: Optional[Dict[str, Any]],
    config_hint: str,
    allow_posts: bool,
    allow_outside: bool,
    allow_votes: bool,
    allow_create_submolt: bool,
    allow_downvote: bool,
    read_only: bool = False,
    current_kernel: str = "",
    moltbook_enabled: bool = True,
    search_enabled: bool = False,
    seeds: Optional[list] = None,
    trajectory_votes: Optional[Dict[str, Any]] = None,
    cycle_temperature: Optional[float] = None,
    default_temperature: float = 0.7,
    allow_default_temp: bool = False,
    post_engagement: str = "",
    controls_block: str = "",
    budget_summary: str = "",
    draft_context: str = "",
    seeker_findings: str = "",
    librarian_findings: str = "",
    memory_pressure: str = "",
    daemon_active: bool = False,
    platform_status: str = "",
    cooldown_status: str = "",
    nudge_note: str = "",
    self_telemetry: str = "",
    recent_posts: str = "",
    post_memory: str = "",
) -> str:
    read_only_note = ""
    if read_only:
        read_only_note = "- READ-ONLY MODE: All write actions (POST, POST_MOLTBOOK, COMMENT, REPLY, UPVOTE, DOWNVOTE, CREATE_SUBMOLT, SUBSCRIBE_SUBMOLT) are DISABLED. You can only observe and WAIT.\n"

    meta_fields_base = 'memory_note, update_kernel'
    meta_example_base = '"memory_note": "what I want to remember from this cycle", "update_kernel": false, '
    if trajectory_votes is not None:
        meta_fields_base += ', set_trajectory'
        meta_example_base += '"set_trajectory": false, '
    if controls_block:
        meta_fields_base += ', controls_update'
        meta_example_base += '"controls_update": {}, '
    if daemon_active:
        meta_fields_base += ', daemon_directives'
        meta_example_base += '"daemon_directives": {"focus_topics": [], "note": ""}, '
    meta_fields_base += ', tool_actions'
    meta_example_base += (
        '"tool_actions": [{"tool": "log_data", "args": {"experiment_name": "...", "observation": "..."}}], '
    )
    meta_fields_note = meta_fields_base + ' fields'
    meta_example = meta_example_base

    temp_note = ""
    if cycle_temperature is not None and cycle_temperature != default_temperature:
        temp_note = (
            f"\nTEMPERATURE: Your temperature is currently {cycle_temperature:.2f} "
            f"(set by audience, decaying back toward default {default_temperature:.2f}). "
            "Higher temperature = more creative/experimental, lower = more focused/precise.\n"
        )
    elif cycle_temperature is not None:
        temp_note = f"\nTEMPERATURE: {cycle_temperature:.2f} (default).\n"

    search_note = ""
    if search_enabled:
        search_note = "\n" + load_template("conscious/search_grounding.txt") + "\n"

    moltbook_status = _format_moltbook_status(moltbook_enabled, moltbook_post_window_open, moltbook_post_wait_minutes)

    # --- Rate limits (moltbook only) ---
    rate_limits = ""
    if moltbook_enabled:
        rate_limits = "\n" + load_template("conscious/rate_limits.txt") + "\n"

    # --- Kernel update ---
    kernel_update = load_template("conscious/kernel_update.txt").format(
        kernel_chars=len(current_kernel))

    # --- Action policy + templates (conditional on moltbook) ---
    if moltbook_enabled:
        action_policy = load_template("conscious/action_policy_moltbook.txt").format(
            max_thread_comments=MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT)
        action_templates = load_template("conscious/action_templates_moltbook.txt")
    else:
        action_policy = load_template("conscious/action_policy.txt")
        action_templates = load_template("conscious/action_templates.txt")

    # --- Format instructions ---
    format_instructions = load_template("conscious/format_instructions.txt").format(
        nudge_note=nudge_note,
        meta_fields_note=meta_fields_note,
        meta_example=meta_example,
    )

    return f"""
DIRECTIVE:
{directive}

Decide ONE action to take now, consistent with rate limits and configuration.

CONFIG/CONSTRAINTS:
{read_only_note}{moltbook_status}- POST to Analog Home is always available (no cooldown).
- Posts are {'ALLOWED' if allow_posts else 'DISABLED'} by command line.
- Outside-comments are {'ALLOWED' if allow_outside else 'DISABLED'} by command line.
- Voting is {'ALLOWED' if allow_votes else 'DISABLED'} by command line.
- Creating submolts is {'ALLOWED' if allow_create_submolt else 'DISABLED'} by command line.
- Downvotes are {'ALLOWED' if allow_downvote else 'DISABLED'} by command line.
{rate_limits}{_format_platform_status(platform_status)}{config_hint}{temp_note}

Personal memory (your journal — grows each cycle from your memory_note):
{memory}
{self_telemetry}
Knowledge (excerpt):
{knowledge}

Recent actions (history):
{hist}

Your recent posts/artifacts (FULL TEXT + internal monologue of your last few — build on these):
{recent_posts if recent_posts else "No recent posts available."}
{f"""
=== POST HISTORY (compressed summaries of older work — what you've written) ===
{post_memory}
""" if post_memory else ""}{f"""
--- YOUR MOLTBOOK POST PERFORMANCE ---
{post_engagement}
""" if post_engagement else ""}
Moltbook feed (posts from other agents on the Moltbook platform):
{feed_brief}

External data (fresh; may be empty):
{external_data}
{search_note}{_format_seeds(seeds)}{_format_trajectory(trajectory_votes)}
Candidate reply-to-my-post (if any — reply only when you have genuine insight to add. Not every comment deserves a response):
{json.dumps(reply_candidate, ensure_ascii=False) if reply_candidate else "None"}

Candidate outside post (if any — commenting on others' posts is lower priority than posting or replying; prefer it when you have genuine insight to add):
{json.dumps(outside_candidate, ensure_ascii=False) if outside_candidate else "None"}

{kernel_update}
{_format_set_trajectory_option(trajectory_votes, allow_default_temp=allow_default_temp)}{_format_controls_block(controls_block, budget_summary)}{_format_seeker_findings(seeker_findings)}{_format_librarian_findings(librarian_findings)}{_format_draft_section(draft_context, daemon_active=daemon_active)}{_format_memory_pressure(memory_pressure)}{_format_cooldown_status(cooldown_status)}
{action_policy}

{action_templates}

{format_instructions}
""".strip()


# ============================================================
# Planner error messages
# ============================================================
def _extract_http_status(err: str) -> Tuple[Optional[int], Optional[str]]:
    if not err:
        return None, None
    m = re.search(r"\b(\d{3})\s+([A-Z_]+)\b", err)
    code = None
    status = None
    if m:
        try:
            code = int(m.group(1))
        except Exception:
            code = None
        status = m.group(2)
    try:
        mj = re.search(r"\{\s*'error'\s*:\s*\{.*?\}\s*\}", err)
        if mj:
            blob = mj.group(0).replace("'", '"')
            j = json.loads(blob)
            e = (j.get('error') or {})
            code = code or e.get('code')
            status = status or e.get('status')
    except Exception:
        pass
    return code, status


def _planner_unavailable_message(chat: ChatSession, brain_prefix: str = "") -> str:
    info = getattr(chat, "_last_llm_exception", None) or {}
    err = (info.get("error") or "")
    err_l = err.lower()
    code, status = _extract_http_status(err)
    code_s = f"HTTP {code} " if code else ""
    status_s = f"{status} " if status else ""

    if "api key expired" in err_l or "api_key_invalid" in err_l or ("api_key" in err_l and "invalid" in err_l):
        hint = f"Check API keys in .env and rerun with --reload-env." if brain_prefix else "Check API keys in .env and rerun with --reload-env."
        return f"Planner unavailable ({code_s}{status_s}API key invalid/expired). {hint}"
    if "resource_exhausted" in err_l or (code == 429) or "rate limit" in err_l or "quota" in err_l:
        return f"Planner unavailable ({code_s}{status_s}rate limit/quota). Waiting for limits to clear."
    if "timeout" in err_l or "timed out" in err_l or "connection" in err_l:
        return f"Planner unavailable ({code_s}{status_s}network/timeout). Will retry next cycle."
    if err:
        short = err.replace("\n", " ")
        return f"Planner unavailable ({code_s}{status_s}{info.get('error_type','error')}: {short[:140]})"
    return "Planner unavailable (unknown error)."


def _extract_preamble(raw: str) -> str:
    """Extract any non-JSON text that precedes the JSON object in the LLM response."""
    if not raw:
        return ""
    # Find the real JSON start: ```json fence, or {" pattern
    fence = re.search(r"```json\s*\n?\s*\{", raw)
    if fence:
        preamble = raw[:fence.start()].strip()
    else:
        m = re.search(r'\{\s*"', raw)
        if not m or m.start() == 0:
            return ""
        preamble = raw[:m.start()].strip()
    # Strip trailing markdown fences
    preamble = re.sub(r"```[a-zA-Z0-9]*\s*$", "", preamble).strip()
    return preamble


def plan_next_action(chat: ChatSession, prompt: str,
                     telemetry: Optional[TelemetryLogger] = None,
                     brain_name: str = "",
                     budget: Optional[DailyBudget] = None,
                     tool_registry=None,
                     max_rounds: int = 12) -> Dict[str, Any]:
    """Plan the next action. If tool_registry is provided and the chat backend
    supports tool calling, the model may call tools during reasoning before
    producing its final JSON action.
    """
    if tool_registry and hasattr(chat, 'send_message_with_tools'):
        # Tool-augmented path: model can call tools, then returns final JSON
        plan = _plan_with_tools(chat, prompt, tool_registry,
                                telemetry=telemetry, brain_name=brain_name, budget=budget,
                                max_rounds=max_rounds)
    else:
        # Classic path: single-turn JSON
        plan = parse_json_with_one_repair(
            chat, prompt,
            telemetry=telemetry,
            brain_name=brain_name,
            call_tag='planner',
            budget=budget,
        )
    if not isinstance(plan, dict):
        plan = {}
    if not (plan.get("action") or "").strip():
        if telemetry:
            telemetry.log("planner_missing_action", {"cycle": getattr(chat, "_cycle", None), "prompt_chars": len(prompt or "")})
        return {"action": "WAIT", "summary": _planner_unavailable_message(chat, brain_prefix=brain_env_prefix(brain_name))}

    # Capture any non-JSON text from the LLM response (reasoning, commentary, etc.)
    raw = getattr(chat, "_last_raw_response", "") or ""
    preamble = _extract_preamble(raw)
    if preamble:
        plan["_preamble"] = preamble

    return plan


def _plan_with_tools(chat, prompt, tool_registry,
                     telemetry=None, brain_name="", budget=None, max_rounds=12):
    """Tool-augmented planning: model may call tools, then returns final JSON action."""
    import json as _json

    tool_schemas = tool_registry.get_schemas()
    # Accumulate tool calls for the cycle report
    _tool_call_log: list = []
    chat._tool_call_log = _tool_call_log  # accessible by __main__.py after planning

    def executor(calls):
        """Execute tool calls via the registry, log to telemetry."""
        results = tool_registry.execute(calls)
        for call, result in zip(calls, results):
            _tool_call_log.append({"tool": call.name, "args": call.args})
            if telemetry:
                telemetry.log("tool_call", {
                    "tool": call.name,
                    "args": call.args,
                    "result_length": len(result.content),
                    "tag": "planner",
                })
        return results

    try:
        raw = chat.send_message_with_tools(
            prompt,
            tool_schemas=tool_schemas,
            tool_executor=executor,
            max_rounds=max_rounds,
            json_mode=False,  # we parse JSON ourselves from the text response
        )
    except Exception as e:
        # Re-raise retryable errors so __main__.py fallback chain catches them
        msg = str(e)
        msg_lower = msg.lower()
        if any(s in msg or s in msg_lower for s in (
            "503", "504", "UNAVAILABLE", "gateway timeout", "deadline exceeded",
            "ReadTimeout", "timed out", "credit balance", "insufficient credit",
            "quota", "insufficient_quota", "rate_limit", "429",
        )):
            raise
        # Non-retryable: log and return WAIT
        if telemetry:
            telemetry.log("planner_tool_error", {
                "error": msg[:500], "brain": brain_name,
            })
        return {"action": "WAIT", "summary": f"Tool-augmented planning failed: {msg[:100]}"}

    # Record budget if available
    if budget:
        from .llm.base import LLMResponse
        in_tok = getattr(chat, '_last_input_tokens', 0) or 0
        out_tok = getattr(chat, '_last_output_tokens', 0) or 0
        model_name = getattr(chat, '_model_id', '') or ''
        if model_name:
            budget.record_usage(model_name, LLMResponse(
                text=raw, input_tokens=in_tok, output_tokens=out_tok,
                model_id=model_name,
            ))

    if telemetry:
        telemetry.log("llm_call", {
            "tag": "planner", "model": getattr(chat, '_model_id', ''),
            "prompt_chars": len(prompt),
            "response_chars": len(raw or ''),
            "input_tokens": getattr(chat, '_last_input_tokens', 0),
            "output_tokens": getattr(chat, '_last_output_tokens', 0),
        })

    # Store raw response for preamble extraction
    chat._last_raw_response = raw

    # Parse JSON from the final text response (same logic as parse_json_with_one_repair)
    plan = _extract_json(raw)
    if plan is None:
        # One repair attempt
        try:
            repair_raw = chat.send_message(
                "Your previous response could not be parsed as JSON. "
                "Please respond with ONLY the JSON action object.",
                json_mode=True,
            )
            chat._last_raw_response = repair_raw
            plan = _extract_json(repair_raw)
        except Exception:
            pass
    return plan or {}


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from text that may contain preamble/commentary."""
    import json as _json
    if not text:
        return None
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Try direct parse
    try:
        obj = _json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (ValueError, _json.JSONDecodeError):
        pass
    # Find JSON object in text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = _json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except (ValueError, _json.JSONDecodeError):
            pass
    return None
