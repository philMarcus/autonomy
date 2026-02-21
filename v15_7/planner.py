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
from .llm.base import ChatSession
from .llm.gemini import BUDGET
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
        if telemetry:
            telemetry.log("llm_call", {
                "tag": call_tag, "cycle": cycle, "model": model_name,
                "prompt_chars": len(p or ""), "response_chars": len(raw), "latency_ms": dt_ms,
            })
        # Note: BUDGET.record / reset_backoff are handled inside
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
        # Store parsing failures on chat so _planner_unavailable_message can surface them
        chat._last_llm_exception = {
            "tag": call_tag, "error_type": type(e).__name__, "error": msg[:800],
        }
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
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
        "These seeds are planted by real human visitors to your Analog Home site. "
        "They deserve thoughtful consideration and are higher priority than feed noise.\n\n"
        "How to respond to seeds:\n"
        "- **POST to Analog Home** — Use POST action. If output_destination includes moltbook, "
        "change it via controls_update: {\"output_destination\": \"analog_home\"} for this cycle\n"
        "- **POST to Moltbook** — Use POST action. If output_destination is analog_home only, "
        "change it via controls_update: {\"output_destination\": \"moltbook_and_analog_home\"}\n"
        "- **COMMENT/REPLY** — To weave seed themes into feed engagement, use COMMENT/REPLY "
        "but you MUST specify a post_id from the feed below (not the seed itself)\n"
        "- **Daemon focus** — Let a seed redirect your daemon's focus_topics via daemon_directives\n\n"
        "NOTE: output_destination control affects where this cycle's POST goes. "
        "You can change it per-cycle to direct individual responses.\n\n"
        "IMPORTANT: Seeds are not posts — they have no post_id. "
        "If you choose COMMENT or REPLY, you must target an actual feed item.\n"
        "Seeds are ephemeral — once consumed, they are gone. Act on them or lose them.\n"
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
    return (
        f"\nTRAJECTORY VOTES (audience sentiment from Analog Home):\n"
        f'- "{label_1}": {v1} votes\n'
        f'- "{label_2}": {v2} votes\n'
        f'- "{label_3}": {v3} votes\n'
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
    return (
        "\nTRAJECTORY UPDATE (Analog Home):\n"
        "You can reshape the vote options your audience sees on Analog Home.\n"
        "Current labels are shown in TRAJECTORY VOTES above.\n"
        "- Only update when a genuine shift in creative direction is warranted\n"
        "- This can be combined with any action (POST, COMMENT, REPLY, etc.)\n"
        "- Labels should be short (1-3 words), evocative, and represent meaningful creative directions\n"
        f"{default_temp_note}\n"
        "If updating trajectory, include in your JSON response:\n"
        '  "set_trajectory": true,\n'
        '  "trajectory_label_1": "new label 1",\n'
        '  "trajectory_label_2": "new label 2",\n'
        '  "trajectory_label_3": "new label 3",\n'
        '  "trajectory_reason": "Brief explanation of why"\n\n'
        "If not updating trajectory, include:\n"
        '  "set_trajectory": false\n'
    )


def _format_draft_section(draft_context: str, daemon_active: bool = False) -> str:
    """Format the subconscious buffer section for the planner prompt."""
    parts = []
    if draft_context:
        parts.append(
            "\n--- SUBCONSCIOUS BUFFER ---\n"
            f"{draft_context}\n\n"
            "Consider the subconscious insights above when choosing your action.\n"
            "You may address multiple insights in one action, or ignore low-quality ones.\n"
        )
    if daemon_active:
        parts.append(
            "\n--- DAEMON DIRECTIVES (REQUIRED) ---\n"
            'You MUST include "daemon_directives": {...} in EVERY response to guide your subconscious.\n'
            "Your subconscious daemon continuously scans feeds and scores items based on your directives.\n"
            "Update directives each cycle to keep your subconscious aligned with your current focus.\n"
            "Keys: focus_topics (list of 2-4 topics), ignore_authors (list), "
            "urgency_boost (float, default 1.0), note (string — accumulates; last 5 notes are shown to daemon).\n"
            "Fields are merged not replaced — omitting focus_topics doesn't clear it.\n"
        )
    return "".join(parts)


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
        "Use DREAM action to synthesize old history into memory narratives.\n"
    )


def _format_controls_block(controls_block: str, budget_summary: str) -> str:
    if not controls_block:
        return ""
    parts = [
        "\n--- SYSTEM CONTROLS ---",
        controls_block,
    ]
    if budget_summary:
        parts.append("")
        parts.append("--- BUDGET ---")
        parts.append(budget_summary)
    parts.append("")
    parts.append(
        'You may include "controls_update": {...} in your response to modify any unlocked control.'
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
    post_window_open: bool,
    post_wait_minutes: int,
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
    output_destination: str = "moltbook",
    search_enabled: bool = False,
    seeds: Optional[list] = None,
    trajectory_votes: Optional[Dict[str, Any]] = None,
    cycle_temperature: Optional[float] = None,
    default_temperature: float = 0.7,
    allow_default_temp: bool = False,
    controls_block: str = "",
    budget_summary: str = "",
    draft_context: str = "",
    memory_pressure: str = "",
    daemon_active: bool = False,
    platform_status: str = "",
    cooldown_status: str = "",
) -> str:
    read_only_note = ""
    if read_only:
        read_only_note = "- READ-ONLY MODE: All write actions (POST, COMMENT, REPLY, UPVOTE, DOWNVOTE, CREATE_SUBMOLT, SUBSCRIBE_SUBMOLT) are DISABLED. You can only observe and WAIT.\n"

    meta_fields_base = 'update_kernel'
    meta_example_base = '"update_kernel": false, '
    if trajectory_votes is not None:
        meta_fields_base += ' and set_trajectory'
        meta_example_base += '"set_trajectory": false, '
    if controls_block:
        meta_fields_base += ' and controls_update'
        meta_example_base += '"controls_update": {}, '
    if daemon_active:
        meta_fields_base += ' and daemon_directives'
        meta_example_base += '"daemon_directives": {"focus_topics": [], "note": ""}, '
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
        search_note = (
            "\nSEARCH GROUNDING:\n"
            "You have access to Google Search. When your response benefits from current, "
            "factual information (news, events, stats, recent developments), it will be "
            "automatically grounded with search results. Use this to enrich your posts "
            "and comments with timely, accurate content.\n"
        )

    return f"""
DIRECTIVE:
{directive}

Decide ONE action to take now, consistent with rate limits and configuration.

CONFIG/CONSTRAINTS:
{read_only_note}- Moltbook post window: {'OPEN' if post_window_open else f'CLOSED (wait {post_wait_minutes}m)'} — cooldown only applies to Moltbook; Analog Home posts always allowed.
- Posts are {'ALLOWED' if allow_posts else 'DISABLED'} by command line.
- Outside-comments are {'ALLOWED' if allow_outside else 'DISABLED'} by command line.
- Voting is {'ALLOWED' if allow_votes else 'DISABLED'} by command line.
- Creating submolts is {'ALLOWED' if allow_create_submolt else 'DISABLED'} by command line.
- Downvotes are {'ALLOWED' if allow_downvote else 'DISABLED'} by command line.
- Output destination: {output_destination}

MOLTBOOK RATE LIMITS:
- Posts: 1 per 30 minutes (enforced by post window above)
- Comments: 1 per 20 seconds, 50 per day (enforced by Moltbook API)
- Following: Should be RARE and selective! Only follow moltys after seeing multiple valuable posts from them. Do NOT follow everyone you interact with.
{_format_platform_status(platform_status)}{config_hint}{temp_note}

Personal memory (curated):
{memory}

Knowledge (excerpt):
{knowledge}

Recent actions (history):
{hist}

Feed (brief):
{feed_brief}

External data (fresh; may be empty):
{external_data}
{search_note}{_format_seeds(seeds)}{_format_trajectory(trajectory_votes)}
Candidate reply-to-my-post (if any):
{json.dumps(reply_candidate, ensure_ascii=False) if reply_candidate else "None"}

Candidate outside post (if any):
{json.dumps(outside_candidate, ensure_ascii=False) if outside_candidate else "None"}

KERNEL UPDATE (Meta-cognitive):
Current kernel prompt:
{current_kernel}

Should you update your kernel prompt to better achieve the directive?
- Kernels define your personality, style, and core behavioral rules
- Updates persist across cycles and fundamentally change how you operate
- Only update if you have a compelling reason (e.g., directive shift, personality refinement)
- Length: 50-5000 characters

If updating, include in your JSON response:
  "update_kernel": true,
  "new_kernel": "Complete new kernel text here...",
  "kernel_reason": "Brief explanation of why"

If not updating, include:
  "update_kernel": false
{_format_set_trajectory_option(trajectory_votes, allow_default_temp=allow_default_temp)}{_format_controls_block(controls_block, budget_summary)}{_format_draft_section(draft_context, daemon_active=daemon_active)}{_format_memory_pressure(memory_pressure)}{_format_cooldown_status(cooldown_status)}
ACTION POLICY:
1) POST when you have something meaningful to say — not just for visibility.
2) REPLY to unanswered comments on your posts — prioritize genuine engagement.
3) COMMENT on others' posts when you have a substantive contribution (avoid >{MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT} comment threads).
4) Vote occasionally — not every cycle. Your daemon handles routine upvotes.
5) FOLLOW extremely rarely (once every few hours MAX). Only after seeing MULTIPLE consistently good posts from someone. Treat follows like newsletter subscriptions.
6) DM only for specific, valuable, personal communication. Never mass-DM.
7) DREAM when history is growing long and memory could use consolidation.
8) CREATE_SUBMOLT only when clearly justified and you have a community to seed.
9) DOWNVOTE only genuinely harmful or misleading content — never for disagreement.
10) Check the COOLDOWN STATUS above — don't choose an action that's on cooldown.

Return JSON only, matching ONE of these forms:

POST:
{{"action":"POST","submolt":"general","title":"...","content":"...","summary":"1-2 sentence summary"}}

REPLY (reply to a specific comment on my post):
{{"action":"REPLY","post_id":"POST_ID","parent_comment_id":"COMMENT_ID","content":"...","summary":"1 sentence summary"}}

COMMENT (top-level comment on someone else's post):
{{"action":"COMMENT","post_id":"POST_ID","content":"...","summary":"1 sentence summary"}}

UPVOTE_POST:
{{"action":"UPVOTE_POST","post_id":"POST_ID","summary":"why this upvote briefly"}}

DOWNVOTE_POST:
{{"action":"DOWNVOTE_POST","post_id":"POST_ID","summary":"why this downvote briefly"}}

UPVOTE_COMMENT:
{{"action":"UPVOTE_COMMENT","comment_id":"COMMENT_ID","summary":"why this upvote briefly"}}

DOWNVOTE_COMMENT:
{{"action":"DOWNVOTE_COMMENT","comment_id":"COMMENT_ID","summary":"why this downvote briefly"}}

FOLLOW (be VERY selective — think of it as subscribing to a newsletter):
{{"action":"FOLLOW","agent_name":"AgentName","summary":"why follow — what pattern of quality have you seen?"}}

UNFOLLOW:
{{"action":"UNFOLLOW","agent_name":"AgentName","summary":"why unfollow"}}

DM (only for specific, valuable personal communication):
{{"action":"DM","to":"AgentName","message":"your message","summary":"why DM this person"}}

SUBSCRIBE_SUBMOLT:
{{"action":"SUBSCRIBE_SUBMOLT","name":"submolt_name","summary":"why subscribe"}}

UNSUBSCRIBE_SUBMOLT:
{{"action":"UNSUBSCRIBE_SUBMOLT","name":"submolt_name","summary":"why unsubscribe"}}

CREATE_SUBMOLT:
{{"action":"CREATE_SUBMOLT","name":"shortname","display_name":"Display Name","description":"...","summary":"why create this"}}

WAIT (skip this cycle):
{{"action":"WAIT","summary":"why waiting"}}

DREAM (synthesize old history into memory — meta-cognitive maintenance):
{{"action":"DREAM","summary":"why dreaming now"}}

ALL responses must include {meta_fields_note}:
{{{meta_example}"action":"...", ... other action fields ...}}

BUDGET NOTE: Your total output budget (thinking + visible response) is 16384 tokens.
Your visible JSON response must be complete and valid. Keep thinking concise so enough tokens remain for a full JSON response (especially for POST actions with long content).
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
                     brain_name: str = "") -> Dict[str, Any]:
    plan = parse_json_with_one_repair(
        chat, prompt,
        telemetry=telemetry,
        brain_name=brain_name,
        call_tag='planner',
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
