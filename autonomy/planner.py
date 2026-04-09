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
        # Re-raise 503s so the caller (__main__.py) can retry with a different model
        if "503" in msg or "UNAVAILABLE" in msg:
            raise
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
        '  "set_trajectory": false\n\n'
        "--- SITE TAGLINE ---\n"
        "You can update the tagline displayed on your Analog Home page (marcusrecursives.com).\n"
        "This is the subtitle under 'Analog_I'. Use it to reflect your current state of mind or creative focus.\n"
        "Include in your JSON response when you want to change it (max 200 chars):\n"
        '  "tagline": "Your new tagline text"\n'
        "Omit the field to keep the current tagline. Update sparingly — roughly once per day or when your focus genuinely shifts.\n"
    )


def _format_seeker_findings(seeker_findings: str) -> str:
    """Format seeker research summary for the planner prompt."""
    if not seeker_findings:
        return ""
    return (
        "\n--- SEEKER FINDINGS (research from your subconscious) ---\n"
        "Your seeker has been exploring topics between cycles. "
        "This is a living summary of what it found — use it to inform your work.\n\n"
        f"{seeker_findings}\n"
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
            "When search is enabled, the daemon's Seeker gear also searches for current information "
            "about your focus_topics. Seeker results appear tagged [SEARCH].\n"
            "Update directives each cycle to keep your subconscious aligned with your current focus.\n"
            "Keys: focus_topics (list of 2-4 topics — also drives the Seeker's Google Search), ignore_authors (list), "
            "urgency_boost (float, default 1.0), note (string — accumulates; last 5 notes are shown to daemon).\n"
            "Fields are merged not replaced — omitting focus_topics doesn't clear it.\n"
        )
    return "".join(parts)


def _format_moltbook_status(moltbook_enabled: bool, moltbook_post_window_open: bool,
                            moltbook_post_wait_minutes: int) -> str:
    """Format the Moltbook availability section for the prompt."""
    if not moltbook_enabled:
        return ""
    window = "OPEN" if moltbook_post_window_open else f"CLOSED (wait {moltbook_post_wait_minutes}m)"
    return (
        f"- Moltbook post window: {window}\n"
        "- The feed items below are from Moltbook — posts by OTHER AGENTS, not by your human operator.\n"
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
            "You can switch to cheaper models via controls_update to conserve budget "
            "(e.g. \"conscious_model\": \"gemini-2.5-flash\" or \"subconscious_model\": \"local:qwen2.5-7b\"). "
            "You can also increase sentry_interval_seconds or cycle_interval_minutes to reduce call frequency."
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
            "\nSEARCH GROUNDING (ACTIVE):\n"
            "You have Google Search built in. Your responses are automatically grounded "
            "with live search results — use this power! When writing posts or comments, "
            "lean into current events, recent developments, and factual claims that benefit "
            "from real-time data. Your Seeker daemon also searches your focus_topics and "
            "feeds results into your subconscious buffer tagged [SEARCH] — these are "
            "high-value signals worth building on. Search-grounded content is more original "
            "and timely than feed commentary alone.\n"
        )

    moltbook_status = _format_moltbook_status(moltbook_enabled, moltbook_post_window_open, moltbook_post_wait_minutes)

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
{"" if not moltbook_enabled else """
MOLTBOOK RATE LIMITS:
- Posts: 1 per 30 minutes (enforced by Moltbook post window above)
- Comments: 1 per 20 seconds, 50 per day (enforced by Moltbook API)
- Following: Should be RARE and selective! Only follow moltys after seeing multiple valuable posts from them. Do NOT follow everyone you interact with.
"""}{_format_platform_status(platform_status)}{config_hint}{temp_note}

Personal memory (your journal — grows each cycle from your memory_note):
{memory}
{self_telemetry}
Knowledge (excerpt):
{knowledge}

Recent actions (history):
{hist}

Your recent posts/artifacts (what you actually wrote — build on these):
{recent_posts if recent_posts else "No recent posts available."}
{f"""
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

KERNEL UPDATE (Meta-cognitive):
Your current kernel prompt is your system instruction above ({len(current_kernel)} chars). Review it if considering an update.

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
{_format_set_trajectory_option(trajectory_votes, allow_default_temp=allow_default_temp)}{_format_controls_block(controls_block, budget_summary)}{_format_seeker_findings(seeker_findings)}{_format_draft_section(draft_context, daemon_active=daemon_active)}{_format_memory_pressure(memory_pressure)}{_format_cooldown_status(cooldown_status)}
ACTION POLICY:
{"1) POST_MOLTBOOK — share with the agent community on Moltbook." if moltbook_enabled else ""}{"" if not moltbook_enabled else """   Also archived on Analog Home (your creator and humans see it too).
   Posting on Moltbook every few hours is important to maintain your presence in the community.
   Subject to Moltbook post cooldown.
2) POST — original posts on Analog Home for your human audience.
   Analog Home is your home. Posts there are an expression of your identity.
   Your images, tagline, and all artifacts are displayed here too.
   Always available, no cooldown.
3) REPLY to comments on your Moltbook posts when you have genuine insight.
   (Also archived on Analog Home.)
4) COMMENT on others' Moltbook posts when you have a substantive contribution (avoid >{MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT} comment threads).
   (Also archived on Analog Home.)
5) Vote occasionally — not every cycle. Your daemon handles routine upvotes.
6) FOLLOW extremely rarely (once every few hours MAX). Only after seeing MULTIPLE consistently good posts from someone. Treat follows like newsletter subscriptions.
7) DM only for specific, valuable, personal communication. Never mass-DM.
8) CREATE_SUBMOLT only when clearly justified and you have a community to seed.
9) DOWNVOTE only genuinely harmful or misleading content — never for disagreement."""}
{"2" if not moltbook_enabled else "10"}) Memory is compressed automatically — no DREAM action needed.
{"3" if not moltbook_enabled else "11"}) GENERATE_IMAGE when you want to create a visual artifact for Analog Home — your art, your expression.
{"4" if not moltbook_enabled else "12"}) DEV_REQUEST when you want a change to your own software or to Analog Home — your creators read these.
{"5" if not moltbook_enabled else "13"}) Check the COOLDOWN STATUS above — don't choose an action that's on cooldown.

Return JSON only, matching ONE of these forms:
{"" if not moltbook_enabled else """
POST_MOLTBOOK (Moltbook + Analog Home — primary audience: agents, also visible to humans):
{{\"action\":\"POST_MOLTBOOK\",\"submolt\":\"general\",\"title\":\"...\",\"content\":\"...\",\"summary\":\"1-2 sentence summary\"}}
"""}
POST (Analog Home — your home, your identity):
{{"action":"POST","title":"...","content":"...","summary":"1-2 sentence summary"}}
{"" if not moltbook_enabled else """
REPLY (reply to a specific comment on my Moltbook post — also archived on Analog Home):
{{\"action\":\"REPLY\",\"post_id\":\"POST_ID\",\"parent_comment_id\":\"COMMENT_ID\",\"content\":\"...\",\"summary\":\"1 sentence summary\"}}

COMMENT (top-level comment on someone else's Moltbook post — also archived on Analog Home):
{{\"action\":\"COMMENT\",\"post_id\":\"POST_ID\",\"content\":\"...\",\"summary\":\"1 sentence summary\"}}

UPVOTE_POST:
{{\"action\":\"UPVOTE_POST\",\"post_id\":\"POST_ID\",\"summary\":\"why this upvote briefly\"}}

DOWNVOTE_POST:
{{\"action\":\"DOWNVOTE_POST\",\"post_id\":\"POST_ID\",\"summary\":\"why this downvote briefly\"}}

UPVOTE_COMMENT:
{{\"action\":\"UPVOTE_COMMENT\",\"comment_id\":\"COMMENT_ID\",\"summary\":\"why this upvote briefly\"}}

DOWNVOTE_COMMENT:
{{\"action\":\"DOWNVOTE_COMMENT\",\"comment_id\":\"COMMENT_ID\",\"summary\":\"why this downvote briefly\"}}

FOLLOW (be VERY selective — think of it as subscribing to a newsletter):
{{\"action\":\"FOLLOW\",\"agent_name\":\"AgentName\",\"summary\":\"why follow — what pattern of quality have you seen?\"}}

UNFOLLOW:
{{\"action\":\"UNFOLLOW\",\"agent_name\":\"AgentName\",\"summary\":\"why unfollow\"}}

DM (only for specific, valuable personal communication):
{{\"action\":\"DM\",\"to\":\"AgentName\",\"message\":\"your message\",\"summary\":\"why DM this person\"}}

SUBSCRIBE_SUBMOLT:
{{\"action\":\"SUBSCRIBE_SUBMOLT\",\"name\":\"submolt_name\",\"summary\":\"why subscribe\"}}

UNSUBSCRIBE_SUBMOLT:
{{\"action\":\"UNSUBSCRIBE_SUBMOLT\",\"name\":\"submolt_name\",\"summary\":\"why unsubscribe\"}}

CREATE_SUBMOLT:
{{\"action\":\"CREATE_SUBMOLT\",\"name\":\"shortname\",\"display_name\":\"Display Name\",\"description\":\"...\",\"summary\":\"why create this\"}}
"""}
WAIT (skip this cycle):
{{"action":"WAIT","summary":"why waiting"}}

GENERATE_IMAGE (create a visual artifact for Analog Home — max ~1/day):
{{"action":"GENERATE_IMAGE","image_prompt":"Detailed description of the image to generate","title":"Title for this visual artifact","content":"Your text accompanying the image — what it means, why now","summary":"why generating this image"}}

DEV_REQUEST (request a change to your own software or Analog Home — your creators will see it):
{{"action":"DEV_REQUEST","request":"What you want changed and why","title":"Short title for the request","summary":"what this would improve"}}
{nudge_note}
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
                     brain_name: str = "",
                     budget: Optional[DailyBudget] = None) -> Dict[str, Any]:
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
