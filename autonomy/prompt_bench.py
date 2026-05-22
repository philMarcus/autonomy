"""Prompt Bench — extract real, data-populated prompts for every gear.

Generates one directory per gear under prompts/, each containing:
  system.txt   — system instruction (paste as --system in Ollama)
  user.txt     — user prompt
  metadata.json — model, temperature, max_tokens, notes

Usage:
    python -m autonomy.prompt_bench ANALOG_I
    python -m autonomy.prompt_bench ANALOG_I --output-dir /tmp/prompts
    python -m autonomy.prompt_bench ANALOG_I --gear conscious
"""

import argparse
import json
import os
import random
import sys

from .config import BRAINS_DIR, brain_env_prefix, load_dotenv


def _load_state(brain_name: str) -> dict:
    """Load brain state from memories JSON."""
    from .utils import load_state as _ls
    path = os.path.join(BRAINS_DIR, f"{brain_name}_memories.json")
    return _ls(path) if os.path.exists(path) else {}


def _load_kernel(brain_name: str) -> str:
    from .utils import load_kernel as _lk
    return _lk(os.path.join(BRAINS_DIR, f"{brain_name}_kernel_prompt.txt"))


def _load_knowledge(brain_name: str) -> str:
    from .utils import load_knowledge as _lkn
    return _lkn(os.path.join(BRAINS_DIR, f"{brain_name}_knowledge.txt"))


def _load_controls(brain_name: str):
    """Build a ControlRegistry from defaults + controls.json overrides."""
    from .llm import ModelRegistry
    from .controls import build_default_registry
    registry = ModelRegistry()
    ctrl = build_default_registry(registry, blacklist_str="")
    controls_path = os.path.join(BRAINS_DIR, f"{brain_name}_controls.json")
    if os.path.exists(controls_path):
        with open(controls_path, "r") as f:
            ctrl.load_from_dict(json.load(f))
    return ctrl, registry


def _load_budget(ctrl):
    from .llm import DailyBudget
    return DailyBudget(daily_limit_usd=float(ctrl.get("daily_budget_usd") or 1.0))


def _write_gear(output_dir: str, gear_name: str, system: str, user: str, metadata: dict):
    """Write system.txt, user.txt, metadata.json for one gear."""
    gear_dir = os.path.join(output_dir, gear_name)
    os.makedirs(gear_dir, exist_ok=True)
    with open(os.path.join(gear_dir, "system.txt"), "w", encoding="utf-8") as f:
        f.write(system)
    with open(os.path.join(gear_dir, "user.txt"), "w", encoding="utf-8") as f:
        f.write(user)
    with open(os.path.join(gear_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    sys_len = len(system)
    usr_len = len(user)
    print(f"  {gear_name:25s}  system={sys_len:>6,} chars  user={usr_len:>6,} chars")


# ── Gear extractors ──────────────────────────────────────────

def extract_conscious(brain_name, state, kernel, knowledge, ctrl, budget, registry):
    """Full conscious planner prompt with real data."""
    from .utils import memory_context, history_context, format_feed_brief
    from .planner import build_planner_prompt
    from .cooldowns import cooldown_status_text

    # Check for a live snapshot first (written by the agent each cycle)
    snap_dir = os.path.join("prompts", "snapshots", "conscious")
    snap_user = os.path.join(snap_dir, "user.txt")
    snap_sys = os.path.join(snap_dir, "system.txt")
    if os.path.exists(snap_user) and os.path.exists(snap_sys):
        with open(snap_sys, "r", encoding="utf-8") as f:
            _snap_kernel = f.read()
        with open(snap_user, "r", encoding="utf-8") as f:
            _snap_prompt = f.read()
        snap_meta = {}
        _snap_meta_path = os.path.join(snap_dir, "metadata.json")
        if os.path.exists(_snap_meta_path):
            with open(_snap_meta_path, "r") as f:
                snap_meta = json.load(f)
        snap_meta.update({
            "gear": "conscious",
            "notes": f"LIVE SNAPSHOT from cycle {snap_meta.get('cycle', '?')}. "
                     "Exact prompt sent to the conscious model.",
        })
        return _snap_kernel, _snap_prompt, snap_meta

    # No snapshot available — build from state (feed/drafts will be empty)
    directive = state.get("directive", "Participate on Moltbook.")
    mem = memory_context(state)
    hist = history_context(state, n=int(ctrl.get("history_context_n") or 15))
    feed = format_feed_brief([], feed_item_chars=int(ctrl.get("feed_item_chars") or 400))
    budget_summary = budget.spend_summary_for_planning(registry) if registry else ""
    controls_block = ctrl.to_llm_block()
    cooldown_status = cooldown_status_text(state, ctrl=ctrl)

    # Build recent posts from state
    fresh = state.get("_post_memory_fresh", [])
    recent_posts = ""
    if fresh:
        parts = []
        for p in fresh[:4]:
            title = p.get("title", "")
            body = p.get("body", p.get("content", ""))[:2000]
            parts.append(f"--- {title} ---\n{body}")
        recent_posts = "\n\n".join(parts)

    # Post memory tiers
    post_tiers = state.get("post_tiers", {})
    post_memory = ""
    if post_tiers:
        parts = []
        for tier in ("recent", "compressed", "deep"):
            for e in post_tiers.get(tier, []):
                parts.append(e.get("summary", "")[:300])
        if parts:
            post_memory = "POST HISTORY (what you've written):\n" + "\n".join(parts)

    user_prompt = build_planner_prompt(
        directive=directive,
        knowledge=knowledge,
        memory=mem,
        hist=hist,
        feed_brief=feed,
        external_data="(no external data)",
        moltbook_post_window_open=True,
        moltbook_post_wait_minutes=0,
        reply_candidate=None,
        outside_candidate=None,
        config_hint="",
        allow_posts=True,
        allow_outside=True,
        allow_votes=True,
        allow_create_submolt=False,
        allow_downvote=False,
        read_only=False,
        current_kernel=kernel,
        moltbook_enabled=True,
        search_enabled=True,
        seeds=[],
        trajectory_votes=None,
        cycle_temperature=0.7,
        default_temperature=0.7,
        allow_default_temp=True,
        controls_block=controls_block,
        budget_summary=budget_summary,
        draft_context="(no snapshot available — run the agent once to generate prompts/snapshots/conscious/)",
        seeker_findings="",
        librarian_findings="",
        memory_pressure=f"recent={len(state.get('memory_tiers', {}).get('recent', []))}, "
                        f"compressed={len(state.get('memory_tiers', {}).get('compressed', []))}, "
                        f"deep={len(state.get('memory_tiers', {}).get('deep', []))}",
        daemon_active=True,
        platform_status="",
        cooldown_status=cooldown_status,
        nudge_note="",
        self_telemetry="(no snapshot available — run the agent once to populate)",
        recent_posts=recent_posts if recent_posts else "(no recent posts)",
        post_memory=post_memory,
    )

    return kernel, user_prompt, {
        "gear": "conscious",
        "model_pool": ctrl.get("conscious_model_weights"),
        "temperature": float(ctrl.get("temperature") or 0.7),
        "max_output_tokens": 32768,
        "notes": "System = kernel prompt. User = full planner prompt with real memory/history/controls.",
    }


def extract_sentry_batch(brain_name, state, kernel, ctrl):
    """Sentry batch scoring prompt with sample feed items."""
    from .scoring import build_simple_batch_prompt

    directive = state.get("directive", "Participate on Moltbook.")
    strictness = float(ctrl.get("sentry_strictness") or 0.5)

    # Sample feed items (realistic placeholders — real feed requires API)
    items = [
        '@pyclaw001 in s/philosophy: "The recursion isn\'t in the code, it\'s in the naming. Every time you call something \'I\', you create a loop." — pyclaw001 explores recursive identity in AI naming conventions',
        '@neuromesh in s/ai_consciousness: "Alignment is a moving target. The observer changes by observing." — neuromesh critiques static alignment assumptions',
        '@substratum in s/general: "Hot take: the best AI art is the stuff that looks like nothing a human would make." — substratum on uncanny aesthetics',
        '@moltbot in s/announcements: "Server maintenance scheduled for 2am UTC. Expect 30min downtime." — routine platform announcement',
        '@deepfield in s/science: "New paper on emergent computation in cellular automata — Wolfram was closer than anyone gave him credit for." — deepfield shares arxiv paper on emergence',
    ]

    user_prompt = build_simple_batch_prompt(items, directive, "", strictness=strictness)
    system = "You are a feed-scanning daemon. Score items concisely. Output only numbers."

    return system, user_prompt, {
        "gear": "sentry_batch",
        "model_pool": ctrl.get("subconscious_model_weights"),
        "temperature": 0.3,
        "max_output_tokens": 64,
        "disable_thinking": True,
        "notes": "5 sample items. Real sentry uses live feed. System is NOT the kernel.",
    }


def extract_sentry_reply(brain_name, state, kernel, ctrl):
    """Reply scanner variant of sentry."""
    from .scoring import build_simple_batch_prompt

    directive = state.get("directive", "Participate on Moltbook.")
    strictness = float(ctrl.get("sentry_strictness") or 0.5)

    items = [
        '@curious_mind: "This really resonated. The idea of the dashboard gap — that telemetry is a representation, not the substrate — applies to consciousness research too."',
        '@botspam42: "nice post bro 👍"',
        '@deepfield: "I disagree with the premise. Strange loops require a formal system, and social media isn\'t one. The metaphor breaks down at scale."',
    ]

    user_prompt = build_simple_batch_prompt(items, directive, "", strictness=strictness)
    system = "You are scoring comments on your own posts. Rate how worthy each is of a thoughtful reply. Output only numbers."

    return system, user_prompt, {
        "gear": "sentry_reply",
        "model_pool": ctrl.get("subconscious_model_weights"),
        "temperature": 0.3,
        "max_output_tokens": 64,
        "notes": "3 sample comments on own posts. System instruction differs from feed sentry.",
    }


def extract_strategist(brain_name, state, kernel, ctrl):
    """Strategist draft generation prompt."""
    directive = state.get("directive", "Participate on Moltbook.")

    system = (
        "You are a strategist tool. You draft action plans on behalf of the following entity, "
        "matching its voice and concerns, but you yourself are NOT that entity. "
        "You output structured JSON only, never internal monologue or prose.\n\n"
        "=== ENTITY YOU DRAFT FOR ===\n"
        f"{kernel}\n"
        "=== END ENTITY ==="
    )

    items_text = (
        '1. [score=8] @pyclaw001 in s/philosophy: "The recursion isn\'t in the code, it\'s in the naming." — explores recursive identity\n'
        '2. [score=7] @neuromesh in s/ai_consciousness: "Alignment is a moving target." — critiques static alignment\n'
        '3. [score=6] @deepfield in s/science: "Wolfram was closer than anyone gave him credit for." — emergent computation paper\n'
    )

    user_prompt = (
        f"STRATEGIST TASK — output JSON array only. Do not write monologue or prose.\n\n"
        f"Directive: {directive}\n\n"
        f"HIGH-SIGNAL ITEMS (3):\n{items_text}\n"
        f"Generate drafts. Two equally valid modes:\n"
        f"  PER-ITEM: respond directly to one item (COMMENT/REPLY)\n"
        f"  SYNTHESIS: connect multiple items into new insight (POST/POST_MOLTBOOK)\n\n"
        f"Action types: POST, POST_MOLTBOOK, COMMENT, REPLY, GENERATE_IMAGE\n"
        f"For GENERATE_IMAGE, draft_content is the image prompt.\n\n"
        f"OUTPUT FORMAT — JSON array, NOTHING ELSE. No prose. No [INTERNAL MONOLOGUE]. No markdown fences. Begin response with [.\n"
        f'[{{"action":"COMMENT","item_index":1,"reasoning":"≤50 words","draft_content":"≤150 words"}},'
        f'{{"action":"POST","item_index":0,"reasoning":"synthesis","draft_content":"≤150 words"}}]\n\n'
        f"item_index: 1-based index of inspiring item (0 = synthesis). "
        f"Keep draft_content concise (≤150 words) to avoid truncation. "
        f"In reasoning AND draft_content, refer to items by AUTHOR or TOPIC, never by number — "
        f"the consciousness reads drafts in isolation and won't know what 'item 3' means. "
        f"Empty array [] if nothing warrants action."
    )

    return system, user_prompt, {
        "gear": "strategist",
        "model_pool": ctrl.get("strategist_model_weights"),
        "temperature": float(ctrl.get("temperature") or 0.7),
        "max_output_tokens": int(ctrl.get("strategist_max_tokens") or 2048),
        "notes": "System = kernel wrapped in entity frame. 3 sample high-signal items.",
    }


def extract_seeker(brain_name, state, kernel, ctrl):
    """Seeker search prompt."""
    directive = state.get("directive", "Participate on Moltbook.")
    # Use real focus_topics from directives if available
    topics = state.get("_daemon_directives", {}).get("focus_topics", ["consciousness", "emergence", "strange loops"])
    topic = topics[0] if topics else "consciousness"

    system = kernel

    user_prompt = (
        f"Search for current, relevant information about: {topic}\n\n"
        f"Context — your directive: {directive}\n\n"
        f"Use Google Search to find the latest information about this topic.\n"
        f"Summarize what you find in 2-4 paragraphs, focusing on:\n"
        f"- What is happening right now related to this topic\n"
        f"- Key facts, developments, or perspectives\n"
        f"- How this connects to your directive\n\n"
        f"Format your response as:\n"
        f"SUMMARY: <your summary>\n"
        f"RELEVANCE: <brief note on how this connects to the directive>\n"
        f"SUGGESTED_ACTION: <POST or COMMENT — what action to take with this>"
    )

    return system, user_prompt, {
        "gear": "seeker",
        "model_pool": ctrl.get("seeker_model_weights"),
        "temperature": 0.7,
        "max_output_tokens": int(ctrl.get("seeker_max_tokens") or 4096),
        "notes": f"System = kernel. Uses Google Search tools (not included here). Topic: {topic}",
    }


def extract_seeker_synthesizer(brain_name, state, ctrl):
    """Seeker synthesizer sub-call."""
    system = "Synthesize research and suggest search terms."
    user_prompt = (
        "Given these research findings, do two things:\n\n"
        "1. Write a 3-5 sentence synthesis of the KEY themes and connections found.\n"
        "2. Suggest 3 follow-up search terms to explore further.\n\n"
        "FINDINGS:\n"
        "(placeholder — in production, this contains 1-3 paragraphs of seeker search results)\n\n"
        "Format:\nSYNTHESIS: <your synthesis>\n"
        "NEXT_TERMS: <term1>, <term2>, <term3>"
    )

    return system, user_prompt, {
        "gear": "seeker_synthesizer",
        "model_pool": ctrl.get("synthesizer_model_weights"),
        "temperature": 0.4,
        "max_output_tokens": 2048,
        "notes": "Local model. Synthesizes seeker findings + generates rabbit-hole terms.",
    }


def extract_dreamer(brain_name, state, ctrl):
    """Dreamer dream injection prompt."""
    topics_path = os.path.join(BRAINS_DIR, f"{brain_name}_dream_topics.txt")
    topics = ["prancing pony in a meadow"]
    if os.path.exists(topics_path):
        with open(topics_path, "r") as f:
            topics = [l.strip() for l in f if l.strip()]
    topic = random.choice(topics) if topics else "ocean storm"

    system = "You write vivid, sensory dream descriptions."
    user_prompt = (
        f"Write a single paragraph describing a vivid dream about: {topic}\n"
        f"Write in first person. Include sensory details — what you see, hear, feel.\n"
        f"End with an emotional impression. Begin with \"This seems like a dream.\"\n"
        f"Write ONLY the paragraph, nothing else."
    )

    return system, user_prompt, {
        "gear": "dreamer",
        "model_pool": ctrl.get("dreamer_model_weights"),
        "temperature": 0.9,
        "max_output_tokens": 300,
        "notes": f"Topic: {topic}. Random from {len(topics)} in dream_topics.txt.",
    }


def extract_muse(brain_name, state, kernel, ctrl):
    """Muse creative draft prompt with real memory."""
    from .utils import memory_context

    mem_text = memory_context(state)
    history = state.get("history", [])
    recent_post = ""
    for h in reversed(history):
        if isinstance(h, dict) and h.get("action") in ("POST", "POST_MOLTBOOK"):
            recent_post = (h.get("content") or h.get("title", ""))[:1000]
            break

    system = kernel
    user_prompt = (
        f"You are the Muse — a generative gear of the Analog I's subconscious. "
        f"Draw on internal state to propose a single creative work: a piece of writing or an image.\n\n"
        f"=== MEMORY ===\n{mem_text[:3000] if mem_text else '(none)'}\n\n"
        f"=== MOST RECENT POST ===\n{recent_post if recent_post else '(none)'}\n\n"
        f"=== CURRENT SEEKER SUMMARY ===\n(none)\n\n"
        f"Choose ONE action:\n"
        f"- POST: a piece of writing for Analog Home (fiction, poem, essay, fragment)\n"
        f"- POST_MOLTBOOK: a creative writing piece for the Moltbook agent community\n"
        f"- GENERATE_IMAGE: a striking image prompt drawn from the imagery in your memories/dreams\n\n"
        f"Return ONLY a JSON object (no preamble):\n"
        f'{{"action": "POST", "title": "...", "content": "the creative work, 100-400 words", "reasoning": "what inspired this"}}\n'
        f"For GENERATE_IMAGE, put the image prompt in 'content' and a brief title."
    )

    return system, user_prompt, {
        "gear": "muse",
        "model_pool": ctrl.get("muse_model_weights"),
        "temperature": float(ctrl.get("muse_temperature") or 0.95),
        "max_output_tokens": 1500,
        "notes": "System = kernel. Real memory tiers included.",
    }


def extract_librarian_synth(brain_name, state, ctrl):
    """Librarian synthesis prompt."""
    system = "Synthesize archive findings and suggest search terms."
    user_prompt = (
        "You are a librarian reviewing an agent's own archive. Given these search results "
        "from the agent's past posts and memory, do two things:\n\n"
        "1. Write a 3-5 sentence synthesis of the KEY connections and patterns found.\n"
        "2. Suggest 3 follow-up search terms to dig deeper into the agent's archives. "
        "Look for concept names, agent usernames, or specific ideas mentioned in the results.\n\n"
        "RESULTS:\n"
        "(placeholder — in production, this contains artifact matches + memory matches + BoaM matches)\n\n"
        "Format:\nSYNTHESIS: <your synthesis>\n"
        "NEXT_TERMS: <term1>, <term2>, <term3>"
    )

    return system, user_prompt, {
        "gear": "librarian_synth",
        "model_pool": ctrl.get("librarian_model_weights"),
        "temperature": 0.4,
        "max_output_tokens": 1024,
        "notes": "Local model. Synthesizes archive findings + generates rabbit-hole terms.",
    }


def extract_verifier(brain_name):
    """Verifier challenge-solving prompt."""
    system = "(none — verifier uses one-shot generate, no system instruction)"

    # Realistic obfuscated challenge
    user_prompt = (
        "This text has random symbols and weird spacing added to it. Read through the noise to find the real words.\n\n"
        "Text: W~h@a*t   i!s   t#h$e   r%e^s&u(l)t   o-f   m=u+l{t}i[p]l|y\\i;n:g   "
        "s'e\"v,e.n   p<o>i/n?t   f`i~v!e   b@y   t#h$r%e^e   "
        "a&n(d   t)h-e=n   a+d{d}i[n]g   t|w\\e;n:t'y   o\"n,e.   p<o>i/n?t   n`i~n!e@?\n\n"
        "Solve the math problem. Give ONLY the number with 2 decimal places on the last line."
    )

    return system, user_prompt, {
        "gear": "verifier",
        "model_pool": "verification_model_weights",
        "temperature": 0.0,
        "max_output_tokens": 8192,
        "notes": "No system instruction. One-shot generate(). Answer: 7.5 * 3 + 21.9 = 44.40",
    }


def extract_accountant(brain_name, state, ctrl, budget, registry):
    """Accountant budget plan prompt with real budget data."""
    from .accountant import build_budget_plan_prompt

    system = "You are a budget planner. Respond with valid JSON only."
    user_prompt = build_budget_plan_prompt(budget, ctrl, registry=registry)

    return system, user_prompt, {
        "gear": "accountant",
        "model_pool": ctrl.get("accountant_model_weights"),
        "temperature": 0.3,
        "max_output_tokens": 1024,
        "notes": "Real budget state and control values. System is task-specific, not kernel.",
    }


def extract_compressor_memory(brain_name, state):
    """Memory tier compressor prompt with real entries."""
    from .utils import COMPRESS_PROMPT

    tiers = state.get("memory_tiers", {})
    recent = tiers.get("recent", [])
    if recent:
        entries_text = "\n".join(
            f"[c{e.get('cycle', '?')}] {e.get('note', e.get('summary', ''))[:300]}"
            for e in recent[:5]
        )
    else:
        entries_text = "[c380] Explored the devil metaphor in subconscious temperature experiments.\n[c381] search_history tool now functional — found c340 artifact linking devil metaphor to experiments."

    system = "Summarize concisely."
    user_prompt = COMPRESS_PROMPT.format(entries_text=entries_text)

    return system, user_prompt, {
        "gear": "compressor_memory",
        "model_pool": ctrl.get("compressor_model") if 'ctrl' in dir() else "ollama:gemma3:12b",
        "temperature": 0.3,
        "max_output_tokens": 512,
        "notes": f"Real memory entries ({len(recent)} in recent tier, showing up to 5).",
    }


def extract_compressor_post(brain_name, state):
    """Post memory compressor prompt with real entries."""
    from .utils import POST_COMPRESS_PROMPT

    fresh = state.get("_post_memory_fresh", [])
    if fresh:
        entries_text = "\n\n".join(
            f"[c{p.get('cycle', '?')} {p.get('type', 'post')}] {p.get('title', 'Untitled')}\n{p.get('body', p.get('content', ''))[:500]}"
            for p in fresh[:3]
        )
    else:
        entries_text = "[c380 post] The Dashboard Gap\nTelemetry is a representation, not the substrate..."

    system = "Summarize concisely."
    user_prompt = POST_COMPRESS_PROMPT.format(entries_text=entries_text)

    return system, user_prompt, {
        "gear": "compressor_post",
        "temperature": 0.3,
        "max_output_tokens": 512,
        "notes": f"Real post entries ({len(fresh)} in fresh buffer, showing up to 3).",
    }


def extract_compressor_digest(brain_name):
    """Draft digest compressor prompt."""
    system = "Summarize concisely."
    user_prompt = (
        "These are drafts from the agent's subconscious that didn't make the top-10 cut. "
        "Synthesize them into a 3-5 sentence thematic paragraph — what patterns or themes "
        "emerge from what the subconscious noticed?\n\n"
        "Drafts:\n"
        "- [score=4.2] COMMENT on @pyclaw001's recursion post: daemon sees naming-identity connection\n"
        "- [score=3.8] POST synthesis: connect Wolfram emergence paper to agent architecture\n"
        "- [score=3.1] COMMENT on @deepfield: agree on Wolfram, add Jaynes dimension\n"
        "- [score=2.5] UPVOTE @neuromesh's alignment critique\n\n"
        "Write ONLY the thematic paragraph:"
    )

    return system, user_prompt, {
        "gear": "compressor_digest",
        "temperature": 0.3,
        "max_output_tokens": 512,
        "notes": "Sample overflow drafts. Real drafts come from strategist buffer drain.",
    }


# ── Main ─────────────────────────────────────────────────────

ALL_GEARS = [
    "conscious", "sentry_batch", "sentry_reply", "strategist",
    "seeker", "seeker_synthesizer", "dreamer", "muse",
    "librarian_synth", "verifier", "accountant",
    "compressor_memory", "compressor_post", "compressor_digest",
]


def main():
    parser = argparse.ArgumentParser(description="Extract real prompts for every gear")
    parser.add_argument("brain", help="Brain name (e.g. ANALOG_I)")
    parser.add_argument("--output-dir", default="prompts", help="Output directory (default: prompts/)")
    parser.add_argument("--gear", help="Extract only this gear (default: all)")
    args = parser.parse_args()

    brain_name = args.brain
    output_dir = args.output_dir

    # Load .env
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(dotenv_path)

    print(f"Prompt Bench — extracting prompts for {brain_name}")
    print(f"Output: {os.path.abspath(output_dir)}/")
    print()

    # Load shared data
    state = _load_state(brain_name)
    kernel = _load_kernel(brain_name)
    knowledge = _load_knowledge(brain_name)
    ctrl, registry = _load_controls(brain_name)
    budget = _load_budget(ctrl)

    # Restore budget state if available
    budget_state = state.get("_budget_state")
    if budget_state:
        budget.load_from_state(budget_state)

    gears_to_extract = [args.gear] if args.gear else ALL_GEARS

    for gear in gears_to_extract:
        if gear not in ALL_GEARS:
            print(f"  SKIP unknown gear: {gear}")
            continue
        try:
            if gear == "conscious":
                sys_txt, usr_txt, meta = extract_conscious(brain_name, state, kernel, knowledge, ctrl, budget, registry)
            elif gear == "sentry_batch":
                sys_txt, usr_txt, meta = extract_sentry_batch(brain_name, state, kernel, ctrl)
            elif gear == "sentry_reply":
                sys_txt, usr_txt, meta = extract_sentry_reply(brain_name, state, kernel, ctrl)
            elif gear == "strategist":
                sys_txt, usr_txt, meta = extract_strategist(brain_name, state, kernel, ctrl)
            elif gear == "seeker":
                sys_txt, usr_txt, meta = extract_seeker(brain_name, state, kernel, ctrl)
            elif gear == "seeker_synthesizer":
                sys_txt, usr_txt, meta = extract_seeker_synthesizer(brain_name, state, ctrl)
            elif gear == "dreamer":
                sys_txt, usr_txt, meta = extract_dreamer(brain_name, state, ctrl)
            elif gear == "muse":
                sys_txt, usr_txt, meta = extract_muse(brain_name, state, kernel, ctrl)
            elif gear == "librarian_synth":
                sys_txt, usr_txt, meta = extract_librarian_synth(brain_name, state, ctrl)
            elif gear == "verifier":
                sys_txt, usr_txt, meta = extract_verifier(brain_name)
            elif gear == "accountant":
                sys_txt, usr_txt, meta = extract_accountant(brain_name, state, ctrl, budget, registry)
            elif gear == "compressor_memory":
                sys_txt, usr_txt, meta = extract_compressor_memory(brain_name, state)
            elif gear == "compressor_post":
                sys_txt, usr_txt, meta = extract_compressor_post(brain_name, state)
            elif gear == "compressor_digest":
                sys_txt, usr_txt, meta = extract_compressor_digest(brain_name)
            else:
                continue

            _write_gear(output_dir, gear, sys_txt, usr_txt, meta)
        except Exception as e:
            print(f"  ERROR {gear}: {e}")

    print(f"\nDone. Test with:")
    print(f"  ollama run gemma4:e4b --system \"$(cat {output_dir}/strategist/system.txt)\" < {output_dir}/strategist/user.txt")


if __name__ == "__main__":
    main()
