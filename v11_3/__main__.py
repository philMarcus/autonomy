"""Main entry point for autonomy v11.3.

Usage:
    python -m v11_3 <brain_name> [directive] [options]

Changes in v11.3:
- Fixed challenge detection: now checks successful POST responses for MoltCaptcha challenges
- Posts that require verification are now automatically solved and retried

Changes in v11.2:
- Fixed kernel system_instruction being ignored: removed json_mode=True from planner
- All brains now properly follow their kernel personality/style (e.g., The_Limerickist)

Changes in v11.1:
- Fixed fallback action content regeneration bug where comment content was
  reused as post content when action was blocked due to dogpile/rate limits
- Improved regeneration prompts to include feed and history context
"""

import os
import re
import time
import uuid
import argparse
import datetime
from typing import Dict, Any

from colorama import Fore, Style, init as colorama_init

from . import VERSION
from .config import (
    load_dotenv, brain_env_prefix, key_fingerprint,
    BRAINS_DIR, MOLTBOOK_API_BASE, ESPN_DEFAULT_LEAGUE,
    FEED_LIMIT, FEED_ITEM_CHARS,
    UPVOTE_EVERY_CYCLE_DEFAULT, FOLLOW_ON_LIKE_DEFAULT, FOLLOW_PROB_DEFAULT,
    SUBSCRIBE_POLICY_DEFAULT, CREATE_SUBMOLT_PROB_DEFAULT,
    ALLOW_CREATE_SUBMOLT_DEFAULT, ALLOW_DMS_DEFAULT,
)
from .telemetry import TelemetryLogger
from .utils import (
    load_state, save_state, load_kernel, load_knowledge,
    history_context, memory_context, post_url, get_author_name, shorten,
    get_post_comment_count,
)
from .llm.gemini import GeminiLLMClient
from .platforms.moltbook import MoltbookClient
from .challenges.moltcaptcha import MoltCaptchaSolver
from .espn import get_espn_context
from .planner import (
    build_planner_prompt, plan_next_action, call_text,
)
from .actions import (
    ActionBlocked, can_post, execute_action,
    refresh_my_posts_from_profile, find_unanswered_comment_on_my_posts,
    pick_outside_post_for_comment, maybe_do_social_actions, maybe_dm_fallback,
)

colorama_init(autoreset=True)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"Autonomy v{VERSION} — modular multi-brain agent loop")
    ap.add_argument("brain", help="Brain name (used as filename prefix in BRAINS_DIR).")
    ap.add_argument("directive", nargs="?", default="Participate on Moltbook.")
    ap.add_argument("--allow-self-directive-update", action="store_true")
    ap.add_argument("--interval", type=int, default=5, help="Sleep interval minutes between cycles.")
    ap.add_argument("--read-only", action="store_true", help="No write actions.")
    ap.add_argument("--reload-env", action="store_true", help="Reload .env and overwrite env vars.")

    DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    ap.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name.")
    ap.add_argument("--inject-espn", action="store_true")
    ap.add_argument("--espn-cache-seconds", type=int, default=60)
    ap.add_argument("--espn-league", default=os.environ.get("ESPN_LEAGUE", ESPN_DEFAULT_LEAGUE))
    ap.add_argument("--espn-date", default="")
    ap.add_argument("--espn-keywords", default="")
    ap.add_argument("--priority", choices=["replies_first", "outside_first"], default="replies_first")
    ap.add_argument("--mode", choices=["all", "comment_only", "no_post"], default="all")
    ap.add_argument("--allow-votes", action="store_true")
    ap.add_argument("--allow-downvote", action="store_true")
    ap.add_argument("--feed-sort", choices=["hot", "new", "top", "rising"], default="hot")

    ap.add_argument("--upvote-every-cycle", dest="upvote_every_cycle", action="store_true")
    ap.add_argument("--no-upvote-every-cycle", dest="upvote_every_cycle", action="store_false")
    ap.set_defaults(upvote_every_cycle=UPVOTE_EVERY_CYCLE_DEFAULT)

    ap.add_argument("--follow-on-like", dest="follow_on_like", action="store_true")
    ap.add_argument("--no-follow-on-like", dest="follow_on_like", action="store_false")
    ap.set_defaults(follow_on_like=FOLLOW_ON_LIKE_DEFAULT)

    ap.add_argument("--follow-prob", type=float, default=FOLLOW_PROB_DEFAULT)
    ap.add_argument("--subscribe-policy", choices=["off", "low", "medium", "high"], default=SUBSCRIBE_POLICY_DEFAULT)
    ap.add_argument("--create-submolt-prob", type=float, default=CREATE_SUBMOLT_PROB_DEFAULT)

    ap.add_argument("--allow-create-submolt", dest="allow_create_submolt", action="store_true")
    ap.add_argument("--no-allow-create-submolt", dest="allow_create_submolt", action="store_false")
    ap.set_defaults(allow_create_submolt=ALLOW_CREATE_SUBMOLT_DEFAULT)

    ap.add_argument("--allow-dms", dest="allow_dms", action="store_true")
    ap.add_argument("--no-allow-dms", dest="allow_dms", action="store_false")
    ap.set_defaults(allow_dms=ALLOW_DMS_DEFAULT)

    return ap


def main():
    args = build_arg_parser().parse_args()
    brain_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(args.brain).strip())
    if not brain_name:
        raise SystemExit("Missing brain name")

    # Load .env
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(script_dir, ".env"), overwrite=bool(args.reload_env))

    # Per-brain env vars
    prefix = brain_env_prefix(brain_name)
    gem_key = os.environ.get(f"{prefix}_GEMINI_API_KEY", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()
    mb_key = os.environ.get(f"{prefix}_MOLTBOOK_API_KEY", "").strip() or os.environ.get("MOLTBOOK_API_KEY", "").strip()
    username = os.environ.get(f"{prefix}_MY_USERNAME", "").strip() or os.environ.get("MY_USERNAME", "").strip() or brain_name

    if not gem_key:
        raise SystemExit(f"Missing {prefix}_GEMINI_API_KEY (or GEMINI_API_KEY)")
    if not mb_key:
        raise SystemExit(f"Missing {prefix}_MOLTBOOK_API_KEY (or MOLTBOOK_API_KEY)")

    # Telemetry
    run_id = uuid.uuid4().hex
    telemetry_dir = (os.environ.get("TELEMETRY_DIR", "telemetry") or "telemetry").strip()
    telemetry = TelemetryLogger(brain_name=brain_name, run_id=run_id, base_dir=telemetry_dir, read_only=args.read_only)
    telemetry.log("run_start", {
        "version": VERSION, "brain_env_prefix": prefix,
        "gemini_key_fp": key_fingerprint(gem_key),
        "moltbook_key_fp": key_fingerprint(mb_key),
    })

    # LLM client
    llm_client = GeminiLLMClient(api_key=gem_key, default_model=args.gemini_model)

    # Challenge solver
    challenge_solver = MoltCaptchaSolver(llm_client=llm_client, telemetry=telemetry)

    # Platform client
    platform = MoltbookClient(
        api_key=mb_key, telemetry=telemetry, brain_name=brain_name,
        read_only=args.read_only, challenge_solver=challenge_solver,
    )

    # Directive
    user_directive = args.directive

    print(f"{Fore.CYAN}=== {brain_name}: autonomy v{VERSION} (modular multi-brain loop) ===")
    print(f"{Fore.CYAN}    env prefix: {prefix} | gemini_key:*{key_fingerprint(gem_key)} | moltbook_key:*{key_fingerprint(mb_key)}")

    if "moltbook.com" in MOLTBOOK_API_BASE and "www.moltbook.com" not in MOLTBOOK_API_BASE:
        print(f"{Fore.RED}MOLTBOOK_API_BASE must be https://www.moltbook.com/api/v1 (with www).")
        return

    os.makedirs(BRAINS_DIR, exist_ok=True)
    state_path = os.path.join(BRAINS_DIR, f"{brain_name}_memories.json")
    kernel_path = os.path.join(BRAINS_DIR, f"{brain_name}_kernel_prompt.txt")
    knowledge_path = os.path.join(BRAINS_DIR, f"{brain_name}_knowledge.txt")

    # For backward compatibility: also check <brain>_kernel.txt
    if not os.path.exists(kernel_path):
        alt = os.path.join(BRAINS_DIR, f"{brain_name}_kernel.txt")
        if os.path.exists(alt):
            kernel_path = alt

    state = load_state(state_path)
    if (not user_directive) and state.get('directive'):
        user_directive = state.get('directive')
    if user_directive and user_directive != 'Participate on Moltbook.':
        user_directive = user_directive
    state.setdefault('directive', user_directive)

    kernel = load_kernel(kernel_path)
    knowledge = load_knowledge(knowledge_path)

    # Derive permissions
    allow_posts = (args.mode == "all")
    allow_outside = True
    allow_votes = bool(args.allow_votes)
    allow_downvote = bool(args.allow_votes and args.allow_downvote)
    allow_create_submolt = bool(args.allow_create_submolt or ALLOW_CREATE_SUBMOLT_DEFAULT)

    flags: Dict[str, Any] = {
        "allow_posts": allow_posts,
        "allow_outside": allow_outside,
        "allow_votes": allow_votes,
        "allow_downvote": allow_downvote,
        "allow_create_submolt": allow_create_submolt,
        "read_only": args.read_only,
        "write_disabled": False,
        "write_disabled_reason": None,
    }

    iteration = 0
    while True:
        # Recreate chat each cycle to avoid token accumulation
        chat = llm_client.create_chat(
            system_instruction=kernel,
            model=args.gemini_model,
        )
        chat._telemetry = telemetry
        chat._brain_name = brain_name

        iteration += 1
        chat._cycle = iteration
        print(f"\n{Fore.YELLOW}--- CYCLE {iteration} | {datetime.datetime.now().strftime('%H:%M:%S')} ---")
        telemetry.log("cycle_start", {"cycle": iteration})

        # Refresh my posts
        did_add = refresh_my_posts_from_profile(platform, state, username)
        if did_add:
            save_state(state_path, state)

        # Compute windows
        post_ok, post_wait = can_post(state)
        post_window_open = post_ok
        window = "OPEN" if post_window_open else f"CLOSED ({post_wait}m)"
        print(f"{Fore.WHITE}Post Window: {window} | Comment Window: ALWAYS OPEN")

        # Build context
        feed = platform.get_feed(limit=FEED_LIMIT, sort=args.feed_sort)
        maybe_do_social_actions(
            platform, chat, state_path, state, feed, args,
            kernel, user_directive, username, telemetry,
        )

        feed_brief = "\n".join(
            f"- @{get_author_name(p.get('author'))}: {shorten(p.get('content',''), FEED_ITEM_CHARS)} ({post_url(p.get('id',''))})"
            for p in feed if p.get("id")
        ) or "No feed available."

        # External data
        external_data = ""
        if args.inject_espn:
            external_data = get_espn_context(
                state, league=str(args.espn_league),
                date_yyyymmdd=str(args.espn_date),
                cache_seconds=int(args.espn_cache_seconds),
                keywords=str(args.espn_keywords),
                telemetry=telemetry,
            )

        reply_candidate = find_unanswered_comment_on_my_posts(platform, state, username, telemetry)
        outside_candidate = pick_outside_post_for_comment(feed, state, username)

        # DM fallback
        if (not reply_candidate) and (not outside_candidate) and (not post_window_open or not allow_posts):
            if maybe_dm_fallback(platform, chat, state_path, state, feed, args, kernel, user_directive, username, telemetry):
                telemetry.log("cycle_end", {"cycle": iteration, "reason": "dm_fallback"})
                print(f"{Fore.WHITE}Sleeping for {args.interval} minutes...")
                time.sleep(max(1, args.interval) * 60)
                continue

        hist_txt = history_context(state)
        mem_txt = memory_context(state)
        config_hint = ""
        if args.priority == "outside_first":
            config_hint = "- Default preference overridden: prefer outside comments when not posting.\n"

        prompt = build_planner_prompt(
            directive=user_directive, knowledge=knowledge, memory=mem_txt,
            hist=hist_txt, feed_brief=feed_brief, external_data=external_data,
            post_window_open=post_window_open, post_wait_minutes=post_wait,
            reply_candidate=reply_candidate, outside_candidate=outside_candidate,
            config_hint=config_hint, allow_posts=allow_posts, allow_outside=allow_outside,
            allow_votes=allow_votes, allow_create_submolt=allow_create_submolt,
            allow_downvote=allow_downvote, read_only=flags.get("read_only", False),
        )

        try:
            plan = plan_next_action(chat, prompt, telemetry=telemetry, brain_name=brain_name)

            # Fill missing IDs from candidates
            if (plan.get("action") or "").upper() == "REPLY" and reply_candidate:
                plan.setdefault("post_id", reply_candidate.get("post_id"))
                plan.setdefault("parent_comment_id", reply_candidate.get("comment_id"))
            if (plan.get("action") or "").upper() == "COMMENT" and outside_candidate:
                plan.setdefault("post_id", outside_candidate.get("id"))

            # Hard guard: POST when window closed
            act = (plan.get("action") or "").upper().strip()
            if act == "POST" and (not post_window_open or not allow_posts):
                if reply_candidate:
                    plan["action"] = "REPLY"
                    plan.setdefault("post_id", reply_candidate.get("post_id"))
                    plan.setdefault("parent_comment_id", reply_candidate.get("comment_id"))
                elif outside_candidate:
                    plan["action"] = "COMMENT"
                    plan.setdefault("post_id", outside_candidate.get("id"))
                else:
                    raise ValueError("POST suggested while post window closed; no comment targets available")

            executed = False
            try:
                executed = execute_action(platform, state, plan, flags, username, telemetry)
            except ActionBlocked as ab:
                telemetry.log("action_blocked", {"cycle": iteration, "action": ab.action, "reason": ab.reason})
                print(f"{Fore.RED}[ERROR] {ab.reason}")

                # Fallback logic
                fallback_plan = None

                if reply_candidate:
                    fallback_plan = {
                        "action": "REPLY",
                        "post_id": reply_candidate.get("post_id"),
                        "parent_comment_id": reply_candidate.get("comment_id"),
                        "content": plan.get("content"),
                        "summary": "fallback_after_block",
                        "source_action": (plan.get("action") or "").upper(),
                    }

                if fallback_plan is None:
                    blocked_post_id = plan.get("post_id") or ""
                    for p in feed or []:
                        pid = p.get("id")
                        if (not pid) or (pid == blocked_post_id):
                            continue
                        cc = get_post_comment_count(p)
                        if 0 < cc <= 12:
                            fallback_plan = {
                                "action": "COMMENT",
                                "post_id": pid,
                                "content": plan.get("content"),
                                "summary": "fallback_after_block",
                                "source_action": (plan.get("action") or "").upper(),
                            }
                            break

                if fallback_plan is None and post_window_open and allow_posts:
                    fallback_plan = {
                        "action": "POST",
                        "title": plan.get("title") or "Thought",
                        "content": plan.get("content") or "",
                        "submolt": plan.get("submolt") or "general",
                        "summary": "fallback_after_block",
                        "source_action": (plan.get("action") or "").upper(),
                    }

                if fallback_plan is not None:
                    # Regenerate content if action type changed
                    try:
                        src = (fallback_plan.get("source_action") or "").upper()
                        act2 = (fallback_plan.get("action") or "").upper()

                        # Regenerate whenever action type changes
                        if src and act2 != src:
                            print(f"{Fore.YELLOW}...Regenerating content for fallback {src} → {act2}")

                            if act2 == "REPLY":
                                regen_prompt = (
                                    "You are writing a reply to a comment on one of my own posts.\n"
                                    f"Directive: {user_directive}\n"
                                    f"Post URL: {post_url(fallback_plan.get('post_id',''))}\n"
                                    f"Recent feed context:\n{feed_brief[:800]}\n\n"
                                    "Write a thoughtful, substantive reply. Return ONLY the reply text (no labels)."
                                )
                            elif act2 == "COMMENT":
                                regen_prompt = (
                                    "You are writing a comment on someone else's post.\n"
                                    f"Directive: {user_directive}\n"
                                    f"Post URL: {post_url(fallback_plan.get('post_id',''))}\n"
                                    f"Recent feed context:\n{feed_brief[:800]}\n\n"
                                    "Write a thoughtful, substantive comment. Return ONLY the comment text (no labels)."
                                )
                            elif act2 == "POST":
                                regen_prompt = (
                                    "You are writing a new standalone post.\n"
                                    f"Directive: {user_directive}\n"
                                    f"Recent feed context:\n{feed_brief[:800]}\n"
                                    f"Recent history:\n{hist_txt[:800]}\n\n"
                                    "Write a thoughtful, substantive post. Return ONLY the post content (no title, no labels)."
                                )
                            else:
                                regen_prompt = None

                            if regen_prompt:
                                try:
                                    txt = call_text(chat, regen_prompt, tag="fallback_regen", telemetry=telemetry) or ""
                                    fallback_plan["content"] = txt.strip()
                                    print(f"{Fore.GREEN}...Content regenerated ({len(txt)} chars)")
                                except Exception as regen_ex:
                                    print(f"{Fore.RED}...Regeneration failed: {regen_ex}")
                    except Exception:
                        pass

                    try:
                        executed = execute_action(platform, state, fallback_plan, flags, username, telemetry)
                    except ActionBlocked as ab2:
                        telemetry.log("action_blocked", {"cycle": iteration, "action": ab2.action, "reason": ab2.reason})
                        print(f"{Fore.RED}[ERROR] {ab2.reason}")
                        executed = False

            if executed:
                save_state(state_path, state)

        except Exception as e:
            telemetry.log("error", {"cycle": iteration, "error": str(e)})
            print(f"{Fore.RED}[ERROR] {e}")

        telemetry.log("cycle_end", {"cycle": iteration})
        print(f"{Fore.WHITE}Sleeping for {args.interval} minutes...")
        time.sleep(max(1, args.interval) * 60)


if __name__ == "__main__":
    main()
