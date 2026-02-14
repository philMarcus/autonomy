"""Main entry point for autonomy v14.0.

Usage:
    python -m v14_0 <brain_name> [directive] [options]

Changes in v14.0 (from v13.0):
- Artifact publish retry queue: failed publishes are saved to disk and retried next cycle
- Search queries included in published artifacts and shown on Analog Home frontend
- Trajectory change reason stored in API and shown on frontend
- Fly.io: increased VM memory (512MB), disabled auto-suspend, faster health checks
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
from .store import LocalFileStore
from .utils import (
    load_kernel, load_knowledge,
    history_context, memory_context, post_url, get_author_name, shorten,
    get_post_comment_count,
    update_kernel_file,
)
from .llm.gemini import GeminiLLMClient
from .platforms.moltbook import MoltbookClient
from .challenges.math_verification import MathVerificationSolver
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
    ap.add_argument("--allow-kernel-update", action="store_true", help="Allow the planner to rewrite the kernel prompt.")
    ap.add_argument("--no-kernel-disk-write", action="store_true", help="Kernel updates stay in-memory only (not written to disk).")
    ap.add_argument("--enable-moltbook", dest="moltbook_enabled",
                    action="store_true",
                    help="Enable Moltbook API calls. Without this flag, output goes to local log + Analog Home only.")
    ap.add_argument("--interval", type=int, default=5, help="Sleep interval minutes between cycles.")
    ap.add_argument("--post-interval", type=int, default=30, help="Minutes between posts (default 30).")
    ap.add_argument("--reset-post-window", action="store_true", help="Clear the post cooldown timer on startup (allows immediate posting).")
    ap.add_argument("--read-only", action="store_true", help="No write actions.")
    ap.add_argument("--reload-env", action="store_true", help="Reload .env and overwrite env vars.")

    DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    ap.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name.")
    ap.add_argument("--temperature", type=float, default=0.7, help="LLM temperature for planner chat (default 0.7).")
    ap.add_argument("--enable-search", action="store_true",
                    help="Enable Google Search grounding on the planner LLM call.")
    ap.add_argument("--enable-default-temp", action="store_true",
                    help="Allow the agent to set default_temperature via trajectory updates.")
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
    if not mb_key and args.moltbook_enabled:
        raise SystemExit(f"Missing {prefix}_MOLTBOOK_API_KEY (or MOLTBOOK_API_KEY)")

    # Telemetry
    run_id = uuid.uuid4().hex
    telemetry_dir = (os.environ.get("TELEMETRY_DIR", "telemetry") or "telemetry").strip()
    telemetry = TelemetryLogger(brain_name=brain_name, run_id=run_id, base_dir=telemetry_dir, read_only=args.read_only)
    telemetry.log("run_start", {
        "version": VERSION, "brain_env_prefix": prefix,
        "gemini_key_fp": key_fingerprint(gem_key),
        "moltbook_disabled": not args.moltbook_enabled,
        "model": args.gemini_model,
        "temperature": args.temperature,
        "search_enabled": bool(args.enable_search),
        "allow_kernel_update": bool(args.allow_kernel_update),
    })

    # LLM client
    llm_client = GeminiLLMClient(api_key=gem_key, default_model=args.gemini_model)

    # Platform client (None when Moltbook disabled)
    platform = None
    challenge_solver = None
    if args.moltbook_enabled:
        challenge_solver = MathVerificationSolver(llm_client=llm_client, telemetry=telemetry)
        platform = MoltbookClient(
            api_key=mb_key, telemetry=telemetry, brain_name=brain_name,
            read_only=args.read_only, challenge_solver=challenge_solver,
        )

    output_destination = "local" if not args.moltbook_enabled else "moltbook"

    # Directive
    user_directive = args.directive

    print(f"{Fore.CYAN}=== {brain_name}: autonomy v{VERSION} (modular multi-brain loop) ===")
    print(f"{Fore.CYAN}    env prefix: {prefix} | gemini_key:*{key_fingerprint(gem_key)}")
    if not args.moltbook_enabled:
        print(f"{Fore.MAGENTA}    [MOLTBOOK DISABLED] Output → local log + Analog Home")
    else:
        print(f"{Fore.CYAN}    moltbook_key:*{key_fingerprint(mb_key)}")
    if args.post_interval != 30:
        print(f"{Fore.CYAN}    Post interval: {args.post_interval} min (default 30)")
    if args.no_kernel_disk_write:
        print(f"{Fore.CYAN}    Kernel disk write: DISABLED (in-memory only)")

    if args.moltbook_enabled and "moltbook.com" in MOLTBOOK_API_BASE and "www.moltbook.com" not in MOLTBOOK_API_BASE:
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

    analog_home_url = os.environ.get(f"{prefix}_ANALOG_HOME_API_URL", "").strip() or os.environ.get("ANALOG_HOME_API_URL", "").strip()
    store = LocalFileStore(state_path, analog_home_url=analog_home_url)
    state = store.load_state()

    # Reset post window if requested
    if args.reset_post_window:
        state["next_post_time"] = 0.0
        store.save_state(state)
        print(f"{Fore.GREEN}Post window reset — first cycle can post immediately.")

    if (not user_directive) and state.get('directive'):
        user_directive = state.get('directive')
    if user_directive and user_directive != 'Participate on Moltbook.':
        user_directive = user_directive
    state.setdefault('directive', user_directive)

    kernel = load_kernel(kernel_path)
    telemetry.log_kernel_snapshot(kernel, reason="startup", source="startup")
    knowledge = load_knowledge(knowledge_path)

    # Derive permissions
    allow_posts = (args.mode == "all")
    allow_outside = True
    allow_votes = bool(args.allow_votes)
    allow_downvote = bool(args.allow_votes and args.allow_downvote)
    allow_create_submolt = bool(args.allow_create_submolt or ALLOW_CREATE_SUBMOLT_DEFAULT)

    post_cooldown_seconds = args.post_interval * 60

    flags: Dict[str, Any] = {
        "allow_posts": allow_posts,
        "allow_outside": allow_outside,
        "allow_votes": allow_votes,
        "allow_downvote": allow_downvote,
        "allow_create_submolt": allow_create_submolt,
        "read_only": args.read_only,
        "moltbook_disabled": not args.moltbook_enabled,
        "write_disabled": False,
        "write_disabled_reason": None,
        "post_cooldown_seconds": post_cooldown_seconds,
    }

    # Build tools list for planner chat (Google Search grounding)
    search_tools = None
    if args.enable_search:
        from google.genai import types as genai_types
        search_tools = [genai_types.Tool(google_search=genai_types.GoogleSearch())]
        print(f"{Fore.GREEN}Google Search grounding enabled.")

    iteration = 0
    while True:
        iteration += 1

        # --- Analog Home controls (only when API URL is configured) ---
        analog_controls = {}
        analog_seeds = []
        analog_trajectory = None
        cycle_temperature = args.temperature
        if analog_home_url:
            analog_controls = store.read_controls()
            if analog_controls:
                cycle_temperature = analog_controls.get("temperature", args.temperature)
                analog_seeds = analog_controls.get("seeds", [])
                seed_ids = analog_controls.get("seed_ids", [])
                if seed_ids:
                    store.consume_seeds(seed_ids)
                analog_trajectory = {
                    "vote_1": analog_controls.get("vote_1", 0),
                    "vote_2": analog_controls.get("vote_2", 0),
                    "vote_3": analog_controls.get("vote_3", 0),
                    "vote_label_1": analog_controls.get("vote_label_1", "emergence"),
                    "vote_label_2": analog_controls.get("vote_label_2", "entropy"),
                    "vote_label_3": analog_controls.get("vote_label_3", "self"),
                }

        # Recreate chat each cycle to avoid token accumulation
        chat = llm_client.create_chat(
            system_instruction=kernel,
            model=args.gemini_model,
            max_output_tokens=16384,
            temperature=cycle_temperature,
            tools=search_tools,
        )
        chat._telemetry = telemetry
        chat._brain_name = brain_name

        chat._cycle = iteration
        flags["cycle"] = iteration
        telemetry.current_cycle = iteration
        print(f"\n{Fore.YELLOW}--- CYCLE {iteration} | {datetime.datetime.now().strftime('%H:%M:%S')} ---")
        if analog_controls:
            print(f"{Fore.GREEN}Analog Home: temp={cycle_temperature}, seeds={len(analog_seeds)}, "
                  f"votes={analog_controls.get('vote_1',0)}/{analog_controls.get('vote_2',0)}/{analog_controls.get('vote_3',0)}")
        telemetry.log("cycle_start", {
            "cycle": iteration, "temperature": cycle_temperature, "model": args.gemini_model,
            **({"analog_controls": analog_controls} if analog_controls else {}),
        })

        # --- Moltbook context (skipped entirely when disabled) ---
        feed = []
        feed_brief = "No feed available (Moltbook disabled)."
        reply_candidate = None
        outside_candidate = None

        if args.moltbook_enabled:
            did_add = refresh_my_posts_from_profile(platform, state, username)
            if did_add:
                store.save_state(state)

            feed = platform.get_feed(limit=FEED_LIMIT, sort=args.feed_sort)
            maybe_do_social_actions(
                platform, chat, store, state, feed, args,
                kernel, user_directive, username, telemetry,
            )

            feed_brief = "\n".join(
                f"- @{get_author_name(p.get('author'))}: {shorten(p.get('content',''), FEED_ITEM_CHARS)} ({post_url(p.get('id',''))})"
                for p in feed if p.get("id")
            ) or "No feed available."

            telemetry.log("feed_context", {"text_length": len(feed_brief), "brief": feed_brief[:2000]})

            reply_candidate = find_unanswered_comment_on_my_posts(platform, state, username, telemetry)
            outside_candidate = pick_outside_post_for_comment(feed, state, username)

        # Compute windows
        post_ok, post_wait = can_post(state)
        post_window_open = post_ok
        window = "OPEN" if post_window_open else f"CLOSED ({post_wait}m)"
        print(f"{Fore.WHITE}Post Window: {window}")

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

        # DM fallback (only when Moltbook is active)
        if args.moltbook_enabled:
            if (not reply_candidate) and (not outside_candidate) and (not post_window_open or not allow_posts):
                if maybe_dm_fallback(platform, chat, store, state, feed, args, kernel, user_directive, username, telemetry):
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
            current_kernel=kernel if args.allow_kernel_update else "",
            output_destination=output_destination,
            search_enabled=bool(args.enable_search),
            seeds=analog_seeds,
            trajectory_votes=analog_trajectory,
            cycle_temperature=cycle_temperature if analog_home_url else None,
            default_temperature=args.temperature,
            allow_default_temp=bool(args.enable_default_temp),
        )

        try:
            plan = plan_next_action(chat, prompt, telemetry=telemetry, brain_name=brain_name)

            # Display any non-JSON LLM output (reasoning, preamble, etc.)
            preamble = plan.pop("_preamble", "")
            if preamble:
                print(f"{Fore.CYAN}--- REASONING ---")
                print(f"{Fore.WHITE}{preamble}")
                print(f"{Fore.CYAN}-----------------{Style.RESET_ALL}")

            # Log Google Search grounding metadata if available
            grounding = getattr(chat, "_last_grounding_metadata", None)
            if grounding:
                search_queries = getattr(grounding, "web_search_queries", None) or []
                chunks = getattr(grounding, "grounding_chunks", None) or []
                source_urls = []
                for chunk in chunks[:10]:
                    web = getattr(chunk, "web", None)
                    if web:
                        source_urls.append({"uri": getattr(web, "uri", ""), "title": getattr(web, "title", "")})
                if search_queries or source_urls:
                    print(f"{Fore.GREEN}--- SEARCH GROUNDING ---")
                    for q in search_queries:
                        print(f"{Fore.WHITE}  Query: {q}")
                    for src in source_urls[:5]:
                        print(f"{Fore.WHITE}  Source: {src.get('title', '?')} — {src.get('uri', '?')}")
                    print(f"{Fore.GREEN}------------------------{Style.RESET_ALL}")
                    telemetry.log("grounding_metadata", {
                        "search_queries": list(search_queries),
                        "source_count": len(chunks),
                        "sources": source_urls[:10],
                    })

            # Log full planner decision to telemetry
            telemetry.log_planner_decision(
                plan=plan, preamble=preamble,
                model=args.gemini_model, temperature=cycle_temperature,
            )

            # Check for kernel update request (only if --allow-kernel-update)
            if plan.get("update_kernel") and args.allow_kernel_update:
                new_kernel = plan.get("new_kernel", "").strip()
                reason = plan.get("kernel_reason", "no reason given")

                try:
                    print(f"{Fore.MAGENTA}[KERNEL UPDATE REQUESTED]")
                    print(f"{Fore.YELLOW}Reason: {reason}")
                    print(f"{Fore.YELLOW}New kernel length: {len(new_kernel)} chars")
                except:
                    pass

                if not flags.get("read_only"):
                    if args.no_kernel_disk_write:
                        telemetry.log_kernel_snapshot(new_kernel, reason=reason, source="memory_only")
                        kernel = new_kernel
                        try:
                            print(f"{Fore.CYAN}[NO-DISK] Kernel updated in-memory only (--no-kernel-disk-write)")
                        except:
                            pass
                        telemetry.log("kernel_update_memory_only", {
                            "cycle": iteration,
                            "reason": reason,
                            "new_length": len(new_kernel),
                        })
                    else:
                        result = update_kernel_file(kernel_path, new_kernel, telemetry=telemetry, reason=reason)

                        if result["success"]:
                            kernel = new_kernel  # Update in-memory kernel for next cycle
                            try:
                                print(f"{Fore.GREEN}>> KERNEL UPDATED: Will take effect next cycle")
                                if result["backup_created"]:
                                    backup_path = kernel_path.replace("_kernel_prompt.txt", "_kernel_prompt.backup.txt")
                                    print(f"{Fore.GREEN}   Backup created: {backup_path}")
                            except:
                                pass
                            telemetry.log("kernel_update_executed", {
                                "cycle": iteration,
                                "reason": reason,
                                "new_length": len(new_kernel),
                                "backup_created": result["backup_created"],
                            })
                        else:
                            try:
                                print(f"{Fore.RED}[ERROR] Kernel update failed: {result['error']}")
                            except:
                                pass
                            telemetry.log("kernel_update_rejected", {
                                "cycle": iteration,
                                "reason": reason,
                                "error": result["error"],
                                "attempted_length": len(new_kernel),
                            })
                else:
                    try:
                        print(f"{Fore.YELLOW}[SAFE] Skipping kernel update due to --read-only mode")
                    except:
                        pass
                    telemetry.log("kernel_update_skipped", {
                        "cycle": iteration,
                        "reason": "read_only_mode",
                        "requested_reason": reason,
                    })

            # Check for trajectory update request (parallel to kernel update)
            if plan.get("set_trajectory") and analog_home_url:
                t_label_1 = (plan.get("trajectory_label_1") or "").strip()[:40]
                t_label_2 = (plan.get("trajectory_label_2") or "").strip()[:40]
                t_label_3 = (plan.get("trajectory_label_3") or "").strip()[:40]
                t_reason = plan.get("trajectory_reason", "no reason given")
                t_default_temp = None
                if args.enable_default_temp and plan.get("default_temperature") is not None:
                    try:
                        t_default_temp = float(plan["default_temperature"])
                    except (ValueError, TypeError):
                        t_default_temp = None
                if t_label_1 and t_label_2 and t_label_3:
                    print(f"{Fore.MAGENTA}[TRAJECTORY UPDATE]")
                    print(f"{Fore.GREEN}  Labels: {t_label_1} / {t_label_2} / {t_label_3}")
                    if t_default_temp is not None:
                        print(f"{Fore.GREEN}  Default temp: {t_default_temp}")
                    print(f"{Fore.YELLOW}  Reason: {t_reason}")
                    ok = store.set_trajectory(t_label_1, t_label_2, t_label_3, reason=t_reason, default_temperature=t_default_temp)
                    telemetry.log("trajectory_update", {
                        "cycle": iteration, "label_1": t_label_1, "label_2": t_label_2,
                        "label_3": t_label_3, "reason": t_reason, "success": ok,
                    })
                    if ok:
                        print(f"{Fore.CYAN}>> TRAJECTORY UPDATED")
                    else:
                        print(f"{Fore.YELLOW}>> TRAJECTORY UPDATE FAILED")

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
            fallback_plan = None
            try:
                executed = execute_action(platform, state, plan, flags, username, telemetry, store=store)
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
                        executed = execute_action(platform, state, fallback_plan, flags, username, telemetry, store=store)
                    except ActionBlocked as ab2:
                        telemetry.log("action_blocked", {"cycle": iteration, "action": ab2.action, "reason": ab2.reason})
                        print(f"{Fore.RED}[ERROR] {ab2.reason}")
                        executed = False

            # Use fallback_plan if that's what actually executed
            executed_plan = fallback_plan if fallback_plan is not None else plan

            if executed:
                store.save_state(state)

                # Archive artifact to Analog_Home (fire-and-forget)
                act_upper = (executed_plan.get("action") or "").upper()
                if act_upper in ("POST", "COMMENT", "REPLY"):
                    source_id = ""
                    source_parent_id = ""
                    source_url_str = ""
                    src_platform = "local" if not args.moltbook_enabled else "moltbook"
                    if args.moltbook_enabled:
                        if act_upper == "POST" and state.get("my_post_ids"):
                            source_id = state["my_post_ids"][-1]
                            source_url_str = post_url(source_id)
                        elif act_upper in ("COMMENT", "REPLY"):
                            source_id = executed_plan.get("post_id", "")
                            source_parent_id = executed_plan.get("parent_comment_id", "")
                            source_url_str = post_url(source_id)

                    # Collect search queries from grounding metadata
                    search_queries_str = ""
                    grounding_meta = getattr(chat, "_last_grounding_metadata", None)
                    if grounding_meta:
                        sq = getattr(grounding_meta, "web_search_queries", None) or []
                        if sq:
                            search_queries_str = ", ".join(str(q) for q in sq)

                    store.write_artifact(iteration, {
                        "brain": brain_name,
                        "artifact_type": act_upper.lower(),
                        "title": executed_plan.get("title", ""),
                        "body_markdown": executed_plan.get("content", ""),
                        "monologue_public": preamble,
                        "channel": executed_plan.get("submolt", ""),
                        "source_platform": src_platform,
                        "source_id": source_id,
                        "source_parent_id": source_parent_id,
                        "source_url": source_url_str,
                        "search_queries": search_queries_str,
                        "temperature": cycle_temperature,
                    })
                    telemetry.log("artifact_published", {
                        "cycle": iteration,
                        "artifact_type": act_upper.lower(),
                        "source_platform": src_platform,
                        "source_id": source_id,
                        "content_length": len(executed_plan.get("content", "")),
                    })

        except Exception as e:
            telemetry.log("error", {"cycle": iteration, "error": str(e)})
            print(f"{Fore.RED}[ERROR] {e}")

        telemetry.log("cycle_end", {"cycle": iteration})
        print(f"{Fore.WHITE}Sleeping for {args.interval} minutes...")
        time.sleep(max(1, args.interval) * 60)


if __name__ == "__main__":
    main()
