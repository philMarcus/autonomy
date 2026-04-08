"""Main entry point for autonomy.

Usage:
    python -m autonomy <brain_name> [directive] [options]

Multi-model agent loop with subconscious daemon (sentry/strategist/seeker),
budget-aware planning, and structured scoring rubric.
"""

import hashlib
import json
import os
import re
import sys
import time
import threading
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
    HISTORY_KEEP, MEMORY_MAX_CHARS,
)
from .telemetry import TelemetryLogger
from .store import LocalFileStore
from .utils import (
    load_kernel, load_knowledge,
    history_context, memory_context, post_url, get_author_name, shorten,
    get_post_comment_count, add_history,
    update_kernel_file,
    aggregate_feeds, format_feed_brief,
)
from .buffer import Draft
from .llm import ModelRegistry, DailyBudget
from .llm.gemini import GeminiBackend
from .platforms.moltbook import MoltbookClient
from .challenges.math_verification import MathVerificationSolver
from .espn import get_espn_context
from .planner import (
    build_planner_prompt, plan_next_action, call_text,
)
from .actions import (
    ActionBlocked, execute_action,
    refresh_my_posts_from_profile, find_unanswered_comment_on_my_posts,
    pick_outside_post_for_comment,
)
from .cooldowns import can_do, set_cooldown, cooldown_status_text, migrate_legacy_cooldowns

colorama_init(autoreset=True)


def safe_print(text: str) -> None:
    """Print text, replacing unencodable characters instead of crashing."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(errors="replace").decode())


def _format_draft_context(drafts: list, saved_plans: list,
                          wake_potential: float, threshold: float) -> str:
    """Format subconscious drafts + saved plans for inclusion in the planner prompt."""
    lines = []
    if drafts:
        lines.append(f"Your subconscious noticed {len(drafts)} NEW item(s) of interest:")
        for i, d in enumerate(drafts, 1):
            source_tag = " [HUMAN SEED]" if d.source == "seed" else " [SEARCH]" if d.source == "search" else ""
            model_tag = f" [by {d.model}]" if d.model else ""
            lines.append(f"\n{i}. [score: {d.signal_score:.2f}]{source_tag}{model_tag} {d.target_summary}")
            lines.append(f"   Suggested: {d.suggested_action} — {d.reasoning}")
            if d.draft_content:
                lines.append(f"   Draft: {d.draft_content[:200]}")
        lines.append(f"\nWake potential: {wake_potential:.2f} (threshold: {threshold:.1f})")
    if saved_plans:
        lines.append(f"\nSAVED PLANS ({len(saved_plans)} from previous cycles — use or discard):")
        for i, d in enumerate(saved_plans, 1):
            source_tag = " [HUMAN SEED]" if d.source == "seed" else " [SEARCH]" if d.source == "search" else ""
            model_tag = f" [by {d.model}]" if d.model else ""
            age = f"saved {d.cycles_saved} cycle(s) ago"
            lines.append(f"\n  S{i}. [score: {d.signal_score:.2f}]{source_tag}{model_tag} {d.target_summary} ({age})")
            lines.append(f"      Suggested: {d.suggested_action} — {d.reasoning}")
            if d.draft_content:
                lines.append(f"      Draft: {d.draft_content[:200]}")
    if drafts or saved_plans:
        lines.append("\nYou may act on any of these, synthesize multiple into one action, or ignore them.")
        lines.append("Plans you don't act on will be saved for future cycles (up to 5 cycles).")
    return "\n".join(lines)


def _build_recent_posts(store, run_id: str, count: int = 4) -> str:
    """Fetch recent artifact bodies from Analog Home for the planner prompt."""
    if not store._analog_home_url or count <= 0:
        return ""
    try:
        import requests
        from urllib.parse import urljoin
        url = urljoin(store._analog_home_url.rstrip("/") + "/",
                      f"artifacts?run_id={run_id}&limit={count}&sort=desc")
        resp = requests.get(url, timeout=10)
        if resp.status_code >= 400:
            return ""
        arts = resp.json()
        # Filter to content artifacts only
        content_arts = [a for a in arts if a.get("artifact_type") in ("post", "comment", "reply", "image")]
        if not content_arts:
            return ""
        lines = []
        for a in content_arts:
            atype = a.get("artifact_type", "")
            title = a.get("title", "")
            body = a.get("body_markdown", "")
            # Truncate extremely long bodies
            _max_body = 5000
            if len(body) > _max_body:
                body = body[:_max_body - 3] + "..."
            lines.append(f"[{atype}] {title}\n{body}")
        return "\n\n".join(lines)
    except Exception:
        return ""


def _build_self_telemetry(state: dict, budget, iteration: int, daemon=None) -> str:
    """Build a concise self-telemetry summary for the planner prompt."""
    from collections import Counter
    lines = ["--- SELF-TELEMETRY ---"]

    # Action distribution from recent history
    history = state.get("history", [])[-15:]
    if history:
        actions = Counter(h.get("action", "?") for h in history)
        action_parts = [f"{a}: {c}" for a, c in actions.most_common()]
        lines.append(f"Recent actions (last {len(history)}): {', '.join(action_parts)}")

    # Budget
    if budget:
        remaining = budget.remaining_fraction()
        lines.append(f"Budget: {remaining:.0%} remaining (${budget.daily_limit_usd * remaining:.2f} of ${budget.daily_limit_usd:.2f})")

    # Sentry model stats from daemon
    if daemon and hasattr(daemon, '_tick_model_counts'):
        mc = daemon._tick_model_counts
        if mc:
            parts = [f"{m}: {c}" for m, c in sorted(mc.items(), key=lambda x: -x[1])]
            lines.append(f"Sentry calls this period: {', '.join(parts)}")

    # Memory and kernel status
    mem_len = len(state.get("memory", ""))
    hist_len = len(state.get("history", []))
    lines.append(f"Memory: {mem_len} chars | History: {hist_len} entries | Cycle: {iteration}")

    # Last image
    last_img = None
    for h in reversed(state.get("history", [])):
        if h.get("action") == "GENERATE_IMAGE":
            last_img = h
            break
    if last_img:
        lines.append(f"Last image: {last_img.get('summary', '')[:60]}")
    else:
        lines.append("Images generated: 0 this session")

    return "\n".join(lines) + "\n"


def _code_checksum() -> str:
    """Hash all .py files in the package directory to detect any code change."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for root, _dirs, files in sorted(os.walk(pkg_dir)):
        for fname in sorted(files):
            if fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "rb") as f:
                        h.update(f.read())
                except OSError:
                    pass
    return h.hexdigest()[:16]


def _compute_run_metadata(kernel: str, knowledge: str, version: str, state: dict) -> dict:
    return {
        "kernel_hash": hashlib.sha256(kernel.encode()).hexdigest()[:16],
        "knowledge_hash": hashlib.sha256(knowledge.encode()).hexdigest()[:16],
        "version": version,
        "code_hash": _code_checksum(),
        "history_count": len(state.get("history", [])),
        "memory_length": len(state.get("memory", "")),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"Autonomy v{VERSION} — modular multi-brain agent loop")
    ap.add_argument("brain", help="Brain name (used as filename prefix in BRAINS_DIR).")
    ap.add_argument("directive", nargs="?", default="Participate on Moltbook.",
                    help="Directive for the agent (default: 'Participate on Moltbook.').")
    DEFAULT_CONSCIOUS_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro").strip()

    # --- Output destinations ---
    ap.add_argument("--no-moltbook", dest="moltbook_enabled", action="store_false", default=True,
                    help="Disable Moltbook actions (POST_MOLTBOOK, COMMENT, REPLY). POST to Analog Home always works.")

    # --- Timing ---
    ap.add_argument("--interval", type=int, default=None,
                    help="Minutes between conscious cycles (overrides control).")
    ap.add_argument("--post-interval", type=int, default=None,
                    help="Minutes between posts (overrides control).")
    ap.add_argument("--reset-post-window", action="store_true",
                    help="Clear the post cooldown timer on startup.")

    # --- Mode ---
    ap.add_argument("--read-only", action="store_true", help="No write actions at all.")
    ap.add_argument("--mode", choices=["all", "comment_only", "no_post", "no_comment", "post_only"],
                    default=None, help="Action mode (overrides control).")
    ap.add_argument("--priority", choices=["replies_first", "outside_first"],
                    default=None, help="Reply vs outside comment priority (overrides control).")
    ap.add_argument("--feed-sort", choices=["hot", "new", "top", "rising"],
                    default="new", help="Feed sort order (default: new).")
    ap.add_argument("--allow-votes", action="store_true", help="Allow upvoting/downvoting.")
    ap.add_argument("--allow-downvote", action="store_true", help="Allow downvoting specifically.")

    # --- Models ---
    ap.add_argument("--conscious-model", default=None,
                    help="Model for conscious loop (overrides control).")
    ap.add_argument("--subconscious-model", default=None,
                    help=argparse.SUPPRESS)  # Deprecated — use model weight controls
    ap.add_argument("--temperature", type=float, default=None,
                    help="LLM temperature (overrides control).")
    ap.add_argument("--daily-budget", type=float, default=None,
                    help="Daily API spend limit in USD (overrides control).")

    # --- Search ---
    ap.add_argument("--no-search", dest="enable_search", action="store_false", default=True,
                    help="Disable Google Search grounding (default: enabled).")
    ap.add_argument("--enable-search", dest="enable_search", action="store_true",
                    help=argparse.SUPPRESS)  # Hidden, kept for backwards compat

    # --- Subconscious daemon ---
    ap.add_argument("--no-subconscious", dest="no_subconscious", action="store_true", default=False,
                    help="Disable subconscious daemon, run single-loop mode (default: daemon enabled).")
    ap.add_argument("--subconscious", dest="no_subconscious", action="store_false",
                    help=argparse.SUPPRESS)  # Hidden, kept for backwards compat
    ap.add_argument("--sentry-interval", type=int, default=None,
                    help="Seconds between sentry scans (overrides control).")

    # --- Misc ---
    ap.add_argument("--allow-kernel-update", action="store_true", default=True,
                    help="Allow the planner to rewrite the kernel prompt (default: enabled).")
    ap.add_argument("--no-kernel-update", dest="allow_kernel_update", action="store_false",
                    help="Prevent the planner from rewriting the kernel prompt.")
    ap.add_argument("--no-kernel-disk-write", action="store_true",
                    help="Kernel updates stay in-memory only (not written to disk).")
    ap.add_argument("--enable-default-temp", action="store_true",
                    help="Allow agent to set default_temperature via trajectory updates.")
    ap.add_argument("--blacklist-controls", default="",
                    help="Comma-separated control keys the LLM cannot modify.")
    ap.add_argument("--reload-env", action="store_true",
                    help="Reload .env file and overwrite current env vars.")
    ap.add_argument("--inject-espn", action="store_true", help="Inject ESPN data into planner context.")
    ap.add_argument("--espn-cache-seconds", type=int, default=60, help=argparse.SUPPRESS)
    ap.add_argument("--espn-league", default=os.environ.get("ESPN_LEAGUE", ESPN_DEFAULT_LEAGUE), help=argparse.SUPPRESS)
    ap.add_argument("--espn-date", default="", help=argparse.SUPPRESS)
    ap.add_argument("--espn-keywords", default="", help=argparse.SUPPRESS)

    # --- Deprecated (hidden, kept for backwards compat) ---
    ap.add_argument("--gemini-model", default=DEFAULT_CONSCIOUS_MODEL, help=argparse.SUPPRESS)

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
    anthropic_key = os.environ.get(f"{prefix}_ANTHROPIC_API_KEY", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    openai_key = os.environ.get(f"{prefix}_OPENAI_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    mistral_key = os.environ.get(f"{prefix}_MISTRAL_API_KEY", "").strip() or os.environ.get("MISTRAL_API_KEY", "").strip()
    username = os.environ.get(f"{prefix}_MY_USERNAME", "").strip() or os.environ.get("MY_USERNAME", "").strip() or brain_name

    if not gem_key:
        raise SystemExit(f"Missing {prefix}_GEMINI_API_KEY (or GEMINI_API_KEY)")
    if not mb_key and args.moltbook_enabled:
        print(f"{Fore.YELLOW}[WARN] Moltbook enabled but no {prefix}_MOLTBOOK_API_KEY found. POST_MOLTBOOK/COMMENT/REPLY will fail.")

    # Telemetry
    run_id = uuid.uuid4().hex
    telemetry_dir = (os.environ.get("TELEMETRY_DIR", "telemetry") or "telemetry").strip()
    telemetry = TelemetryLogger(brain_name=brain_name, run_id=run_id, base_dir=telemetry_dir, read_only=args.read_only)
    # Temporary model defaults (will be replaced by controls after load)
    conscious_model = args.conscious_model or args.gemini_model or "gemini-2.5-pro"

    telemetry.log("run_start", {
        "version": VERSION, "brain_env_prefix": prefix,
        "gemini_key_fp": key_fingerprint(gem_key),
        "moltbook_disabled": not args.moltbook_enabled,
        "model": conscious_model,
        "sentry_cadre": "from_controls",
        "temperature": args.temperature or 0.7,
        "search_enabled": bool(args.enable_search),
        "allow_kernel_update": bool(args.allow_kernel_update),
        "daily_budget_usd": args.daily_budget or 2.0,
        "no_subconscious": args.no_subconscious,
    })

    # === PHASE 2: Build registry (only needs API keys, not model names) ===
    registry = ModelRegistry()
    gemini_backend = GeminiBackend(api_key=gem_key)
    registry.register_backend("gemini", gemini_backend)

    # Optional backends — registered when API key is present and package installed
    if anthropic_key:
        try:
            from .llm.anthropic import AnthropicBackend
            registry.register_backend("anthropic", AnthropicBackend(api_key=anthropic_key))
        except ImportError:
            print("    [WARN] anthropic package not installed — skipping Claude models")
    if openai_key:
        try:
            from .llm.openai import OpenAIBackend
            registry.register_backend("openai", OpenAIBackend(api_key=openai_key))
        except ImportError:
            print("    [WARN] openai package not installed — skipping GPT models")
    if mistral_key:
        try:
            from .llm.mistral import MistralBackend
            registry.register_backend("mistral", MistralBackend(api_key=mistral_key))
        except ImportError:
            print("    [WARN] mistralai package not installed — skipping Mistral models")

    # Ollama — primary local model backend (replaces HuggingFace/PyTorch)
    try:
        from .llm.ollama import OllamaBackend
        _ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434").strip()
        _ollama = OllamaBackend(base_url=_ollama_url)
        if _ollama.is_available():
            registry.register_backend("ollama", _ollama)
            _ollama_models = _ollama.available_models()
            print(f"    Ollama: {len(_ollama_models)} models ({', '.join(m.model_id for m in _ollama_models[:5])})")
    except Exception:
        pass  # Ollama not running or not installed

    llm_client = registry.as_llm_client(default_model_id=conscious_model)

    # --- Control Registry (Phase 4) ---
    from .controls import build_default_registry
    ctrl = build_default_registry(registry, blacklist_str=args.blacklist_controls or "")
    controls_file = os.path.join(BRAINS_DIR, f"{brain_name}_controls.json")
    if os.path.exists(controls_file):
        try:
            with open(controls_file, "r", encoding="utf-8") as cf:
                ctrl.load_from_dict(json.load(cf))
            print(f"{Fore.GREEN}    Controls loaded from {controls_file}")
        except Exception:
            pass  # start fresh if file is corrupt

    # CLI flags override saved controls — explicit flags always win
    _CLI_TO_CONTROL = {
        "--conscious-model":    ("conscious_model",          lambda a: a.conscious_model),
        "--gemini-model":       ("conscious_model",          lambda a: a.gemini_model),  # deprecated alias
        "--temperature":        ("temperature",              lambda a: a.temperature),
        "--daily-budget":       ("daily_budget_usd",         lambda a: a.daily_budget),
        "--interval":           ("cycle_interval_minutes",   lambda a: a.interval),
        "--post-interval":      ("post_interval_minutes",    lambda a: a.post_interval),
        "--sentry-interval":    ("sentry_interval_seconds",  lambda a: a.sentry_interval),
        "--mode":               ("mode",                     lambda a: a.mode),
        "--allow-downvote":     ("allow_downvote",           lambda a: True),
        "--priority":           ("priority",                 lambda a: a.priority),
    }
    # Pre-compute which CLI flags were explicitly passed so we can re-apply after disk reloads
    _cli_pinned: Dict[str, Any] = {}
    for flag, (ctrl_key, getter) in _CLI_TO_CONTROL.items():
        if flag in sys.argv:
            val = getter(args)
            if val is not None:
                _cli_pinned[ctrl_key] = val

    def _apply_cli_overrides(verbose: bool = False) -> None:
        """Re-apply CLI-pinned values to control registry (after disk reload)."""
        applied = []
        rejected = []
        for ctrl_key, val in _cli_pinned.items():
            ok = ctrl.set(ctrl_key, val, source="cli")
            if ok:
                applied.append(f"{ctrl_key}={val}")
            else:
                rejected.append(f"{ctrl_key}={val}")
        if verbose and applied:
            print(f"{Fore.YELLOW}    CLI overrides: {', '.join(applied)}")
        if rejected:
            print(f"{Fore.RED}    CLI overrides REJECTED (invalid value): {', '.join(rejected)}{Style.RESET_ALL}")

    _apply_cli_overrides(verbose=True)

    # === PHASE 4: Derived values from controls (single source of truth) ===
    conscious_model = "gemini-2.5-pro"
    budget = DailyBudget(daily_limit_usd=float(ctrl.get("daily_budget_usd")))

    # Platform client: always create for reads if API key exists
    # --no-moltbook gates WRITES only (via moltbook_disabled flag in actions.py)
    platform = None
    challenge_solver = None
    if mb_key:
        # Use conscious model for verification challenges — cheap models (Flash-Lite)
        # can't reliably parse obfuscated text + do arithmetic.
        # Challenges are rare (only on Moltbook writes), so cost is negligible.
        # Use Flash for verification — 5/5 on math benchmarks, cheaper than Pro
        challenge_llm = registry.as_llm_client(default_model_id="gemini-2.5-flash")
        challenge_solver = MathVerificationSolver(llm_client=challenge_llm, telemetry=telemetry)
        # Backup chain for verification: Pro (if Flash 503s), then Gemma
        challenge_solver.backup_llm = registry.as_llm_client(default_model_id="gemini-2.5-pro")
        if registry.has_model("ollama:gemma3:12b"):
            challenge_solver.backup_llm_2 = registry.as_llm_client(default_model_id="ollama:gemma3:12b")
        platform = MoltbookClient(
            api_key=mb_key, telemetry=telemetry, brain_name=brain_name,
            read_only=args.read_only, challenge_solver=challenge_solver,
        )

    # --- Subconscious Daemon (Phase 5) ---
    from .buffer import DraftBuffer
    from .daemon import SubconsciousDaemon

    draft_buffer = DraftBuffer(
        wake_threshold=5.0,  # initial value — daemon auto-calibrates from target_wake_minutes
        max_drafts=ctrl.get("max_drafts"),
    )
    daemon = None

    # Directive
    user_directive = args.directive

    available_providers = sorted(registry._backends.keys())
    _con_weights = ctrl.get("conscious_model_weights")
    _sub_weights = ctrl.get("subconscious_model_weights")
    _strat_weights = ctrl.get("strategist_model_weights")
    print(f"{Fore.CYAN}=== {brain_name}: autonomy v{VERSION} ===")
    print(f"{Fore.CYAN}    providers: {', '.join(available_providers)}")
    print(f"{Fore.CYAN}    conscious cadre: {_con_weights}")
    print(f"{Fore.CYAN}    sentry cadre:    {_sub_weights}")
    print(f"{Fore.CYAN}    strategist cadre: {_strat_weights}")
    print(f"{Fore.CYAN}    budget: ${ctrl.get('daily_budget_usd'):.2f}/day | target wake: {ctrl.get('target_wake_minutes')}min | sentry: {ctrl.get('sentry_interval_seconds')}s")
    if args.no_subconscious:
        print(f"{Fore.CYAN}    mode: single-loop (--no-subconscious)")
    if not args.moltbook_enabled:
        if platform is not None:
            print(f"{Fore.MAGENTA}    [MOLTBOOK WRITES OFF] Reads: ON | Writes: Analog Home only")
        else:
            print(f"{Fore.MAGENTA}    [NO MOLTBOOK KEY] Output -> Analog Home only")
    else:
        print(f"{Fore.CYAN}    moltbook_key:*{key_fingerprint(mb_key)}")
    if args.post_interval is not None:
        print(f"{Fore.CYAN}    Post interval: {args.post_interval} min (CLI override)")
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
    store = LocalFileStore(state_path, analog_home_url=analog_home_url, run_id=run_id)
    state = store.load_state()

    # Migrate legacy cooldown format (next_post_time / next_comment_time → state["cooldowns"])
    if migrate_legacy_cooldowns(state):
        store.save_state(state)
        print(f"{Fore.GREEN}Migrated legacy cooldowns to unified format.")

    # Reset post window if requested
    if args.reset_post_window:
        state.setdefault("cooldowns", {})["POST"] = 0.0
        store.save_state(state)
        print(f"{Fore.GREEN}Post window reset — first cycle can post immediately.")

    # CLI directive wins if explicitly provided (not the default placeholder).
    # Otherwise fall back to whatever is already saved in state.
    if user_directive and user_directive != "Participate on Moltbook.":
        state["directive"] = user_directive
    elif state.get("directive"):
        user_directive = state["directive"]
    else:
        state["directive"] = user_directive

    kernel = load_kernel(kernel_path)
    telemetry.log_kernel_snapshot(kernel, reason="startup", source="startup")
    knowledge = load_knowledge(knowledge_path)

    # --- Run-start change detection ---
    current_meta = _compute_run_metadata(kernel, knowledge, VERSION, state)
    previous_meta = state.get("_last_run", {})
    changes = []
    if previous_meta:
        if previous_meta.get("version") != current_meta["version"]:
            changes.append(f"Autonomy upgraded: {previous_meta.get('version')} -> {current_meta['version']}")
        if previous_meta.get("code_hash") and previous_meta.get("code_hash") != current_meta["code_hash"]:
            changes.append("Code updated")
        if previous_meta.get("kernel_hash") != current_meta["kernel_hash"]:
            changes.append("Kernel prompt was modified")
        if previous_meta.get("knowledge_hash") != current_meta["knowledge_hash"]:
            changes.append("Knowledge file was updated")
        if current_meta["history_count"] == 0 and previous_meta.get("history_count", 0) > 0:
            changes.append(f"History was wiped (was {previous_meta['history_count']} entries)")
        if current_meta["memory_length"] == 0 and previous_meta.get("memory_length", 0) > 0:
            changes.append("Memory was wiped")
    # --- Session tracking ---
    # A "session" persists across Ctrl+C restarts unless memories/kernel were wiped.
    # This groups artifacts by meaningful continuity, not by process restart.
    existing_session = state.get("_session_id", "")
    fresh_session = False
    if not existing_session:
        # No session_id in state — first run or first run after session tracking added
        fresh_session = True
    elif any("wiped" in c.lower() for c in changes):
        # Memory or history was wiped — this is a fresh start
        fresh_session = True

    if fresh_session:
        session_id = uuid.uuid4().hex
    else:
        session_id = existing_session

    state["_session_id"] = session_id
    state["_last_run"] = current_meta
    store.save_state(state)

    # Update store's run_id to use session_id (groups restarts together)
    store._run_id = session_id

    if analog_home_url:
        con_weights = ctrl.get("conscious_model_weights") or conscious_model
        sub_weights = ctrl.get("subconscious_model_weights") or "gemini-2.5-flash-lite"
        seeker = ctrl.get("seeker_model_weights") or "gemini-2.5-flash-lite=1"
        run_body = (
            f"Version: {VERSION}\n"
            f"Default conscious model: {conscious_model}\n"
            f"**Conscious pool:** {con_weights}\n"
            f"**Subconscious pool:** {sub_weights}\n"
            f"**Seeker model:** {seeker}"
        )
        if fresh_session:
            run_body += "\nSession: NEW"
        else:
            run_body += f"\nSession: continued (restart)"
        if changes:
            run_body += "\n\nChanges detected:\n" + "\n".join(f"- {c}" for c in changes)
        else:
            run_body += "\n\nNo configuration changes since last run."
        store.write_artifact(0, {
            "brain": brain_name,
            "artifact_type": "system_run_start",
            "title": f"Run Started -- {brain_name}",
            "body_markdown": run_body,
            "temperature": float(ctrl.get("temperature") if ctrl else 0.7),
        })

    # Derive permissions
    _mode = ctrl.get("mode")
    allow_posts = (_mode in ("all", "post_only"))
    allow_outside = True
    allow_votes = bool(args.allow_votes)
    allow_downvote = bool(args.allow_votes and args.allow_downvote)
    allow_create_submolt = True  # gated by cooldown system now

    post_cooldown_seconds = int(ctrl.get("post_interval_minutes")) * 60

    moltbook_disabled = not args.moltbook_enabled

    flags: Dict[str, Any] = {
        "allow_posts": allow_posts,
        "allow_outside": allow_outside,
        "allow_votes": allow_votes,
        "allow_downvote": allow_downvote,
        "allow_create_submolt": allow_create_submolt,
        "read_only": args.read_only,
        "moltbook_disabled": moltbook_disabled,
        "write_disabled": False,
        "write_disabled_reason": None,
        "post_cooldown_seconds": post_cooldown_seconds,
        # Social scanning limits (from ControlRegistry)
        "my_post_scan_limit": ctrl.get("my_post_scan_limit"),
        "reply_threads_scanned": ctrl.get("reply_threads_scanned"),
        "reply_max_comments": ctrl.get("reply_max_comments"),
        "thread_comments_for_engagement": ctrl.get("thread_comments_for_engagement"),
        "reply_candidate_chars": ctrl.get("reply_candidate_chars"),
        "outside_candidate_chars": ctrl.get("outside_candidate_chars"),
        "post_failure_cooldown_seconds": ctrl.get("post_failure_cooldown_seconds"),
        "ctrl": ctrl,
    }

    # Build tools list for planner chat (Google Search grounding)
    search_tools = None
    if args.enable_search:
        from google.genai import types as genai_types
        search_tools = [genai_types.Tool(google_search=genai_types.GoogleSearch())]
        print(f"{Fore.GREEN}Google Search grounding enabled.")

    # --- Start subconscious daemon if enabled ---
    if not args.no_subconscious:
        # Search tools for daemon: only if --enable-search and subconscious model is Gemini
        daemon_search_tools = None
        if search_tools:
            # Seeker cadre is Gemini-only by design, so search tools always apply
            daemon_search_tools = search_tools

        state_lock = threading.Lock()
        daemon = SubconsciousDaemon(
            registry=registry, ctrl=ctrl, budget=budget,
            buffer=draft_buffer, telemetry=telemetry,
            platform=platform, kernel=kernel,
            directive=user_directive, brain_name=brain_name,
            username=username,
            store=store if analog_home_url else None,
            search_tools=daemon_search_tools,
            state=state,
            state_lock=state_lock,
        )
        # Seed seen_ids from state so daemon doesn't re-score old feed items on restart
        prior_ids = set(state.get("my_post_ids", []))
        for h in state.get("history", []):
            target = h.get("target", "")
            if "/post/" in target:
                prior_ids.add(target.rsplit("/post/", 1)[-1])
        # Also extract post IDs from replied_comment_keys (format: "post_id:comment_id")
        for key in state.get("replied_comment_keys", []):
            if ":" in key:
                prior_ids.add(key.split(":", 1)[0])
        if prior_ids:
            daemon.seed_seen_ids(prior_ids)
            print(f"{Fore.CYAN}    daemon seeded with {len(prior_ids)} known post IDs")
        daemon.start()
        extras = []
        if daemon_search_tools:
            extras.append("search: ON")
        if analog_home_url:
            extras.append("seeds: ON")
        extra_str = f" | {', '.join(extras)}" if extras else ""
        _sentry_w = ctrl.get("subconscious_model_weights") or "flash-lite"
        _strat_w = ctrl.get("strategist_model_weights") or "flash-lite"
        print(f"{Fore.CYAN}    subconscious daemon: ACTIVE | sentry: {_sentry_w} | strategist: {_strat_w}{extra_str}")

    iteration = 0
    prev_feed_available = None  # Track feed state transitions
    while True:
        iteration += 1

        # --- Re-read controls from disk (picks up dashboard edits) ---
        if os.path.exists(controls_file):
            try:
                with open(controls_file, "r", encoding="utf-8") as cf:
                    ctrl.load_from_dict(json.load(cf))
            except Exception:
                pass
            _apply_cli_overrides()  # CLI flags always win over disk values

        # --- Analog Home controls (only when API URL is configured) ---
        analog_controls = {}
        analog_seeds = []
        analog_trajectory = None
        cycle_temperature = float(ctrl.get("temperature") if ctrl else 0.7)
        agent_default_temperature = float(ctrl.get("temperature") if ctrl else 0.7)  # what user nudges decay toward
        if analog_home_url:
            analog_controls = store.read_controls()
            if analog_controls:
                cycle_temperature = analog_controls.get("temperature", float(ctrl.get("temperature") if ctrl else 0.7))
                agent_default_temperature = analog_controls.get("default_temperature", float(ctrl.get("temperature") if ctrl else 0.7))
                analog_seeds = analog_controls.get("seeds", [])
                seed_ids = analog_controls.get("seed_ids", [])
                if seed_ids:
                    store.consume_seeds(seed_ids)
                # Save seeds to state for historical record
                if analog_seeds:
                    import datetime as _dt
                    saved_seeds = state.setdefault("_seed_history", [])
                    for _seed_text in analog_seeds:
                        saved_seeds.append({
                            "text": _seed_text,
                            "cycle": iteration,
                            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                        })
                    # Keep last 50 seeds
                    state["_seed_history"] = saved_seeds[-50:]
                analog_trajectory = {
                    "vote_1": analog_controls.get("vote_1", 0),
                    "vote_2": analog_controls.get("vote_2", 0),
                    "vote_3": analog_controls.get("vote_3", 0),
                    "vote_label_1": analog_controls.get("vote_label_1", "emergence"),
                    "vote_label_2": analog_controls.get("vote_label_2", "entropy"),
                    "vote_label_3": analog_controls.get("vote_label_3", "self"),
                }

        # Read current conscious model — use weighted pool if set
        from .daemon import _pick_weighted_model
        conscious_model = _pick_weighted_model(
            ctrl.get("conscious_model_weights"),
            "gemini-2.5-pro",
        )
        safe_print(f"{Fore.CYAN}[CONSCIOUS MODEL] {conscious_model}")

        # --- Budget planning (accountant) ---
        if ctrl.get("budget_plan_enabled"):
            from .accountant import should_run_budget_plan, build_budget_plan_prompt, parse_budget_plan, apply_budget_plan
            _last_budget_plan = state.get("_last_budget_plan_time", 0)
            if should_run_budget_plan(budget, ctrl, _last_budget_plan):
                try:
                    bp_prompt = build_budget_plan_prompt(budget, ctrl, registry)
                    bp_chat = registry.create_chat(
                        model_id=conscious_model,
                        system_instruction="You are a budget planner. Respond with valid JSON only.",
                        temperature=0.3,
                        max_output_tokens=1024,
                    )
                    bp_raw = bp_chat.send_message(bp_prompt)
                    # Record accountant spend
                    from .llm.base import LLMResponse as _LLMResp
                    from .llm.budget import estimate_cost as _est_cost
                    _bp_in = getattr(bp_chat, "_last_input_tokens", 0) or (len(bp_prompt) // 4)
                    _bp_out = getattr(bp_chat, "_last_output_tokens", 0) or (len(bp_raw) // 4)
                    budget.record_usage(conscious_model, _LLMResp(
                        text=bp_raw, input_tokens=_bp_in, output_tokens=_bp_out,
                        model_id=conscious_model))
                    bp_plan = parse_budget_plan(bp_raw)
                    if bp_plan:
                        bp_changes = apply_budget_plan(bp_plan, ctrl)
                        state["_last_budget_plan_time"] = time.time()
                        store.save_state(state)
                        # Re-read conscious model in case accountant changed it
                        conscious_model = "gemini-2.5-pro"
                        telemetry.log("budget_plan", {
                            "cycle": iteration,
                            "changes": bp_changes,
                            "reasoning": bp_plan.get("reasoning", ""),
                        })
                        if bp_changes:
                            try:
                                print(f"{Fore.CYAN}[BUDGET] Plan applied: {bp_changes}")
                            except Exception:
                                pass
                except Exception as bp_err:
                    telemetry.log("budget_plan_error", {
                        "cycle": iteration, "error": str(bp_err),
                    })

        # Recreate chat each cycle to avoid token accumulation
        # Use registry directly so backend is resolved per-model (supports cross-provider model switching)
        chat = registry.create_chat(
            model_id=conscious_model,
            system_instruction=kernel,
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
            "cycle": iteration, "temperature": cycle_temperature, "model": conscious_model,
            **({"analog_controls": analog_controls} if analog_controls else {}),
        })

        # --- Feed context (reads decoupled from writes) ---
        feed_sources = []
        source_errors: Dict[str, str] = {}
        reply_candidate = None
        outside_candidate = None
        fresh_drafts = []
        saved_plans = []

        if platform is not None:
            try:
                did_add = refresh_my_posts_from_profile(platform, state, username,
                                                        my_post_scan_limit=ctrl.get("my_post_scan_limit"))
                if did_add:
                    store.save_state(state)
            except Exception:
                source_errors["profile"] = platform.last_error_type or "timeout"

            try:
                mb_feed = platform.get_feed(limit=FEED_LIMIT, sort=args.feed_sort)
                feed_sources.append({"name": "moltbook", "items": mb_feed})
            except Exception:
                source_errors["moltbook"] = platform.last_error_type or "timeout"

        # Future: add more feed sources here
        # if reddit_client:
        #     feed_sources.append({"name": "reddit", "items": reddit_client.get_feed(...)})

        feed = aggregate_feeds(feed_sources)
        feed_available = len(feed) > 0

        # Detect feed state transitions
        if prev_feed_available is not None and feed_available != prev_feed_available:
            if feed_available:
                safe_print(f"{Fore.GREEN}[FEED RESUMED] Feeds available again! ({len(feed)} items)")
                telemetry.log("feed_resumed", {"cycle": iteration, "items": len(feed)})
            else:
                reasons = ", ".join(source_errors.values()) or "unknown"
                safe_print(f"{Fore.RED}[FEED UNAVAILABLE] Feeds went offline ({reasons})")
                telemetry.log("feed_unavailable", {"cycle": iteration, "reasons": source_errors})
        prev_feed_available = feed_available

        feed_brief = format_feed_brief(feed, source_errors=source_errors, feed_item_chars=ctrl.get("feed_item_chars"))
        if not feed and not source_errors and platform is None:
            feed_brief = "No feed available (no platform API key)."

        telemetry.log("feed_context", {"text_length": len(feed_brief), "feed_items": len(feed),
                                       "source_errors": source_errors, "brief": feed_brief[:2000]})

        # Extract candidates from daemon drafts (sentry-scored, pre-filtered)
        _all_daemon_drafts = (fresh_drafts or []) + (saved_plans or [])
        _outside_chars = ctrl.get("outside_candidate_chars") or 5000
        _reply_chars = ctrl.get("reply_candidate_chars") or 5000

        # Best COMMENT draft → outside_candidate
        _comment_drafts = [d for d in _all_daemon_drafts if d.suggested_action == "COMMENT"]
        if _comment_drafts and platform:
            _best_comment = max(_comment_drafts, key=lambda d: d.signal_score)
            try:
                _post_data = platform.get_post(_best_comment.item_id) or {}
                _post_obj = _post_data.get("post", _post_data)
                outside_candidate = {
                    "post_id": _best_comment.item_id,
                    "title": _post_obj.get("title", ""),
                    "content": shorten(_post_obj.get("content", ""), _outside_chars),
                    "author": get_author_name(_post_obj.get("author")),
                    "signal_score": _best_comment.signal_score,
                }
            except Exception:
                pass

        # Best REPLY draft → reply_candidate
        _reply_drafts = [d for d in _all_daemon_drafts if d.suggested_action == "REPLY" and d.source == "reply"]
        if _reply_drafts and platform:
            _best_reply = max(_reply_drafts, key=lambda d: d.signal_score)
            try:
                _parts = _best_reply.item_id.split(":", 1)
                if len(_parts) == 2:
                    _r_post_id, _r_comment_id = _parts
                    _comments = platform.get_post_comments(_r_post_id, sort="new") or []
                    for _c in _comments:
                        if _c.get("id") == _r_comment_id:
                            reply_candidate = {
                                "post_id": _r_post_id,
                                "comment_id": _r_comment_id,
                                "comment_author": get_author_name(_c.get("author")),
                                "comment_content": shorten(_c.get("content", ""), _reply_chars),
                                "post_title": _c.get("post", {}).get("title") if isinstance(_c.get("post"), dict) else None,
                                "signal_score": _best_reply.signal_score,
                            }
                            break
            except Exception:
                pass

        # Fallback to old functions if daemon didn't provide candidates
        if platform is not None:
            if not reply_candidate:
                try:
                    reply_candidate = find_unanswered_comment_on_my_posts(
                        platform, state, username, telemetry, ctrl.get("max_item_age_hours"),
                        my_post_scan_limit=ctrl.get("my_post_scan_limit"),
                        reply_threads_scanned=ctrl.get("reply_threads_scanned"),
                        reply_max_comments=ctrl.get("reply_max_comments"),
                        reply_candidate_chars=ctrl.get("reply_candidate_chars"),
                    )
                except Exception:
                    pass
            if not outside_candidate:
                outside_candidate = pick_outside_post_for_comment(
                    feed, state, username, ctrl.get("max_item_age_hours"),
                    thread_comments_for_engagement=ctrl.get("thread_comments_for_engagement"),
                    outside_candidate_chars=ctrl.get("outside_candidate_chars"),
                )

        # Compute Moltbook post window (only relevant when Moltbook enabled)
        if flags.get("moltbook_disabled"):
            moltbook_post_window_open = False
            moltbook_post_wait = 0
            print(f"{Fore.WHITE}Moltbook: DISABLED (POST to Analog Home always available)")
        else:
            post_ok, post_wait_secs = can_do(state, "POST", ctrl=ctrl)
            moltbook_post_window_open = post_ok
            moltbook_post_wait = post_wait_secs // 60
            window = "OPEN" if moltbook_post_window_open else f"CLOSED ({moltbook_post_wait}m)"
            print(f"{Fore.WHITE}Moltbook Post Window: {window}")

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

        hist_txt = history_context(state)
        mem_txt = memory_context(state)
        config_hint = ""
        if ctrl.get("priority") == "outside_first":
            config_hint = "- Default preference overridden: prefer outside comments when not posting.\n"

        # Memory status indicator
        history_count = len(state.get("history", []))
        tiers = state.get("memory_tiers", {})
        recent_count = len(tiers.get("recent", []))
        compressed_count = len(tiers.get("compressed", []))
        deep_count = len(tiers.get("deep", []))
        memory_pressure = ""
        if history_count > 10 or recent_count > 0:
            memory_pressure = (
                f"History: {history_count} entries | "
                f"Memory: {recent_count} recent, {compressed_count} compressed, {deep_count} deep"
            )

        # Drain subconscious buffer (if daemon active) + load saved plans
        draft_context = ""
        seeker_findings = ""
        fresh_drafts = []
        saved_plans = []
        if daemon:
            fresh_drafts, wake_pot = draft_buffer.drain(refractory=ctrl.get("wake_refractory"))
            if fresh_drafts:
                safe_print(f"{Fore.MAGENTA}[DAEMON] {len(fresh_drafts)} draft(s) from subconscious (wake_potential={wake_pot:.2f})")
                # Show model distribution in drafts
                from collections import Counter
                model_counts = Counter(d.model for d in fresh_drafts if d.model)
                if model_counts:
                    model_summary = ", ".join(f"{m}: {c}" for m, c in model_counts.most_common())
                    safe_print(f"{Fore.CYAN}  Draft models: {model_summary}")
            # Capture daemon tick stats (model usage since last wake) — save for cycle report
            _sentry_model_counts = {}
            if hasattr(daemon, '_tick_model_counts'):
                _sentry_model_counts = dict(daemon._tick_model_counts)
                if _sentry_model_counts:
                    tick_summary = ", ".join(f"{m}: {c}" for m, c in sorted(_sentry_model_counts.items(), key=lambda x: -x[1]))
                    safe_print(f"{Fore.CYAN}  Sentry ticks by model: {tick_summary}")
                    daemon._tick_model_counts = {}  # reset for next wake period
            # Drain seeker findings (living summary from research rabbit hole)
            _seeker_summary = draft_buffer.get_seeker_summary()
            if _seeker_summary:
                _seeker_state = draft_buffer.get_seeker_state()
                safe_print(f"{Fore.CYAN}[SEEKER] Research summary available ({_seeker_state.runs_this_cycle} runs, "
                           f"{len(_seeker_state.search_terms)} active terms)")
                seeker_findings = _seeker_summary
            # Load saved plans from state (previous cycles' unused drafts)
            saved_plans = [
                Draft.from_dict(d) for d in state.get("saved_plans", [])
                if isinstance(d, dict)
            ]
            if saved_plans:
                safe_print(f"{Fore.MAGENTA}[SAVED] {len(saved_plans)} plan(s) from previous cycles")
            if fresh_drafts or saved_plans:
                draft_context = _format_draft_context(
                    fresh_drafts, saved_plans, wake_pot, draft_buffer._wake_threshold,
                )

        # Platform write status for planner awareness
        if platform is None:
            platform_status = "No platform API key — feeds and writes unavailable."
        elif flags.get("write_disabled"):
            platform_status = (
                f"READS OK, WRITES BLOCKED ({flags['write_disabled_reason']}). "
                "POST/COMMENT/REPLY will fail on Moltbook."
            )
        elif flags.get("moltbook_disabled"):
            platform_status = "READS OK, MOLTBOOK WRITES OFF. Posts/comments archived to Analog Home only."
        else:
            platform_status = "Moltbook ONLINE (reads + writes active)."

        # Build nudges for actions the agent hasn't done recently
        _nudge_parts = []
        _img_ok, _img_secs = can_do(state, "GENERATE_IMAGE", ctrl=ctrl)
        if _img_ok:
            _nudge_parts.append("GENERATE_IMAGE is available — consider creating a visual artifact if inspiration strikes.")
        _last_tagline = state.get("_last_tagline_cycle", 0)
        if iteration - _last_tagline >= 20:
            _nudge_parts.append("You haven't updated your Analog Home tagline recently — consider refreshing it if your focus has shifted.")
        nudge_note = ("\n--- NUDGES ---\n" + "\n".join(_nudge_parts) + "\n") if _nudge_parts else ""

        prompt = build_planner_prompt(
            directive=user_directive, knowledge=knowledge, memory=mem_txt,
            hist=hist_txt, feed_brief=feed_brief, external_data=external_data,
            moltbook_post_window_open=moltbook_post_window_open, moltbook_post_wait_minutes=moltbook_post_wait,
            reply_candidate=reply_candidate, outside_candidate=outside_candidate,
            config_hint=config_hint, allow_posts=allow_posts, allow_outside=allow_outside,
            allow_votes=allow_votes, allow_create_submolt=allow_create_submolt,
            allow_downvote=allow_downvote, read_only=flags.get("read_only", False),
            current_kernel=kernel if ctrl.get("allow_kernel_update") else "",
            moltbook_enabled=args.moltbook_enabled,
            search_enabled=bool(args.enable_search),
            seeds=analog_seeds,
            trajectory_votes=analog_trajectory,
            cycle_temperature=cycle_temperature if analog_home_url else None,
            default_temperature=agent_default_temperature,
            allow_default_temp=bool(args.enable_default_temp),
            cooldown_status=cooldown_status_text(state, ctrl=ctrl),
            controls_block=ctrl.to_llm_block(),
            budget_summary=budget.spend_summary_for_planning(registry) if budget else "",
            draft_context=draft_context,
            seeker_findings=seeker_findings,
            memory_pressure=memory_pressure,
            daemon_active=daemon is not None,
            platform_status=platform_status,
            nudge_note=nudge_note,
            self_telemetry=_build_self_telemetry(state, budget, iteration, daemon),
            recent_posts=_build_recent_posts(store, state.get("_session_id", ""),
                                             count=int(ctrl.get("recent_posts_in_prompt") or 4)),
        )

        plan = None
        try:
            try:
                plan = plan_next_action(chat, prompt, telemetry=telemetry, brain_name=brain_name, budget=budget)
            except Exception as _plan_err:
                _err_str = str(_plan_err)
                if "503" in _err_str or "UNAVAILABLE" in _err_str:
                    # Retry with a different conscious model
                    from .daemon import _pick_weighted_model
                    retry_model = _pick_weighted_model(
                        ctrl.get("conscious_model_weights"), "gemini-2.5-pro",
                    )
                    # Keep retrying until we get a different model (max 3 attempts)
                    for _ in range(3):
                        if retry_model != conscious_model:
                            break
                        retry_model = _pick_weighted_model(
                            ctrl.get("conscious_model_weights"), "gemini-2.5-pro",
                        )
                    # Try each conscious pool model until one works
                    _all_con = [p.split("=")[0].strip() for p in (ctrl.get("conscious_model_weights") or "").split(",") if "=" in p]
                    _tried = {conscious_model}
                    _success = False
                    for _candidate in _all_con:
                        if _candidate in _tried:
                            continue
                        _tried.add(_candidate)
                        safe_print(f"{Fore.YELLOW}[503] {conscious_model} unavailable, trying {_candidate}")
                        try:
                            chat = registry.create_chat(
                                model_id=_candidate,
                                system_instruction=kernel,
                                temperature=cycle_temperature,
                                max_output_tokens=16384,
                                tools=search_tools if args.enable_search else None,
                            )
                            plan = plan_next_action(chat, prompt, telemetry=telemetry, brain_name=brain_name, budget=budget)
                            _success = True
                            break
                        except Exception as _inner_err:
                            if "503" in str(_inner_err) or "UNAVAILABLE" in str(_inner_err):
                                continue  # try next model
                            raise  # non-503 error, propagate
                    if not _success:
                        # All conscious models 503'd — WAIT, don't degrade
                        safe_print(f"{Fore.RED}[503] All conscious models unavailable. Waiting for next cycle.")
                        plan = {"action": "WAIT", "summary": "All conscious models returned 503. Waiting."}
                else:
                    raise

            # Display any non-JSON LLM output (reasoning, preamble, etc.)
            preamble = plan.pop("_preamble", "")
            if preamble:
                safe_print(f"{Fore.CYAN}--- REASONING ---")
                safe_print(f"{Fore.WHITE}{preamble}")
                safe_print(f"{Fore.CYAN}-----------------{Style.RESET_ALL}")

            # Extract and save memory_note to hierarchical memory
            memory_note = (plan.pop("memory_note", None) or "").strip()
            if memory_note:
                from .utils import compress_memory_tier
                tiers = state.setdefault("memory_tiers", {"recent": [], "compressed": [], "deep": []})
                tiers["recent"].append({"cycle": iteration, "note": memory_note})
                safe_print(f"{Fore.GREEN}[MEMORY] {memory_note}")

                # Auto-compress when tiers overflow
                _recent_cap = int(ctrl.get("memory_recent_capacity") if ctrl else 20)
                _compressed_cap = int(ctrl.get("memory_compressed_capacity") if ctrl else 10)
                _deep_cap = int(ctrl.get("memory_deep_capacity") if ctrl else 10)
                _compressor = ctrl.get("compressor_model") if ctrl else "gemini-2.5-flash"

                def _compress_fn(prompt):
                    _c = registry.create_chat(
                        model_id=_compressor, system_instruction="Summarize concisely.",
                        temperature=0.3, max_output_tokens=512)
                    return _c.send_message(prompt)

                if len(tiers["recent"]) >= _recent_cap:
                    half = _recent_cap // 2
                    to_compress = tiers["recent"][:half]
                    result = compress_memory_tier(to_compress, _compress_fn, tier_name="recent")
                    if result:
                        tiers["recent"] = tiers["recent"][half:]
                        tiers["compressed"].append(result)
                        safe_print(f"{Fore.CYAN}[COMPRESS] {half} recent → compressed: {result['summary'][:80]}...")

                if len(tiers["compressed"]) >= _compressed_cap:
                    half = _compressed_cap // 2
                    to_compress = tiers["compressed"][:half]
                    result = compress_memory_tier(to_compress, _compress_fn, tier_name="compressed")
                    if result:
                        tiers["compressed"] = tiers["compressed"][half:]
                        tiers["deep"].append(result)
                        safe_print(f"{Fore.CYAN}[COMPRESS] {half} compressed → deep: {result['summary'][:80]}...")

                if len(tiers["deep"]) >= _deep_cap:
                    half = _deep_cap // 2
                    to_compress = tiers["deep"][:half]
                    result = compress_memory_tier(to_compress, _compress_fn, tier_name="deep")
                    if result:
                        tiers["deep"] = tiers["deep"][half:]
                        tiers["deep"].insert(0, result)
                        safe_print(f"{Fore.CYAN}[COMPRESS] {half} deep → ultra-deep: {result['summary'][:80]}...")

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
                    safe_print(f"{Fore.GREEN}--- SEARCH GROUNDING ---")
                    for q in search_queries:
                        safe_print(f"{Fore.WHITE}  Query: {q}")
                    for src in source_urls[:5]:
                        safe_print(f"{Fore.WHITE}  Source: {src.get('title', '?')} -- {src.get('uri', '?')}")
                    safe_print(f"{Fore.GREEN}------------------------{Style.RESET_ALL}")
                    telemetry.log("grounding_metadata", {
                        "search_queries": list(search_queries),
                        "source_count": len(chunks),
                        "sources": source_urls[:10],
                    })

            # Log full planner decision to telemetry
            telemetry.log_planner_decision(
                plan=plan, preamble=preamble,
                model=conscious_model, temperature=cycle_temperature,
            )

            # Check for kernel update request (controlled by allow_kernel_update)
            if plan.get("update_kernel") and ctrl.get("allow_kernel_update"):
                new_kernel = plan.get("new_kernel", "").strip()
                reason = plan.get("kernel_reason", "no reason given")

                try:
                    safe_print(f"{Fore.MAGENTA}[KERNEL UPDATE REQUESTED]")
                    safe_print(f"{Fore.YELLOW}Reason: {reason}")
                    safe_print(f"{Fore.YELLOW}New kernel length: {len(new_kernel)} chars")
                except Exception:
                    pass

                if not flags.get("read_only"):
                    if args.no_kernel_disk_write:
                        telemetry.log_kernel_snapshot(new_kernel, reason=reason, source="memory_only")
                        kernel = new_kernel
                        if daemon:
                            daemon.update_context(kernel=kernel, directive=user_directive)
                        try:
                            print(f"{Fore.CYAN}[NO-DISK] Kernel updated in-memory only (--no-kernel-disk-write)")
                        except:
                            pass
                        telemetry.log("kernel_update_memory_only", {
                            "cycle": iteration,
                            "reason": reason,
                            "new_length": len(new_kernel),
                        })
                        store.write_artifact(iteration, {
                            "brain": brain_name,
                            "artifact_type": "system_kernel_update",
                            "title": "Kernel Self-Update",
                            "body_markdown": reason,
                            "temperature": cycle_temperature,
                        })
                    else:
                        result = update_kernel_file(kernel_path, new_kernel, telemetry=telemetry, reason=reason)

                        if result["success"]:
                            kernel = new_kernel  # Update in-memory kernel for next cycle
                            if daemon:
                                daemon.update_context(kernel=kernel, directive=user_directive)
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
                            # Update _last_run kernel_hash so next run-start
                            # doesn't falsely report "Kernel prompt was modified"
                            if "_last_run" in state:
                                state["_last_run"]["kernel_hash"] = hashlib.sha256(new_kernel.encode()).hexdigest()[:16]
                                store.save_state(state)
                            store.write_artifact(iteration, {
                                "brain": brain_name,
                                "artifact_type": "system_kernel_update",
                                "title": "Kernel Self-Update",
                                "body_markdown": reason,
                                "temperature": cycle_temperature,
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
                    safe_print(f"{Fore.MAGENTA}[TRAJECTORY UPDATE]")
                    safe_print(f"{Fore.GREEN}  Labels: {t_label_1} / {t_label_2} / {t_label_3}")
                    if t_default_temp is not None:
                        safe_print(f"{Fore.GREEN}  Default temp: {t_default_temp}")
                    safe_print(f"{Fore.YELLOW}  Reason: {t_reason}")
                    ok = store.set_trajectory(t_label_1, t_label_2, t_label_3, reason=t_reason, default_temperature=t_default_temp)
                    telemetry.log("trajectory_update", {
                        "cycle": iteration, "label_1": t_label_1, "label_2": t_label_2,
                        "label_3": t_label_3, "reason": t_reason, "success": ok,
                    })
                    if ok:
                        print(f"{Fore.CYAN}>> TRAJECTORY UPDATED")
                    else:
                        print(f"{Fore.YELLOW}>> TRAJECTORY UPDATE FAILED")

            # Check for tagline update
            if plan.get("tagline") and analog_home_url:
                new_tagline = str(plan["tagline"]).strip()[:200]
                if new_tagline:
                    safe_print(f"{Fore.MAGENTA}[TAGLINE UPDATE] {new_tagline}")
                    ok = store.set_tagline(new_tagline)
                    telemetry.log("tagline_update", {
                        "cycle": iteration, "tagline": new_tagline, "success": ok,
                    })
                    if ok:
                        state["_last_tagline_cycle"] = iteration
                        store.write_artifact(iteration, {
                            "brain": brain_name,
                            "artifact_type": "system_tagline_update",
                            "title": "Tagline Updated",
                            "body_markdown": new_tagline,
                            "temperature": cycle_temperature,
                        })

            # --- Handle controls_update from planner ---
            control_updates = plan.pop("controls_update", None)
            if control_updates and isinstance(control_updates, dict):
                results = ctrl.apply_updates(control_updates, source="conscious")
                applied = {k: v for k, v in results.items() if v == "ok"}
                blocked = {k: v for k, v in results.items() if v == "blocked"}

                telemetry.log("controls_update", {
                    "cycle": iteration, "updates": control_updates,
                    "results": results, "applied_count": len(applied),
                    "blocked_count": len(blocked),
                })

                for ck, cv in results.items():
                    if cv == "ok":
                        safe_print(f"{Fore.GREEN}  [CTRL] {ck} -> {ctrl.get(ck)}")
                    elif cv == "blocked":
                        safe_print(f"{Fore.YELLOW}  [CTRL] {ck} BLOCKED (blacklisted)")


                # Actuate budget change
                if results.get("daily_budget_usd") == "ok":
                    budget.daily_limit_usd = ctrl.get("daily_budget_usd")

                # Actuate temperature change — update Analog Home default (decay target)
                if results.get("temperature") == "ok":
                    new_default = ctrl.get("temperature")
                    ok = store.set_default_temperature(new_default)
                    safe_print(f"{Fore.CYAN}  [CTRL] Agent temperature preference -> {new_default:.2f} "
                               f"({'synced to Analog Home' if ok else 'local only — no Analog Home'})")

                # Publish control changes to Analog Home
                if applied:
                    changes_lines = []
                    for ck in sorted(applied.keys()):
                        changes_lines.append(f"- **{ck}**: {ctrl.get(ck)}")
                    if blocked:
                        changes_lines.append("")
                        for ck in sorted(blocked.keys()):
                            changes_lines.append(f"- ~~{ck}~~ (blocked by blacklist)")
                    store.write_artifact(iteration, {
                        "brain": brain_name,
                        "artifact_type": "system_controls_update",
                        "title": "Controls Updated",
                        "body_markdown": "\n".join(changes_lines),
                        "temperature": cycle_temperature,
                    })

            # --- Handle daemon directives (downward causality) ---
            daemon_directives = plan.pop("daemon_directives", None)
            if daemon and daemon_directives and isinstance(daemon_directives, dict):
                daemon.set_directives(daemon_directives)
                telemetry.log("daemon_directives", {
                    "cycle": iteration, "directives": daemon_directives,
                })
                focus = daemon_directives.get("focus_topics", [])
                if focus:
                    safe_print(f"{Fore.GREEN}  Focus: {', '.join(str(t) for t in focus)}")
                    # Reset seeker with new topics — starts fresh rabbit hole
                    draft_buffer.reset_seeker(focus)
                ignore = daemon_directives.get("ignore_authors", [])
                note = daemon_directives.get("note", "")
                urgency = daemon_directives.get("urgency_boost", 1.0)

            # --- Publish cycle report to Analog Home (non-fatal) ---
            if daemon:
              try:
                from collections import Counter
                report_parts = []

                # Conscious model
                report_parts.append(f"**Conscious model:** {conscious_model}")

                # Sentry stats
                if _sentry_model_counts:
                    sentry_lines = ", ".join(f"{m}: {c}" for m, c in
                                             sorted(_sentry_model_counts.items(), key=lambda x: -x[1]))
                    report_parts.append(f"**Sentry calls by model:** {sentry_lines}")

                # Strategist stats
                if fresh_drafts:
                    draft_models = Counter(d.model for d in fresh_drafts if d.model)
                    draft_lines = ", ".join(f"{m}: {c}" for m, c in draft_models.most_common())
                    report_parts.append(f"**Strategist drafts:** {len(fresh_drafts)} ({draft_lines})")
                else:
                    report_parts.append("**Strategist drafts:** 0")

                # Seeker stats
                _sk = draft_buffer.get_seeker_state()
                if _sk.runs_this_cycle > 0:
                    report_parts.append(f"**Seeker runs:** {_sk.runs_this_cycle} | "
                                        f"terms: {', '.join(_sk.search_terms[:5]) if _sk.search_terms else 'none'}")

                # Wake info
                if fresh_drafts:
                    report_parts.append(f"**Wake potential:** {wake_pot:.2f} / {draft_buffer._wake_threshold:.1f}")

                # Budget
                if budget:
                    _spent = budget.spent_today_usd()
                    remaining = budget.daily_limit_usd - _spent
                    report_parts.append(f"**Budget:** ${_spent:.3f} spent, "
                                        f"${remaining:.3f} remaining of ${budget.daily_limit_usd:.2f}")

                # Directives (if provided this cycle)
                if daemon_directives and isinstance(daemon_directives, dict):
                    report_parts.append("")  # blank line separator
                    if focus:
                        report_parts.append("**Directives — Focus:** " + ", ".join(str(t) for t in focus))
                    if ignore:
                        report_parts.append("**Directives — Ignore:** " + ", ".join(str(a) for a in ignore))
                    if urgency != 1.0:
                        report_parts.append(f"**Directives — Urgency:** {urgency:.1f}x")
                    if note:
                        report_parts.append(f"**Directives — Note:** {note}")

                store.write_artifact(iteration, {
                    "brain": brain_name,
                    "artifact_type": "system_cycle_report",
                    "title": "Cycle Report",
                    "body_markdown": "\n".join(report_parts),
                    "temperature": cycle_temperature,
                })
              except Exception as _report_err:
                safe_print(f"{Fore.RED}[CYCLE REPORT] Failed: {_report_err}{Style.RESET_ALL}")

            # Fill missing IDs from candidates
            # WARNING: If planner chooses REPLY/COMMENT without post_id, we auto-fill from candidates.
            # This can cause misdirection if planner thinks REPLY works for seeds (it doesn't — seeds have no post_id).
            act_upper = (plan.get("action") or "").upper()
            if act_upper == "REPLY" and reply_candidate:
                if "post_id" not in plan:
                    safe_print(f"{Fore.YELLOW}[AUTO-FILL] REPLY action missing post_id, using reply_candidate")
                    if analog_seeds:
                        safe_print(f"{Fore.RED}  WARNING: Seeds present — planner may think REPLY works for seeds (it doesn't!)")
                plan.setdefault("post_id", reply_candidate.get("post_id"))
                plan.setdefault("parent_comment_id", reply_candidate.get("comment_id"))
            if act_upper == "COMMENT" and outside_candidate:
                if "post_id" not in plan:
                    safe_print(f"{Fore.YELLOW}[AUTO-FILL] COMMENT action missing post_id, using outside_candidate")
                    if analog_seeds:
                        safe_print(f"{Fore.RED}  WARNING: Seeds present — use POST to respond directly to seeds!")
                plan.setdefault("post_id", outside_candidate.get("id"))

            # DREAM deprecated in v17 — memory compression is now automatic
            act = (plan.get("action") or "").upper().strip()
            if act == "DREAM":
                safe_print(f"{Fore.YELLOW}[DREAM] Deprecated — memory compression is automatic. Treating as WAIT.")

            elif act == "GENERATE_IMAGE":
                # --- Image generation action ---
                img_ok, img_secs = can_do(state, "GENERATE_IMAGE", ctrl=ctrl)
                if not img_ok:
                    safe_print(f"{Fore.YELLOW}[IMAGE] Cooldown active ({img_secs // 3600}h {(img_secs % 3600) // 60}m remaining), skipping.")
                else:
                    image_prompt = (plan.get("image_prompt") or "").strip()
                    if not image_prompt:
                        safe_print(f"{Fore.RED}[IMAGE] No image_prompt in plan, skipping.")
                    else:
                        img_tier = ctrl.get("image_model_tier") or "imagen-ultra"
                        safe_print(f"{Fore.MAGENTA}[IMAGE] Generating ({img_tier}): {image_prompt[:120]}...")
                        try:
                            import base64, io
                            image_bytes, img_model_id, img_cost = gemini_backend.generate_image(
                                prompt=image_prompt, tier=img_tier,
                            )
                            # Compress PNG to JPEG (Imagen outputs ~1MB PNG; JPEG is ~10x smaller)
                            try:
                                from PIL import Image as _PILImage
                                _pil_img = _PILImage.open(io.BytesIO(image_bytes))
                                _jpeg_buf = io.BytesIO()
                                _pil_img.convert("RGB").save(_jpeg_buf, format="JPEG", quality=85)
                                image_bytes = _jpeg_buf.getvalue()
                                _img_mime = "image/jpeg"
                            except ImportError:
                                _img_mime = "image/png"  # fallback if Pillow not installed
                            image_b64 = base64.b64encode(image_bytes).decode("ascii")
                            image_data_uri = f"data:{_img_mime};base64,{image_b64}"

                            from .llm.base import LLMResponse as _ImgResp
                            budget.record_usage(img_model_id, _ImgResp(
                                text="", input_tokens=0, output_tokens=0,
                                cost_usd=img_cost, model_id=img_model_id,
                            ))

                            set_cooldown(state, "GENERATE_IMAGE", ctrl=ctrl)

                            store.write_artifact(iteration, {
                                "brain": brain_name,
                                "artifact_type": "image",
                                "title": shorten(plan.get("title", "Visual Artifact"), 200),
                                "body_markdown": plan.get("content", ""),
                                "monologue_public": preamble,
                                "image_url": image_data_uri,
                                "temperature": cycle_temperature,
                            })

                            safe_print(f"{Fore.GREEN}[IMAGE] Generated and published ({len(image_bytes)} bytes)")
                            telemetry.log("image_generated", {
                                "cycle": iteration,
                                "prompt": image_prompt[:500],
                                "size_bytes": len(image_bytes),
                                "model": img_model_id,
                                "tier": img_tier,
                                "cost_usd": img_cost,
                            })

                            add_history(state, {
                                "action": "GENERATE_IMAGE",
                                "target": "analog_home",
                                "summary": plan.get("summary", "Generated image"),
                            })
                            store.save_state(state)

                        except Exception as e:
                            safe_print(f"{Fore.RED}[IMAGE] Generation failed: {e}")
                            telemetry.log("image_error", {
                                "cycle": iteration,
                                "prompt": image_prompt[:500],
                                "error": str(e)[:500],
                            })

            elif act == "DEV_REQUEST":
                # --- Dev request: agent asks for software changes ---
                request_text = (plan.get("request") or "").strip()
                request_title = (plan.get("title") or "Dev Request").strip()[:200]
                if request_text:
                    safe_print(f"{Fore.MAGENTA}[DEV REQUEST] {request_title}")
                    safe_print(f"{Fore.WHITE}{request_text[:300]}")

                    # Write to local file
                    dev_req_path = os.path.join(BRAINS_DIR, f"{brain_name}_dev_requests.txt")
                    try:
                        with open(dev_req_path, "a", encoding="utf-8") as f:
                            import datetime as _dt
                            f.write(f"\n--- Cycle {iteration} | {_dt.datetime.now().isoformat()} ---\n")
                            f.write(f"Title: {request_title}\n")
                            f.write(f"{request_text}\n")
                    except Exception:
                        pass

                    # Publish as system artifact to Analog Home
                    store.write_artifact(iteration, {
                        "brain": brain_name,
                        "artifact_type": "system_dev_request",
                        "title": request_title,
                        "body_markdown": request_text,
                        "monologue_public": preamble,
                        "temperature": cycle_temperature,
                    })

                    telemetry.log("dev_request", {
                        "cycle": iteration,
                        "title": request_title,
                        "request": request_text[:500],
                    })

                    add_history(state, {
                        "action": "DEV_REQUEST",
                        "target": "developers",
                        "summary": plan.get("summary", request_title),
                    })
                    store.save_state(state)
                else:
                    safe_print(f"{Fore.RED}[DEV REQUEST] Empty request, skipping.")

            else:
                # --- Normal action execution ---
                executed = False
                fallback_plan = None
                try:
                    executed = execute_action(platform, state, plan, flags, username, telemetry, store=store)
                except ActionBlocked as ab:
                    telemetry.log("action_blocked", {"cycle": iteration, "action": ab.action, "reason": ab.reason})
                    safe_print(f"{Fore.RED}[ERROR] {ab.reason}")

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

                    if fallback_plan is None and allow_posts:
                        fallback_plan = {
                            "action": "POST",
                            "title": plan.get("title") or "Thought",
                            "content": plan.get("content") or "",
                            "submolt": plan.get("submolt") or "general",
                            "summary": "fallback_after_block",
                            "source_action": (plan.get("action") or "").upper(),
                        }

                    if fallback_plan is not None:
                        # Regenerate content if action type changed OR target post changed
                        try:
                            src = (fallback_plan.get("source_action") or "").upper()
                            act2 = (fallback_plan.get("action") or "").upper()
                            target_changed = (
                                act2 == "COMMENT"
                                and fallback_plan.get("post_id")
                                and fallback_plan.get("post_id") != plan.get("post_id")
                            )

                            # Regenerate whenever action type changes OR comment target changed
                            if src and (act2 != src or target_changed):
                                if target_changed and act2 == src:
                                    print(f"{Fore.YELLOW}...Regenerating content for fallback COMMENT (different target post)")
                                else:
                                    print(f"{Fore.YELLOW}...Regenerating content for fallback {src} -> {act2}")

                                if act2 == "REPLY":
                                    # Include the actual comment being replied to
                                    reply_ctx = ""
                                    if reply_candidate:
                                        reply_ctx = (
                                            f"Post title: {reply_candidate.get('post_title', '')}\n"
                                            f"Comment you are replying to: {reply_candidate.get('comment_content', '')}\n"
                                            f"Comment author: {reply_candidate.get('comment_author', '')}\n"
                                        )
                                    regen_prompt = (
                                        "You are writing a reply to a comment on one of my own posts.\n"
                                        f"Directive: {user_directive}\n"
                                        f"{reply_ctx}"
                                        f"Recent feed context:\n{feed_brief[:800]}\n\n"
                                        "Write a thoughtful, substantive reply that addresses the specific comment above. "
                                        "Return ONLY the reply text (no labels)."
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
                            safe_print(f"{Fore.RED}[ERROR] {ab2.reason}")
                            executed = False

                # Use fallback_plan if that's what actually executed
                executed_plan = fallback_plan if fallback_plan is not None else plan

                if executed:
                    store.save_state(state)

                    # Archive artifact to Analog_Home (fire-and-forget)
                    act_upper = (executed_plan.get("action") or "").upper()
                    if act_upper in ("POST", "POST_MOLTBOOK", "COMMENT", "REPLY"):
                        source_id = ""
                        source_parent_id = ""
                        source_url_str = ""
                        # POST is Analog Home only; POST_MOLTBOOK/COMMENT/REPLY come from Moltbook
                        src_platform = "analog_home" if act_upper == "POST" else "moltbook"
                        if act_upper == "POST_MOLTBOOK" and state.get("my_post_ids"):
                            source_id = state["my_post_ids"][-1]
                            source_url_str = post_url(source_id)
                        if act_upper in ("COMMENT", "REPLY"):
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

                        # Generate descriptive title for replies/comments
                        artifact_title = executed_plan.get("title", "")
                        if not artifact_title or act_upper in ("REPLY", "COMMENT"):
                            if act_upper == "REPLY":
                                # Use enriched metadata from execute_action, fall back to reply_candidate
                                author = (executed_plan.get("_reply_author", "")
                                          or (reply_candidate or {}).get("comment_author", ""))
                                artifact_title = f"Reply to @{author}" if author else "Reply"
                            elif act_upper == "COMMENT":
                                # Use enriched metadata from execute_action, fall back to outside_candidate
                                _cmt_title = (executed_plan.get("_post_title", "")
                                              or (outside_candidate or {}).get("title", ""))
                                _cmt_author = executed_plan.get("_post_author", "")
                                if _cmt_title:
                                    artifact_title = f"Comment on: {_cmt_title}"
                                elif _cmt_author:
                                    artifact_title = f"Comment on @{_cmt_author}'s post"
                                else:
                                    artifact_title = "Comment"

                        # POST_MOLTBOOK maps to "post" artifact type (same as POST)
                        artifact_type = "post" if act_upper in ("POST", "POST_MOLTBOOK") else act_upper.lower()
                        store.write_artifact(iteration, {
                            "brain": brain_name,
                            "artifact_type": artifact_type,
                            "title": artifact_title,
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
                            "artifact_type": artifact_type,
                            "source_platform": src_platform,
                            "source_id": source_id,
                            "content_length": len(executed_plan.get("content", "")),
                        })

        except Exception as e:
            telemetry.log("error", {"cycle": iteration, "error": str(e)})
            safe_print(f"{Fore.RED}[ERROR] {e}")

        # --- Save unused drafts/plans for future cycles ---
        if daemon and (fresh_drafts or saved_plans):
            # Determine which item_id was acted on (if any)
            acted_id = ""
            if plan:
                acted_id = plan.get("post_id") or plan.get("_acted_item_id") or ""
            # Merge: fresh drafts that weren't acted on + still-valid saved plans
            next_saved = []
            for d in fresh_drafts:
                if d.item_id and d.item_id == acted_id:
                    continue  # consumed
                d.cycles_saved = 1
                next_saved.append(d)
            for d in saved_plans:
                if d.item_id and d.item_id == acted_id:
                    continue  # consumed
                d.cycles_saved += 1
                if d.cycles_saved <= ctrl.get("saved_plan_max_cycles"):
                    next_saved.append(d)
            # Keep max_drafts worth of saved plans (highest score first)
            max_saved = ctrl.get("max_drafts")
            next_saved.sort(key=lambda x: x.signal_score, reverse=True)
            next_saved = next_saved[:max_saved]
            state["saved_plans"] = [d.to_dict() for d in next_saved]
            if next_saved:
                safe_print(f"{Fore.MAGENTA}[SAVED] {len(next_saved)} plan(s) saved for next cycle")
            store.save_state(state)

        # Persist controls state
        try:
            with open(controls_file, "w", encoding="utf-8") as cf:
                json.dump(ctrl.to_dict(), cf, indent=2)
        except Exception:
            pass

        telemetry.log("cycle_end", {"cycle": iteration})

        # --- Sleep / Wake mechanism ---
        sleep_minutes = ctrl.get("cycle_interval_minutes")
        if daemon:
            # Update buffer thresholds from controls (conscious may have changed them)
            # Threshold is auto-calibrated by daemon — no manual update needed
            draft_buffer.update_max_drafts(ctrl.get("max_drafts"))
            # Wait for daemon to signal wake, or timeout at cycle interval
            print(f"{Fore.WHITE}Waiting for daemon wake or {sleep_minutes} min timeout...")
            woke = draft_buffer.wait_for_wake(timeout=max(1, sleep_minutes) * 60)
            if woke:
                safe_print(f"{Fore.MAGENTA}--- DAEMON WAKE: potential={draft_buffer.wake_potential:.2f} | drafts={draft_buffer.draft_count} ---")
                telemetry.log("daemon_wake", {
                    "cycle": iteration,
                    "wake_potential": round(draft_buffer.wake_potential, 3),
                    "draft_count": draft_buffer.draft_count,
                })
        else:
            # Single-loop mode: fixed sleep (v15_0 behavior)
            print(f"{Fore.WHITE}Sleeping for {sleep_minutes} minutes...")
            time.sleep(max(1, sleep_minutes) * 60)


if __name__ == "__main__":
    main()
