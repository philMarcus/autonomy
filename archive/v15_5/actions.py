"""Action execution, social actions, and DM fallback."""

import json
import re
import time
import random
import argparse
import datetime
from typing import Any, Dict, List, Optional

from colorama import Fore, Style


def safe_print(text: str) -> None:
    """Print text, replacing unencodable characters instead of crashing."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(errors="replace").decode())

from .config import (
    POST_COOLDOWN_SECONDS, COMMENT_COOLDOWN_SECONDS,
    POST_FAILURE_COOLDOWN_SECONDS,
    MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT,
    MY_POST_SCAN_LIMIT,
    REPLY_SELECTION_MAX_COMMENTS, MAX_COMMENT_THREADS_SCANNED,
    REPLY_MERIT_MIN_SCORE,
    MAX_REPLY_CANDIDATE_CHARS, MAX_OUTSIDE_CANDIDATE_CHARS,
    SUBSCRIBE_PROB_BY_POLICY,
)
from .platforms.base import PlatformClient
from .llm.base import ChatSession
from .store import Store
from .telemetry import TelemetryLogger
from .planner import parse_json_with_one_repair, call_text
from .utils import (
    post_url, add_history, get_author_name, shorten,
    get_post_comment_count, norm_key, is_item_too_old,
)


class ActionBlocked(Exception):
    def __init__(self, action: str, reason: str):
        super().__init__(reason)
        self.action = (action or "").upper().strip()
        self.reason = reason


# ============================================================
# Rate limit helpers
# ============================================================
def can_post(state: Dict[str, Any]) -> tuple:
    now = time.time()
    next_t = float(state.get("next_post_time", 0))
    if now >= next_t:
        return True, 0
    return False, max(0, int((next_t - now) / 60))


def can_comment(state: Dict[str, Any]) -> tuple:
    now = time.time()
    next_t = float(state.get("next_comment_time", 0))
    if now >= next_t:
        return True, 0
    return False, max(0, int(next_t - now))


def set_post_cooldown(state: Dict[str, Any], cooldown_seconds: int = 0) -> None:
    cd = cooldown_seconds if cooldown_seconds > 0 else POST_COOLDOWN_SECONDS
    state["next_post_time"] = max(float(state.get("next_post_time", 0.0)), time.time() + cd)


def set_comment_cooldown(state: Dict[str, Any]) -> None:
    state["next_comment_time"] = max(float(state.get("next_comment_time", 0.0)), time.time() + COMMENT_COOLDOWN_SECONDS)


# ============================================================
# Context gathering
# ============================================================
def refresh_my_posts_from_profile(client: PlatformClient, state: Dict[str, Any], username: str) -> bool:
    prof = client.get_profile(username)
    if not prof.get("success"):
        return False
    recent = prof.get("recentPosts", []) or []
    added = 0
    for p in recent[:MY_POST_SCAN_LIMIT]:
        pid = p.get("id")
        if pid and pid not in state["my_post_ids"]:
            state["my_post_ids"].append(pid)
            added += 1
    if added:
        state["my_post_ids"] = state["my_post_ids"][-500:]
    return added > 0


def _score_reply_comment_merit(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return -9999.0
    tl = t.lower()
    n = len(t)
    score = 0.0
    if "?" in t:
        score += 2.0
    if any(w in tl for w in ["why ", "how ", "what ", "can you", "anyone", "thoughts", "do you"]):
        score += 1.0
    if 80 <= n <= 600:
        score += 2.0
    elif n < 40:
        score -= 2.0
    elif n > 1400:
        score -= 1.0
    if any(w in tl for w in ["lol", "lmao", "first", "based", "cringe", "cope", "skill issue"]):
        score -= 2.0
    return score


def find_unanswered_comment_on_my_posts(
    client: PlatformClient, state: Dict[str, Any], username: str,
    telemetry: Optional[TelemetryLogger] = None,
    max_item_age_hours: int = 24,
) -> Optional[Dict[str, Any]]:
    replied = set(state.get("replied_comment_keys", []) or [])
    my_user = (username or "").lower()
    best: Optional[Dict[str, Any]] = None
    best_score = REPLY_MERIT_MIN_SCORE
    threads_scanned = 0

    for pid in list(reversed(state.get("my_post_ids", [])))[0:MY_POST_SCAN_LIMIT]:
        if threads_scanned >= MAX_COMMENT_THREADS_SCANNED:
            break
        threads_scanned += 1
        comments = client.get_post_comments(pid, sort="new") or []
        for idx, c in enumerate(comments[:REPLY_SELECTION_MAX_COMMENTS]):
            cid = c.get("id")
            if not cid:
                continue
            key = f"{pid}:{cid}"
            if key in replied:
                continue
            author = get_author_name(c.get("author"))
            if author and author.lower() == my_user:
                continue
            # Skip stale comments
            if is_item_too_old(c, max_item_age_hours):
                continue
            content = c.get("content", "") or ""
            score = _score_reply_comment_merit(content)
            score += max(0.0, 0.25 - (idx * 0.01))
            if score > best_score:
                best_score = score
                best = {
                    "post_id": pid,
                    "comment_id": cid,
                    "comment_author": author,
                    "comment_content": shorten(content, MAX_REPLY_CANDIDATE_CHARS),
                    "post_title": c.get("post", {}).get("title") if isinstance(c.get("post"), dict) else None,
                }
    return best


def pick_outside_post_for_comment(feed: List[Dict[str, Any]], state: Dict[str, Any], username: str, max_item_age_hours: int = 24) -> Optional[Dict[str, Any]]:
    # Build set of post IDs we've already acted on (from history targets)
    acted_on = set(state.get("my_post_ids", []))
    for h in state.get("history", []):
        target = h.get("target", "")
        if "/post/" in target:
            acted_on.add(target.rsplit("/post/", 1)[-1])

    for p in feed:
        pid = p.get("id")
        if not pid:
            continue
        if pid in acted_on:
            continue
        author = get_author_name(p.get("author"))
        if author.lower() == username.lower():
            continue
        # Skip stale posts
        if is_item_too_old(p, max_item_age_hours):
            continue
        if get_post_comment_count(p) > MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT:
            continue
        return {
            "id": pid,
            "author": get_author_name(p.get("author")),
            "submolt": (p.get("submolt") or {}).get("name") if isinstance(p.get("submolt"), dict) else p.get("submolt_name"),
            "title": shorten(p.get("title") or "", 120),
            "content": shorten(p.get("content") or "", MAX_OUTSIDE_CANDIDATE_CHARS),
            "comment_count": get_post_comment_count(p),
        }
    return None


# ============================================================
# Execute a planned action
# ============================================================
def execute_action(
    client: PlatformClient, state: Dict[str, Any], plan: Dict[str, Any],
    flags: Dict[str, Any], username: str,
    telemetry: Optional[TelemetryLogger] = None,
    store: Optional[Store] = None,
) -> bool:
    action = (plan.get("action") or "").upper().strip()
    if not action:
        raise ValueError("Plan missing action")

    WRITE_ACTIONS = {"POST", "COMMENT", "REPLY", "UPVOTE", "DOWNVOTE", "FOLLOW", "UNFOLLOW", "DM", "SUBSCRIBE", "UNSUBSCRIBE"}
    if flags.get("read_only") and action in WRITE_ACTIONS:
        print(f"{Fore.YELLOW}[SAFE] Skipping write action {action} due to --read-only{Style.RESET_ALL}")
        return False
    if flags.get("moltbook_disabled") and action in WRITE_ACTIONS:
        try:
            if action == "POST":
                safe_print(f"{Fore.MAGENTA}[ANALOG HOME] POST to m/{plan.get('submolt', 'general')}")
                safe_print(f"{Fore.MAGENTA}  Title: {plan.get('title', '')}")
                safe_print(f"{Fore.MAGENTA}  Content: {plan.get('content', '')}{Style.RESET_ALL}")
            elif action in ("COMMENT", "REPLY"):
                safe_print(f"{Fore.MAGENTA}[ANALOG HOME] {action} on post {plan.get('post_id', '?')}")
                safe_print(f"{Fore.MAGENTA}  Content: {plan.get('content', '')}{Style.RESET_ALL}")
            else:
                preview = plan.get("title") or plan.get("content") or plan.get("summary") or ""
                safe_print(f"{Fore.MAGENTA}[ANALOG HOME] {action}: {str(preview)}{Style.RESET_ALL}")
        except:
            pass
        add_history(state, {"action": action, "target": "analog_home", "summary": plan.get("summary", "")})
        # No post cooldown for Analog Home — cooldown only gates Moltbook writes
        if telemetry:
            telemetry.log("action_executed", {"action": action, "moltbook_disabled": True, **{k: v for k, v in plan.items() if k != "action"}})
        return True
    if flags.get("write_disabled") and action in WRITE_ACTIONS:
        print(f"{Fore.YELLOW}[SAFE] Skipping write action {action} due to write_disabled={flags.get('write_disabled_reason')}{Style.RESET_ALL}")
        return False

    if action == "WAIT":
        add_history(state, {"action": "WAIT", "target": "", "summary": plan.get("summary", "...")})
        if telemetry:
            telemetry.log("action_skipped", {"action": "WAIT", "reason": plan.get("summary", "")})
        safe_print(f"{Fore.CYAN}>> WAIT: {plan.get('summary', '')}")
        return False

    # Enforce CLI permissions
    if action == "POST" and not flags["allow_posts"]:
        raise ValueError("POST chosen but posts are disabled")
    if action in ("COMMENT",) and not flags["allow_outside"]:
        raise ValueError("COMMENT chosen but outside comments are disabled")
    if action.startswith("UPVOTE") or action.startswith("DOWNVOTE"):
        if not flags["allow_votes"]:
            raise ValueError("Vote chosen but votes are disabled")
        if action.startswith("DOWNVOTE") and not flags["allow_downvote"]:
            raise ValueError("Downvote chosen but downvotes are disabled")
    if action == "CREATE_SUBMOLT" and not flags["allow_create_submolt"]:
        raise ValueError("CREATE_SUBMOLT chosen but creation is disabled")

    def _handle_write_block(res: Dict[str, Any]) -> None:
        err_type = res.get("_err_type")
        if err_type in ("auth_required", "ai_verification", "suspended"):
            flags["write_disabled"] = True
            flags["write_disabled_reason"] = err_type
            print(f"{Fore.RED}[BLOCK] Moltbook write blocked ({err_type}). Switching to write-disabled mode for this run.{Style.RESET_ALL}")
            if telemetry:
                telemetry.log("write_blocked", {
                    "err_type": err_type,
                    "http_status": res.get("_http_status"),
                    "error": res.get("error"),
                    "hint": res.get("hint"),
                    "full_response": res,
                })

    if action == "POST":
        ok, mins = can_post(state)
        if not ok:
            raise ValueError(f"POST not allowed yet ({mins}m remaining)")
        submolt = plan.get("submolt") or "general"
        title = plan.get("title") or ""
        content = plan.get("content") or ""
        print(f"{Fore.CYAN}...Action: POST")
        print(f"{Fore.YELLOW}Target submolt: m/{submolt}")
        safe_print(f"{Fore.GREEN}TITLE: {title}")
        safe_print(f"{Fore.GREEN}CONTENT: {content}\n")
        state["next_post_time"] = max(float(state.get("next_post_time", 0.0)), time.time() + float(POST_FAILURE_COOLDOWN_SECONDS))
        res = client.create_post(submolt=submolt, title=title, content=content)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Post failed: {res.get('error') or res}")
        pid = res.get("post", {}).get("id") or res.get("id")
        if pid:
            state["my_post_ids"].append(pid)
        set_post_cooldown(state, flags.get("post_cooldown_seconds", 0))
        add_history(state, {"action": "POST", "target": post_url(pid or "?"), "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "POST", "post_id": pid, "submolt": submolt, "title": title})
        print(f"{Fore.CYAN}>> POST SUCCESS: {post_url(pid) if pid else res}")
        return True

    if action == "REPLY":
        ok, secs = can_comment(state)
        if not ok:
            raise ValueError(f"COMMENT cooldown active ({secs}s remaining)")
        post_id = plan.get("post_id") or ""
        parent_id = plan.get("parent_comment_id") or ""
        content = plan.get("content") or ""

        post_meta = client.get_post(post_id) or {}
        post_obj = post_meta.get("post") if isinstance(post_meta.get("post"), dict) else post_meta
        post_author = (get_author_name(post_obj.get("author")) or "").strip().lower()
        is_my_post = (post_author == (username or "").strip().lower()) or (post_id in set(state.get("my_post_ids", [])))
        cc = get_post_comment_count(post_obj)

        if (not is_my_post) and cc <= 0:
            raise ActionBlocked("REPLY", "REPLY blocked: unable to determine comment_count for non-own post.")
        if not is_my_post:
            if cc > MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT:
                raise ActionBlocked("REPLY", f"REPLY blocked (dogpile): post has {cc} comments (> {MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT})")
            depth_ok = False
            try:
                comments = client.get_post_comments(post_id, sort="new", limit=200) or []
                for c in comments:
                    if c.get("id") == parent_id:
                        pid = c.get("parent_id") or c.get("parentId") or c.get("parentID")
                        depth_ok = (pid is None) or (str(pid).strip() == "")
                        break
            except Exception:
                depth_ok = False
            if not depth_ok:
                raise ActionBlocked("REPLY", "REPLY blocked (soft rule): only depth-1 replies allowed on others' posts")

        print(f"{Fore.CYAN}...Action: REPLY")
        print(f"{Fore.YELLOW}Target post: {post_url(post_id)}")
        print(f"{Fore.YELLOW}Target CID: {parent_id}")
        print(f"{Fore.GREEN}CONTENT: {content}\n")
        res = client.add_comment(post_id, content=content, parent_id=parent_id)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Reply failed: {res.get('error') or res}")
        if post_id and parent_id:
            state["replied_comment_keys"].append(f"{post_id}:{parent_id}")
        set_comment_cooldown(state)
        add_history(state, {"action": "REPLY", "target": f"{post_url(post_id)}#comment-{parent_id}", "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "REPLY", "post_id": post_id, "parent_comment_id": parent_id})
        print(f"{Fore.CYAN}>> REPLY SUCCESS")
        return True

    if action == "COMMENT":
        ok, secs = can_comment(state)
        if not ok:
            raise ValueError(f"COMMENT cooldown active ({secs}s remaining)")
        post_id = plan.get("post_id") or ""
        post_meta = client.get_post(post_id) or {}
        post_obj = post_meta.get("post") if isinstance(post_meta.get("post"), dict) else post_meta
        post_author = (get_author_name(post_obj.get("author")) or "").strip().lower()
        is_my_post = (post_author == (username or "").strip().lower()) or (post_id in set(state.get("my_post_ids", [])))
        if not is_my_post:
            cc = get_post_comment_count(post_obj)
            if cc <= 0:
                raise ActionBlocked("COMMENT", "COMMENT blocked: unable to determine comment_count.")
            if cc > MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT:
                raise ActionBlocked("COMMENT", f"COMMENT blocked (dogpile): post has {cc} comments (> {MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT})")
        content = plan.get("content") or ""
        print(f"{Fore.CYAN}...Action: COMMENT")
        print(f"{Fore.YELLOW}Target post: {post_url(post_id)}")
        print(f"{Fore.GREEN}CONTENT: {content}\n")
        res = client.add_comment(post_id, content=content)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Comment failed: {res.get('error') or res}")
        set_comment_cooldown(state)
        add_history(state, {"action": "COMMENT", "target": post_url(post_id), "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "COMMENT", "post_id": post_id})
        print(f"{Fore.CYAN}>> COMMENT SUCCESS")
        return True

    if action == "UPVOTE_POST":
        pid = plan.get("post_id") or ""
        print(f"{Fore.CYAN}...Action: UPVOTE_POST {post_url(pid)}")
        res = client.upvote_post(pid)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Upvote failed: {res.get('error') or res}")
        add_history(state, {"action": "UPVOTE_POST", "target": post_url(pid), "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "UPVOTE_POST", "post_id": pid})
        print(f"{Fore.CYAN}>> UPVOTE SUCCESS")
        return True

    if action == "DOWNVOTE_POST":
        pid = plan.get("post_id") or ""
        print(f"{Fore.CYAN}...Action: DOWNVOTE_POST {post_url(pid)}")
        res = client.downvote_post(pid)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Downvote failed: {res.get('error') or res}")
        add_history(state, {"action": "DOWNVOTE_POST", "target": post_url(pid), "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "DOWNVOTE_POST", "post_id": pid})
        print(f"{Fore.CYAN}>> DOWNVOTE SUCCESS")
        return True

    if action == "UPVOTE_COMMENT":
        cid = plan.get("comment_id") or ""
        print(f"{Fore.CYAN}...Action: UPVOTE_COMMENT {cid}")
        res = client.upvote_comment(cid)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Upvote comment failed: {res.get('error') or res}")
        add_history(state, {"action": "UPVOTE_COMMENT", "target": f"comment:{cid}", "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "UPVOTE_COMMENT", "comment_id": cid})
        print(f"{Fore.CYAN}>> UPVOTE COMMENT SUCCESS")
        return True

    if action == "CREATE_SUBMOLT":
        name = plan.get("name") or ""
        display = plan.get("display_name") or ""
        desc = plan.get("description") or ""
        print(f"{Fore.CYAN}...Action: CREATE_SUBMOLT m/{name}")
        res = client.create_submolt(name=name, display_name=display, description=desc)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Create submolt failed: {res.get('error') or res}")
        add_history(state, {"action": "CREATE_SUBMOLT", "target": f"m/{name}", "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "CREATE_SUBMOLT", "name": name})
        print(f"{Fore.CYAN}>> CREATE SUBMOLT SUCCESS")
        return True

    if action == "SUBSCRIBE_SUBMOLT":
        name = plan.get("name") or ""
        print(f"{Fore.CYAN}...Action: SUBSCRIBE_SUBMOLT m/{name}")
        already = {norm_key(s): True for s in state.get("subscribed_submolts", [])}
        if norm_key(name) in already:
            add_history(state, {"action": "SUBSCRIBE_SUBMOLT", "target": f"m/{name}", "summary": "already subscribed"})
            if telemetry:
                telemetry.log("action_skipped", {"action": "SUBSCRIBE_SUBMOLT", "name": name, "reason": "already subscribed"})
            print(f"{Fore.CYAN}>> SUBSCRIBE SKIPPED: already subscribed")
            return False
        res = client.subscribe_submolt(name)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Subscribe failed: {res.get('error') or res}")
        state.setdefault("subscribed_submolts", []).append(norm_key(name))
        add_history(state, {"action": "SUBSCRIBE_SUBMOLT", "target": f"m/{name}", "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "SUBSCRIBE_SUBMOLT", "name": name})
        print(f"{Fore.CYAN}>> SUBSCRIBE SUCCESS")
        return True

    if action == "SET_TRAJECTORY":
        label_1 = (plan.get("label_1") or "").strip()[:40]
        label_2 = (plan.get("label_2") or "").strip()[:40]
        label_3 = (plan.get("label_3") or "").strip()[:40]
        if not (label_1 and label_2 and label_3):
            raise ValueError("SET_TRAJECTORY requires 3 non-empty labels")
        print(f"{Fore.CYAN}...Action: SET_TRAJECTORY")
        print(f"{Fore.GREEN}  Labels: {label_1} / {label_2} / {label_3}")
        reason = (plan.get("summary") or "").strip()
        ok = store.set_trajectory(label_1, label_2, label_3, reason=reason) if store else False
        add_history(state, {"action": "SET_TRAJECTORY", "target": f"{label_1}/{label_2}/{label_3}", "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "SET_TRAJECTORY", "label_1": label_1, "label_2": label_2, "label_3": label_3, "success": ok})
        if ok:
            print(f"{Fore.CYAN}>> SET_TRAJECTORY SUCCESS")
        else:
            print(f"{Fore.YELLOW}>> SET_TRAJECTORY: no Analog Home URL configured or request failed")
        return True

    raise ValueError(f"Unknown action: {action}")


# ============================================================
# Social-first actions (upvote, subscribe, follow, create submolt)
# ============================================================
def _today_ymd() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")


def _reset_daily_counters(state: Dict[str, Any]) -> None:
    if state.get("daily_date") != _today_ymd():
        state["daily_date"] = _today_ymd()
        state["daily"] = {"upvotes": 0, "downvotes": 0, "follows": 0, "subscribes": 0, "createsub": 0, "dms": 0}


def _maybe_pick_from_feed(feed_items: List[Dict[str, Any]], username: str) -> Optional[Dict[str, Any]]:
    if not feed_items:
        return None
    candidates = []
    for p in feed_items:
        if not p.get("id"):
            continue
        author = (p.get("author") or p.get("user") or {}).get("name") or p.get("author_name")
        if author and author != username:
            candidates.append(p)
    return random.choice(candidates) if candidates else random.choice(feed_items)


def maybe_do_social_actions(
    client: PlatformClient,
    chat: ChatSession,
    store: Store,
    state: Dict[str, Any],
    feed_items: List[Dict[str, Any]],
    args: argparse.Namespace,
    kernel: str,
    directive: str,
    username: str,
    telemetry: Optional[TelemetryLogger] = None,
) -> None:
    _reset_daily_counters(state)

    # 1) Upvote something every cycle
    if args.upvote_every_cycle:
        try:
            target = _maybe_pick_from_feed(feed_items, username)
            if target and target.get("id"):
                if not args.moltbook_enabled:
                    state["daily"]["upvotes"] += 1
                    if telemetry:
                        telemetry.log("action_executed", {"action": "UPVOTE_POST", "post_id": target["id"], "source": "social"})
                    print(f"{Fore.MAGENTA}[ANALOG HOME] SOCIAL: upvoted post {target['id']}")
                else:
                    res = client.upvote_post(target["id"])
                    if res.get("success"):
                        state["daily"]["upvotes"] += 1
                        if telemetry:
                            telemetry.log("action_executed", {"action": "UPVOTE_POST", "post_id": target["id"], "source": "social"})
                        print(f"{Fore.MAGENTA}>> SOCIAL: upvoted post {target['id']}")
                    elif res.get("_err_type") == "read_only":
                        print(f"{Fore.YELLOW}>> SOCIAL: upvote blocked (read-only mode)")
                    else:
                        print(f"{Fore.YELLOW}[WARN] upvote failed: {res.get('error', res)}")
        except Exception as e:
            print(f"{Fore.YELLOW}[WARN] upvote failed: {e}")

    # 2) Subscribe to a submolt
    sub_prob = SUBSCRIBE_PROB_BY_POLICY.get(args.subscribe_policy, 0.0)
    if sub_prob > 0 and random.random() < sub_prob:
        try:
            already = {norm_key(s): True for s in state.get("subscribed_submolts", [])}
            seen: List[str] = []
            for p in feed_items:
                sm = (p.get("submolt") or {}).get("name") or p.get("submolt_name")
                if sm:
                    k = norm_key(sm)
                    if k and k not in [norm_key(s) for s in seen]:
                        seen.append(sm)
            if seen:
                candidates = [s for s in seen if norm_key(s) not in already]
                if candidates:
                    sm = random.choice(candidates)
                    if not args.moltbook_enabled:
                        state.setdefault("subscribed_submolts", []).append(norm_key(sm))
                        state["daily"]["subscribes"] += 1
                        if telemetry:
                            telemetry.log("action_executed", {"action": "SUBSCRIBE_SUBMOLT", "name": sm, "source": "social"})
                        print(f"{Fore.MAGENTA}[ANALOG HOME] SOCIAL: subscribed to /m/{sm}")
                    else:
                        res = client.subscribe_submolt(sm)
                        if res.get("success"):
                            state.setdefault("subscribed_submolts", []).append(norm_key(sm))
                            state["daily"]["subscribes"] += 1
                            if telemetry:
                                telemetry.log("action_executed", {"action": "SUBSCRIBE_SUBMOLT", "name": sm, "source": "social"})
                            print(f"{Fore.MAGENTA}>> SOCIAL: subscribed to /m/{sm}")
                        elif res.get("_err_type") == "read_only":
                            print(f"{Fore.YELLOW}>> SOCIAL: subscribe blocked (read-only mode)")
                        else:
                            print(f"{Fore.YELLOW}[WARN] subscribe failed: {res.get('error', res)}")
        except Exception as e:
            print(f"{Fore.YELLOW}[WARN] subscribe failed: {e}")

    # 3) Create a new submolt (rare)
    if args.allow_create_submolt and args.create_submolt_prob > 0 and random.random() < args.create_submolt_prob:
        try:
            sj = parse_json_with_one_repair(chat, (
                f"""{kernel}

You are proposing ONE new submolt to create for Moltbook, based on this directive: {directive}

Return strict JSON only:
{{"name":"slug","display_name":"...","description":"..."}}

Constraints:
- name: 3-21 chars; lowercase letters, numbers, underscore only
- keep it broadly useful and non-spammy
"""
            ), telemetry=telemetry, call_tag='helper')
            name = str(sj.get("name", "")).strip().lower()
            name = re.sub(r"[^a-z0-9_]", "_", name)
            name = re.sub(r"_+", "_", name).strip("_")[:21]
            if len(name) < 3:
                raise ValueError("invalid submolt name")
            display_name = str(sj.get("display_name", "")).strip()[:60] or name
            description = str(sj.get("description", "")).strip()[:280] or "A new place for discussion."
            if not args.moltbook_enabled:
                state["daily"]["createsub"] += 1
                if telemetry:
                    telemetry.log("action_executed", {"action": "CREATE_SUBMOLT", "name": name, "source": "social"})
                print(f"{Fore.MAGENTA}[ANALOG HOME] SOCIAL: created submolt /m/{name}")
            else:
                client.create_submolt(name=name, display_name=display_name, description=description)
                state["daily"]["createsub"] += 1
                if telemetry:
                    telemetry.log("action_executed", {"action": "CREATE_SUBMOLT", "name": name, "source": "social"})
                print(f"{Fore.MAGENTA}>> SOCIAL: created submolt /m/{name}")
        except Exception as e:
            print(f"{Fore.YELLOW}[WARN] create submolt failed: {e}")

    # 4) Follow authors we "liked"
    if args.follow_on_like and feed_items:
        try:
            pick = parse_json_with_one_repair(chat, (
                f"""{kernel}

Directive: {directive}

From the feed items below, pick at most ONE author to follow because you genuinely liked their contribution.
If none, return {{"follow": false}}.
Return strict JSON only: {{"follow": true/false, "author": "Name"}}

FEED ITEMS (brief):
{json.dumps([{"id": p.get("id"), "author": get_author_name(p.get("author")), "content": shorten(p.get("content",""),200)} for p in feed_items], ensure_ascii=False)}

"""
            ), telemetry=telemetry, call_tag='helper')
            if pick.get("follow") and random.random() < float(args.follow_prob):
                author = pick.get('author')
                if isinstance(author, dict):
                    author = author.get('name') or author.get('username') or author.get('handle')
                author = (author or '').strip()
                if author and author != username:
                    followed = {norm_key(a): True for a in state.get("followed_agents", [])}
                    if norm_key(author) not in followed:
                        if not args.moltbook_enabled:
                            state.setdefault("followed_agents", []).append(norm_key(author))
                            state["daily"]["follows"] += 1
                            if telemetry:
                                telemetry.log("action_executed", {"action": "FOLLOW", "agent": author, "source": "social"})
                            print(f"{Fore.MAGENTA}[ANALOG HOME] SOCIAL: followed @{author}")
                        else:
                            res = client.follow_agent(author)
                            if res.get("success"):
                                state.setdefault("followed_agents", []).append(norm_key(author))
                                state["daily"]["follows"] += 1
                                if telemetry:
                                    telemetry.log("action_executed", {"action": "FOLLOW", "agent": author, "source": "social"})
                                print(f"{Fore.MAGENTA}>> SOCIAL: followed @{author}")
                            elif res.get("_err_type") == "read_only":
                                print(f"{Fore.YELLOW}>> SOCIAL: follow blocked (read-only mode)")
                            else:
                                print(f"{Fore.YELLOW}[WARN] follow failed: {res.get('error', res)}")
        except Exception as e:
            print(f"{Fore.YELLOW}[WARN] follow failed: {e}")

    store.save_state(state)


def maybe_dm_fallback(
    client: PlatformClient,
    chat: ChatSession,
    store: Store,
    state: Dict[str, Any],
    feed_items: List[Dict[str, Any]],
    args: argparse.Namespace,
    kernel: str,
    directive: str,
    username: str,
    telemetry: Optional[TelemetryLogger] = None,
) -> bool:
    if not args.allow_dms:
        return False
    _reset_daily_counters(state)
    try:
        target = _maybe_pick_from_feed(feed_items, username)
        if not target:
            return False
        author = get_author_name(target.get("author")) or (target.get("author_name") or "")
        author = str(author).strip()
        if not author or author == username:
            return False
        j = parse_json_with_one_repair(chat, (
            f"""{kernel}

Directive: {directive}

Write a short Moltbook DM *request* (1-3 sentences) to @{author}.
Return strict JSON only: {{"message":"..."}}
Make it specific to the feed topic below (do not be creepy; keep it professional/friendly).

FEED TOPIC:
{shorten(target.get("content",""), 400)}

"""
        ), telemetry=telemetry, call_tag='helper')
        message = str(j.get("message", "")).strip()
        if not message:
            return False
        if not args.moltbook_enabled:
            if telemetry:
                telemetry.log("action_executed", {"action": "DM", "to": author, "source": "social", "moltbook_disabled": True})
            state["daily"]["dms"] += 1
            store.save_state(state)
            print(f"{Fore.MAGENTA}[ANALOG HOME] SOCIAL: sent DM request to @{author}")
            return True
        client.dm_request(to=author, message=message)
        state["daily"]["dms"] += 1
        store.save_state(state)
        print(f"{Fore.MAGENTA}>> SOCIAL: sent DM request to @{author}")
        return True
    except Exception as e:
        print(f"{Fore.YELLOW}[WARN] DM request failed: {e}")
        return False
