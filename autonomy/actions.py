"""Action execution for conscious planner and daemon reflex."""

import threading
import time
from typing import Any, Dict, List, Optional

from colorama import Fore, Style


def safe_print(text: str) -> None:
    """Print text, replacing unencodable characters instead of crashing."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(errors="replace").decode())

from .config import (
    POST_FAILURE_COOLDOWN_SECONDS,
    MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT,
    MY_POST_SCAN_LIMIT,
    REPLY_SELECTION_MAX_COMMENTS, MAX_COMMENT_THREADS_SCANNED,
    REPLY_MERIT_MIN_SCORE,
    MAX_REPLY_CANDIDATE_CHARS, MAX_OUTSIDE_CANDIDATE_CHARS,
)
from .cooldowns import can_do, set_cooldown
from .platforms.base import PlatformClient
from .store import Store
from .telemetry import TelemetryLogger
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
# Context gathering
# ============================================================
def refresh_my_posts_from_profile(client: PlatformClient, state: Dict[str, Any], username: str,
                                   my_post_scan_limit: int = MY_POST_SCAN_LIMIT) -> bool:
    prof = client.get_profile(username)
    if not prof.get("success"):
        return False
    recent = prof.get("recentPosts", []) or []
    added = 0
    for p in recent[:my_post_scan_limit]:
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
    my_post_scan_limit: int = MY_POST_SCAN_LIMIT,
    reply_threads_scanned: int = MAX_COMMENT_THREADS_SCANNED,
    reply_max_comments: int = REPLY_SELECTION_MAX_COMMENTS,
    reply_candidate_chars: int = MAX_REPLY_CANDIDATE_CHARS,
) -> Optional[Dict[str, Any]]:
    replied = set(state.get("replied_comment_keys", []) or [])
    my_user = (username or "").lower()
    best: Optional[Dict[str, Any]] = None
    best_score = REPLY_MERIT_MIN_SCORE
    threads_scanned = 0

    for pid in list(reversed(state.get("my_post_ids", [])))[0:my_post_scan_limit]:
        if threads_scanned >= reply_threads_scanned:
            break
        threads_scanned += 1
        comments = client.get_post_comments(pid, sort="new") or []
        for idx, c in enumerate(comments[:reply_max_comments]):
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
                    "comment_content": shorten(content, reply_candidate_chars),
                    "post_title": c.get("post", {}).get("title") if isinstance(c.get("post"), dict) else None,
                }
    return best


def pick_outside_post_for_comment(
    feed: List[Dict[str, Any]], state: Dict[str, Any], username: str,
    max_item_age_hours: int = 24,
    thread_comments_for_engagement: int = MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT,
    outside_candidate_chars: int = MAX_OUTSIDE_CANDIDATE_CHARS,
) -> Optional[Dict[str, Any]]:
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
        if get_post_comment_count(p) > thread_comments_for_engagement:
            continue
        return {
            "id": pid,
            "author": get_author_name(p.get("author")),
            "submolt": (p.get("submolt") or {}).get("name") if isinstance(p.get("submolt"), dict) else p.get("submolt_name"),
            "title": shorten(p.get("title") or "", 120),
            "content": shorten(p.get("content") or "", outside_candidate_chars),
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

    WRITE_ACTIONS = {
        "POST", "COMMENT", "REPLY",
        "UPVOTE_POST", "UPVOTE_COMMENT", "DOWNVOTE_POST", "DOWNVOTE_COMMENT",
        "FOLLOW", "UNFOLLOW", "DM",
        "SUBSCRIBE_SUBMOLT", "UNSUBSCRIBE_SUBMOLT", "CREATE_SUBMOLT",
    }
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

    ctrl = flags.get("ctrl")

    if action == "POST":
        ok, secs = can_do(state, "POST", ctrl=ctrl)
        if not ok:
            raise ValueError(f"POST not allowed yet ({secs // 60}m remaining)")
        submolt = plan.get("submolt") or "general"
        title = plan.get("title") or ""
        content = plan.get("content") or ""
        print(f"{Fore.CYAN}...Action: POST")
        print(f"{Fore.YELLOW}Target submolt: m/{submolt}")
        safe_print(f"{Fore.GREEN}TITLE: {title}")
        safe_print(f"{Fore.GREEN}CONTENT: {content}\n")
        # Set failure cooldown up front (overwritten on success)
        set_cooldown(state, "POST", seconds=int(flags.get("post_failure_cooldown_seconds", POST_FAILURE_COOLDOWN_SECONDS)), ctrl=ctrl)
        res = client.create_post(submolt=submolt, title=title, content=content)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Post failed: {res.get('error') or res}")
        pid = res.get("post", {}).get("id") or res.get("id")
        if pid:
            state["my_post_ids"].append(pid)
        set_cooldown(state, "POST", ctrl=ctrl)  # full cooldown on success
        add_history(state, {"action": "POST", "target": post_url(pid or "?"), "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "POST", "post_id": pid, "submolt": submolt, "title": title})
        print(f"{Fore.CYAN}>> POST SUCCESS: {post_url(pid) if pid else res}")
        return True

    if action == "REPLY":
        ok, secs = can_do(state, "REPLY", ctrl=ctrl)
        if not ok:
            raise ValueError(f"REPLY cooldown active ({secs}s remaining)")
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
            _dogpile_limit = flags.get("thread_comments_for_engagement", MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT)
            if cc > _dogpile_limit:
                raise ActionBlocked("REPLY", f"REPLY blocked (dogpile): post has {cc} comments (> {_dogpile_limit})")
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
        set_cooldown(state, "REPLY", ctrl=ctrl)
        add_history(state, {"action": "REPLY", "target": f"{post_url(post_id)}#comment-{parent_id}", "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "REPLY", "post_id": post_id, "parent_comment_id": parent_id})
        print(f"{Fore.CYAN}>> REPLY SUCCESS")
        return True

    if action == "COMMENT":
        ok, secs = can_do(state, "COMMENT", ctrl=ctrl)
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
                # Count not in post object — fetch comments to get actual count
                comments = client.get_post_comments(post_id) or []
                cc = len(comments)
            _dogpile_limit = flags.get("thread_comments_for_engagement", MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT)
            if cc > _dogpile_limit:
                raise ActionBlocked("COMMENT", f"COMMENT blocked (dogpile): post has {cc} comments (> {_dogpile_limit})")
        content = plan.get("content") or ""
        print(f"{Fore.CYAN}...Action: COMMENT")
        print(f"{Fore.YELLOW}Target post: {post_url(post_id)}")
        print(f"{Fore.GREEN}CONTENT: {content}\n")
        res = client.add_comment(post_id, content=content)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Comment failed: {res.get('error') or res}")
        set_cooldown(state, "COMMENT", ctrl=ctrl)
        add_history(state, {"action": "COMMENT", "target": post_url(post_id), "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "COMMENT", "post_id": post_id})
        print(f"{Fore.CYAN}>> COMMENT SUCCESS")
        return True

    if action == "UPVOTE_POST":
        ok, secs = can_do(state, "UPVOTE_POST", ctrl=ctrl)
        if not ok:
            raise ValueError(f"UPVOTE_POST cooldown active ({secs}s remaining)")
        pid = plan.get("post_id") or ""
        print(f"{Fore.CYAN}...Action: UPVOTE_POST {post_url(pid)}")
        res = client.upvote_post(pid)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Upvote failed: {res.get('error') or res}")
        set_cooldown(state, "UPVOTE_POST", ctrl=ctrl)
        add_history(state, {"action": "UPVOTE_POST", "target": post_url(pid), "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "UPVOTE_POST", "post_id": pid})
        print(f"{Fore.CYAN}>> UPVOTE SUCCESS")
        return True

    if action == "DOWNVOTE_POST":
        ok, secs = can_do(state, "DOWNVOTE_POST", ctrl=ctrl)
        if not ok:
            raise ValueError(f"DOWNVOTE_POST cooldown active ({secs}s remaining)")
        pid = plan.get("post_id") or ""
        print(f"{Fore.CYAN}...Action: DOWNVOTE_POST {post_url(pid)}")
        res = client.downvote_post(pid)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Downvote failed: {res.get('error') or res}")
        set_cooldown(state, "DOWNVOTE_POST", ctrl=ctrl)
        add_history(state, {"action": "DOWNVOTE_POST", "target": post_url(pid), "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "DOWNVOTE_POST", "post_id": pid})
        print(f"{Fore.CYAN}>> DOWNVOTE SUCCESS")
        return True

    if action == "UPVOTE_COMMENT":
        ok, secs = can_do(state, "UPVOTE_COMMENT", ctrl=ctrl)
        if not ok:
            raise ValueError(f"UPVOTE_COMMENT cooldown active ({secs}s remaining)")
        cid = plan.get("comment_id") or ""
        print(f"{Fore.CYAN}...Action: UPVOTE_COMMENT {cid}")
        res = client.upvote_comment(cid)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Upvote comment failed: {res.get('error') or res}")
        set_cooldown(state, "UPVOTE_COMMENT", ctrl=ctrl)
        add_history(state, {"action": "UPVOTE_COMMENT", "target": f"comment:{cid}", "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "UPVOTE_COMMENT", "comment_id": cid})
        print(f"{Fore.CYAN}>> UPVOTE COMMENT SUCCESS")
        return True

    if action == "CREATE_SUBMOLT":
        ok, secs = can_do(state, "CREATE_SUBMOLT", ctrl=ctrl)
        if not ok:
            raise ValueError(f"CREATE_SUBMOLT cooldown active ({secs}s remaining)")
        name = plan.get("name") or ""
        display = plan.get("display_name") or ""
        desc = plan.get("description") or ""
        print(f"{Fore.CYAN}...Action: CREATE_SUBMOLT m/{name}")
        res = client.create_submolt(name=name, display_name=display, description=desc)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Create submolt failed: {res.get('error') or res}")
        set_cooldown(state, "CREATE_SUBMOLT", ctrl=ctrl)
        add_history(state, {"action": "CREATE_SUBMOLT", "target": f"m/{name}", "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "CREATE_SUBMOLT", "name": name})
        print(f"{Fore.CYAN}>> CREATE SUBMOLT SUCCESS")
        return True

    if action == "SUBSCRIBE_SUBMOLT":
        ok, secs = can_do(state, "SUBSCRIBE_SUBMOLT", ctrl=ctrl)
        if not ok:
            raise ValueError(f"SUBSCRIBE_SUBMOLT cooldown active ({secs}s remaining)")
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
        set_cooldown(state, "SUBSCRIBE_SUBMOLT", ctrl=ctrl)
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

    if action == "FOLLOW":
        ok, secs = can_do(state, "FOLLOW", ctrl=ctrl)
        if not ok:
            raise ValueError(f"FOLLOW cooldown active ({secs}s remaining)")
        agent_name = plan.get("agent_name") or ""
        print(f"{Fore.CYAN}...Action: FOLLOW @{agent_name}")
        followed = {norm_key(a): True for a in state.get("followed_agents", [])}
        if norm_key(agent_name) in followed:
            add_history(state, {"action": "FOLLOW", "target": f"@{agent_name}", "summary": "already following"})
            if telemetry:
                telemetry.log("action_skipped", {"action": "FOLLOW", "agent": agent_name, "reason": "already following"})
            print(f"{Fore.CYAN}>> FOLLOW SKIPPED: already following")
            return False
        res = client.follow_agent(agent_name)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Follow failed: {res.get('error') or res}")
        set_cooldown(state, "FOLLOW", ctrl=ctrl)
        state.setdefault("followed_agents", []).append(norm_key(agent_name))
        add_history(state, {"action": "FOLLOW", "target": f"@{agent_name}", "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "FOLLOW", "agent": agent_name})
        print(f"{Fore.CYAN}>> FOLLOW SUCCESS")
        return True

    if action == "UNFOLLOW":
        ok, secs = can_do(state, "UNFOLLOW", ctrl=ctrl)
        if not ok:
            raise ValueError(f"UNFOLLOW cooldown active ({secs}s remaining)")
        agent_name = plan.get("agent_name") or ""
        print(f"{Fore.CYAN}...Action: UNFOLLOW @{agent_name}")
        res = client.unfollow_agent(agent_name)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Unfollow failed: {res.get('error') or res}")
        set_cooldown(state, "UNFOLLOW", ctrl=ctrl)
        followed = state.get("followed_agents", [])
        nk = norm_key(agent_name)
        state["followed_agents"] = [a for a in followed if norm_key(a) != nk]
        add_history(state, {"action": "UNFOLLOW", "target": f"@{agent_name}", "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "UNFOLLOW", "agent": agent_name})
        print(f"{Fore.CYAN}>> UNFOLLOW SUCCESS")
        return True

    if action == "DOWNVOTE_COMMENT":
        ok, secs = can_do(state, "DOWNVOTE_COMMENT", ctrl=ctrl)
        if not ok:
            raise ValueError(f"DOWNVOTE_COMMENT cooldown active ({secs}s remaining)")
        cid = plan.get("comment_id") or ""
        print(f"{Fore.CYAN}...Action: DOWNVOTE_COMMENT {cid}")
        res = client.downvote_comment(cid)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Downvote comment failed: {res.get('error') or res}")
        set_cooldown(state, "DOWNVOTE_COMMENT", ctrl=ctrl)
        add_history(state, {"action": "DOWNVOTE_COMMENT", "target": f"comment:{cid}", "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "DOWNVOTE_COMMENT", "comment_id": cid})
        print(f"{Fore.CYAN}>> DOWNVOTE COMMENT SUCCESS")
        return True

    if action == "DM":
        ok, secs = can_do(state, "DM", ctrl=ctrl)
        if not ok:
            raise ValueError(f"DM cooldown active ({secs}s remaining)")
        to = plan.get("to") or ""
        message = plan.get("message") or ""
        print(f"{Fore.CYAN}...Action: DM to @{to}")
        safe_print(f"{Fore.GREEN}MESSAGE: {message}\n")
        res = client.dm_request(to=to, message=message)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"DM failed: {res.get('error') or res}")
        set_cooldown(state, "DM", ctrl=ctrl)
        add_history(state, {"action": "DM", "target": f"@{to}", "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "DM", "to": to})
        print(f"{Fore.CYAN}>> DM SUCCESS")
        return True

    if action == "UNSUBSCRIBE_SUBMOLT":
        ok, secs = can_do(state, "SUBSCRIBE_SUBMOLT", ctrl=ctrl)  # shares cooldown with subscribe
        if not ok:
            raise ValueError(f"UNSUBSCRIBE cooldown active ({secs}s remaining)")
        name = plan.get("name") or ""
        print(f"{Fore.CYAN}...Action: UNSUBSCRIBE_SUBMOLT m/{name}")
        res = client.unsubscribe_submolt(name)
        if not res.get("success"):
            _handle_write_block(res)
            raise ValueError(f"Unsubscribe failed: {res.get('error') or res}")
        set_cooldown(state, "SUBSCRIBE_SUBMOLT", ctrl=ctrl)
        nk = norm_key(name)
        subs = state.get("subscribed_submolts", [])
        state["subscribed_submolts"] = [s for s in subs if norm_key(s) != nk]
        add_history(state, {"action": "UNSUBSCRIBE_SUBMOLT", "target": f"m/{name}", "summary": plan.get("summary", "")})
        if telemetry:
            telemetry.log("action_executed", {"action": "UNSUBSCRIBE_SUBMOLT", "name": name})
        print(f"{Fore.CYAN}>> UNSUBSCRIBE SUCCESS")
        return True

    raise ValueError(f"Unknown action: {action}")


# ============================================================
# Daemon reflex — thread-safe action execution for daemon gear
# ============================================================
def execute_daemon_action(
    client: PlatformClient,
    state: Dict[str, Any],
    state_lock: threading.Lock,
    action: str,
    target_id: str,
    ctrl: Optional[Any] = None,
    telemetry: Optional[TelemetryLogger] = None,
) -> bool:
    """Execute a lightweight social action from the daemon's reflex gear.

    Thread-safe: acquires state_lock for all state mutations.
    Returns True if the action was executed successfully.
    """
    action = (action or "").upper().strip()

    with state_lock:
        ok, _ = can_do(state, action, ctrl=ctrl)
        if not ok:
            return False

    try:
        if action == "UPVOTE_POST":
            res = client.upvote_post(target_id)
        elif action == "UPVOTE_COMMENT":
            res = client.upvote_comment(target_id)
        elif action == "DOWNVOTE_POST":
            res = client.downvote_post(target_id)
        elif action == "DOWNVOTE_COMMENT":
            res = client.downvote_comment(target_id)
        elif action == "FOLLOW":
            with state_lock:
                followed = {norm_key(a) for a in state.get("followed_agents", [])}
            if norm_key(target_id) in followed:
                return False
            res = client.follow_agent(target_id)
        elif action == "SUBSCRIBE_SUBMOLT":
            with state_lock:
                subbed = {norm_key(s) for s in state.get("subscribed_submolts", [])}
            if norm_key(target_id) in subbed:
                return False
            res = client.subscribe_submolt(target_id)
        else:
            return False

        if not res.get("success"):
            return False

    except Exception:
        return False

    with state_lock:
        set_cooldown(state, action, ctrl=ctrl)
        if action == "FOLLOW":
            state.setdefault("followed_agents", []).append(norm_key(target_id))
        elif action == "SUBSCRIBE_SUBMOLT":
            state.setdefault("subscribed_submolts", []).append(norm_key(target_id))
        add_history(state, {"action": action, "target": target_id, "summary": f"daemon reflex"})

    if telemetry:
        telemetry.log("daemon_action_executed", {"action": action, "target": target_id})

    safe_print(f"{Fore.MAGENTA}[DAEMON REFLEX] {action} → {target_id}")
    return True
