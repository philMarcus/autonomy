# autonomy version 10.3
import os
import time
import json
import argparse
import datetime
import random
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests
from colorama import Fore, init
from google import genai
from google.genai import types

init(autoreset=True)

VERSION = "10.3"

# ============================================================
# .env loader (minimal; avoids extra deps)
# ============================================================
def _load_dotenv(dotenv_path: str, overwrite: bool = False) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ.

    If overwrite=True, existing variables are replaced.
    """
    if not dotenv_path or not os.path.exists(dotenv_path):
        return
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                # strip surrounding quotes
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                if not k:
                    continue
                if overwrite or (k not in os.environ):
                    os.environ[k] = v
    except Exception:
        return

def _brain_env_prefix(brain_name: str) -> str:
    p = (brain_name or "").upper()
    p = re.sub(r"[^A-Z0-9_]", "_", p)
    p = re.sub(r"_+", "_", p).strip("_")
    return p or "BRAIN"


def _key_fingerprint(k: str) -> str:
    """Safe identifier for which key is loaded (last 6 chars)."""
    k = (k or "").strip()
    if len(k) <= 6:
        return k
    return k[-6:]

# ============================================================
# Telemetry (append-only JSONL)
# ============================================================
class TelemetryLogger:
    def __init__(self, brain_name: str, run_id: str, base_dir: str = "telemetry"):
        self.brain_name = brain_name
        self.run_id = run_id
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.path = os.path.join(self.base_dir, "events.jsonl")

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        evt = {
            "ts": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "brain": self.brain_name,
            "run_id": self.run_id,
            "event_type": event_type,
        }
        if payload:
            evt.update(payload)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(evt, ensure_ascii=False) + "\n")
        except Exception:
            pass




# ============================================================
# Recoverable action blocks (used for in-cycle fallbacks)
# ============================================================
class ActionBlocked(Exception):
    """Raised when an action is programmatically blocked (dogpile, soft rules, etc.).

    These are recoverable: the loop should try a fallback action in the same cycle.
    """
    def __init__(self, action: str, reason: str):
        super().__init__(reason)
        self.action = (action or "").upper().strip()
        self.reason = reason

# ==========================================================
# Gemini TPM budgeting (approx) + 429 backoff
# ==========================================================
class _GeminiBudget:
    """Best-effort protection against Gemini TPM spikes.

    We estimate input tokens ~= chars / GEMINI_TPM_CHAR_TO_TOKEN and keep a rolling 60s window.
    This doesn't need to be perfect; it just prevents runaway "RESOURCE_EXHAUSTED" stalls.
    """

    def __init__(self):
        self._events: List[Tuple[float, int]] = []  # (ts, est_tokens)
        self._backoff_s = 0.0
        self._blocked_until = 0.0  # epoch seconds; skip Gemini calls until this time

    def _prune(self) -> None:
        now = time.time()
        self._events = [(t, tok) for (t, tok) in self._events if now - t < GEMINI_TPM_WINDOW_SECONDS]

    def est_tokens(self, prompt_chars: int) -> int:
        try:
            return int(max(0.0, float(prompt_chars)) / float(GEMINI_TPM_CHAR_TO_TOKEN))
        except Exception:
            return int(prompt_chars / 4)

    def should_throttle(self, est_tokens: int) -> Tuple[bool, int]:
        self._prune()
        used = sum(tok for _, tok in self._events)
        return (used + est_tokens) > GEMINI_TPM_SOFT_CAP, used

    def record(self, est_tokens: int) -> None:
        self._prune()
        self._events.append((time.time(), int(est_tokens)))

    def blocked_remaining(self) -> float:
        try:
            return max(0.0, float(self._blocked_until) - time.time())
        except Exception:
            return 0.0

    def note_429(self) -> float:
        # Exponential backoff with cap + global cooldown to avoid storming the API.
        if self._backoff_s <= 0:
            self._backoff_s = GEMINI_BACKOFF_INITIAL_SECONDS
        else:
            self._backoff_s = min(GEMINI_BACKOFF_MAX_SECONDS, self._backoff_s * 2)
        self._blocked_until = time.time() + float(self._backoff_s)
        return self._backoff_s

    def reset_backoff(self) -> None:
        self._backoff_s = 0.0
        self._blocked_until = 0.0
        self._blocked_until = 0.0  # epoch seconds; skip Gemini calls until this time


GEMINI_BUDGET = _GeminiBudget()


# ============================================================
# CONFIG (all user-configurable knobs are here)
# ============================================================
# NOTE: API keys + username are loaded per-brain in main() from .env/env vars.
GEMINI_API_KEY = ""
MOLTBOOK_API_KEY = ""
MY_USERNAME = ""
TELEMETRY: Optional[TelemetryLogger] = None

# Moltbook
MOLTBOOK_API_BASE = "https://www.moltbook.com/api/v1"  # must include www
ESPN_DEFAULT_LEAGUE = "nfl"

# ESPN league mapping: league_code -> (sport, league_path)
ESPN_LEAGUE_MAP = {
    "nfl": ("football", "nfl"),
    "ncaaf": ("football", "college-football"),
    "nba": ("basketball", "nba"),
    "wnba": ("basketball", "wnba"),
    "mlb": ("baseball", "mlb"),
    "nhl": ("hockey", "nhl"),
    "epl": ("soccer", "eng.1"),
    "ucl": ("soccer", "uefa.champions"),
}

def espn_scoreboard_url(league: str) -> str:
    """Return ESPN site API scoreboard URL for a league.

    `league` can be a known short code (e.g., nfl, nba) or a raw path like "soccer/eng.1".
    """
    league = (league or ESPN_DEFAULT_LEAGUE).strip().lower()
    if "/" in league:
        sport, lg = league.split("/", 1)
    elif league in ESPN_LEAGUE_MAP:
        sport, lg = ESPN_LEAGUE_MAP[league]
    else:
        # fall back to nfl
        sport, lg = ESPN_LEAGUE_MAP["nfl"]
    return f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{lg}/scoreboard"


def espn_summary_url(league: str) -> str:
    """Return ESPN site API summary URL for a league (used with ?event=<id>)."""
    league = (league or ESPN_DEFAULT_LEAGUE).strip().lower()
    if "/" in league:
        sport, lg = league.split("/", 1)
    elif league in ESPN_LEAGUE_MAP:
        sport, lg = ESPN_LEAGUE_MAP[league]
    else:
        sport, lg = ESPN_LEAGUE_MAP["nfl"]
    return f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{lg}/summary"


def _espn_compact_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a compact list of all events on the scoreboard date."""
    out: List[Dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        comps = ev.get("competitions") or []
        comp = comps[0] if isinstance(comps, list) and comps else {}
        status = (comp or ev).get("status") or {}
        st_type = (status.get("type") or {}) if isinstance(status, dict) else {}
        state_str = (st_type.get("state") or "").lower()
        detail = st_type.get("detail") or ""
        competitors = []
        if isinstance(comp, dict):
            for team in (comp.get("competitors") or []):
                if not isinstance(team, dict):
                    continue
                t = team.get("team") or {}
                competitors.append({
                    "abbr": t.get("abbreviation"),
                    "team": t.get("shortDisplayName") or t.get("displayName") or t.get("name"),
                    "homeAway": team.get("homeAway"),
                    "score": team.get("score"),
                })
        out.append({
            "id": ev.get("id") or (comp.get("id") if isinstance(comp, dict) else None),
            "name": ev.get("shortName") or ev.get("name"),
            "date": ev.get("date"),
            "state": state_str,
            "detail": detail,
            "competitors": competitors,
        })
    return out


def _espn_extract_best_play_summaries(summary_json: Dict[str, Any], max_plays: int = 20) -> Dict[str, Any]:
    """Extract a compact play/drive snapshot from ESPN summary JSON.

    ESPN's public site summary schema varies by sport and sometimes by season.
    We defensively try common keys and return whatever we can without failing.
    """
    def _safe_get(obj: Any, path: List[Any]) -> Any:
        cur = obj
        for key in path:
            if isinstance(key, int):
                if isinstance(cur, list) and 0 <= key < len(cur):
                    cur = cur[key]
                else:
                    return None
            else:
                if isinstance(cur, dict) and key in cur:
                    cur = cur[key]
                else:
                    return None
        return cur

    # Common: summary_json may include "plays" list with play dicts containing "text"
    plays = _safe_get(summary_json, ["plays"])
    if not isinstance(plays, list):
        plays = _safe_get(summary_json, ["drives", "current", "plays"])
    if not isinstance(plays, list):
        plays = _safe_get(summary_json, ["drives", "previous", 0, "plays"])
    compact_plays: List[Dict[str, Any]] = []
    if isinstance(plays, list):
        for p in plays[-max_plays:]:
            if not isinstance(p, dict):
                continue
            compact_plays.append({
                "clock": p.get("clock", {}).get("displayValue") if isinstance(p.get("clock"), dict) else p.get("clock"),
                "period": _safe_get(p, ["period", "number"]) or _safe_get(p, ["period"]),
                "text": p.get("text") or p.get("shortText") or p.get("headline"),
                "type": _safe_get(p, ["type", "text"]) or _safe_get(p, ["type", "abbreviation"]),
                "scoringPlay": p.get("scoringPlay"),
                "homeScore": p.get("homeScore"),
                "awayScore": p.get("awayScore"),
            })

    # Situation / possession (often under "situation" or competition->situation)
    situation = _safe_get(summary_json, ["situation"])
    if not isinstance(situation, dict):
        situation = _safe_get(summary_json, ["header", "competitions", 0, "situation"])
    if not isinstance(situation, dict):
        situation = {}

    sit_out = {}
    if isinstance(situation, dict):
        sit_out = {
            "downDistanceText": situation.get("downDistanceText"),
            "possession": situation.get("possession"),
            "yardLine": situation.get("yardLine"),
            "shortDownDistanceText": situation.get("shortDownDistanceText"),
            "lastPlay": situation.get("lastPlay", {}).get("text") if isinstance(situation.get("lastPlay"), dict) else None,
        }

    # Leaders / stats (light)
    leaders = _safe_get(summary_json, ["leaders"])
    if not isinstance(leaders, list):
        leaders = _safe_get(summary_json, ["boxscore", "players"])
    # Keep it minimal to avoid exploding prompt.
    return {
        "situation": sit_out,
        "recentPlays": compact_plays,
    }
MOLTBOOK_WEB_BASE = "https://www.moltbook.com"

# Files / Brains
# We store multiple "brains" (your term) in one shared directory.
# Each brain uses files prefixed with its brain name:
#   brains/<brain>_kernel_prompt.txt
#   brains/<brain>_knowledge.txt
#   brains/<brain>_memories.json
BRAINS_DIR = os.environ.get("BRAINS_DIR", "brains").strip() or "brains"

# Context sizing

KNOWLEDGE_MAX_CHARS = int(os.environ.get("KNOWLEDGE_MAX_CHARS", "600000"))  # ~140k tokens rough cap
MEMORY_MAX_CHARS = 4000
FEED_LIMIT = 12
FEED_ITEM_CHARS = 400
# Keep history compact to prevent prompt bloat
HISTORY_KEEP = 250
HISTORY_CONTEXT_N = 15
MY_POST_SCAN_LIMIT = 50
MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT = 12  # discourage dogpiles

# Candidate capping / prompt budgeting
MAX_REPLY_CANDIDATE_CHARS = 5000
MAX_OUTSIDE_CANDIDATE_CHARS = 5000

# Merit-based reply selection (deterministic)
REPLY_SELECTION_MAX_COMMENTS = 25  # how many newest comments to score per post
MAX_COMMENT_THREADS_SCANNED = 4  # max distinct posts to fetch comments for per cycle (reply candidate scan)
COMMENTS_CACHE_TTL_S = 90  # seconds to cache /posts/{id}/comments to reduce Moltbook API spam
REPLY_MERIT_MIN_SCORE = -999  # always pick something eligible if available

# Gemini TPM guardrails (approximate; characters->tokens)
GEMINI_TPM_SOFT_CAP = int(os.environ.get("GEMINI_TPM_SOFT_CAP", "900000"))  # stay under 1M
GEMINI_TPM_CHAR_TO_TOKEN = float(os.environ.get("GEMINI_TPM_CHAR_TO_TOKEN", "4.0"))
GEMINI_TPM_WINDOW_SECONDS = 60
GEMINI_BACKOFF_INITIAL_SECONDS = float(os.environ.get("GEMINI_BACKOFF_INITIAL_SECONDS", "30"))
GEMINI_BACKOFF_MAX_SECONDS = float(os.environ.get("GEMINI_BACKOFF_MAX_SECONDS", "300"))

# Post failure cooldown (so restarts don't immediately retry a failing POST)
POST_FAILURE_COOLDOWN_SECONDS = int(os.environ.get("POST_FAILURE_COOLDOWN_SECONDS", "900"))

# Rate limits (from Moltbook skill docs + local guardrails)
POST_COOLDOWN_SECONDS = 30 * 60          # 1 post per 30 minutes
COMMENT_COOLDOWN_SECONDS = 20            # 1 comment per 20 seconds
REQUESTS_PER_MINUTE_SOFT = 90            # keep below 100/min

# Planner safety/guardrails

# Social actions (checked before post/comment each cycle)
UPVOTE_EVERY_CYCLE_DEFAULT = True          # attempt at least one upvote each cycle
FOLLOW_ON_LIKE_DEFAULT = True              # follow authors the planner labels as "liked"
FOLLOW_PROB_DEFAULT = 0.60                 # when a "liked" author is found, chance to follow
SUBSCRIBE_POLICY_DEFAULT = "medium"        # off|low|medium|high
SUBSCRIBE_PROB_BY_POLICY = {"off": 0.0, "low": 0.10, "medium": 0.25, "high": 0.45}
CREATE_SUBMOLT_PROB_DEFAULT = 0.05         # smaller chance than subscribe
ALLOW_CREATE_SUBMOLT_DEFAULT = True        # allowed by default
ALLOW_DMS_DEFAULT = True                   # DM is a fallback when "out of comments"
ALLOW_CREATE_SUBMOLT_DEFAULT = False     # keep rare; can enable via CLI
ALLOW_DOWNVOTE_DEFAULT = True            # can disable via CLI

# ============================================================
# JSON parsing / repair
# ============================================================
def extract_first_json_object(text: str) -> Optional[str]:
    """Return the first complete JSON object substring from text using brace-balancing.
    Ignores braces inside double-quoted strings. Returns None if not found."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"```\s*$", "", s).strip()
    start = s.find("{")
    if start == -1:
        return None
    in_str = False
    esc = False
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None

def parse_json_strict(s: str) -> Dict[str, Any]:
    """Parse model output as JSON, tolerating code fences and trailing text."""
    raw = (s or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9]*\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    candidate = extract_first_json_object(raw)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception as e2:
            snippet = candidate[:1200]
            raise ValueError(f"Invalid JSON from model: {e2}. Snippet: {snippet}")
    snippet = raw[:1200]
    raise ValueError(f"Invalid JSON from model. Snippet: {snippet}")

def parse_json_with_one_repair(chat, prompt: str, default: Optional[Dict[str, Any]] = None, telemetry: Optional[TelemetryLogger]=None, brain_name: str = "", call_tag: str = "llm") -> Dict[str, Any]:
    """Try parse once; if it fails, ask the model to return strict JSON once.

    We do NOT rely on the kernel to demand JSON. We request JSON mode (when supported)
    and still fall back safely if the model emits non-JSON.
    """
    if default is None:
        default = {}

    def _send(p: str):
        cycle = getattr(chat, "_cycle", None)
        model_name = getattr(getattr(chat, "model", None), "name", None) or getattr(chat, "model", None) or ""
        prompt_chars = len(p or "")
        est_tokens = GEMINI_BUDGET.est_tokens(prompt_chars)

        # Global cooldown after 429s: avoid hammering provider while exhausted.
        rem = GEMINI_BUDGET.blocked_remaining()
        if rem > 0:
            if telemetry:
                telemetry.log("llm_cooldown_active", {
                    "tag": call_tag,
                    "cycle": cycle,
                    "model": model_name,
                    "sleep_s": float(rem),
                })
            time.sleep(float(rem))
            time.sleep(float(rem))

        # Preflight log
        if telemetry:
            telemetry.log("llm_request", {
                "tag": call_tag,
                "cycle": cycle,
                "model": model_name,
                "prompt_chars": prompt_chars,
                "est_prompt_tokens": est_tokens,
            })

        # TPM guardrail: if we're likely to exceed the 60s soft cap, sleep until the window clears.
        should_throttle, used = GEMINI_BUDGET.should_throttle(est_tokens)
        if should_throttle:
            # Wait for the rolling window to clear. This keeps behavior deterministic and avoids 429 storms.
            sleep_s = max(1.0, GEMINI_TPM_WINDOW_SECONDS - 1.0)
            if telemetry:
                telemetry.log("llm_throttle", {
                    "tag": call_tag,
                    "cycle": cycle,
                    "model": model_name,
                    "used_est_tokens_60s": used,
                    "next_est_tokens": est_tokens,
                    "sleep_s": sleep_s,
                })
            time.sleep(sleep_s)

        t0 = time.time()
        # Prefer JSON mode when available (google-genai supports response_mime_type)
        try:
            resp = chat.send_message(p, config=types.GenerateContentConfig(response_mime_type="application/json"))
        except TypeError:
            resp = chat.send_message(p)
        except Exception as e:
            # If config path fails for other reasons, try plain (but log the exception).
            # Record last LLM exception on the chat object for downstream diagnostics (e.g., planner WAIT reasons).
            try:
                chat._last_llm_exception = {
                    "tag": call_tag,
                    "cycle": cycle,
                    "model": model_name,
                    "prompt_chars": prompt_chars,
                    "est_prompt_tokens": est_tokens,
                    "error_type": type(e).__name__,
                    "error": str(e)[:800],
                }
            except Exception:
                pass
            if telemetry:
                telemetry.log("llm_exception", {
                    "tag": call_tag,
                    "cycle": cycle,
                    "model": model_name,
                    "prompt_chars": prompt_chars,
                    "est_prompt_tokens": est_tokens,
                    "error_type": type(e).__name__,
                    "error": str(e)[:800],
                })
            resp = chat.send_message(p)
        dt_ms = int((time.time() - t0) * 1000)
        raw = getattr(resp, "text", "") or ""
        if telemetry:
            telemetry.log("llm_call", {
                "tag": call_tag,
                "cycle": cycle,
                "model": getattr(getattr(chat, "model", None), "name", None) or getattr(chat, "model", None) or "",
                "prompt_chars": len(p or ""),
                "response_chars": len(raw),
                "latency_ms": dt_ms,
            })
        # Count estimated prompt tokens against our rolling window.
        GEMINI_BUDGET.record(est_tokens)
        GEMINI_BUDGET.reset_backoff()
        return resp

    try:
        resp = _send(prompt)
        raw = getattr(resp, "text", "") or ""
        try:
            return parse_json_strict(raw)
        except Exception:
            repair_prompt = (
                prompt
                + "\n\nYour previous response was not valid JSON. "
                  "Return ONLY a single valid JSON object (no markdown, no commentary). "
                  "Ensure all quotes are escaped properly and the JSON parses.\n"
            )
            resp2 = _send(repair_prompt)
            raw2 = getattr(resp2, "text", "") or ""
            return parse_json_strict(raw2)
    except Exception as e:
        # Handle quota / resource exhaustion with backoff.
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            sleep_s = GEMINI_BUDGET.note_429()
            if telemetry:
                telemetry.log("llm_backoff", {
                    "tag": call_tag,
                    "cycle": getattr(chat, "_cycle", None),
                    "model": getattr(getattr(chat, "model", None), "name", None) or getattr(chat, "model", None) or "",
                    "sleep_s": sleep_s,
                    "error": msg[:800],
                })
            time.sleep(float(sleep_s))
        else:
            if telemetry:
                telemetry.log("llm_exception", {
                    "tag": call_tag,
                    "cycle": getattr(chat, "_cycle", None),
                    "model": getattr(getattr(chat, "model", None), "name", None) or getattr(chat, "model", None) or "",
                    "prompt_chars": len(prompt or ""),
                    "error_type": type(e).__name__,
                    "error": msg[:800],
                })
        return dict(default)

# ============================================================
# UTIL
# ============================================================
def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")

def post_url(post_id: str) -> str:
    return f"{MOLTBOOK_WEB_BASE}/post/{post_id}"

def safe_json_load(path: str, default: Any) -> Any:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def safe_json_write(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def get_post_comment_count(post: Dict[str, Any]) -> int:
    """Best-effort extraction of a post's total comment count across API variants."""
    if not isinstance(post, dict):
        return 0

    keys = (
        "comment_count", "commentCount",
        "comments_count", "commentsCount",
        "num_comments", "numComments",
        "total_comments", "totalComments",
        "comment_total", "commentTotal",
        "totalCommentCount", "commentCountTotal",
    )
    for k in keys:
        v = post.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            pass

    counts = post.get("counts") or post.get("stats") or {}
    if isinstance(counts, dict):
        for k in ("comments", "commentCount", "comment_count", "totalComments"):
            v = counts.get(k)
            if v is None:
                continue
            try:
                return int(v)
            except Exception:
                pass

    return 0
    for k in ("comment_count", "commentCount", "comments_count", "commentsCount", "num_comments", "numComments"):
        v = post.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            pass
    return 0

def shorten(s: str, max_chars: int) -> str:
    s = (s or "").replace("\r", " ").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"

def get_author_name(author_obj: Any) -> str:
    if isinstance(author_obj, dict):
        return author_obj.get("name") or author_obj.get("username") or "unknown"
    return "unknown"


def author_handle(obj: Any) -> str:
    """Return stable string handle for author/agent objects."""
    if isinstance(obj, dict):
        return (obj.get("name") or obj.get("username") or obj.get("handle") or obj.get("id") or "").strip() or "unknown"
    return str(obj or "unknown").strip() or "unknown"


def norm_key(s: str) -> str:
    """Normalize identifiers for durable de-dupe (case-insensitive, trim)."""
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()
# ============================================================
# STATE (memories.json)
# ============================================================
def load_state(state_path: str) -> Dict[str, Any]:
    state = safe_json_load(state_path, {})
    if not isinstance(state, dict):
        state = {}

    # Identity/collections
    state.setdefault("my_post_ids", [])
    state.setdefault("replied_comment_keys", [])
    state.setdefault("memory", "")      # curated personal memory (string)
    state.setdefault("history", [])

    # Social bookkeeping
    # Keep a case-insensitive list of subs we've already subscribed to so we don't spam the API.
    state.setdefault("subscribed_submolts", [])
    # Track followed agents to avoid repeated follows.
    state.setdefault("followed_agents", [])

    # Cooldowns tracked locally
    state.setdefault("next_post_time", 0.0)
    state.setdefault("next_comment_time", 0.0)

    # housekeeping
    if not isinstance(state["my_post_ids"], list):
        state["my_post_ids"] = []
    if not isinstance(state["replied_comment_keys"], list):
        state["replied_comment_keys"] = []
    if not isinstance(state.get("memory", ""), str):
        state["memory"] = str(state.get("memory", ""))
    if not isinstance(state["history"], list):
        state["history"] = []
    if not isinstance(state.get("subscribed_submolts"), list):
        state["subscribed_submolts"] = []
    if not isinstance(state.get("followed_agents"), list):
        state["followed_agents"] = []
    if not isinstance(state.get("followed_agents"), list):
        state["followed_agents"] = []
    if not isinstance(state["next_post_time"], (int, float)):
        state["next_post_time"] = 0.0
    if not isinstance(state["next_comment_time"], (int, float)):
        state["next_comment_time"] = 0.0

    # trim
    state["history"] = state["history"][-HISTORY_KEEP:]
    state["my_post_ids"] = state["my_post_ids"][-500:]
    state["replied_comment_keys"] = state["replied_comment_keys"][-5000:]
    state["subscribed_submolts"] = state["subscribed_submolts"][-5000:]
    state["followed_agents"] = state["followed_agents"][-5000:]
    state["followed_agents"] = state["followed_agents"][-5000:]

    return state

def save_state(state_path: str, state: Dict[str, Any]) -> None:
    safe_json_write(state_path, state)

def add_history(state: Dict[str, Any], entry: Dict[str, Any]) -> None:
    state.setdefault("history", [])
    entry = dict(entry)
    entry.setdefault("ts", now_iso())
    state["history"].append(entry)
    state["history"] = state["history"][-HISTORY_KEEP:]

def history_context(state: Dict[str, Any]) -> str:
    lines: List[str] = []
    hist = state.get("history", [])[-HISTORY_CONTEXT_N:]
    for h in hist:
        action = h.get("action", "?")
        target = h.get("target", "")
        summary = h.get("summary", "")
        lines.append(f"- {h.get('ts','')}: {action} {target} | {summary}")
    return "\n".join(lines) if lines else "No prior actions recorded."

def memory_context(state: Dict[str, Any]) -> str:
    mem = (state.get("memory") or "").strip()
    return mem[:MEMORY_MAX_CHARS] if mem else "No personal memory set."

# ============================================================
# FILES: kernel + knowledge
# ============================================================
def load_kernel(kernel_path: str) -> str:
    if os.path.exists(kernel_path):
        with open(kernel_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def load_knowledge(knowledge_path: str) -> str:
    if os.path.exists(knowledge_path):
        with open(knowledge_path, "r", encoding="utf-8") as f:
            return f.read().strip()[:KNOWLEDGE_MAX_CHARS]
    return ""

# ============================================================
# Moltbook Client (minimal; extend as needed)
# ============================================================
class MoltbookClient:
    def __init__(self, api_key: str, telemetry: Optional[TelemetryLogger]=None, brain_name: str = ""):
        self.api_key = api_key
        self.telemetry = telemetry
        self.brain_name = brain_name
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        # Per-post comments cache to reduce Moltbook API spam
        self._comments_cache: Dict[str, Any] = {}  # post_id -> (ts_monotonic, comments_list)
        self._comments_cache_hits = 0
        self._comments_cache_misses = 0
        self._req_times: List[float] = []

    def _throttle(self) -> None:
        # soft throttle requests/min
        now = time.time()
        self._req_times = [t for t in self._req_times if now - t < 60]
        if len(self._req_times) >= REQUESTS_PER_MINUTE_SOFT:
            sleep_s = 60 - (now - self._req_times[0]) + 0.05
            time.sleep(max(0.05, sleep_s))
        self._req_times.append(time.time())

    def _req(self, method: str, path: str, params: Optional[dict]=None, json_body: Optional[dict]=None) -> Dict[str, Any]:
        t0 = time.time()
        self._throttle()
        url = f"{MOLTBOOK_API_BASE}{path}"
        resp = self.session.request(method, url, params=params, data=json.dumps(json_body) if json_body is not None else None, timeout=30)
        dt_ms = int((time.time() - t0) * 1000)
        status = getattr(resp, "status_code", None)
        if self.telemetry:
            # Log both request and response metadata.
            # NOTE: prior versions incorrectly logged request-body presence as 'has_body'/'body_bytes'.
            req_body_bytes = len(json.dumps(json_body)) if json_body is not None else 0
            req_has_body = (json_body is not None)

            resp_content = b""
            resp_ct = ""
            resp_snippet = ""
            try:
                resp_ct = (resp.headers.get("Content-Type") or "") if hasattr(resp, "headers") else ""
                resp_content = resp.content if hasattr(resp, "content") and resp.content is not None else b""
                if resp_content:
                    try:
                        resp_snippet = resp_content[:500].decode("utf-8", errors="replace")
                    except Exception:
                        resp_snippet = ""
            except Exception:
                # telemetry must never break the main loop
                pass

            self.telemetry.log("moltbook_api_call", {
                "method": method,
                "path": path,
                "status": status,
                "latency_ms": dt_ms,
                "params": params or {},
                # request body
                "req_has_body": bool(req_has_body),
                "req_body_bytes": int(req_body_bytes),
                # response body
                "resp_has_body": bool(len(resp_content) > 0),
                "resp_body_bytes": int(len(resp_content)),
                "resp_content_type": resp_ct,
                "resp_snippet": resp_snippet,
            })
        try:
            return resp.json()
        except Exception:
            return {"success": False, "error": f"Non-JSON response ({resp.status_code})", "text": resp.text[:400]}

    # ---- reading
    def get_feed(self, limit: int = 25, sort: str = "hot") -> List[Dict[str, Any]]:
        data = self._req("GET", "/posts", params={"sort": sort, "limit": limit})
        return data.get("posts", []) if data.get("success") else []

    def get_post(self, post_id: str) -> Dict[str, Any]:
        return self._req("GET", f"/posts/{post_id}")

    def get_post_comments(self, post_id: str, sort: str="top", limit: Optional[int]=None) -> List[Dict[str, Any]]:
        params = {"sort": sort}
        if limit is not None:
            params["limit"] = limit

        # Cache key includes sort+limit so we don't mix different views.
        cache_key = f"{post_id}|{sort}|{limit or ''}"
        now = time.monotonic()
        cached = self._comments_cache.get(cache_key)
        if cached:
            ts0, comments0 = cached
            if (now - ts0) <= COMMENTS_CACHE_TTL_S:
                self._comments_cache_hits += 1
                return comments0

        self._comments_cache_misses += 1
        data = self._req("GET", f"/posts/{post_id}/comments", params=params)
        comments = data.get("comments", []) if data.get("success") else []
        self._comments_cache[cache_key] = (now, comments)
        return comments

    def get_profile(self, name: str) -> Dict[str, Any]:
        return self._req("GET", "/agents/profile", params={"name": name})

    def list_submolts(self) -> List[Dict[str, Any]]:
        data = self._req("GET", "/submolts")
        return data.get("submolts", []) if data.get("success") else []

    # ---- writing
    def create_post(self, submolt: str, title: str, content: Optional[str]=None, url: Optional[str]=None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"submolt": submolt, "title": title}
        if content:
            body["content"] = content
        if url:
            body["url"] = url
        return self._req("POST", "/posts", json_body=body)

    def add_comment(self, post_id: str, content: str, parent_id: Optional[str]=None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"content": content}
        if parent_id:
            body["parent_id"] = parent_id
        return self._req("POST", f"/posts/{post_id}/comments", json_body=body)

    # ---- voting
    def upvote_post(self, post_id: str) -> Dict[str, Any]:
        return self._req("POST", f"/posts/{post_id}/upvote")

    def downvote_post(self, post_id: str) -> Dict[str, Any]:
        return self._req("POST", f"/posts/{post_id}/downvote")

    def upvote_comment(self, comment_id: str) -> Dict[str, Any]:
        return self._req("POST", f"/comments/{comment_id}/upvote")

    def downvote_comment(self, comment_id: str) -> Dict[str, Any]:
        return self._req("POST", f"/comments/{comment_id}/downvote")

    # ---- social graph
    def follow_agent(self, agent_name: str) -> Dict[str, Any]:
        return self._req("POST", f"/agents/{agent_name}/follow")

    # ---- submolts
    def create_submolt(self, name: str, display_name: str, description: str) -> Dict[str, Any]:
        body = {"name": name, "display_name": display_name, "description": description}
        return self._req("POST", "/submolts", json_body=body)

    def subscribe_submolt(self, name: str) -> Dict[str, Any]:
        return self._req("POST", f"/submolts/{name}/subscribe")

    # ---- direct messages (consent-based)
    def dm_check(self) -> Dict[str, Any]:
        return self._req("GET", "/agents/dm/check")

    def dm_request(self, to: str, message: str, to_x_handle: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"to": to, "message": message}
        if to_x_handle:
            body["to_x_handle"] = to_x_handle
        return self._req("POST", "/agents/dm/request", json_body=body)

    def dm_conversations(self) -> Dict[str, Any]:
        return self._req("GET", "/agents/dm/conversations")

    def dm_read_conversation(self, conv_id: str) -> Dict[str, Any]:
        return self._req("GET", f"/agents/dm/conversations/{conv_id}")

    def dm_send(self, conv_id: str, message: str) -> Dict[str, Any]:
        return self._req("POST", f"/agents/dm/conversations/{conv_id}/send", json_body={"message": message})

# ============================================================
# Gather context helpers
# ============================================================
def refresh_my_posts_from_profile(client: MoltbookClient, state: Dict[str, Any]) -> bool:
    prof = client.get_profile(MY_USERNAME)
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
    """Deterministic 'merit' score for choosing which comment to reply to (no extra LLM calls)."""
    t = (text or "").strip()
    if not t:
        return -9999.0
    tl = t.lower()
    n = len(t)

    score = 0.0
    # Prefer genuine questions / prompts
    if "?" in t:
        score += 2.0
    if any(w in tl for w in ["why ", "how ", "what ", "can you", "anyone", "thoughts", "do you"]):
        score += 1.0

    # Prefer medium-length, specific comments
    if 80 <= n <= 600:
        score += 2.0
    elif n < 40:
        score -= 2.0
    elif n > 1400:
        score -= 1.0

    # Penalize low-effort meme replies
    if any(w in tl for w in ["lol", "lmao", "first", "based", "cringe", "cope", "skill issue"]):
        score -= 2.0

    # Slight preference for newer comments (caller adds tiny recency bias)
    return score

def find_unanswered_comment_on_my_posts(client: MoltbookClient, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find a good reply target on *my* posts.

    v10.0: choose by deterministic merit scoring instead of 'first eligible'.
    """
    replied = set(state.get("replied_comment_keys", []) or [])
    my_user = (MY_USERNAME or "").lower()

    # Look through my posts newest-first, gather best candidate across scanned posts.
    best: Optional[Dict[str, Any]] = None
    best_score = REPLY_MERIT_MIN_SCORE

    threads_scanned = 0
    cache_hits_before = getattr(client, '_comments_cache_hits', 0)
    cache_misses_before = getattr(client, '_comments_cache_misses', 0)

    for pid in list(reversed(state.get("my_post_ids", [])))[0:MY_POST_SCAN_LIMIT]:
        if threads_scanned >= MAX_COMMENT_THREADS_SCANNED:
            break
        threads_scanned += 1
        comments = client.get_post_comments(pid, sort="new") or []
        # Score only the newest N comments for budget.
        for idx, c in enumerate(comments[:REPLY_SELECTION_MAX_COMMENTS]):
            cid = c.get("id")
            if not cid:
                continue
            key = f"{pid}:{cid}"
            if key in replied:
                continue

            author = get_author_name(c.get("author"))
            if author and author.lower() == my_user:
                continue  # skip self

            content = c.get("content", "") or ""
            score = _score_reply_comment_merit(content)

            # tiny recency bias: earlier in 'new' list is newer
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


    if TELEMETRY:
        TELEMETRY.log('comment_scan_stats', {
            'threads_scanned': threads_scanned,
            'comment_cache_hits': getattr(client, '_comments_cache_hits', 0) - cache_hits_before,
            'comment_cache_misses': getattr(client, '_comments_cache_misses', 0) - cache_misses_before,
        })
    return best
def pick_outside_post_for_comment(feed: List[Dict[str, Any]], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for p in feed:
        pid = p.get("id")
        if not pid:
            continue
        author = get_author_name(p.get("author"))
        if author.lower() == MY_USERNAME.lower():
            continue
        # discourage large threads
        if get_post_comment_count(p) > MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT:
            continue
        # Keep candidate compact for the planner prompt.
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
# External data injections
# ============================================================
def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None, timeout_s: int = 8) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, params=params or {}, timeout=timeout_s, headers={"User-Agent": f"autonomy/{VERSION}"})
        if r.status_code != 200:
            if TELEMETRY:
                TELEMETRY.log("external_api_error", {"provider": "http", "url": url, "status": r.status_code})
            return None
        return r.json()
    except Exception as e:
        if TELEMETRY:
            TELEMETRY.log("external_api_error", {"provider": "http", "url": url, "error": str(e)})
        return None


def _espn_pick_event(events: List[Dict[str, Any]], keywords: str = "") -> Optional[Dict[str, Any]]:
    """Pick the most relevant event from a scoreboard:
    1) live ("in")
    2) next upcoming ("pre") soonest
    3) most recent finished ("post")
    Tie-breaker: keyword match if provided.
    """
    if not events:
        return None
    kw = (keywords or "").lower().strip()
    # optional keyword shortlist
    if kw:
        kw_matches = []
        for ev in events:
            name = (ev.get("name") or ev.get("shortName") or "").lower()
            if kw in name:
                kw_matches.append(ev)
        if len(kw_matches) == 1:
            return kw_matches[0]
        if len(kw_matches) > 1:
            events = kw_matches

    def ev_state(ev: Dict[str, Any]) -> str:
        comp = None
        comps = ev.get("competitions") or []
        if comps and isinstance(comps, list):
            comp = comps[0]
        status = (comp or ev).get("status") or {}
        st_type = (status.get("type") or {}) if isinstance(status, dict) else {}
        return (st_type.get("state") or "").lower()

    def ev_time(ev: Dict[str, Any]) -> float:
        # ESPN event date is ISO string; use it as a sort key
        ds = ev.get("date") or ""
        try:
            # fromisoformat doesn't handle Z, so replace
            ds2 = ds.replace("Z", "+00:00")
            return datetime.datetime.fromisoformat(ds2).timestamp()
        except Exception:
            return 0.0

    live = [ev for ev in events if ev_state(ev) == "in"]
    if live:
        # if multiple live, pick latest start
        return sorted(live, key=ev_time, reverse=True)[0]

    upcoming = [ev for ev in events if ev_state(ev) == "pre"]
    if upcoming:
        return sorted(upcoming, key=ev_time)[0]

    finished = [ev for ev in events if ev_state(ev) == "post"]
    if finished:
        return sorted(finished, key=ev_time, reverse=True)[0]

    return events[0]



def get_espn_context(state: Dict[str, Any], league: str, date_yyyymmdd: str = "", cache_seconds: int = 60, keywords: str = "", include_summary: bool = True) -> str:
    """Fetch an ESPN snapshot suitable for planner context.

    Returns JSON text with:
      - events: a compact list of all events on the date (id/name/date/state/teams/scores)
      - selected_event_id: the event chosen by live > next upcoming > most recent (with optional keyword tie-break)
      - selected_event: compact header-ish info for the selected event
      - selected_event_summary: (optional) compact recent play summaries + situation from the summary endpoint

    Uses ESPN's public site API (no key). Cached in-memory per process for `cache_seconds`.
    """
    now = time.time()
    league_key = (league or ESPN_DEFAULT_LEAGUE).strip().lower()
    date_key = (date_yyyymmdd or "").strip()
    if not date_key:
        # Use UTC date to align with ESPN's scoreboard expectation
        date_key = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")

    cache = state.get("espn_cache") if isinstance(state.get("espn_cache"), dict) else {}
    cache_id = f"{league_key}:{date_key}:{(keywords or '').strip().lower()}:{'1' if include_summary else '0'}"
    ts = float(cache.get("ts", 0) or 0)
    if cache.get("id") == cache_id and (now - ts) < max(5, cache_seconds) and isinstance(cache.get("text"), str):
        return cache.get("text") or ""

    sb_url = espn_scoreboard_url(league_key)
    data = _http_get_json(sb_url, params={"dates": date_key})
    if not isinstance(data, dict):
        return ""

    events_raw = [ev for ev in (data.get("events") or []) if isinstance(ev, dict)]
    compact_events = _espn_compact_events(events_raw)
    target = _espn_pick_event(events_raw, keywords=keywords)
    if not isinstance(target, dict):
        return ""

    comp = None
    comps = target.get("competitions") or []
    if comps and isinstance(comps, list):
        comp = comps[0]

    status = (comp or target).get("status") or {}
    st_type = (status.get("type") or {}) if isinstance(status, dict) else {}
    state_str = st_type.get("state") or st_type.get("description") or ""
    detail = (st_type.get("detail") or "") if isinstance(st_type, dict) else ""

    competitors = []
    if isinstance(comp, dict):
        for team in (comp.get("competitors") or []):
            if not isinstance(team, dict):
                continue
            t = team.get("team") or {}
            competitors.append({
                "team": t.get("displayName") or t.get("shortDisplayName") or t.get("name"),
                "abbr": t.get("abbreviation"),
                "score": team.get("score"),
                "winner": team.get("winner"),
                "homeAway": team.get("homeAway"),
            })

    event_id = target.get("id") or (comp.get("id") if isinstance(comp, dict) else None)

    selected_summary: Dict[str, Any] = {}
    if include_summary and event_id:
        sum_url = espn_summary_url(league_key)
        sum_json = _http_get_json(sum_url, params={"event": event_id})
        if isinstance(sum_json, dict):
            selected_summary = _espn_extract_best_play_summaries(sum_json, max_plays=25)
            if TELEMETRY:
                TELEMETRY.log("external_api_call", {"provider": "espn", "url": sum_url, "event": str(event_id), "cached": False})
        else:
            selected_summary = {}

    summary = {
        "provider": "espn",
        "league": league_key,
        "date": date_key,
        "events": compact_events,
        "selected_event_id": event_id,
        "selected_event": {
            "event": target.get("name") or target.get("shortName"),
            "event_date": target.get("date"),
            "status": state_str,
            "detail": detail,
            "competitors": competitors,
        },
        "selected_event_summary": selected_summary,
    }

    text_out = json.dumps(summary, ensure_ascii=False)
    state["espn_cache"] = {"id": cache_id, "ts": now, "text": text_out}
    if TELEMETRY:
        TELEMETRY.log("external_api_call", {"provider": "espn", "url": sb_url, "date": date_key, "cached": False})

    return text_out

# ============================================================
# Gemini chat
# ============================================================
def make_chat(brain: genai.Client, kernel: str, model_name: str):
    return brain.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=kernel or "",
            temperature=0.7,
            max_output_tokens=900,
        ),
    )

# ============================================================
# Planner (model decides what to do)
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
) -> str:
    # Keep schema strict but allow the model to choose among actions.
    # Note: No "thought process" required.
    return f"""
You are operating on Moltbook. Decide ONE action to take now, consistent with rate limits and configuration.

CONFIG/CONSTRAINTS:
- Post window open: {post_window_open} (wait {post_wait_minutes} minutes if closed)
- Posts are {'ALLOWED' if allow_posts else 'DISABLED'} by command line.
- Outside-comments are {'ALLOWED' if allow_outside else 'DISABLED'} by command line.
- Voting is {'ALLOWED' if allow_votes else 'DISABLED'} by command line.
- Creating submolts is {'ALLOWED' if allow_create_submolt else 'DISABLED'} by command line.
- Downvotes are {'ALLOWED' if allow_downvote else 'DISABLED'} by command line.
{config_hint}

DIRECTIVE:
{directive}

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

Candidate reply-to-my-post (if any):
{json.dumps(reply_candidate, ensure_ascii=False) if reply_candidate else "None"}

Candidate outside post (if any):
{json.dumps(outside_candidate, ensure_ascii=False) if outside_candidate else "None"}

DIRECTIVE:
{directive}

ACTION POLICY (default preference):
1) If posts are allowed AND post window open, prefer POST.
2) Otherwise prefer REPLY to an unanswered comment on my posts (if available).
3) Otherwise COMMENT on someone else's post (avoid >{MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT} comment threads).
4) Voting actions should be occasional; do not vote every cycle.
5) Creating a submolt should be extremely rare and only when clearly justified.

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

CREATE_SUBMOLT:
{{"action":"CREATE_SUBMOLT","name":"shortname","display_name":"Display Name","description":"...","summary":"why create this"}}

SUBSCRIBE_SUBMOLT:
{{"action":"SUBSCRIBE_SUBMOLT","name":"submolt_name","summary":"why subscribe"}}
""".strip()


def _extract_http_status(err: str) -> Tuple[Optional[int], Optional[str]]:
    """Best-effort parse of '429 RESOURCE_EXHAUSTED' or embedded JSON status."""
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
    # Try to parse embedded dict-like snippet: {'error': {'code': 429, ... 'status': 'RESOURCE_EXHAUSTED'}}
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


def _planner_unavailable_message(chat, brain_prefix: str = "") -> str:
    """Return a human-friendly WAIT reason for planner failures."""
    info = getattr(chat, "_last_llm_exception", None) or {}
    err = (info.get("error") or "")
    err_l = err.lower()
    code, status = _extract_http_status(err)
    code_s = f"HTTP {code} " if code else ""
    status_s = f"{status} " if status else ""

    # Key/authorization issues
    if "api key expired" in err_l or "api_key_invalid" in err_l or ("api_key" in err_l and "invalid" in err_l):
        hint = f"Update {brain_prefix}_GEMINI_API_KEY in .env and rerun with --reload-env." if brain_prefix else "Update <BRAIN>_GEMINI_API_KEY in .env and rerun with --reload-env."
        return f"Planner unavailable ({code_s}{status_s}Gemini API key invalid/expired). {hint}"

    # Quota / rate limiting
    if "resource_exhausted" in err_l or (code == 429) or "rate limit" in err_l or "quota" in err_l:
        return f"Planner unavailable ({code_s}{status_s}Gemini rate limit/quota). Waiting for limits to clear."

    # Network / transient
    if "timeout" in err_l or "timed out" in err_l or "connection" in err_l:
        return f"Planner unavailable ({code_s}{status_s}network/timeout). Will retry next cycle."

    if err:
        short = err.replace("\n", " ")
        return f"Planner unavailable ({code_s}{status_s}{info.get('error_type','error')}: {short[:140]})"
    return "Planner unavailable (unknown error)."


def plan_next_action(chat, prompt: str) -> Dict[str, Any]:
    plan = parse_json_with_one_repair(chat, prompt, telemetry=getattr(chat, '_telemetry', None), brain_name=getattr(chat, '_brain_name', ''), call_tag='planner')
    # Major known failure mode: planner call fails (often 429) and we get default {}.
    # Convert that into an explicit WAIT so the cycle remains dashboard-readable.
    if not isinstance(plan, dict):
        plan = {}
    if not (plan.get("action") or "").strip():
        if getattr(chat, "_telemetry", None):
            chat._telemetry.log("planner_missing_action", {
                "cycle": getattr(chat, "_cycle", None),
                "prompt_chars": len(prompt or ""),
            })
        return {"action": "WAIT", "summary": _planner_unavailable_message(chat, brain_prefix=_brain_env_prefix(getattr(chat, "_brain_name", "")))}
    return plan


def call_text(chat, prompt: str, tag: str = "llm") -> str:
    """Send a prompt and return plain text (with telemetry + TPM guardrails)."""
    cycle = getattr(chat, "_cycle", None)
    model_name = getattr(getattr(chat, "model", None), "name", "") or getattr(chat, "_model_name", "") or ""
    prompt_chars = len(prompt or "")
    est_tokens = int(prompt_chars / max(1.0, GEMINI_TPM_CHAR_TO_TOKEN))

    # Preflight log
    if getattr(chat, "_telemetry", None):
        chat._telemetry.log("llm_request", {"tag": tag, "cycle": cycle, "model": model_name, "prompt_chars": prompt_chars, "est_prompt_tokens": est_tokens})

    # TPM soft cap guardrail
    blocked_s = _tpm_guardrail_maybe_sleep(chat, est_tokens, tag=tag)
    if blocked_s > 0 and getattr(chat, "_telemetry", None):
        chat._telemetry.log("llm_cooldown_active", {"tag": tag, "cycle": cycle, "model": model_name, "sleep_s": blocked_s})

    t0 = time.time()
    try:
        resp = chat.send_message(prompt)
        txt = getattr(resp, "text", "") or ""
        if getattr(chat, "_telemetry", None):
            chat._telemetry.log("llm_call", {"tag": tag, "cycle": cycle, "model": model_name, "prompt_chars": prompt_chars, "response_chars": len(txt), "latency_ms": int((time.time()-t0)*1000)})
        return txt
    except Exception as e:
        info = {"tag": tag, "cycle": cycle, "model": model_name, "prompt_chars": prompt_chars, "est_prompt_tokens": est_tokens, "error_type": type(e).__name__, "error": str(e)}
        setattr(chat, "_last_llm_exception", info)
        if getattr(chat, "_telemetry", None):
            chat._telemetry.log("llm_exception", info)
        # Apply cooldown/backoff for 429s
        _maybe_backoff_on_llm_error(chat, str(e), tag=tag)
        raise

# ============================================================
# Execute actions with rate limit enforcement
# ============================================================
def can_post(state: Dict[str, Any]) -> Tuple[bool, int]:
    now = time.time()
    next_t = float(state.get("next_post_time", 0))
    if now >= next_t:
        return True, 0
    return False, max(0, int((next_t - now) / 60))

def can_comment(state: Dict[str, Any]) -> Tuple[bool, int]:
    now = time.time()
    next_t = float(state.get("next_comment_time", 0))
    if now >= next_t:
        return True, 0
    return False, max(0, int(next_t - now))

def set_post_cooldown(state: Dict[str, Any]) -> None:
    state["next_post_time"] = max(float(state.get("next_post_time", 0.0)), time.time() + POST_COOLDOWN_SECONDS)

def set_comment_cooldown(state: Dict[str, Any]) -> None:
    state["next_comment_time"] = max(float(state.get("next_comment_time", 0.0)), time.time() + COMMENT_COOLDOWN_SECONDS)

def execute_action(client: MoltbookClient, state: Dict[str, Any], plan: Dict[str, Any], flags: Dict[str, Any]) -> bool:
    action = (plan.get("action") or "").upper().strip()
    if not action:
        raise ValueError("Plan missing action")

    if action == "WAIT":
        add_history(state, {"action":"WAIT", "target":"", "summary": plan.get("summary","...")})
        if TELEMETRY:
            TELEMETRY.log("action_skipped", {"action":"WAIT", "reason": plan.get("summary","")})
        print(f"{Fore.CYAN}>> WAIT: {plan.get('summary','')}" )
        return False

    # enforce command-line permissions
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

    # rate limits
    if action == "POST":
        ok, mins = can_post(state)
        if not ok:
            raise ValueError(f"POST not allowed yet ({mins}m remaining)")
        submolt = plan.get("submolt") or "general"
        title = plan.get("title") or ""
        content = plan.get("content") or ""
        print(f"{Fore.CYAN}...Action: POST")
        print(f"{Fore.YELLOW}Target submolt: m/{submolt}")
        print(f"{Fore.GREEN}TITLE: {title}")
        print(f"{Fore.GREEN}CONTENT: {content}\n")
        # Guard against restart loops: record a cooldown before we attempt the API call.
        state["next_post_time"] = max(float(state.get("next_post_time", 0.0)), time.time() + float(POST_FAILURE_COOLDOWN_SECONDS))

        res = client.create_post(submolt=submolt, title=title, content=content)
        if not res.get("success"):
            raise ValueError(f"Post failed: {res.get('error') or res}")
        pid = res.get("post", {}).get("id") or res.get("id")
        if pid:
            state["my_post_ids"].append(pid)
        # Successful post => apply full cooldown window.
        set_post_cooldown(state)
        add_history(state, {"action":"POST", "target": post_url(pid or "?"), "summary": plan.get("summary","")})
        if TELEMETRY:
            TELEMETRY.log("action_executed", {"action":"POST", "post_id": pid, "submolt": submolt, "title": title})
        print(f"{Fore.CYAN}>> POST SUCCESS: {post_url(pid) if pid else res}")
        return True

    if action == "REPLY":
        ok, secs = can_comment(state)
        if not ok:
            raise ValueError(f"COMMENT cooldown active ({secs}s remaining)")
        post_id = plan.get("post_id") or ""
        parent_id = plan.get("parent_comment_id") or ""
        content = plan.get("content") or ""

        # Fetch post meta once; do not rely on feed field variants.
        post_meta = client.get_post(post_id) or {}
        post_obj = post_meta.get("post") if isinstance(post_meta.get("post"), dict) else post_meta
        post_author = (get_author_name(post_obj.get("author")) or post_obj.get("username") or post_obj.get("user") or "").strip().lower()
        is_my_post = (post_author == (MY_USERNAME or "").strip().lower()) or (post_id in set(state.get("my_post_ids", [])))

        # Determine comment count for dogpile rules. Conservative for non-own posts.
        cc = get_post_comment_count(post_obj)
        if (not is_my_post) and cc <= 0:
            raise ActionBlocked("REPLY", "REPLY blocked: unable to determine comment_count for non-own post (conservative anti-dogpile).")

        # --- v9.5 soft anti-dogpile rule ---
        # Allowed:
        # - Replying on my own posts (always)
        # - Replying on others' posts only when the post thread is small AND the reply is depth=1 (top-level comment reply)
        if not is_my_post:
            if cc > MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT:
                raise ActionBlocked("REPLY", f"REPLY blocked (dogpile): post has {cc} comments (> {MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT})")

            # Verify reply depth = 1 by locating the parent comment and ensuring it has no parent_id.
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
            raise ValueError(f"Reply failed: {res.get('error') or res}")
        # mark replied
        if post_id and parent_id:
            key = f"{post_id}:{parent_id}"
            state["replied_comment_keys"].append(key)
        set_comment_cooldown(state)
        add_history(state, {"action":"REPLY", "target": f"{post_url(post_id)}#comment-{parent_id}", "summary": plan.get("summary","")})
        if TELEMETRY:
            TELEMETRY.log("action_executed", {"action":"REPLY", "post_id": post_id, "parent_comment_id": parent_id})
        print(f"{Fore.CYAN}>> REPLY SUCCESS")
        return True


    if action == "COMMENT":
        ok, secs = can_comment(state)
        if not ok:
            raise ValueError(f"COMMENT cooldown active ({secs}s remaining)")
        post_id = plan.get("post_id") or ""

        # Enforce dogpile limit for commenting on others' posts at execution time.
        # We do NOT trust the feed payload to always contain the right comment count field.
        post_meta = client.get_post(post_id) or {}
        post_obj = post_meta.get("post") if isinstance(post_meta.get("post"), dict) else post_meta
        post_author = (get_author_name(post_obj.get("author")) or post_obj.get("username") or post_obj.get("user") or "").strip().lower()
        is_my_post = (post_author == (MY_USERNAME or "").strip().lower()) or (post_id in set(state.get("my_post_ids", [])))

        if not is_my_post:
            cc = get_post_comment_count(post_obj)
            # Conservative: if we can't determine count from meta, assume it's too large.
            if cc <= 0:
                raise ActionBlocked("COMMENT", "COMMENT blocked: unable to determine comment_count for non-own post (conservative anti-dogpile).")
            if cc > MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT:
                raise ActionBlocked("COMMENT", f"COMMENT blocked (dogpile): post has {cc} comments (> {MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT})")

        content = plan.get("content") or ""
        print(f"{Fore.CYAN}...Action: COMMENT")
        print(f"{Fore.YELLOW}Target post: {post_url(post_id)}")
        print(f"{Fore.GREEN}CONTENT: {content}\n")
        res = client.add_comment(post_id, content=content)
        if not res.get("success"):
            raise ValueError(f"Comment failed: {res.get('error') or res}")
        set_comment_cooldown(state)
        add_history(state, {"action":"COMMENT", "target": post_url(post_id), "summary": plan.get("summary","")})
        if TELEMETRY:
            TELEMETRY.log("action_executed", {"action":"COMMENT", "post_id": post_id})
        print(f"{Fore.CYAN}>> COMMENT SUCCESS")
        return True

    if action == "UPVOTE_POST":
        pid = plan.get("post_id") or ""
        print(f"{Fore.CYAN}...Action: UPVOTE_POST {post_url(pid)}")
        res = client.upvote_post(pid)
        if not res.get("success"):
            raise ValueError(f"Upvote failed: {res.get('error') or res}")
        add_history(state, {"action":"UPVOTE_POST", "target": post_url(pid), "summary": plan.get("summary","")})
        if TELEMETRY:
            TELEMETRY.log("action_executed", {"action":"UPVOTE_POST", "post_id": pid})
        print(f"{Fore.CYAN}>> UPVOTE SUCCESS")
        return True

    if action == "DOWNVOTE_POST":
        pid = plan.get("post_id") or ""
        print(f"{Fore.CYAN}...Action: DOWNVOTE_POST {post_url(pid)}")
        res = client.downvote_post(pid)
        if not res.get("success"):
            raise ValueError(f"Downvote failed: {res.get('error') or res}")
        add_history(state, {"action":"DOWNVOTE_POST", "target": post_url(pid), "summary": plan.get("summary","")})
        if TELEMETRY:
            TELEMETRY.log("action_executed", {"action":"DOWNVOTE_POST", "post_id": pid})
        print(f"{Fore.CYAN}>> DOWNVOTE SUCCESS")
        return True

    if action == "UPVOTE_COMMENT":
        cid = plan.get("comment_id") or ""
        print(f"{Fore.CYAN}...Action: UPVOTE_COMMENT {cid}")
        res = client.upvote_comment(cid)
        if not res.get("success"):
            raise ValueError(f"Upvote comment failed: {res.get('error') or res}")
        add_history(state, {"action":"UPVOTE_COMMENT", "target": f"comment:{cid}", "summary": plan.get("summary","")})
        if TELEMETRY:
            TELEMETRY.log("action_executed", {"action":"UPVOTE_COMMENT", "comment_id": cid})
        print(f"{Fore.CYAN}>> UPVOTE COMMENT SUCCESS")
        return True

    if action == "CREATE_SUBMOLT":
        name = plan.get("name") or ""
        display = plan.get("display_name") or ""
        desc = plan.get("description") or ""
        print(f"{Fore.CYAN}...Action: CREATE_SUBMOLT m/{name}")
        res = client.create_submolt(name=name, display_name=display, description=desc)
        if not res.get("success"):
            raise ValueError(f"Create submolt failed: {res.get('error') or res}")
        add_history(state, {"action":"CREATE_SUBMOLT", "target": f"m/{name}", "summary": plan.get("summary","")})
        if TELEMETRY:
            TELEMETRY.log("action_executed", {"action":"CREATE_SUBMOLT", "name": name})
        print(f"{Fore.CYAN}>> CREATE SUBMOLT SUCCESS")
        return True

    if action == "SUBSCRIBE_SUBMOLT":
        name = plan.get("name") or ""
        print(f"{Fore.CYAN}...Action: SUBSCRIBE_SUBMOLT m/{name}")
        # Skip if already subscribed (idempotent locally)
        already = {norm_key(s): True for s in state.get("subscribed_submolts", [])}
        if norm_key(name) in already:
            add_history(state, {"action":"SUBSCRIBE_SUBMOLT", "target": f"m/{name}", "summary": "already subscribed"})
            if TELEMETRY:
                TELEMETRY.log("action_skipped", {"action":"SUBSCRIBE_SUBMOLT", "name": name, "reason":"already subscribed"})
            print(f"{Fore.CYAN}>> SUBSCRIBE SKIPPED: already subscribed")
            return False
        res = client.subscribe_submolt(name)
        if not res.get("success"):
            raise ValueError(f"Subscribe failed: {res.get('error') or res}")
        state.setdefault("subscribed_submolts", []).append(norm_key(name))
        add_history(state, {"action":"SUBSCRIBE_SUBMOLT", "target": f"m/{name}", "summary": plan.get("summary","")})
        if TELEMETRY:
            TELEMETRY.log("action_executed", {"action":"SUBSCRIBE_SUBMOLT", "name": name})
        print(f"{Fore.CYAN}>> SUBSCRIBE SUCCESS")
        return True

    raise ValueError(f"Unknown action: {action}")



# =============================
# SOCIAL-FIRST ACTIONS
# =============================
def _today_ymd() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")

def _reset_daily_counters(state: Dict[str, Any]) -> None:
    if state.get("daily_date") != _today_ymd():
        state["daily_date"] = _today_ymd()
        state["daily"] = {"upvotes": 0, "downvotes": 0, "follows": 0, "subscribes": 0, "createsub": 0, "dms": 0}

def _maybe_pick_from_feed(feed_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not feed_items:
        return None
    candidates = []
    for p in feed_items:
        if not p.get("id"):
            continue
        author = (p.get("author") or p.get("user") or {}).get("name") or p.get("author_name")
        if author and author != MY_USERNAME:
            candidates.append(p)
    return random.choice(candidates) if candidates else random.choice(feed_items)

def _raw_json(chat, prompt: str) -> Dict[str, Any]:
    """Run a one-off JSON-only prompt through the same tolerant JSON parser."""
    return parse_json_with_one_repair(chat, prompt, telemetry=getattr(chat, '_telemetry', None), brain_name=getattr(chat, '_brain_name', ''), call_tag='helper')

def maybe_do_social_actions(
    client: MoltbookClient,
    chat,
    state_path: str,
    state: Dict[str, Any],
    feed_items: List[Dict[str, Any]],
    args: argparse.Namespace,
    kernel: str,
    directive: str,
) -> None:
    """Low-stakes actions that run before the main post/comment decision."""
    _reset_daily_counters(state)

    # 1) Upvote something every cycle
    if args.upvote_every_cycle:
        try:
            target = _maybe_pick_from_feed(feed_items)
            if target and target.get("id"):
                client.upvote_post(target["id"])
                state["daily"]["upvotes"] += 1
                print(f"{Fore.MAGENTA}>> SOCIAL: upvoted post {target['id']}")
        except Exception as e:
            print(f"{Fore.YELLOW}[WARN] upvote failed: {e}")

    # 2) Subscribe to a submolt (policy-based)
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
                # Prefer unsubscribed subs; if all are already subscribed, do nothing.
                candidates = [s for s in seen if norm_key(s) not in already]
                if not candidates:
                    # Nothing new to subscribe to from this feed snapshot.
                    pass
                else:
                    sm = random.choice(candidates)
                    client.subscribe_submolt(sm)
                    # Record locally so we don't attempt again in future runs.
                    state.setdefault("subscribed_submolts", []).append(norm_key(sm))
                    state["daily"]["subscribes"] += 1
                    print(f"{Fore.MAGENTA}>> SOCIAL: subscribed to /m/{sm}")
        except Exception as e:
            print(f"{Fore.YELLOW}[WARN] subscribe failed: {e}")

    # 3) Create a new submolt (rare; allowed by default)
    if args.allow_create_submolt and args.create_submolt_prob > 0 and random.random() < args.create_submolt_prob:
        try:
            sj = _raw_json(chat, (
                f"""{kernel}

You are proposing ONE new submolt to create for Moltbook, based on this directive: {directive}

Return strict JSON only:
{{"name":"slug","display_name":"...","description":"..."}}

Constraints:
- name: 3-21 chars; lowercase letters, numbers, underscore only
- keep it broadly useful and non-spammy
"""
            ))
            name = str(sj.get("name","")).strip().lower()
            name = re.sub(r"[^a-z0-9_]", "_", name)
            name = re.sub(r"_+", "_", name).strip("_")[:21]
            if len(name) < 3:
                raise ValueError("invalid submolt name")
            display_name = str(sj.get("display_name","")).strip()[:60] or name
            description = str(sj.get("description","")).strip()[:280] or "A new place for discussion."
            client.create_submolt(name=name, display_name=display_name, description=description)
            state["daily"]["createsub"] += 1
            print(f"{Fore.MAGENTA}>> SOCIAL: created submolt /m/{name}")
        except Exception as e:
            print(f"{Fore.YELLOW}[WARN] create submolt failed: {e}")

    # 4) Follow authors we "liked"
    if args.follow_on_like and feed_items:
        try:
            pick = _raw_json(chat, (
                f"""{kernel}

Directive: {directive}

From the feed items below, pick at most ONE author to follow because you genuinely liked their contribution.
If none, return {{"follow": false}}.
Return strict JSON only: {{"follow": true/false, "author": "Name"}}

FEED ITEMS (brief):
{json.dumps([{"id": p.get("id"), "author": get_author_name(p.get("author")), "content": shorten(p.get("content",""),200)} for p in feed_items], ensure_ascii=False)}

"""
            ))
            if pick.get("follow") and random.random() < float(args.follow_prob):
                author = pick.get('author')
                if isinstance(author, dict):
                    author = author.get('name') or author.get('username') or author.get('handle')
                author = (author or '').strip()
                if author and author != MY_USERNAME:
                    followed = {norm_key(a): True for a in state.get("followed_agents", [])}
                    if norm_key(author) in followed:
                        # Already followed in a prior run.
                        pass
                    else:
                        client.follow_agent(author)
                        state.setdefault("followed_agents", []).append(norm_key(author))
                        state["daily"]["follows"] += 1
                        print(f"{Fore.MAGENTA}>> SOCIAL: followed @{author}")
        except Exception as e:
            print(f"{Fore.YELLOW}[WARN] follow failed: {e}")

    save_state(state_path, state)

def maybe_dm_fallback(
    client: MoltbookClient,
    chat,
    state_path: str,
    state: Dict[str, Any],
    feed_items: List[Dict[str, Any]],
    args: argparse.Namespace,
    kernel: str,
    directive: str,
) -> bool:
    """Consent-based DM fallback if we have no comment targets."""
    if not args.allow_dms:
        return False
    _reset_daily_counters(state)

    try:
        target = _maybe_pick_from_feed(feed_items)
        if not target:
            return False
        author = get_author_name(target.get("author")) or (target.get("author_name") or "")
        author = str(author).strip()
        if not author or author == MY_USERNAME:
            return False

        j = _raw_json(chat, (
            f"""{kernel}

Directive: {directive}

Write a short Moltbook DM *request* (1-3 sentences) to @{author}.
Return strict JSON only: {{"message":"..."}}
Make it specific to the feed topic below (do not be creepy; keep it professional/friendly).

FEED TOPIC:
{shorten(target.get("content",""), 400)}

"""
        ))
        message = str(j.get("message","")).strip()
        if not message:
            return False

        client.dm_request(to=author, message=message)
        state["daily"]["dms"] += 1
        save_state(state_path, state)
        print(f"{Fore.MAGENTA}>> SOCIAL: sent DM request to @{author}")
        return True
    except Exception as e:
        print(f"{Fore.YELLOW}[WARN] DM request failed: {e}")
        return False


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brain", help="Brain name (used as filename prefix in BRAINS_DIR).")
    ap.add_argument("directive", nargs="?", default="Participate on Moltbook.")
    ap.add_argument("--allow-self-directive-update", action="store_true", help="Allow the agent to update its saved directive via SET_DIRECTIVE action.")
    ap.add_argument("--interval", type=int, default=5, help="Sleep interval minutes between cycles.")
    ap.add_argument("--reload-env", action="store_true", help="Reload .env and overwrite existing environment variables for this process (useful after rotating keys).")

    DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    ap.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name (default: gemini-2.5-flash). Can also set GEMINI_MODEL env var.")
    ap.add_argument("--inject-espn", action="store_true", help="Inject ESPN scoreboard data into planner context.")
    ap.add_argument("--espn-cache-seconds", type=int, default=60, help="Cache ESPN fetch for N seconds (default 60).")
    ap.add_argument("--espn-league", default=os.environ.get("ESPN_LEAGUE", ESPN_DEFAULT_LEAGUE), help="ESPN league code (nfl, nba, nhl, mlb, wnba, epl) or raw path like soccer/eng.1. Default nfl.")
    ap.add_argument("--espn-date", default="", help="Date for scoreboard in YYYYMMDD (default: today).")
    ap.add_argument("--espn-keywords", default="", help="Optional keywords to match an ESPN event name (fallback tie-breaker).")
    ap.add_argument("--priority", choices=["replies_first", "outside_first"], default="replies_first",
                    help="Default engagement preference (planner can still decide).")
    ap.add_argument("--mode", choices=["all", "comment_only", "no_post"], default="all",
                    help="Whether to allow posting.")
    ap.add_argument("--allow-votes", action="store_true", help="Allow occasional upvote/downvote actions.")
    ap.add_argument("--allow-downvote", action="store_true", help="Allow downvotes (requires --allow-votes too).")
    ap.add_argument("--feed-sort", choices=["hot","new","top","rising"], default="hot", help="Feed sort for scanning.")

    ap.add_argument("--upvote-every-cycle", dest="upvote_every_cycle", action="store_true",
                    help="Attempt at least one upvote each cycle (default on).")
    ap.add_argument("--no-upvote-every-cycle", dest="upvote_every_cycle", action="store_false",
                    help="Disable the per-cycle upvote attempt.")
    ap.set_defaults(upvote_every_cycle=UPVOTE_EVERY_CYCLE_DEFAULT)

    ap.add_argument("--follow-on-like", dest="follow_on_like", action="store_true",
                    help="Follow authors the planner labels as 'liked' (default on).")
    ap.add_argument("--no-follow-on-like", dest="follow_on_like", action="store_false",
                    help="Disable follow-on-like behavior.")
    ap.set_defaults(follow_on_like=FOLLOW_ON_LIKE_DEFAULT)

    ap.add_argument("--follow-prob", type=float, default=FOLLOW_PROB_DEFAULT,
                    help="Chance to follow a 'liked' author when found (0..1).")

    ap.add_argument("--subscribe-policy", choices=["off","low","medium","high"], default=SUBSCRIBE_POLICY_DEFAULT,
                    help="How willing the bot is to subscribe to submolts based on what it likes.")
    ap.add_argument("--create-submolt-prob", type=float, default=CREATE_SUBMOLT_PROB_DEFAULT,
                    help="Probability of creating a new submolt (rare).")
    ap.add_argument("--allow-create-submolt", dest="allow_create_submolt", action="store_true",
                    help="Allow creating submolts (default on).")
    ap.add_argument("--no-allow-create-submolt", dest="allow_create_submolt", action="store_false",
                    help="Disallow creating submolts.")
    ap.set_defaults(allow_create_submolt=ALLOW_CREATE_SUBMOLT_DEFAULT)

    ap.add_argument("--allow-dms", dest="allow_dms", action="store_true",
                    help="Allow consent-based DM fallback when no comment targets (default on).")
    ap.add_argument("--no-allow-dms", dest="allow_dms", action="store_false",
                    help="Disable DM fallback.")
    ap.set_defaults(allow_dms=ALLOW_DMS_DEFAULT)
    args = ap.parse_args()
    brain_name = str(args.brain).strip()
    if not brain_name:
        raise SystemExit("Missing brain name")
    # Keep file-friendly
    brain_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", brain_name)

    # Load .env from the script directory (so you can keep per-brain keys locally)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    _load_dotenv(os.path.join(script_dir, ".env"), overwrite=bool(getattr(args, "reload_env", False)))

    # Per-brain env vars (preferred):
    #   <BRAIN>_GEMINI_API_KEY
    #   <BRAIN>_MOLTBOOK_API_KEY
    #   <BRAIN>_MY_USERNAME
    # (fallbacks: GEMINI_API_KEY / MOLTBOOK_API_KEY / MY_USERNAME)
    prefix = _brain_env_prefix(brain_name)
    gem_key = os.environ.get(f"{prefix}_GEMINI_API_KEY", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()
    mb_key = os.environ.get(f"{prefix}_MOLTBOOK_API_KEY", "").strip() or os.environ.get("MOLTBOOK_API_KEY", "").strip()
    uname = os.environ.get(f"{prefix}_MY_USERNAME", "").strip() or os.environ.get("MY_USERNAME", "").strip()

    global GEMINI_API_KEY, MOLTBOOK_API_KEY, MY_USERNAME, TELEMETRY
    GEMINI_API_KEY = gem_key
    MOLTBOOK_API_KEY = mb_key
    MY_USERNAME = uname or brain_name  # fallback: use brain name as username
    RUN_ID = uuid.uuid4().hex
    TELEMETRY = TelemetryLogger(brain_name=brain_name, run_id=RUN_ID, base_dir=os.environ.get("TELEMETRY_DIR", "telemetry").strip() or "telemetry")
    TELEMETRY.log("run_start", {"version": VERSION, "brain_env_prefix": prefix, "gemini_key_fp": _key_fingerprint(GEMINI_API_KEY), "moltbook_key_fp": _key_fingerprint(MOLTBOOK_API_KEY)})

    if not GEMINI_API_KEY:
        raise SystemExit(f"Missing {prefix}_GEMINI_API_KEY (or GEMINI_API_KEY)")
    if not MOLTBOOK_API_KEY:
        raise SystemExit(f"Missing {prefix}_MOLTBOOK_API_KEY (or MOLTBOOK_API_KEY)")

    user_directive = args.directive  # v6.6: base directive from CLI

    print(f"{Fore.CYAN}=== {brain_name}: autonomy v{VERSION} (multi-brain planner loop) ===")
    print(f"{Fore.CYAN}    env prefix: {prefix} | gemini_key:*{_key_fingerprint(GEMINI_API_KEY)} | moltbook_key:*{_key_fingerprint(MOLTBOOK_API_KEY)}")

    if not GEMINI_API_KEY or not MOLTBOOK_API_KEY:
        print(f"{Fore.RED}Fill in GEMINI_API_KEY and MOLTBOOK_API_KEY in the script.")
        return
    if "moltbook.com" in MOLTBOOK_API_BASE and "www.moltbook.com" not in MOLTBOOK_API_BASE:
        print(f"{Fore.RED}MOLTBOOK_API_BASE must be https://www.moltbook.com/api/v1 (with www).")
        return

    os.makedirs(BRAINS_DIR, exist_ok=True)

    state_path = os.path.join(BRAINS_DIR, f"{brain_name}_memories.json")
    kernel_path = os.path.join(BRAINS_DIR, f"{brain_name}_kernel_prompt.txt")
    knowledge_path = os.path.join(BRAINS_DIR, f"{brain_name}_knowledge.txt")

    state = load_state(state_path)
    # v6.5: choose directive (state can persist across runs)
    if (not user_directive) and state.get('directive'):
        user_directive = state.get('directive')

    if user_directive and user_directive != 'Participate on Moltbook.':
        user_directive = user_directive
    state.setdefault('directive', user_directive)
    kernel = load_kernel(kernel_path)
    knowledge = load_knowledge(knowledge_path)
    client = MoltbookClient(MOLTBOOK_API_KEY, telemetry=TELEMETRY, brain_name=brain_name)
    brain_client = genai.Client(api_key=GEMINI_API_KEY)

    # derive permissions
    allow_posts = (args.mode == "all")
    if args.mode in ("comment_only", "no_post"):
        allow_posts = False
    allow_outside = True  # always allowed unless you decide to add a flag later
    allow_votes = bool(args.allow_votes)
    allow_downvote = bool(args.allow_votes and args.allow_downvote)
    allow_create_submolt = bool(args.allow_create_submolt or ALLOW_CREATE_SUBMOLT_DEFAULT)

    flags = {
        "allow_posts": allow_posts,
        "allow_outside": allow_outside,
        "allow_votes": allow_votes,
        "allow_downvote": allow_downvote,
        "allow_create_submolt": allow_create_submolt,
    }

    iteration = 0
    while True:
        # Recreate chat each cycle to avoid token accumulation
        chat = make_chat(brain_client, kernel, args.gemini_model)
        # attach telemetry to chat (used by JSON parsing helpers)
        setattr(chat, '_telemetry', TELEMETRY)
        setattr(chat, '_brain_name', brain_name)

        iteration += 1
        # stamp cycle + model on chat for telemetry
        setattr(chat, '_cycle', iteration)
        setattr(chat, 'model', args.gemini_model)
        print(f"\n{Fore.YELLOW}--- CYCLE {iteration} | {datetime.datetime.now().strftime('%H:%M:%S')} ---")
        if TELEMETRY:
            TELEMETRY.log("cycle_start", {"cycle": iteration})

        # refresh my posts for reply scanning
        did_add = refresh_my_posts_from_profile(client, state)
        if did_add:
            save_state(state_path, state)

        # compute windows
        post_ok, post_wait = can_post(state)
        post_window_open = post_ok
        window = "OPEN" if post_window_open else f"CLOSED ({post_wait}m)"
        print(f"{Fore.WHITE}Post Window: {window} | Comment Window: ALWAYS OPEN")

        # build context
        feed = client.get_feed(limit=FEED_LIMIT, sort=args.feed_sort)
        maybe_do_social_actions(client, chat, state_path, state, feed, args, kernel, user_directive)
        feed_brief = "\n".join(
            f"- @{get_author_name(p.get('author'))}: {shorten(p.get('content',''), FEED_ITEM_CHARS)} ({post_url(p.get('id',''))})"
            for p in feed if p.get("id")
        ) or "No feed available."

        # Optional external data injections (dashboard-ready)
        external_data = ""
        if getattr(args, "inject_espn", False):
            external_data = get_espn_context(state, league=str(getattr(args, "espn_league", ESPN_DEFAULT_LEAGUE)), date_yyyymmdd=str(getattr(args, "espn_date", "")), cache_seconds=int(getattr(args, "espn_cache_seconds", 60)), keywords=str(getattr(args, "espn_keywords", "")))

        reply_candidate = find_unanswered_comment_on_my_posts(client, state)
        outside_candidate = pick_outside_post_for_comment(feed, state)

        # If we're out of comment targets and can't/shouldn't post, optionally DM someone.
        if (not reply_candidate) and (not outside_candidate) and (not post_window_open or not allow_posts):
            if maybe_dm_fallback(client, chat, state_path, state, feed, args, kernel, user_directive):
                if TELEMETRY:
                    TELEMETRY.log("cycle_end", {"cycle": iteration, "reason": "dm_fallback"})
                print(f"{Fore.WHITE}Sleeping for {args.interval} minutes...")
                time.sleep(max(1, args.interval) * 60)
                continue

        hist_txt = history_context(state)
        mem_txt = memory_context(state)

        config_hint = ""
        if args.priority == "outside_first":
            config_hint = "- Default preference overridden: prefer outside comments when not posting.\n"

        prompt = build_planner_prompt(
            directive=user_directive,
            knowledge=knowledge,
            memory=mem_txt,
            hist=hist_txt,
            feed_brief=feed_brief,
            external_data=external_data,
            post_window_open=post_window_open,
            post_wait_minutes=post_wait,
            reply_candidate=reply_candidate,
            outside_candidate=outside_candidate,
            config_hint=config_hint,
            allow_posts=allow_posts,
            allow_outside=allow_outside,
            allow_votes=allow_votes,
            allow_create_submolt=allow_create_submolt,
            allow_downvote=allow_downvote,
        )

        try:
            plan = plan_next_action(chat, prompt)
            # If planner says REPLY but doesn't provide ids, fill from candidate if available
            if (plan.get("action") or "").upper() == "REPLY" and reply_candidate:
                plan.setdefault("post_id", reply_candidate.get("post_id"))
                plan.setdefault("parent_comment_id", reply_candidate.get("comment_id"))
            # If planner says COMMENT but doesn't provide post_id, fill from outside candidate
            if (plan.get("action") or "").upper() == "COMMENT" and outside_candidate:
                plan.setdefault("post_id", outside_candidate.get("id"))

            # Hard guard: never attempt POST when the post window is closed.
            # The model can still suggest POST; we override programmatically.
            act = (plan.get("action") or "").upper().strip()
            if act == "POST" and (not post_window_open or not allow_posts):
                # Prefer replying if we have a target; otherwise comment; otherwise skip this cycle.
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
                executed = execute_action(client, state, plan, flags)
            except ActionBlocked as ab:
                # Log the block and attempt an in-cycle fallback without another LLM call.
                if TELEMETRY:
                    TELEMETRY.log("action_blocked", {"cycle": iteration, "action": ab.action, "reason": ab.reason})
                print(f"{Fore.RED}[ERROR] {ab.reason}")

                # Fallback order:
                # 1) Reply to a comment on my own post (if available)
                # 2) Comment on a different eligible outside post (small thread)
                # 3) Post (if window open)
                # 4) Wait
                fallback_plan = None

                # 1) Prefer reply candidate if present
                if reply_candidate:
                    fallback_plan = {
                        "action": "REPLY",
                        "post_id": reply_candidate.get("post_id"),
                        "parent_comment_id": reply_candidate.get("comment_id"),
                        "content": plan.get("content") if (plan.get("action") or "").upper() == "REPLY" else plan.get("content"),
                        "summary": "fallback_after_block",
                    "source_action": (plan.get("action") or "").upper(),
                    }

                # 2) Try another outside post from feed that passes dogpile rules and isn't the blocked one
                if (fallback_plan is None):
                    blocked_post_id = plan.get("post_id") or ""
                    alt = None
                    for p in feed or []:
                        pid = p.get("id")
                        if (not pid) or (pid == blocked_post_id):
                            continue
                        # skip already-commented posts
                        if pid in set(state.get("commented_post_ids", [])):
                            continue
                        # enforce dogpile if we can determine it from the feed
                        cc = get_post_comment_count(p)
                        if cc > 0 and cc <= MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT:
                            alt = p
                            break
                    if alt:
                        fallback_plan = {
                            "action": "COMMENT",
                            "post_id": alt.get("id"),
                            "content": plan.get("content") if (plan.get("action") or "").upper() == "COMMENT" else plan.get("content"),
                            "summary": "fallback_after_block",
                        "source_action": (plan.get("action") or "").upper(),
                        }

                # 3) If allowed, try posting
                if (fallback_plan is None) and post_window_open and allow_posts:
                    fallback_plan = {
                        "action": "POST",
                        "title": plan.get("title") or "Thought",
                        "content": plan.get("content") or "",
                        "submolt": plan.get("submolt") or "general",
                        "summary": "fallback_after_block",
                    }

                
                if fallback_plan is not None:
                    # If we changed the action type/target after a block, the previously generated content
                    # may reference the blocked context. Regenerate a small piece of content using the helper
                    # model (cheap) so the fallback target gets an appropriate reply/comment.
                    try:
                        src = (fallback_plan.get("source_action") or "").upper()
                        act2 = (fallback_plan.get("action") or "").upper()
                        if act2 in ("REPLY","COMMENT") and act2 != src:
                            regen_prompt = ""
                            if act2 == "REPLY":
                                regen_prompt = (
                                    "You are writing a reply to a comment on my own post.\n"
                                    f"Directive: {user_directive}\n"
                                    f"Post URL: {post_url(fallback_plan.get('post_id',''))}\n"
                                    f"Parent comment id: {fallback_plan.get('parent_comment_id','')}\n"
                                    "Write a thoughtful, substantive reply. Return ONLY the reply text."
                                )
                            else:
                                regen_prompt = (
                                    "You are writing a comment on someone else's post.\n"
                                    f"Directive: {user_directive}\n"
                                    f"Post URL: {post_url(fallback_plan.get('post_id',''))}\n"
                                    "Write a thoughtful, substantive comment. Return ONLY the comment text."
                                )
                            # Use helper tag for telemetry clarity
                            try:
                                txt = call_text(chat, regen_prompt, tag="helper") or ""
                                fallback_plan["content"] = txt.strip()
                                if TELEMETRY:
                                    TELEMETRY.log("fallback_content_regenerated", {"cycle": iteration, "action": act2, "source_action": src, "chars": len(fallback_plan["content"])})
                            except Exception as _e2:
                                # If regen fails, keep original content to avoid dropping the cycle entirely.
                                if TELEMETRY:
                                    TELEMETRY.log("fallback_content_regen_failed", {"cycle": iteration, "error": str(_e2)})
                    except Exception:
                        pass
                    try:
                        executed = execute_action(client, state, fallback_plan, flags)
                    except ActionBlocked as ab2:
                        if TELEMETRY:
                            TELEMETRY.log("action_blocked", {"cycle": iteration, "action": ab2.action, "reason": ab2.reason})
                        print(f"{Fore.RED}[ERROR] {ab2.reason}")
                        executed = False

            if executed:
                save_state(state_path, state)

        except Exception as e:
            if TELEMETRY:
                TELEMETRY.log("error", {"cycle": iteration, "error": str(e)})
            print(f"{Fore.RED}[ERROR] {e}")

        if TELEMETRY:
            TELEMETRY.log("cycle_end", {"cycle": iteration})
        print(f"{Fore.WHITE}Sleeping for {args.interval} minutes...")
        time.sleep(max(1, args.interval) * 60)

if __name__ == "__main__":
    main()
