"""Configuration loading, constants, and environment helpers."""

import os
import re
from typing import Dict

# ============================================================
# .env loader (minimal; avoids extra deps)
# ============================================================
def load_dotenv(dotenv_path: str, overwrite: bool = False) -> None:
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
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                if not k:
                    continue
                if overwrite or (k not in os.environ):
                    os.environ[k] = v
    except Exception:
        return


def brain_env_prefix(brain_name: str) -> str:
    p = (brain_name or "").upper()
    p = re.sub(r"[^A-Z0-9_]", "_", p)
    p = re.sub(r"_+", "_", p).strip("_")
    return p or "BRAIN"


def key_fingerprint(k: str) -> str:
    k = (k or "").strip()
    if len(k) <= 6:
        return k
    return k[-6:]


# ============================================================
# Moltbook
# ============================================================
MOLTBOOK_API_BASE = "https://www.moltbook.com/api/v1"
MOLTBOOK_WEB_BASE = "https://www.moltbook.com"

# ============================================================
# ESPN
# ============================================================
ESPN_DEFAULT_LEAGUE = "nfl"
ESPN_LEAGUE_MAP: Dict[str, tuple] = {
    "nfl": ("football", "nfl"),
    "ncaaf": ("football", "college-football"),
    "nba": ("basketball", "nba"),
    "wnba": ("basketball", "wnba"),
    "mlb": ("baseball", "mlb"),
    "nhl": ("hockey", "nhl"),
    "epl": ("soccer", "eng.1"),
    "ucl": ("soccer", "uefa.champions"),
}

# ============================================================
# Brains / Files
# ============================================================
BRAINS_DIR = os.environ.get("BRAINS_DIR", "brains").strip() or "brains"

# ============================================================
# Context sizing
# ============================================================
KNOWLEDGE_MAX_CHARS = int(os.environ.get("KNOWLEDGE_MAX_CHARS", "600000"))
MEMORY_MAX_CHARS = 4000
FEED_LIMIT = 12
FEED_ITEM_CHARS = 400
HISTORY_KEEP = 250
HISTORY_CONTEXT_N = 15
MY_POST_SCAN_LIMIT = 50
MAX_THREAD_COMMENTS_FOR_OUTSIDE_ENGAGEMENT = 12

# Candidate capping / prompt budgeting
MAX_REPLY_CANDIDATE_CHARS = 5000
MAX_OUTSIDE_CANDIDATE_CHARS = 5000

# Merit-based reply selection
REPLY_SELECTION_MAX_COMMENTS = 25
MAX_COMMENT_THREADS_SCANNED = 4
COMMENTS_CACHE_TTL_S = 90
REPLY_MERIT_MIN_SCORE = -999

# ============================================================
# LLM guardrails (provider-agnostic defaults)
# ============================================================
LLM_TPM_SOFT_CAP = int(os.environ.get("GEMINI_TPM_SOFT_CAP", "900000"))
LLM_TPM_CHAR_TO_TOKEN = float(os.environ.get("GEMINI_TPM_CHAR_TO_TOKEN", "4.0"))
LLM_TPM_WINDOW_SECONDS = 60
LLM_BACKOFF_INITIAL_SECONDS = float(os.environ.get("GEMINI_BACKOFF_INITIAL_SECONDS", "30"))
LLM_BACKOFF_MAX_SECONDS = float(os.environ.get("GEMINI_BACKOFF_MAX_SECONDS", "300"))

# ============================================================
# Rate limits (from Moltbook skill docs + local guardrails)
# ============================================================
POST_FAILURE_COOLDOWN_SECONDS = int(os.environ.get("POST_FAILURE_COOLDOWN_SECONDS", "900"))
POST_COOLDOWN_SECONDS = 30 * 60
COMMENT_COOLDOWN_SECONDS = 20
REQUESTS_PER_MINUTE_SOFT = 90

# ============================================================
# Social action defaults
# ============================================================
UPVOTE_EVERY_CYCLE_DEFAULT = True
FOLLOW_ON_LIKE_DEFAULT = False
FOLLOW_PROB_DEFAULT = 0.60
SUBSCRIBE_POLICY_DEFAULT = "medium"
SUBSCRIBE_PROB_BY_POLICY = {"off": 0.0, "low": 0.10, "medium": 0.25, "high": 0.45}
CREATE_SUBMOLT_PROB_DEFAULT = 0.05
ALLOW_CREATE_SUBMOLT_DEFAULT = False
ALLOW_DMS_DEFAULT = True
ALLOW_DOWNVOTE_DEFAULT = True
