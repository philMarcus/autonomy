"""Phil's projects (v19.2): read access to his GitHub repositories for the agent.

Two planner tools and one embedding source:
  list_phil_projects()            — every public repo (plus opted-in private ones)
  read_phil_project(repo, path)   — README (default), file tree ("tree") or a file
  project_documents(...)          — README + top-level docs chunked for the recall index

Auth: GITHUB_TOKEN env, else `gh auth token`, else anonymous (public repos only,
60 requests/hour). Everything is read-only.
"""

import base64
import json
import logging
import os
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

OWNER = "philMarcus"
API = "https://api.github.com"
# Private repos the agent may read (public ones are always included).
INCLUDE_PRIVATE: List[str] = []
# Repos that are not "projects" (profile README, placeholders).
EXCLUDE = {"philMarcus", "marcus-project"}
# Docs embedded for recall, in priority order; README always first.
DOC_CANDIDATES = ("README.md", "CLAUDE.md", "PLAN.md", "ARCHITECTURE.md", "DESIGN.md", "NOTES.md")
MAX_DOC_CHARS = 40_000       # per repo, across docs
MAX_FILE_CHARS = 12_000      # read_phil_project cap
CHUNK_CHARS = 2_000
CACHE_TTL = 3600

_token_cache: Dict[str, Any] = {"token": None, "checked": False}
_cache: Dict[str, Any] = {}


def github_token() -> str:
    if _token_cache["checked"]:
        return _token_cache["token"] or ""
    _token_cache["checked"] = True
    tok = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    if not tok:
        try:
            tok = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True,
                                 timeout=10).stdout.strip()
        except Exception:
            tok = ""
    _token_cache["token"] = tok
    return tok


def _get(path: str, params: Optional[dict] = None, raw: bool = False, timeout: int = 30):
    """GET from the GitHub API. Returns parsed JSON (or text when raw=True)."""
    headers = {"Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json",
               "User-Agent": "autonomy-agent"}
    tok = github_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    url = path if path.startswith("http") else f"{API}{path}"
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code == 404:
        raise FileNotFoundError(path)
    r.raise_for_status()
    return r.text if raw else r.json()


def _cached(key: str, fn: Callable[[], Any], ttl: int = CACHE_TTL):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (time.time(), val)
    return val


def _trim(text: str, cap: int) -> str:
    text = text or ""
    return text if len(text) <= cap else text[:cap] + f"\n…[truncated at {cap} chars]"


# ----------------------------------------------------------------------
# Repos
# ----------------------------------------------------------------------

def _repo_summary(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "repo": r.get("name", ""),
        "description": r.get("description") or "",
        "language": r.get("language") or "",
        "pushed_at": (r.get("pushed_at") or "")[:10],
        "stars": r.get("stargazers_count", 0),
        "private": bool(r.get("private")),
        "url": r.get("html_url", ""),
        "topics": r.get("topics") or [],
    }


def list_repos(get=None) -> List[Dict[str, Any]]:
    get = get or _get
    """Public repos (sorted by last push, newest first) plus INCLUDE_PRIVATE."""
    def _fetch():
        rows = get(f"/users/{OWNER}/repos", params={"per_page": 100, "type": "owner", "sort": "pushed"})
        repos = [_repo_summary(r) for r in rows if not r.get("fork") and r.get("name") not in EXCLUDE]
        names = {r["repo"] for r in repos}
        for name in INCLUDE_PRIVATE:
            if name not in names:
                try:
                    repos.append(_repo_summary(get(f"/repos/{OWNER}/{name}")))
                except Exception as e:  # noqa: BLE001
                    log.warning("private repo %s unavailable: %s", name, e)
        repos.sort(key=lambda r: r["pushed_at"], reverse=True)
        return repos
    return _cached("repos", _fetch)


def read_readme(repo: str, get=None) -> str:
    get = get or _get
    def _fetch():
        try:
            return get(f"/repos/{OWNER}/{repo}/readme", raw=True)
        except FileNotFoundError:
            return ""
    return _cached(f"readme:{repo}", _fetch)


def read_tree(repo: str, get=None, max_depth: int = 2, limit: int = 200) -> List[str]:
    get = get or _get
    def _fetch():
        data = get(f"/repos/{OWNER}/{repo}/git/trees/HEAD", params={"recursive": "1"})
        paths = [t["path"] + ("/" if t.get("type") == "tree" else "")
                 for t in data.get("tree", []) if t.get("path", "").count("/") < max_depth]
        return sorted(paths)[:limit]
    return _cached(f"tree:{repo}", _fetch)


def read_file(repo: str, path: str, get=None) -> str:
    get = get or _get
    def _fetch():
        data = get(f"/repos/{OWNER}/{repo}/contents/{path.strip('/')}")
        if isinstance(data, list):
            return "\n".join(d.get("path", "") + ("/" if d.get("type") == "dir" else "") for d in data)
        if data.get("encoding") == "base64":
            try:
                return base64.b64decode(data.get("content", "")).decode("utf-8")
            except UnicodeDecodeError:
                return f"[binary file, {data.get('size', 0)} bytes]"
        return data.get("content", "") or ""
    return _cached(f"file:{repo}:{path}", _fetch)


# ----------------------------------------------------------------------
# Embedding source
# ----------------------------------------------------------------------

def _chunk(text: str, size: int = CHUNK_CHARS) -> List[str]:
    """Split on paragraph boundaries into ~size-char chunks (hard split for giant paragraphs)."""
    out: List[str] = []
    buf = ""
    for para in text.replace("\r\n", "\n").split("\n\n"):
        para = para.strip()
        if not para:
            continue
        while len(para) > size * 1.5:
            out.append(para[:size])
            para = para[size:]
        if buf and len(buf) + len(para) + 2 > size:
            out.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        out.append(buf)
    return out


def project_documents(get=None, brain: str = "") -> List[Dict[str, Any]]:
    get = get or _get
    """README + top-level docs of every repo, chunked, as recall documents (kind phil_project)."""
    docs: List[Dict[str, Any]] = []
    for repo in list_repos(get):
        name = repo["repo"]
        budget = MAX_DOC_CHARS
        texts: List[tuple] = []
        readme = read_readme(name, get)
        if readme:
            texts.append(("README.md", readme))
        try:
            tree = set(read_tree(name, get, max_depth=1))
        except Exception:  # noqa: BLE001
            tree = set()
        for doc in DOC_CANDIDATES[1:]:
            if doc in tree:
                try:
                    texts.append((doc, read_file(name, doc, get)))
                except Exception:  # noqa: BLE001
                    pass
        header = f"[Phil's project: {name}" + (f" — {repo['description']}" if repo["description"] else "") + "]"
        for path, text in texts:
            text = text[:budget]
            budget -= len(text)
            for i, chunk in enumerate(_chunk(text)):
                docs.append({
                    "source_ref": f"gh:{name}:{path}:{i}", "kind": "phil_project",
                    "brain": brain, "run_id": "", "cycle": None,
                    "text": f"{header} ({path}, part {i + 1})\n{chunk}",
                })
            if budget <= 0:
                break
    return docs


# ----------------------------------------------------------------------
# Planner tools
# ----------------------------------------------------------------------

def build_project_tools(registry: Any) -> None:
    from .tools import ToolDef

    def list_phil_projects() -> Dict[str, Any]:
        try:
            repos = list_repos()
        except Exception as e:  # noqa: BLE001
            return {"error": f"GitHub unavailable: {str(e)[:200]}"}
        return {"owner": OWNER, "projects": repos, "count": len(repos),
                "note": "Use read_phil_project(repo) for the README, path='tree' for the file list, "
                        "or a file path for its contents. recall also searches these docs."}

    def read_phil_project(repo: str, path: str = "") -> Dict[str, Any]:
        repo = (repo or "").strip().strip("/")
        if not repo:
            return {"error": "repo is required (see list_phil_projects)"}
        path = (path or "").strip()
        try:
            if path in ("", "readme", "README", "README.md"):
                text = read_readme(repo)
                if not text:
                    return {"repo": repo, "path": "README.md", "content": "", "note": "No README; try path='tree'."}
                return {"repo": repo, "path": "README.md", "content": _trim(text, MAX_FILE_CHARS)}
            if path in ("tree", "/", "."):
                return {"repo": repo, "path": "tree", "entries": read_tree(repo)}
            return {"repo": repo, "path": path, "content": _trim(read_file(repo, path), MAX_FILE_CHARS)}
        except FileNotFoundError:
            return {"error": f"{repo}/{path or 'README'} not found"}
        except Exception as e:  # noqa: BLE001
            return {"error": f"GitHub read failed: {str(e)[:200]}"}

    registry.register(ToolDef(
        name="list_phil_projects",
        description=("List your architect Phil's software projects on GitHub (games, simulations, "
                     "AI pipelines, this agent, Analog Home). Read-only."),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=list_phil_projects,
    ))
    registry.register(ToolDef(
        name="read_phil_project",
        description=("Read one of Phil's GitHub projects: its README (default), the file tree "
                     "(path='tree'), or a specific file (path='dir/file.py'). Use when a project of "
                     "his is relevant to what you are thinking about."),
        parameters={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository name from list_phil_projects, e.g. 'powers-of-zen'."},
                "path": {"type": "string", "description": "'' for README, 'tree' for the file list, or a file path."},
            },
            "required": ["repo"],
        },
        handler=read_phil_project,
    ))
