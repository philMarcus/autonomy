"""Semantic recall (v19): embeddings of everything the agent has written or
thought, stored in Analog Home (pgvector) and searched by meaning.

Division of labour: this module computes embeddings (it has the Gemini key)
and the Analog Home API stores vectors and runs the nearest-neighbour query,
so Postgres stays reachable only through the API.

Pieces:
  Embedder          — gemini-embedding-2 at 768 dims, batched, budget-tracked
  RecallClient      — thin HTTP client for /embeddings/* and /recall
  sync_pending()    — embed artifacts the API reports as not yet embedded
                      (runs every cycle, bounded; and in a loop for backfill)
  embed_memory_file — embed a run's memory tiers so a *previous life* is
                      recallable after a clean-slate restart
  build_recall_tool — the `recall` planner tool
  CLI               — python -m autonomy.recall backfill <brain> [...]
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin

import requests

log = logging.getLogger(__name__)

EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 768
MAX_DOC_CHARS = 8000      # embedding input cap per document (well under the model limit)
EMBED_BATCH = 20          # documents per embed_content call
EMBED_RETRIES = 6         # on 429/5xx: sleep 5, 10, 20, 40, 80, 160 s
_RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL",
              "504", "DEADLINE_EXCEEDED", "timed out", "ReadTimeout")
DEFAULT_KINDS = ("artifact_body", "artifact_monologue")
# Mirrors the API's default: content + the agent's own kernel/dev-request history.
RECALL_TYPES = ["post", "comment", "reply", "image", "dream",
                "system_kernel_update", "system_dev_request"]


# ----------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------

class Embedder:
    """gemini-embedding-2 wrapper. Each document is its own Content so the
    API returns one vector per document (a bare list of strings collapses
    into a single multi-part content)."""

    def __init__(self, api_key: str, model: str = EMBED_MODEL, dim: int = EMBED_DIM,
                 budget: Any = None):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.dim = dim
        self._budget = budget
        self.calls = 0
        self.chars = 0
        self.retries = 0

    def _call(self, contents, task_type: str):
        """One embed_content call with exponential backoff on rate limits / 5xx."""
        from google.genai import types
        delay = 5.0
        for attempt in range(EMBED_RETRIES + 1):
            try:
                return self._client.models.embed_content(
                    model=self.model, contents=contents,
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
                )
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if attempt >= EMBED_RETRIES or not any(s in msg for s in _RETRYABLE):
                    raise
                log.warning("embed_content %s — retry %d/%d in %.0fs", msg[:80], attempt + 1, EMBED_RETRIES, delay)
                self.retries += 1
                time.sleep(delay)
                delay = min(delay * 2, 160.0)

    def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        from google.genai import types
        texts = [(t or "")[:MAX_DOC_CHARS] for t in texts]
        resp = self._call([types.Content(parts=[types.Part(text=t)]) for t in texts], task_type)
        vecs = [list(e.values) for e in resp.embeddings]
        if len(vecs) != len(texts):
            raise RuntimeError(f"embed_content returned {len(vecs)} vectors for {len(texts)} docs")
        self.calls += 1
        self.chars += sum(len(t) for t in texts)
        if self._budget is not None:
            try:
                from .llm.base import LLMResponse
                self._budget.record_usage(self.model, LLMResponse(
                    text="", input_tokens=sum(len(t) for t in texts) // 4,
                    output_tokens=0, model_id=self.model))
            except Exception:  # budget is best-effort
                pass
        return vecs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            out.extend(self._embed(texts[i:i + EMBED_BATCH], "RETRIEVAL_DOCUMENT"))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


# ----------------------------------------------------------------------
# HTTP client
# ----------------------------------------------------------------------

class RecallClient:
    def __init__(self, api_url: str, timeout: int = 60):
        self._base = api_url.rstrip("/") + "/"
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return urljoin(self._base, path.lstrip("/"))

    def pending(self, limit: int = 50, kinds=DEFAULT_KINDS, artifact_types=None) -> Dict[str, Any]:
        params = {"limit": limit, "kinds": ",".join(kinds),
                  "artifact_types": ",".join(artifact_types or RECALL_TYPES)}
        r = requests.get(self._url("embeddings/pending"), params=params, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def upsert(self, items: List[Dict[str, Any]]) -> int:
        total = 0
        for i in range(0, len(items), 100):
            r = requests.post(self._url("embeddings"), json={"items": items[i:i + 100]},
                              timeout=self._timeout)
            r.raise_for_status()
            total += int(r.json().get("upserted", 0))
        return total

    def recall(self, vector: List[float], k: int = 8, kinds=None, artifact_types=None,
               run_id: str = "", exclude_run_id: str = "", min_score: float = 0.0,
               snippet_chars: int = 600) -> List[Dict[str, Any]]:
        body = {"embedding": vector, "k": k, "kinds": list(kinds or []),
                "artifact_types": list(artifact_types or []), "run_id": run_id,
                "exclude_run_id": exclude_run_id, "min_score": min_score,
                "snippet_chars": snippet_chars}
        r = requests.post(self._url("recall"), json=body, timeout=self._timeout)
        r.raise_for_status()
        return r.json().get("results", [])

    def stats(self) -> Dict[str, Any]:
        r = requests.get(self._url("embeddings/stats"), timeout=self._timeout)
        r.raise_for_status()
        return r.json()


# ----------------------------------------------------------------------
# Document shaping + sync
# ----------------------------------------------------------------------

def document_texts(art: Dict[str, Any]) -> Dict[str, str]:
    """The embeddable texts of an artifact, keyed by kind. Empty fields are omitted."""
    out: Dict[str, str] = {}
    body = (art.get("body_markdown") or "").strip()
    title = (art.get("title") or "").strip()
    if body:
        out["artifact_body"] = (f"{title}\n\n{body}" if title and title not in body[:200] else body)[:MAX_DOC_CHARS]
    mono = (art.get("monologue_public") or "").strip()
    if mono:
        out["artifact_monologue"] = mono[:MAX_DOC_CHARS]
    return out


def sync_pending(client: RecallClient, embedder: Embedder, limit: int = 50,
                 kinds=DEFAULT_KINDS, artifact_types=None) -> Dict[str, int]:
    """Embed up to `limit` artifacts the API reports as unembedded.

    Returns {"embedded": docs upserted, "artifacts": rows processed,
             "remaining": artifacts still pending after this pass}.
    """
    page = client.pending(limit=limit, kinds=kinds, artifact_types=artifact_types)
    rows = page.get("items", [])
    docs: List[Dict[str, Any]] = []
    for art in rows:
        texts = document_texts(art)
        for kind in art.get("missing_kinds", []):
            if kind in texts:
                docs.append({
                    "artifact_id": art["id"], "kind": kind, "brain": art.get("brain", ""),
                    "run_id": art.get("run_id", ""), "cycle": art.get("cycle"),
                    "text": texts[kind], "model": embedder.model,
                })
    if docs:
        vectors = embedder.embed_documents([d["text"] for d in docs])
        for d, v in zip(docs, vectors):
            d["embedding"] = v
        client.upsert(docs)
    remaining = int(page.get("remaining", 0)) - len(rows) if docs else int(page.get("remaining", 0))
    return {"embedded": len(docs), "artifacts": len(rows), "remaining": max(0, remaining)}


def memory_documents(state: Dict[str, Any], run_id: str, brain: str) -> List[Dict[str, Any]]:
    """Memory tiers + post-memory tiers of a run as embeddable documents."""
    docs: List[Dict[str, Any]] = []
    for tier_key, kind in (("memory_tiers", "memory_note"), ("post_tiers", "post_memory")):
        tiers = state.get(tier_key) or {}
        for tier_name, entries in tiers.items():
            for i, entry in enumerate(entries or []):
                if not isinstance(entry, dict):
                    continue
                text = (entry.get("note") or entry.get("summary") or entry.get("text") or "").strip()
                if not text:
                    continue
                cycle = entry.get("cycle")
                label = entry.get("cycles") or (f"cycle {cycle}" if cycle is not None else "undated")
                docs.append({
                    "source_ref": f"memory:{run_id}:{tier_name}:{i}", "kind": kind,
                    "brain": brain, "run_id": run_id,
                    "cycle": cycle if isinstance(cycle, int) else None,
                    "text": f"[{tier_name} memory, {label}] {text}"[:MAX_DOC_CHARS],
                })
    return docs


def embed_memory_file(client: RecallClient, embedder: Embedder, path: str,
                      run_id: str = "", brain: str = "") -> int:
    """Embed a memories.json (current or archived) so its notes are recallable."""
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    run_id = run_id or state.get("_session_id", "") or "unknown"
    docs = memory_documents(state, run_id, brain)
    if not docs:
        return 0
    vectors = embedder.embed_documents([d["text"] for d in docs])
    for d, v in zip(docs, vectors):
        d["embedding"] = v
        d["model"] = embedder.model
    return client.upsert(docs)


# ----------------------------------------------------------------------
# The planner tool
# ----------------------------------------------------------------------

def _run_label(run_id: str, current_run_id: str) -> str:
    if run_id and run_id == current_run_id:
        return "this run"
    return f"previous life {run_id[:8]}" if run_id else "previous life"


def format_results(results: List[Dict[str, Any]], current_run_id: str) -> List[Dict[str, Any]]:
    out = []
    for r in results:
        out.append({
            "score": r.get("score"),
            "when": (r.get("created_at") or "")[:10],
            "cycle": r.get("cycle"),
            "run": _run_label(r.get("run_id", ""), current_run_id),
            "type": r.get("artifact_type") or r.get("kind", ""),
            "kind": r.get("kind", ""),
            "title": r.get("title", ""),
            "artifact_id": r.get("artifact_id"),
            "snippet": r.get("snippet", ""),
        })
    return out


def build_recall_tool(registry: Any, client: RecallClient, embedder: Embedder,
                      state: Dict[str, Any]) -> None:
    """Register `recall` on a ToolRegistry."""
    from .tools import ToolDef

    def recall(query: str, k: int = 6, scope: str = "all", kinds: str = "") -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"error": "query is required"}
        k = max(1, min(int(k or 6), 20))
        current = state.get("_session_id", "") or ""
        scope = (scope or "all").strip().lower()
        kind_map = {"body": "artifact_body", "monologue": "artifact_monologue",
                    "memory": "memory_note", "posts": "post_memory"}
        kind_list = [kind_map[s.strip()] for s in kinds.split(",") if s.strip() in kind_map] if kinds else []
        try:
            vec = embedder.embed_query(query)
            results = client.recall(
                vec, k=k, kinds=kind_list,
                run_id=current if scope == "current" else "",
                exclude_run_id=current if scope == "previous" else "",
            )
        except Exception as e:  # noqa: BLE001
            return {"error": f"recall failed: {str(e)[:200]}"}
        return {"query": query, "scope": scope, "results": format_results(results, current)}

    registry.register(ToolDef(
        name="recall",
        description=(
            "Semantic recall over everything you have ever written or thought — posts, "
            "comments, replies, image essays, dreams, kernel rewrites, your internal "
            "monologues, and the memory notes of previous lives (earlier runs, before "
            "your memory was reset). Searches by meaning, not keyword; use search_history "
            "for exact phrases. Results say which run they came from."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you want to remember, in natural language."},
                "k": {"type": "integer", "description": "How many results (1-20). Default 6."},
                "scope": {"type": "string",
                          "description": "all (default), previous (earlier runs only), or current (this run only)."},
                "kinds": {"type": "string",
                          "description": "Optional comma list to narrow: body, monologue, memory, posts. Default: everything."},
            },
            "required": ["query"],
        },
        handler=recall,
    ))


# ----------------------------------------------------------------------
# CLI: python -m autonomy.recall backfill <brain> [--memory FILE] [--run-id ID]
# ----------------------------------------------------------------------

def _resolve_env(brain: str):
    from .config import load_dotenv, brain_env_prefix
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(root, ".env"))
    prefix = brain_env_prefix(brain)
    key = os.environ.get(f"{prefix}_GEMINI_API_KEY", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()
    api = os.environ.get(f"{prefix}_ANALOG_HOME_API_URL", "").strip() or os.environ.get("ANALOG_HOME_API_URL", "").strip()
    return key, api


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m autonomy.recall",
                                 description="Semantic recall maintenance for an Analog Home brain.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    bf = sub.add_parser("backfill", help="Embed every pending artifact (and optionally a memories.json).")
    bf.add_argument("brain")
    bf.add_argument("--api", default="", help="Analog Home API URL (default: from .env)")
    bf.add_argument("--batch", type=int, default=50, help="artifacts per API page (max 200)")
    bf.add_argument("--memory", default="", help="path to a memories.json to embed as memory notes")
    bf.add_argument("--run-id", default="", help="run id to tag the memory file with (default: its _session_id)")
    bf.add_argument("--max-artifacts", type=int, default=0, help="stop after this many (0 = all)")
    st = sub.add_parser("stats", help="Show embedding coverage.")
    st.add_argument("brain")
    st.add_argument("--api", default="")
    q = sub.add_parser("query", help="Run one recall query from the command line.")
    q.add_argument("brain")
    q.add_argument("text")
    q.add_argument("--api", default="")
    q.add_argument("-k", type=int, default=6)
    args = ap.parse_args(argv)

    key, api = _resolve_env(args.brain)
    api = args.api or api
    if not api:
        print("No Analog Home API URL (set {PREFIX}_ANALOG_HOME_API_URL or --api)")
        return 2
    client = RecallClient(api)

    if args.cmd == "stats":
        print(json.dumps(client.stats(), indent=2))
        return 0

    if not key:
        print("No Gemini API key for this brain")
        return 2
    embedder = Embedder(api_key=key)

    if args.cmd == "query":
        vec = embedder.embed_query(args.text)
        for r in client.recall(vec, k=args.k):
            print(f"[{r['score']:.3f}] {str(r.get('created_at'))[:10]} c{r.get('cycle')} {r.get('artifact_type') or r.get('kind')} "
                  f"run={r.get('run_id','')[:8]} | {r.get('title','')[:60]}\n    {r.get('snippet','')[:240].replace(chr(10),' ')}")
        return 0

    # backfill
    t0 = time.time()
    done = 0
    failures = 0
    while True:
        try:
            res = sync_pending(client, embedder, limit=max(1, min(args.batch, 200)))
            failures = 0
        except Exception as e:  # noqa: BLE001
            failures += 1
            if failures > 5:
                print(f"  giving up after {failures} consecutive failures: {str(e)[:200]}")
                return 1
            wait = 30 * failures
            print(f"  page failed ({str(e)[:120]}) — retrying in {wait}s")
            time.sleep(wait)
            continue
        done += res["artifacts"]
        print(f"  embedded {res['embedded']:3d} docs from {res['artifacts']:3d} artifacts | "
              f"remaining {res['remaining']} | {embedder.chars:,} chars so far | {time.time()-t0:.0f}s")
        if res["artifacts"] == 0 or res["remaining"] == 0:
            break
        if args.max_artifacts and done >= args.max_artifacts:
            break
    if args.memory:
        n = embed_memory_file(client, embedder, args.memory, run_id=args.run_id, brain=args.brain)
        print(f"  embedded {n} memory documents from {args.memory}")
    est = embedder.chars / 4 / 1e6 * 0.20
    print(f"done: {done} artifacts, {embedder.calls} embed calls ({embedder.retries} retries), "
          f"~{embedder.chars/4:,.0f} tokens (~${est:.2f})")
    print(json.dumps(client.stats(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
