"""Deterministic memory-invariant checks (v19.3.1) — agent request, cycle 20, ask #2a.

The compressor templates emit an `[INVARIANTS & CONDITIONAL GATES]` block in every
compressed/deep memory entry. When those entries are folded again by the LLM, a
bullet can silently vanish or be reworded into something weaker. This module is
the out-of-band check the agent asked for:

  * after a fold, every invariant bullet in the INPUT entries must still appear in
    the OUTPUT; a missing one is appended back verbatim ("omitted" is detected by
    token overlap — heavy rewording counts as omission, so both versions are kept);
  * bullets get a stable id tag `[inv:xxxxxx]` (hash of the normalised text) so the
    agent can refer to them;
  * `retire_invariant(id, reason)` is the only way an invariant leaves: it is
    recorded in state["_retired_invariants"] and stripped from every tier, and the
    enforcer will not restore it again.

No LLM is involved anywhere here.
"""

import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

INV_HEADER = "[INVARIANTS & CONDITIONAL GATES]"
_HEADER_RE = re.compile(r"^\s*\[[A-Z][A-Z &/]+\]\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-•*]\s+(.*)$")
_TAG_RE = re.compile(r"\[inv:([0-9a-f]{6})\]\s*")
_STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "be", "for", "on", "as", "by",
         "with", "that", "this", "it", "its", "not", "no", "must", "should", "do", "if", "when", "from",
         "than", "into", "at", "any", "all", "vs"}
SURVIVAL_THRESHOLD = 0.6     # share of an input bullet's content tokens that must appear in some output bullet
WARN_BLOCK_CHARS = 4000      # telemetry warning when a block grows past this


def strip_tag(text: str) -> str:
    return _TAG_RE.sub("", text).strip()


def normalize(text: str) -> str:
    t = strip_tag(text).lower()
    t = re.sub(r"[*_`]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def inv_id(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:6]


def tokens(text: str) -> Set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9'-]*", normalize(text))
    return {w for w in words if w not in _STOP and len(w) > 1}


def split_sections(summary: str) -> Tuple[str, List[str], str]:
    """(text before the header, raw lines of the invariants section, text after)."""
    lines = (summary or "").replace("\r\n", "\n").split("\n")
    start = next((i for i, l in enumerate(lines) if l.strip() == INV_HEADER), None)
    if start is None:
        return summary or "", [], ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _HEADER_RE.match(lines[j]) and lines[j].strip() != INV_HEADER:
            end = j
            break
    return "\n".join(lines[:start]), lines[start + 1:end], "\n".join(lines[end:])


def parse_bullets(section_lines: Iterable[str]) -> List[str]:
    """Top-level bullets, each with its indented continuation lines folded in."""
    out: List[str] = []
    for line in section_lines:
        m = _BULLET_RE.match(line)
        if m and len(m.group(1)) == 0:
            out.append(m.group(2).strip())
        elif out and line.strip():
            out[-1] = out[-1] + " " + line.strip().lstrip("-•* ").strip()
    return [b for b in out if b and normalize(b) not in ("none", "none.")]


def parse_invariants(summary: str) -> List[Dict[str, str]]:
    _, sec, _ = split_sections(summary)
    return [{"id": inv_id(b), "text": strip_tag(b)} for b in parse_bullets(sec)]


def survives(bullet: str, candidates: List[str], threshold: float = SURVIVAL_THRESHOLD) -> bool:
    toks = tokens(bullet)
    if not toks:
        return True
    for c in candidates:
        if len(toks & tokens(c)) / len(toks) >= threshold:
            return True
    return False


def render_block(bullets: List[str]) -> str:
    if not bullets:
        return f"{INV_HEADER}\n- None"
    return INV_HEADER + "\n" + "\n".join(f"- [inv:{inv_id(b)}] {strip_tag(b)}" for b in bullets)


def enforce(input_entries: List[Dict[str, Any]], summary: str,
            retired_ids: Optional[Set[str]] = None) -> Tuple[str, Dict[str, Any]]:
    """Guarantee every input invariant survives the fold; drop retired ones; tag ids."""
    retired_ids = retired_ids or set()
    inputs: List[str] = []
    for e in input_entries:
        if e.get("summary"):
            inputs.extend(strip_tag(b) for b in parse_bullets(split_sections(e["summary"])[1]))
    pre, sec, post = split_sections(summary)
    out_bullets = [strip_tag(b) for b in parse_bullets(sec)]
    restored: List[str] = []
    for b in inputs:
        if inv_id(b) in retired_ids:
            continue
        if not survives(b, out_bullets):
            out_bullets.append(b)
            restored.append(b)
    removed = [b for b in out_bullets if inv_id(b) in retired_ids]
    out_bullets = [b for b in out_bullets if inv_id(b) not in retired_ids]
    seen: Set[str] = set()
    deduped: List[str] = []
    for b in out_bullets:
        if inv_id(b) not in seen:
            seen.add(inv_id(b))
            deduped.append(b)
    block = render_block(deduped)
    if not deduped and INV_HEADER not in (summary or ""):
        new_summary = summary  # nothing to keep and no section to rewrite — leave the text alone
    elif INV_HEADER in (summary or ""):
        new_summary = "\n".join(p for p in (pre.rstrip(), block, post.lstrip("\n")) if p)
    else:
        new_summary = block + "\n\n" + (summary or "")
    report = {"inputs": len(inputs), "kept": len(deduped), "restored": restored,
              "retired_removed": removed, "block_chars": len(block),
              "warn": len(block) > WARN_BLOCK_CHARS}
    return new_summary, report


# ----------------------------------------------------------------------
# State-level view + retirement
# ----------------------------------------------------------------------

def active_invariants(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for tier in ("deep", "compressed"):
        for entry in (state.get("memory_tiers") or {}).get(tier, []):
            for inv in parse_invariants(entry.get("summary", "")):
                if inv["id"] in seen:
                    continue
                seen.add(inv["id"])
                out.append({**inv, "tier": tier, "cycles": entry.get("cycles")})
    return out


def retire(state: Dict[str, Any], target_id: str, reason: str, cycle: Optional[int]) -> Optional[str]:
    """Strip the invariant from every tier's summary and record it. Returns its text."""
    target_id = (target_id or "").strip().lower().replace("inv:", "")
    found: Optional[str] = None
    tiers = state.get("memory_tiers") or {}
    for tier in ("compressed", "deep"):
        for entry in tiers.get(tier, []):
            s = entry.get("summary", "")
            pre, sec, post = split_sections(s)
            bullets = [strip_tag(b) for b in parse_bullets(sec)]
            keep = [b for b in bullets if inv_id(b) != target_id]
            if len(keep) != len(bullets):
                found = found or next(b for b in bullets if inv_id(b) == target_id)
                entry["summary"] = "\n".join(p for p in (pre.rstrip(), render_block(keep), post.lstrip("\n")) if p)
    if found is None:
        return None
    import datetime as _dt
    state.setdefault("_retired_invariants", []).append({
        "id": target_id, "text": found, "reason": (reason or "").strip()[:300], "cycle": cycle,
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="minutes"),
    })
    return found


def retired_ids(state: Dict[str, Any]) -> Set[str]:
    return {r.get("id", "") for r in state.get("_retired_invariants", [])}


def build_invariant_tools(registry: Any, state: Dict[str, Any], cycle_getter) -> None:
    from .tools import ToolDef

    def list_invariants(include_retired: bool = False) -> Dict[str, Any]:
        active = active_invariants(state)
        out: Dict[str, Any] = {"active": active, "active_count": len(active),
                               "note": "ids are the [inv:xxxxxx] tags in your compressed/deep memory; "
                                       "retire_invariant(id, reason) removes one for good"}
        if include_retired:
            out["retired"] = state.get("_retired_invariants", [])
        return out

    def retire_invariant(id: str, reason: str) -> Dict[str, Any]:
        if not (reason or "").strip():
            return {"error": "reason is required — say why this invariant no longer holds"}
        text = retire(state, id, reason, cycle_getter())
        if text is None:
            return {"error": f"no active invariant with id {id!r} (see list_invariants)"}
        return {"status": "retired", "id": id.replace("inv:", ""), "text": text}

    registry.register(ToolDef(
        name="list_invariants",
        description=("List the invariants and conditional gates currently held in your compressed/deep "
                     "memory, with their ids. The harness guarantees these survive every memory fold "
                     "until you retire them."),
        parameters={"type": "object", "properties": {
            "include_retired": {"type": "boolean", "description": "Also list retired invariants with reasons."}},
            "required": []},
        handler=list_invariants,
    ))
    registry.register(ToolDef(
        name="retire_invariant",
        description=("Retire an invariant that no longer holds (falsified, superseded, or obsolete). It is "
                     "removed from memory and will not be restored by the harness; the retirement and your "
                     "reason are kept on record. This is the only way an invariant leaves."),
        parameters={"type": "object", "properties": {
            "id": {"type": "string", "description": "The invariant's id, e.g. 'inv:3f9a1c' or '3f9a1c'."},
            "reason": {"type": "string", "description": "Why it no longer holds."}},
            "required": ["id", "reason"]},
        handler=retire_invariant,
        mode="write",
    ))
