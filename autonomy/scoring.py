"""Structured sentry scoring rubric for v16.2.

Replaces free-form 0.0-1.0 float scoring with a multi-criteria rubric
using 0-3 integer scales. Line-based output format works across all models
including small local models that can't use json_mode.

Criteria:
    relevance (0-3):     How relevant to the directive
    novelty (0-3):       How new or surprising
    actionability (0-3): Can the agent meaningfully engage

Final score = weighted average normalized to 0.0-1.0.
"""

import re
from typing import Dict, List, Optional, Tuple

# Default weights (overridden by ControlRegistry at runtime)
DEFAULT_WEIGHTS = {
    "relevance": 0.45,
    "novelty": 0.30,
    "actionability": 0.25,
}

CRITERIA = [
    {
        "name": "relevance",
        "question": "How relevant is this to your directive?",
        "anchors": {
            0: "unrelated",
            1: "tangentially related",
            2: "directly relevant",
            3: "core topic",
        },
    },
    {
        "name": "novelty",
        "question": "How new or surprising is this?",
        "anchors": {
            0: "generic/repetitive",
            1: "common knowledge",
            2: "recent development",
            3: "breaking/unprecedented",
        },
    },
    {
        "name": "actionability",
        "question": "Can you meaningfully engage with this?",
        "anchors": {
            0: "nothing to add",
            1: "could acknowledge",
            2: "could contribute insight",
            3: "must respond",
        },
    },
]

CRITERION_NAMES = [c["name"] for c in CRITERIA]


def build_sentry_prompt(
    item_text: str,
    directive: str,
    directives_text: str = "",
) -> str:
    """Build a structured scoring prompt with rubric criteria and anchors.

    The output format is line-based (not JSON) so that even small local
    models can comply reliably.
    """
    directive_section = ""
    if directives_text:
        directive_section = f"\nConscious directives:\n{directives_text}\n"

    criteria_block = ""
    for c in CRITERIA:
        anchors = "  ".join(f"{k} = {v}" for k, v in c["anchors"].items())
        criteria_block += f"{c['name']} ({c['question']}):\n  {anchors}\n"

    return (
        f"Score this feed item on three criteria. "
        f"For each, answer with ONLY a number 0-3.\n\n"
        f"Directive: {directive}\n"
        f"{directive_section}\n"
        f"Feed item:\n{item_text}\n\n"
        f"CRITERIA:\n{criteria_block}\n"
        f"RESPOND WITH EXACTLY THIS FORMAT:\n"
        f"relevance: <0-3>\n"
        f"novelty: <0-3>\n"
        f"actionability: <0-3>\n"
        f"reason: <one sentence>\n"
    )


def build_batch_sentry_prompt(
    items: List[str],
    directive: str,
    directives_text: str = "",
) -> str:
    """Build a batch scoring prompt for multiple feed items in one LLM call.

    Each item_text should be pre-formatted (Author/Title/Content).
    Returns a prompt that asks for scores for all items at once.
    """
    directive_section = ""
    if directives_text:
        directive_section = f"\nConscious directives:\n{directives_text}\n"

    criteria_block = ""
    for c in CRITERIA:
        anchors = "  ".join(f"{k} = {v}" for k, v in c["anchors"].items())
        criteria_block += f"  {c['name']}: {anchors}\n"

    items_block = ""
    for i, item_text in enumerate(items, 1):
        items_block += f"\n--- ITEM {i} ---\n{item_text}\n"

    response_format = ""
    for i in range(1, len(items) + 1):
        response_format += f"\nITEM {i}:\nrelevance: <0-3>\nnovelty: <0-3>\nactionability: <0-3>\n"

    return (
        f"Score each feed item on three criteria (0-3 each).\n\n"
        f"Directive: {directive}\n"
        f"{directive_section}\n"
        f"Criteria:\n{criteria_block}\n"
        f"{items_block}\n"
        f"Respond in EXACTLY this format:{response_format}"
    )


def parse_batch_rubric_response(text: str, num_items: int) -> List[Dict]:
    """Parse batch rubric scores from model output.

    Splits by ITEM markers, then parses each chunk with the single-item parser.
    Returns a list of dicts (one per item). Missing items get all-zero scores.
    """
    results: List[Dict] = []

    # Split by ITEM markers
    chunks = re.split(r'ITEM\s+(\d+)\s*:', text, flags=re.IGNORECASE)
    # chunks alternates: [preamble, "1", chunk1_text, "2", chunk2_text, ...]

    parsed_by_index: Dict[int, Dict] = {}
    for i in range(1, len(chunks) - 1, 2):
        try:
            idx = int(chunks[i]) - 1  # 0-based
            chunk_text = chunks[i + 1]
            parsed_by_index[idx] = parse_rubric_response(chunk_text)
        except (ValueError, IndexError):
            pass

    # Build result list, filling missing items with zeros
    for i in range(num_items):
        if i in parsed_by_index:
            results.append(parsed_by_index[i])
        else:
            results.append({name: 0 for name in CRITERION_NAMES} | {"reason": "", "raw": ""})

    return results


def build_simple_batch_prompt(
    items: List[str],
    directive: str,
    directives_text: str = "",
) -> str:
    """Simplified batch scoring prompt — one number (0-9) per item.

    Designed for models that can't handle the multi-criterion format.
    """
    directive_section = ""
    if directives_text:
        directive_section = f"\nFocus: {directives_text}\n"

    items_block = ""
    for i, item_text in enumerate(items, 1):
        items_block += f"{i}. {item_text}\n"

    return (
        f"Score each item 0-9 on relevance to this directive: {directive}\n"
        f"{directive_section}\n"
        f"Scale: 0-2 = irrelevant/noise, 3-5 = tangential, 6-7 = relevant, 8-9 = core topic\n\n"
        f"{items_block}\n"
        f"Reply with ONLY one number per line (no text, no labels):\n"
    )


def parse_simple_batch_response(text: str, num_items: int) -> List[Dict]:
    """Parse simple 0-9 scores (one per line) into rubric-compatible dicts."""
    import re
    numbers = re.findall(r'\b(\d)\b', text)
    results = []
    for i in range(num_items):
        if i < len(numbers):
            score_9 = min(9, max(0, int(numbers[i])))
            # Map 0-9 to 0-3 per criterion (approximate)
            score_3 = round(score_9 / 3.0)
            score_3 = min(3, score_3)
            results.append({
                "relevance": score_3,
                "novelty": score_3,
                "actionability": score_3,
                "reason": "",
                "raw": text[:200],
            })
        else:
            results.append({name: 0 for name in CRITERION_NAMES} | {"reason": "", "raw": ""})
    return results


def parse_rubric_response(text: str) -> Dict:
    """Parse rubric scores from model output.

    Returns dict with keys: relevance, novelty, actionability, reason, raw.
    All scores are ints 0-3. Missing scores default to 0.

    Parsing strategy (multiple fallbacks):
    1. Line-based regex: ``criterion_name: digit``
    2. Ordered digit extraction: first three 0-3 digits found
    3. JSON fallback: try to parse as JSON with score keys
    4. Default: all zeros (item safely skipped)
    """
    scores: Dict[str, int] = {}
    reason = ""

    # --- Strategy 1: named criterion lines ---
    for name in CRITERION_NAMES:
        m = re.search(
            rf'{name}\s*[:=]\s*([0-3])',
            text,
            re.IGNORECASE,
        )
        if m:
            scores[name] = int(m.group(1))

    # Extract reason
    m_reason = re.search(r'reason\s*[:=]\s*(.+)', text, re.IGNORECASE)
    if m_reason:
        reason = m_reason.group(1).strip()

    # --- Strategy 2: ordered digits if we're missing any ---
    if len(scores) < len(CRITERION_NAMES):
        digits = re.findall(r'\b([0-3])\b', text)
        # Filter out digits that are part of larger numbers
        clean_digits = []
        for m in re.finditer(r'(?<!\d)([0-3])(?!\d)', text):
            clean_digits.append(int(m.group(1)))
        if len(clean_digits) >= len(CRITERION_NAMES):
            for i, name in enumerate(CRITERION_NAMES):
                if name not in scores:
                    scores[name] = clean_digits[i]

    # --- Strategy 3: JSON fallback ---
    if len(scores) < len(CRITERION_NAMES):
        try:
            import json
            # Try to find JSON object in text
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(text[start:end + 1])
                for name in CRITERION_NAMES:
                    if name not in scores and name in data:
                        val = int(data[name])
                        scores[name] = max(0, min(3, val))
                if not reason and "reason" in data:
                    reason = str(data["reason"])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # --- Strategy 4: default to 0 ---
    for name in CRITERION_NAMES:
        if name not in scores:
            scores[name] = 0

    return {
        **scores,
        "reason": reason,
        "raw": text[:500],
    }


def compute_score(
    criteria_scores: Dict[str, int],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Weighted average of criteria scores, normalized to 0.0-1.0.

    Each criterion is scored 0-3, so score/3 gives 0.0-1.0 per criterion.
    Weights are normalized (divided by sum) so they don't need to sum to 1.0.

    Examples:
        All 0s -> 0.0
        All 1s -> ~0.33  (below 0.6 threshold)
        All 2s -> ~0.67  (triggers strategist)
        All 3s -> 1.0
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Normalize weights
    total_weight = sum(weights.get(name, 0.0) for name in CRITERION_NAMES)
    if total_weight <= 0:
        total_weight = 1.0

    score = 0.0
    for name in CRITERION_NAMES:
        w = weights.get(name, 0.0) / total_weight
        s = criteria_scores.get(name, 0) / 3.0
        score += w * s

    return round(max(0.0, min(1.0, score)), 4)


def weights_from_controls(ctrl) -> Dict[str, float]:
    """Read rubric weights from a ControlRegistry instance."""
    return {
        "relevance": float(ctrl.get("sentry_weight_relevance")),
        "novelty": float(ctrl.get("sentry_weight_novelty")),
        "actionability": float(ctrl.get("sentry_weight_actionability")),
    }
