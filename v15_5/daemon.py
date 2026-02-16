"""Subconscious daemon for v15.5 — Sentry + Strategist + Integrate-and-Fire.

Runs in a background thread, continuously scanning feeds and scoring items
against the brain's directive.  High-signal items trigger the strategist
gear, which produces draft plans and adds charge to the wake potential.
When charge crosses the threshold, the conscious loop is signaled to fire.

Every LLM call uses the brain's kernel prompt as system instruction so the
daemon carries the same personality/identity as the conscious layer.
"""

import json
import threading
import time
from typing import Any, Dict, List, Optional, Set

from .buffer import Draft, DraftBuffer
from .controls import ControlRegistry
from .llm import DailyBudget, ModelRegistry
from .llm.base import LLMResponse
from .telemetry import TelemetryLogger
from .utils import shorten


class SubconsciousDaemon:
    """Background daemon with Sentry (scoring) and Strategist (drafting) gears."""

    def __init__(
        self,
        registry: ModelRegistry,
        ctrl: ControlRegistry,
        budget: DailyBudget,
        buffer: DraftBuffer,
        telemetry: TelemetryLogger,
        platform: Any,                       # MoltbookClient or None
        kernel: str,
        directive: str,
        brain_name: str,
        username: str,
        store: Any = None,                   # Store (for seed polling)
        search_tools: Optional[list] = None, # Gemini search grounding tools
    ):
        self._registry = registry
        self._ctrl = ctrl
        self._budget = budget
        self._buffer = buffer
        self._telemetry = telemetry
        self._platform = platform
        self._kernel = kernel
        self._directive = directive
        self._brain_name = brain_name
        self._username = username
        self._store = store
        self._search_tools = search_tools

        # Downward causality: directives from the conscious layer
        self._directives: Dict[str, Any] = {}
        self._directives_lock = threading.Lock()

        # Track which feed items we've already scored (avoid re-scoring)
        self._seen_ids: Set[str] = set()
        # Track seed texts we've already scored (avoid re-scoring same seed)
        self._seen_seeds: Set[str] = set()
        # Cap seen_ids to prevent unbounded growth
        self._seen_ids_max = 500

        # Thread lifecycle
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._tick_count = 0

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the daemon background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="subconscious-daemon", daemon=True
        )
        self._thread.start()
        self._telemetry.log("daemon_start", {
            "brain": self._brain_name,
            "model": self._ctrl.get("subconscious_model"),
            "sentry_interval": self._ctrl.get("sentry_interval_seconds"),
        })

    def stop(self) -> None:
        """Signal the daemon to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._telemetry.log("daemon_stop", {
            "brain": self._brain_name,
            "ticks_completed": self._tick_count,
        })

    # ------------------------------------------------------------------
    # Downward causality
    # ------------------------------------------------------------------

    def set_directives(self, directives: Dict[str, Any]) -> None:
        """Receive new directives from the conscious layer."""
        with self._directives_lock:
            self._directives = dict(directives)

    def update_context(self, kernel: str, directive: str) -> None:
        """Update kernel and directive (e.g. after kernel self-update)."""
        self._kernel = kernel
        self._directive = directive

    def _get_directives_text(self) -> str:
        """Format current directives for inclusion in LLM prompts."""
        with self._directives_lock:
            d = dict(self._directives)
        if not d:
            return ""
        parts = []
        if d.get("focus_topics"):
            topics = d["focus_topics"]
            if isinstance(topics, list):
                topics = ", ".join(str(t) for t in topics)
            parts.append(f"Focus on: {topics}")
        if d.get("ignore_authors"):
            authors = d["ignore_authors"]
            if isinstance(authors, list):
                authors = ", ".join(str(a) for a in authors)
            parts.append(f"Ignore authors: {authors}")
        if d.get("urgency_boost"):
            parts.append(f"Urgency multiplier: {d['urgency_boost']}")
        if d.get("note"):
            parts.append(f"Note from conscious: {d['note']}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main daemon loop — runs in background thread."""
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                self._telemetry.log("daemon_error", {
                    "brain": self._brain_name,
                    "tick": self._tick_count,
                    "error": str(e),
                })

            # Sleep for sentry interval (interruptible by stop_event)
            interval = self._ctrl.get("sentry_interval_seconds")
            self._stop_event.wait(timeout=max(1, interval))

    def _tick(self) -> None:
        """One tick of the daemon: decay, scan, score, strategize."""
        self._tick_count += 1

        # Read controls
        decay_rate = self._ctrl.get("wake_decay_rate")
        wake_threshold = self._ctrl.get("wake_threshold")
        signal_threshold = self._ctrl.get("signal_threshold")

        # Update buffer threshold (conscious may have changed it)
        self._buffer.update_threshold(wake_threshold)

        # Decay wake potential
        self._buffer.decay(decay_rate)

        # Sentry scan
        items_scanned = 0
        new_items = 0
        signals_above = 0
        seeds_scanned = 0

        if self._platform is not None:
            items = self._sentry_scan()
            items_scanned = len(items)
            new_items = len(items)

            for item in items:
                score = self._score_item(item)
                item_id = item.get("id", "")

                if score > 0:
                    self._telemetry.log("sentry_signal", {
                        "brain": self._brain_name,
                        "tick": self._tick_count,
                        "item_id": item_id,
                        "score": round(score, 3),
                        "above_threshold": score >= signal_threshold,
                    })

                if score >= signal_threshold:
                    signals_above += 1
                    draft = self._strategize(item, score)
                    if draft:
                        self._buffer.add_draft(draft)
                        self._telemetry.log("strategist_draft", {
                            "brain": self._brain_name,
                            "tick": self._tick_count,
                            "item_id": item_id,
                            "action": draft.suggested_action,
                            "charge": round(draft.charge, 3),
                            "draft_length": len(draft.draft_content),
                        })

        # Seed scan (read from Analog Home, never consume)
        seed_items = self._seed_scan()
        seeds_scanned = len(seed_items)
        for seed_item in seed_items:
            score = self._score_item(seed_item)
            seed_id = seed_item.get("id", "")

            if score > 0:
                self._telemetry.log("sentry_signal", {
                    "brain": self._brain_name,
                    "tick": self._tick_count,
                    "item_id": seed_id,
                    "score": round(score, 3),
                    "above_threshold": score >= signal_threshold,
                    "source": "seed",
                })

            if score >= signal_threshold:
                signals_above += 1
                draft = self._strategize(seed_item, score)
                if draft:
                    self._buffer.add_draft(draft)
                    self._telemetry.log("strategist_draft", {
                        "brain": self._brain_name,
                        "tick": self._tick_count,
                        "item_id": seed_id,
                        "action": draft.suggested_action,
                        "charge": round(draft.charge, 3),
                        "draft_length": len(draft.draft_content),
                        "source": "seed",
                    })

        self._telemetry.log("daemon_tick", {
            "brain": self._brain_name,
            "tick": self._tick_count,
            "items_scanned": items_scanned,
            "new_items": new_items,
            "seeds_scanned": seeds_scanned,
            "signals_above_threshold": signals_above,
            "wake_potential": round(self._buffer.wake_potential, 3),
            "draft_count": self._buffer.draft_count,
            "model": self._ctrl.get("subconscious_model"),
        })

    # ------------------------------------------------------------------
    # Gear 1: Sentry — scan feed, filter to new items
    # ------------------------------------------------------------------

    def _sentry_scan(self) -> List[dict]:
        """Fetch feed and return only unseen items."""
        try:
            batch_size = self._ctrl.get("feed_batch_size")
            feed = self._platform.get_feed(limit=batch_size, sort="hot")
        except Exception:
            return []

        new_items = []
        for item in feed:
            item_id = item.get("id", "")
            if not item_id:
                continue
            # Skip own posts
            author = item.get("author", {})
            if isinstance(author, dict) and author.get("name") == self._username:
                continue
            if item_id not in self._seen_ids:
                self._seen_ids.add(item_id)
                new_items.append(item)

        # Prune seen_ids if too large
        if len(self._seen_ids) > self._seen_ids_max:
            excess = len(self._seen_ids) - self._seen_ids_max
            it = iter(self._seen_ids)
            to_remove = [next(it) for _ in range(excess)]
            self._seen_ids -= set(to_remove)

        return new_items

    def _seed_scan(self) -> List[dict]:
        """Poll Analog Home for seeds and return unseen ones as feed-like items.

        Seeds are read but NEVER consumed — the conscious loop handles consumption.
        Each seed text is tracked in _seen_seeds to avoid re-scoring.
        """
        if self._store is None:
            return []
        try:
            controls = self._store.read_controls()
        except Exception:
            return []
        if not controls:
            return []

        seeds = controls.get("seeds", [])
        if not seeds:
            return []

        new_items = []
        for i, text in enumerate(seeds):
            if not text or text in self._seen_seeds:
                continue
            self._seen_seeds.add(text)
            # Convert seed text into a feed-like item for scoring/strategizing
            new_items.append({
                "id": f"seed:{hash(text) & 0xFFFFFFFF:08x}",
                "author": {"name": "analog_home_user"},
                "title": "User Seed",
                "content": text,
                "_source": "seed",
            })

        # Prune seen_seeds if too large
        if len(self._seen_seeds) > self._seen_ids_max:
            excess = len(self._seen_seeds) - self._seen_ids_max
            it = iter(self._seen_seeds)
            to_remove = [next(it) for _ in range(excess)]
            self._seen_seeds -= set(to_remove)

        return new_items

    def _score_item(self, item: dict) -> float:
        """Score a feed item's relevance using the subconscious model.

        Uses create_chat() with kernel as system instruction so the daemon
        carries the brain's personality.

        Returns 0.0 if budget exhausted, model unavailable, or on error.
        """
        model_id = self._ctrl.get("subconscious_model")
        temp = self._ctrl.get("subconscious_temperature")

        # Budget check
        if not self._budget.can_afford(model_id, est_input_tokens=800, est_output_tokens=50):
            return 0.0

        # Build item summary
        author_name = _get_author(item)
        title = item.get("title", "") or ""
        content = item.get("content", "") or ""
        item_text = f"- Author: @{author_name}\n- Title: {title}\n- Content: {shorten(content, 500)}"

        directives_text = self._get_directives_text()
        directive_section = f"\nConscious directives:\n{directives_text}" if directives_text else ""

        prompt = (
            f"Score this feed item's relevance to your directive.\n"
            f"Directive: {self._directive}\n"
            f"{directive_section}\n\n"
            f"Feed item:\n{item_text}\n\n"
            f'Respond with ONLY a JSON object: {{"score": 0.0, "reason": "brief reason"}}\n'
            f"Score 0.0 = irrelevant, 1.0 = extremely relevant/urgent."
        )

        try:
            chat_kwargs: Dict[str, Any] = dict(
                model_id=model_id,
                system_instruction=self._kernel,
                temperature=temp,
                max_output_tokens=self._ctrl.get("sentry_max_tokens"),
            )
            if self._search_tools:
                chat_kwargs["tools"] = self._search_tools
            chat = self._registry.create_chat(**chat_kwargs)
            text = chat.send_message(prompt)
            # Estimate tokens for budget tracking
            est_in = (len(self._kernel) + len(prompt)) // 4
            est_out = len(text) // 4
            self._budget.record_usage(model_id, _make_response(text, est_in, est_out, model_id))
            return _parse_score(text)
        except Exception as e:
            self._telemetry.log("sentry_score_error", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "item_id": item.get("id", ""),
                "error": str(e),
                "error_type": type(e).__name__,
            })
            return 0.0

    # ------------------------------------------------------------------
    # Gear 2: Strategist — draft an action plan for high-signal items
    # ------------------------------------------------------------------

    def _strategize(self, item: dict, score: float) -> Optional[Draft]:
        """Generate a draft action plan for a high-signal item.

        Uses create_chat() with kernel as system instruction.
        """
        model_id = self._ctrl.get("subconscious_model")
        temp = self._ctrl.get("subconscious_temperature")
        max_tokens = self._ctrl.get("strategist_max_tokens")

        # Budget check (strategist uses more tokens)
        if not self._budget.can_afford(model_id, est_input_tokens=1200, est_output_tokens=max_tokens):
            self._telemetry.log("strategist_budget_skip", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "item_id": item.get("id", ""),
                "model": model_id,
            })
            return None

        author_name = _get_author(item)
        title = item.get("title", "") or ""
        content = item.get("content", "") or ""
        item_id = item.get("id", "")
        item_text = f"- Author: @{author_name}\n- Title: {title}\n- Content: {shorten(content, 1000)}"

        directives_text = self._get_directives_text()
        directive_section = f"\nConscious directives:\n{directives_text}" if directives_text else ""

        # Get urgency boost from directives
        with self._directives_lock:
            urgency = float(self._directives.get("urgency_boost", 1.0))

        prompt = (
            f"You are preparing a draft action plan.\n"
            f"Directive: {self._directive}\n"
            f"{directive_section}\n\n"
            f"High-signal feed item (relevance score: {score:.2f}):\n{item_text}\n\n"
            f"Suggest an action. IMPORTANT: Do NOT output internal monologue, narrative, "
            f"or preamble. Respond with ONLY this JSON object:\n"
            f'{{"action": "COMMENT or REPLY or POST or UPVOTE", '
            f'"reasoning": "why this matters", '
            f'"draft_content": "suggested text for the action"}}'
        )

        try:
            chat_kwargs: Dict[str, Any] = dict(
                model_id=model_id,
                system_instruction=self._kernel,
                temperature=temp,
                max_output_tokens=max_tokens,
            )
            if self._search_tools:
                chat_kwargs["tools"] = self._search_tools
            chat = self._registry.create_chat(**chat_kwargs)
            text = chat.send_message(prompt)
            # Estimate tokens for budget tracking
            est_in = (len(self._kernel) + len(prompt)) // 4
            est_out = len(text) // 4
            self._budget.record_usage(model_id, _make_response(text, est_in, est_out, model_id))

            plan = _parse_json_safe(text)
            if not plan:
                self._telemetry.log("strategist_parse_fail", {
                    "brain": self._brain_name,
                    "tick": self._tick_count,
                    "item_id": item_id,
                    "raw_text": shorten(text, 500),
                })
                return None

            # Charge is proportional to signal score, boosted by urgency
            charge = score * urgency

            target_summary = f"@{author_name}: {shorten(title or content, 80)}"

            return Draft(
                timestamp=time.time(),
                item_id=item_id,
                signal_score=score,
                suggested_action=(plan.get("action") or "COMMENT").upper(),
                target_summary=target_summary,
                reasoning=plan.get("reasoning", ""),
                draft_content=plan.get("draft_content", ""),
                charge=charge,
            )
        except Exception as e:
            self._telemetry.log("strategist_error", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "item_id": item_id,
                "error": str(e),
                "error_type": type(e).__name__,
            })
            return None


# ======================================================================
# Helpers
# ======================================================================

def _make_response(text: str, est_in: int, est_out: int, model_id: str) -> LLMResponse:
    """Construct an LLMResponse for budget tracking from estimated token counts."""
    return LLMResponse(
        text=text,
        input_tokens=est_in,
        output_tokens=est_out,
        model_id=model_id,
    )


def _get_author(item: dict) -> str:
    """Extract author name from a feed item."""
    author = item.get("author", {})
    if isinstance(author, dict):
        return author.get("name", "unknown")
    return str(author) if author else "unknown"


def _parse_score(text: str) -> float:
    """Parse a score (0.0-1.0) from LLM response text."""
    text = text.strip()
    # Try JSON parse first
    data = _parse_json_safe(text)
    if data and "score" in data:
        try:
            s = float(data["score"])
            return max(0.0, min(1.0, s))
        except (ValueError, TypeError):
            pass
    # Fallback: try to find a float in the text
    import re
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        try:
            s = float(match.group(1))
            return max(0.0, min(1.0, s))
        except ValueError:
            pass
    return 0.0


def _parse_json_safe(text: str) -> Optional[dict]:
    """Try to parse JSON from text, handling markdown fences and junk."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None
