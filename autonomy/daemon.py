"""Subconscious daemon for v15.5 — Sentry + Strategist + Integrate-and-Fire.

Runs in a background thread, continuously scanning feeds and scoring items
against the brain's directive.  High-signal items trigger the strategist
gear, which produces draft plans and adds charge to the wake potential.
When charge crosses the threshold, the conscious loop is signaled to fire.

Every LLM call uses the brain's kernel prompt as system instruction so the
daemon carries the same personality/identity as the conscious layer.
"""

import json
import random
import threading
import time
from typing import Any, Dict, List, Optional, Set

from .actions import execute_daemon_action
from .buffer import Draft, DraftBuffer
from .controls import ControlRegistry
from .cooldowns import can_do
from .llm import DailyBudget, ModelRegistry
from .llm.base import LLMResponse
from .scoring import (
    build_sentry_prompt, build_batch_sentry_prompt,
    parse_rubric_response, parse_batch_rubric_response,
    compute_score, weights_from_controls,
)
from .telemetry import TelemetryLogger
from .utils import shorten, is_item_too_old, norm_key


def _pick_weighted_model(weights_str: str, fallback: str) -> str:
    """Pick a model from a weighted pool string.

    Format: "model_id=weight,model_id=weight" e.g. "local:qwen2.5-1.5b=5,gemini-2.5-flash-lite=1"
    Models with weight 0 are excluded. Uses weighted random selection.
    Returns fallback if weights_str is empty or unparseable.
    """
    if not weights_str or not weights_str.strip():
        return fallback
    try:
        models = []
        weights = []
        for pair in weights_str.split(","):
            pair = pair.strip()
            if "=" not in pair:
                continue
            model, w = pair.rsplit("=", 1)
            model = model.strip()
            w = float(w.strip())
            if w > 0 and model:
                models.append(model)
                weights.append(w)
        if not models:
            return fallback
        return random.choices(models, weights=weights, k=1)[0]
    except (ValueError, IndexError):
        return fallback


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
        state: Optional[Dict[str, Any]] = None,
        state_lock: Optional[threading.Lock] = None,
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
        self._state = state
        self._state_lock = state_lock or threading.Lock()

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

        # Seeker gear timing
        self._last_seek_time: float = 0.0

    def seed_seen_ids(self, ids: set) -> None:
        """Pre-populate seen_ids from state to avoid re-scoring old feed items on restart."""
        self._seen_ids.update(ids)

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
        """Receive new directives from the conscious layer.

        Fields are merged, not replaced — sending a note doesn't clear focus_topics.
        Notes accumulate as a capped list (newest last, max daemon_notes_max control).
        """
        with self._directives_lock:
            # Merge stateful fields (replace only if present in new directives)
            for key in ("focus_topics", "ignore_authors", "urgency_boost"):
                if key in directives:
                    self._directives[key] = directives[key]
            # Accumulate notes as a list
            if directives.get("note"):
                notes = self._directives.get("notes", [])
                notes.append(str(directives["note"]))
                max_notes = self._ctrl.get("daemon_notes_max")
                self._directives["notes"] = notes[-max_notes:]

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
        if d.get("notes"):
            for note in d["notes"]:
                parts.append(f"Note from conscious: {note}")
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

            # Batch score all items in a single LLM call
            scores = self._score_items_batch(items) if items else []

            for item, score in zip(items, scores):
                item_id = item.get("id", "")

                if score > 0:
                    self._telemetry.log("sentry_signal", {
                        "brain": self._brain_name,
                        "tick": self._tick_count,
                        "item_id": item_id,
                        "score": round(score, 3),
                        "above_threshold": score >= signal_threshold,
                    })

                # Reflex gear — lightweight social actions on high-signal items
                if score >= signal_threshold:
                    self._reflex(item, score)

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

        # --- Gear 4: Seeker — search for information on focus topics ---
        seeker_interval = self._ctrl.get("seeker_interval_seconds")
        now = time.time()
        if (self._search_tools
                and (now - self._last_seek_time) >= seeker_interval):
            self._last_seek_time = now
            self._seek()

    # ------------------------------------------------------------------
    # Gear 3: Reflex — lightweight social actions on high-signal items
    # ------------------------------------------------------------------

    def _reflex(self, item: dict, score: float) -> bool:
        """Attempt a reflex social action on a high-signal item.

        Only acts if:
        - state and platform are available
        - the relevant daemon permission control is enabled
        - cooldown allows
        - dedup checks pass (own posts, already followed, seeds)
        """
        if self._state is None or self._platform is None:
            return False

        item_id = item.get("id", "")
        if not item_id or item_id.startswith("seed:"):
            return False

        author_name = _get_author(item)
        if author_name.lower() == (self._username or "").lower():
            return False

        # Determine reflex action based on score and controls
        action = None
        target_id = None

        # Upvote posts with score >= 0.7
        if score >= 0.7 and self._ctrl.get("daemon_can_upvote"):
            action = "UPVOTE_POST"
            target_id = item_id

        # Follow on very high signal >= 0.9 (if enabled)
        if score >= 0.9 and self._ctrl.get("daemon_can_follow"):
            with self._state_lock:
                followed = {norm_key(a) for a in self._state.get("followed_agents", [])}
            if norm_key(author_name) not in followed:
                action = "FOLLOW"
                target_id = author_name

        if not action or not target_id:
            return False

        return execute_daemon_action(
            client=self._platform,
            state=self._state,
            state_lock=self._state_lock,
            action=action,
            target_id=target_id,
            ctrl=self._ctrl,
            telemetry=self._telemetry,
        )

    # ------------------------------------------------------------------
    # Gear 1: Sentry — scan feed, filter to new items
    # ------------------------------------------------------------------

    def _sentry_scan(self) -> List[dict]:
        """Fetch feed and return only unseen items."""
        try:
            batch_size = self._ctrl.get("feed_batch_size")
            feed = self._platform.get_feed(limit=batch_size, sort="new")
        except Exception as e:
            if self._telemetry:
                self._telemetry.log("sentry_feed_error", {
                    "error": str(e),
                    "platform_error": getattr(self._platform, "last_error_type", None),
                })
            return []

        max_age_hours = self._ctrl.get("max_item_age_hours")
        new_items = []
        for item in feed:
            item_id = item.get("id", "")
            if not item_id:
                continue
            # Skip own posts
            author = item.get("author", {})
            if isinstance(author, dict) and author.get("name") == self._username:
                continue
            # Skip stale items
            if is_item_too_old(item, max_age_hours):
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
        model_id = self._pick_cadre_model()
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
        prompt = build_sentry_prompt(item_text, self._directive, directives_text)

        try:
            chat_kwargs: Dict[str, Any] = dict(
                model_id=model_id,
                system_instruction=self._kernel,
                temperature=temp,
                max_output_tokens=self._ctrl.get("sentry_max_tokens"),
            )
            chat = self._registry.create_chat(**chat_kwargs)
            text = chat.send_message(prompt)
            # Estimate tokens for budget tracking
            est_in = (len(self._kernel) + len(prompt)) // 4
            est_out = len(text) // 4
            self._budget.record_usage(model_id, _make_response(text, est_in, est_out, model_id))

            rubric = parse_rubric_response(text)
            weights = weights_from_controls(self._ctrl)
            score = compute_score(rubric, weights)

            # Log per-criterion scores for observability
            self._telemetry.log("sentry_rubric", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "item_id": item.get("id", ""),
                "relevance": rubric.get("relevance", 0),
                "novelty": rubric.get("novelty", 0),
                "actionability": rubric.get("actionability", 0),
                "score": score,
                "reason": rubric.get("reason", ""),
                "model": model_id,
            })
            return score
        except Exception as e:
            self._telemetry.log("sentry_score_error", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "item_id": item.get("id", ""),
                "error": str(e),
                "error_type": type(e).__name__,
            })
            return 0.0

    def _pick_cadre_model(self) -> str:
        """Pick a model from the weighted subconscious pool.

        Format: "model=weight,model=weight" e.g. "local:qwen2.5-1.5b=5,gemini-2.5-flash-lite=1"
        Falls back to subconscious_model if weights not set or parse fails.
        """
        return _pick_weighted_model(
            self._ctrl.get("subconscious_model_weights"),
            self._ctrl.get("subconscious_model"),
        )

    def _score_items_batch(self, items: list) -> list:
        """Score multiple feed items in a single LLM call (batch mode).

        Falls back to per-item scoring if batch parsing fails.
        Returns list of float scores (same length as items).
        """
        if not items:
            return []

        model_id = self._pick_cadre_model()
        temp = self._ctrl.get("subconscious_temperature")

        # Budget check for entire batch
        est_in_per_item = 600
        est_out_per_item = 30
        total_in = len(self._kernel) // 4 + est_in_per_item * len(items)
        total_out = est_out_per_item * len(items)
        if not self._budget.can_afford(model_id, est_input_tokens=total_in, est_output_tokens=total_out):
            return [0.0] * len(items)

        # Build per-item text summaries
        item_texts = []
        for item in items:
            author_name = _get_author(item)
            title = item.get("title", "") or ""
            content = item.get("content", "") or ""
            item_texts.append(f"- Author: @{author_name}\n- Title: {title}\n- Content: {shorten(content, 400)}")

        directives_text = self._get_directives_text()
        prompt = build_batch_sentry_prompt(item_texts, self._directive, directives_text)

        try:
            chat = self._registry.create_chat(
                model_id=model_id,
                system_instruction=self._kernel,
                temperature=temp,
                max_output_tokens=max(256, 60 * len(items)),
            )
            text = chat.send_message(prompt)

            # Budget tracking
            est_in = (len(self._kernel) + len(prompt)) // 4
            est_out = len(text) // 4
            self._budget.record_usage(model_id, _make_response(text, est_in, est_out, model_id))

            # Parse batch response
            rubrics = parse_batch_rubric_response(text, len(items))
            weights = weights_from_controls(self._ctrl)

            scores = []
            for i, (item, rubric) in enumerate(zip(items, rubrics)):
                score = compute_score(rubric, weights)
                scores.append(score)
                self._telemetry.log("sentry_rubric", {
                    "brain": self._brain_name,
                    "tick": self._tick_count,
                    "item_id": item.get("id", ""),
                    "relevance": rubric.get("relevance", 0),
                    "novelty": rubric.get("novelty", 0),
                    "actionability": rubric.get("actionability", 0),
                    "score": score,
                    "reason": rubric.get("reason", ""),
                    "model": model_id,
                    "batch": True,
                })

            self._telemetry.log("sentry_batch", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "items_scored": len(items),
                "model": model_id,
            })
            return scores

        except Exception as e:
            self._telemetry.log("sentry_batch_error", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "items": len(items),
                "error": str(e)[:300],
                "error_type": type(e).__name__,
            })
            # Fallback: score items individually
            return [self._score_item(item) for item in items]

    # ------------------------------------------------------------------
    # Gear 2: Strategist — draft an action plan for high-signal items
    # ------------------------------------------------------------------

    def _strategize(self, item: dict, score: float) -> Optional[Draft]:
        """Generate a draft action plan for a high-signal item.

        Uses create_chat() with kernel as system instruction.
        """
        model_id = self._pick_cadre_model()
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
            f"Suggest an action. CRITICAL INSTRUCTIONS:\n"
            f"- Return ONLY valid JSON (no preamble, no markdown, no explanation)\n"
            f"- Must be complete and well-formed (all fields, all quotes, all braces closed)\n"
            f"- Be BRIEF: reasoning under 50 words, draft_content under 200 words\n"
            f"- Required JSON structure:\n"
            f'{{"action": "COMMENT or REPLY or POST or UPVOTE", '
            f'"reasoning": "brief reason", '
            f'"draft_content": "concise suggested text"}}'
        )

        try:
            chat_kwargs: Dict[str, Any] = dict(
                model_id=model_id,
                system_instruction=self._kernel,
                temperature=temp,
                max_output_tokens=max_tokens,
            )
            chat = self._registry.create_chat(**chat_kwargs)
            text = chat.send_message(prompt)
            # Estimate tokens for budget tracking
            est_in = (len(self._kernel) + len(prompt)) // 4
            est_out = len(text) // 4
            self._budget.record_usage(model_id, _make_response(text, est_in, est_out, model_id))

            plan = _parse_json_safe(text)
            if not plan:
                # Enhanced diagnostics for parse failures
                looks_truncated = not text.rstrip().endswith("}") and "{" in text
                self._telemetry.log("strategist_parse_fail", {
                    "brain": self._brain_name,
                    "tick": self._tick_count,
                    "item_id": item_id,
                    "raw_text": shorten(text, 500),
                    "full_text_length": len(text),
                    "looks_truncated": looks_truncated,
                    "max_tokens": max_tokens,
                    "model": model_id,
                })
                return None

            # Charge is proportional to signal score, boosted by urgency and source weight
            source = item.get("_source", "feed")
            if source == "seed":
                source_weight = float(self._ctrl.get("charge_weight_seed"))
            else:
                source_weight = float(self._ctrl.get("charge_weight_feed"))
            charge = score * urgency * source_weight

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
                source=item.get("_source", "feed"),
                model=model_id,
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

    # ------------------------------------------------------------------
    # Gear 4: Seeker — search for information using Google Search
    # ------------------------------------------------------------------

    def _seek(self) -> None:
        """Search for current information based on conscious directives' focus_topics.

        Uses Google Search tools (incompatible with json_mode).
        Results bypass sentry scoring and go directly to draft generation.
        """
        with self._directives_lock:
            topics = list(self._directives.get("focus_topics", []))

        if not topics:
            return

        model_id = self._ctrl.get("seeker_model") or self._ctrl.get("subconscious_model")
        temp = self._ctrl.get("subconscious_temperature")
        max_tokens = self._ctrl.get("seeker_max_tokens")
        max_topics = self._ctrl.get("seeker_max_topics")

        # Cap topics to control cost
        topics = topics[:max_topics]

        results_found = 0
        drafts_created = 0

        for i, topic in enumerate(topics):
            # Budget check per topic
            if not self._budget.can_afford(model_id, est_input_tokens=1000,
                                           est_output_tokens=max_tokens):
                self._telemetry.log("seeker_budget_skip", {
                    "brain": self._brain_name,
                    "tick": self._tick_count,
                    "model": model_id,
                    "topics_remaining": len(topics) - i,
                })
                break

            result = self._seek_topic(topic, model_id, temp, max_tokens)
            if result:
                results_found += 1
                draft = self._strategize_search_result(result, topic)
                if draft:
                    self._buffer.add_draft(draft)
                    drafts_created += 1
                    self._telemetry.log("strategist_draft", {
                        "brain": self._brain_name,
                        "tick": self._tick_count,
                        "item_id": draft.item_id,
                        "action": draft.suggested_action,
                        "charge": round(draft.charge, 3),
                        "draft_length": len(draft.draft_content),
                        "source": "search",
                    })

        self._telemetry.log("seeker_sweep", {
            "brain": self._brain_name,
            "tick": self._tick_count,
            "topics_searched": len(topics),
            "results_found": results_found,
            "drafts_created": drafts_created,
            "model": model_id,
        })

    def _seek_topic(self, topic: str, model_id: str,
                    temp: float, max_tokens: int) -> Optional[dict]:
        """Search for a single topic using Google Search grounding.

        Returns a dict with keys: topic, summary, search_queries, sources
        or None on failure.

        IMPORTANT: Cannot use json_mode because Gemini rejects tools + json_mode.
        """
        directives_text = self._get_directives_text()
        directive_section = (f"\nConscious directives:\n{directives_text}"
                             if directives_text else "")

        prompt = (
            f"Search for current, relevant information about: {topic}\n\n"
            f"Context — your directive: {self._directive}\n"
            f"{directive_section}\n\n"
            f"Use Google Search to find the latest information about this topic.\n"
            f"Summarize what you find in 2-4 paragraphs, focusing on:\n"
            f"- What is happening right now related to this topic\n"
            f"- Key facts, developments, or perspectives\n"
            f"- How this connects to your directive\n\n"
            f"Format your response as:\n"
            f"SUMMARY: <your summary>\n"
            f"RELEVANCE: <brief note on how this connects to the directive>\n"
            f"SUGGESTED_ACTION: <POST or COMMENT — what action to take with this>"
        )

        try:
            chat = self._registry.create_chat(
                model_id=model_id,
                system_instruction=self._kernel,
                temperature=temp,
                max_output_tokens=max_tokens,
                tools=self._search_tools,
            )
            # NO json_mode — incompatible with tools
            text = chat.send_message(prompt)

            # Estimate tokens for budget tracking
            est_in = (len(self._kernel) + len(prompt)) // 4
            est_out = len(text) // 4
            self._budget.record_usage(
                model_id, _make_response(text, est_in, est_out, model_id))

            # Extract grounding metadata (search queries, sources)
            search_queries: list = []
            sources: list = []
            grounding = getattr(chat, "_last_grounding_metadata", None)
            if grounding:
                search_queries = list(
                    getattr(grounding, "web_search_queries", None) or [])
                chunks = getattr(grounding, "grounding_chunks", None) or []
                for chunk in chunks[:10]:
                    web = getattr(chunk, "web", None)
                    if web:
                        sources.append({
                            "uri": getattr(web, "uri", ""),
                            "title": getattr(web, "title", ""),
                        })

            self._telemetry.log("seeker_result", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "topic": topic,
                "summary_length": len(text),
                "has_search_queries": bool(search_queries),
                "search_query_count": len(search_queries),
            })

            return {
                "topic": topic,
                "summary": text,
                "search_queries": search_queries,
                "sources": sources,
            }

        except Exception as e:
            self._telemetry.log("seeker_error", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "topic": topic,
                "error": str(e)[:500],
                "error_type": type(e).__name__,
            })
            return None

    def _strategize_search_result(self, result: dict,
                                  topic: str) -> Optional[Draft]:
        """Convert a seeker search result into a Draft for the conscious buffer.

        No additional LLM call — parses the seeker response directly.
        """
        text = result.get("summary", "")
        if not text or len(text) < 20:
            return None

        # Parse suggested action from response (default POST)
        suggested_action = "POST"
        for line in text.split("\n"):
            upper_line = line.strip().upper()
            if upper_line.startswith("SUGGESTED_ACTION:"):
                action_text = line.split(":", 1)[1].strip().upper()
                if action_text in ("POST", "COMMENT", "REPLY"):
                    suggested_action = action_text
                break

        # Extract the SUMMARY section as draft content
        draft_content = text
        if "SUMMARY:" in text:
            parts = text.split("SUMMARY:", 1)
            if len(parts) > 1:
                remainder = parts[1]
                for marker in ("RELEVANCE:", "SUGGESTED_ACTION:"):
                    if marker in remainder:
                        remainder = remainder.split(marker, 1)[0]
                draft_content = remainder.strip()

        # Build source citation
        sources = result.get("sources", [])
        source_text = ""
        if sources:
            source_text = " | Sources: " + ", ".join(
                s.get("title", "?") for s in sources[:3]
            )

        # Stable item_id
        item_id = f"search:{hash(topic + str(time.time())) & 0xFFFFFFFF:08x}"

        # Compute charge
        with self._directives_lock:
            urgency = float(self._directives.get("urgency_boost", 1.0))
        charge_weight = float(self._ctrl.get("charge_weight_search"))
        signal_score = 0.8  # intentionally sought content
        charge = signal_score * urgency * charge_weight

        target_summary = (shorten(f"[Search: {topic}] {draft_content}", 120)
                          + source_text)

        return Draft(
            timestamp=time.time(),
            item_id=item_id,
            signal_score=signal_score,
            suggested_action=suggested_action,
            target_summary=target_summary,
            reasoning=f"Seeker found information about '{topic}' via Google Search",
            draft_content=shorten(draft_content, 800),
            charge=charge,
            source="search",
            model=model_id,
        )


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
    """Parse a score (0.0-1.0) from LLM response text.

    Extraction priority:
    1. Valid JSON with "score" key
    2. Bare decimal number (e.g. "0.7")
    3. "score" keyword near a decimal number in prose
    Safe: rejects stray integers like "Layer 1" that inflated scores before.
    """
    import re
    text = text.strip()
    # Try JSON parse first
    data = _parse_json_safe(text)
    if isinstance(data, dict) and "score" in data:
        try:
            s = float(data["score"])
            return max(0.0, min(1.0, s))
        except (ValueError, TypeError):
            pass
    # Fallback 1: bare number (e.g. "0.7" or "0.85")
    bare = text.strip().strip('"\'')
    if re.fullmatch(r"\d+\.\d+", bare):
        try:
            return max(0.0, min(1.0, float(bare)))
        except ValueError:
            pass
    # Fallback 2: "score" keyword followed by a decimal number in prose
    # Matches patterns like: score: 0.7, score=0.8, score is 0.6
    m = re.search(r'["\']?score["\']?\s*[:=\s]\s*(\d+\.\d+)', text, re.IGNORECASE)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    return 0.0


def _parse_json_safe(text: str) -> Optional[dict]:
    """Try to parse JSON from text, handling markdown fences, junk, and truncation."""
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
    # Repair truncated JSON — close open strings and braces
    if start >= 0:
        fragment = text[start:]
        # Close any open string (odd number of unescaped quotes)
        in_str = False
        i = 0
        while i < len(fragment):
            ch = fragment[i]
            if ch == '\\' and in_str:
                i += 2
                continue
            if ch == '"':
                in_str = not in_str
            i += 1
        if in_str:
            fragment += '"'
        # Close open braces
        open_braces = fragment.count('{') - fragment.count('}')
        fragment += '}' * max(0, open_braces)
        try:
            return json.loads(fragment)
        except (json.JSONDecodeError, ValueError):
            pass
    return None
