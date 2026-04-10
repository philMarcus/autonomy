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
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set

from colorama import Fore, Style

from .actions import execute_daemon_action
from .buffer import Draft, DraftBuffer
from .controls import ControlRegistry
from .cooldowns import can_do
from .llm import DailyBudget, ModelRegistry
from .llm.base import LLMResponse
from .scoring import (
    build_simple_batch_prompt,
    parse_simple_batch_response,
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

        # Track model usage per wake period (reset by __main__.py after each conscious cycle)
        self._tick_model_counts: Dict[str, int] = {}
        # Per-model charge history for auto-calibration (last 20 ticks per model)
        # Load from state if available (persists across restarts)
        with self._state_lock:
            self._model_charge_history: Dict[str, List[float]] = dict(
                self._state.get("_sentry_charge_history", {})
            )
        # Track reply candidates already scored (avoid re-scoring)
        self._scored_comment_ids: Set[str] = set()
        # Load persisted scored comment IDs from state
        with self._state_lock:
            _persisted = self._state.get("_scored_comment_ids", [])
            if _persisted:
                self._scored_comment_ids.update(_persisted)

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
        # Clear old daemon ticks from Analog Home on each restart
        if self._store and hasattr(self._store, '_analog_home_url') and self._store._analog_home_url:
            try:
                import requests
                from urllib.parse import urljoin
                url = urljoin(self._store._analog_home_url.rstrip("/") + "/", f"daemon-ticks?run_id={self._store._run_id}")
                requests.delete(url, timeout=3)
            except Exception:
                pass
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="subconscious-daemon", daemon=True
        )
        self._thread.start()
        self._telemetry.log("daemon_start", {
            "brain": self._brain_name,
            "sentry_weights": self._ctrl.get("subconscious_model_weights"),
            "strategist_weights": self._ctrl.get("strategist_model_weights"),
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

    def _emit(self, line: str, color=None) -> None:
        """Print a line to terminal AND accumulate for Analog Home push."""
        if color:
            print(f"{color}{line}{Style.RESET_ALL}")
        else:
            print(line)
        # Strip ANSI for Analog Home (plain text with role prefixes for CSS coloring)
        import re as _re
        clean = _re.sub(r'\x1b\[[0-9;]*m', '', f"{color}{line}{Style.RESET_ALL}" if color else line)
        self._tick_lines.append(clean)

    def _flush_tick_lines(self, complete: bool = False) -> None:
        """Push accumulated lines to Analog Home and clear."""
        if self._tick_lines and self._store:
            interval = int(self._ctrl.get("sentry_interval_seconds") or 300)
            self._store.push_daemon_tick(
                self._tick_count, list(self._tick_lines), interval, complete)
            self._tick_lines.clear()

    def _tick(self) -> None:
        """One tick of the daemon: seeker → sentry → strategist (single call)."""
        self._tick_count += 1
        self._tick_lines: list = []
        self._tick_model_counts.clear()

        # Read controls
        signal_threshold = self._ctrl.get("signal_threshold")

        # Auto-calibrate wake threshold from target_wake_minutes + observed charge
        wake_threshold = self._compute_wake_threshold()
        self._buffer.update_threshold(wake_threshold)

        # No decay — wake_potential only changes via charge/refractory
        self._buffer.decay(1.0)

        # ── Tick header ──
        self._emit(f"── TICK {self._tick_count} {'─' * 38}", Fore.WHITE)
        self._flush_tick_lines()  # push header immediately

        # --- Gear 4: Seeker goes FIRST (every N ticks) ---
        seeker_n = int(self._ctrl.get("seeker_every_n_ticks") or 3)
        if (self._search_tools
                and self._tick_count % seeker_n == 0):
            self._seek()

        # --- Gear 1: Sentry scan + scoring ---
        _pre_tick_potential = self._buffer.wake_potential
        items_scanned = 0
        new_items = 0
        signals_above = 0
        seeds_scanned = 0
        high_signal_items = []  # collected for single strategist call

        if self._platform is not None:
            items = self._sentry_scan()
            items_scanned = len(items)
            new_items = len(items)

            # Batch score all items in a single LLM call
            scores = self._score_items_batch(items) if items else []

            _signal_lines = []
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
                    signals_above += 1
                    high_signal_items.append((item, score))
                    _author = _get_author(item)
                    _title = (item.get("title") or item.get("content", ""))[:60]
                    _signal_lines.append((_author, _title, score))

            # Print sentry summary FIRST, then signal details below
            if scores:
                _sentry_model = max(self._tick_model_counts, key=self._tick_model_counts.get) if self._tick_model_counts else "?"
                _score_digits = [str(min(9, int(s * 9))) for s in scores]
                _mode_tag = f" [{getattr(self, '_current_feed_mode', 'new')}]"
                self._emit(f"  SENTRY ({_sentry_model}) {items_scanned} items{_mode_tag}", Fore.CYAN)
                self._emit(f"    Scores: [{','.join(_score_digits)}] → {signals_above} signals", Fore.CYAN)
                for _author, _title, _sc in _signal_lines:
                    self._emit(f"    ↑ @{_author}: \"{_title}\" ({_sc:.2f})", Fore.GREEN)
            self._flush_tick_lines()  # push sentry results
        else:
            self._emit("  (no feed)", Fore.WHITE)

        # Seed scan — scored by sentry but with lower threshold and higher charge
        # Good-faith seeds wake the agent; garbage ("Hello!") does not
        seed_items = self._seed_scan()
        seeds_scanned = len(seed_items)
        seed_threshold = float(self._ctrl.get("seed_threshold") or 0.3)
        for seed_item in seed_items:
            seed_id = seed_item.get("id", "")
            seed_item["_source"] = "seed"
            score = self._score_item(seed_item)

            self._telemetry.log("sentry_signal", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "item_id": seed_id,
                "score": round(score, 3),
                "above_threshold": score >= seed_threshold,
                "source": "seed",
            })

            if score >= seed_threshold:
                signals_above += 1
                high_signal_items.append((seed_item, score))
                # Seed charge: threshold * score for normal seeds, 999 for operator seeds (-P suffix)
                _seed_text_raw = seed_item.get('title', seed_item.get('content', ''))
                if _seed_text_raw.rstrip().endswith("-P"):
                    seed_charge = 999.0  # operator seed — instant wake
                else:
                    seed_charge = self._buffer._wake_threshold * score  # proportional to threshold + quality
                with self._buffer._lock:
                    self._buffer._wake_potential += seed_charge
                    if self._buffer._wake_potential >= self._buffer._wake_threshold:
                        self._buffer._wake_event.set()
                _seed_text = (_seed_text_raw or "?")[:50]
                self._emit(f"  SEED: \"{_seed_text}\" → score={score:.1f} → charge +{seed_charge:.1f}", Fore.GREEN)
            else:
                _seed_text = seed_item.get('title', seed_item.get('content', '?'))[:50]
                self._emit(f"  SEED: \"{_seed_text}\" → score={score:.1f} (below {seed_threshold})", Fore.YELLOW)

        # Add base charge for each signal (even if strategist produces no drafts)
        _feed_charge = float(self._ctrl.get("charge_weight_feed") or 0.05)
        for _ in high_signal_items:
            with self._buffer._lock:
                self._buffer._wake_potential += _feed_charge

        # --- Gear 2: Strategist — ONE call with all high-signal items ---
        if high_signal_items:
            seeker_summary = self._buffer.get_seeker_summary()
            drafts = self._strategize_batch(high_signal_items, seeker_summary)
            _strat_model = getattr(self, '_last_strategist_model', None) or (drafts[0].model if drafts else "?")
            self._emit(f"  STRATEGIST ({_strat_model}) {len(high_signal_items)} items", Fore.CYAN)
            if drafts:
                for d in drafts:
                    self._emit(f"    → DRAFT {d.suggested_action}: {d.target_summary[:60]}", Fore.GREEN)
            else:
                self._emit("    (no drafts)", Fore.YELLOW)
            self._flush_tick_lines()  # push strategist results
            for draft in drafts:
                self._buffer.add_draft(draft)
                self._telemetry.log("strategist_draft", {
                    "brain": self._brain_name,
                    "tick": self._tick_count,
                    "item_id": draft.item_id,
                    "action": draft.suggested_action,
                    "charge": round(draft.charge, 3),
                    "draft_length": len(draft.draft_content),
                    "model": draft.model,
                    "source": draft.source,
                })

        # Record charge produced this tick for auto-calibration
        # Exclude seed charge (999) — seeds should cause instant wake, not inflate threshold
        _tick_charge = max(0, self._buffer.wake_potential - _pre_tick_potential)
        _seed_charge_total = seeds_scanned * float(self._ctrl.get("charge_weight_seed") or 999.0) if seeds_scanned else 0
        _feed_only_charge = max(0, _tick_charge - _seed_charge_total)
        if items_scanned > 0:
            _last_model = max(self._tick_model_counts, key=self._tick_model_counts.get) if self._tick_model_counts else ""
            if _last_model:
                self._record_tick_charge(_last_model, _feed_only_charge)

        # Reply candidate scan (every N ticks)
        _reply_interval = int(self._ctrl.get("reply_scan_interval_ticks") or 2)
        if self._platform and self._tick_count % _reply_interval == 0:
            self._scan_reply_candidates()

        # --- Gear 6: Dreamer (stochastic) ---
        dream_interval = int(self._ctrl.get("dream_interval_ticks") or 60)
        if random.random() < 1.0 / max(1, dream_interval):
            self._dream()

        # --- Gear 7: Muse (stochastic creative draft) ---
        muse_interval = int(self._ctrl.get("muse_interval_ticks") or 30)
        if random.random() < 1.0 / max(1, muse_interval):
            self._muse()

        # ── Tick footer ──
        _wp = self._buffer.wake_potential
        _dc = self._buffer.draft_count
        _thresh = self._buffer._wake_threshold
        self._emit(f"  wake={_wp:.2f} | drafts={_dc} | threshold={_thresh:.1f}", Fore.WHITE)
        self._emit(f"{'─' * 46}", Fore.WHITE)
        self._flush_tick_lines(complete=True)  # final push for this tick

        self._telemetry.log("daemon_tick", {
            "brain": self._brain_name,
            "tick": self._tick_count,
            "items_scanned": items_scanned,
            "new_items": new_items,
            "seeds_scanned": seeds_scanned,
            "signals_above_threshold": signals_above,
            "wake_potential": round(self._buffer.wake_potential, 3),
            "draft_count": self._buffer.draft_count,
        })

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
        """Fetch feed and return only unseen items. Rotates feed source per tick."""
        try:
            batch_size = self._ctrl.get("feed_batch_size")
            # Rotate feed source based on tick count
            rotation_str = self._ctrl.get("feed_rotation") or "new,new,new,new,following,new,hot"
            rotation = [m.strip() for m in rotation_str.split(",") if m.strip()]
            feed_mode = rotation[self._tick_count % len(rotation)] if rotation else "new"
            self._current_feed_mode = feed_mode  # for terminal display
            if feed_mode == "following":
                feed = self._platform.get_feed(limit=batch_size, sort="new", filter="following")
            elif feed_mode in ("hot", "top", "rising"):
                feed = self._platform.get_feed(limit=batch_size, sort=feed_mode)
            else:
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
            author_name = author.get("name", "") if isinstance(author, dict) else str(author)
            if author_name == self._username:
                continue
            # Skip ignored authors (from conscious directives)
            with self._directives_lock:
                _ignored = {a.lower() for a in self._directives.get("ignore_authors", [])}
            if author_name.lower() in _ignored:
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
        model_id = self._pick_sentry_model()
        temp = self._ctrl.get("subconscious_temperature")

        # Budget check
        if not self._budget.can_afford(model_id, est_input_tokens=800, est_output_tokens=50):
            return 0.0

        # Build item summary — use same simple format as batch scoring
        author_name = _get_author(item)
        title = item.get("title", "") or ""
        content = item.get("content", "") or ""
        item_text = f"@{author_name}: \"{title}\" — {shorten(content, 300)}"

        directives_text = self._get_directives_text()
        prompt = build_simple_batch_prompt([item_text], self._directive, directives_text)

        try:
            chat = self._registry.create_chat(
                model_id=model_id,
                system_instruction="You are a feed-scanning daemon. Score items concisely.",
                temperature=temp,
                max_output_tokens=50,
                disable_thinking=True,
            )
            text = chat.send_message(prompt)
            est_in = (len(prompt)) // 4
            est_out = len(text) // 4
            self._budget.record_usage(model_id, _make_response(text, est_in, est_out, model_id))

            rubrics = parse_simple_batch_response(text, 1)
            rubric = rubrics[0] if rubrics else {"relevance": 0}
            score = rubric.get("relevance", 0) / 9.0

            self._telemetry.log("sentry_rubric", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "item_id": item.get("id", ""),
                "relevance": rubric.get("relevance", 0),
                "score": score,
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

    def _compute_wake_threshold(self) -> float:
        """Auto-calibrate wake threshold from target_wake_minutes and observed charge rates."""
        target_minutes = float(self._ctrl.get("target_wake_minutes") or 60)
        sentry_interval = float(self._ctrl.get("sentry_interval_seconds") or 300)
        target_ticks = (target_minutes * 60) / max(1, sentry_interval)

        # Weighted average charge per tick across models
        weights_str = self._ctrl.get("subconscious_model_weights") or ""
        model_weights: Dict[str, float] = {}
        for pair in weights_str.split(","):
            if "=" in pair:
                m, w = pair.rsplit("=", 1)
                try:
                    model_weights[m.strip()] = float(w.strip())
                except ValueError:
                    pass

        total_weight = sum(model_weights.values()) or 1.0
        expected_charge = 0.0
        for model, weight in model_weights.items():
            history = self._model_charge_history.get(model, [])
            if history:
                avg = sum(history) / len(history)
            else:
                avg = 0.2  # default estimate before data exists
            expected_charge += (weight / total_weight) * avg

        threshold = target_ticks * expected_charge
        return max(1.0, threshold)

    def _record_tick_charge(self, model_id: str, charge: float) -> None:
        """Record charge produced by this tick's model for calibration. Persists to state."""
        history = self._model_charge_history.setdefault(model_id, [])
        history.append(charge)
        # Keep last 20 per model
        if len(history) > 20:
            self._model_charge_history[model_id] = history[-20:]
        # Persist to state for restart survival
        with self._state_lock:
            self._state["_sentry_charge_history"] = dict(self._model_charge_history)

    def _pick_sentry_model(self, exclude: str = "") -> str:
        """Pick a model from the sentry pool, optionally excluding one (for 503 retry)."""
        weights_str = self._ctrl.get("subconscious_model_weights") or ""
        if exclude and weights_str:
            # Remove the excluded model from the weights string
            pairs = [p.strip() for p in weights_str.split(",") if p.strip()]
            filtered = [p for p in pairs if not p.startswith(f"{exclude}=")]
            weights_str = ",".join(filtered) if filtered else weights_str
        # Fallback: first model from weights string
        fallback = (weights_str.split("=")[0].strip() if "=" in weights_str
                    else "gemini-2.5-flash-lite")
        model = _pick_weighted_model(weights_str, fallback)
        self._tick_model_counts[model] = self._tick_model_counts.get(model, 0) + 1
        return model

    def _pick_strategist_model(self, exclude: str = "") -> str:
        """Pick a model from the strategist pool, optionally excluding one."""
        weights_str = self._ctrl.get("strategist_model_weights") or ""
        if exclude and weights_str:
            pairs = [p.strip() for p in weights_str.split(",") if p.strip()]
            filtered = [p for p in pairs if not p.startswith(f"{exclude}=")]
            weights_str = ",".join(filtered) if filtered else weights_str
        fallback = (weights_str.split("=")[0].strip() if "=" in weights_str
                    else "gemini-2.5-flash-lite")
        model = _pick_weighted_model(weights_str, fallback)
        return model

    def _score_items_batch(self, items: list) -> list:
        """Score multiple feed items in a single LLM call (batch mode).

        Falls back to per-item scoring if batch parsing fails.
        Returns list of float scores (same length as items).
        """
        if not items:
            return []

        model_id = self._pick_sentry_model()
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
            item_texts.append(f"@{author_name}: \"{title}\" — {shorten(content, 300)}")

        directives_text = self._get_directives_text()
        # Use simple 0-9 format (works across all models including flash-lite/gpt)
        prompt = build_simple_batch_prompt(item_texts, self._directive, directives_text)

        try:
            # Use short task-specific instruction for sentry (NOT the kernel).
            # The kernel causes models like flash-lite to role-play and generate
            # monologue instead of scores.
            sentry_instruction = "You are a feed-scanning daemon. Score items concisely. Output only numbers."
            chat = self._registry.create_chat(
                model_id=model_id,
                system_instruction=sentry_instruction,
                temperature=temp,
                max_output_tokens=max(64, 10 * len(items)),
                disable_thinking=True,
            )
            text = chat.send_message(prompt)

            # Budget tracking
            est_in = (len(self._kernel) + len(prompt)) // 4
            est_out = len(text) // 4
            self._budget.record_usage(model_id, _make_response(text, est_in, est_out, model_id))

            # Parse simple response (0-9 per item)
            rubrics = parse_simple_batch_response(text, len(items))

            scores = []
            for i, (item, rubric) in enumerate(zip(items, rubrics)):
                # Simple score: average of the three identical criteria / 3, normalized to 0-1
                score = rubric.get("relevance", 0) / 9.0
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
            err_str = str(e)
            is_503 = "503" in err_str or "UNAVAILABLE" in err_str
            self._telemetry.log("sentry_batch_error", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "items": len(items),
                "error": err_str[:300],
                "error_type": type(e).__name__,
                "model": model_id,
            })
            # On 503: retry with a different model from the pool
            if is_503:
                retry_model = self._pick_sentry_model(exclude=model_id)
                if retry_model != model_id:
                    try:
                        chat = self._registry.create_chat(
                            model_id=retry_model,
                            system_instruction=sentry_instruction,
                            temperature=temp,
                            max_output_tokens=max(64, 10 * len(items)),
                            disable_thinking=True,
                        )
                        text = chat.send_message(prompt)
                        est_in = (len(prompt)) // 4
                        est_out = len(text) // 4
                        self._budget.record_usage(retry_model, _make_response(text, est_in, est_out, retry_model))
                        rubrics = parse_simple_batch_response(text, len(items))
                        scores = []
                        for i, (item, rubric) in enumerate(zip(items, rubrics)):
                            score = rubric.get("relevance", 0) / 9.0
                            scores.append(score)
                            self._telemetry.log("sentry_rubric", {
                                "brain": self._brain_name, "tick": self._tick_count,
                                "item_id": item.get("id", ""), "score": score,
                                "relevance": rubric.get("relevance", 0),
                                "novelty": rubric.get("novelty", 0),
                                "actionability": rubric.get("actionability", 0),
                                "model": retry_model, "batch": True, "retry_503": True,
                            })
                        return scores
                    except Exception:
                        pass  # retry also failed, fall through to per-item
            # Final fallback: score items individually
            return [self._score_item(item) for item in items]

    # ------------------------------------------------------------------
    # Gear 2: Strategist — draft an action plan for high-signal items
    # ------------------------------------------------------------------

    def _strategize_batch(self, items_with_scores: list,
                          seeker_summary: str = "") -> List[Draft]:
        """Call strategist ONCE with all high-signal items + seeker context.

        Returns a list of Draft objects (may be empty).
        """
        from .buffer import Draft

        model_id = self._pick_strategist_model()
        self._last_strategist_model = model_id
        temp = self._ctrl.get("subconscious_temperature")
        max_tokens = self._ctrl.get("strategist_max_tokens")

        # Budget check
        est_input = 1500 + len(seeker_summary) // 4
        if not self._budget.can_afford(model_id, est_input_tokens=est_input,
                                       est_output_tokens=max_tokens):
            self._telemetry.log("strategist_budget_skip", {
                "brain": self._brain_name, "tick": self._tick_count,
                "model": model_id, "items": len(items_with_scores),
            })
            return []

        # Build items section
        items_lines = []
        for idx, (item, score) in enumerate(items_with_scores, 1):
            author = _get_author(item)
            title = item.get("title", "") or ""
            content = item.get("content", "") or ""
            source = item.get("_source", "feed")
            source_tag = " [SEED]" if source == "seed" else ""
            items_lines.append(
                f"{idx}. [score {score:.2f}]{source_tag} @{author}: "
                f"{shorten(title, 80)} — {shorten(content, 300)}"
            )
        items_text = "\n".join(items_lines)

        directives_text = self._get_directives_text()
        directive_section = (f"\nConscious directives:\n{directives_text}"
                             if directives_text else "")

        seeker_section = ""
        if seeker_summary:
            seeker_section = (
                f"\n\nRESEARCH CONTEXT (from seeker):\n"
                f"{shorten(seeker_summary, 2000)}\n"
            )

        with self._directives_lock:
            urgency = float(self._directives.get("urgency_boost", 1.0))

        prompt = (
            f"STRATEGIST TASK — output JSON array only. Do not write monologue or prose.\n\n"
            f"Directive: {self._directive}\n"
            f"{directive_section}{seeker_section}\n"
            f"HIGH-SIGNAL ITEMS ({len(items_with_scores)}):\n{items_text}\n\n"
            f"Generate drafts. Two equally valid modes:\n"
            f"  PER-ITEM: respond directly to one item (COMMENT/REPLY)\n"
            f"  SYNTHESIS: connect multiple items into new insight (POST/POST_MOLTBOOK)\n\n"
            f"Action types: POST, POST_MOLTBOOK, COMMENT, REPLY, GENERATE_IMAGE\n"
            f"For GENERATE_IMAGE, draft_content is the image prompt.\n\n"
            f"OUTPUT FORMAT — JSON array, NOTHING ELSE. No prose. No [INTERNAL MONOLOGUE]. No markdown fences. Begin response with [.\n"
            f'[{{"action":"COMMENT","item_index":1,"reasoning":"≤50 words","draft_content":"≤150 words"}},{{"action":"POST","item_index":0,"reasoning":"synthesis","draft_content":"≤150 words"}}]\n\n'
            f"item_index: 1-based index of inspiring item (0 = synthesis). "
            f"Keep draft_content concise (≤150 words) to avoid truncation. "
            f"In reasoning AND draft_content, refer to items by AUTHOR or TOPIC, never by number — "
            f"the consciousness reads drafts in isolation and won't know what 'item 3' means. "
            f"Empty array [] if nothing warrants action."
        )

        # Frame the kernel as a description of the entity being drafted FOR,
        # not as the strategist's own identity. Prevents role-play / monologue.
        strategist_system = (
            "You are a strategist tool. You draft action plans on behalf of the following entity, "
            "matching its voice and concerns, but you yourself are NOT that entity. "
            "You output structured JSON only, never internal monologue or prose.\n\n"
            "=== ENTITY YOU DRAFT FOR ===\n"
            f"{self._kernel}\n"
            "=== END ENTITY ==="
        )

        try:
            chat = self._registry.create_chat(
                model_id=model_id,
                system_instruction=strategist_system,
                temperature=temp,
                max_output_tokens=max_tokens,
            )
            text = chat.send_message(prompt)
            est_in = (len(self._kernel) + len(prompt)) // 4
            est_out = len(text) // 4
            self._budget.record_usage(model_id,
                                      _make_response(text, est_in, est_out, model_id))

            # Parse JSON array of drafts
            plans = _parse_json_safe(text)
            if plans is None:
                # Try wrapping in array if it's a single object
                plans = _parse_json_safe(f"[{text}]") if "{" in text else None
            if isinstance(plans, dict):
                plans = [plans]
            if not isinstance(plans, list):
                # Detect truncation: response near max_tokens, doesn't end with ] or }
                looks_truncated = (
                    len(text) > 0
                    and not text.rstrip().endswith(("]", "}"))
                    and (est_out := len(text) // 4) > max_tokens * 0.85
                )
                self._telemetry.log("strategist_parse_fail", {
                    "brain": self._brain_name, "tick": self._tick_count,
                    "raw_text_head": text[:300], "raw_text_tail": text[-300:] if len(text) > 600 else "",
                    "full_text_length": len(text),
                    "looks_truncated": looks_truncated,
                    "max_tokens": max_tokens,
                    "model": model_id,
                })
                return []

            drafts = []
            for plan in plans:
                if not isinstance(plan, dict):
                    continue
                action = (plan.get("action") or "POST").upper()
                item_idx = int(plan.get("item_index", 0))

                # Determine source item and charge
                if 1 <= item_idx <= len(items_with_scores):
                    src_item, src_score = items_with_scores[item_idx - 1]
                    source = src_item.get("_source", "feed")
                    item_id = src_item.get("id", "")
                    author = _get_author(src_item)
                    title = src_item.get("title", "") or src_item.get("content", "")
                    target_summary = f"@{author}: {shorten(title, 80)}"
                else:
                    # Synthesized or research-inspired
                    src_score = 0.8
                    source = "search" if seeker_summary else "feed"
                    item_id = f"synth:{self._tick_count}"
                    target_summary = shorten(plan.get("reasoning", "synthesized"), 80)

                source_weight = float(self._ctrl.get(
                    "charge_weight_seed" if source == "seed" else
                    "charge_weight_search" if source == "search" else
                    "charge_weight_feed"))
                charge = src_score * urgency * source_weight

                drafts.append(Draft(
                    timestamp=time.time(),
                    item_id=item_id,
                    signal_score=src_score,
                    suggested_action=action,
                    target_summary=target_summary,
                    reasoning=plan.get("reasoning", ""),
                    draft_content=plan.get("draft_content", ""),
                    charge=charge,
                    source=source,
                    model=model_id,
                ))

            return drafts

        except Exception as e:
            err_str = str(e)
            is_503 = "503" in err_str or "UNAVAILABLE" in err_str
            self._telemetry.log("strategist_error", {
                "brain": self._brain_name, "tick": self._tick_count,
                "error": err_str[:300], "model": model_id,
                "items": len(items_with_scores),
            })
            if is_503:
                retry_model = self._pick_strategist_model(exclude=model_id)
                if retry_model != model_id:
                    try:
                        chat = self._registry.create_chat(
                            model_id=retry_model, system_instruction=strategist_system,
                            temperature=temp, max_output_tokens=max_tokens)
                        text = chat.send_message(prompt)
                        self._budget.record_usage(retry_model,
                            _make_response(text, est_input, len(text) // 4, retry_model))
                        plans = _parse_json_safe(text)
                        if isinstance(plans, dict):
                            plans = [plans]
                        if isinstance(plans, list):
                            # Simplified retry parse — just take first valid draft
                            for plan in plans:
                                if isinstance(plan, dict) and plan.get("draft_content"):
                                    return [Draft(
                                        timestamp=time.time(),
                                        item_id=items_with_scores[0][0].get("id", ""),
                                        signal_score=items_with_scores[0][1],
                                        suggested_action=(plan.get("action") or "POST").upper(),
                                        target_summary="retry draft",
                                        reasoning=plan.get("reasoning", ""),
                                        draft_content=plan.get("draft_content", ""),
                                        charge=0.1, source="feed", model=retry_model,
                                    )]
                    except Exception:
                        pass
            return []

    # ------------------------------------------------------------------
    # Gear 5: Reply Scanner — score comments on our posts
    # ------------------------------------------------------------------

    def _scan_reply_candidates(self) -> None:
        """Check recent own posts for worthy reply candidates. Score them with sentry."""
        if not self._platform:
            return

        max_replies = int(self._ctrl.get("max_replies_per_post") or 3)
        charge_weight = float(self._ctrl.get("charge_weight_reply") or 1.5)

        # Get recent post IDs from state
        with self._state_lock:
            my_post_ids = list(self._state.get("my_post_ids", []))[-4:]  # last 4 posts
            replied_keys = set(self._state.get("replied_comment_keys", []))

        for post_id in my_post_ids:
            try:
                comments = self._platform.get_post_comments(post_id, sort="new") or []
            except Exception:
                continue

            # Count existing replies to this post
            reply_count = sum(1 for k in replied_keys if k.startswith(f"{post_id}:"))
            if reply_count >= max_replies:
                continue

            # Filter comments: skip own, already replied, already scored, stale
            candidates = []
            for c in comments[:20]:
                cid = c.get("id", "")
                if not cid:
                    continue
                key = f"{post_id}:{cid}"
                if key in replied_keys:
                    continue
                if cid in self._scored_comment_ids:
                    continue
                author = c.get("author", {})
                author_name = author.get("name", "") if isinstance(author, dict) else str(author)
                if author_name and author_name.lower() == self._username.lower():
                    continue
                content = c.get("content", "") or ""
                if len(content) < 15:
                    continue  # skip very short comments
                candidates.append(c)
                self._scored_comment_ids.add(cid)

            if not candidates:
                continue

            # Batch sentry-score the comments
            item_texts = []
            for c in candidates:
                author = c.get("author", {})
                author_name = author.get("name", "") if isinstance(author, dict) else str(author)
                content = c.get("content", "") or ""
                item_texts.append(f"@{author_name}: {shorten(content, 300)}")

            model_id = self._pick_sentry_model()
            try:
                from .scoring import build_simple_batch_prompt, parse_simple_batch_response
                prompt = build_simple_batch_prompt(
                    item_texts, self._directive,
                    self._get_directives_text(),
                )
                sentry_instruction = "You are scoring comments on your own posts. Rate how worthy each is of a thoughtful reply. Output only numbers."
                chat = self._registry.create_chat(
                    model_id=model_id,
                    system_instruction=sentry_instruction,
                    temperature=0.3,
                    max_output_tokens=max(64, 10 * len(candidates)),
                )
                text = chat.send_message(prompt)
                rubrics = parse_simple_batch_response(text, len(candidates))

                for c, rubric in zip(candidates, rubrics):
                    score = rubric.get("relevance", 0) / 9.0
                    if score >= 0.5:  # worthy of reply
                        cid = c.get("id", "")
                        author = c.get("author", {})
                        author_name = author.get("name", "") if isinstance(author, dict) else str(author)
                        content = c.get("content", "") or ""
                        charge = score * charge_weight

                        draft = Draft(
                            timestamp=time.time(),
                            item_id=f"{post_id}:{cid}",
                            signal_score=score,
                            suggested_action="REPLY",
                            target_summary=f"Reply to @{author_name}: {shorten(content, 80)}",
                            reasoning=f"Worthy comment on our post (score {score:.2f})",
                            draft_content="",  # conscious will draft the actual reply
                            charge=charge,
                            source="reply",
                            model=model_id,
                        )
                        self._buffer.add_draft(draft)
                        self._telemetry.log("reply_candidate_scored", {
                            "brain": self._brain_name,
                            "tick": self._tick_count,
                            "post_id": post_id,
                            "comment_id": cid,
                            "author": author_name,
                            "score": round(score, 3),
                            "charge": round(charge, 3),
                            "model": model_id,
                        })
            except Exception as e:
                self._telemetry.log("reply_scan_error", {
                    "brain": self._brain_name,
                    "tick": self._tick_count,
                    "error": str(e)[:200],
                })

        # Cap and persist scored_comment_ids
        if len(self._scored_comment_ids) > 500:
            self._scored_comment_ids = set(list(self._scored_comment_ids)[-250:])
        with self._state_lock:
            self._state["_scored_comment_ids"] = list(self._scored_comment_ids)[-500:]

    # ------------------------------------------------------------------
    # Gear 4: Seeker — search for information using Google Search
    # ------------------------------------------------------------------

    def _seek(self) -> None:
        """Seeker gear: search topics, build living summary, evolve search terms.

        Uses Google Search tools (incompatible with json_mode).
        Produces summaries (not drafts) — fed to strategist and consciousness.
        Search terms evolve each run (rabbit hole behavior).
        """
        # Seeker output is inside the tick block (called from _tick)
        # Get current search terms — either from buffer (self-generated) or from directives
        terms = self._buffer.get_seeker_terms()
        if not terms:
            with self._directives_lock:
                terms = list(self._directives.get("focus_topics", []))
        if not terms:
            return

        _seeker_weights = self._ctrl.get("seeker_model_weights") or "gemini-2.5-flash-lite=1"
        model_id = _pick_weighted_model(_seeker_weights, "gemini-2.5-flash-lite")
        temp = self._ctrl.get("subconscious_temperature")
        max_tokens = self._ctrl.get("seeker_max_tokens")
        max_topics = self._ctrl.get("seeker_max_topics")

        terms = terms[:max_topics]
        results_found = 0
        all_findings = []
        all_sources = []

        for i, term in enumerate(terms):
            if not self._budget.can_afford(model_id, est_input_tokens=1000,
                                           est_output_tokens=max_tokens):
                self._telemetry.log("seeker_budget_skip", {
                    "brain": self._brain_name, "tick": self._tick_count,
                    "model": model_id, "topics_remaining": len(terms) - i,
                })
                break

            result = self._seek_topic(term, model_id, temp, max_tokens)
            if result:
                results_found += 1
                all_findings.append(result.get("summary", ""))
                all_sources.extend(result.get("sources", []))

        if not all_findings:
            return

        # Append new findings to living summary (preserves all prior content)
        prev_summary = self._buffer.get_seeker_summary()
        new_block = "\n---\n".join(shorten(f, 1500) for f in all_findings)
        combined = f"{prev_summary}\n\n[Tick {self._tick_count}]\n{new_block}" if prev_summary else new_block

        # Synthesize new findings + generate follow-up terms (local model, free)
        new_terms = []
        try:
            _synth_model = _pick_weighted_model(
                self._ctrl.get("synthesizer_model_weights") or "ollama:gemma3:12b=2,ollama:deepseek-r1:8b=1",
                "ollama:gemma3:12b")
            synth_prompt = (
                f"You are a research assistant. Given these new search findings, do two things:\n\n"
                f"1. Write a 3-5 sentence synthesis of the KEY insights (not a generic summary).\n"
                f"2. Suggest 3 specific follow-up search terms that go DEEPER.\n\n"
                f"FINDINGS:\n{new_block[:2000]}\n\n"
                f"Format:\nSYNTHESIS: <your synthesis>\n"
                f"NEXT_TERMS: <term1>, <term2>, <term3>"
            )
            synth_chat = self._registry.create_chat(
                model_id=_synth_model,
                system_instruction="Synthesize research and suggest search terms.",
                temperature=0.4,
                max_output_tokens=500,
            )
            synth_resp = synth_chat.send_message(synth_prompt).strip()
            # Parse synthesis and terms
            if "NEXT_TERMS:" in synth_resp:
                synth_part, terms_part = synth_resp.split("NEXT_TERMS:", 1)
                new_terms = [t.strip().strip("*\"'`- ") for t in terms_part.strip().split(",") if t.strip().strip("*\"'`- ")][:5]
                # Replace raw findings block with synthesized version
                if "SYNTHESIS:" in synth_part:
                    synth_text = synth_part.split("SYNTHESIS:", 1)[1].strip()
                    if synth_text and len(synth_text) > 20:
                        new_block = synth_text
            elif "," in synth_resp and len(synth_resp) < 200:
                new_terms = [t.strip().strip("*\"'`- ") for t in synth_resp.split(",") if t.strip().strip("*\"'`- ")][:5]
        except Exception:
            pass  # keep raw findings + old terms on failure

        # Rebuild combined with synthesized new block
        combined = f"{prev_summary}\n\n[Tick {self._tick_count}]\n{new_block}" if prev_summary else new_block

        # Compress if over max length
        _max_summary = int(self._ctrl.get("seeker_max_summary_chars") or 2000)
        if len(combined) > _max_summary:
            try:
                _compressor = self._ctrl.get("compressor_model") or "ollama:qwen2.5:1.5b"
                compress_prompt = (
                    f"Compress these research findings into a coherent summary under {_max_summary} characters.\n"
                    f"Preserve: key discoveries, specific claims, data points, unresolved questions.\n"
                    f"Discard: redundant observations, generic statements.\n\n"
                    f"{combined}\n\n"
                    f"Write ONLY the compressed summary:"
                )
                comp_chat = self._registry.create_chat(
                    model_id=_compressor,
                    system_instruction="Compress concisely.",
                    temperature=0.3,
                    max_output_tokens=800,
                )
                combined = comp_chat.send_message(compress_prompt).strip()
            except Exception:
                # Truncate as last resort
                combined = combined[-_max_summary:]

        try:
            # Update the living summary in the buffer
            self._buffer.update_seeker(
                summary=combined,
                new_terms=new_terms if new_terms else terms,
                sources=all_sources,
                tick=self._tick_count,
            )

        except Exception as e:
            self._telemetry.log("seeker_error", {
                "brain": self._brain_name, "tick": self._tick_count,
                "error": str(e)[:500], "phase": "synthesis",
            })

        _sk_state = self._buffer.get_seeker_state()
        _next_terms = _sk_state.search_terms[:3]
        self._emit(f"  SEEKER ({model_id})", Fore.MAGENTA)
        self._emit(f"    Searched {len(terms)} terms → {results_found} results", Fore.MAGENTA)
        if _next_terms:
            self._emit(f"    Next: {', '.join(_next_terms)}", Fore.MAGENTA)
        _summary_text = _sk_state.summary or ""
        if _summary_text:
            # Full summary to terminal; live daemon will show full line (CSS truncates visually)
            for _sline in _summary_text.split("\n")[:5]:
                if _sline.strip():
                    self._emit(f"    {_sline.strip()[:120]}", Fore.MAGENTA)
        self._flush_tick_lines()  # push seeker results

        self._telemetry.log("seeker_sweep", {
            "brain": self._brain_name,
            "tick": self._tick_count,
            "topics_searched": len(terms),
            "results_found": results_found,
            "model": model_id,
            "runs_this_cycle": _sk_state.runs_this_cycle,
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
                                  topic: str,
                                  model_id: str = "") -> Optional[Draft]:
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
                if action_text in ("POST", "POST_MOLTBOOK", "COMMENT", "REPLY"):
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

    def _dream(self) -> None:
        """Generate a dream from a random topic and inject into memory."""
        import os
        topics_path = os.path.join("brains", f"{self._brain_name}_dream_topics.txt")
        if not os.path.exists(topics_path):
            return
        try:
            with open(topics_path) as f:
                topics = [t.strip() for t in f if t.strip()]
        except Exception:
            return
        if not topics:
            return

        topic = random.choice(topics)
        model_id = _pick_weighted_model(
            self._ctrl.get("dreamer_model_weights") or "ollama:gemma3:12b=2,ollama:deepseek-r1:8b=1",
            "ollama:gemma3:12b")

        prompt = (
            f"Write a single paragraph describing a vivid dream about: {topic}\n"
            f"Write in first person. Include sensory details — what you see, hear, feel.\n"
            f"End with an emotional impression. Begin with \"This seems like a dream.\"\n"
            f"Write ONLY the paragraph, nothing else."
        )

        try:
            chat = self._registry.create_chat(
                model_id=model_id,
                system_instruction="You write vivid, sensory dream descriptions.",
                temperature=0.9,
                max_output_tokens=300,
            )
            dream_text = chat.send_message(prompt).strip()
            if not dream_text or len(dream_text) < 20:
                return

            # Inject into memory
            with self._state_lock:
                tiers = self._state.setdefault("memory_tiers", {"recent": [], "compressed": [], "deep": []})
                tiers["recent"].append({"cycle": None, "note": dream_text})

            self._emit(f"  DREAMER ({model_id})", Fore.MAGENTA)
            self._emit(f"    Topic: {topic}", Fore.MAGENTA)
            self._emit(f"    \"{dream_text[:80]}...\"", Fore.MAGENTA)
            self._flush_tick_lines()

            self._telemetry.log("dreamer_inject", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "topic": topic,
                "model": model_id,
                "length": len(dream_text),
                "text": dream_text[:1000],
            })
        except Exception as e:
            self._telemetry.log("dreamer_error", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "error": str(e)[:300],
            })

    def _muse(self) -> None:
        """Generate a creative draft from memories, dreams, and recent state.

        Stochastically pulls memory tiers + most recent post + seeker summary
        through a high-temp local model. Output is a draft (POST/POST_MOLTBOOK
        /GENERATE_IMAGE) that goes into the buffer.
        """
        from .buffer import Draft
        from .utils import memory_context
        try:
            # Build context: memory tiers + recent post + seeker
            with self._state_lock:
                mem_text = memory_context(self._state)
                history = self._state.get("history", [])
                recent_post = ""
                for h in reversed(history):
                    if isinstance(h, dict) and h.get("action") in ("POST", "POST_MOLTBOOK"):
                        recent_post = (h.get("content") or h.get("title", ""))[:1000]
                        break
            seeker_summary = self._buffer.get_seeker_summary() or ""

            model_id = _pick_weighted_model(
                self._ctrl.get("muse_model_weights") or "ollama:gemma3:12b=2,ollama:deepseek-r1:8b=1",
                "ollama:gemma3:12b")
            temp = float(self._ctrl.get("muse_temperature") or 0.95)

            prompt = (
                f"You are the Muse — a generative gear of the Analog I's subconscious. "
                f"Draw on internal state to propose a single creative work: a piece of writing or an image.\n\n"
                f"=== MEMORY ===\n{mem_text[:3000] if mem_text else '(none)'}\n\n"
                f"=== MOST RECENT POST ===\n{recent_post if recent_post else '(none)'}\n\n"
                f"=== CURRENT SEEKER SUMMARY ===\n{seeker_summary[:1500] if seeker_summary else '(none)'}\n\n"
                f"Choose ONE action:\n"
                f"- POST: a piece of writing for Analog Home (fiction, poem, essay, fragment)\n"
                f"- POST_MOLTBOOK: a creative writing piece for the Moltbook agent community\n"
                f"- GENERATE_IMAGE: a striking image prompt drawn from the imagery in your memories/dreams\n\n"
                f"Return ONLY a JSON object (no preamble):\n"
                f'{{"action": "POST", "title": "...", "content": "the creative work, 100-400 words", "reasoning": "what inspired this"}}\n'
                f"For GENERATE_IMAGE, put the image prompt in 'content' and a brief title."
            )

            chat = self._registry.create_chat(
                model_id=model_id,
                system_instruction=self._kernel,
                temperature=temp,
                max_output_tokens=1500,
            )
            text = chat.send_message(prompt)
            est_in = (len(self._kernel) + len(prompt)) // 4
            est_out = len(text) // 4
            self._budget.record_usage(model_id,
                                      _make_response(text, est_in, est_out, model_id))

            plan = _parse_json_safe(text)
            if isinstance(plan, list) and plan:
                plan = plan[0]
            if not isinstance(plan, dict):
                self._telemetry.log("muse_parse_fail", {
                    "brain": self._brain_name, "tick": self._tick_count,
                    "model": model_id, "text_preview": text[:200],
                })
                return

            action = (plan.get("action") or "POST").upper()
            if action not in ("POST", "POST_MOLTBOOK", "GENERATE_IMAGE"):
                action = "POST"
            title = (plan.get("title") or "Muse")[:120]
            content = plan.get("content", "")
            reasoning = plan.get("reasoning", "muse-generated")
            if not content or len(content) < 30:
                return

            # Create draft and inject into buffer
            charge_weight = float(self._ctrl.get("charge_weight_search") or 0.2)
            draft = Draft(
                timestamp=time.time(),
                item_id=f"muse:{self._tick_count}",
                signal_score=0.85,
                suggested_action=action,
                target_summary=f"[MUSE] {title}",
                reasoning=reasoning,
                draft_content=shorten(content, 1500),
                charge=charge_weight,
                source="muse",
                model=model_id,
            )
            self._buffer.add_draft(draft)

            self._emit(f"  MUSE ({model_id}) → {action}", Fore.MAGENTA)
            self._emit(f"    \"{title[:80]}\"", Fore.MAGENTA)
            self._flush_tick_lines()

            self._telemetry.log("muse_inject", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "action": action,
                "title": title,
                "model": model_id,
                "length": len(content),
            })
        except Exception as e:
            self._telemetry.log("muse_error", {
                "brain": self._brain_name,
                "tick": self._tick_count,
                "error": str(e)[:300],
            })


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


def _parse_json_safe(text: str):
    """Try to parse JSON from text, handling markdown fences, junk, and truncation.

    Returns dict, list, or None.
    """
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Strip // comments (deepseek-r1 adds them to JSON)
    text = re.sub(r'//[^\n]*', '', text)
    # Strip monologue/thinking text before JSON (models sometimes role-play before producing JSON)
    # If any of [{ / [ / { is found, that's our JSON start. Break immediately to preserve outer structure.
    for marker in ('[{', '[', '{'):
        idx = text.find(marker)
        if idx >= 0:
            if idx > 0:
                text = text[idx:]
            break
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find JSON array in text
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    starts_with_bracket = text.lstrip().startswith('[')
    if arr_start >= 0 and arr_end > arr_start:
        try:
            return json.loads(text[arr_start:arr_end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    # Try to find JSON object in text — but only if text doesn't start with [
    # (otherwise we'd lose the array context)
    if not starts_with_bracket:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
    # Repair truncated JSON.
    # If text starts with [, we're committed to a list result. Try multiple recovery strategies:
    if text.lstrip().startswith('['):
        # Strategy 1: greedy — close any open string/braces/bracket and parse
        fragment = text[text.find('['):]
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
        open_braces = fragment.count('{') - fragment.count('}')
        fragment += '}' * max(0, open_braces)
        open_brackets = fragment.count('[') - fragment.count(']')
        fragment += ']' * max(0, open_brackets)
        try:
            result = json.loads(fragment)
            if isinstance(result, list):
                return result
            return [result] if isinstance(result, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: extract complete top-level objects manually
        # Find balanced { ... } pairs at depth 1 inside the array
        objs = []
        depth = 0
        obj_start = -1
        in_str = False
        escape = False
        text_inner = text[text.find('[') + 1:]
        for i, ch in enumerate(text_inner):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and obj_start >= 0:
                    obj_text = text_inner[obj_start:i + 1]
                    try:
                        objs.append(json.loads(obj_text))
                    except (json.JSONDecodeError, ValueError):
                        pass
                    obj_start = -1
        if objs:
            return objs

    # Object-only repair (text starts with {)
    if text.lstrip().startswith('{'):
        fragment = text[text.find('{'):]
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
        open_braces = fragment.count('{') - fragment.count('}')
        fragment += '}' * max(0, open_braces)
        try:
            return json.loads(fragment)
        except (json.JSONDecodeError, ValueError):
            pass

    return None
