"""Google Gemini LLM implementation."""

import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from .base import LLMClient, ChatSession
from ..config import (
    LLM_TPM_SOFT_CAP, LLM_TPM_CHAR_TO_TOKEN, LLM_TPM_WINDOW_SECONDS,
    LLM_BACKOFF_INITIAL_SECONDS, LLM_BACKOFF_MAX_SECONDS,
)


# ============================================================
# TPM budgeting (approximate) + 429 backoff
# ============================================================
class GeminiBudget:
    def __init__(self):
        self._events: List[Tuple[float, int]] = []
        self._backoff_s = 0.0
        self._blocked_until = 0.0

    def _prune(self) -> None:
        now = time.time()
        self._events = [(t, tok) for (t, tok) in self._events if now - t < LLM_TPM_WINDOW_SECONDS]

    def est_tokens(self, prompt_chars: int) -> int:
        try:
            return int(max(0.0, float(prompt_chars)) / float(LLM_TPM_CHAR_TO_TOKEN))
        except Exception:
            return int(prompt_chars / 4)

    def should_throttle(self, est_tokens: int) -> Tuple[bool, int]:
        self._prune()
        used = sum(tok for _, tok in self._events)
        return (used + est_tokens) > LLM_TPM_SOFT_CAP, used

    def record(self, est_tokens: int) -> None:
        self._prune()
        self._events.append((time.time(), int(est_tokens)))

    def blocked_remaining(self) -> float:
        try:
            return max(0.0, float(self._blocked_until) - time.time())
        except Exception:
            return 0.0

    def note_429(self) -> float:
        if self._backoff_s <= 0:
            self._backoff_s = LLM_BACKOFF_INITIAL_SECONDS
        else:
            self._backoff_s = min(LLM_BACKOFF_MAX_SECONDS, self._backoff_s * 2)
        self._blocked_until = time.time() + float(self._backoff_s)
        return self._backoff_s

    def reset_backoff(self) -> None:
        self._backoff_s = 0.0
        self._blocked_until = 0.0


# Module-level singleton budget tracker
BUDGET = GeminiBudget()


# ============================================================
# Gemini ChatSession wrapper
# ============================================================
class GeminiChatSession(ChatSession):
    def __init__(self, chat, model_name: str):
        self._chat = chat
        self.model_name = model_name

    def send_message(self, prompt: str, json_mode: bool = False) -> str:
        prompt_chars = len(prompt or "")
        est_tokens = BUDGET.est_tokens(prompt_chars)

        # Global cooldown after 429s
        rem = BUDGET.blocked_remaining()
        if rem > 0:
            time.sleep(float(rem))

        # TPM guardrail
        should_throttle, used = BUDGET.should_throttle(est_tokens)
        if should_throttle:
            sleep_s = max(1.0, LLM_TPM_WINDOW_SECONDS - 1.0)
            time.sleep(sleep_s)

        t0 = time.time()
        try:
            if json_mode:
                resp = self._chat.send_message(
                    prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
            else:
                resp = self._chat.send_message(prompt)
        except TypeError:
            # Fallback if config not supported
            resp = self._chat.send_message(prompt)

        raw = getattr(resp, "text", "") or ""
        BUDGET.record(est_tokens)
        BUDGET.reset_backoff()
        return raw


# ============================================================
# Gemini LLMClient
# ============================================================
class GeminiLLMClient(LLMClient):
    def __init__(self, api_key: str, default_model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.default_model = default_model
        self._client = genai.Client(api_key=api_key)

    @property
    def raw_client(self) -> genai.Client:
        """Access the underlying genai.Client (e.g. for challenge solvers)."""
        return self._client

    def create_chat(
        self,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 900,
        model: str = "",
    ) -> GeminiChatSession:
        model_name = model or self.default_model
        chat = self._client.chats.create(
            model=model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction or "",
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return GeminiChatSession(chat, model_name)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.9,
        max_output_tokens: int = 200,
        model: str = "",
    ) -> str:
        model_name = model or self.default_model
        response = self._client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return (response.text or "").strip()
