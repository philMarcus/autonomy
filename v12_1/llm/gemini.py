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
# Gemini ChatSession wrapper — uses stateless generate_content
# with manual history tracking to avoid the chats.create
# first-message empty-response bug in Gemini 2.5 Flash.
# ============================================================
class GeminiChatSession(ChatSession):
    def __init__(self, client: genai.Client, model_name: str,
                 system_instruction: str, temperature: float,
                 max_output_tokens: int):
        self._client = client
        self.model_name = model_name
        self._system_instruction = system_instruction
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._history: List[types.Content] = []

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

        # Build contents: history + new user message
        contents = list(self._history) + [
            types.Content(role="user", parts=[types.Part(text=prompt)])
        ]

        config_kwargs: Dict[str, Any] = {
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
        }
        if self._system_instruction:
            config_kwargs["system_instruction"] = self._system_instruction
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        # No ThinkingConfig override — let models use their default thinking
        # behavior. The large max_output_tokens (16384) provides enough
        # headroom for thinking + a complete JSON response.

        try:
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except TypeError:
            # Fallback without json_mode if config not supported
            config_kwargs.pop("response_mime_type", None)
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )

        raw = self._extract_text(resp)

        # Append exchange to history for multi-turn support (repair prompts)
        self._history.append(
            types.Content(role="user", parts=[types.Part(text=prompt)])
        )
        if raw:
            self._history.append(
                types.Content(role="model", parts=[types.Part(text=raw)])
            )

        BUDGET.record(est_tokens)
        BUDGET.reset_backoff()
        return raw

    @staticmethod
    def _extract_text(resp) -> str:
        """Get text from a Gemini response, trying .text then individual parts."""
        raw = getattr(resp, "text", "") or ""
        if not raw:
            try:
                parts = resp.candidates[0].content.parts
                raw = "\n".join(p.text for p in parts if getattr(p, "text", None))
            except Exception:
                raw = ""
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
        return GeminiChatSession(
            client=self._client,
            model_name=model_name,
            system_instruction=system_instruction or "",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

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
