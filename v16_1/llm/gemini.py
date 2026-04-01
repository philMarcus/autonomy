"""Google Gemini LLM backend for v15.0.

Implements ModelBackend with all Gemini models.  Retains the stateless
generate_content approach with manual history tracking (works around the
Gemini 2.5 Flash first-message empty-response bug).
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from .base import ChatSession, LLMResponse, ModelBackend, ModelInfo
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
# Gemini ChatSession — stateless generate_content with history
# ============================================================
class GeminiChatSession(ChatSession):
    def __init__(self, client: genai.Client, model_name: str,
                 system_instruction: str, temperature: float,
                 max_output_tokens: int,
                 tools: Optional[list] = None):
        self._client = client
        self.model_name = model_name
        self._system_instruction = system_instruction
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._tools = tools
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
        if self._tools:
            config_kwargs["tools"] = self._tools

        try:
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except TypeError:
            config_kwargs.pop("response_mime_type", None)
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            # Trigger exponential backoff on 429 / rate-limit errors
            exc_str = str(exc)
            if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                wait = BUDGET.note_429()
                # Re-raise so caller sees the error
            raise

        # Capture grounding metadata if available
        self._last_grounding_metadata = None
        try:
            self._last_grounding_metadata = resp.candidates[0].grounding_metadata
        except (IndexError, AttributeError):
            pass

        raw = self._extract_text(resp)

        # Append exchange to history for multi-turn support
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
        raw = getattr(resp, "text", "") or ""
        if not raw:
            try:
                parts = resp.candidates[0].content.parts
                raw = "\n".join(p.text for p in parts if getattr(p, "text", None))
            except Exception:
                raw = ""
        return raw


# ============================================================
# Gemini ModelBackend
# ============================================================

# All Gemini models this backend serves
GEMINI_MODELS: List[ModelInfo] = [
    # --- 2.5 family ---
    ModelInfo(
        model_id="gemini-2.5-flash",
        provider="gemini",
        display_name="Gemini 2.5 Flash",
        input_cost_per_1k=0.0003,
        output_cost_per_1k=0.0025,
        supports_tools=True,
    ),
    ModelInfo(
        model_id="gemini-2.5-flash-lite",
        provider="gemini",
        display_name="Gemini 2.5 Flash-Lite",
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0004,
        supports_tools=True,
    ),
    ModelInfo(
        model_id="gemini-2.5-pro",
        provider="gemini",
        display_name="Gemini 2.5 Pro",
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.01,
        supports_tools=True,
    ),
    # --- 3.x family ---
    ModelInfo(
        model_id="gemini-3-flash-preview",
        provider="gemini",
        display_name="Gemini 3 Flash Preview",
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.003,
        supports_tools=True,
    ),
    ModelInfo(
        model_id="gemini-3-pro-preview",
        provider="gemini",
        display_name="Gemini 3 Pro Preview",
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.012,
        supports_tools=True,
    ),
    ModelInfo(
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        display_name="Gemini 3.1 Pro Preview",
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.012,
        supports_tools=True,
    ),
    ModelInfo(
        model_id="gemini-3.1-flash-lite-preview",
        provider="gemini",
        display_name="Gemini 3.1 Flash-Lite Preview",
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.0015,
        supports_tools=True,
    ),
    # --- 2.0 family (deprecated June 2026) ---
    ModelInfo(
        model_id="gemini-2.0-flash",
        provider="gemini",
        display_name="Gemini 2.0 Flash (deprecated)",
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0004,
        supports_tools=True,
    ),
    ModelInfo(
        model_id="gemini-2.0-flash-lite",
        provider="gemini",
        display_name="Gemini 2.0 Flash-Lite (deprecated)",
        input_cost_per_1k=0.000075,
        output_cost_per_1k=0.0003,
    ),
]


class GeminiBackend(ModelBackend):
    """Gemini provider backend — serves all Gemini model variants."""

    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    @property
    def raw_client(self) -> genai.Client:
        """Access the underlying genai.Client (e.g. for challenge solvers)."""
        return self._client

    def create_chat(
        self,
        model_id: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        tools: Optional[list] = None,
    ) -> GeminiChatSession:
        return GeminiChatSession(
            client=self._client,
            model_name=model_id,
            system_instruction=system_instruction or "",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=tools,
        )

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
    ) -> LLMResponse:
        t0 = time.time()
        response = self._client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        latency_ms = int((time.time() - t0) * 1000)
        text = (response.text or "").strip()

        # Try to extract token counts from usage metadata
        input_tokens = 0
        output_tokens = 0
        try:
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        except (AttributeError, TypeError):
            pass

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=model_id,
            latency_ms=latency_ms,
        )

    def available_models(self) -> List[ModelInfo]:
        return list(GEMINI_MODELS)
