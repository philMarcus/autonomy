"""OpenAI LLM backend for v15.0.

Implements ModelBackend for GPT models via the OpenAI Python SDK.
Requires: pip install openai
"""

import time
from typing import Any, Dict, List, Optional

from .base import ChatSession, LLMResponse, ModelBackend, ModelInfo


# All OpenAI models this backend serves
OPENAI_MODELS: List[ModelInfo] = [
    ModelInfo(
        model_id="gpt-4o-mini",
        provider="openai",
        display_name="GPT-4o Mini",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        max_context_tokens=128_000,
        supports_json_mode=True,
    ),
    ModelInfo(
        model_id="gpt-4o",
        provider="openai",
        display_name="GPT-4o",
        input_cost_per_1k=0.0025,
        output_cost_per_1k=0.01,
        max_context_tokens=128_000,
        supports_json_mode=True,
    ),
]


class OpenAIChatSession(ChatSession):
    """Multi-turn chat via OpenAI Chat Completions with manual history."""

    def __init__(self, client: Any, model_id: str,
                 system_instruction: str, temperature: float,
                 max_output_tokens: int):
        self._client = client
        self.model_name = model_id
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._messages: List[Dict[str, str]] = []
        if system_instruction:
            self._messages.append({"role": "system", "content": system_instruction})

    def send_message(self, prompt: str, json_mode: bool = False) -> str:
        self._messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": list(self._messages),
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""

        self._messages.append({"role": "assistant", "content": text})
        return text


class OpenAIBackend(ModelBackend):
    """OpenAI provider backend — serves GPT models."""

    def __init__(self, api_key: str):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI backend requires the 'openai' package. "
                "Install it with: pip install openai"
            )
        self._client = openai.OpenAI(api_key=api_key)

    def create_chat(
        self,
        model_id: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        tools: Optional[list] = None,
    ) -> OpenAIChatSession:
        return OpenAIChatSession(
            client=self._client,
            model_id=model_id,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
    ) -> LLMResponse:
        t0 = time.time()
        resp = self._client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        latency_ms = int((time.time() - t0) * 1000)

        text = resp.choices[0].message.content or ""
        input_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(resp.usage, "completion_tokens", 0) or 0

        return LLMResponse(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=model_id,
            latency_ms=latency_ms,
        )

    def available_models(self) -> List[ModelInfo]:
        return list(OPENAI_MODELS)
