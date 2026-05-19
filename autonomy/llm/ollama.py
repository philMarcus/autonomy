"""Ollama backend for local model inference via REST API.

Ollama runs models natively with GGUF quantization and keeps them warm
in memory. Much faster than HuggingFace/PyTorch (1-5s vs 30-250s).

Requires: Ollama running at http://localhost:11434 (default)
Install: https://ollama.ai
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from .base import ChatSession, LLMResponse, ModelBackend, ModelInfo, ToolCall, ToolResult


class OllamaChatSession(ChatSession):
    """Stateful multi-turn chat via Ollama's /api/chat endpoint."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        disable_thinking: bool = False,
    ):
        self.model_name = model_name
        self._base_url = base_url
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._disable_thinking = disable_thinking
        self._history: List[Dict[str, str]] = []
        self._last_input_tokens = 0
        self._last_output_tokens = 0

        if system_instruction:
            self._history.append({"role": "system", "content": system_instruction})

    def send_message(self, prompt: str, json_mode: bool = False) -> str:
        """Send a message and return the model's text response."""
        self._history.append({"role": "user", "content": prompt})

        # Strip the "ollama:" prefix to get the actual Ollama model name
        ollama_model = self.model_name
        if ollama_model.startswith("ollama:"):
            ollama_model = ollama_model[7:]

        payload: Dict[str, Any] = {
            "model": ollama_model,
            "messages": self._history,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_output_tokens,
            },
        }
        # Disable thinking when requested (e.g. sentry scoring — thinking blocks confuse parsers)
        if self._disable_thinking:
            payload["think"] = False
        if json_mode:
            payload["format"] = "json"

        resp = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        msg = data.get("message", {}) or {}
        text = (msg.get("content") or "").strip()
        # Ollama returns the thought trace separately from content for thinking models
        # (qwen3, deepseek-r1). Callers can read `session._last_thinking` to surface
        # reasoning without risking JSON parse failures on the primary response.
        self._last_thinking = (msg.get("thinking") or "").strip()

        # Track token usage
        self._last_input_tokens = data.get("prompt_eval_count", 0) or 0
        self._last_output_tokens = data.get("eval_count", 0) or 0

        # Append assistant response to history
        self._history.append({"role": "assistant", "content": text})

        return text

    def send_message_with_tools(
        self,
        prompt: str,
        tool_schemas: List[Dict],
        tool_executor: Callable[[List[ToolCall]], List[ToolResult]],
        max_rounds: int = 3,
        json_mode: bool = False,
    ) -> str:
        """Tool calling stub — falls back to text-only send_message.

        TODO: implement native Ollama tool calling.
        """
        log = logging.getLogger("autonomy.llm.ollama")
        log.debug("send_message_with_tools: native tool calling not yet "
                   "implemented for Ollama — falling back to send_message")
        return self.send_message(prompt, json_mode=json_mode)


class OllamaBackend(ModelBackend):
    """Ollama provider backend — serves models running in Ollama."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._base_url = base_url.rstrip("/")
        self._models_cache: Optional[List[ModelInfo]] = None

    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def _discover_models(self) -> List[ModelInfo]:
        """Query Ollama for available models."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name", "")
                if not name:
                    continue
                size_bytes = m.get("size", 0)
                size_gb = size_bytes / 1e9 if size_bytes else 0
                # Estimate context from model details (default 128K)
                details = m.get("details", {})
                param_size = details.get("parameter_size", "")

                models.append(ModelInfo(
                    model_id=f"ollama:{name}",
                    provider="ollama",
                    display_name=f"{name} ({size_gb:.1f}GB, Ollama)",
                    is_local=True,
                    input_cost_per_1k=0.0,
                    output_cost_per_1k=0.0,
                    max_context_tokens=128_000,
                    supports_json_mode=True,
                ))
            return models
        except Exception:
            return []

    def create_chat(
        self,
        model_id: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        tools: Optional[list] = None,
        disable_thinking: bool = False,
        **kwargs,
    ) -> OllamaChatSession:
        return OllamaChatSession(
            base_url=self._base_url,
            model_name=model_id,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            disable_thinking=disable_thinking,
        )

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """One-shot generation via /api/generate."""
        ollama_model = model_id
        if ollama_model.startswith("ollama:"):
            ollama_model = ollama_model[7:]

        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
            },
        }
        # Disable thinking for tasks that need clean output (verification, scoring)
        if kwargs.get("disable_thinking", False):
            payload["think"] = False

        t0 = time.time()
        resp = requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = int((time.time() - t0) * 1000)

        text = data.get("response", "").strip()
        input_tokens = data.get("prompt_eval_count", 0) or 0
        output_tokens = data.get("eval_count", 0) or 0

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
            model_id=model_id,
            latency_ms=latency_ms,
        )

    def available_models(self) -> List[ModelInfo]:
        if self._models_cache is None:
            self._models_cache = self._discover_models()
        return self._models_cache
