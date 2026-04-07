"""Ollama backend for local model inference via REST API.

Ollama runs models natively with GGUF quantization and keeps them warm
in memory. Much faster than HuggingFace/PyTorch (1-5s vs 30-250s).

Requires: Ollama running at http://localhost:11434 (default)
Install: https://ollama.ai
"""

import time
from typing import Any, Dict, List, Optional

import requests

from .base import ChatSession, LLMResponse, ModelBackend, ModelInfo


class OllamaChatSession(ChatSession):
    """Stateful multi-turn chat via Ollama's /api/chat endpoint."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
    ):
        self.model_name = model_name
        self._base_url = base_url
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
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
        if json_mode:
            payload["format"] = "json"

        resp = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        text = data.get("message", {}).get("content", "").strip()

        # Track token usage
        self._last_input_tokens = data.get("prompt_eval_count", 0) or 0
        self._last_output_tokens = data.get("eval_count", 0) or 0

        # Append assistant response to history
        self._history.append({"role": "assistant", "content": text})

        return text


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
    ) -> OllamaChatSession:
        return OllamaChatSession(
            base_url=self._base_url,
            model_name=model_id,
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
        """One-shot generation via /api/generate."""
        ollama_model = model_id
        if ollama_model.startswith("ollama:"):
            ollama_model = ollama_model[7:]

        t0 = time.time()
        resp = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_output_tokens,
                },
            },
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
