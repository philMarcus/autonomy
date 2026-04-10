"""Model registry: maps model ID strings to provider backends.

Usage:
    registry = ModelRegistry()
    registry.register_backend("gemini", gemini_backend)
    chat = registry.create_chat("gemini-2.5-flash", system_instruction="...")
    resp = registry.generate("gemini-2.5-flash", prompt="...")
"""

from typing import Dict, List, Optional

from .base import (
    ChatSession, CompatAdapter, LLMClient, LLMResponse,
    ModelBackend, ModelInfo,
)


class ModelRegistry:
    """Central registry mapping model_id strings to provider backends."""

    def __init__(self):
        self._backends: Dict[str, ModelBackend] = {}       # provider -> backend
        self._models: Dict[str, ModelInfo] = {}             # model_id -> info
        self._model_to_provider: Dict[str, str] = {}        # model_id -> provider

    def register_backend(self, provider: str, backend: ModelBackend) -> None:
        """Register a provider backend and all its models."""
        self._backends[provider] = backend
        for info in backend.available_models():
            self._models[info.model_id] = info
            self._model_to_provider[info.model_id] = provider

    def get_backend(self, model_id: str) -> ModelBackend:
        """Resolve model_id to its backend. Raises ValueError if unknown."""
        provider = self._model_to_provider.get(model_id)
        if not provider:
            raise ValueError(
                f"Unknown model: {model_id!r}. "
                f"Available: {sorted(self._models.keys())}"
            )
        return self._backends[provider]

    def get_info(self, model_id: str) -> ModelInfo:
        """Get static metadata for a model."""
        if model_id not in self._models:
            raise ValueError(f"Unknown model: {model_id!r}")
        return self._models[model_id]

    # ------------------------------------------------------------------
    # Convenience: delegate to the resolved backend
    # ------------------------------------------------------------------

    def create_chat(self, model_id: str, **kwargs) -> ChatSession:
        return self.get_backend(model_id).create_chat(model_id, **kwargs)

    def generate(self, model_id: str, **kwargs) -> LLMResponse:
        return self.get_backend(model_id).generate(model_id, **kwargs)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_models(
        self,
        local_only: bool = False,
        api_only: bool = False,
        provider: str = "",
    ) -> List[ModelInfo]:
        models = list(self._models.values())
        if local_only:
            models = [m for m in models if m.is_local]
        if api_only:
            models = [m for m in models if not m.is_local]
        if provider:
            models = [m for m in models if m.provider == provider]
        return models

    def has_model(self, model_id: str) -> bool:
        return model_id in self._models

    # ------------------------------------------------------------------
    # v14 compatibility: produce an LLMClient for existing code
    # ------------------------------------------------------------------

    def as_llm_client(self, default_model_id: str) -> LLMClient:
        """Return a v14-compatible LLMClient wrapping this registry."""
        backend = self.get_backend(default_model_id)
        return CompatAdapter(backend, default_model_id)
