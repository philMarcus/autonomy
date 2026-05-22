"""Prompt template loader — cached file reads from the prompt_templates directory."""

import os
from typing import Dict

_CACHE: Dict[str, str] = {}
_DIR = os.path.dirname(__file__)


def load_template(name: str) -> str:
    """Load a prompt template by relative path, cached after first read.

    Args:
        name: Relative path like "conscious/action_policy.txt"

    Returns:
        Template contents as a string. Use .format(**kwargs) to fill {slots}.
    """
    if name not in _CACHE:
        path = os.path.join(_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            _CACHE[name] = f.read()
    return _CACHE[name]


def clear_cache() -> None:
    """Clear the template cache (useful for testing after editing templates)."""
    _CACHE.clear()
