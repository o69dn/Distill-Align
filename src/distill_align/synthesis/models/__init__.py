"""LLM model integrations for various providers.

The :mod:`~.registry` module provides a provider registry that maps names
(e.g. ``"openai"``, ``"anthropic"``) to metadata and determines which client
class to use.  Built-in providers are registered at import time; custom
providers can be added from config files.
"""

from .registry import (
    ProviderInfo,
    clear_custom,
    get,
    list_all,
    list_names,
    list_select_choices,
    register,
    register_builtins,
)

__all__ = [
    "ProviderInfo",
    "clear_custom",
    "get",
    "list_all",
    "list_names",
    "list_select_choices",
    "register",
    "register_builtins",
]
