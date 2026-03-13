"""Abstract LLM provider protocol and factory."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, TypedDict, Union

from tensorboard_ai.constants import DEFAULT_MODELS, DEFAULT_PROVIDER, ENV_MODEL, ENV_PROVIDER


class ChatMessage(TypedDict):
    role: str  # "user" or "assistant"
    content: str


class LLMProvider(Protocol):
    def stream_chat(
        self,
        messages: List[ChatMessage],
        system_prompt: str,
        model: Optional[str] = None,
        use_reasoning: bool = False,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]: ...


def get_provider(name: Optional[str] = None) -> Any:
    """Create an LLM provider by name. Falls back to env var / default."""
    provider_name = name or os.environ.get(ENV_PROVIDER, DEFAULT_PROVIDER)
    model = os.environ.get(ENV_MODEL) or DEFAULT_MODELS.get(provider_name)

    if provider_name == "anthropic":
        from tensorboard_ai.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model=model)
    elif provider_name == "openai":
        from tensorboard_ai.providers.openai import OpenAIProvider

        return OpenAIProvider(model=model)
    elif provider_name == "bedrock":
        from tensorboard_ai.providers.bedrock import BedrockProvider

        return BedrockProvider(model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")


class ProviderRegistry:
    """Lazy-loading registry of LLM providers.

    Creates provider instances on first use and caches them.
    """

    def __init__(self, default_provider: str):
        self.default_provider = default_provider
        self._cache: Dict[str, Any] = {}

    def get(self, name: Optional[str] = None) -> Any:
        """Get (or create) a provider by name."""
        provider_name = name or self.default_provider
        if provider_name not in self._cache:
            self._cache[provider_name] = get_provider(provider_name)
        return self._cache[provider_name]
