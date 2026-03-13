"""Tests for the provider factory."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from tensorboard_ai.providers.base import get_provider


def test_get_provider_anthropic():
    with patch.dict(os.environ, {}, clear=True):
        with patch("tensorboard_ai.providers.anthropic.anthropic"):
            provider = get_provider("anthropic")
    assert type(provider).__name__ == "AnthropicProvider"


def test_get_provider_openai():
    with patch.dict(os.environ, {}, clear=True):
        with patch("tensorboard_ai.providers.openai.openai"):
            provider = get_provider("openai")
    assert type(provider).__name__ == "OpenAIProvider"


def test_get_provider_bedrock():
    with patch.dict(os.environ, {}, clear=True):
        with patch("tensorboard_ai.providers.bedrock.boto3"):
            provider = get_provider("bedrock")
    assert type(provider).__name__ == "BedrockProvider"


def test_get_provider_unsupported():
    with pytest.raises(ValueError, match="Unsupported provider"):
        get_provider("unsupported_llm")


def test_get_provider_env_var_fallback():
    with patch.dict(os.environ, {"TENSORBOARD_AI_PROVIDER": "openai"}, clear=False):
        with patch("tensorboard_ai.providers.openai.openai"):
            provider = get_provider()
    assert type(provider).__name__ == "OpenAIProvider"


def test_get_provider_default():
    with patch.dict(os.environ, {}, clear=True):
        with patch("tensorboard_ai.providers.anthropic.anthropic"):
            provider = get_provider()
    assert type(provider).__name__ == "AnthropicProvider"
