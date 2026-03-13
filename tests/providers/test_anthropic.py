"""Tests for the Anthropic provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tensorboard_ai.providers.anthropic import AnthropicProvider


@pytest.fixture
def mock_anthropic():
    with patch("tensorboard_ai.providers.anthropic.anthropic") as mock_mod:
        mock_client = MagicMock()
        mock_mod.AsyncAnthropic.return_value = mock_client
        yield mock_mod, mock_client


def test_init_default_model(mock_anthropic):
    provider = AnthropicProvider()
    assert provider.model == "claude-sonnet-4-6"


def test_init_custom_model(mock_anthropic):
    provider = AnthropicProvider(model="claude-opus-4-20250514")
    assert provider.model == "claude-opus-4-20250514"


def test_init_with_api_key(mock_anthropic):
    mock_mod, _ = mock_anthropic
    AnthropicProvider(api_key="test-key")
    mock_mod.AsyncAnthropic.assert_called_with(api_key="test-key")


@pytest.mark.asyncio
async def test_stream_chat(mock_anthropic):
    _, mock_client = mock_anthropic
    provider = AnthropicProvider()

    # Create a mock async context manager for stream
    chunks = ["Hello", " world"]
    mock_stream_cm = AsyncMock()
    mock_stream_cm.__aenter__.return_value = mock_stream_cm

    async def mock_text_stream():
        for chunk in chunks:
            yield chunk

    mock_stream_cm.text_stream = mock_text_stream()

    # Mock get_final_message for usage tracking
    mock_final = MagicMock()
    mock_final.usage.input_tokens = 50
    mock_final.usage.output_tokens = 30
    mock_stream_cm.get_final_message = AsyncMock(return_value=mock_final)

    mock_client.messages.stream.return_value = mock_stream_cm

    messages = [{"role": "user", "content": "Hi"}]
    result = []
    async for chunk in provider.stream_chat(messages, "system prompt"):
        result.append(chunk)

    assert result[:-1] == ["Hello", " world"]
    assert result[-1] == {"usage": {"input_tokens": 50, "output_tokens": 30}}
    mock_client.messages.stream.assert_called_once_with(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        system="system prompt",
        messages=messages,
    )
