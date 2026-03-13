"""Tests for the OpenAI provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tensorboard_ai.providers.openai import OpenAIProvider


@pytest.fixture
def mock_openai():
    with patch("tensorboard_ai.providers.openai.openai") as mock_mod:
        mock_client = MagicMock()
        mock_mod.AsyncOpenAI.return_value = mock_client
        yield mock_mod, mock_client


def test_init_default_model(mock_openai):
    provider = OpenAIProvider()
    assert provider.model == "gpt-4o"


def test_init_custom_model(mock_openai):
    provider = OpenAIProvider(model="gpt-4-turbo")
    assert provider.model == "gpt-4-turbo"


def test_init_with_api_key(mock_openai):
    mock_mod, _ = mock_openai
    OpenAIProvider(api_key="test-key")
    mock_mod.AsyncOpenAI.assert_called_with(api_key="test-key")


@pytest.mark.asyncio
async def test_stream_chat_prepends_system(mock_openai):
    _, mock_client = mock_openai
    provider = OpenAIProvider()

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hello"
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = " world"
    chunk2.usage = None

    async def mock_stream():
        for c in [chunk1, chunk2]:
            yield c

    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

    messages = [{"role": "user", "content": "Hi"}]
    result = []
    async for chunk in provider.stream_chat(messages, "system prompt"):
        result.append(chunk)

    assert result == ["Hello", " world"]

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["messages"][0] == {"role": "system", "content": "system prompt"}
    assert call_kwargs["messages"][1] == {"role": "user", "content": "Hi"}
    assert call_kwargs["stream"] is True


@pytest.mark.asyncio
async def test_stream_chat_skips_empty_delta(mock_openai):
    _, mock_client = mock_openai
    provider = OpenAIProvider()

    chunk_empty = MagicMock()
    chunk_empty.choices = [MagicMock()]
    chunk_empty.choices[0].delta.content = None
    chunk_empty.usage = None

    chunk_real = MagicMock()
    chunk_real.choices = [MagicMock()]
    chunk_real.choices[0].delta.content = "hi"
    chunk_real.usage = None

    async def mock_stream():
        for c in [chunk_empty, chunk_real]:
            yield c

    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

    result = []
    async for chunk in provider.stream_chat([{"role": "user", "content": "x"}], "sys"):
        result.append(chunk)

    assert result == ["hi"]
