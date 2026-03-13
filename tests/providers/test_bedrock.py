"""Tests for the Bedrock provider."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from tensorboard_ai.providers.bedrock import BedrockProvider, _ensure_region_prefix


@pytest.fixture
def mock_boto3():
    with patch("tensorboard_ai.providers.bedrock.boto3") as mock:
        mock_session = MagicMock()
        mock.Session.return_value = mock_session
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        yield mock, mock_session, mock_client


class TestRegionPrefix:
    def test_adds_us_prefix(self):
        assert _ensure_region_prefix("anthropic.claude-sonnet-4-6", "us-east-1") == \
            "us.anthropic.claude-sonnet-4-6"

    def test_adds_eu_prefix(self):
        assert _ensure_region_prefix("anthropic.claude-sonnet-4-6", "eu-west-1") == \
            "eu.anthropic.claude-sonnet-4-6"

    def test_adds_ap_prefix(self):
        assert _ensure_region_prefix("anthropic.claude-opus-4-6-v1", "ap-northeast-1") == \
            "ap.anthropic.claude-opus-4-6-v1"

    def test_already_correct_prefix(self):
        assert _ensure_region_prefix("us.anthropic.claude-sonnet-4-6", "us-east-1") == \
            "us.anthropic.claude-sonnet-4-6"

    def test_replaces_wrong_prefix(self):
        assert _ensure_region_prefix("us.anthropic.claude-sonnet-4-6", "eu-west-1") == \
            "eu.anthropic.claude-sonnet-4-6"


def test_init_iam_role_no_explicit_creds(mock_boto3):
    """IAM role auth: no explicit creds → boto3 default credential chain."""
    mock, mock_session, _ = mock_boto3
    with patch.dict(os.environ, {}, clear=True):
        provider = BedrockProvider()
    mock.Session.assert_called_once_with(region_name="us-east-1")
    mock_session.client.assert_called_once_with("bedrock-runtime")
    assert provider.model == "us.anthropic.claude-sonnet-4-6"


def test_init_explicit_key_auth(mock_boto3):
    """Explicit key auth: creds passed directly."""
    mock, _, _ = mock_boto3
    with patch.dict(os.environ, {}, clear=True):
        BedrockProvider(
            aws_access_key_id="AKIA...",
            aws_secret_access_key="secret",
            aws_session_token="token",
            region="eu-west-1",
        )
    mock.Session.assert_called_once_with(
        region_name="eu-west-1",
        aws_access_key_id="AKIA...",
        aws_secret_access_key="secret",
        aws_session_token="token",
    )


def test_init_env_var_creds(mock_boto3):
    """Explicit key auth via env vars."""
    mock, _, _ = mock_boto3
    env = {
        "AWS_ACCESS_KEY_ID": "env-key",
        "AWS_SECRET_ACCESS_KEY": "env-secret",
        "AWS_REGION": "ap-northeast-1",
    }
    with patch.dict(os.environ, env, clear=True):
        provider = BedrockProvider()
    mock.Session.assert_called_once_with(
        region_name="ap-northeast-1",
        aws_access_key_id="env-key",
        aws_secret_access_key="env-secret",
    )
    assert provider.model == "ap.anthropic.claude-sonnet-4-6"


def test_init_region_fallback(mock_boto3):
    """Region falls back to us-east-1."""
    mock, _, _ = mock_boto3
    with patch.dict(os.environ, {}, clear=True):
        provider = BedrockProvider()
    assert provider.region == "us-east-1"


def test_init_custom_model_gets_prefix(mock_boto3):
    with patch.dict(os.environ, {}, clear=True):
        provider = BedrockProvider(model="anthropic.claude-opus-4-6-v1")
    assert provider.model == "us.anthropic.claude-opus-4-6-v1"


def test_init_eu_region_model(mock_boto3):
    with patch.dict(os.environ, {}, clear=True):
        provider = BedrockProvider(model="anthropic.claude-sonnet-4-6", region="eu-west-1")
    assert provider.model == "eu.anthropic.claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_stream_chat_converse_stream(mock_boto3):
    """Test converse_stream parsing: contentBlockDelta events."""
    _, _, mock_client = mock_boto3
    with patch.dict(os.environ, {}, clear=True):
        provider = BedrockProvider()

    mock_client.converse_stream.return_value = {
        "stream": [
            {"contentBlockDelta": {"delta": {"text": "Hello"}}},
            {"contentBlockStart": {"contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " world"}}},
            {"contentBlockDelta": {"delta": {}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    }

    messages = [{"role": "user", "content": "Hi"}]
    result = []
    async for chunk in provider.stream_chat(messages, "system prompt"):
        result.append(chunk)

    assert result == ["Hello", " world"]

    call_kwargs = mock_client.converse_stream.call_args[1]
    assert call_kwargs["modelId"] == "us.anthropic.claude-sonnet-4-6"
    assert call_kwargs["system"] == [{"text": "system prompt"}]
    assert call_kwargs["messages"] == [{"role": "user", "content": [{"text": "Hi"}]}]


@pytest.mark.asyncio
async def test_stream_chat_model_override_gets_prefix(mock_boto3):
    """Model override from frontend also gets region prefix."""
    _, _, mock_client = mock_boto3
    with patch.dict(os.environ, {}, clear=True):
        provider = BedrockProvider()

    mock_client.converse_stream.return_value = {"stream": []}

    messages = [{"role": "user", "content": "Hi"}]
    async for _ in provider.stream_chat(messages, "sys", model="anthropic.claude-opus-4-6-v1"):
        pass

    call_kwargs = mock_client.converse_stream.call_args[1]
    assert call_kwargs["modelId"] == "us.anthropic.claude-opus-4-6-v1"
