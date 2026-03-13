"""AWS Bedrock provider."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, List, Optional

import boto3

from tensorboard_ai.constants import (
    DEFAULT_MODELS,
    ENV_AWS_ACCESS_KEY_ID,
    ENV_AWS_REGION,
    ENV_AWS_SECRET_ACCESS_KEY,
    ENV_AWS_SESSION_TOKEN,
)
from tensorboard_ai.providers.base import ChatMessage

_executor = ThreadPoolExecutor(max_workers=4)


def _region_prefix(region: str) -> str:
    """Extract region prefix for Bedrock cross-region model IDs.

    e.g. "us-east-1" → "us", "eu-west-1" → "eu", "ap-northeast-1" → "ap"
    """
    return region.split("-")[0] if region else "us"


def _ensure_region_prefix(model_id: str, region: str) -> str:
    """Add region prefix to a Bedrock model ID if not already present.

    e.g. "anthropic.claude-sonnet-4-6" + "us-east-1"
      → "us.anthropic.claude-sonnet-4-6"

    Already-prefixed IDs (like "us.anthropic.claude-...") are returned as-is.
    """
    prefix = _region_prefix(region)
    # Already has a region prefix (e.g. "us.anthropic..." or "eu.anthropic...")
    if model_id.split(".")[0] == prefix:
        return model_id
    # Check if the first segment looks like a region prefix (2-letter)
    first = model_id.split(".")[0]
    if len(first) <= 2 and first.isalpha():
        # Has a different region prefix — replace it
        return prefix + "." + ".".join(model_id.split(".")[1:])
    return prefix + "." + model_id


class BedrockProvider:
    def __init__(
        self,
        model: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region: Optional[str] = None,
    ):
        access_key = aws_access_key_id or os.environ.get(ENV_AWS_ACCESS_KEY_ID)
        secret_key = aws_secret_access_key or os.environ.get(ENV_AWS_SECRET_ACCESS_KEY)
        session_token = aws_session_token or os.environ.get(ENV_AWS_SESSION_TOKEN)
        self.region = region or os.environ.get(ENV_AWS_REGION, "us-east-1")
        self.model = _ensure_region_prefix(model or DEFAULT_MODELS["bedrock"], self.region)

        session_kwargs: dict = {"region_name": self.region}
        if access_key and secret_key:
            session_kwargs["aws_access_key_id"] = access_key
            session_kwargs["aws_secret_access_key"] = secret_key
            if session_token:
                session_kwargs["aws_session_token"] = session_token

        self._session = boto3.Session(**session_kwargs)
        self._client = self._session.client("bedrock-runtime")

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        system_prompt: str,
        model: Optional[str] = None,
        use_reasoning: bool = False,
    ) -> AsyncIterator[str]:
        bedrock_messages = [
            {"role": m["role"], "content": [{"text": m["content"]}]} for m in messages
        ]

        resolved_model = self.model
        if model:
            resolved_model = _ensure_region_prefix(model, self.region)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            _executor,
            lambda: self._client.converse_stream(
                modelId=resolved_model,
                messages=bedrock_messages,
                system=[{"text": system_prompt}],
            ),
        )

        event_stream = response["stream"]
        for event in event_stream:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                text = delta.get("text", "")
                if text:
                    yield text
            if "metadata" in event:
                usage = event["metadata"].get("usage", {})
                if usage:
                    yield {
                        "usage": {
                            "input_tokens": usage.get("inputTokens", 0),
                            "output_tokens": usage.get("outputTokens", 0),
                        }
                    }
