"""Anthropic Claude provider."""

from __future__ import annotations

from typing import AsyncIterator, List, Optional

import anthropic

from tensorboard_ai.constants import DEFAULT_MODELS
from tensorboard_ai.providers.base import ChatMessage


class AnthropicProvider:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or DEFAULT_MODELS["anthropic"]
        self.client = (
            anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()
        )

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        system_prompt: str,
        model: Optional[str] = None,
        use_reasoning: bool = False,
    ) -> AsyncIterator[str]:
        resolved_model = model or self.model
        kwargs: dict = {
            "model": resolved_model,
            "max_tokens": 16000,
            "system": system_prompt,
            "messages": messages,
        }

        if use_reasoning:
            kwargs["temperature"] = 1  # required for extended thinking
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

        async with self.client.messages.stream(**kwargs) as stream:
            if not use_reasoning:
                async for text in stream.text_stream:
                    yield text
            else:
                async for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield delta.text
            final = await stream.get_final_message()
            yield {
                "usage": {
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                }
            }
