"""OpenAI provider."""

from __future__ import annotations

from typing import Any, AsyncIterator, List, Optional

import openai

from tensorboard_ai.constants import DEFAULT_MODELS
from tensorboard_ai.providers.base import ChatMessage


class OpenAIProvider:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or DEFAULT_MODELS["openai"]
        self.client = openai.AsyncOpenAI(api_key=api_key) if api_key else openai.AsyncOpenAI()

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        system_prompt: str,
        model: Optional[str] = None,
        use_reasoning: bool = False,
    ) -> AsyncIterator[str]:
        openai_messages: List[Any] = [{"role": "system", "content": system_prompt}]
        openai_messages.extend({"role": m["role"], "content": m["content"]} for m in messages)

        stream: Any = await self.client.chat.completions.create(
            model=model or self.model,
            messages=openai_messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if chunk.usage:
                yield {
                    "usage": {
                        "input_tokens": chunk.usage.prompt_tokens,
                        "output_tokens": chunk.usage.completion_tokens,
                    }
                }
