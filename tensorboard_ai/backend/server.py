"""WSGI request handlers for the TensorBoard AI plugin."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List

from werkzeug import Request, Response

from tensorboard_ai.backend.chart_gen import extract_chart_spec, extract_data_requests
from tensorboard_ai.backend.data_access import TBDataAccess
from tensorboard_ai.prompts.system import build_system_prompt, detect_training_type
from tensorboard_ai.providers.base import ChatMessage, LLMProvider, ProviderRegistry


def _run_async_generator(async_gen_func: Callable, *args: Any):
    """Run an async generator in a new event loop, yielding results."""
    loop = asyncio.new_event_loop()
    try:
        agen = async_gen_func(*args)
        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()


def _sse_event(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")


def chat_handler(data_access: TBDataAccess, provider_or_registry) -> Callable:
    """Return a WSGI handler for the /chat SSE endpoint.

    Accepts either a single LLMProvider or a ProviderRegistry.
    """

    def _resolve_provider(provider_name: str | None):
        if isinstance(provider_or_registry, ProviderRegistry):
            return provider_or_registry.get(provider_name)
        return provider_or_registry

    def handler(environ: dict, start_response: Callable) -> Any:
        request = Request(environ)
        if request.method != "POST":
            response = Response("Method not allowed", status=405)
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response(environ, start_response)

        try:
            body = json.loads(request.get_data(as_text=True))
        except (json.JSONDecodeError, TypeError):
            response = Response("Invalid JSON", status=400)
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response(environ, start_response)

        messages: List[ChatMessage] = body.get("messages", [])
        if not messages:
            response = Response("No messages provided", status=400)
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response(environ, start_response)

        # Optional settings from the request body
        provider_name: str | None = body.get("provider") or None
        model_override: str | None = body.get("model") or None
        language: str = body.get("language", "")
        use_reasoning: bool = body.get("use_reasoning", False)

        provider = _resolve_provider(provider_name)

        data_context = data_access.get_data_context()
        runs_and_tags = data_access.list_runs_and_tags()
        training_type = detect_training_type(runs_and_tags)
        hyperparameters_context = data_access.get_hyperparameters_context()
        system_prompt = build_system_prompt(
            data_context, training_type, hyperparameters_context
        )

        if language:
            system_prompt += f"\n\nIMPORTANT: Always respond in {language}."

        def generate():
            full_response = []
            usage_info = None
            try:
                for chunk in _run_async_generator(
                    provider.stream_chat,
                    messages,
                    system_prompt,
                    model_override,
                    use_reasoning,
                ):
                    if isinstance(chunk, dict) and "usage" in chunk:
                        usage_info = chunk["usage"]
                        continue
                    full_response.append(chunk)
                    yield _sse_event({"type": "text", "content": chunk})

                full_text = "".join(full_response)

                data_requests = extract_data_requests(full_text)
                chart_spec = extract_chart_spec(full_text)
                if chart_spec is not None:
                    chart_data: Dict[str, Dict[str, list]] = {}
                    if data_requests:
                        pairs = [(r["run"], r["tag"]) for r in data_requests]
                        chart_data = data_access.read_scalars_for_tags(pairs)
                    yield _sse_event(
                        {
                            "type": "chart",
                            "spec": chart_spec,
                            "chart_data": chart_data,
                        }
                    )

            except Exception as e:
                yield _sse_event({"type": "error", "content": str(e)})

            done_data: dict = {"type": "done"}
            if usage_info:
                done_data["usage"] = usage_info
            yield _sse_event(done_data)

        start_response(
            "200 OK",
            [
                ("Content-Type", "text/event-stream"),
                ("Cache-Control", "no-cache"),
                ("Connection", "keep-alive"),
                ("X-Accel-Buffering", "no"),
                ("X-Content-Type-Options", "nosniff"),
            ],
        )
        return generate()

    return handler


def runs_handler(data_access: TBDataAccess) -> Callable:
    """Return a WSGI handler for the /runs endpoint."""

    def handler(environ: dict, start_response: Callable) -> Any:
        runs_and_tags = data_access.list_runs_and_tags()
        response = Response(
            json.dumps(runs_and_tags),
            content_type="application/json",
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response(environ, start_response)

    return handler


def scalars_handler(data_access: TBDataAccess) -> Callable:
    """Return a WSGI handler for the /scalars endpoint (all scalar data)."""

    def handler(environ: dict, start_response: Callable) -> Any:
        runs_and_tags = data_access.list_runs_and_tags()
        pairs = []
        for run, tags in runs_and_tags.items():
            for tag in tags.get("scalars", []):
                pairs.append((run, tag))

        if not pairs:
            response = Response(json.dumps({}), content_type="application/json")
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response(environ, start_response)

        scalars = data_access.read_scalars_for_tags(pairs)
        response = Response(json.dumps(scalars), content_type="application/json")
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response(environ, start_response)

    return handler



def health_handler() -> Callable:
    """Return a WSGI handler for the /health endpoint."""

    def handler(environ: dict, start_response: Callable) -> Any:
        response = Response(
            json.dumps({"status": "ok"}),
            content_type="application/json",
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response(environ, start_response)

    return handler
