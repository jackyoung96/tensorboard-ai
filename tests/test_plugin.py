"""Tests for the TensorBoard AI plugin."""

from __future__ import annotations

import json
import os
from io import BytesIO
from unittest.mock import MagicMock, patch


from tensorboard_ai.backend.server import chat_handler, health_handler, runs_handler
from tensorboard_ai.backend.data_access import TBDataAccess


def make_environ(method="GET", path="/", body=None, content_type="application/json"):
    """Create a minimal WSGI environ dict."""
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "SERVER_NAME": "localhost",
        "SERVER_PORT": "6006",
        "HTTP_HOST": "localhost:6006",
        "wsgi.url_scheme": "http",
        "CONTENT_TYPE": content_type,
    }
    if body is not None:
        body_bytes = body.encode("utf-8") if isinstance(body, str) else body
        environ["wsgi.input"] = BytesIO(body_bytes)
        environ["CONTENT_LENGTH"] = str(len(body_bytes))
    else:
        environ["wsgi.input"] = BytesIO(b"")
        environ["CONTENT_LENGTH"] = "0"
    return environ


def collect_response(app, environ):
    """Call a WSGI app and collect status + headers + body."""
    status_holder = {}
    headers_holder = {}

    def start_response(status, headers):
        status_holder["status"] = status
        headers_holder["headers"] = dict(headers)

    body_parts = list(app(environ, start_response))
    body = b"".join(part.encode("utf-8") if isinstance(part, str) else part for part in body_parts)
    return status_holder.get("status", ""), headers_holder.get("headers", {}), body


class TestPlugin:
    def test_init_with_mock_context(self, mock_tb_context):
        with patch.dict(os.environ, {}, clear=True):
            from tensorboard_ai.plugin import TensorBoardAIPlugin

            plugin = TensorBoardAIPlugin(mock_tb_context)
            assert plugin.plugin_name == "tensorboard_ai"

    def test_is_active(self, mock_tb_context):
        with patch.dict(os.environ, {}, clear=True):
            from tensorboard_ai.plugin import TensorBoardAIPlugin

            plugin = TensorBoardAIPlugin(mock_tb_context)
            assert plugin.is_active() is True

    def test_frontend_metadata(self, mock_tb_context):
        with patch.dict(os.environ, {}, clear=True):
            from tensorboard_ai.plugin import TensorBoardAIPlugin

            plugin = TensorBoardAIPlugin(mock_tb_context)
            meta = plugin.frontend_metadata()
            assert meta.es_module_path == "/index.js"
            assert meta.tab_name == "AI"

    def test_get_plugin_apps(self, mock_tb_context):
        with patch.dict(os.environ, {}, clear=True):
            from tensorboard_ai.plugin import TensorBoardAIPlugin

            plugin = TensorBoardAIPlugin(mock_tb_context)
            apps = plugin.get_plugin_apps()
            assert "/chat" in apps
            assert "/runs" in apps
            assert "/health" in apps
            assert "/config" in apps


class TestConfigHandler:
    def test_config_debug_off(self, mock_tb_context):
        with patch.dict(os.environ, {}, clear=True):
            from tensorboard_ai.plugin import TensorBoardAIPlugin

            plugin = TensorBoardAIPlugin(mock_tb_context)
            apps = plugin.get_plugin_apps()
            handler = apps["/config"]
            environ = make_environ()
            status, _, body = collect_response(handler, environ)
            assert "200" in status
            data = json.loads(body)
            assert data["debug"] is False

    def test_config_debug_on(self, mock_tb_context):
        mock_tb_context.flags.tensorboard_ai_debug = True
        with patch.dict(os.environ, {}, clear=True):
            from tensorboard_ai.plugin import TensorBoardAIPlugin

            plugin = TensorBoardAIPlugin(mock_tb_context)
            apps = plugin.get_plugin_apps()
            handler = apps["/config"]
            environ = make_environ()
            status, _, body = collect_response(handler, environ)
            assert "200" in status
            data = json.loads(body)
            assert data["debug"] is True

    def test_config_providers(self, mock_tb_context):
        with patch.dict(os.environ, {}, clear=True):
            from tensorboard_ai.plugin import TensorBoardAIPlugin

            plugin = TensorBoardAIPlugin(mock_tb_context)
            apps = plugin.get_plugin_apps()
            handler = apps["/config"]
            environ = make_environ()
            status, _, body = collect_response(handler, environ)
            data = json.loads(body)
            assert data["default_provider"] == "anthropic"
            assert "providers" in data
            assert "anthropic" in data["providers"]
            assert "openai" in data["providers"]
            assert "bedrock" in data["providers"]
            assert len(data["providers"]["anthropic"]["models"]) > 0


class TestHealthHandler:
    def test_health(self):
        handler = health_handler()
        environ = make_environ()
        status, headers, body = collect_response(handler, environ)
        assert "200" in status
        data = json.loads(body)
        assert data == {"status": "ok"}


class TestRunsHandler:
    def test_runs(self, mock_data_provider):
        da = TBDataAccess(data_provider=mock_data_provider)
        handler = runs_handler(da)
        environ = make_environ()
        status, headers, body = collect_response(handler, environ)
        assert "200" in status
        data = json.loads(body)
        assert "train" in data
        assert "eval" in data


class TestChatHandler:
    def test_chat_method_not_allowed(self, mock_data_provider, mock_provider):
        da = TBDataAccess(data_provider=mock_data_provider)
        handler = chat_handler(da, mock_provider)
        environ = make_environ(method="GET")
        status, _, body = collect_response(handler, environ)
        assert "405" in status

    def test_chat_invalid_json(self, mock_data_provider, mock_provider):
        da = TBDataAccess(data_provider=mock_data_provider)
        handler = chat_handler(da, mock_provider)
        environ = make_environ(method="POST", body="not json")
        status, _, body = collect_response(handler, environ)
        assert "400" in status

    def test_chat_no_messages(self, mock_data_provider, mock_provider):
        da = TBDataAccess(data_provider=mock_data_provider)
        handler = chat_handler(da, mock_provider)
        environ = make_environ(method="POST", body='{"messages": []}')
        status, _, body = collect_response(handler, environ)
        assert "400" in status

    def test_chat_sse_streaming(self, mock_data_provider, mock_provider):
        da = TBDataAccess(data_provider=mock_data_provider)
        handler = chat_handler(da, mock_provider)
        body = json.dumps({"messages": [{"role": "user", "content": "Hello"}]})
        environ = make_environ(method="POST", body=body)

        status_holder = {}

        def start_response(status, headers):
            status_holder["status"] = status
            status_holder["headers"] = dict(headers)

        chunks = list(handler(environ, start_response))
        assert "200" in status_holder["status"]
        assert status_holder["headers"]["Content-Type"] == "text/event-stream"

        # Parse SSE events
        events = []
        for chunk in chunks:
            text = chunk if isinstance(chunk, str) else chunk.decode("utf-8")
            for line in text.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

        text_events = [e for e in events if e["type"] == "text"]
        assert len(text_events) > 0

        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert "usage" in done_events[0]
        assert done_events[0]["usage"]["input_tokens"] == 100
        assert done_events[0]["usage"]["output_tokens"] == 50

    def test_chat_with_chart_response(self, mock_data_provider):
        """Test that chart specs in LLM response produce chart SSE events."""
        chart_response = """Here is a chart:
```data_request
[{"run": "train", "tag": "loss"}]
```
```plotly
{"data": [{"x": [], "y": [], "type": "scatter"}], "layout": {"title": "Loss"}}
```"""

        class ChartProvider:
            async def stream_chat(self, messages, system_prompt, model=None, use_reasoning=False):
                yield chart_response
                yield {"usage": {"input_tokens": 200, "output_tokens": 100}}

        da = TBDataAccess(data_provider=mock_data_provider)
        handler = chat_handler(da, ChartProvider())
        body = json.dumps({"messages": [{"role": "user", "content": "plot loss"}]})
        environ = make_environ(method="POST", body=body)

        status_holder = {}

        def start_response(status, headers):
            status_holder["status"] = status

        chunks = list(handler(environ, start_response))

        events = []
        for chunk in chunks:
            text = chunk if isinstance(chunk, str) else chunk.decode("utf-8")
            for line in text.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

        chart_events = [e for e in events if e["type"] == "chart"]
        assert len(chart_events) == 1
        assert chart_events[0]["spec"]["layout"]["title"] == "Loss"
        assert "chart_data" in chart_events[0]
