"""TensorBoard AI Plugin — main plugin class."""

from __future__ import annotations

import json
import mimetypes
import os
from typing import Any, Callable, Dict

from werkzeug import Request, Response

from tensorboard.plugins import base_plugin

from tensorboard_ai.backend.data_access import TBDataAccess
from tensorboard_ai.backend.server import (
    chat_handler,
    health_handler,
    runs_handler,
    scalars_handler,
)
from tensorboard_ai.constants import (
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    ENV_MODEL,
    ENV_PROVIDER,
    MODEL_CHOICES,
    PLUGIN_NAME,
)
from tensorboard_ai.providers.base import ProviderRegistry

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# CLI flag name
_FLAG_DEBUG = "tensorboard_ai_debug"


def _static_file_handler(filename: str) -> Callable:
    """Return a WSGI handler that serves a static file."""
    filepath = os.path.join(_STATIC_DIR, filename)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    def handler(environ: dict, start_response: Callable) -> Any:
        try:
            with open(filepath, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            response = Response("Not found", status=404)
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response(environ, start_response)
        response = Response(body, content_type=content_type)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response(environ, start_response)

    return handler


def _config_handler(config: dict) -> Callable:
    """Return a WSGI handler that serves plugin configuration to the frontend."""
    config_json = json.dumps(config).encode("utf-8")

    def handler(environ: dict, start_response: Callable) -> Any:
        response = Response(config_json, content_type="application/json")
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response(environ, start_response)

    return handler


class TensorBoardAIPlugin(base_plugin.TBPlugin):
    """TensorBoard plugin that adds conversational AI capabilities."""

    plugin_name = PLUGIN_NAME

    def __init__(self, context: Any):
        self._data_access = TBDataAccess(
            data_provider=context.data_provider,
        )
        self._debug = getattr(context.flags, _FLAG_DEBUG, False) if context.flags else False

        # Determine default provider/model from env
        self._default_provider = os.environ.get(ENV_PROVIDER, DEFAULT_PROVIDER)
        self._default_model = os.environ.get(ENV_MODEL) or DEFAULT_MODELS.get(
            self._default_provider, ""
        )
        self._registry = ProviderRegistry(self._default_provider)

    def get_plugin_apps(self) -> Dict[str, Any]:
        config = {
            "debug": self._debug,
            "default_provider": self._default_provider,
            "default_model": self._default_model,
            "providers": {
                name: {
                    "default_model": DEFAULT_MODELS[name],
                    "models": MODEL_CHOICES.get(name, []),
                }
                for name in ("anthropic", "openai", "bedrock")
            },
        }
        return {
            "/chat": chat_handler(self._data_access, self._registry),
            "/runs": runs_handler(self._data_access),
            "/scalars": scalars_handler(self._data_access),
            "/health": health_handler(),
            "/config": _config_handler(config),
            "/index.js": _static_file_handler("index.js"),
            "/chat.js": _static_file_handler("chat.js"),
            "/chart_renderer.js": _static_file_handler("chart_renderer.js"),
            "/plotly-basic.min.js": _static_file_handler("plotly-basic.min.js"),
            "/marked.min.js": _static_file_handler("marked.min.js"),
        }

    def is_active(self) -> bool:
        return True

    def frontend_metadata(self) -> base_plugin.FrontendMetadata:
        return base_plugin.FrontendMetadata(
            es_module_path="/index.js",
            tab_name="AI",
        )


class TensorBoardAILoader(base_plugin.TBLoader):
    """Custom loader that registers --tensorboard_ai_debug CLI flag."""

    def define_flags(self, parser):
        parser.add_argument(
            "--tensorboard_ai_debug",
            action="store_true",
            default=False,
            help="Enable debug mode (show raw prompts in the AI panel).",
        )

    def load(self, context):
        return TensorBoardAIPlugin(context)
