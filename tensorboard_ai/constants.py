"""Shared constants for tensorboard-ai."""

PLUGIN_NAME = "tensorboard_ai"

DEFAULT_PROVIDER = "anthropic"

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "bedrock": "anthropic.claude-sonnet-4-6",
}

MODEL_CHOICES = {
    "anthropic": [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5-20251001",
    ],
    "openai": [
        "gpt-5.4-pro",
        "gpt-5.4",
        "gpt-5-mini",
        "gpt-4o",
    ],
    "bedrock": [
        "anthropic.claude-opus-4-6-v1",
        "anthropic.claude-sonnet-4-6",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
    ],
}

ENV_PROVIDER = "TENSORBOARD_AI_PROVIDER"
ENV_MODEL = "TENSORBOARD_AI_MODEL"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
ENV_AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
ENV_AWS_SESSION_TOKEN = "AWS_SESSION_TOKEN"
ENV_AWS_REGION = "AWS_REGION"

MAX_SCALAR_POINTS = 1000
