"""Shared test fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


@dataclass
class ScalarPoint:
    step: int
    value: float
    wall_time: float = 0.0


def make_mock_data_provider(
    scalars: Dict[str, Dict[str, List[ScalarPoint]]] | None = None,
    tensors: Dict[str, Dict[str, Any]] | None = None,
    blob_sequences: Dict[str, Dict[str, Any]] | None = None,
) -> MagicMock:
    """Create a mock DataProvider with configurable data."""
    provider = MagicMock()

    scalar_listing = {}
    if scalars:
        for run, tags in scalars.items():
            scalar_listing[run] = {tag: MagicMock() for tag in tags}
    provider.list_scalars.return_value = scalar_listing

    tensor_listing = {}
    if tensors:
        for run, tags in tensors.items():
            tensor_listing[run] = {tag: MagicMock() for tag in tags}
    provider.list_tensors.return_value = tensor_listing

    blob_listing = {}
    if blob_sequences:
        for run, tags in blob_sequences.items():
            blob_listing[run] = {tag: MagicMock() for tag in tags}
    provider.list_blob_sequences.return_value = blob_listing

    def read_scalars_side_effect(
        ctx=None, experiment_id="", plugin_name="", run_tag_filter=None, downsample=1000
    ):
        if not scalars or run_tag_filter is None:
            return {}
        # Support both RunTagFilter objects and plain dicts (for flexibility)
        if hasattr(run_tag_filter, "runs"):
            filter_runs = run_tag_filter.runs  # frozenset or None
            filter_tags = run_tag_filter.tags  # frozenset or None
        else:
            filter_runs = set(run_tag_filter.keys())
            filter_tags = None

        result = {}
        for run, tags in scalars.items():
            if filter_runs is not None and run not in filter_runs:
                continue
            for tag, points in tags.items():
                if filter_tags is not None and tag not in filter_tags:
                    continue
                result.setdefault(run, {})[tag] = points
        return result

    provider.read_scalars.side_effect = read_scalars_side_effect

    # Default: hparams API raises (not implemented by mock)
    provider.list_hyperparameters.side_effect = Exception("not implemented")

    return provider


@pytest.fixture
def sample_scalars():
    return {
        "train": {
            "loss": [
                ScalarPoint(step=0, value=2.5, wall_time=1000.0),
                ScalarPoint(step=100, value=1.0, wall_time=1100.0),
                ScalarPoint(step=200, value=0.5, wall_time=1200.0),
            ],
            "accuracy": [
                ScalarPoint(step=0, value=0.1, wall_time=1000.0),
                ScalarPoint(step=100, value=0.7, wall_time=1100.0),
                ScalarPoint(step=200, value=0.9, wall_time=1200.0),
            ],
        },
        "eval": {
            "loss": [
                ScalarPoint(step=0, value=3.0, wall_time=1000.0),
                ScalarPoint(step=100, value=1.5, wall_time=1100.0),
            ],
        },
    }


@pytest.fixture
def mock_data_provider(sample_scalars):
    return make_mock_data_provider(
        scalars=sample_scalars,
        tensors={"train": {"weights": None}},
        blob_sequences={"train": {"images": None}},
    )


@pytest.fixture
def mock_tb_context(mock_data_provider):
    context = MagicMock()
    context.data_provider = mock_data_provider
    context.flags = MagicMock()
    context.flags.tensorboard_ai_debug = False
    return context


@pytest.fixture
def mock_provider():
    """Mock LLM provider that yields canned responses."""

    class MockLLMProvider:
        def __init__(self, response: str = "This is a test response."):
            self.response = response
            self.last_messages = None
            self.last_system_prompt = None

        async def stream_chat(self, messages, system_prompt, model=None, use_reasoning=False):
            self.last_messages = messages
            self.last_system_prompt = system_prompt
            for word in self.response.split(" "):
                yield word + " "
            yield {"usage": {"input_tokens": 100, "output_tokens": 50}}

    return MockLLMProvider()
