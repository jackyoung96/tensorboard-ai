"""Tests for the data access layer."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from tensorboard_ai.backend.data_access import TBDataAccess
from tests.conftest import make_mock_data_provider


@pytest.fixture
def data_access(mock_data_provider):
    return TBDataAccess(data_provider=mock_data_provider)


def test_list_runs_and_tags(data_access):
    result = data_access.list_runs_and_tags()
    assert "train" in result
    assert "eval" in result
    assert sorted(result["train"]["scalars"]) == ["accuracy", "loss"]
    assert result["train"]["tensors"] == ["weights"]
    assert result["train"]["blob_sequences"] == ["images"]
    assert result["eval"]["scalars"] == ["loss"]


def test_read_scalar_summary(data_access):
    summary = data_access.read_scalar_summary("train", "loss")
    assert summary["run"] == "train"
    assert summary["tag"] == "loss"
    assert summary["count"] == 3
    assert summary["min"] == 0.5
    assert summary["max"] == 2.5
    assert summary["first"] == 2.5
    assert summary["last"] == 0.5
    assert summary["step_range"] == [0, 200]
    assert len(summary["sampled_points"]) == 3


def test_read_scalar_summary_empty():
    provider = make_mock_data_provider(scalars={})
    da = TBDataAccess(data_provider=provider)
    summary = da.read_scalar_summary("nonexistent", "loss")
    assert summary["count"] == 0


def test_read_scalar_summary_downsample(data_access, mock_data_provider):
    data_access.read_scalar_summary("train", "loss", downsample=500)
    call_kwargs = mock_data_provider.read_scalars.call_args[1]
    assert call_kwargs["downsample"] == 500


def test_read_scalars_for_tags(data_access):
    result = data_access.read_scalars_for_tags([("train", "loss"), ("eval", "loss")])
    assert "train" in result
    assert "eval" in result
    assert len(result["train"]["loss"]) == 3
    assert result["train"]["loss"][0]["step"] == 0
    assert result["train"]["loss"][0]["value"] == 2.5
    assert result["train"]["loss"][0]["wall_time"] == 1000.0


def test_get_data_context(data_access):
    context = data_access.get_data_context()
    assert "Available training data:" in context
    assert "Run: train" in context
    assert "loss" in context
    assert "accuracy" in context
    assert "weights" in context
    assert "images" in context


def test_get_data_context_empty():
    provider = make_mock_data_provider()
    da = TBDataAccess(data_provider=provider)
    context = da.get_data_context()
    assert context == "No training data available."


# --- Hyperparameter tests ---


@dataclass(frozen=True)
class FakeHyperparameterValue:
    hyperparameter_name: str
    value: object = None


@dataclass(frozen=True)
class FakeSessionRun:
    experiment_id: str = ""
    run: str = ""


@dataclass(frozen=True)
class FakeSessionGroup:
    root: FakeSessionRun = None
    sessions: list = None
    hyperparameter_values: list = None

    def __post_init__(self):
        if self.root is None:
            object.__setattr__(self, "root", FakeSessionRun())
        if self.sessions is None:
            object.__setattr__(self, "sessions", [])
        if self.hyperparameter_values is None:
            object.__setattr__(self, "hyperparameter_values", [])


@dataclass(frozen=True)
class FakeListHPResult:
    hyperparameters: list = None
    session_groups: list = None

    def __post_init__(self):
        if self.hyperparameters is None:
            object.__setattr__(self, "hyperparameters", [])
        if self.session_groups is None:
            object.__setattr__(self, "session_groups", [])


def _make_hp_provider(session_groups):
    """Create a mock provider with hparams data."""
    provider = make_mock_data_provider()
    provider.list_hyperparameters.side_effect = None
    provider.list_hyperparameters.return_value = FakeListHPResult(
        session_groups=session_groups,
    )
    return provider


class TestReadHyperparameters:
    def test_hparams_api(self):
        groups = [
            FakeSessionGroup(
                root=FakeSessionRun(run="run1"),
                sessions=[FakeSessionRun(run="run1")],
                hyperparameter_values=[
                    FakeHyperparameterValue("lr", 0.001),
                    FakeHyperparameterValue("batch_size", 32),
                ],
            ),
            FakeSessionGroup(
                root=FakeSessionRun(run="run2"),
                sessions=[FakeSessionRun(run="run2")],
                hyperparameter_values=[
                    FakeHyperparameterValue("lr", 0.01),
                    FakeHyperparameterValue("batch_size", 64),
                ],
            ),
        ]
        da = TBDataAccess(data_provider=_make_hp_provider(groups))
        result = da.read_hyperparameters()
        assert result["run1"]["lr"] == 0.001
        assert result["run2"]["batch_size"] == 64

    def test_hparams_api_empty_falls_through(self):
        """Empty hparams result → tries text plugin fallback."""
        provider = make_mock_data_provider()
        provider.list_hyperparameters.side_effect = None
        provider.list_hyperparameters.return_value = FakeListHPResult()
        # Text plugin also empty
        provider.list_tensors.return_value = {}
        da = TBDataAccess(data_provider=provider)
        assert da.read_hyperparameters() == {}

    def test_hparams_api_exception_falls_through(self):
        """Hparams exception → tries text fallback."""
        provider = make_mock_data_provider()
        # list_hyperparameters already raises by default from conftest
        provider.list_tensors.return_value = {}
        da = TBDataAccess(data_provider=provider)
        assert da.read_hyperparameters() == {}


class TestGetHyperparametersContext:
    def test_differing_hparams(self):
        groups = [
            FakeSessionGroup(
                root=FakeSessionRun(run="run1"),
                sessions=[FakeSessionRun(run="run1")],
                hyperparameter_values=[
                    FakeHyperparameterValue("lr", 0.001),
                    FakeHyperparameterValue("epochs", 10),
                ],
            ),
            FakeSessionGroup(
                root=FakeSessionRun(run="run2"),
                sessions=[FakeSessionRun(run="run2")],
                hyperparameter_values=[
                    FakeHyperparameterValue("lr", 0.01),
                    FakeHyperparameterValue("epochs", 10),
                ],
            ),
        ]
        da = TBDataAccess(data_provider=_make_hp_provider(groups))
        ctx = da.get_hyperparameters_context()
        assert "Differing across runs" in ctx
        assert "lr" in ctx
        assert "run1=0.001" in ctx
        assert "run2=0.01" in ctx
        assert "Common across all runs" in ctx
        assert "epochs: 10" in ctx

    def test_all_same_hparams(self):
        groups = [
            FakeSessionGroup(
                root=FakeSessionRun(run="run1"),
                sessions=[FakeSessionRun(run="run1")],
                hyperparameter_values=[
                    FakeHyperparameterValue("lr", 0.001),
                ],
            ),
            FakeSessionGroup(
                root=FakeSessionRun(run="run2"),
                sessions=[FakeSessionRun(run="run2")],
                hyperparameter_values=[
                    FakeHyperparameterValue("lr", 0.001),
                ],
            ),
        ]
        da = TBDataAccess(data_provider=_make_hp_provider(groups))
        ctx = da.get_hyperparameters_context()
        assert "Differing" not in ctx
        assert "Common across all runs" in ctx
        assert "lr: 0.001" in ctx

    def test_single_run_shows_all(self):
        groups = [
            FakeSessionGroup(
                root=FakeSessionRun(run="run1"),
                sessions=[FakeSessionRun(run="run1")],
                hyperparameter_values=[
                    FakeHyperparameterValue("lr", 0.001),
                    FakeHyperparameterValue("batch_size", 32),
                ],
            ),
        ]
        da = TBDataAccess(data_provider=_make_hp_provider(groups))
        ctx = da.get_hyperparameters_context()
        assert "Hyperparameters:" in ctx
        assert "lr: 0.001" in ctx
        assert "batch_size: 32" in ctx
        assert "Differing" not in ctx

    def test_no_hparams_empty_string(self):
        provider = make_mock_data_provider()
        provider.list_tensors.return_value = {}
        da = TBDataAccess(data_provider=provider)
        assert da.get_hyperparameters_context() == ""
