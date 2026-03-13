"""Tests for chart generation and validation."""

from __future__ import annotations

from tensorboard_ai.backend.chart_gen import extract_chart_spec, extract_data_requests


def test_extract_chart_spec_valid():
    text = """Here is a chart:
```plotly
{"data": [{"x": [1,2], "y": [3,4], "type": "scatter"}], "layout": {"title": "Test"}}
```
"""
    spec = extract_chart_spec(text)
    assert spec is not None
    assert len(spec["data"]) == 1
    assert spec["layout"]["title"] == "Test"


def test_extract_chart_spec_no_block():
    assert extract_chart_spec("No chart here") is None


def test_extract_chart_spec_invalid_json():
    text = """```plotly
{invalid json}
```"""
    assert extract_chart_spec(text) is None


def test_extract_chart_spec_schema_violation():
    text = """```plotly
{"data": "not an array", "layout": {}}
```"""
    assert extract_chart_spec(text) is None


def test_extract_chart_spec_missing_layout():
    text = """```plotly
{"data": []}
```"""
    assert extract_chart_spec(text) is None


def test_extract_data_requests_valid():
    text = """```data_request
[{"run": "train", "tag": "loss"}, {"run": "eval", "tag": "loss"}]
```"""
    requests = extract_data_requests(text)
    assert len(requests) == 2
    assert requests[0] == {"run": "train", "tag": "loss"}
    assert requests[1] == {"run": "eval", "tag": "loss"}


def test_extract_data_requests_no_block():
    assert extract_data_requests("No data request") == []


def test_extract_data_requests_invalid_json():
    text = """```data_request
{invalid}
```"""
    assert extract_data_requests(text) == []


def test_extract_data_requests_not_a_list():
    text = """```data_request
{"run": "train", "tag": "loss"}
```"""
    assert extract_data_requests(text) == []


def test_extract_data_requests_filters_invalid_entries():
    text = """```data_request
[{"run": "train", "tag": "loss"}, {"invalid": true}, {"run": "x"}]
```"""
    requests = extract_data_requests(text)
    assert len(requests) == 1
    assert requests[0] == {"run": "train", "tag": "loss"}
