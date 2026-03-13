"""Chart generation and validation utilities."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import jsonschema

PLOTLY_SCHEMA = {
    "type": "object",
    "required": ["data", "layout"],
    "properties": {
        "data": {
            "type": "array",
            "items": {"type": "object"},
        },
        "layout": {
            "type": "object",
        },
    },
}


def extract_chart_spec(text: str) -> Optional[Dict[str, Any]]:
    """Extract and validate a Plotly JSON spec from ```plotly``` blocks."""
    pattern = r"```plotly\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    raw = match.group(1).strip()
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError:
        return None

    try:
        jsonschema.validate(instance=spec, schema=PLOTLY_SCHEMA)
    except jsonschema.ValidationError:
        return None

    return dict(spec)


def extract_data_requests(text: str) -> List[Dict[str, str]]:
    """Extract data request blocks: ```data_request``` → [{run, tag}]."""
    pattern = r"```data_request\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return []

    raw = match.group(1).strip()
    try:
        requests = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(requests, list):
        return []

    valid = []
    for req in requests:
        if isinstance(req, dict) and "run" in req and "tag" in req:
            valid.append({"run": req["run"], "tag": req["tag"]})
    return valid
