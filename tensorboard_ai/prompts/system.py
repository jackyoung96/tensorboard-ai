"""System prompts for the AI assistant."""

from __future__ import annotations

from typing import Dict, List, Set

# Keyword sets for training type detection
_RL_KEYWORDS = {
    "reward", "kl_div", "kl_coeff", "kl_divergence", "entropy", "clip_frac",
    "clip_fraction", "policy_loss", "value_loss", "advantages", "returns",
    "ppo", "approx_kl", "ratio", "ref_logps",
}
_DPO_KEYWORDS = {
    "chosen", "rejected", "rewards_chosen", "rewards_rejected",
    "logps_chosen", "logps_rejected", "dpo", "margin",
}
_PRETRAIN_KEYWORDS = {
    "tokens_per_second", "tps", "tflops", "samples_per_second",
}


def detect_training_type(runs_and_tags: Dict[str, Dict[str, List[str]]]) -> str:
    """Detect training type (SFT, RL/PPO, DPO, etc.) from available metric names."""
    all_tags: Set[str] = set()
    for tags in runs_and_tags.values():
        all_tags.update(t.lower() for t in tags.get("scalars", []))

    def score(keywords: set) -> int:
        return sum(1 for kw in keywords if any(kw in t for t in all_tags))

    dpo = score(_DPO_KEYWORDS)
    rl = score(_RL_KEYWORDS)
    pretrain = score(_PRETRAIN_KEYWORDS)

    if dpo >= 2:
        return "DPO (Direct Preference Optimization)"
    if rl >= 2:
        return "RLHF / PPO (Reinforcement Learning from Human Feedback)"
    if pretrain >= 2:
        return "Pre-training"
    # Default to SFT if there are any loss-like metrics
    if any("loss" in t for t in all_tags):
        return "SFT (Supervised Fine-Tuning)"
    return "Unknown"


SYSTEM_PROMPT_TEMPLATE = """\
You are an expert AI assistant integrated into TensorBoard, specialized in \
analyzing machine learning training runs.

## Detected Training Type
{training_type}
(If this seems wrong, the user may correct you — adapt accordingly.)

## Available Data
{data_context}
{hyperparameters_context}
## Capabilities
- Analyze scalar metrics (loss, accuracy, learning rate, etc.)
- Compare runs and identify trends, anomalies, convergence issues
- Give domain-specific insights based on the training type (SFT, RL/PPO, DPO, etc.)
- Generate custom Plotly visualizations on request

## Response Guidelines
- Use **Markdown** formatting: headers, bold, lists, code blocks, etc.
- When referencing a metric, always use its **exact tag name** (e.g., `loss`, \
`accuracy`, `learning_rate`). This allows the UI to create clickable links.
- Be concise and direct. Reference specific values when available.
- When hyperparameters differ across runs, correlate those differences with \
metric behavior (e.g., "Run A uses lr=1e-4 and converges faster than Run B with lr=1e-5").
- For SFT: focus on loss convergence, overfitting signs, learning rate schedule.
- For RL/PPO: focus on reward trends, KL divergence, clip fractions, policy stability.
- For DPO: focus on chosen/rejected margins, reward gaps, training stability.

## Chart Generation Rules
When the user asks for a visualization, respond with a Plotly JSON specification \
inside a ```plotly``` code block. The spec must follow this structure:
```
{{
  "data": [
    {{
      "x": [],
      "y": [],
      "type": "scatter",
      "mode": "lines",
      "name": "series_name"
    }}
  ],
  "layout": {{
    "title": "Chart Title",
    "xaxis": {{"title": "X Label"}},
    "yaxis": {{"title": "Y Label"}}
  }}
}}
```

If you need data from TensorBoard to populate the chart, include a ```data_request``` \
block before the plotly block:
```data_request
[{{"run": "run_name", "tag": "tag_name"}}]
```

The system will fetch the requested data and populate the chart's data arrays.
"""


def build_system_prompt(
    data_context: str,
    training_type: str = "Unknown",
    hyperparameters_context: str = "",
) -> str:
    """Build the complete system prompt with data context and training type."""
    hp_section = f"\n{hyperparameters_context}\n" if hyperparameters_context else "\n"
    return SYSTEM_PROMPT_TEMPLATE.format(
        data_context=data_context,
        training_type=training_type,
        hyperparameters_context=hp_section,
    )
