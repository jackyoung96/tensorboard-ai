# TensorBoard AI

A TensorBoard plugin that adds conversational AI analysis to your training dashboards. Analyze metrics, compare runs, and generate custom visualizations — all through natural language.

## Quick Start

```bash
pip install tensorboard-ai
```

Or install from source for developers:

```bash
git clone https://github.com/jackyoung96/tensorboard-ai.git
cd tensorboard-ai
pip install -e .
```

Set your API key:

```bash
export ANTHROPIC_API_KEY="your-key"     # Anthropic (default)
# or
export OPENAI_API_KEY="your-key"        # OpenAI
# or use AWS IAM credentials             # Bedrock
```

Launch TensorBoard:

```bash
tensorboard --logdir /path/to/logs
```

Open the **AI** tab in TensorBoard. Select metrics with checkboxes and hit **Send** to start analysis.

![TensorBoard AI Example](https://raw.githubusercontent.com/jackyoung96/tensorboard-ai/master/assets/tensorboard-ai-example.png)
*(Dummy data example)*

## Features

### AI-Powered Training Analysis

- **Metric selection** — per-chart checkbox to include/exclude metrics from AI context, with All/None toolbar toggle
- **Default analysis prompt** — hit Send with empty input for comprehensive analysis of selected metrics
- Automatic training type detection (SFT, RL/PPO, DPO, Pre-training) with domain-specific insights
- Cross-metric correlation analysis (e.g., grad_norm spike when reward drops)
- Hyperparameter-aware analysis — automatically reads hparams and highlights differences across runs

### Multi-Provider LLM Support

- **Anthropic Claude** (default) — including extended thinking mode
- **OpenAI** (GPT-4o, GPT-5 series)
- **AWS Bedrock** — automatic region prefix for cross-region model IDs
- Runtime provider/model switching via sidebar UI (no restart needed)

### Interactive Chat Panel

- Always-visible sidebar with SSE streaming responses
- Markdown rendering with GFM support (tables, code blocks, lists)
- Clickable metric names in responses — scrolls to and highlights the corresponding chart
- Follow-up conversations with full context retention
- **+ New** button to start fresh sessions; previous sessions saved in history dropdown

### Chart Visualization

- Plotly-based interactive charts with TensorBoard-like styling
- EMA smoothing with adjustable weight slider
- Horizontal axis modes: Step / Relative time / Wall time
- AI-generated custom Plotly visualizations via conversation

### Token Optimization

- Smart data sampling: uniform grid + anomaly detection + curvature-based points (max 50 per series)
- Constant value detection — outputs `constant=X` instead of full data
- Run name alias compression (common prefix/suffix removal, UUID/hash stripping)
- Duplicate metric deduplication: identical data across tags grouped together

## Dashboard Layout

Three-panel layout with resizable panels:

| Settings Sidebar | Chart Grid | AI Analysis Panel |
|---|---|---|
| Chart settings, analysis settings, run selection, tag filter | Interactive Plotly charts with per-metric context checkboxes | Streaming chat with markdown, metric links, session history |

### Settings

- **Chart**: Smoothing slider, horizontal axis mode (Step / Relative / Wall)
- **Analysis**: Language (Auto/EN/KO/JA/ZH/ES/FR/DE), provider, model, extended thinking, debug mode
- **Runs**: Checkbox list with color swatches, regex filter, Select All/None
- **Tag Filter**: Filter visible charts by tag name
- **Metrics**: Per-chart checkbox for AI context inclusion; selected metrics sort to top

## Configuration

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `TENSORBOARD_AI_PROVIDER` | LLM provider (`anthropic`, `openai`, `bedrock`) | `anthropic` |
| `TENSORBOARD_AI_MODEL` | Model name override | Provider default |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `AWS_ACCESS_KEY_ID` | AWS access key (Bedrock) | IAM role |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key (Bedrock) | IAM role |
| `AWS_SESSION_TOKEN` | AWS session token (Bedrock) | — |
| `AWS_REGION` | AWS region (Bedrock) | `us-east-1` |

### CLI Options

```bash
# Standard usage
tensorboard --logdir /path/to/logs

# Enable debug mode (shows raw prompt data in AI panel)
tensorboard --logdir /path/to/logs --tensorboard_ai_debug
```

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Lint & format
ruff check .
ruff format .

# Type checking
mypy tensorboard_ai/

# Build package
python -m build
```

## Architecture

```
tensorboard_ai/
├── plugin.py              # TBPlugin + TBLoader — routes, CLI flags
├── backend/
│   ├── server.py          # SSE streaming chat endpoint
│   ├── data_access.py     # Reads scalars, hyperparameters via DataProvider
│   └── chart_gen.py       # Validates LLM-generated Plotly specs
├── providers/
│   ├── base.py            # Abstract provider protocol + registry
│   ├── anthropic.py       # Claude integration
│   ├── openai.py          # OpenAI integration
│   └── bedrock.py         # AWS Bedrock integration
├── prompts/
│   └── system.py          # System prompts with training type detection
└── static/
    ├── index.js           # Dashboard layout, settings, chart grid
    ├── chat.js            # Chat UI, SSE client, markdown rendering
    └── chart_renderer.js  # Plotly charts, data sampling, run aliases
```

## License

Apache-2.0
