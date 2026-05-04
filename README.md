# My Course Agent

A LangGraph-based multi-agent project built as part of a Platzi course on LangChain and LangGraph. It demonstrates different agent patterns — from simple graph-based flows to RAG (Retrieval-Augmented Generation) agents with structured output.

## Agents

| Name | Graph entry | Description |
|------|-------------|-------------|
| `agent` | `src/agents/main.py:agent` | ReAct agent with a weather tool powered by OpenAI GPT |
| `simple` | `src/agents/simple.py:agent` | Simple stateful LangGraph with custom `State` and a single LLM node |
| `rag` | `src/agents/rag.py:agent` | Two-node graph: an **extractor** node (Claude, structured output) + a **conversation** node (GPT with file-search / vector store) |

## Project Structure

```
src/
├── agents/
│   ├── main.py        # ReAct agent (weather tool)
│   ├── simple.py      # Simple stateful graph
│   └── rag.py         # RAG agent (extractor + conversation)
└── support/
    ├── state.py       # Shared graph State definition
    └── nodes/
        └── conversation/
            ├── node.py    # Conversation node (GPT + file_search tool)
            └── tools.py   # (reserved for custom tools)
notebooks/             # Jupyter notebooks for experimentation
```

## Requirements

- Python ≥ 3.13
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- An [OpenAI](https://platform.openai.com/) API key
- An [Anthropic](https://console.anthropic.com/) API key

## Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd my_course_agent
   ```

2. **Install dependencies**

   ```bash
   uv sync
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

## Running the agents

### LangGraph Dev Server

Start the LangGraph development server (all three graphs are served):

```bash
uv run langgraph dev
```

The Studio UI will be available at `http://localhost:2024`. The graphs defined in [langgraph.json](langgraph.json) will be loaded automatically.

### Run a graph directly (Python)

```python
from src.agents.simple import agent

result = agent.invoke({"messages": [("user", "Hello!")]})
print(result)
```

## Development

```bash
# Lint & format
uv run ruff check .
uv run ruff format .

# Type-check
uv run mypy src/
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key — used for GPT models |
| `ANTHROPIC_API_KEY` | Anthropic API key — used for Claude models |

Copy [.env.example](.env.example) to `.env` and fill in your values. Never commit `.env` to version control.
