# LangChain + Ollama Quickstart

A minimal tool-using LangChain agent backed by the local `qwen3:4b` model. No cloud API key is needed.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download) installed and running

The configured local machine already has `qwen3:4b` available. If setting this up elsewhere, pull it with `ollama pull qwen3:4b`.

## Install and run

From this directory, install the project dependencies and run the sample prompt:

```powershell
uv sync
uv run python agent.py
```

Ask a different question by passing it as an argument:

```powershell
uv run python agent.py "What's the weather in Tokyo?"
```

You can also run `python agent.py` after installing the dependencies in your active environment. The `get_weather` tool returns clearly labeled sample data; it is not connected to a live weather service.