# Ollama + LangGraph Travel Agent

A small learning example of a LangGraph agent that uses the local `qwen3:4b` model. A simple keyword-based condition sends each request to either a trip-planning node or a general travel-advice node.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

## Run

From this directory, install the Python dependencies and download the model:

```bash
pip install -r requirements.txt
ollama pull qwen3:4b
python agent.py
```

Enter a request when prompted. For example:

- `Plan a 3-day trip to Kyoto`
- `What should I pack for Iceland?`

The first example follows the `plan_trip` branch; the second follows `travel_advice`. The routing cues are intentionally kept in Python so the conditional edge is easy to see. The LLM generates the response in either branch.
