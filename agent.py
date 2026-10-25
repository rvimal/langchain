"""Minimal LangChain agent using a local Ollama model."""

from __future__ import annotations

import argparse

from langchain.agents import create_agent
from langchain_ollama import ChatOllama


MODEL = "qwen3:4b"


def get_weather(city: str) -> str:
    """Return a sample weather report for a city (demo data only)."""
    return f"It's sunny and 22°C in {city}. (This is sample weather data.)"


def build_agent():
    """Create the quickstart agent and its local chat model."""
    model = ChatOllama(model=MODEL, temperature=0)
    return create_agent(
        model=model,
        tools=[get_weather],
        system_prompt=(
            "You are a helpful assistant. Use the get_weather tool when asked "
            "about weather, and clearly note that its result is sample data."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local LangChain agent.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="What's the weather in San Francisco?",
        help="What you want to ask the agent.",
    )
    args = parser.parse_args()

    result = build_agent().invoke(
        {"messages": [{"role": "user", "content": args.prompt}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
