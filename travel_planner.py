"""
AI Travel Planner: LangChain + LangGraph + local Qwen via Ollama.

SETUP
-----
1. Install Ollama: https://ollama.com/download
2. Pull a Qwen model and make sure Ollama is running:
       ollama pull qwen2.5:7b
       ollama serve        # skip if Ollama already runs as a service
3. Install Python packages:
       pip install langchain-ollama langgraph

RUN
---
    python travel_planner.py

EXAMPLE
-------
    $ python travel_planner.py
    Describe your trip: I want to go to Tokyo for 5 days with a budget of 2000 USD
    --- Understand Travel Request ---
    --- Validate Details ---
    --- Generate Itinerary ---
    --- Check Budget ---
    Final itinerary: Day 1: ... Estimated total cost: 1800 USD

    If you leave out details (e.g. "I want to go somewhere nice"), the agent
    asks a clarification question and waits for your next answer.
"""

import json
import re
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# Local Qwen model served by Ollama. temperature=0 keeps answers stable.
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

MAX_ATTEMPTS = 3  # Limit itinerary regeneration so the loop always ends.


# Shared state: every node reads from and writes to this dictionary.
class TravelState(TypedDict):
    user_request: str   # Raw text from the user
    destination: str    # Extracted by node 1
    days: int           # Extracted by node 1
    budget: float       # Extracted by node 1 (USD)
    missing: list       # Names of missing fields, set by node 2
    itinerary: str      # Written by node 3
    estimated_cost: float  # Written by node 3
    attempts: int       # How many itineraries were generated
    message: str        # Clarification question or final result


def extract_json(text: str) -> dict:
    """Pull the first {...} block out of the model's reply and parse it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


def to_number(value) -> float:
    """Convert a model-provided value to a number, or 0 if not possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# Node 1: ask the LLM to turn free text into structured fields.
def understand_request(state: TravelState) -> dict:
    print("--- Understand Travel Request ---")
    prompt = (
        "Extract trip details from the request below. Reply with ONLY JSON like "
        '{"destination": "", "days": 0, "budget": 0}. Use "" or 0 for anything '
        "not mentioned. Budget is a number in USD.\n\n"
        f"Request: {state['user_request']}"
    )
    data = extract_json(llm.invoke(prompt).content)
    return {
        "destination": str(data.get("destination") or "").strip(),
        "days": int(to_number(data.get("days"))),
        "budget": to_number(data.get("budget")),
        "attempts": 0,
    }


# Node 2: plain Python check that all required fields are present.
def validate_details(state: TravelState) -> dict:
    print("--- Validate Details ---")
    missing = []
    if not state["destination"]:
        missing.append("destination")
    if state["days"] <= 0:
        missing.append("number of days")
    if state["budget"] <= 0:
        missing.append("budget")
    message = ""
    if missing:
        message = "Please tell me your " + ", ".join(missing) + "."
    return {"missing": missing, "message": message}


# Node 3: generate an itinerary; if this is a retry, ask for a cheaper one.
def generate_itinerary(state: TravelState) -> dict:
    print("--- Generate Itinerary ---")
    prompt = (
        f"Plan a {state['days']}-day trip to {state['destination']} "
        f"with a total budget of {state['budget']} USD.\n"
    )
    if state["attempts"] > 0:
        prompt += (
            f"The previous plan cost about {state['estimated_cost']} USD, which is "
            "too expensive. Make it clearly cheaper and stay within budget.\n"
        )
    prompt += (
        "Give a short day-by-day plan. On the LAST line write exactly: "
        "TOTAL_COST: <number in USD>"
    )
    text = llm.invoke(prompt).content

    # Read the cost from the last line, then remove it from the displayed plan.
    match = re.search(r"TOTAL_COST:\s*\$?([\d,\.]+)", text)
    cost = to_number(match.group(1).replace(",", "").rstrip(".")) if match else 0.0
    itinerary = re.sub(r"TOTAL_COST:.*", "", text).strip()
    return {
        "itinerary": itinerary,
        "estimated_cost": cost,
        "attempts": state["attempts"] + 1,
    }


# Node 4: compare the estimated cost with the budget and build the final message.
def check_budget(state: TravelState) -> dict:
    print("--- Check Budget ---")
    over = state["estimated_cost"] > state["budget"]
    note = ""
    if over:
        note = f"\n(Note: estimate {state['estimated_cost']} USD still exceeds budget.)"
    return {
        "message": f"{state['itinerary']}\n\nEstimated total cost: "
        f"{state['estimated_cost']} USD (budget: {state['budget']} USD){note}"
    }


# Conditional edge after Validate Details:
# missing info -> END (the message holds a clarification question)
# otherwise    -> generate the itinerary
def route_after_validation(state: TravelState) -> str:
    return "ask_user" if state["missing"] else "generate"


# Conditional edge after Check Budget:
# over budget (and retries left) -> regenerate a cheaper itinerary
# otherwise                      -> finish
def route_after_budget(state: TravelState) -> str:
    if state["estimated_cost"] > state["budget"] and state["attempts"] < MAX_ATTEMPTS:
        return "regenerate"
    return "done"


# Build the graph: register nodes, then wire up normal and conditional edges.
graph = StateGraph(TravelState)
graph.add_node("understand_request", understand_request)
graph.add_node("validate_details", validate_details)
graph.add_node("generate_itinerary", generate_itinerary)
graph.add_node("check_budget", check_budget)

graph.add_edge(START, "understand_request")
graph.add_edge("understand_request", "validate_details")
graph.add_conditional_edges(
    "validate_details",
    route_after_validation,
    {"ask_user": END, "generate": "generate_itinerary"},
)
graph.add_edge("generate_itinerary", "check_budget")
graph.add_conditional_edges(
    "check_budget",
    route_after_budget,
    {"regenerate": "generate_itinerary", "done": END},
)
app = graph.compile()


if __name__ == "__main__":
    request = input("Describe your trip: ")
    while True:
        result = app.invoke({"user_request": request})
        if result["missing"]:
            # Ask for clarification and combine the answer with the earlier request.
            answer = input(result["message"] + "\n> ")
            request = f"{request}. {answer}"
        else:
            print("\nFinal itinerary:\n" + result["message"])
            break
