"""A tiny LangGraph travel agent backed by a local Ollama model."""

from typing import Literal, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


class TravelState(TypedDict, total=False):
    request: str
    route: Literal["plan", "advice"]
    response: str


# The model runs locally through Ollama. Pull this model once with:
# ollama pull qwen3:4b
llm = ChatOllama(model="qwen3:4b", temperature=0.2)


# Keep the first routing decision deliberately simple for learning purposes.
PLANNING_CUES = (
    "itinerary",
    "plan a trip",
    "plan my trip",
    "travel plan",
    "trip to",
    "vacation",
    "holiday",
    "road trip",
    "days in",
)


def classify_request(state: TravelState) -> dict[str, str]:
    """Choose which travel workflow should handle the request."""
    text = state["request"].lower()
    route = "plan" if any(cue in text for cue in PLANNING_CUES) else "advice"
    return {"route": route}


def choose_route(state: TravelState) -> str:
    """Return the branch name used by the conditional graph edge."""
    return state["route"]


def plan_trip(state: TravelState) -> dict[str, str]:
    prompt = (
        "You are a friendly travel planner. Create a practical, beginner-friendly "
        "trip outline based on the request. Use short sections for a suggested "
        "itinerary, transport, and useful tips. If important details such as dates "
        "or budget are missing, state reasonable assumptions and ask one follow-up "
        "question at the end. Do not invent current prices or availability.\n\n"
        f"Traveler's request: {state['request']}"
    )
    result = llm.invoke(prompt)
    return {"response": str(result.content)}


def travel_advice(state: TravelState) -> dict[str, str]:
    prompt = (
        "You are a helpful travel guide. Answer the travel question clearly and "
        "concisely. Mention when the traveler should verify changing rules or "
        "details with an official source.\n\n"
        f"Traveler's question: {state['request']}"
    )
    result = llm.invoke(prompt)
    return {"response": str(result.content)}


graph_builder = StateGraph(TravelState)
graph_builder.add_node("classify_request", classify_request)
graph_builder.add_node("plan_trip", plan_trip)
graph_builder.add_node("travel_advice", travel_advice)

graph_builder.add_edge(START, "classify_request")
graph_builder.add_conditional_edges(
    "classify_request",
    choose_route,
    {"plan": "plan_trip", "advice": "travel_advice"},
)
graph_builder.add_edge("plan_trip", END)
graph_builder.add_edge("travel_advice", END)

travel_agent = graph_builder.compile()


def main() -> None:
    request = input("What would you like help with for your travels? ").strip()
    if not request:
        print("Please enter a travel question or trip-planning request.")
        return

    result = travel_agent.invoke({"request": request})
    print(f"\n[{result['route']} workflow]\n")
    print(result["response"])


if __name__ == "__main__":
    main()
