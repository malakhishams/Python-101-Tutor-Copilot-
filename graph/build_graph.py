"""
Graph assembly (Story 06 end-to-end wiring).

Flow:
    intake_plan -> react_router --(needs_clarification)--> ask_clarifying_question -> END
                              \--(ok)--> retrieve --(weak retrieval)--> ask_clarifying_question -> END
                                                  \--(ok)--> draft -> reflect -> route_decision
                                                                                     |--(retry)--> retrieve  [loop, bounded]
                                                                                     \--(finalize)--> END

Two separate "weak context" exits feed the same ask_clarifying_question
node: one from react_router (missing debug context, Story 02) and one
from retrieve (weak similarity scores, Story 04). Both set
needs_clarification/clarifying_question on state before reaching it, so
the node itself is trivial -- it just copies clarifying_question into
final_answer.

llm_client / qdrant_client / embedding_client are closed over via
functools.partial when nodes are added, since LangGraph node functions
must be single-argument (state) -> state callables.
"""

from functools import partial

from langgraph.graph import StateGraph, END

from graph.state import TutorState
from graph.nodes.intake_plan import intake_plan_node
from graph.nodes.react_router import react_router_node, route_after_react
from graph.nodes.retrieve import retrieve_node
from graph.nodes.draft import draft_node
from graph.nodes.reflect import reflect_node
from graph.nodes.route_decision import route_decision_node, route_after_reflection


def ask_clarifying_question_node(state: TutorState) -> TutorState:
    """Terminal node for both 'missing context' exits (react_router and
    retrieve). Surfaces the clarifying question as the final_answer so
    run.py has one consistent field to read regardless of which path
    the graph took."""
    state["final_answer"] = state["clarifying_question"]
    return state


def route_after_retrieve(state: TutorState) -> str:
    """Conditional edge after retrieve.py. Weak retrieval sets
    needs_clarification=True (see retrieve.py) -- same signal
    react_router uses, so we can reuse its routing logic name-for-name
    at the call site without importing a second router function."""
    if state["needs_clarification"]:
        return "ask_clarifying_question"
    return "draft"


def build_graph(llm_client, qdrant_client, embedding_client):
    """Constructs and compiles the LangGraph. Returns a runnable graph
    (.invoke(initial_state) -> final_state)."""

    graph = StateGraph(TutorState)

    graph.add_node("intake_plan", partial(intake_plan_node, llm_client=llm_client))
    graph.add_node("react_router", react_router_node)
    graph.add_node(
        "retrieve",
        partial(retrieve_node, qdrant_client=qdrant_client, embedding_client=embedding_client),
    )
    graph.add_node("draft", partial(draft_node, llm_client=llm_client))
    graph.add_node("reflect", partial(reflect_node, llm_client=llm_client))
    graph.add_node("route_decision", route_decision_node)
    graph.add_node("ask_clarifying_question", ask_clarifying_question_node)

    graph.set_entry_point("intake_plan")

    graph.add_edge("intake_plan", "react_router")

    graph.add_conditional_edges(
        "react_router",
        route_after_react,
        {
            "ask_clarifying_question": "ask_clarifying_question",
            "retrieve": "retrieve",
        },
    )

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "ask_clarifying_question": "ask_clarifying_question",
            "draft": "draft",
        },
    )

    graph.add_edge("draft", "reflect")
    graph.add_edge("reflect", "route_decision")

    graph.add_conditional_edges(
        "route_decision",
        route_after_reflection,
        {
            "retrieve": "retrieve",
            "finalize": END,
        },
    )

    graph.add_edge("ask_clarifying_question", END)

    return graph.compile()