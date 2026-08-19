"""
Graph assembly (Stories 01–06 end-to-end wiring).

Flow:
    intake_plan -> react_router

    react_router:
        ├── missing context --> ask_clarifying_question --> END
        ├── tool request ----> tool_handler ------------> END
        └── normal question -> retrieve

    retrieve:
        ├── weak retrieval --> ask_clarifying_question --> END
        └── good retrieval -> draft -> reflect -> route_decision
                                               |
                                               ├── retry --> retrieve
                                               └── finalize -> END

Tool requests are routed through tool_handler.py, where arguments are
validated against Pydantic schemas before the requested tool executes.
Every tool call is logged in state["tool_calls"].
"""

from functools import partial

from langgraph.graph import StateGraph, END

from graph.state import TutorState
from graph.nodes.intake_plan import intake_plan_node
from graph.nodes.react_router import react_router_node, route_after_react
from graph.nodes.retrieve import retrieve_node
from graph.nodes.draft import draft_node
from graph.nodes.reflect import reflect_node
from graph.nodes.route_decision import (
    route_decision_node,
    route_after_reflection,
)
from graph.nodes.tool_handler import tool_handler_node


def ask_clarifying_question_node(state: TutorState) -> TutorState:
    """
    Terminal node for clarification paths.

    Both the ReAct router and retrieval can determine that more
    information is needed. This node provides one consistent final
    answer field for run.py.
    """

    state["final_answer"] = (
        state["clarifying_question"]
        or "Could you provide a little more detail about your question?"
    )

    return state


def route_after_retrieve(state: TutorState) -> str:
    """
    Conditional edge after retrieve.py.

    Weak retrieval sets needs_clarification=True.
    Successful retrieval continues to drafting.
    """

    if state["needs_clarification"]:
        return "ask_clarifying_question"

    return "draft"


def build_graph(llm_client, qdrant_client, embedding_client):
    """
    Construct and compile the LangGraph.

    Clients are injected using functools.partial because LangGraph
    nodes receive state as their primary argument.
    """

    graph = StateGraph(TutorState)

    # ---------------------------------------------------------
    # Add nodes
    # ---------------------------------------------------------

    graph.add_node(
        "intake_plan",
        partial(intake_plan_node, llm_client=llm_client),
    )

    graph.add_node(
        "react_router",
        react_router_node,
    )

    graph.add_node(
        "tool_handler",
        tool_handler_node,
    )

    graph.add_node(
        "retrieve",
        partial(
            retrieve_node,
            qdrant_client=qdrant_client,
            embedding_client=embedding_client,
        ),
    )

    graph.add_node(
        "draft",
        partial(draft_node, llm_client=llm_client),
    )

    graph.add_node(
        "reflect",
        partial(reflect_node, llm_client=llm_client),
    )

    graph.add_node(
        "route_decision",
        route_decision_node,
    )

    graph.add_node(
        "ask_clarifying_question",
        ask_clarifying_question_node,
    )

    # ---------------------------------------------------------
    # Entry point
    # ---------------------------------------------------------

    graph.set_entry_point("intake_plan")

    graph.add_edge(
        "intake_plan",
        "react_router",
    )

    # ---------------------------------------------------------
    # ReAct routing
    #
    # clarification -> ask question
    # tool request  -> safe tool execution
    # normal        -> retrieval
    # ---------------------------------------------------------

    graph.add_conditional_edges(
        "react_router",
        route_after_react,
        {
            "ask_clarifying_question": "ask_clarifying_question",
            "tool_handler": "tool_handler",
            "retrieve": "retrieve",
        },
    )

    # Tool actions are terminal for now.
    graph.add_edge(
        "tool_handler",
        END,
    )

    # ---------------------------------------------------------
    # Retrieval routing
    # ---------------------------------------------------------

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "ask_clarifying_question": "ask_clarifying_question",
            "draft": "draft",
        },
    )

    # ---------------------------------------------------------
    # Agentic RAG flow
    # ---------------------------------------------------------

    graph.add_edge(
        "draft",
        "reflect",
    )

    graph.add_edge(
        "reflect",
        "route_decision",
    )

    graph.add_conditional_edges(
        "route_decision",
        route_after_reflection,
        {
            "retrieve": "retrieve",
            "finalize": END,
        },
    )

    # ---------------------------------------------------------
    # Clarification is terminal
    # ---------------------------------------------------------

    graph.add_edge(
        "ask_clarifying_question",
        END,
    )

    return graph.compile()