"""
Route decision node (Story 06, "Routing" step).

The graph's central control point after reflection: either finalize
(reflection passed, or we've exhausted retries) or loop back for one
retry with the fix directive attached.

Bounded to MAX_RETRIES so a stubborn "fail" verdict can't loop forever --
per the task brief: "repeats retrieve + draft once" (i.e. one retry,
not unlimited).

This node owns retry_count incrementing (not reflect.py) since
incrementing is a routing decision -- "are we going to spend another
loop on this?" -- not a judgment about answer quality.
"""

from graph.state import TutorState

MAX_RETRIES = 1


def route_decision_node(state: TutorState) -> TutorState:
    """LangGraph node function. Reads reflection_verdict, writes
    final_answer when finalizing. Retry bookkeeping (retry_count) is
    incremented here so route_after_reflection can make its decision
    off a state that's already up to date."""

    if state["reflection_verdict"] == "fail" and state["retry_count"] < MAX_RETRIES:
        state["retry_count"] += 1
        # Leave final_answer unset -- route_after_reflection will send
        # this back to retrieve.py for another pass.
        return state

    # Either passed, or retries are exhausted -- finalize with whatever
    # the last draft was. On exhausted retries, this may still carry
    # reflection feedback the student never sees; that's acceptable
    # (better to answer imperfectly than block the student entirely),
    # but it's worth surfacing in Langfuse traces for the QA lead.
    state["final_answer"] = state["draft_answer"]
    return state


def route_after_reflection(state: TutorState) -> str:
    """Conditional edge function for build_graph.py. Returns the name
    of the next node based on whether route_decision_node decided to
    retry or finalize."""
    if state["final_answer"]:
        return "finalize"
    return "retrieve"