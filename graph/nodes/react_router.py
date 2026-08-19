"""
ReAct-style router node (User Story 02).

Decides the next best step instead of guessing: does the Copilot have
enough context to proceed (retrieve + draft), or does it need to ask
one targeted follow-up question first?

Rule-based by design (not LLM-based): the case called out in the brief
-- a debug question with no code/error detail -- is a clean, cheap check
that doesn't benefit from LLM reasoning. This intentionally does NOT
catch fuzzier cases (e.g. a vague concept question with no code attached)
-- documented as a known limitation rather than solved with an extra
LLM call, to keep this node fast and free.
"""

from graph.state import TutorState


def _looks_like_missing_debug_context(state: TutorState) -> bool:
    """Debug intent with no code/error text pasted at all."""
    intent = state["intent"]
    code = state.get("student_code")

    if intent != "debug":
        return False

    if code is None or code.strip() == "":
        return True

    return False


def react_router_node(state: TutorState) -> TutorState:
    """LangGraph node function. Reads intent/student_code, writes
    needs_clarification + clarifying_question back into state."""

    if _looks_like_missing_debug_context(state):
        state["needs_clarification"] = True
        state["clarifying_question"] = (
            "Could you paste the code that's producing this error? "
            "That'll help me point you to the right explanation."
        )
    else:
        state["needs_clarification"] = False
        state["clarifying_question"] = None

    return state


def route_after_react(state: TutorState) -> str:
    """Conditional edge function for build_graph.py. Returns the name
    of the next node based on needs_clarification."""
    if state["needs_clarification"]:
        return "ask_clarifying_question"
    return "retrieve"