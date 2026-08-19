"""
ReAct-style router node (User Story 02 + Story 03 routing).

Decides the next best step instead of guessing:

1. If a debugging question is missing code/error context,
   ask one targeted clarifying question.
2. If the student requests a supported classroom action,
   route safely to the tool handler.
3. Otherwise, proceed to textbook retrieval.

The router is rule-based by design to keep this decision fast,
predictable, and free of unnecessary LLM calls.
"""

from graph.state import TutorState


def _looks_like_missing_debug_context(state: TutorState) -> bool:
    """Return True when this is a debugging question with no code or
    error text provided."""

    intent = state["intent"]
    code = state.get("student_code")

    if intent != "debug":
        return False

    if code is None or code.strip() == "":
        return True

    return False


def _detect_tool_request(state: TutorState) -> tuple[str | None, dict]:
    """
    Detect supported classroom tool requests from the student's question.

    Returns:
        (tool_name, tool_arguments)

    Returns:
        (None, {}) when no supported tool is detected.
    """

    question = state["student_question"].lower()

    # ---------------------------------------------------------
    # create_practice_quiz
    # ---------------------------------------------------------
    if any(word in question for word in ["quiz", "practice quiz", "test me"]):
        return (
            "create_practice_quiz",
            {
                "topic": _extract_topic(question),
                "num_questions": _extract_number(question, default=3),
                "difficulty": "beginner",
            },
        )

    # ---------------------------------------------------------
    # recommend_exercises
    # ---------------------------------------------------------
    if any(
        phrase in question
        for phrase in [
            "recommend exercises",
            "recommend exercise",
            "give me exercises",
            "practice exercises",
        ]
    ):
        return (
            "recommend_exercises",
            {
                "topic": _extract_topic(question),
                "num_exercises": _extract_number(question, default=3),
            },
        )

    # ---------------------------------------------------------
    # escalate_to_ta
    # ---------------------------------------------------------
    if any(
        phrase in question
        for phrase in [
            "escalate to ta",
            "ask a ta",
            "contact a ta",
            "need a ta",
        ]
    ):
        return (
            "escalate_to_ta",
            {
                "reason": "Student requested TA assistance",
                "question": state["student_question"],
            },
        )

    # ---------------------------------------------------------
    # log_student_question
    #
    # This is intentionally not triggered automatically for every
    # question. It only runs when explicitly requested.
    # ---------------------------------------------------------
    if any(
        phrase in question
        for phrase in [
            "log this question",
            "save this question",
            "record this question",
        ]
    ):
        return (
            "log_student_question",
            {
                "topic": _extract_topic(question),
                "question": state["student_question"],
            },
        )

    return None, {}


def _extract_number(question: str, default: int = 3) -> int:
    """
    Extract a small number from the question.

    Example:
        'Create a 5 question quiz about lists'
        -> 5
    """

    for number in range(1, 11):
        if str(number) in question:
            return number

    return default


def _extract_topic(question: str) -> str:
    """
    Simple topic extraction for supported Python 101 topics.

    This can be replaced later with structured LLM/function-calling
    extraction if needed.
    """

    known_topics = [
        "lists",
        "tuples",
        "loops",
        "for loops",
        "while loops",
        "functions",
        "variables",
        "strings",
        "dictionaries",
        "sets",
        "conditionals",
        "if statements",
        "errors",
        "debugging",
        "indentation",
    ]

    for topic in known_topics:
        if topic in question:
            return topic

    return "general Python"


def react_router_node(state: TutorState) -> TutorState:
    """
    LangGraph node function.

    Decides whether to:
    - ask for missing debugging context,
    - execute a supported tool,
    - or continue to textbook retrieval.
    """

    # Reset tool fields so a reused state cannot accidentally carry
    # a previous tool request.
    state["requested_tool"] = None
    state["tool_arguments"] = {}

    # ---------------------------------------------------------
    # Priority 1: Missing debug context
    # ---------------------------------------------------------
    if _looks_like_missing_debug_context(state):
        state["needs_clarification"] = True

        state["clarifying_question"] = (
            "Could you paste the code that's producing this error? "
            "That'll help me point you to the right explanation."
        )

        return state

    # ---------------------------------------------------------
    # Priority 2: Tool request
    # ---------------------------------------------------------
    tool_name, tool_arguments = _detect_tool_request(state)

    if tool_name is not None:
        state["requested_tool"] = tool_name
        state["tool_arguments"] = tool_arguments

        state["needs_clarification"] = False
        state["clarifying_question"] = None

        return state

    # ---------------------------------------------------------
    # Priority 3: Normal RAG flow
    # ---------------------------------------------------------
    state["needs_clarification"] = False
    state["clarifying_question"] = None

    return state


def route_after_react(state: TutorState) -> str:
    """
    Conditional edge function for build_graph.py.

    Priority:
    1. Clarification
    2. Tool handler
    3. Retrieval
    """

    if state["needs_clarification"]:
        return "ask_clarifying_question"

    if state.get("requested_tool"):
        return "tool_handler"

    return "retrieve"