"""
Intake -> Plan node (User Story 01).

Responsibilities:
1. Classify the student's intent: "concept" | "debug" | "quiz_prep"
2. Attach a step-by-step plan for that intent (shown to the student so
   they know the Copilot will work through it methodically)

Deliberately does NOT decide whether clarification is needed -- that's
react_router.py's job (Story 02), kept separate so each node has one
clear responsibility.
"""

from graph.state import TutorState

# One LLM call is used here (intent classification). The plan itself is a
# static template per intent -- the plan is a "show your work" UX feature,
# not something downstream nodes parse, so an extra LLM call to generate
# it isn't worth the cost/latency/variability.

PLAN_TEMPLATES = {
    "concept": [
        "Check if the question has enough context to answer directly",
        "Retrieve the most relevant textbook section",
        "Explain the concept at Level 1 (beginner vocabulary)",
        "Provide a tiny example",
        "Reflect on accuracy and tone, refine if needed",
    ],
    "debug": [
        "Check if both the error and the relevant code were provided",
        "Retrieve the textbook section covering this kind of error",
        "Explain what's causing the error at Level 1",
        "Show a tiny corrected example",
        "Reflect on accuracy and tone, refine if needed",
    ],
    "quiz_prep": [
        "Check if the topic to practice is clear",
        "Retrieve the relevant textbook section for reference",
        "Summarize the key points at Level 1",
        "Offer a tiny practice example",
        "Reflect on accuracy and tone, refine if needed",
    ],
}

INTENT_CLASSIFICATION_PROMPT = """You are classifying a beginner Python student's question into exactly one category.

Categories:
- "concept": student wants a concept explained (e.g. "what's the difference between a list and tuple")
- "debug": student has an error or broken code they want fixed (e.g. "why is my indentation failing")
- "quiz_prep": student wants practice questions or a quiz on a topic

Student question: {question}
Student code/error (if any): {code}

Respond with ONLY one word: concept, debug, or quiz_prep. No punctuation, no explanation."""


def classify_intent(llm_client, question: str, code: str | None) -> str:
    """Calls the LLM to classify intent. Falls back to 'concept' on any
    unexpected output so the graph never gets stuck on a bad classification."""
    prompt = INTENT_CLASSIFICATION_PROMPT.format(
        question=question, code=code or "(none provided)"
    )
    response = llm_client.complete(prompt).strip().lower()

    if response in PLAN_TEMPLATES:
        return response
    return "concept"  # safe default


def intake_plan_node(state: TutorState, llm_client) -> TutorState:
    """LangGraph node function. Reads student_question/student_code,
    writes intent + plan back into state."""
    intent = classify_intent(llm_client, state["student_question"], state.get("student_code"))

    state["intent"] = intent
    state["plan"] = PLAN_TEMPLATES[intent]

    return state