"""
Shared state for the Python 101 Tutor Copilot LangGraph.

Every node reads from and writes to this single TypedDict as it flows
through the graph: intake -> plan -> route -> retrieve -> draft -> reflect
-> (loop back once, or) finalize.
"""

from typing import TypedDict, Optional, List, Dict, Any


class RetrievedChunk(TypedDict):
    """One chunk returned from Qdrant, with citation metadata."""
    text: str
    page: int
    chapter: str
    chunk_id: str
    score: float


class ToolCall(TypedDict):
    """A single logged tool invocation (Story 03)."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any]
    timestamp: str


class TutorState(TypedDict):
    # ---- intake ----
    student_question: str
    student_code: Optional[str]  # pasted code / error text, if provided

    # ---- planning (Story 01) ----
    intent: str  # "concept" | "debug" | "quiz_prep"
    plan: List[str]  # step-by-step plan shown to the student
    needs_clarification: bool
    clarifying_question: Optional[str]

    # ---- retrieval (Stories 04 / 05) ----
    retrieval_query: str  # may be rewritten from the raw question
    retrieved_chunks: List[RetrievedChunk]

    # ---- drafting ----
    draft_answer: str
    citations: List[str]

    # ---- reflection (Story 06) ----
    reflection_verdict: str  # "pass" | "fail"
    reflection_feedback: Optional[str]  # fix directive if verdict == "fail"
    retry_count: int

    # ---- output ----
    final_answer: str

    # ---- tools / logging (Story 03) ----
    tool_calls: List[ToolCall]


def create_initial_state(student_question: str, student_code: Optional[str] = None) -> TutorState:
    """Factory for a fresh TutorState at the start of a run."""
    return TutorState(
        student_question=student_question,
        student_code=student_code,
        intent="",
        plan=[],
        needs_clarification=False,
        clarifying_question=None,
        retrieval_query="",
        retrieved_chunks=[],
        draft_answer="",
        citations=[],
        reflection_verdict="",
        reflection_feedback=None,
        retry_count=0,
        final_answer="",
        tool_calls=[],
    )