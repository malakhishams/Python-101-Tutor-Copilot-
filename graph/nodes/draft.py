"""
Draft node (Story 06, "Drafting" step).

Writes the student-facing answer: a Level 1 explanation grounded in the
retrieved textbook chunks, plus a tiny example, plus citations.

Assumes it is only reached when retrieved_chunks is non-empty -- the
graph should route needs_clarification straight to the clarifying-
question exit instead of calling this node (see route_after_react in
react_router.py, and the equivalent check after retrieve.py).

On a reflection retry (retry_count > 0), reflection_feedback is folded
into the prompt as a fix directive so the second draft actually
addresses what failed the first time, instead of repeating the same
mistake.
"""

from graph.state import TutorState

DRAFT_SYSTEM_PROMPT = """You are a patient tutor for a beginner-level (Level 1) Python course.

Rules:
- Use only the textbook excerpts provided below. Do not use outside knowledge.
- Explain in simple, beginner-friendly language -- avoid jargon and advanced shortcuts.
- Include one tiny, runnable code example that matches your explanation.
- After the explanation, list which excerpt(s) you used, referenced by their chunk_id.
- If the excerpts don't actually answer the question, say so plainly instead of guessing.
"""

DRAFT_USER_PROMPT_TEMPLATE = """Student question: {question}
Student code/error (if any): {code}

Textbook excerpts:
{excerpts}
{fix_directive_block}
Write the explanation, the tiny example, and the citation list now."""

FIX_DIRECTIVE_TEMPLATE = """
A previous attempt at this answer was reviewed and needs fixing. Address this specifically:
{feedback}
"""


def _format_excerpts(chunks: list[dict]) -> str:
    """Turns retrieved_chunks into a numbered block the LLM can cite from
    by chunk_id, with page/chapter shown for traceability."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] chunk_id={chunk['chunk_id']} "
            f"(chapter: {chunk['chapter']}, page: {chunk['page']})\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(lines)


def _extract_citations(chunks: list[dict]) -> list[str]:
    """Citations are derived directly from what was retrieved, not
    parsed out of the LLM's free-text response -- this guarantees every
    citation shown to the student is a real chunk_id that exists in
    Qdrant, even if the LLM's own citation list in the text is messy."""
    return [chunk["chunk_id"] for chunk in chunks]


def draft_node(state: TutorState, llm_client) -> TutorState:
    """LangGraph node function. Reads student_question / student_code /
    retrieved_chunks / reflection_feedback, writes draft_answer +
    citations back into state."""

    chunks = state["retrieved_chunks"]
    excerpts = _format_excerpts(chunks)

    fix_directive_block = ""
    if state.get("retry_count", 0) > 0 and state.get("reflection_feedback"):
        fix_directive_block = FIX_DIRECTIVE_TEMPLATE.format(
            feedback=state["reflection_feedback"]
        )

    user_prompt = DRAFT_USER_PROMPT_TEMPLATE.format(
        question=state["student_question"],
        code=state.get("student_code") or "(none provided)",
        excerpts=excerpts,
        fix_directive_block=fix_directive_block,
    )

    draft_answer = llm_client.complete(
        prompt=user_prompt,
        system=DRAFT_SYSTEM_PROMPT,
    ).strip()

    state["draft_answer"] = draft_answer
    state["citations"] = _extract_citations(chunks)

    return state