"""
Reflection node (Story 06, "Reflection" step).

Checks the draft answer against three criteria before it's allowed to
reach the student:
1. Correctness vs. the retrieved text (no claims unsupported by the excerpts)
2. Level 1 tone (no jargon/advanced shortcuts)
3. Whether the tiny example actually matches the explanation

Writes a pass/fail verdict. On fail, writes a short fix directive that
draft.py folds back into its next attempt (see FIX_DIRECTIVE_TEMPLATE
in draft.py) -- this is what makes the loop "reflect and refine"
instead of "reflect and give up."

Does NOT increment retry_count itself -- that's route_decision.py's
job, since incrementing is a routing decision (should we even retry?),
not a reflection judgment (was this attempt good?).
"""

from graph.state import TutorState


REFLECT_SYSTEM_PROMPT = """You are a strict but fair reviewer for a beginner (Level 1) Python tutoring answer.

Check the draft answer against the retrieved textbook excerpts on four criteria:

1. CORRECTNESS:
Every factual claim in the draft must be supported by the provided textbook excerpts.
Do not allow unsupported outside knowledge or invented facts.

2. TONE:
The language must be beginner-friendly and appropriate for a Level 1 Python student.
Avoid unnecessary jargon, advanced shortcuts, or assumed prior knowledge.

3. EXAMPLE_MATCH:
The tiny code example must actually demonstrate the concept explained in the answer.
The example must not contradict the explanation.

4. GROUNDING:
The answer must clearly rely on the retrieved textbook material and must not make claims that go beyond what the excerpts support.

Respond in EXACTLY this format, nothing else:

VERDICT: pass

or

VERDICT: fail
FEEDBACK: <one or two sentences, specific and actionable, describing exactly what to fix>

Only include FEEDBACK if VERDICT is fail."""


REFLECT_USER_PROMPT_TEMPLATE = """Student question:
{question}

Retrieved textbook excerpts:
{excerpts}

Draft answer to review:
{draft}

Review the draft against the four criteria and return the required VERDICT format."""


def _format_excerpts(chunks: list[dict]) -> str:
    """Format retrieved chunks with citation metadata."""

    lines = []

    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] "
            f"chunk_id={chunk['chunk_id']} | "
            f"chapter={chunk['chapter']} | "
            f"page={chunk['page']}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(lines)

def _parse_reflection_response(response: str) -> tuple[str, str | None]:
    """Parses the constrained VERDICT/FEEDBACK format. Falls back to
    'fail' with a generic feedback message on any malformed response --
    never silently treats an unparseable response as a pass, since that
    would defeat the point of the reflection gate."""
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]

    verdict = "fail"
    feedback = None

    for line in lines:
        if line.lower().startswith("verdict:"):
            value = line.split(":", 1)[1].strip().lower()
            if value in ("pass", "fail"):
                verdict = value
        elif line.lower().startswith("feedback:"):
            feedback = line.split(":", 1)[1].strip()

    if verdict == "fail" and feedback is None:
        feedback = (
            "Reflection response was malformed or incomplete -- "
            "re-check correctness against the excerpts, Level 1 tone, "
            "and that the example matches the explanation."
        )

    return verdict, feedback


def reflect_node(state: TutorState, llm_client) -> TutorState:
    """LangGraph node function. Reads draft_answer / retrieved_chunks,
    writes reflection_verdict + reflection_feedback back into state."""

    user_prompt = REFLECT_USER_PROMPT_TEMPLATE.format(
        question=state["student_question"],
        excerpts=_format_excerpts(state["retrieved_chunks"]),
        draft=state["draft_answer"],
    )

    response = llm_client.complete(prompt=user_prompt, system=REFLECT_SYSTEM_PROMPT)
    verdict, feedback = _parse_reflection_response(response)

    state["reflection_verdict"] = verdict
    state["reflection_feedback"] = feedback

    return state