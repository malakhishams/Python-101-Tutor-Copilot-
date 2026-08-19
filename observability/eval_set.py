"""
Evaluation dataset + scoring (User Story 07).

A small set of Level 1 questions run end-to-end through the graph.
Each run is scored on three metrics using an LLM-as-judge, and the
scores are attached to that run's Langfuse trace so a QA lead can see
groundedness/clarity/correctness per run, not just raw traces.

Usage:
    python -m observability.eval_set
"""

import json
import re

from graph.state import create_initial_state
from graph.build_graph import build_graph
from graph.llm_client import create_llm_client
from observability.langfuse_setup import (
    get_langfuse_client,
    langfuse_is_configured,
    score_current_trace,
    flush_langfuse,
)

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# Small, deliberately varied Level 1 eval set: a mix of concept, debug,
# and quiz_prep questions, including one that should trigger clarification
# (debug with no code) so the eval set exercises more than the happy path.
EVAL_QUESTIONS = [
    {"question": "What is the difference between a list and a tuple?", "code": None},
    {"question": "Why is my indentation failing?", "code": None},  # expect clarification
    {
        "question": "Why do I get an IndexError in this code?",
        "code": "nums = [1, 2, 3]\nprint(nums[3])",
    },
    {"question": "How do for loops work in Python?", "code": None},
    {"question": "What's the difference between == and = in Python?", "code": None},
    {"question": "Create a 3-question beginner quiz about lists", "code": None},
]


JUDGE_SYSTEM_PROMPT = """You are grading a beginner (Level 1) Python tutoring answer on three metrics.
Score each metric from 0.0 to 1.0.

- groundedness: Is every claim in the answer supported by the retrieved textbook excerpts? (1.0 = fully supported, 0.0 = fabricated/unsupported)
- clarity: Is the answer written in simple, beginner-friendly language a Level 1 student could follow? (1.0 = very clear, 0.0 = confusing/jargon-heavy)
- correctness: Is the answer factually correct about Python? (1.0 = fully correct, 0.0 = wrong)

Respond with ONLY a JSON object, no other text:
{"groundedness": <float>, "clarity": <float>, "correctness": <float>}"""

JUDGE_USER_PROMPT = """Student question: {question}

Retrieved textbook excerpts:
{excerpts}

Answer to grade:
{answer}"""


def _format_excerpts(chunks: list[dict]) -> str:
    if not chunks:
        return "(none retrieved -- this was a clarification response)"
    return "\n\n".join(
        f"[{c.get('chunk_id')}] {c.get('text', '')}" for c in chunks
    )


def judge_answer(llm_client, question: str, chunks: list[dict], answer: str) -> dict:
    """LLM-as-judge scoring. Falls back to zeros on any parse failure
    so one bad eval item can't crash the whole run."""
    prompt = JUDGE_USER_PROMPT.format(
        question=question,
        excerpts=_format_excerpts(chunks),
        answer=answer,
    )
    raw = llm_client.complete(prompt=prompt, system=JUDGE_SYSTEM_PROMPT).strip()

    # Strip accidental markdown fences before parsing.
    raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        scores = json.loads(raw)
        return {
            "groundedness": float(scores.get("groundedness", 0.0)),
            "clarity": float(scores.get("clarity", 0.0)),
            "correctness": float(scores.get("correctness", 0.0)),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        print(f"  [judge] could not parse judge output, defaulting to 0.0: {raw!r}")
        return {"groundedness": 0.0, "clarity": 0.0, "correctness": 0.0}


def run_eval() -> None:
    """Runs every question in EVAL_QUESTIONS through the graph, scores
    each result, attaches scores to that run's Langfuse trace, and
    prints a summary table."""

    llm_client = create_llm_client()
    qdrant_client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    embedding_client = SentenceTransformer(EMBEDDING_MODEL_NAME)
    graph = build_graph(llm_client, qdrant_client, embedding_client)

    langfuse = get_langfuse_client()
    use_langfuse = langfuse_is_configured() and langfuse is not None

    results = []

    for i, item in enumerate(EVAL_QUESTIONS, start=1):
        question, code = item["question"], item["code"]
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] {question}")

        initial_state = create_initial_state(student_question=question, student_code=code)

        if use_langfuse:
            with langfuse.start_as_current_observation(
                as_type="span",
                name="eval-run",
                input={"question": question, "code": code},
            ) as trace:
                final_state = graph.invoke(initial_state)

                if final_state.get("needs_clarification"):
                    # Clarification responses aren't graded -- there's no
                    # drafted answer to judge yet.
                    print("  -> clarification requested, skipping scoring")
                    trace.update(output={"final_answer": final_state.get("final_answer")})
                    results.append({"question": question, "skipped": True})
                    continue

                scores = judge_answer(
                    llm_client,
                    question,
                    final_state.get("retrieved_chunks", []),
                    final_state.get("final_answer", ""),
                )

                for metric, value in scores.items():
                    score_current_trace(name=metric, value=value)

                trace.update(output={"final_answer": final_state.get("final_answer"), "scores": scores})
        else:
            final_state = graph.invoke(initial_state)
            if final_state.get("needs_clarification"):
                print("  -> clarification requested, skipping scoring")
                results.append({"question": question, "skipped": True})
                continue
            scores = judge_answer(
                llm_client,
                question,
                final_state.get("retrieved_chunks", []),
                final_state.get("final_answer", ""),
            )

        print(f"  groundedness={scores['groundedness']:.2f}  clarity={scores['clarity']:.2f}  correctness={scores['correctness']:.2f}")
        results.append({"question": question, "scores": scores, "skipped": False})

    flush_langfuse()

    # ---- summary ----
    graded = [r for r in results if not r.get("skipped")]
    print("\n=== EVAL SUMMARY ===")
    print(f"Total questions: {len(results)}  |  Graded: {len(graded)}  |  Clarification-skipped: {len(results) - len(graded)}")

    if graded:
        for metric in ("groundedness", "clarity", "correctness"):
            avg = sum(r["scores"][metric] for r in graded) / len(graded)
            print(f"Average {metric}: {avg:.2f}")

    if not use_langfuse:
        print("\n[note] Langfuse not configured -- scores were computed and printed "
              "but not attached to any trace. Set LANGFUSE_PUBLIC_KEY / "
              "LANGFUSE_SECRET_KEY in .env to persist scores.")


if __name__ == "__main__":
    run_eval()