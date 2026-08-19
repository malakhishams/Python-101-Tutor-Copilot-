"""
Entrypoint (Story 06 + Story 07 demo runner).

Builds the clients, compiles the LangGraph, runs a student question,
and records the important workflow information in Langfuse.

Usage:
    python run.py "What is a Python list?"
    python run.py "Why is my indentation failing?" --code "if True:\nprint('hi')"
"""

import argparse
import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from graph.state import create_initial_state
from graph.build_graph import build_graph
from graph.llm_client import create_llm_client

from observability.langfuse_setup import (
    get_langfuse_client,
    langfuse_is_configured,
    flush_langfuse,
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)


def build_clients():
    """
    Constructs the clients needed by the Tutor Copilot.
    """

    llm_client = create_llm_client()

    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        check_compatibility=False,
    )

    embedding_client = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    return llm_client, qdrant_client, embedding_client


def run_question(question: str, code: str | None = None) -> dict:
    """
    Runs one student question through the full Tutor Copilot graph.

    If Langfuse is configured, the complete run is recorded as one
    trace with observations for the important workflow outputs.
    """

    llm_client, qdrant_client, embedding_client = build_clients()

    graph = build_graph(
        llm_client,
        qdrant_client,
        embedding_client,
    )

    initial_state = create_initial_state(
        student_question=question,
        student_code=code,
    )

    langfuse = get_langfuse_client()

    # ---------------------------------------------------------
    # Run with Langfuse tracing
    # ---------------------------------------------------------

    if langfuse_is_configured() and langfuse is not None:

        with langfuse.start_as_current_observation(
            as_type="span",
            name="python-101-tutor-copilot-run",
            input={
                "student_question": question,
                "student_code": code,
            },
        ) as run_trace:

            final_state = graph.invoke(initial_state)

            # -----------------------------
            # Plan
            # -----------------------------

            with langfuse.start_as_current_observation(
                as_type="span",
                name="intake_plan",
                input={
                    "student_question": question,
                },
            ) as observation:

                observation.update(
                    output={
                        "intent": final_state.get("intent"),
                        "plan": final_state.get("plan"),
                    }
                )

            # -----------------------------
            # Retrieval
            # -----------------------------

            retrieved_chunks = final_state.get(
                "retrieved_chunks",
                [],
            )

            retrieval_data = []

            for chunk in retrieved_chunks:
                retrieval_data.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "page": chunk.get("page"),
                        "chapter": chunk.get("chapter"),
                        "score": chunk.get("score"),
                    }
                )

            with langfuse.start_as_current_observation(
                as_type="span",
                name="retrieval",
                input={
                    "retrieval_query": final_state.get(
                        "retrieval_query"
                    ),
                },
            ) as observation:

                observation.update(
                    output={
                        "num_chunks": len(retrieval_data),
                        "chunks": retrieval_data,
                        "needs_clarification": final_state.get(
                            "needs_clarification"
                        ),
                    }
                )

            # -----------------------------
            # Tool calls
            # -----------------------------

            tool_calls = final_state.get(
                "tool_calls",
                [],
            )

            if tool_calls:

                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="tool_calls",
                    input={
                        "tool_calls": tool_calls,
                    },
                ) as observation:

                    observation.update(
                        output={
                            "num_tool_calls": len(tool_calls),
                            "tool_calls": tool_calls,
                        }
                    )

            # -----------------------------
            # Reflection
            # -----------------------------

            with langfuse.start_as_current_observation(
                as_type="span",
                name="reflection",
                input={
                    "draft_answer": final_state.get(
                        "draft_answer"
                    ),
                },
            ) as observation:

                observation.update(
                    output={
                        "verdict": final_state.get(
                            "reflection_verdict"
                        ),
                        "feedback": final_state.get(
                            "reflection_feedback"
                        ),
                        "retry_count": final_state.get(
                            "retry_count"
                        ),
                    }
                )

            # -----------------------------
            # Final answer
            # -----------------------------

            run_trace.update(
                output={
                    "final_answer": final_state.get(
                        "final_answer"
                    ),
                    "citations": final_state.get(
                        "citations"
                    ),
                    "retry_count": final_state.get(
                        "retry_count"
                    ),
                    "needs_clarification": final_state.get(
                        "needs_clarification"
                    ),
                }
            )

    # ---------------------------------------------------------
    # Run without Langfuse
    # ---------------------------------------------------------

    else:
        final_state = graph.invoke(initial_state)

    # Make sure events are sent before the CLI exits.
    flush_langfuse()

    return final_state


def main():
    parser = argparse.ArgumentParser(
        description="Run one question through the Tutor Copilot graph."
    )

    parser.add_argument(
        "question",
        type=str,
        help="The student's question.",
    )

    parser.add_argument(
        "--code",
        type=str,
        default=None,
        help="Pasted code or error text, if any.",
    )

    args = parser.parse_args()

    final_state = run_question(
        args.question,
        args.code,
    )

    print("\n=== PLAN ===")

    for step in final_state["plan"]:
        print(f"- {step}")

    if final_state.get("needs_clarification"):

        print("\n=== CLARIFYING QUESTION ===")

    else:

        print(
            f"\n=== FINAL ANSWER "
            f"(retries: {final_state['retry_count']}) ==="
        )

    print(final_state["final_answer"])

    if final_state["citations"]:

        print("\n=== CITATIONS ===")

        for citation in final_state["citations"]:
            print(f"- {citation}")


if __name__ == "__main__":
    main()