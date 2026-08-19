"""
Entrypoint (Story 06 demo runner).

Builds the three injected clients (LLM, Qdrant, embeddings), compiles
the graph once, and invokes it for a single student question. This is
the file the demo script (demo/demo_script.md) will call for each
question in the walkthrough.

Usage:
    python -m graph.run "What's the difference between a list and a tuple?"
    python -m graph.run "Why is my indentation failing?" --code "if True:\nprint('hi')"
"""

import argparse
import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from graph.state import create_initial_state
from graph.build_graph import build_graph
from graph.llm_client import create_llm_client

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def build_clients():
    """Constructs the three clients every graph node depends on. Kept
    in one place so run.py and any future eval script (Story 07) build
    them identically."""
    llm_client = create_llm_client()
    qdrant_client = QdrantClient(url=QDRANT_URL)
    embedding_client = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return llm_client, qdrant_client, embedding_client


def run_question(question: str, code: str | None = None) -> dict:
    """Runs a single student question through the full graph and
    returns the final state (so callers can inspect plan, retrieved
    chunks, retry_count, etc. -- not just the final answer)."""
    llm_client, qdrant_client, embedding_client = build_clients()
    graph = build_graph(llm_client, qdrant_client, embedding_client)

    initial_state = create_initial_state(student_question=question, student_code=code)
    final_state = graph.invoke(initial_state)

    return final_state


def main():
    parser = argparse.ArgumentParser(description="Run one question through the Tutor Copilot graph.")
    parser.add_argument("question", type=str, help="The student's question.")
    parser.add_argument("--code", type=str, default=None, help="Pasted code or error text, if any.")
    args = parser.parse_args()

    final_state = run_question(args.question, args.code)

    print("\n=== PLAN ===")
    for step in final_state["plan"]:
        print(f"- {step}")

    if final_state.get("needs_clarification"):
        print("\n=== CLARIFYING QUESTION ===")
    else:
        print(f"\n=== FINAL ANSWER (retries: {final_state['retry_count']}) ===")
    print(final_state["final_answer"])

    if final_state["citations"]:
        print("\n=== CITATIONS ===")
        for c in final_state["citations"]:
            print(f"- {c}")


if __name__ == "__main__":
    main()