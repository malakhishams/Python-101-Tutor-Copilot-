"""
Retrieval node (User Story 04 / 05).

Queries Qdrant for the top-k textbook chunks most relevant to the
student's question, using local sentence-transformers embeddings
(all-MiniLM-L6-v2 -- free, no API cost, same model used in ingestion).

If the best hit is below SCORE_THRESHOLD, retrieval is treated as
"weak": retrieved_chunks is left empty and the node asks a clarifying
question instead of passing noisy context to draft.py. This keeps the
system grounded-by-design (no good sources -> abstain), per the
non-functional requirements in the task brief.
"""

import os
from graph.state import TutorState, RetrievedChunk

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "python101_textbook")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# Cosine similarity threshold below which we don't trust the top hit.
# Tune this once you see real score distributions from your PDF chunks.
SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.35"))

WEAK_RETRIEVAL_CLARIFYING_QUESTION = (
    "I couldn't find a clear match for that in the course textbook. "
    "Could you rephrase your question, or tell me which topic or "
    "chapter it's related to (e.g. loops, functions, lists)?"
)


def _build_query(state: TutorState) -> str:
    """Prefer a rewritten retrieval_query if one was set upstream
    (e.g. by a reflection retry with a fix directive); otherwise fall
    back to the raw student question."""
    query = state.get("retrieval_query") or ""
    if query.strip():
        return query
    return state["student_question"]


def _embed(embedding_client, text: str) -> list[float]:
    """embedding_client is a sentence-transformers SentenceTransformer
    instance (all-MiniLM-L6-v2). .encode() returns a numpy array --
    Qdrant's client wants a plain list of floats."""
    vector = embedding_client.encode(text)
    return vector.tolist()


def retrieve_node(state: TutorState, qdrant_client, embedding_client) -> TutorState:
    """Retrieve relevant textbook chunks from Qdrant."""

    query = _build_query(state)
    state["retrieval_query"] = query

    query_vector = _embed(embedding_client, query)

    result = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=TOP_K,
    )

    hits = result.points

    if not hits or hits[0].score < SCORE_THRESHOLD:
        state["retrieved_chunks"] = []
        state["needs_clarification"] = True
        state["clarifying_question"] = (
            WEAK_RETRIEVAL_CLARIFYING_QUESTION
        )
        return state

    chunks: list[RetrievedChunk] = []

    for hit in hits:
        payload = hit.payload or {}

        chunks.append(
            RetrievedChunk(
                text=payload.get("text", ""),
                page=payload.get("page", -1),
                chapter=payload.get("chapter", ""),
                chunk_id=payload.get(
                    "chunk_id",
                    str(hit.id),
                ),
                score=hit.score,
            )
        )

    state["retrieved_chunks"] = chunks

    # Retrieval succeeded.
    state["needs_clarification"] = False
    state["clarifying_question"] = None

    return state

    chunks: list[RetrievedChunk] = []
    for hit in hits:
        payload = hit.payload or {}
        chunks.append(
            RetrievedChunk(
                text=payload.get("text", ""),
                page=payload.get("page", -1),
                chapter=payload.get("chapter", ""),
                chunk_id=payload.get("chunk_id", str(hit.id)),
                score=hit.score,
            )
        )