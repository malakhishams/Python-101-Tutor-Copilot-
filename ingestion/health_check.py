"""
Qdrant knowledge-base health check.

Usage:

    python -m ingestion.health_check
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


load_dotenv()


QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "python101_textbook",
)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)


def main() -> None:
    """Check collection health and perform a sample retrieval."""

    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {QDRANT_COLLECTION}")

    client = QdrantClient(
        url=QDRANT_URL
    )

    if not client.collection_exists(QDRANT_COLLECTION):
        raise RuntimeError(
            f"Collection '{QDRANT_COLLECTION}' does not exist."
        )

    info = client.get_collection(
        QDRANT_COLLECTION
    )

    print(f"Points: {info.points_count}")

    if not info.points_count:
        raise RuntimeError(
            f"Collection '{QDRANT_COLLECTION}' is empty."
        )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    query = "What is a Python list?"

    vector = model.encode(
        query,
        normalize_embeddings=True,
    )

    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector.tolist(),
        limit=3,
    )

    hits = result.points

    print("\n=== SAMPLE QUERY ===")
    print(f"Question: {query}")
    print(f"Hits: {len(hits)}")

    for index, hit in enumerate(hits, start=1):
        payload = hit.payload or {}

        print(f"\n--- Hit {index} ---")
        print(f"Score: {hit.score:.4f}")
        print(f"Page: {payload.get('page')}")
        print(f"Chapter: {payload.get('chapter')}")
        print(f"Chunk: {payload.get('chunk_id')}")
        print(
            f"Text: {payload.get('text', '')[:300]}..."
        )

    print("\n=== HEALTH CHECK PASSED ===")


if __name__ == "__main__":
    main()