"""
Build the Qdrant knowledge base.

Usage:

    python -m ingestion.embed_and_index

The script reads data/course_pack.pdf, creates embeddings using
all-MiniLM-L6-v2, and stores the vectors in Qdrant.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from ingestion.chunker import chunk_pages
from ingestion.loader import load_pdf


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = PROJECT_ROOT / "data" / "course_pack.pdf"

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

BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE", "64")
)


def build_index() -> None:
    """Load, chunk, embed, and index the course pack."""

    print(f"Loading PDF: {PDF_PATH}")

    pages = load_pdf(PDF_PATH)

    print(f"Loaded {len(pages)} pages.")

    chunks = chunk_pages(
        pages,
        source=PDF_PATH.name,
    )

    if not chunks:
        raise RuntimeError("No chunks were produced from the PDF.")

    print(f"Created {len(chunks)} chunks.")

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    # Determine embedding dimensionality automatically.
    sample_vector = embedding_model.encode(
        chunks[0].text,
        normalize_embeddings=True,
    )

    vector_size = len(sample_vector)

    print(f"Embedding dimension: {vector_size}")

    print(f"Connecting to Qdrant: {QDRANT_URL}")

    qdrant_client = QdrantClient(
        url=QDRANT_URL
    )

    # Recreate the collection so ingestion is repeatable.
    # This is appropriate for a course-pack refresh workflow.
    if qdrant_client.collection_exists(QDRANT_COLLECTION):
        print(
            f"Deleting existing collection: "
            f"{QDRANT_COLLECTION}"
        )

        qdrant_client.delete_collection(
            collection_name=QDRANT_COLLECTION
        )

    print(
        f"Creating collection: "
        f"{QDRANT_COLLECTION}"
    )

    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    print("Generating embeddings and uploading vectors...")

    point_id = 0

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]

        texts = [chunk.text for chunk in batch]

        vectors = embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points: list[PointStruct] = []

        for chunk, vector in zip(batch, vectors):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "text": chunk.text,
                        "source": chunk.source,
                        "page": chunk.page,
                        "chapter": chunk.chapter,
                        "chunk_id": chunk.chunk_id,
                    },
                )
            )

            point_id += 1

        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
        )

        print(
            f"Indexed {min(start + BATCH_SIZE, len(chunks))}"
            f"/{len(chunks)} chunks"
        )

    collection_info = qdrant_client.get_collection(
        QDRANT_COLLECTION
    )

    print("\n=== INGESTION COMPLETE ===")
    print(f"Collection: {QDRANT_COLLECTION}")
    print(f"Vectors: {collection_info.points_count}")


if __name__ == "__main__":
    build_index()