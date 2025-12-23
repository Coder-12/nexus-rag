import os
import time

from src.retrieval.chunk import Chunk
from src.retrieval.embeddings import EmbeddingGenerator
from src.retrieval.vector_store import PineconeVectorStore


def test_e2e_embedding_and_vectorstore():
    # -----------------------------
    # Arrange
    # -----------------------------
    chunk = Chunk(
        chunk_id="e2e::doc::section::chunk::0",
        doc_id="e2e_doc",
        section_id="e2e_section",
        text=(
            "Transformers are neural network architectures based on "
            "self-attention mechanisms that enable parallel sequence modeling."
        ),
        token_count=20,
        chunk_index=0,
        total_chunks=1,
        section_path=["Overview"],
        metadata={
            "tier": "Tier-1",
            "cluster": "Core Architecture",
        },
    )

    # -----------------------------
    # Phase C: Embedding
    # -----------------------------
    embedder = EmbeddingGenerator(batch_size=1)
    embedded = embedder.embed_chunks([chunk])

    assert len(embedded) == 1
    assert embedded[0]["chunk_id"] == chunk.chunk_id
    assert len(embedded[0]["vector"]) == 3072

    # -----------------------------
    # Phase D: Vector Store
    # -----------------------------
    store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME", "nexus-rag"),
        namespace="e2e_test",
    )

    # Safe re-run
    store.delete_by_doc_id("e2e_doc")

    store.upsert(embedded)

    # Pinecone is eventually consistent
    time.sleep(2)

    # -----------------------------
    # Query
    # -----------------------------
    query_vector = embedded[0]["vector"]
    results = store.query(query_vector, top_k=3)

    # -----------------------------
    # Assertions (CORRECT)
    # -----------------------------
    assert results.matches is not None
    assert len(results.matches) >= 1

    top_match = results.matches[0]

    assert top_match["id"] == chunk.chunk_id
    assert top_match["metadata"]["doc_id"] == "e2e_doc"
    assert top_match["metadata"]["section_id"] == "e2e_section"