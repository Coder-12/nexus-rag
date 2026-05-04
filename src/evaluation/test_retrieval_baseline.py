"""
Baseline Retrieval Test — Nexus-RAG
Validates ingestion quality and dense retrieval health
before introducing routing or reranking layers.
"""

import os
import sys
from openai import OpenAI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.vector_store import PineconeVectorStore


# -----------------------------
# Test Queries (Positive Cases)
# -----------------------------
TEST_QUERIES = [
    # Core Architecture
    "How does the attention mechanism work in transformers?",

    # Training & Optimization
    "What is the difference between fine-tuning and RLHF?",

    # RAG & Retrieval
    "How do embeddings enable semantic search in RAG systems?",

    # Cross-cluster reasoning
    "How are transformers used in large language models?",

    # Self-referential grounding
    "What is retrieval augmented generation?",
]

# -----------------------------
# Negative / Out-of-domain Query
# -----------------------------
NEGATIVE_QUERY = "How do quantum error correction codes work?"


def test_baseline_retrieval():
    """
    Validates:
    - Dense semantic retrieval correctness
    - Metadata integrity
    - Score sanity
    - Out-of-domain behavior
    """

    client = OpenAI()

    vector_store = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        namespace="tier1_v1",
    )

    print("\n🧪 Nexus-RAG — Baseline Retrieval Validation")
    print("=" * 80)

    # -----------------------------
    # Positive Queries
    # -----------------------------
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 80)

        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=query,
        )
        query_vector = response.data[0].embedding

        results = vector_store.query(
            vector=query_vector,
            top_k=3,
        )

        # ---- Assertions: retrieval health ----
        assert results.matches is not None
        assert len(results.matches) > 0

        scores = []

        for j, match in enumerate(results.matches, 1):
            score = match.score
            scores.append(score)

            metadata = match.metadata

            # Metadata must exist
            assert "doc_id" in metadata
            assert "section_path" in metadata

            print(f"   {j}. Score: {score:.4f} | Doc: {metadata['doc_id']}")
            print(f"      Section: {metadata['section_path']}")
            print(f"      Text: {metadata.get('text', '')[:150]}...")
            print()

        # Score sanity: not random, not pathological
        assert max(scores) > 0.45
        assert max(scores) < 0.85

    # -----------------------------
    # Negative / Out-of-Domain Test
    # -----------------------------
    print("\n🔍 Negative Query (Out-of-Domain)")
    print("-" * 80)
    print(f"Query: {NEGATIVE_QUERY}")

    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=NEGATIVE_QUERY,
    )
    query_vector = response.data[0].embedding

    results = vector_store.query(
        vector=query_vector,
        top_k=3,
    )

    scores = [m.score for m in results.matches]

    for j, match in enumerate(results.matches, 1):
        print(
            f"   {j}. Score: {match.score:.4f} | Doc: {match.metadata.get('doc_id')}"
        )

    # Out-of-domain queries should not be confidently matched
    assert max(scores) < 0.75

    print("\n" + "=" * 80)
    print("✅ Baseline retrieval validation PASSED")
    print("System ready for routing and agentic layers.\n")


if __name__ == "__main__":
    test_baseline_retrieval()
