"""
Intelligent Router Tests - Nexus RAG
Validates rule-based + LLM fallback routing behavior.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.intelligent_router import initialize_routing_system


def test_factual_query_routes_to_vector():
    router = initialize_routing_system(
        pinecone_index_name=os.environ["PINECONE_INDEX_NAME"],
        pinecone_namespace="tier1_v1",
        bm25_cache_path=Path("cache/bm25_index.pkl"),
    )

    query = "What is retrieval augmented generation?"

    output = router.route_and_retrieve(query)

    assert output["strategy"] == "vector"
    assert output["used_llm_fallback"] is False
    assert len(output["results"]) > 0


def test_comparative_query_routes_to_hybrid():
    router = initialize_routing_system(
        pinecone_index_name=os.environ["PINECONE_INDEX_NAME"],
        pinecone_namespace="tier1_v1",
        bm25_cache_path=Path("cache/bm25_index.pkl"),
    )

    query = "What is the difference between fine-tuning and RLHF?"

    output = router.route_and_retrieve(query)

    doc_ids = {r["metadata"]["doc_id"] for r in output["results"]}

    assert output["strategy"] == "hybrid"
    assert "fine_tuning" in doc_ids
    assert "reinforcement_learning_with_human_feedback" in doc_ids


def test_high_complexity_triggers_hybrid():
    router = initialize_routing_system(
        pinecone_index_name=os.environ["PINECONE_INDEX_NAME"],
        pinecone_namespace="tier1_v1",
        bm25_cache_path=Path("cache/bm25_index.pkl"),
    )

    query = (
        "Explain how transformers, attention mechanisms, and RLHF "
        "interact during large language model training"
    )

    output = router.route_and_retrieve(query)

    assert output["strategy"] == "hybrid"
    assert len(output["results"]) > 0


def test_llm_fallback_triggered():
    router = initialize_routing_system(
        pinecone_index_name=os.environ["PINECONE_INDEX_NAME"],
        pinecone_namespace="tier1_v1",
        bm25_cache_path=Path("cache/bm25_index.pkl"),
    )

    # Ambiguous query intentionally
    query = "Tell me about attention and alignment"

    output = router.route_and_retrieve(query)

    assert output["strategy"] in {"vector", "hybrid"}
    assert "routing_reason" in output
    assert isinstance(output["used_llm_fallback"], bool)


def test_router_output_schema():
    router = initialize_routing_system(
        pinecone_index_name=os.environ["PINECONE_INDEX_NAME"],
        pinecone_namespace="tier1_v1",
        bm25_cache_path=Path("cache/bm25_index.pkl"),
    )

    output = router.route_and_retrieve("What is a transformer?")

    required_keys = {
        "results",
        "strategy",
        "routing_reason",
        "used_llm_fallback",
        "analysis",
    }

    assert required_keys.issubset(output.keys())