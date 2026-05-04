"""
Minimal Reranking Sanity Test - Nexus RAG
Validates reranking integration without assumptions.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
print(PROJECT_ROOT)
# Your actual imports (as you have them)
from src.retrieval.hybrid_retrieval import initialize_hybrid_system


def test_reranking_sanity():
    """
    Minimal test: Does reranking work without breaking anything?
    """
    print("\n" + "=" * 80)
    print("RERANKING SANITY TEST")
    print("=" * 80)
    
    # Initialize with reranking
    hybrid = initialize_hybrid_system(
        pinecone_index_name=os.environ["PINECONE_INDEX_NAME"],
        pinecone_namespace="tier1_v1",
        cohere_api_key=os.environ["COHERE_API_KEY"],
        bm25_cache_path=Path("cache/bm25_index.pkl"),
        vector_weight=0.5,
        bm25_weight=0.5
    )
    
    # Test query (Query 2 from baseline)
    query = "What is the difference between fine-tuning and RLHF?"
    
    print(f"\nQuery: {query}")
    print("-" * 80)
    
    # Retrieve
    results = hybrid.retrieve(query, top_k=100)
    
    print(f"\n✅ Retrieved {len(results)} results")
    
    # Sanity checks
    assert len(results) > 0, "No results returned"
    assert len(results) <= 20, f"Expected ≤20 results, got {len(results)}"
    
    # Check schema
    first = results[0]
    assert "chunk_id" in first, "Missing chunk_id"
    assert "score" in first, "Missing score"
    assert "metadata" in first, "Missing metadata"
    
    print("✅ Schema valid")
    
    # Check metadata has text (required for reranking)
    has_text = "text" in first["metadata"]
    print(f"{'✅' if has_text else '⚠️'} metadata.text exists: {has_text}")
    
    if not has_text:
        print("   WARNING: Reranking may not work without text field")
    
    # Check both docs retrieved
    doc_ids = {r["metadata"]["doc_id"] for r in results[:10]}
    
    has_fine_tuning = "fine_tuning" in doc_ids
    has_rlhf = "reinforcement_learning_with_human_feedback" in doc_ids
    
    print(f"\nTop 10 docs: {doc_ids}")
    print(f"{'✅' if has_fine_tuning else '❌'} fine_tuning retrieved: {has_fine_tuning}")
    print(f"{'✅' if has_rlhf else '❌'} RLHF retrieved: {has_rlhf}")
    
    # Show top 5 with scores
    print(f"\nTop 5 Results:")
    for i, r in enumerate(results[:5], 1):
        doc = r["metadata"]["doc_id"]
        score = r["score"]
        print(f"  {i}. {doc[:40]:40} | {score:.4f}")
    
    print("\n" + "=" * 80)
    
    if has_fine_tuning and has_rlhf:
        print("✅ SANITY TEST PASSED - Reranking working")
    else:
        print("⚠️ WARNING - One doc missing, check reranking")
    
    print("=" * 80)


if __name__ == "__main__":
    test_reranking_sanity()
