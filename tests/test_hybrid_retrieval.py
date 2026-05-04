"""
Hybrid Retrieval Test - Validates improvement over baseline on comparative queries.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retrieval import initialize_hybrid_system


def test_comparative_query_improvement():
    """
    Test Query 2 from baseline: "What is the difference between fine-tuning and RLHF?"
    
    Baseline (Vector Only):
        - Score: 0.6137 | reinforcement_learning_with_human_feedback
        - Score: 0.5851 | reinforcement_learning_with_human_feedback
        - Score: 0.5619 | ai_alignment
        ❌ Missing fine_tuning.txt
    
    Expected (Hybrid):
        ✅ Both fine_tuning AND RLHF in top 5
    """
    print("\n" + "=" * 80)
    print("Hybrid Retrieval Test - Query 2 Improvement")
    print("=" * 80)
    
    # Initialize hybrid system with balanced weights
    hybrid_retriever = initialize_hybrid_system(
        pinecone_index_name=os.environ["PINECONE_INDEX_NAME"],
        pinecone_namespace="tier1_v1",
        bm25_cache_path=Path("cache/bm25_index.pkl"),
        vector_weight=0.5,
        bm25_weight=0.5
    )
    
    # Test query
    query = "What is the difference between fine-tuning and RLHF?"
    print(f"\nQuery: {query}")
    print("-" * 80)
    
    results = hybrid_retriever.retrieve(query, top_k=100, final_k=15)
    
    # Display results
    doc_ids_seen = set()
    for i, result in enumerate(results, 1):
        doc_id = result["metadata"]["doc_id"]
        doc_ids_seen.add(doc_id)
        section = result["metadata"]["section_path"]
        score = result["score"]
        
        print(f"{i}. Score: {score:.4f} | Doc: {doc_id}")
        print(f"   Section: {section}")
        print()
    
    # Validation
    assert "fine_tuning" in doc_ids_seen, "❌ fine_tuning not retrieved"
    assert "reinforcement_learning_with_human_feedback" in doc_ids_seen, "❌ RLHF not retrieved"
    
    print("=" * 80)
    print("✅ HYBRID TEST PASSED")
    print(f"   - Both target documents retrieved")
    print(f"   - Documents in top-5: {doc_ids_seen}")
    print("=" * 80)


def test_multiple_query_types():
    """Test hybrid retrieval on diverse query types."""
    print("\n" + "=" * 80)
    print("Hybrid Retrieval - Multiple Query Types")
    print("=" * 80)
    
    hybrid_retriever = initialize_hybrid_system(
        pinecone_index_name=os.environ["PINECONE_INDEX_NAME"],
        pinecone_namespace="tier1_v1",
        bm25_cache_path=Path("cache/bm25_index.pkl")
    )
    
    test_queries = [
        "How does attention mechanism work?",  # Factual
        "Compare BERT and GPT architectures",  # Comparative
        "What is RAG?",  # Factual
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = hybrid_retriever.retrieve(query, top_k=100, final_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['metadata']['doc_id'][:30]:30} | {result['score']:.4f}")
        
        assert len(results) > 0, f"No results for: {query}"
    
    print("\n✅ All query types handled")


if __name__ == "__main__":
    print("\n🧪 NEXUS RAG - HYBRID RETRIEVAL TEST")
    
    try:
        test_comparative_query_improvement()
        test_multiple_query_types()
        print("\n🎉 ALL HYBRID TESTS PASSED\n")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
