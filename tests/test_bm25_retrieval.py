"""
BM25 Retrieval Tests - Nexus RAG
Unit tests for BM25 sparse retrieval system.

Test Coverage:
1. Index building from chunks
2. Tokenization quality (technical terms)
3. Comparative query handling (both entities)
4. Keyword precision (exact term matching)
5. Cache persistence (save/load)
"""

import os
import sys
from pathlib import Path
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.document_loader import load_documents
from src.retrieval.chunking import SectionAwareChunker
from src.retrieval.bm25_retriever import (
    TechnicalTokenizer,
    BM25Index,
    BM25Retriever,
    initialize_bm25_system
)


# -----------------------------
# Test 1: Index Building
# -----------------------------
def test_bm25_index_building():
    """Validates that BM25 index builds successfully from chunks"""
    print("\n" + "=" * 80)
    print("Test 1: BM25 Index Building")
    print("=" * 80)
    
    # Load documents and create chunks
    data_dir = PROJECT_ROOT / "data" / "raw"
    documents = load_documents(data_dir)
    
    chunker = SectionAwareChunker()
    all_chunks = []
    for doc in documents[:3]:  # Test with 3 docs for speed
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
    
    print(f"Test corpus: {len(all_chunks)} chunks from 3 documents")
    
    # Build index
    tokenizer = TechnicalTokenizer()
    bm25_index = BM25Index(all_chunks, tokenizer)
    
    # Assertions
    assert len(bm25_index.chunk_ids) == len(all_chunks), "Chunk IDs mismatch"
    assert len(bm25_index.tokenized_corpus) == len(all_chunks), "Tokenized corpus mismatch"
    assert bm25_index.bm25 is not None, "BM25 model not initialized"
    
    print(f"✅ Index built successfully")
    print(f"   - {len(bm25_index.chunk_ids)} chunk IDs")
    print(f"   - {len(bm25_index.tokenized_corpus)} tokenized documents")
    print(f"   - BM25 model: {type(bm25_index.bm25).__name__}")
    
    return bm25_index


# -----------------------------
# Test 2: Tokenization Quality
# -----------------------------
def test_tokenization():
    """Validates that technical terms are handled correctly"""
    print("\n" + "=" * 80)
    print("Test 2: Tokenization Quality")
    print("=" * 80)
    
    tokenizer = TechnicalTokenizer()
    
    # Test cases
    test_cases = [
        # (input, expected_tokens)
        (
            "BERT and GPT use transformers",
            ["bert", "and", "gpt", "use", "transformers"]
        ),
        (
            "encoder-decoder architecture",
            ["encoder-decoder", "architecture"]
        ),
        (
            "self-attention mechanism in multi-head attention",
            ["self-attention", "mechanism", "in", "multi-head", "attention"]
        ),
        (
            "fine-tuning with RLHF",
            ["fine-tuning", "with", "rlhf"]
        ),
        (
            "zero-shot and few-shot learning",
            ["zero-shot", "and", "few-shot", "learning"]
        ),
    ]
    
    for i, (input_text, expected) in enumerate(test_cases, 1):
        tokens = tokenizer.tokenize(input_text)
        
        print(f"\nCase {i}:")
        print(f"  Input: '{input_text}'")
        print(f"  Tokens: {tokens}")
        print(f"  Expected: {expected}")
        
        # Check that important terms are present
        for term in expected:
            assert term in tokens, f"Expected term '{term}' not found in tokens"
        
        print(f"  ✅ Pass")
    
    print(f"\n✅ All tokenization tests passed")


# -----------------------------
# Test 3: Comparative Query
# -----------------------------
def test_comparative_query():
    """Validates that comparative queries retrieve from both documents"""
    print("\n" + "=" * 80)
    print("Test 3: Comparative Query Handling")
    print("=" * 80)
    
    # Load full corpus
    data_dir = PROJECT_ROOT / "data" / "raw"
    documents = load_documents(data_dir)
    
    chunker = SectionAwareChunker()
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
    
    print(f"Full corpus: {len(all_chunks)} chunks from {len(documents)} documents")
    
    # Build index
    bm25_index = BM25Index(all_chunks)
    bm25_retriever = BM25Retriever(bm25_index)
    
    from src.retrieval.vector_store import PineconeVectorStore
    
    store = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        namespace="tier1_v1",
    )
    
    # Get BM25 results
    bm25_results = bm25_retriever.search(
        "difference between fine-tuning and RLHF",
        top_k=5,
    )
    bm25_ids = [cid for cid, _ in bm25_results]

    # Try fetching those exact IDs from Pinecone
    fetch = store.index.fetch(
        ids=bm25_ids,
        namespace="tier1_v1",
    )

    print("BM25 IDs:", bm25_ids)
    print("Fetched IDs:", fetch.vectors.keys())
    
    # Test comparative query
    query = "difference between fine-tuning and RLHF"
    print(f"\nQuery: '{query}'")
    
    results = bm25_retriever.search(query, top_k=10)
    
    print(f"\nTop 10 Results:")
    doc_ids_seen = set()
    for i, (chunk_id, score) in enumerate(results, 1):
        doc_id = chunk_id.split("::")[0]
        doc_ids_seen.add(doc_id)
        print(f"  {i}. {doc_id[:30]:30} | Score: {score:.4f}")
    
    # Assertions
    assert len(results) > 0, "No results returned"
    
    # Should retrieve from BOTH fine_tuning and RLHF documents
    assert "fine_tuning" in doc_ids_seen, "fine_tuning document not retrieved"
    assert "reinforcement_learning_with_human_feedback" in doc_ids_seen, "RLHF document not retrieved"
    
    print(f"\n✅ Comparative query test passed")
    print(f"   - Retrieved from both target documents")
    print(f"   - Documents seen: {len(doc_ids_seen)}")


# -----------------------------
# Test 4: Keyword Precision
# -----------------------------
def test_keyword_precision():
    """Validates exact term matching works correctly"""
    print("\n" + "=" * 80)
    print("Test 4: Keyword Precision")
    print("=" * 80)
    
    # Load corpus
    data_dir = PROJECT_ROOT / "data" / "raw"
    documents = load_documents(data_dir)
    
    chunker = SectionAwareChunker()
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
    
    # Build index
    bm25_index = BM25Index(all_chunks)
    bm25_retriever = BM25Retriever(bm25_index)
    
    # Test queries with expected top doc
    test_queries = [
        ("BERT architecture", "bert_architecture"),
        ("GPT decoder", "gpt_architecture"),
        ("attention mechanism", "attention_mechanism"),
        ("transformer architecture", "transformer_architecture"),
    ]
    
    for query, expected_doc in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected top doc: {expected_doc}")
        
        results = bm25_retriever.search(query, top_k=5)
        
        assert len(results) > 0, f"No results for query: {query}"
        
        top_chunk_id, top_score = results[0]
        top_doc_id = top_chunk_id.split("::")[0]
        
        print(f"Top result: {top_doc_id} (score: {top_score:.4f})")
        
        # Should rank target document highly (in top 3)
        top_3_docs = [results[i][0].split("::")[0] for i in range(min(3, len(results)))]
        assert expected_doc in top_3_docs, f"Expected doc '{expected_doc}' not in top 3"
        
        print(f"✅ Pass (target doc in top 3)")
    
    print(f"\n✅ All keyword precision tests passed")


# -----------------------------
# Test 5: Cache Persistence
# -----------------------------
def test_cache_persistence():
    """Validates that index can be saved and loaded from disk"""
    print("\n" + "=" * 80)
    print("Test 5: Cache Persistence")
    print("=" * 80)
    
    # Load small corpus
    data_dir = PROJECT_ROOT / "data" / "raw"
    documents = load_documents(data_dir)
    
    chunker = SectionAwareChunker()
    all_chunks = []
    for doc in documents[:2]:  # 2 docs for speed
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
    
    print(f"Test corpus: {len(all_chunks)} chunks")
    
    # Build index
    original_index = BM25Index(all_chunks)
    
    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_bm25.pkl"
        
        print(f"\nSaving to: {cache_path}")
        original_index.save(cache_path)
        
        assert cache_path.exists(), "Cache file not created"
        print(f"✅ Cache saved ({cache_path.stat().st_size / 1024:.2f} KB)")
        
        # Load from cache
        print(f"\nLoading from cache...")
        loaded_index = BM25Index.load(cache_path)
        
        # Verify loaded index matches original
        assert len(loaded_index.chunk_ids) == len(original_index.chunk_ids), "Chunk IDs mismatch"
        assert len(loaded_index.tokenized_corpus) == len(original_index.tokenized_corpus), "Corpus mismatch"
        
        print(f"✅ Cache loaded successfully")
        print(f"   - {len(loaded_index.chunk_ids)} chunk IDs")
        print(f"   - {len(loaded_index.tokenized_corpus)} tokenized documents")
        
        # Test that loaded index works for search
        retriever = BM25Retriever(loaded_index)
        results = retriever.search("transformer attention", top_k=5)
        
        assert len(results) > 0, "Loaded index cannot perform search"
        print(f"✅ Loaded index is functional ({len(results)} results)")
    
    print(f"\n✅ Cache persistence test passed")


# -----------------------------
# Test 6: Integration Test
# -----------------------------
def test_full_integration():
    """End-to-end test: build → cache → load → query"""
    print("\n" + "=" * 80)
    print("Test 6: Full Integration Test")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "integration_test.pkl"
        
        # Load corpus
        data_dir = PROJECT_ROOT / "data" / "raw"
        documents = load_documents(data_dir)
        
        chunker = SectionAwareChunker()
        all_chunks = []
        for doc in documents[:5]:  # 5 docs for reasonable speed
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        print(f"Test corpus: {len(all_chunks)} chunks")
        
        # Initialize system (builds and caches)
        print("\n[Step 1] Building and caching index...")
        retriever = initialize_bm25_system(
            cache_path=cache_path,
            chunks=all_chunks,
            force_rebuild=True
        )
        
        # Query
        print("\n[Step 2] Testing query...")
        query = "how does attention mechanism work"
        results = retriever.search(query, top_k=5)
        
        assert len(results) > 0, "Query returned no results"
        print(f"✅ Query returned {len(results)} results")
        
        # Load from cache and query again
        print("\n[Step 3] Loading from cache and re-querying...")
        retriever_2 = initialize_bm25_system(
            cache_path=cache_path,
            chunks=None  # Should load from cache
        )
        
        results_2 = retriever_2.search(query, top_k=5)
        
        assert len(results_2) > 0, "Query from cached index failed"
        assert results[0][0] == results_2[0][0], "Results differ between fresh and cached index"
        
        print(f"✅ Cached index produces same results")
        
        print(f"\n✅ Integration test passed")


# -----------------------------
# Run All Tests
# -----------------------------
def run_all_tests():
    """Execute all BM25 tests"""
    print("\n" + "=" * 80)
    print("NEXUS RAG - BM25 RETRIEVAL TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Debug", test_bm25_ids_exist_in_pinecone),
        ("Index Building", test_bm25_index_building),
        ("Tokenization", test_tokenization),
        ("Comparative Query", test_comparative_query),
        ("Keyword Precision", test_keyword_precision),
        ("Cache Persistence", test_cache_persistence),
        ("Full Integration", test_full_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {test_name}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}")
            print(f"   {type(e).__name__}: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total: {passed + failed}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - BM25 SYSTEM READY")
    else:
        print(f"\n⚠️  {failed} test(s) failed - review errors above")
    
    return failed == 0


def test_bm25_ids_exist_in_pinecone():
    bm25 = initialize_bm25_system(Path("cache/bm25_index.pkl"))
    from src.retrieval.vector_store import PineconeVectorStore
    store = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        namespace="tier1_v1",
    )

    chunk_id, _ = bm25.search("fine-tuning", top_k=1)[0]

    res = store.index.fetch(ids=[chunk_id], namespace="tier1_v1")
    assert chunk_id in res.vectors
    
if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
