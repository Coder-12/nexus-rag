from src.retrieval.vector_store import PineconeVectorStore
import os


def test_pinecone_adapter_smoke():
    store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME", "nexus-rag"),
        namespace="test"
    )

    stats = store.stats()
    assert "total_vector_count" in stats
