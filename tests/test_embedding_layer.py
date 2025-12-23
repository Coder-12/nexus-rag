from src.retrieval.embeddings import EmbeddingGenerator
from src.retrieval.chunk import Chunk


def test_embedding_generator_sanity():
    chunks = [
        Chunk(
            chunk_id="doc::sec::chunk::0",
            doc_id="doc",
            section_id="sec",
            text="Transformers use attention mechanisms.",
            token_count=6,
            chunk_index=0,
            total_chunks=1,
            section_path=["Overview"],
            metadata={"tier": "Tier-1", "cluster": "Core Architecture"},
        )
    ]

    generator = EmbeddingGenerator(batch_size=1)
    results = generator.embed_chunks(chunks)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc::sec::chunk::0"
    assert len(results[0]["vector"]) == 3072
    assert "doc_id" in results[0]["metadata"]
