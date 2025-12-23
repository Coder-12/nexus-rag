import pytest

from src.retrieval.document import Document, Section
from src.retrieval.chunking import SectionAwareChunker


def test_section_aware_chunking_sanity():
    # -----------------------------
    # Arrange: mock document
    # -----------------------------
    document = Document(
        doc_id="test_transformer",
        title="Transformer Architecture",
        source="Wikipedia",
        domain="AI / ML",
        cluster="Core Architecture",
        tier="Tier-1",
        raw_text="dummy",
        sections=[
            Section(
                section_id="test_transformer::section::0",
                title="Overview",
                text=("Transformers are neural network architectures "
                      "based on attention mechanisms. " * 120),
                path=["Overview"],
                order=0,
            ),
            Section(
                section_id="test_transformer::section::1",
                title="Encoder-Decoder Architecture",
                text=("The encoder-decoder architecture consists of "
                      "stacked self-attention layers. " * 140),
                path=["Architecture", "Encoder-Decoder"],
                order=1,
            ),
        ],
        metadata={"language": "en"},
    )

    chunker = SectionAwareChunker(
        target_tokens=512,
        overlap_tokens=50,
    )

    # -----------------------------
    # Act: chunk document
    # -----------------------------
    chunks = chunker.chunk_document(document)

    # -----------------------------
    # Assert: Phase B invariants
    # -----------------------------
    assert len(chunks) > 0, "No chunks produced"

    # 1. Chunk size sanity
    assert all(
        c.token_count <= 700 for c in chunks
    ), "Chunk exceeds max token threshold"

    # 2. Deterministic & unique IDs
    assert len(set(c.chunk_id for c in chunks)) == len(chunks)

    # 3. Section provenance preserved
    assert all(
        c.section_id for c in chunks
    ), "Chunk missing section_id"

    # 4. Section boundaries respected
    section_ids = {c.section_id for c in chunks}
    assert section_ids == {
        "test_transformer::section::0",
        "test_transformer::section::1",
    }

    # 5. Section path preserved
    assert all(
        isinstance(c.section_path, list) and len(c.section_path) > 0
        for c in chunks
    )