"""
Tests for AnswerSynthesizer (Phase H1)

Validates:
- Evidence ranking by score
- Per-document diversity
- Overview/definition prioritization
- Token + chunk budget enforcement
- Citation correctness
- Refusal behavior
- Traceability (used_chunk_ids)
"""

import pytest
from src.generation.answer_synthesizer import AnswerSynthesizer


# ---------------------------------------------------------------------
# Test Utilities
# ---------------------------------------------------------------------

def fake_chunk(
    *,
    chunk_id="c1",
    score=1.0,
    doc_id="doc",
    section="1. Overview and Definition",
    text="Some relevant content"
):
    return {
        "chunk_id": chunk_id,
        "score": score,
        "metadata": {
            "doc_id": doc_id,
            "section_path": section,
            "text": text,
        },
    }


# ---------------------------------------------------------------------
# Evidence Selection Tests
# ---------------------------------------------------------------------

def test_selects_highest_scoring_chunk_survives():
    synthesizer = AnswerSynthesizer(max_context_chunks=3)

    chunks = [
        fake_chunk(chunk_id="c1", score=0.2),
        fake_chunk(chunk_id="c2", score=0.9),
        fake_chunk(chunk_id="c3", score=0.7),
    ]

    selected = synthesizer._select_chunks(chunks)

    assert len(selected) == 1
    assert selected[0]["chunk_id"] == "c2"


def test_limits_chunks_per_document():
    synthesizer = AnswerSynthesizer(max_chunks_per_doc=1)

    chunks = [
        fake_chunk(chunk_id="c1", doc_id="doc1", score=0.9),
        fake_chunk(chunk_id="c2", doc_id="doc1", score=0.8),
        fake_chunk(chunk_id="c3", doc_id="doc2", score=0.7),
    ]

    selected = synthesizer._select_chunks(chunks)
    doc_ids = [c["metadata"]["doc_id"] for c in selected]

    assert doc_ids.count("doc1") == 1
    assert doc_ids.count("doc2") == 1


def test_overview_section_not_excluded():
    synthesizer = AnswerSynthesizer(max_context_chunks=2)

    chunks = [
        fake_chunk(
            chunk_id="deep",
            section="10. Mathematical Derivations",
            score=0.95,
        ),
        fake_chunk(
            chunk_id="overview",
            section="1. Overview and Definition",
            score=0.85,
        ),
    ]

    selected = synthesizer._select_chunks(chunks)

    sections = [c["metadata"]["section_path"].lower() for c in selected]
    assert any("overview" in s for s in sections)


def test_avoids_redundant_sections():
    synthesizer = AnswerSynthesizer()

    chunks = [
        fake_chunk(chunk_id="c1", section="1. Overview", score=0.9),
        fake_chunk(chunk_id="c2", section="1. Overview", score=0.8),
    ]

    selected = synthesizer._select_chunks(chunks)

    assert len(selected) == 1


def test_max_context_chunks_is_upper_bound():
    synthesizer = AnswerSynthesizer(max_context_chunks=2)

    chunks = [
        fake_chunk(chunk_id=f"c{i}", score=1.0 - i * 0.1)
        for i in range(5)
    ]

    selected = synthesizer._select_chunks(chunks)

    assert len(selected) <= 2


def test_enforces_token_budget():
    synthesizer = AnswerSynthesizer(max_context_tokens=60)

    chunks = [
        fake_chunk(chunk_id="c1", text="word " * 40, score=0.9),
        fake_chunk(chunk_id="c2", text="word " * 40, score=0.8),
    ]

    selected = synthesizer._select_chunks(chunks)

    assert len(selected) == 1


# ---------------------------------------------------------------------
# Context + Citation Tests
# ---------------------------------------------------------------------

def test_build_context_and_citations():
    synthesizer = AnswerSynthesizer()

    chunks = [
        fake_chunk(
            chunk_id="c1",
            doc_id="docA",
            section="1. Overview",
            text="Definition text"
        ),
        fake_chunk(
            chunk_id="c2",
            doc_id="docB",
            section="2. Details",
            text="Details text"
        ),
    ]

    context, used_chunk_ids, citations = synthesizer._build_context(chunks)

    assert "Definition text" in context
    assert "Details text" in context

    assert used_chunk_ids == ["c1", "c2"]

    assert citations == [
        {"doc_id": "docA", "section": "1. Overview"},
        {"doc_id": "docB", "section": "2. Details"},
    ]


def test_citations_match_used_chunks():
    synthesizer = AnswerSynthesizer()

    chunks = [
        fake_chunk(chunk_id="c1", doc_id="docA"),
        fake_chunk(chunk_id="c2", doc_id="docB"),
    ]

    _, used_chunk_ids, citations = synthesizer._build_context(chunks)

    cited_docs = {c["doc_id"] for c in citations}
    used_docs = {c["metadata"]["doc_id"] for c in chunks}

    assert cited_docs == used_docs
    assert len(used_chunk_ids) == len(citations)


# ---------------------------------------------------------------------
# Refusal Behavior Tests
# ---------------------------------------------------------------------

def test_refusal_when_no_chunks():
    synthesizer = AnswerSynthesizer()

    result = synthesizer.synthesize(
        query="What is quantum gravity?",
        retrieved_chunks=[],
    )

    assert result["refused"] is True
    assert result["confidence"] == 0.0
    assert "don't have enough information" in result["answer"].lower()
    assert result["citations"] == []
    assert result["used_chunk_ids"] == []


# ---------------------------------------------------------------------
# Traceability Tests
# ---------------------------------------------------------------------

def test_used_chunk_ids_propagated_to_output(monkeypatch):
    synthesizer = AnswerSynthesizer()

    chunks = [
        fake_chunk(chunk_id="c1", score=0.9),
    ]

    # Mock OpenAI response to avoid real API call
    def mock_completion(*args, **kwargs):
        class FakeResponse:
            class Choice:
                message = type("msg", (), {
                    "content": """{
                        "answer": "Test answer",
                        "citations": [{"doc_id": "doc", "section": "1. Overview"}],
                        "confidence": 0.8,
                        "used_chunk_ids": ["c1"],
                        "refused": false
                    }"""
                })

            choices = [Choice()]

        return FakeResponse()

    monkeypatch.setattr(
        synthesizer.client.chat.completions,
        "create",
        mock_completion,
    )

    result = synthesizer.synthesize("test query", chunks)

    assert result["used_chunk_ids"] == ["c1"]
    assert result["refused"] is False
