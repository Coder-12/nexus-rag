import pytest

from src.generation.answer_synthesizer import (
    RetrievalAgreementScorer,
    AttributionScorer,
    SourceAgreementScorer,
    ConfidenceCalibrator,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def chunks_three_docs():
    return [
        {
            "chunk_id": "c1",
            "metadata": {
                "doc_id": "doc1",
                "section_path": "1. Overview",
                "text": "RAG combines retrieval and generation.",
            },
        },
        {
            "chunk_id": "c2",
            "metadata": {
                "doc_id": "doc2",
                "section_path": "1. Overview",
                "text": "Retrieval augmented generation uses external knowledge.",
            },
        },
        {
            "chunk_id": "c3",
            "metadata": {
                "doc_id": "doc3",
                "section_path": "2. Details",
                "text": "Generation is grounded in retrieved documents.",
            },
        },
    ]


@pytest.fixture
def answer_with_citations():
    return {
        "answer": "RAG combines retrieval and generation.",
        "citations": [
            {"doc_id": "doc1", "section": "1. Overview"},
            {"doc_id": "doc2", "section": "1. Overview"},
        ],
    }


# ---------------------------------------------------------------------
# Retrieval Agreement
# ---------------------------------------------------------------------

def test_retrieval_agreement_full_overlap():
    scorer = RetrievalAgreementScorer()

    vector_ids = ["a", "b", "c"]
    bm25_ids = ["a", "b", "c"]

    score = scorer.score(vector_ids, bm25_ids)
    assert score == 1.0


def test_retrieval_agreement_partial_overlap():
    scorer = RetrievalAgreementScorer()

    vector_ids = ["a", "b", "c"]
    bm25_ids = ["b", "d", "e"]

    score = scorer.score(vector_ids, bm25_ids)
    assert 0.0 < score < 1.0


def test_retrieval_agreement_no_overlap():
    scorer = RetrievalAgreementScorer()

    vector_ids = ["a", "b"]
    bm25_ids = ["c", "d"]

    score = scorer.score(vector_ids, bm25_ids)
    assert score == 0.0


# ---------------------------------------------------------------------
# Attribution Scoring
# ---------------------------------------------------------------------

def test_attribution_score_high(answer_with_citations, chunks_three_docs):
    scorer = AttributionScorer()

    score = scorer.score(answer_with_citations, chunks_three_docs)
    assert score >= 0.6


def test_attribution_score_zero_when_no_citations(chunks_three_docs):
    scorer = AttributionScorer()

    answer = {"answer": "RAG explanation", "citations": []}
    score = scorer.score(answer, chunks_three_docs)

    assert score == 0.0


# ---------------------------------------------------------------------
# Source Agreement
# ---------------------------------------------------------------------

def test_source_agreement_multi_doc(chunks_three_docs):
    scorer = SourceAgreementScorer()

    score = scorer.score(chunks_three_docs)
    assert score == 1.0


def test_source_agreement_single_doc():
    scorer = SourceAgreementScorer()

    chunks = [
        {
            "chunk_id": "c1",
            "metadata": {"doc_id": "doc1", "text": "Only one doc"},
        }
    ]

    score = scorer.score(chunks)
    assert score < 1.0


# ---------------------------------------------------------------------
# Confidence Calibration
# ---------------------------------------------------------------------

def test_confidence_increases_with_agreement(chunks_three_docs):
    calibrator = ConfidenceCalibrator()

    low_conf = calibrator.calibrate(
        support_score=0.9,
        contradiction=False,
        llm_used=False,
        chunks=chunks_three_docs,
        retrieval_agreement=0.0,
        attribution_score=0.0,
        source_agreement=0.0,
    )

    high_conf = calibrator.calibrate(
        support_score=0.9,
        contradiction=False,
        llm_used=False,
        chunks=chunks_three_docs,
        retrieval_agreement=1.0,
        attribution_score=1.0,
        source_agreement=1.0,
    )

    assert high_conf > low_conf


def test_confidence_penalized_by_contradiction(chunks_three_docs):
    calibrator = ConfidenceCalibrator()

    no_contradiction = calibrator.calibrate(
        support_score=0.9,
        contradiction=False,
        llm_used=False,
        chunks=chunks_three_docs,
        retrieval_agreement=1.0,
        attribution_score=1.0,
        source_agreement=1.0,
    )

    with_contradiction = calibrator.calibrate(
        support_score=0.9,
        contradiction=True,
        llm_used=False,
        chunks=chunks_three_docs,
        retrieval_agreement=1.0,
        attribution_score=1.0,
        source_agreement=1.0,
    )

    assert with_contradiction < no_contradiction


def test_confidence_lower_when_llm_used(chunks_three_docs):
    calibrator = ConfidenceCalibrator()

    no_llm = calibrator.calibrate(
        support_score=0.9,
        contradiction=False,
        llm_used=False,
        chunks=chunks_three_docs,
        retrieval_agreement=1.0,
        attribution_score=1.0,
        source_agreement=1.0,
    )

    with_llm = calibrator.calibrate(
        support_score=0.9,
        contradiction=False,
        llm_used=True,
        chunks=chunks_three_docs,
        retrieval_agreement=1.0,
        attribution_score=1.0,
        source_agreement=1.0,
    )

    assert with_llm < no_llm


def test_confidence_bounded_between_0_and_1(chunks_three_docs):
    calibrator = ConfidenceCalibrator()

    score = calibrator.calibrate(
        support_score=10.0,   # extreme
        contradiction=False,
        llm_used=False,
        chunks=chunks_three_docs,
        retrieval_agreement=10.0,
        attribution_score=10.0,
        source_agreement=10.0,
    )

    assert 0.0 <= score <= 1.0

def test_soft_attribution_penalty_applied():
    calibrator = ConfidenceCalibrator()

    confidence = calibrator.calibrate(
        support_score=1.0,
        contradiction=False,
        llm_used=False,
        chunks=[{"metadata": {"doc_id": "doc1"}}],
        retrieval_agreement=1.0,
        attribution_score=0.2,
        source_agreement=1.0,
        answer_text="word " * 60,
    )

    assert confidence < 1.0

def test_no_penalty_for_short_answer():
    calibrator = ConfidenceCalibrator()

    confidence = calibrator.calibrate(
        support_score=1.0,
        contradiction=False,
        llm_used=False,
        chunks=[{"metadata": {"doc_id": "doc1"}}],
        retrieval_agreement=1.0,
        attribution_score=0.2,
        source_agreement=1.0,
        answer_text="Short factual answer.",
    )

    assert confidence >= 0.9
