import pytest
from unittest.mock import MagicMock

from src.generation.answer_synthesizer import (
    AnswerSynthesizer,
    AnswerValidator,
    ContradictionDetector,
    ConfidenceCalibrator,
)

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def chunks_single_doc():
    return [
        {
            "chunk_id": "c1",
            "score": 1.0,
            "metadata": {
                "doc_id": "doc1",
                "section_path": "1. Overview",
                "text": "Retrieval augmented generation combines retrieval and generation."
            },
        }
    ]


@pytest.fixture
def chunks_multi_doc():
    return [
        {
            "chunk_id": "c1",
            "score": 1.0,
            "metadata": {
                "doc_id": "doc1",
                "section_path": "1. Overview",
                "text": "Retrieval augmented generation combines retrieval and generation."
            },
        },
        {
            "chunk_id": "c2",
            "score": 0.9,
            "metadata": {
                "doc_id": "doc2",
                "section_path": "1. Definition",
                "text": "RAG systems retrieve documents before generating answers."
            },
        },
    ]


@pytest.fixture
def supported_answer():
    return {
        "answer": "Retrieval augmented generation combines retrieval and generation.",
        "citations": [],
        "confidence": 0.0,
        "used_chunk_ids": [],
        "refused": False,
    }


@pytest.fixture
def unsupported_answer():
    return {
        "answer": "RAG was invented in 2035.",
        "citations": [],
        "confidence": 0.0,
        "used_chunk_ids": [],
        "refused": False,
    }

# ---------------------------------------------------------------------
# AnswerValidator Tests
# ---------------------------------------------------------------------

def test_support_score_high_for_supported_answer(chunks_single_doc, supported_answer):
    validator = AnswerValidator()
    score = validator.check_support(supported_answer, chunks_single_doc)
    assert 0.0 <= score <= 1.0


def test_support_score_low_for_unsupported_answer(chunks_single_doc, unsupported_answer):
    validator = AnswerValidator()
    score = validator.check_support(unsupported_answer, chunks_single_doc)
    assert score <= 0.2

# ---------------------------------------------------------------------
# ContradictionDetector Tests
# ---------------------------------------------------------------------

def test_no_contradiction_when_consistent(chunks_single_doc, supported_answer):
    detector = ContradictionDetector()
    assert detector.detect(supported_answer, chunks_single_doc) is False


def test_detects_simple_contradiction():
    detector = ContradictionDetector()
    answer = {"answer": "RAG is a retrieval system."}
    chunks = [
        {
            "metadata": {
                "text": "RAG is not a retrieval system."
            }
        }
    ]
    assert detector.detect(answer, chunks) is True

# ---------------------------------------------------------------------
# ConfidenceCalibrator Tests
# ---------------------------------------------------------------------

def test_confidence_increases_with_diversity(chunks_multi_doc):
    calibrator = ConfidenceCalibrator()

    low = calibrator.calibrate(
        support_score=1.0,
        contradiction=False,
        llm_used=False,
        chunks=chunks_multi_doc[:1],
    )

    high = calibrator.calibrate(
        support_score=1.0,
        contradiction=False,
        llm_used=False,
        chunks=chunks_multi_doc,
    )

    assert high > low


def test_contradiction_penalizes_confidence(chunks_multi_doc):
    calibrator = ConfidenceCalibrator()

    ok = calibrator.calibrate(
        support_score=1.0,
        contradiction=False,
        llm_used=False,
        chunks=chunks_multi_doc,
    )

    bad = calibrator.calibrate(
        support_score=1.0,
        contradiction=True,
        llm_used=False,
        chunks=chunks_multi_doc,
    )

    assert bad < ok


def test_confidence_clamped_between_0_and_1(chunks_multi_doc):
    calibrator = ConfidenceCalibrator()

    confidence = calibrator.calibrate(
        support_score=10.0,  # pathological
        contradiction=False,
        llm_used=False,
        chunks=chunks_multi_doc,
    )

    assert 0.0 <= confidence <= 1.0

# ---------------------------------------------------------------------
# AnswerSynthesizer Integration Tests (Mocked LLM)
# ---------------------------------------------------------------------

def test_refusal_when_confidence_too_low(monkeypatch, chunks_single_doc):
    synthesizer = AnswerSynthesizer()

    # Mock LLM response
    monkeypatch.setattr(
        synthesizer.client.chat.completions,
        "create",
        lambda **_: MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"answer": "Unknown fact"}'))]
        ),
    )

    result = synthesizer.synthesize("What is RAG?", chunks_single_doc)

    assert result["refused"] is True
    assert result["confidence"] == 0.0


def test_llm_verifier_triggered_on_low_support(monkeypatch, chunks_single_doc):
    synthesizer = AnswerSynthesizer()

    # Primary answer LLM
    monkeypatch.setattr(
        synthesizer.client.chat.completions,
        "create",
        lambda **_: MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"answer": "RAG was invented in 2035."}'
            ))]
        ),
    )

    # Verifier LLM
    monkeypatch.setattr(
        synthesizer,
        "_llm_verify_answer",
        lambda *args, **kwargs: "unsupported",
    )

    result = synthesizer.synthesize("What is RAG?", chunks_single_doc)

    assert result["refused"] is True


def test_detects_semantic_contradiction():
    detector = ContradictionDetector()

    chunks = [
        {
            "metadata": {
                "text": "RAG combines retrieval with generation.",
            }
        }
    ]

    answer = {
        "answer": "RAG does not use retrieval at all."
    }

    assert detector.detect(answer, chunks) is True


def test_no_contradiction_when_supported():
    detector = ContradictionDetector()

    chunks = [
        {
            "metadata": {
                "text": "RAG combines retrieval with generation.",
            }
        }
    ]

    answer = {
        "answer": "RAG uses retrieval together with generation."
    }

    assert detector.detect(answer, chunks) is False


# ---------------------------------------------------------------------
# Confidence Regression Tests (Monotonicity & Stability)
# ---------------------------------------------------------------------

def test_confidence_monotonic_with_support(chunks_multi_doc):
    calibrator = ConfidenceCalibrator()

    low = calibrator.calibrate(
        support_score=0.3,
        contradiction=False,
        llm_used=False,
        chunks=chunks_multi_doc,
    )

    high = calibrator.calibrate(
        support_score=0.9,
        contradiction=False,
        llm_used=False,
        chunks=chunks_multi_doc,
    )

    assert high > low


def test_confidence_penalized_when_llm_used(chunks_multi_doc):
    calibrator = ConfidenceCalibrator()

    no_llm = calibrator.calibrate(
        support_score=0.9,
        contradiction=False,
        llm_used=False,
        chunks=chunks_multi_doc,
        retrieval_agreement=1.0,
        attribution_score=1.0,
        source_agreement=1.0,
    )

    with_llm = calibrator.calibrate(
        support_score=0.9,
        contradiction=False,
        llm_used=True,
        chunks=chunks_multi_doc,
        retrieval_agreement=1.0,
        attribution_score=1.0,
        source_agreement=1.0,
    )

    assert with_llm < no_llm


# ---------------------------------------------------------------------
# Golden Failure Cases (Never Regress)
# ---------------------------------------------------------------------
def test_golden_failure_historical_fabrication(chunks_single_doc):
    synthesizer = AnswerSynthesizer()

    fake_answer = {
        "answer": "RAG was invented in 2035.",
        "citations": [],
        "confidence": 0.0,
        "used_chunk_ids": [],
        "refused": False,
    }

    support = synthesizer.validator.check_support(fake_answer, chunks_single_doc)
    contradiction = synthesizer.contradiction_detector.detect(fake_answer, chunks_single_doc)

    confidence = synthesizer.confidence_calibrator.calibrate(
        support_score=support,
        contradiction=contradiction,
        llm_used=False,
        chunks=chunks_single_doc,
    )

    assert confidence < 0.35


def test_golden_failure_logical_negation(chunks_single_doc):
    detector = ContradictionDetector()

    answer = {
        "answer": "RAG does not use retrieval."
    }

    assert detector.detect(answer, chunks_single_doc) is True