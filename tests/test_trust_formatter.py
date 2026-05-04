import pytest

from src.generation.trust_formatter import TrustFormatter


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def formatter():
    return TrustFormatter()


@pytest.fixture
def base_trust_inputs():
    return {
        "support_score": 0.9,
        "retrieval_agreement": 0.8,
        "attribution_score": 0.7,
        "source_agreement": 0.6,
        "contradiction": False,
    }


@pytest.fixture
def base_citations():
    return [
        {"doc_id": "doc1", "section": "1. Overview"},
        {"doc_id": "doc2", "section": "2. Definition"},
    ]


@pytest.fixture
def base_meta():
    return {
        "latency_ms": 120,
        "model": "gpt-4o-mini",
    }


# ---------------------------------------------------------------------
# Happy-path formatting
# ---------------------------------------------------------------------

def test_formats_successful_answer(
    formatter,
    base_trust_inputs,
    base_citations,
    base_meta,
):
    result = formatter.format(
        answer_text="RAG combines retrieval with generation.",
        confidence_score=0.82,
        citations=base_citations,
        trust_inputs=base_trust_inputs,
        refused=False,
        refusal_reason=None,
        meta=base_meta,
    )

    assert "answer" in result
    assert result["answer"]["text"].startswith("RAG combines")
    assert result["answer"]["type"] in {"direct", "summary", "comparison"}

    assert result["confidence"]["score"] == 0.82
    assert result["confidence"]["level"] == "high"

    assert result["refusal"] is None
    assert result["meta"] == base_meta


# ---------------------------------------------------------------------
# Confidence level bucketing
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "score,expected",
    [
        (0.85, "high"),
        (0.6, "medium"),
        (0.2, "low"),
    ],
)
def test_confidence_level_mapping(formatter, score, expected):
    result = formatter.format(
        answer_text="Test",
        confidence_score=score,
        citations=[],
        trust_inputs={},
        refused=False,
        refusal_reason=None,
        meta={},
    )

    assert result["confidence"]["level"] == expected


# ---------------------------------------------------------------------
# Trust signal interpretation
# ---------------------------------------------------------------------

def test_trust_signals_strong(formatter, base_trust_inputs):
    result = formatter.format(
        answer_text="Test",
        confidence_score=0.8,
        citations=[],
        trust_inputs=base_trust_inputs,
        refused=False,
        refusal_reason=None,
        meta={},
    )

    signals = result["trust_signals"]

    assert signals["evidence_supported"] is True
    assert signals["multi_source"] is True
    assert signals["retrieval_agreement"] == "strong"
    assert signals["attribution_quality"] in {"strong", "partial"}


def test_trust_signals_weak(formatter):
    trust_inputs = {
        "support_score": 0.2,
        "retrieval_agreement": 0.1,
        "attribution_score": 0.0,
        "source_agreement": 0.0,
        "contradiction": True,
    }

    result = formatter.format(
        answer_text="Test",
        confidence_score=0.15,
        citations=[],
        trust_inputs=trust_inputs,
        refused=False,
        refusal_reason=None,
        meta={},
    )

    signals = result["trust_signals"]

    assert signals["evidence_supported"] is False
    assert signals["multi_source"] is False
    assert signals["retrieval_agreement"] == "weak"
    assert signals["attribution_quality"] == "weak"


# ---------------------------------------------------------------------
# Citation formatting
# ---------------------------------------------------------------------

def test_citations_formatted_correctly(
    formatter,
    base_trust_inputs,
    base_citations,
):
    result = formatter.format(
        answer_text="Test",
        confidence_score=0.7,
        citations=base_citations,
        trust_inputs=base_trust_inputs,
        refused=False,
        refusal_reason=None,
        meta={},
    )

    citations = result["citations"]
    assert len(citations) == 2

    for c in citations:
        assert "doc_id" in c
        assert "section" in c
        assert c["used_for"] == "supporting evidence"


# ---------------------------------------------------------------------
# Refusal formatting
# ---------------------------------------------------------------------

def test_refusal_response_structure(formatter):
    result = formatter.format(
        answer_text="",
        confidence_score=0.0,
        citations=[],
        trust_inputs={},
        refused=True,
        refusal_reason="insufficient_evidence",
        meta={"latency_ms": 50},
    )

    assert result["answer"] is None
    assert result["confidence"]["score"] == 0.0
    assert result["refusal"]["refused"] is True
    assert "enough reliable information" in result["refusal"]["message"].lower()
    assert result["meta"]["latency_ms"] == 50


# ---------------------------------------------------------------------
# Schema stability regression test
# ---------------------------------------------------------------------

def test_response_schema_is_stable(formatter, base_trust_inputs):
    result = formatter.format(
        answer_text="Test",
        confidence_score=0.5,
        citations=[],
        trust_inputs=base_trust_inputs,
        refused=False,
        refusal_reason=None,
        meta={},
    )

    expected_keys = {
        "answer",
        "confidence",
        "citations",
        "trust_signals",
        "refusal",
        "meta",
    }

    assert set(result.keys()) == expected_keys
