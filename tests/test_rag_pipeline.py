import pytest
import logging
from unittest.mock import MagicMock

from src.pipeline.rag_pipeline import RAGPipeline
from src.generation.trust_formatter import TrustFormatter
from src.generation.answer_synthesizer import AnswerSynthesizer
from src.generation.refusal import RefusalReason
import time


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def mock_router():
    router = MagicMock()
    router.route_and_retrieve.return_value = {
        "strategy": "hybrid",
        "results": [
            {
                "chunk_id": "c1",
                "score": 1.0,
                "metadata": {
                    "doc_id": "doc1",
                    "section_path": "1. Overview",
                    "text": "RAG combines retrieval and generation."
                },
            }
        ],
    }
    return router


@pytest.fixture
def mock_synthesizer():
    synthesizer = MagicMock()
    synthesizer.synthesize.return_value = {
        "answer": "RAG combines retrieval and generation.",
        "confidence": 0.82,
        "citations": [
            {"doc_id": "doc1", "section": "1. Overview"}
        ],
        "used_chunk_ids": ["c1"],
        "refused": False,
        "retrieval_agreement": 1.0,
        "attribution_score": 1.0,
        "source_agreement": 0.5,
        "contradiction": False,
    }
    return synthesizer


@pytest.fixture
def trust_formatter():
    return TrustFormatter()


@pytest.fixture
def pipeline(mock_router, mock_synthesizer, trust_formatter):
    return RAGPipeline(
        router=mock_router,
        synthesizer=mock_synthesizer,
        trust_formatter=trust_formatter,
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_pipeline_returns_trust_formatted_response(pipeline):
    output = pipeline.run("What is RAG?")
    print(f"output= {output}")

    # ---- Schema checks ----
    assert "answer" in output
    assert "confidence" in output
    assert "citations" in output
    assert "trust_signals" in output
    assert "meta" in output

    assert "text" in output["answer"]
    assert "level" in output['confidence']
    assert "score" in output['confidence']
    

    # ---- Content checks ----
    assert output["answer"]["text"] == "RAG combines retrieval and generation."
    assert output["confidence"]["score"] == 0.82
    assert output["confidence"]["level"] in {"high", "medium", "low"}

    # ---- Metadata ----
    assert "latency_ms" in output["meta"]
    assert output["meta"]["strategy"] == "hybrid"


def test_pipeline_calls_components_in_order(pipeline, mock_router, mock_synthesizer):
    pipeline.run("What is RAG?")

    mock_router.route_and_retrieve.assert_called_once()
    mock_synthesizer.synthesize.assert_called_once()


def test_pipeline_refusal_propagates_to_user():
    router = MagicMock()
    router.route_and_retrieve.return_value = {
        "strategy": "vector",
        "results": [],
    }

    synthesizer = MagicMock()
    synthesizer.synthesize.return_value = {
        "answer": "",
        "confidence": 0.0,
        "citations": [],
        "used_chunk_ids": [],
        "refused": True,
    }

    pipeline = RAGPipeline(
        router=router,
        synthesizer=synthesizer,
        trust_formatter=TrustFormatter(),
    )

    output = pipeline.run("Unknown question")

    assert output["answer"] is None
    assert output["confidence"]["score"] == 0.0
    assert output["confidence"]["level"] == "low"
    
    assert output["refusal"]["refused"] is True
    assert output["refusal"]["reason"].lower() in {
        "insufficient_evidence",
        "low_confidence",
        "unsupported",
        "contradiction",
    }


def test_pipeline_does_not_leak_internal_fields(pipeline):
    output = pipeline.run("What is RAG?")

    forbidden_keys = {
        "support_score",
        "retrieval_agreement",
        "attribution_score",
        "source_agreement",
        "contradiction",
        "used_chunk_ids",
    }

    for key in forbidden_keys:
        assert key not in output


def test_pipeline_is_structurally_deterministic(pipeline):
    out1 = pipeline.run("What is RAG?")
    out2 = pipeline.run("What is RAG?")

    assert out1["confidence"]["level"] == out2["confidence"]["level"]
    assert out1["trust_signals"] == out2["trust_signals"]
    assert out1["refusal"] == out2["refusal"]


def test_short_query_handling(pipeline):
    out = pipeline.run("RAG?")
    assert "answer" in out


def test_empty_query_refusal(pipeline):
    out = pipeline.run("")
    print(f"out: {out}")
    assert out["refusal"]["refused"] is True


def test_pipeline_expands_multihop_queries_with_subquery_evidence(trust_formatter):
    router = MagicMock()
    router.route_and_retrieve.side_effect = [
        {
            "strategy": "hybrid",
            "analysis": {
                "intent": "relationship",
                "entities": ["in-context learning", "alignment risks"],
                "keywords": [],
                "confidence": 0.8,
                "complexity": 7,
                "keyword_density": 0.2,
                "requires_multiple_docs": True,
            },
            "results": [
                {
                    "chunk_id": "icl",
                    "score": 0.9,
                    "metadata": {
                        "doc_id": "doc1",
                        "section_path": "1. ICL",
                        "text": "In-context learning adapts behavior from examples in a prompt.",
                    },
                }
            ],
        },
        {
            "strategy": "hybrid",
            "results": [
                {
                    "chunk_id": "alignment",
                    "score": 0.85,
                    "metadata": {
                        "doc_id": "doc2",
                        "section_path": "2. Alignment",
                        "text": "Alignment risks include prompt injection and misuse.",
                    },
                }
            ],
        },
        {
            "strategy": "hybrid",
            "results": [
                {
                    "chunk_id": "icl",
                    "score": 0.8,
                    "metadata": {
                        "doc_id": "doc1",
                        "section_path": "1. ICL",
                        "text": "In-context learning adapts behavior from examples in a prompt.",
                    },
                }
            ],
        },
        {
            "strategy": "hybrid",
            "results": [],
        },
    ]

    synthesizer = MagicMock()
    synthesizer.synthesize.return_value = {
        "answer": "In-context learning can interact with alignment risks through prompt injection.",
        "confidence": 0.78,
        "citations": [{"doc_id": "doc1", "section": "1. ICL"}],
        "used_chunk_ids": ["icl", "alignment"],
        "refused": False,
        "retrieval_agreement": 0.8,
        "attribution_score": 0.8,
        "source_agreement": 0.6,
        "contradiction": False,
    }

    pipeline = RAGPipeline(
        router=router,
        synthesizer=synthesizer,
        trust_formatter=trust_formatter,
    )

    output = pipeline.run("How does in-context learning relate to alignment risks?")

    assert router.route_and_retrieve.call_count > 1
    passed_chunks = synthesizer.synthesize.call_args.kwargs["retrieved_chunks"]
    assert {c["chunk_id"] for c in passed_chunks} == {"icl", "alignment"}
    assert output["meta"]["query_plan"]["requires_decomposition"] is True
    assert output["meta"]["evidence_audit"]["sufficient"] is True


def test_long_query_does_not_crash(pipeline):
    long_query = "What is RAG? " * 500
    out = pipeline.run(long_query)
    assert "meta" in out


def test_all_refusal_reasons_are_valid_strings():
    for reason in RefusalReason:
        assert isinstance(reason.value, str)


def test_trust_formatter_handles_all_refusal_reasons():
    tf = TrustFormatter()
    for reason in RefusalReason:
        out = tf.format(
            answer_text="",
            confidence_score=0.0,
            citations=[],
            trust_inputs={},
            refused=True,
            refusal_reason=reason.value,
            meta={}
        )
        assert out["refusal"]["reason"] == reason.value


def test_pipeline_latency_reasonable(pipeline):
    start = time.time()
    pipeline.run("What is RAG?")
    elapsed = time.time() - start
    assert elapsed < 5.0   # seconds, not ms


def test_answer_trace_logging_does_not_crash(caplog, mock_router, trust_formatter):
    synthesizer = AnswerSynthesizer()

    pipeline = RAGPipeline(
        router=mock_router,              # mocked routing
        synthesizer=synthesizer,    # REAL synthesizer
        trust_formatter=trust_formatter,
    )

    caplog.set_level(logging.INFO, logger="src.generation.answer_synthesizer")

    pipeline.run("What is RAG?")

    traces = [
        rec.message
        for rec in caplog.records
        if "ANSWER_TRACE" in rec.message
    ]

    assert len(traces) == 1
