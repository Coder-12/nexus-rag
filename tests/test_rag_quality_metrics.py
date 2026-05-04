from src.evaluation.aggregate_rag_quality_metrics import aggregate


def test_retrieval_ranking_context_and_latency_metrics():
    rows = [
        {
            "question_id": "A",
            "expected_docs": ["doc_a", "doc_b"],
            "model_answer": "grounded answer",
            "citations": [{"doc_id": "doc_a", "section": "1"}],
            "meta": {
                "latency_ms": 100,
                "retrieved_doc_ids": ["noise", "doc_a", "doc_b"],
                "used_chunk_ids": ["c2"],
                "evidence_context": [
                    {"chunk_id": "c1", "doc_id": "noise", "text": "noise"},
                    {"chunk_id": "c2", "doc_id": "doc_a", "text": "support"},
                ],
                "evidence_audit": {"coverage_score": 1.0, "sufficient": True},
                "query_plan": {"requires_decomposition": False},
                "evidence_recovered": False,
            },
            "judge": {"score": 5, "hallucination": False},
            "faithfulness": {
                "faithful": True,
                "faithfulness_score": 0.9,
                "citation_supported": True,
            },
        },
        {
            "question_id": "B",
            "expected_docs": ["doc_c"],
            "model_answer": "unsupported answer",
            "citations": [{"doc_id": "noise", "section": "1"}],
            "meta": {
                "latency_ms": 1000,
                "retrieved_doc_ids": ["noise", "doc_c"],
                "used_chunk_ids": ["c3"],
                "evidence_context": [
                    {"chunk_id": "c3", "doc_id": "noise", "text": "noise"},
                ],
                "evidence_audit": {"coverage_score": 0.5, "sufficient": False},
                "query_plan": {"requires_decomposition": True},
                "evidence_recovered": True,
            },
            "judge": {"score": 3, "hallucination": True},
            "faithfulness": {
                "faithful": False,
                "faithfulness_score": 0.2,
                "citation_supported": False,
            },
        },
    ]

    metrics = aggregate(rows)

    retrieval = metrics["retrieval_quality"]
    assert retrieval["recall_at_k"][1] == 0.0
    assert retrieval["recall_at_k"][3] == 1.0
    assert round(retrieval["mrr_at_10"], 3) == 0.5
    assert 0.0 < retrieval["ndcg_at_10"] < 1.0

    context = metrics["context_quality"]
    assert context["context_precision"] == 0.5
    assert context["context_recall"] == 0.25
    assert context["citation_precision"] == 0.5
    assert context["citation_recall"] == 0.25

    faithfulness = metrics["faithfulness"]
    assert faithfulness["coverage"] == 1.0
    assert faithfulness["faithful_rate"] == 0.5
    assert faithfulness["avg_faithfulness_score"] == 0.55

    latency = metrics["latency"]
    assert latency["p90_ms"] == 1000
    assert latency["p95_ms"] == 1000
    assert latency["p99_ms"] == 1000
    assert latency["max_ms"] == 1000
