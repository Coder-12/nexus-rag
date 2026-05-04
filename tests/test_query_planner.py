from src.retrieval.query_planner import QueryPlanner


def _chunk(chunk_id: str, text: str, score: float = 0.8, doc_id: str = "doc"):
    return {
        "chunk_id": chunk_id,
        "score": score,
        "metadata": {
            "doc_id": doc_id,
            "section_path": "1. Test",
            "text": text,
        },
    }


def test_simple_factual_query_does_not_decompose():
    planner = QueryPlanner()

    plan = planner.plan(
        "What is RAG?",
        analysis={
            "intent": "factual",
            "entities": ["RAG"],
            "keywords": ["RAG"],
        },
    )

    assert plan.intent == "factual"
    assert plan.requires_decomposition is False
    assert plan.subqueries == []
    assert plan.answer_contract == "direct grounded answer"


def test_relationship_query_creates_generic_subqueries():
    planner = QueryPlanner()

    plan = planner.plan(
        "How does in-context learning relate to alignment risks?",
        analysis={
            "intent": "relationship",
            "entities": ["in-context learning", "alignment risks"],
            "keywords": [],
        },
    )

    assert plan.requires_decomposition is True
    assert "in context learning" in plan.evidence_requirements
    assert "alignment risks" in plan.evidence_requirements
    assert len(plan.subqueries) >= 2
    assert plan.answer_contract == "cover each evidence hop before the conclusion"


def test_evidence_audit_reports_missing_requirements():
    planner = QueryPlanner()
    plan = planner.plan(
        "How does in-context learning relate to alignment risks?",
        analysis={
            "intent": "relationship",
            "entities": ["in-context learning", "alignment risks"],
            "keywords": [],
        },
    )

    audit = planner.audit_coverage(
        plan,
        [_chunk("c1", "In-context learning adapts behavior from examples in a prompt.")],
    )

    assert audit["coverage_score"] == 0.5
    assert "in context learning" in audit["covered_requirements"]
    assert "alignment risks" in audit["missing_requirements"]
    assert audit["sufficient"] is False


def test_evidence_audit_passes_when_requirements_are_covered():
    planner = QueryPlanner()
    plan = planner.plan(
        "How does in-context learning relate to alignment risks?",
        analysis={
            "intent": "relationship",
            "entities": ["in-context learning", "alignment risks"],
            "keywords": [],
        },
    )

    audit = planner.audit_coverage(
        plan,
        [
            _chunk("c1", "In-context learning adapts behavior from prompt examples."),
            _chunk("c2", "Alignment risks include prompt injection and misuse.", doc_id="doc2"),
        ],
    )

    assert audit["coverage_score"] == 1.0
    assert audit["sufficient"] is True
    assert audit["unique_docs"] == 2
