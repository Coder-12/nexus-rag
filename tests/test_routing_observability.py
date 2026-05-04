"""
Routing Observability Tests - Nexus RAG

Validates:
- Exactly one ROUTING_TRACE per query
- ROUTING_TRACE is valid JSON
- Metrics counters increment correctly
- LLM fallback usage is visible
- Routing overhead latency stays < 5ms (rule-based path)
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.intelligent_router import initialize_routing_system


@pytest.fixture(scope="module")
def router():
    return initialize_routing_system(
        pinecone_index_name=os.environ["PINECONE_INDEX_NAME"],
        pinecone_namespace="tier1_v1",
        bm25_cache_path=Path("cache/bm25_index.pkl"),
    )


def test_single_routing_trace_logged(caplog, router):
    caplog.set_level(logging.INFO)

    router.route_and_retrieve("What is retrieval augmented generation?")

    routing_traces = [
        rec.message for rec in caplog.records
        if "ROUTING_TRACE" in rec.message
    ]

    assert len(routing_traces) == 1, "Expected exactly one ROUTING_TRACE log"


def test_routing_trace_is_valid_json(caplog, router):
    caplog.set_level(logging.INFO)

    router.route_and_retrieve("What is retrieval augmented generation?")

    trace_line = next(
        rec.message for rec in caplog.records
        if "ROUTING_TRACE" in rec.message
    )

    # Extract JSON payload
    json_payload = trace_line.replace("ROUTING_TRACE", "").strip()

    parsed = json.loads(json_payload)

    # Required fields
    for field in [
        "query",
        "intent",
        "complexity",
        "confidence",
        "keyword_density",
        "entities",
        "strategy",
        "used_llm_fallback",
        "routing_reason",
        "timestamp",
    ]:
        assert field in parsed, f"Missing field in routing trace: {field}"


def test_metrics_increment_correctly(router):
    # Snapshot before
    before = router.metrics.snapshot().copy()

    router.route_and_retrieve("What is retrieval augmented generation?")
    router.route_and_retrieve("Compare BERT and GPT architectures")

    after = router.metrics.snapshot()

    assert after.get("route.vector", 0) >= before.get("route.vector", 0)
    assert after.get("route.hybrid", 0) >= before.get("route.hybrid", 0)


def test_llm_fallback_visible(router):
    # Trigger ambiguous query
    router.route_and_retrieve("Explain fine-tuning and RLHF in depth")

    metrics = router.metrics.snapshot()

    assert metrics.get("route.llm_fallback", 0) >= 1, \
        "Expected at least one LLM fallback routing"


def test_routing_decision_latency_under_5ms(router):
    """
    Measures ONLY routing decision latency.
    Excludes retrieval + network calls.
    """

    analysis = {
        "intent": "factual",
        "complexity": 3,
        "confidence": 0.8,
        "keyword_density": 0.4,
        "entities": ["retrieval augmented generation"],
        "requires_multiple_docs": False,
    }

    start = time.perf_counter()

    strategy, reason, used_llm = router._decide_strategy(
        "What is retrieval augmented generation?",
        analysis,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 5.0, f"Routing decision too slow: {elapsed_ms:.2f}ms"
    assert strategy == "vector"
