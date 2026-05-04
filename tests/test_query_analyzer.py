"""
Query Analyzer Tests - Nexus RAG
Validates structured query understanding for routing decisions.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.query_analyzer import QueryAnalyzer


def _assert_analysis_schema(analysis: dict):
    required_keys = {
        "intent",
        "complexity",
        "entities",
        "requires_multiple_docs",
        "keywords",
        "confidence",
        "keyword_density",
    }
    missing = required_keys - analysis.keys()
    assert not missing, f"Missing keys in analysis: {missing}"


def test_factual_query():
    analyzer = QueryAnalyzer()

    query = "What is a transformer architecture?"
    analysis = analyzer.analyze(query)

    _assert_analysis_schema(analysis)

    assert analysis["intent"] == "factual"
    assert analysis["complexity"] <= 4
    assert analysis["confidence"] >= 0.6
    assert analysis["requires_multiple_docs"] is False


def test_comparative_query():
    analyzer = QueryAnalyzer()

    query = "What is the difference between BERT and GPT?"
    analysis = analyzer.analyze(query)

    _assert_analysis_schema(analysis)

    assert analysis["intent"] == "comparative"
    assert analysis["requires_multiple_docs"] is True
    assert len(analysis["entities"]) >= 2
    assert analysis["confidence"] >= 0.6


def test_analytical_query():
    analyzer = QueryAnalyzer()

    query = "How does attention mechanism work in transformers?"
    analysis = analyzer.analyze(query)

    _assert_analysis_schema(analysis)

    assert analysis["intent"] == "analytical"
    assert analysis["complexity"] >= 4


def test_relationship_query():
    analyzer = QueryAnalyzer()

    query = "How are transformers and large language models related?"
    analysis = analyzer.analyze(query)

    _assert_analysis_schema(analysis)

    assert analysis["intent"] == "relationship"
    assert analysis["requires_multiple_docs"] is True


def test_fallback_logic():
    analyzer = QueryAnalyzer()

    # Force fallback by passing something pathological
    query = "??? ??? ???"
    analysis = analyzer._fallback_analysis(query)

    _assert_analysis_schema(analysis)

    assert analysis["intent"] in {
        "factual",
        "analytical",
        "comparative",
        "relationship",
    }
    assert 1 <= analysis["complexity"] <= 10
    assert isinstance(analysis["confidence"], float)