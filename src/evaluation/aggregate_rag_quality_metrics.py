"""
Production/research-grade RAG quality metrics.

This complements the LLM judge aggregate by separating answer quality from
retrieval, evidence coverage, latency, and agentic planning behavior.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable


RESULTS_PATH = Path(os.getenv("EVAL_RESULTS_PATH", "outputs/evaluation_results.jsonl"))
RANKING_KS = (1, 3, 5, 10)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval results file: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(value: float) -> str:
    return f"{value:.2%}"


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def normalize_doc_id(doc_ref: str | None) -> str:
    if not doc_ref:
        return ""
    doc_ref = str(doc_ref).strip()
    if not doc_ref or doc_ref.lower() in {"none", "not", "n/a", "na"}:
        return ""
    doc_ref = doc_ref.split()[0].strip()
    if doc_ref.endswith(".txt"):
        doc_ref = doc_ref[:-4]
    aliases = {
        "ai_alignment_tier1_core": "ai_alignment",
        "attention_mechanism_tier1_core": "attention_mechanism",
        "bert_language_model_tier1_core": "bert_architecture",
        "embeddings_machine_learning_tier1_core": "embeddings",
        "fine_tuning_deep_learning_tier1_core": "fine_tuning",
        "gpt_generative_pretrained_transformer_tier1_core": "gpt_architecture",
        "in_context_learning_tier1_core": "in_context_learning",
        "large_language_model_llm_tier1_core": "large_language_models",
        "prompt_engineering_tier1_core": "prompt_engineering",
        "production_rag_evaluation_tier1_core": "production_rag_evaluation",
        "retrieval_augmented_generation_rag_tier1_core": "retrieval_augmented_generation",
        "rlhf_reinforcement_learning_human_feedback_tier1_core": "reinforcement_learning_with_human_feedback",
        "semantic_search_tier1_core": "semantic_search",
        "seq2seq_sequence_to_sequence_tier1_core": "encoder_decoder_models",
        "transfer_learning_tier1_core": "transfer_learning",
        "transformer_architecture_tier1_core": "transformer_architecture",
        "vector_database_tier1_core": "vector_database",
    }
    return aliases.get(doc_ref, doc_ref)


def dedupe(values: Iterable[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        normalized = normalize_doc_id(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def expected_docs(record: dict) -> set[str]:
    return {
        normalized
        for doc in record.get("expected_docs", [])
        if (normalized := normalize_doc_id(doc))
    }


def ranked_retrieved_docs(record: dict) -> list[str]:
    return dedupe(record.get("meta", {}).get("retrieved_doc_ids", []))


def citation_docs(record: dict) -> list[str]:
    return dedupe(citation.get("doc_id") for citation in record.get("citations", []))


def context_docs(record: dict) -> list[str]:
    meta = record.get("meta", {})
    evidence_context = meta.get("evidence_context", [])
    chunk_to_doc = {
        item.get("chunk_id"): normalize_doc_id(item.get("doc_id"))
        for item in evidence_context
        if item.get("chunk_id")
    }

    used_chunk_ids = meta.get("used_chunk_ids", [])
    used_docs = [
        chunk_to_doc.get(chunk_id, "")
        for chunk_id in used_chunk_ids
    ]
    used_docs = dedupe(used_docs)
    if used_docs:
        return used_docs

    cited = citation_docs(record)
    if cited:
        return cited

    return dedupe(item.get("doc_id") for item in evidence_context)


def is_refusal(record: dict) -> bool:
    answer = (record.get("model_answer") or "").strip().lower()
    return (
        not answer
        or answer.startswith("i don't have enough information")
        or answer.startswith("i do not have enough information")
        or answer.startswith("cannot answer")
        or answer.startswith("insufficient information")
    )


def doc_recall(record: dict) -> float | None:
    expected = expected_docs(record)
    if not expected:
        return None

    retrieved = set(ranked_retrieved_docs(record))
    if not retrieved:
        return 0.0
    return len(expected & retrieved) / len(expected)


def recall_at_k(record: dict, k: int) -> float | None:
    expected = expected_docs(record)
    if not expected:
        return None
    retrieved = set(ranked_retrieved_docs(record)[:k])
    return len(expected & retrieved) / len(expected)


def hit_at_k(record: dict, k: int) -> float | None:
    value = recall_at_k(record, k)
    if value is None:
        return None
    return 1.0 if value > 0 else 0.0


def reciprocal_rank(record: dict, k: int = 10) -> float | None:
    expected = expected_docs(record)
    if not expected:
        return None
    for idx, doc_id in enumerate(ranked_retrieved_docs(record)[:k], 1):
        if doc_id in expected:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(record: dict, k: int = 10) -> float | None:
    expected = expected_docs(record)
    if not expected:
        return None
    ranked = ranked_retrieved_docs(record)[:k]
    dcg = 0.0
    for idx, doc_id in enumerate(ranked, 1):
        rel = 1.0 if doc_id in expected else 0.0
        dcg += rel / math.log2(idx + 1)

    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def precision_recall_for_docs(record: dict, docs: list[str]) -> tuple[float | None, float | None]:
    expected = expected_docs(record)
    if not expected:
        return None, None
    observed = set(docs)
    if not observed:
        return 0.0, 0.0
    precision = len(expected & observed) / len(observed)
    recall = len(expected & observed) / len(expected)
    return precision, recall


def faithfulness(record: dict) -> dict | None:
    value = record.get("faithfulness")
    return value if isinstance(value, dict) else None


def aggregate(rows: list[dict]) -> dict:
    total = len(rows)
    scores = [float(r.get("judge", {}).get("score", 0)) for r in rows]
    hallucinations = [bool(r.get("judge", {}).get("hallucination")) for r in rows]
    refusals = [is_refusal(r) for r in rows]

    metas = [r.get("meta", {}) for r in rows]
    audits = [m.get("evidence_audit", {}) for m in metas]
    plans = [m.get("query_plan", {}) for m in metas]

    latencies = [
        float(m.get("latency_ms", 0.0))
        for m in metas
        if isinstance(m.get("latency_ms", 0.0), (int, float))
    ]
    coverage_scores = [
        float(a.get("coverage_score", 0.0))
        for a in audits
        if "coverage_score" in a
    ]
    doc_recalls = [value for r in rows if (value := doc_recall(r)) is not None]
    recall_by_k = {
        k: [value for r in rows if (value := recall_at_k(r, k)) is not None]
        for k in RANKING_KS
    }
    hit_by_k = {
        k: [value for r in rows if (value := hit_at_k(r, k)) is not None]
        for k in RANKING_KS
    }
    mrr_values = [value for r in rows if (value := reciprocal_rank(r, 10)) is not None]
    ndcg_values = [value for r in rows if (value := ndcg_at_k(r, 10)) is not None]

    context_pairs = [
        precision_recall_for_docs(r, context_docs(r))
        for r in rows
        if expected_docs(r)
    ]
    citation_pairs = [
        precision_recall_for_docs(r, citation_docs(r))
        for r in rows
        if expected_docs(r)
    ]
    context_precisions = [p for p, _ in context_pairs if p is not None]
    context_recalls = [r for _, r in context_pairs if r is not None]
    citation_precisions = [p for p, _ in citation_pairs if p is not None]
    citation_recalls = [r for _, r in citation_pairs if r is not None]

    faithfulness_rows = [f for r in rows if (f := faithfulness(r)) is not None]
    faithful_values = [bool(f.get("faithful")) for f in faithfulness_rows]
    faithfulness_scores = [
        float(f.get("faithfulness_score", 0.0))
        for f in faithfulness_rows
        if isinstance(f.get("faithfulness_score", 0.0), (int, float))
    ]
    citation_supported_values = [
        bool(f.get("citation_supported"))
        for f in faithfulness_rows
    ]
    retrieved_doc_counts = [
        len(r.get("meta", {}).get("retrieved_doc_ids", []))
        for r in rows
    ]
    citation_counts = [len(r.get("citations", [])) for r in rows]

    decomposed = [bool(p.get("requires_decomposition")) for p in plans]
    evidence_recovered = [bool(m.get("evidence_recovered")) for m in metas]
    audit_sufficient = [bool(a.get("sufficient")) for a in audits if a]

    grounded_pass = [
        (r.get("judge", {}).get("score", 0) >= 4)
        and not r.get("judge", {}).get("hallucination", False)
        and (not a or a.get("sufficient", True))
        for r, a in zip(rows, audits)
    ]

    by_section = defaultdict(list)
    for r in rows:
        by_section[r.get("section", "UNKNOWN")].append(r)

    return {
        "total": total,
        "answer_quality": {
            "avg_score": avg(scores),
            "accuracy_pct": sum(scores) / max(1, total * 5),
            "score_counts": dict(sorted(Counter(scores).items())),
            "hallucination_rate": avg(hallucinations),
            "refusal_rate": avg(refusals),
            "grounded_pass_rate": avg(grounded_pass),
        },
        "retrieval_quality": {
            "expected_doc_recall": avg(doc_recalls),
            "expected_doc_recall_coverage": len(doc_recalls) / max(1, total),
            "recall_at_k": {k: avg(values) for k, values in recall_by_k.items()},
            "hit_at_k": {k: avg(values) for k, values in hit_by_k.items()},
            "mrr_at_10": avg(mrr_values),
            "ndcg_at_10": avg(ndcg_values),
            "avg_retrieved_docs": avg(retrieved_doc_counts),
            "avg_citations": avg(citation_counts),
            "avg_evidence_coverage": avg(coverage_scores),
            "evidence_sufficient_rate": avg(audit_sufficient),
        },
        "context_quality": {
            "context_precision": avg(context_precisions),
            "context_recall": avg(context_recalls),
            "citation_precision": avg(citation_precisions),
            "citation_recall": avg(citation_recalls),
        },
        "faithfulness": {
            "coverage": len(faithfulness_rows) / max(1, total),
            "faithful_rate": avg(faithful_values),
            "avg_faithfulness_score": avg(faithfulness_scores),
            "citation_supported_rate": avg(citation_supported_values),
        },
        "agentic_behavior": {
            "decomposition_rate": avg(decomposed),
            "evidence_recovery_rate": avg(evidence_recovered),
        },
        "latency": {
            "avg_ms": avg(latencies),
            "p50_ms": median(latencies) if latencies else 0.0,
            "p90_ms": percentile(latencies, 0.90),
            "p95_ms": percentile(latencies, 0.95),
            "p99_ms": percentile(latencies, 0.99),
            "max_ms": max(latencies) if latencies else 0.0,
        },
        "sections": {
            section: {
                "count": len(section_rows),
                "avg_score": avg(float(r.get("judge", {}).get("score", 0)) for r in section_rows),
                "hallucination_rate": avg(bool(r.get("judge", {}).get("hallucination")) for r in section_rows),
                "doc_recall": avg(v for r in section_rows if (v := doc_recall(r)) is not None),
                "recall_at_5": avg(v for r in section_rows if (v := recall_at_k(r, 5)) is not None),
                "mrr_at_10": avg(v for r in section_rows if (v := reciprocal_rank(r, 10)) is not None),
                "ndcg_at_10": avg(v for r in section_rows if (v := ndcg_at_k(r, 10)) is not None),
            }
            for section, section_rows in sorted(by_section.items())
        },
    }


def print_report(metrics: dict) -> None:
    print("\n" + "=" * 80)
    print("NEXUS-RAG — PRODUCTION QUALITY METRICS")
    print("=" * 80)

    print(f"\nTotal Questions: {metrics['total']}")

    answer = metrics["answer_quality"]
    print("\nANSWER QUALITY")
    print(f"  Avg Judge Score       : {answer['avg_score']:.2f}/5")
    print(f"  Accuracy              : {pct(answer['accuracy_pct'])}")
    print(f"  Grounded Pass Rate    : {pct(answer['grounded_pass_rate'])}")
    print(f"  Hallucination Rate    : {pct(answer['hallucination_rate'])}")
    print(f"  Refusal Rate          : {pct(answer['refusal_rate'])}")
    print(f"  Score Counts          : {answer['score_counts']}")

    retrieval = metrics["retrieval_quality"]
    print("\nRETRIEVAL / EVIDENCE")
    print(f"  Expected Doc Recall   : {pct(retrieval['expected_doc_recall'])}")
    print(f"  Recall Coverage       : {pct(retrieval['expected_doc_recall_coverage'])}")
    for k, value in retrieval["recall_at_k"].items():
        print(f"  Recall@{k:<2}             : {pct(value)}")
    for k, value in retrieval["hit_at_k"].items():
        print(f"  Hit@{k:<2}                : {pct(value)}")
    print(f"  MRR@10                : {retrieval['mrr_at_10']:.3f}")
    print(f"  nDCG@10               : {retrieval['ndcg_at_10']:.3f}")
    print(f"  Avg Retrieved Docs    : {retrieval['avg_retrieved_docs']:.2f}")
    print(f"  Avg Citations         : {retrieval['avg_citations']:.2f}")
    print(f"  Avg Evidence Coverage : {pct(retrieval['avg_evidence_coverage'])}")
    print(f"  Evidence Sufficiency  : {pct(retrieval['evidence_sufficient_rate'])}")

    context = metrics["context_quality"]
    print("\nCONTEXT / CITATION QUALITY")
    print(f"  Context Precision     : {pct(context['context_precision'])}")
    print(f"  Context Recall        : {pct(context['context_recall'])}")
    print(f"  Citation Precision    : {pct(context['citation_precision'])}")
    print(f"  Citation Recall       : {pct(context['citation_recall'])}")

    faithful = metrics["faithfulness"]
    print("\nFAITHFULNESS")
    print(f"  Verifier Coverage     : {pct(faithful['coverage'])}")
    print(f"  Faithful Rate         : {pct(faithful['faithful_rate'])}")
    print(f"  Avg Faithfulness      : {pct(faithful['avg_faithfulness_score'])}")
    print(f"  Citation Supported    : {pct(faithful['citation_supported_rate'])}")

    agentic = metrics["agentic_behavior"]
    print("\nAGENTIC BEHAVIOR")
    print(f"  Decomposition Rate    : {pct(agentic['decomposition_rate'])}")
    print(f"  Evidence Recovery Rt  : {pct(agentic['evidence_recovery_rate'])}")

    latency = metrics["latency"]
    print("\nLATENCY")
    print(f"  Avg                   : {latency['avg_ms']:.1f} ms")
    print(f"  P50                   : {latency['p50_ms']:.1f} ms")
    print(f"  P90                   : {latency['p90_ms']:.1f} ms")
    print(f"  P95                   : {latency['p95_ms']:.1f} ms")
    print(f"  P99                   : {latency['p99_ms']:.1f} ms")
    print(f"  Max                   : {latency['max_ms']:.1f} ms")

    print("\nSECTION BREAKDOWN")
    for section, stats in metrics["sections"].items():
        print(
            f"  [{section}] count={stats['count']} "
            f"avg={stats['avg_score']:.2f} "
            f"halluc={pct(stats['hallucination_rate'])} "
            f"doc_recall={pct(stats['doc_recall'])} "
            f"recall@5={pct(stats['recall_at_5'])} "
            f"mrr@10={stats['mrr_at_10']:.3f} "
            f"ndcg@10={stats['ndcg_at_10']:.3f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_report(aggregate(load_rows(RESULTS_PATH)))
