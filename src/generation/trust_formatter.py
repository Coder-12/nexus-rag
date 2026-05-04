"""
Trust Formatter - Nexus RAG
User-facing trust and confidence presentation layer.
"""

from typing import Dict, List, Optional

from src.generation.refusal import RefusalReason

REFUSAL_EXPLANATIONS = {
    "no_relevant_documents": (
        "This question appears outside the ingested AI/ML corpus, so I should not invent an answer. "
        "Try asking about RAG, retrieval, embeddings, LLMs, alignment, evaluation, or production readiness."
    ),
    "temporal_out_of_scope": (
        "This asks for current or time-sensitive information that is not available in the static corpus. "
        "I can answer the underlying AI/ML concept if you phrase it without needing live updates."
    ),
    "unsupported_specificity": (
        "The corpus does not support that level of exact detail. I can answer at the conceptual level covered by the sources."
    ),
    "false_premise": (
        "The question contains a premise that conflicts with the corpus, so I should correct the premise instead of accepting it."
    ),
    RefusalReason.NO_RETRIEVAL.value: (
        "I couldn't ground that in the available sources. Try an AI/ML interview question about RAG, embeddings, vector search, LLMs, alignment, evaluation, or production readiness."
    ),
    RefusalReason.INSUFFICIENT_EVIDENCE.value: (
        "I couldn't ground that in the current AI/ML corpus. Try asking about RAG, retrieval, embeddings, LLMs, alignment, evaluation, or production readiness."
    ),
    RefusalReason.LOW_CONFIDENCE.value: (
        "The available evidence is too weak or incomplete to provide a reliable answer. Ask within the AI/ML interview scope for a grounded response."
    ),
    RefusalReason.CONTRADICTION.value: (
        "The available sources contain conflicting information, so I can't provide a reliable answer without resolving the conflict."
    ),
    RefusalReason.UNSUPPORTED_CLAIM.value: (
        "The answer could not be supported by the retrieved sources, so I won't present it as grounded."
    ),
}


class TrustFormatter:
    """
    Formats AnswerSynthesizer output into a stable,
    user-facing trust-aware response schema.
    """

    # -------------------------------
    # Public API
    # -------------------------------

    def format(
        self,
        *,
        answer_text: str,
        confidence_score: float,
        citations: List[Dict],
        trust_inputs: Dict,
        refused: bool,
        refusal_reason: Optional[str],
        meta: Dict,
    ) -> Dict:
        """
        Format final response for UI / API consumers.
        """

        if refused:
            return self._format_refusal(refusal_reason, meta)

        return {
            "answer": {
                "text": answer_text,
                "type": self._infer_answer_type(answer_text),
            },
            "confidence": {
                "score": round(confidence_score, 2),
                "level": self._confidence_level(confidence_score),
                "explanation": self._confidence_explanation(trust_inputs),
            },
            "citations": self._format_citations(citations),
            "trust_signals": self._trust_signals(trust_inputs),
            "refusal": None,
            "meta": meta,
        }

    # -------------------------------
    # Confidence helpers
    # -------------------------------

    def _confidence_level(self, score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.45:
            return "medium"
        return "low"

    def _confidence_explanation(self, trust: Dict) -> str:
        if trust.get("contradiction"):
            return (
                "Some parts of the answer may conflict with the available sources, "
                "reducing overall confidence."
            )

        signals = []

        if trust.get("support_score", 0) >= 0.7:
            signals.append("strong evidence support")

        if trust.get("retrieval_agreement", 0) >= 0.5:
            signals.append("consistent retrieval across methods")

        if trust.get("source_agreement", 0) >= 0.5:
            signals.append("multiple independent sources")

        if not signals:
            return (
                "The answer is based on limited or weakly supported evidence, "
                "resulting in lower confidence."
            )

        return (
            "The answer is supported by "
            + ", ".join(signals)
            + " with no detected contradictions."
        )

    # -------------------------------
    # Trust signals
    # -------------------------------

    def _trust_signals(self, trust: Dict) -> Dict:
        return {
            "evidence_supported": trust.get("support_score", 0) >= 0.7,
            "multi_source": trust.get("source_agreement", 0) >= 0.5,
            "contradiction_checked": True,
            "retrieval_agreement": self._bucket(trust.get("retrieval_agreement", 0)),
            "attribution_quality": self._bucket(trust.get("attribution_score", 0)),
        }

    def _bucket(self, score: float) -> str:
        if score >= 0.7:
            return "strong"
        if score >= 0.4:
            return "partial"
        return "weak"

    # -------------------------------
    # Citations
    # -------------------------------

    def _format_citations(self, citations: List[Dict]) -> List[Dict]:
        formatted = []
        for c in citations:
            formatted.append(
                {
                    "doc_id": c["doc_id"],
                    "section": c["section"],
                    "used_for": "supporting evidence",
                }
            )
        return formatted

    # -------------------------------
    # Refusal handling
    # -------------------------------

    def _format_refusal(self, reason: Optional[str], meta: Dict) -> Dict:
        explanation = REFUSAL_EXPLANATIONS.get(
            reason,
            "I couldn't ground that in the current AI/ML corpus. "
            "Try asking about RAG, retrieval, embeddings, vector search, LLMs, alignment, evaluation, or production readiness."
        )
        return {
            "answer": None,
            "confidence": {
                "score": 0.0,
                "level": "low",
                "explanation": "The system could not produce a reliable answer.",
            },
            "citations": [],
            "trust_signals": {
                "evidence_supported": False,
                "multi_source": False,
                "contradiction_checked": True,
                "retrieval_agreement": "weak",
                "attribution_quality": "weak",
            },
            "refusal": {
                "refused": True,
                "reason": reason or "insufficient_evidence",
                "message": explanation,
            },
            "meta": meta,
        }

    # -------------------------------
    # Utilities
    # -------------------------------

    def _infer_answer_type(self, text: str) -> str:
        if "compare" in text.lower() or "difference" in text.lower():
            return "comparison"
        if len(text.split()) > 60:
            return "summary"
        return "direct"
