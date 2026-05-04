"""
Answer critic for intent-aware reflexion.

This module deliberately keeps the first reflexion pass deterministic. It is
used as a cheap guardrail after synthesis to catch incomplete answers before
they leave the generation layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class CritiqueResult:
    intent_matched: bool
    grounded: bool
    complete: bool
    missing_elements: List[str]
    unsupported_claims: List[str]
    needs_repair: bool
    repair_instruction: str
    risk: str

    def to_dict(self) -> Dict:
        return asdict(self)


class AnswerCritic:
    """
    Lightweight structured critic for RAG answers.

    The critic does not try to re-answer the question. It checks whether the
    produced answer satisfies the detected intent and covers the key domain
    entities implied by the question.
    """

    ENTITY_PATTERNS: Sequence[Tuple[str, Sequence[str], Sequence[str]]] = (
        ("BERT", ("bert",), ("bert",)),
        ("GPT", ("gpt",), ("gpt",)),
        (
            "encoder-decoder",
            ("encoder-decoder", "encoder decoder", "seq2seq", "sequence-to-sequence"),
            ("encoder-decoder", "encoder decoder", "seq2seq", "sequence-to-sequence", "cross-attention"),
        ),
        (
            "chain-of-thought",
            ("chain-of-thought", "chain of thought", "cot"),
            ("chain-of-thought", "chain of thought", "cot", "step by step"),
        ),
        (
            "tree-of-thought",
            ("tree-of-thought", "tree of thought", "tot"),
            ("tree-of-thought", "tree of thought", "tot", "multiple reasoning paths", "branches"),
        ),
        ("RAG", ("rag", "retrieval-augmented generation", "retrieval augmented generation"), ("rag", "retrieval", "retrieval-augmented generation")),
        ("RLHF", ("rlhf", "human preferences", "human feedback"), ("rlhf", "human feedback", "reward model", "ppo", "preferences")),
        ("HNSW", ("hnsw", "hierarchical navigable small world"), ("hnsw", "hierarchical navigable small world")),
        ("BM25", ("bm25",), ("bm25", "lexical")),
        ("embeddings", ("embedding", "embeddings", "vector representations"), ("embedding", "embeddings", "vector")),
        ("semantic search", ("semantic search", "meaning"), ("semantic search", "meaning", "embeddings", "vector")),
        ("fine-tuning", ("fine-tuning", "fine tuning", "retraining"), ("fine-tuning", "fine tuning", "weight updates", "weights")),
        ("in-context learning", ("in-context learning", "in context learning", "few examples", "examples in the prompt"), ("in-context learning", "in context learning", "prompt", "examples", "no weight updates")),
        ("outer alignment", ("outer alignment", "intended objective", "specification problem"), ("outer alignment", "intended objective", "specification", "reward hacking")),
        ("inner alignment", ("inner alignment",), ("inner alignment", "mesa", "learned objective")),
        ("prompt injection", ("prompt injection",), ("prompt injection",)),
        ("emergent abilities", ("emergent", "suddenly gain new skills", "smaller models lack"), ("emergent", "not present in smaller models", "scale")),
    )

    REFUSAL_MARKERS = (
        "i don't have enough information",
        "do not have enough information",
        "not covered by the corpus",
        "not covered in the corpus",
        "insufficient information",
    )

    CONCEPT_REQUIREMENTS: Sequence[Tuple[Sequence[str], Sequence[str], str]] = (
        (("false premise",), ("false premise", "not true", "not always", "does not", "not better", "not identical"), "false-premise correction"),
        (("better than gpt", "all language tasks"), ("false premise", "depends on the task", "task determines"), "task-dependent BERT/GPT correction"),
        (("rag eliminates hallucinations",), ("reduces", "does not eliminate", "not eliminate"), "RAG reduces not eliminates hallucination"),
        (("fine-tuning always outperforms",), ("not always", "trade-off", "depends"), "fine-tuning vs ICL trade-off"),
        (("what is the best",), ("depends", "no single", "use case", "task"), "ambiguity acknowledgement"),
        (("same as few-shot learning",), ("related", "not identical", "broader", "few-shot"), "ICL/few-shot distinction"),
        (("does rag use fine",), ("depends", "basic rag", "advanced", "not required"), "RAG/fine-tuning qualification"),
        (("chunk size",), ("smaller", "larger", "trade", "depends"), "chunk-size trade-off"),
        (("catastrophic forgetting",), ("overwrite", "prior", "knowledge"), "catastrophic forgetting mechanism"),
        (("technical jargon",), ("dense", "rare", "bm25", "hybrid"), "rare-term hybrid retrieval diagnosis"),
        (("reward model", "biased"), ("sycophancy", "reward hacking", "goodhart", "proxy"), "reward-model failure modes"),
        (("complete path", "rag system"), ("embedding", "vector", "retrieval", "context", "generation"), "end-to-end RAG failure chain"),
    )

    def critique(
        self,
        query: str,
        answer: str,
        chunks: Sequence[Dict],
        intent: str,
    ) -> Dict:
        q = self._normalize(query)
        a = self._normalize(answer)

        if self._is_refusal(a):
            grounded = True
            missing = []
            unsupported = []
            needs_repair = False
            return CritiqueResult(
                intent_matched=True,
                grounded=grounded,
                complete=True,
                missing_elements=missing,
                unsupported_claims=unsupported,
                needs_repair=needs_repair,
                repair_instruction="",
                risk="low",
            ).to_dict()

        missing = self._missing_entities(q, a)
        missing.extend(self._missing_expected_concepts(q, a))
        missing.extend(self._intent_structure_gaps(q, a, intent))

        grounded = bool(chunks)
        unsupported = []
        if not grounded:
            unsupported.append("No retrieved evidence was available for a non-refusal answer.")

        intent_matched = not any(item.startswith("intent:") for item in missing)
        complete = not missing
        needs_repair = bool(missing or unsupported)
        risk = "high" if unsupported else "medium" if missing else "low"

        repair_instruction = ""
        if needs_repair:
            parts = []
            if missing:
                parts.append("cover missing elements: " + ", ".join(missing))
            if unsupported:
                parts.append("avoid unsupported claims")
            repair_instruction = "; ".join(parts)

        return CritiqueResult(
            intent_matched=intent_matched,
            grounded=grounded,
            complete=complete,
            missing_elements=missing,
            unsupported_claims=unsupported,
            needs_repair=needs_repair,
            repair_instruction=repair_instruction,
            risk=risk,
        ).to_dict()

    def score(self, critique: Dict) -> int:
        """Higher is better; useful for accepting or rejecting repair attempts."""
        score = 0
        if critique.get("grounded"):
            score += 3
        if critique.get("intent_matched"):
            score += 2
        if critique.get("complete"):
            score += 2
        score -= len(critique.get("missing_elements", []))
        score -= 2 * len(critique.get("unsupported_claims", []))
        return score

    def _missing_entities(self, query: str, answer: str) -> List[str]:
        if "zero shot" in query and ("chain of thought" in query or "cot" in query):
            return []

        missing = []
        for label, query_terms, answer_terms in self.ENTITY_PATTERNS:
            if self._contains_any(query, query_terms) and not self._contains_any(answer, answer_terms):
                missing.append(label)
        return missing

    def _missing_expected_concepts(self, query: str, answer: str) -> List[str]:
        missing = []
        for query_terms, answer_terms, label in self.CONCEPT_REQUIREMENTS:
            if self._contains_all(query, query_terms) and not self._contains_any(answer, answer_terms):
                missing.append(label)
        return missing

    def _intent_structure_gaps(self, query: str, answer: str, intent: str) -> List[str]:
        gaps = []

        if self._is_list_query(query):
            if self._item_count(answer) < 2:
                gaps.append("intent:list structure")

        if (self._is_relationship_query(query) and not self._is_contrastive_query(query)) or intent in {"multi_hop", "reasoning"}:
            if not self._contains_any(answer, ("interact", "relationship", "connect", "depends on", "through", "by")):
                gaps.append("intent:relationship link")

        if intent == "analytical" and self._contains_any(query, ("evolved", "evolution", "over time", "as models improved")):
            for required in ("past", "present", "implication"):
                if required not in answer:
                    gaps.append(f"intent:analytical {required}")

        if intent == "procedural" or self._contains_any(query, ("how would", "how to", "steps", "pipeline", "implemented")):
            if self._item_count(answer) < 2 and not self._contains_any(answer, ("first", "then", "finally")):
                gaps.append("intent:procedural sequence")

        return gaps

    def _is_list_query(self, query: str) -> bool:
        return self._contains_any(
            query,
            (
                "components",
                "objectives",
                "types",
                "steps",
                "measures",
                "techniques",
                "methods",
                "factors",
                "requirements",
                "what are",
                "list",
            ),
        )

    def _is_relationship_query(self, query: str) -> bool:
        return self._contains_any(
            query,
            (
                "relationship",
                "relate",
                "connect",
                "connection",
                "how does",
                "how do",
                "interact",
            ),
        )

    def _is_contrastive_query(self, query: str) -> bool:
        return self._contains_any(
            query,
            ("difference", "differ", "distinguish", "compare", " vs ", "versus"),
        )

    def _item_count(self, answer: str) -> int:
        bullets = len(re.findall(r"(^|\n)\s*(?:[-*]|\d+\.)\s+", answer))
        if bullets:
            return bullets
        separators = answer.count(";") + answer.count(":")
        return separators + 1 if separators else 1

    def _is_refusal(self, answer: str) -> bool:
        return self._contains_any(answer, self.REFUSAL_MARKERS)

    def _contains_any(self, text: str, terms: Iterable[str]) -> bool:
        return any(self._normalize(term) in text for term in terms)

    def _contains_all(self, text: str, terms: Iterable[str]) -> bool:
        return all(self._normalize(term) in text for term in terms)

    def _normalize(self, text: str) -> str:
        text = text.lower().replace("-", " ")
        return " ".join(text.split())
