import logging
from typing import Dict, List

from src.retrieval.reranking.base import Reranker

logger = logging.getLogger(__name__)


class ScoreFallbackReranker(Reranker):
    """
    Deterministic fallback that preserves the best available retrieval score.
    """

    def __init__(self):
        self.last_status = "not_called"

    def rerank(self, *, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        ranked = sorted(
            chunks,
            key=lambda item: float(item.get("score", 0.0) or 0.0),
            reverse=True,
        )
        self.last_status = "success"
        return ranked[:top_k]


class FallbackReranker(Reranker):
    """
    Reranker chain:
    1. Primary managed reranker, e.g. Cohere
    2. Local cross-encoder reranker when available
    3. Score-based deterministic fallback
    """

    SUCCESS = {"success"}

    def __init__(self, rerankers: List[Reranker]):
        self.rerankers = rerankers
        self.last_status = "not_called"
        self.last_provider = None

    def rerank(self, *, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        if not chunks:
            self.last_status = "empty"
            return []

        for reranker in self.rerankers:
            provider = reranker.__class__.__name__
            ranked = reranker.rerank(query=query, chunks=chunks, top_k=top_k)
            status = getattr(reranker, "last_status", "unknown")
            logger.info(
                "RERANK_PROVIDER_TRACE %s",
                {"provider": provider, "status": status},
            )
            if status in self.SUCCESS:
                self.last_status = "success"
                self.last_provider = provider
                return ranked

        self.last_status = "fallback_exhausted"
        self.last_provider = None
        return chunks[:top_k]
