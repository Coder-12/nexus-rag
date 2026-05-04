import logging
import os
from typing import Dict, List

from src.retrieval.reranking.base import Reranker

logger = logging.getLogger(__name__)


class LocalCrossEncoderReranker(Reranker):
    """
    Optional local cross-encoder reranker.

    This is intentionally lazy: if the model is not installed/cached or the
    runtime cannot load it, the reranker reports unavailable and lets the next
    fallback keep the pipeline healthy.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_documents: int = 40,
    ):
        self.model_name = model_name or os.getenv(
            "LOCAL_RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        self.max_documents = max_documents
        self.model = None
        self.last_status = "not_called"

    def rerank(self, *, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        if not chunks:
            self.last_status = "empty"
            return []

        model = self._load_model()
        if model is None:
            self.last_status = "unavailable"
            return chunks[:top_k]

        candidates = chunks[: self.max_documents]
        pairs = [
            [query, str(chunk.get("metadata", {}).get("text", ""))]
            for chunk in candidates
        ]

        try:
            scores = model.predict(pairs)
        except Exception as exc:
            self.last_status = "error"
            logger.warning(
                "LOCAL_RERANK_FAILED %s",
                {"model": self.model_name, "error": str(exc)},
            )
            return chunks[:top_k]

        ranked = []
        for chunk, score in zip(candidates, scores):
            item = dict(chunk)
            metadata = dict(item.get("metadata", {}))
            metadata["local_rerank_score"] = float(score)
            item["metadata"] = metadata
            item["score"] = float(score)
            ranked.append(item)

        self.last_status = "success"
        ranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        logger.info(
            "LOCAL_RERANK_TRACE %s",
            {"model": self.model_name, "input_chunks": len(candidates), "reranked": len(ranked[:top_k])},
        )
        return ranked[:top_k]

    def _load_model(self):
        if self.model is not None:
            return self.model

        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            logger.warning("LOCAL_RERANK_IMPORT_UNAVAILABLE %s", {"error": str(exc)})
            return None

        try:
            self.model = CrossEncoder(self.model_name)
        except Exception as exc:
            logger.warning(
                "LOCAL_RERANK_MODEL_UNAVAILABLE %s",
                {"model": self.model_name, "error": str(exc)},
            )
            return None

        return self.model
