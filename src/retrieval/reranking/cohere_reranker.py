# src/retrieval/reranking/cohere_reranker.py

import logging
import os
import time
from pathlib import Path
from typing import List, Dict

import cohere

from src.retrieval.reranking.base import Reranker

logger = logging.getLogger(__name__)


class CohereReranker(Reranker):
    """
    Cohere cross-encoder reranker (rerank-english-v3.0).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0",
        max_documents: int = 100,
        cooldown_seconds: int | None = None,
    ):
        self.client = cohere.Client(api_key)
        self.model = model
        self.max_documents = max_documents
        self.last_status = "not_called"
        self.cooldown_seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else int(os.getenv("COHERE_RERANK_COOLDOWN_SECONDS", "300"))
        )
        self.cooldown_path = Path(
            os.getenv("COHERE_RERANK_COOLDOWN_PATH", "cache/cohere_rerank_cooldown.txt")
        )
        self._disabled_until = self._read_disabled_until()

    def rerank(
        self,
        *,
        query: str,
        chunks: List[Dict],
        top_k: int,
    ) -> List[Dict]:

        if not chunks:
            self.last_status = "empty"
            return []

        now = time.time()
        if now < self._disabled_until:
            self.last_status = "cooldown"
            logger.warning(
                "RERANK_SKIPPED_COOLDOWN %s",
                {
                    "model": self.model,
                    "remaining_seconds": round(self._disabled_until - now, 1),
                },
            )
            return chunks[:top_k]

        # Cap documents for safety
        candidates = chunks[: self.max_documents]

        documents = [
            c["metadata"].get("text", "")
            for c in candidates
        ]

        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
            )

            ranked_chunks = [
                candidates[result.index]
                for result in response.results
            ]

            logger.info(
                "RERANK_TRACE",
                extra={
                    "model": self.model,
                    "input_chunks": len(chunks),
                    "reranked": len(ranked_chunks),
                },
            )

            self.last_status = "success"
            return ranked_chunks

        except Exception as e:
            if self._is_rate_limited(e):
                self.last_status = "rate_limited"
                self._disabled_until = time.time() + self.cooldown_seconds
                self._write_disabled_until(self._disabled_until)
                logger.warning(
                    "RERANK_RATE_LIMIT_COOLDOWN %s",
                    {
                        "model": self.model,
                        "cooldown_seconds": self.cooldown_seconds,
                        "error": str(e),
                    },
                )
            else:
                logger.exception(
                    "Reranking failed, falling back to original order due to %s",
                    e,
                )
                self.last_status = "error"
            return candidates[:top_k]

    @staticmethod
    def _is_rate_limited(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code == 429:
            return True
        error_name = error.__class__.__name__.lower()
        return "ratelimit" in error_name or "too_many_requests" in error_name

    def _read_disabled_until(self) -> float:
        try:
            return float(self.cooldown_path.read_text().strip())
        except (FileNotFoundError, ValueError, OSError):
            return 0.0

    def _write_disabled_until(self, disabled_until: float) -> None:
        try:
            self.cooldown_path.parent.mkdir(parents=True, exist_ok=True)
            self.cooldown_path.write_text(str(disabled_until))
        except OSError:
            logger.warning(
                "RERANK_COOLDOWN_PERSIST_FAILED %s",
                {"path": str(self.cooldown_path)},
            )
