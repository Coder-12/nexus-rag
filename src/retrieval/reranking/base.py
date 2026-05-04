# src/retrieval/reranking/base.py

from abc import ABC, abstractmethod
from typing import List, Dict


class Reranker(ABC):
    """
    Abstract reranker interface.
    Reorders retrieved chunks by query relevance.
    """

    @abstractmethod
    def rerank(
        self,
        *,
        query: str,
        chunks: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        """
        Args:
            query: user query
            chunks: retrieved chunks (must preserve schema)
            top_k: number of chunks to return after reranking

        Returns:
            Reranked subset of chunks (length <= top_k)
        """
        pass