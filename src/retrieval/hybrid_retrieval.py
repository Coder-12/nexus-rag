"""
Hybrid Retrieval - Nexus RAG
Combines dense vector search + sparse BM25 using Reciprocal Rank Fusion (RRF).
"""

from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging

from src.retrieval.bm25_retriever import initialize_bm25_system
from src.retrieval.vector_store import PineconeVectorStore
from src.retrieval.reranking.fallback_reranker import FallbackReranker, ScoreFallbackReranker
from src.retrieval.reranking.local_cross_encoder_reranker import LocalCrossEncoderReranker
from src.retrieval.reranking.cohere_reranker import CohereReranker
from openai import OpenAI
import time
import os

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion algorithm for combining ranked lists.
    Formula: score = Σ 1/(k + rank) for each source
    """
    
    def __init__(self, k: int = 60):
        """
        Args:
            k: RRF constant (default: 60 from research)
        """
        self.k = k
    
    def fuse(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4
    ) -> List[Tuple[str, float]]:
        """
        Fuse vector and BM25 results using RRF.
        
        Args:
            vector_results: [(chunk_id, score), ...]
            bm25_results: [(chunk_id, score), ...]
            vector_weight: Weight for vector search (default: 0.6)
            bm25_weight: Weight for BM25 (default: 0.4)
            
        Returns:
            Combined results sorted by fused score
        """
        fused_scores = {}
        
        # Add vector scores
        for rank, (chunk_id, _) in enumerate(vector_results, start=1):
            rrf_score = vector_weight / (self.k + rank)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + rrf_score
        
        # Add BM25 scores
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            rrf_score = bm25_weight / (self.k + rank)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + rrf_score
        
        # Sort by fused score
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results


class HybridRetriever:
    """
    Hybrid retrieval combining vector search and BM25.
    """
    
    def __init__(
        self,
        vector_store: PineconeVectorStore,
        bm25_retriever,
        openai_client: OpenAI,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
        reranker=None,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Pinecone vector store
            bm25_retriever: BM25 retriever instance
            openai_client: OpenAI client for embeddings
            vector_weight: Weight for vector search (0-1)
            bm25_weight: Weight for BM25 (0-1)
            rrf_k: RRF constant
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.openai_client = openai_client
        self.rrf = ReciprocalRankFusion(k=rrf_k)
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.reranker = reranker
    
    def retrieve(
        self,
        query: str,
        top_k: int = 100,
    ) -> List[Dict]:
        """
        Hybrid retrieval: vector + BM25 with RRF fusion.
        
        Args:
            query: Search query
            top_k: Number of candidates from each retriever
            
        Returns:
            List of dicts with chunk_id, score, and metadata
        """
        logger.info(f"Hybrid retrieval: query='{query[:50]}...', top_k={top_k}")
        
        # 1. Vector search
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        query_vector = response.data[0].embedding
        
        vector_results = self.vector_store.query(
            vector=query_vector,
            top_k=top_k
        )
        
        # Limit vector dominance — preserve recall balance
        MAX_VECTOR_CANDIDATES = 40
        FINAL_RETRIEVAL_CANDIDATES = 20
        vector_ranked = [
            (match.id, match.score)
            for match in vector_results.matches[:MAX_VECTOR_CANDIDATES]
        ]
        
        # 2. BM25 search
        bm25_ranked = self.bm25_retriever.search(query, top_k=top_k)
        
        # Keep only meaningful BM25 hits (non-zero score)
        bm25_ranked = [(cid, s) for cid, s in bm25_ranked if s > 0]
        
        logger.info(f"  Vector: {len(vector_ranked)} results")
        logger.info(f"  BM25: {len(bm25_ranked)} results")
        
        # 3. RRF fusion
        fused_results = self.rrf.fuse(
            vector_ranked,
            bm25_ranked,
            self.vector_weight,
            self.bm25_weight
        )
        
        logger.info(
            "RETRIEVAL_TRACE %s",
            {
                "vector": len(vector_ranked),
                "bm25": len(bm25_ranked),
                "fused": len(fused_results),
            }
        )
        
        top_fused = fused_results[:100]   # candidates for reranking
        chunk_ids = [cid for cid, _ in top_fused]

        metadata_batch = self.vector_store.index.fetch(
            ids=chunk_ids,
            namespace=self.vector_store.namespace
        )
        
        rerank_candidates = []
        for chunk_id, score in top_fused:
            if chunk_id in metadata_batch.vectors:
                meta = dict(metadata_batch.vectors[chunk_id].metadata)

                # 🚨 REQUIRED for reranking
                if "text" not in meta:
                    continue

                rerank_candidates.append({
                    "chunk_id": chunk_id,
                    "score": score,
                    "metadata": meta,
                })
        
        if not rerank_candidates:
            logger.warning("RERANK_EMPTY_CANDIDATES")
            reranked_results = []
        else:
            reranked_results = rerank_candidates[:20]
        
        # 4. Find the reranking based top_k
        if self.reranker and rerank_candidates:
            try:
                rerank_start = time.time()
                reranked_results = self.reranker.rerank(
                    query=query,
                    chunks=rerank_candidates,
                    top_k=20,   # 🔑 rerank to tighter set
                )
                rerank_ms = round((time.time() - rerank_start) * 1000, 2)
                logger.info(
                    "RERANK_TRACE %s",
                    {
                        "query": query[:50],
                        "candidates": len(reranked_results),
                        "latency_ms": rerank_ms,
                    }
                )
            except Exception as e:
                logger.warning(
                    "RERANK_FALLBACK %s",
                    {"error": str(e)}
                )
                reranked_results = rerank_candidates[:20]
        
        logger.info(
            "RERANK_EFFECT %s",
            {
                "before": top_fused[:3],
                "after": reranked_results[:3],
            }
        )
        # 5. Fetch metadata for top results
        top_results = reranked_results[:FINAL_RETRIEVAL_CANDIDATES]
        chunk_ids = [chunk['chunk_id'] for chunk in top_results]

        metadata_batch = self.vector_store.index.fetch(
            ids=chunk_ids,
            namespace=self.vector_store.namespace
        )

        # print(f"top_results: {top_results[0]}")
        final_results = []
        for chunk in top_results:
            chunk_id = chunk.get('chunk_id')
            score = chunk.get('score')
            if chunk_id in metadata_batch.vectors:
                metadata = dict(metadata_batch.vectors[chunk_id].metadata)
                metadata['rerank_score'] = score
                final_results.append({
                    "chunk_id": chunk_id,
                    "score": score,
                    "metadata": metadata
                })
        
        seen_docs = set()
        deduped = []

        for r in final_results:
            doc = r["metadata"]["doc_id"]
            if doc not in seen_docs:
                deduped.append(r)
                seen_docs.add(doc)

        final_results = deduped

        return final_results


class EmptyBM25Retriever:
    """
    Vector-only fallback for hosted deployments where the BM25 cache artifact is
    not present yet. This keeps the app available while surfacing a warning.
    """

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        return []


def initialize_hybrid_system(
    pinecone_index_name: str,
    pinecone_namespace: str,
    cohere_api_key: str,
    bm25_cache_path: Path,
    bm25_chunks: Optional[List] = None,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
):
    """
    Initialize hybrid retrieval system.
    
    Args:
        pinecone_index_name: Pinecone index name
        pinecone_namespace: Pinecone namespace
        bm25_cache_path: Path to BM25 cache
        bm25_chunks: Chunks for BM25 (if rebuilding)
        vector_weight: Weight for vector search (default: 0.5)
        bm25_weight: Weight for BM25 (default: 0.5)
        
    Returns:
        HybridRetriever instance
    """
    # Initialize components
    vector_store = PineconeVectorStore(
        index_name=pinecone_index_name,
        namespace=pinecone_namespace
    )
    
    try:
        bm25_retriever = initialize_bm25_system(
            cache_path=bm25_cache_path,
            chunks=bm25_chunks
        )
    except ValueError as exc:
        logger.warning(
            "BM25_UNAVAILABLE_VECTOR_ONLY_FALLBACK %s",
            {"cache_path": str(bm25_cache_path), "error": str(exc)},
        )
        bm25_retriever = EmptyBM25Retriever()
    
    openai_client = OpenAI()
    # reranker = FallbackReranker(
    #     [
    #         CohereReranker(api_key=cohere_api_key),
    #         LocalCrossEncoderReranker(),
    #         ScoreFallbackReranker(),
    #     ]
    # )
    
    rerankers = []

    cohere_enabled = os.getenv("COHERE_RERANK_ENABLED", "true").lower() == "true"

    if cohere_enabled and cohere_api_key:
        rerankers.append(CohereReranker(api_key=cohere_api_key))
    else:
        logger.warning(
            "COHERE_RERANK_DISABLED_OR_MISSING_KEY %s",
            {
                "cohere_enabled": cohere_enabled,
                "has_key": bool(cohere_api_key),
            },
        )

    try:
        rerankers.append(LocalCrossEncoderReranker())
    except Exception as exc:
        logger.warning(
            "LOCAL_RERANKER_UNAVAILABLE %s",
            {"error": str(exc)},
        )

    rerankers.append(ScoreFallbackReranker())

    reranker = FallbackReranker(rerankers)
    
    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        openai_client=openai_client,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        reranker=reranker,
    )
    
    logger.info("✅ Hybrid retrieval system initialized")
    
    return hybrid_retriever
