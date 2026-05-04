"""
Intelligent Router - Nexus RAG
Rule-based routing with LLM fallback for ambiguous queries.
"""

import logging
import json
import time
from typing import Dict, List, Tuple
from pathlib import Path

from src.retrieval.query_analyzer import QueryAnalyzer
from src.retrieval.vector_store import PineconeVectorStore
from src.retrieval.hybrid_retrieval import initialize_hybrid_system
from src.observability.router_metrics import RouterMetrics
from openai import OpenAI

logger = logging.getLogger(__name__)


class IntelligentRouter:
    """
    Agentic routing controller:
    - Rule-based primary routing
    - LLM fallback for ambiguous cases
    """

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        hybrid_retriever,
        query_analyzer: QueryAnalyzer,
        openai_client: OpenAI,
        llm_model: str = "gpt-4o-mini",
    ):
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever
        self.query_analyzer = query_analyzer
        self.openai_client = openai_client
        self.llm_model = llm_model
        self.metrics = RouterMetrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_and_retrieve(
        self,
        query: str,
        top_k: int = 100,
        final_k: int = 5,
    ) -> Dict:
        analysis = self.query_analyzer.analyze(query)

        # Phase 1: rule-based routing only
        strategy, reason = self._rule_based_strategy(analysis)
        used_llm = False
        
        disable_fallback = False
        preview_results = []
        # Phase 3: decide LLM fallback
        if self._needs_llm_fallback(analysis):
            if strategy == "hybrid":
                preview_results = self.hybrid_retriever.retrieve(
                    query=query,
                    top_k=20,
                )
                preview_results = preview_results[:5]
            else:
                preview_results = self._vector_retrieve(
                    query=query,
                    top_k=20,
                    final_k=5,
                )
            
            disable_fallback = self._should_disable_llm_fallback(
                retrieved_chunks=preview_results,
                analysis=analysis,
            )

            if not disable_fallback:
                strategy, reason = self._llm_route_decision(query, analysis)
                used_llm = True
            else:
                reason = f"{reason} + retrieval_confidence_gate"
        
        logger.info(
            "ROUTING_CONFIDENCE_GATE %s",
            json.dumps({
                "intent": analysis["intent"],
                "query_confidence": analysis["confidence"],
                "disable_llm_fallback": disable_fallback,
                "preview_executed": bool(preview_results),
                "avg_top_score": (
                    round(
                        sum(c.get("score", 0) for c in preview_results[:5]) / max(1, len(preview_results[:5])), 3
                    )
                    if preview_results else None
                ),
                "unique_docs": len({c["metadata"].get("doc_id") for c in preview_results}),
                "chunk_count": len(preview_results),
            })
        )
        routing_trace = {
            "query": query,
            "intent": analysis["intent"],
            "complexity": analysis["complexity"],
            "confidence": analysis["confidence"],
            "keyword_density": analysis["keyword_density"],
            "entities": analysis["entities"],
            "strategy": strategy,
            "used_llm_fallback": used_llm,
            "routing_reason": reason,
            "timestamp": time.time(),
        }
        
        logger.info(
            "ROUTING_TRACE %s",
            json.dumps(routing_trace, ensure_ascii=False)
        )

        logger.info(
            "Routing | strategy=%s | llm_used=%s | intent=%s | confidence=%.2f",
            strategy,
            used_llm,
            analysis["intent"],
            analysis["confidence"],
        )
        
        self.metrics.increment(f"route.{strategy}")
        if used_llm:
            self.metrics.increment("route.llm_fallback")

        if strategy == "hybrid":
            results = self.hybrid_retriever.retrieve(
                query=query,
                top_k=top_k,
            )
            results = results[:top_k]
        else:
            results = self._vector_retrieve(
                query=query,
                top_k=top_k,
                final_k=final_k,
            )

        return {
            "results": results,
            "strategy": strategy,
            "routing_reason": reason,
            "used_llm_fallback": used_llm,
            "analysis": analysis,
        }

    def _rule_based_strategy(self, analysis: Dict) -> Tuple[str, str]:
        intent = analysis["intent"]
        complexity = analysis["complexity"]
        confidence = analysis["confidence"]
        density = analysis["keyword_density"]
        multi_doc = analysis["requires_multiple_docs"]
        entities = analysis["entities"]

        if (
            intent == "factual"
            and confidence >= 0.75
            and complexity <= 4
            and not multi_doc
        ):
            return "vector", "high_confidence_factual"

        # Comparative / relationship queries
        if intent in {"comparative", "relationship"}:
            return "hybrid", "comparative_or_relationship_intent"

        # Explicit multi-doc requirement
        if multi_doc:
            return "hybrid", "requires_multiple_docs"

        # High complexity
        if complexity >= 7:
            return "hybrid", "high_complexity"

        # Multiple distinct entities (true comparison)
        if len(entities) >= 2:
            return "hybrid", "multiple_entities_detected"

        # Keyword-heavy but low confidence → hybrid
        if density >= 0.6 and confidence < 0.7:
            return "hybrid", "keyword_heavy_low_confidence"

        return "vector", "default_vector"

    def _needs_llm_fallback(self, analysis: Dict) -> bool:
        """
        Gate expensive LLM routing.
        """
        return (
            analysis["confidence"] < 0.65
            or analysis["complexity"] >= 6
            or (
                len(analysis["entities"]) >= 2
                and analysis["intent"] in {"comparative", "analytical", "relationship"}
            )
        )

    # ------------------------------------------------------------------
    # LLM Routing Agent
    # ------------------------------------------------------------------

    def _llm_route_decision(self, query: str, analysis: Dict) -> Tuple[str, str]:
        """
        LLM decides routing strategy when rules are uncertain.
        """
        prompt = f"""
You are an expert retrieval router for a RAG system.

Query: "{query}"

Query analysis:
{json.dumps(analysis, indent=2)}

Choose the best retrieval strategy:
- "vector": simple factual, single-doc semantic lookup
- "hybrid": comparative, multi-doc, keyword-sensitive queries

Return JSON only:
{{
  "strategy": "vector|hybrid",
  "reason": "short explanation"
}}
"""

        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a retrieval routing expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=120,
        )

        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()

        decision = json.loads(content)

        return decision["strategy"], f"llm_fallback: {decision['reason']}"

    # ------------------------------------------------------------------
    # Vector Retrieval
    # ------------------------------------------------------------------

    def _vector_retrieve(
        self,
        query: str,
        top_k: int,
        final_k: int,
    ) -> List[Dict]:
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=query,
        )

        vector_results = self.vector_store.query(
            vector=response.data[0].embedding,
            top_k=top_k,
        )

        return [
            {
                "chunk_id": match.id,
                "score": match.score,
                "metadata": match.metadata,
            }
            for match in vector_results.matches[:final_k]
        ]
        
    def _should_disable_llm_fallback(
        self,
        retrieved_chunks: List[Dict],
        analysis: Dict,
        min_score: float = 0.72,
    ) -> bool:
        """
        Disable LLM fallback when retrieval is already strong.
        """

        intent = analysis["intent"]
        confidence = analysis["confidence"]

        if intent not in {"comparative", "analytical", "relationship", "procedural"}:
            return False

        if confidence < 0.75 or not retrieved_chunks:
            return False

        scores = [c.get("score", 0.0) for c in retrieved_chunks[:5]]
        avg_score = sum(scores) / max(1, len(scores))

        unique_docs = len({c["metadata"].get("doc_id") for c in retrieved_chunks})
        chunk_count = len(retrieved_chunks)

        return (
            avg_score >= min_score
            and unique_docs >= 2
            and chunk_count >= 3
        )


# ----------------------------------------------------------------------
# System Initialization
# ----------------------------------------------------------------------

def initialize_routing_system(
    pinecone_index_name: str,
    pinecone_namespace: str,
    cohere_api_key: str,
    bm25_cache_path: Path,
):
    vector_store = PineconeVectorStore(
        index_name=pinecone_index_name,
        namespace=pinecone_namespace,
    )

    hybrid_retriever = initialize_hybrid_system(
        pinecone_index_name=pinecone_index_name,
        pinecone_namespace=pinecone_namespace,
        cohere_api_key=cohere_api_key,
        bm25_cache_path=bm25_cache_path,
    )

    query_analyzer = QueryAnalyzer()
    openai_client = OpenAI()

    router = IntelligentRouter(
        vector_store=vector_store,
        hybrid_retriever=hybrid_retriever,
        query_analyzer=query_analyzer,
        openai_client=openai_client,
    )

    logger.info("✅ Agentic intelligent routing system initialized")

    return router
